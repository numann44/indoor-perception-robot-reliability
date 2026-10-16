"""Residual least-squares CV forecasting: sensor history in, target-only loss.

All tests stay closed until a trained checkpoint selection lock is supplied.
Image XY normalized units and calibrated world XZ metres have separate packs.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,platform,random,shlex,subprocess,sys,time,uuid,zipfile
from pathlib import Path
import numpy as np
from .experiments import root_path,json_write,sha
from .motion_data import (FUTURE_TIMES,resample_history,interpolate_labels,fixture_scene_parameters,render_fixture_frame,sensor_world_positions)
from .motion_training import displacement_metrics,train_scale,seed_all
from .tracking import Tracker,bbox_iou


def _dataset_writable(root,domain):
    selection=Path(root)/f"models/v2/selection-forecast-{domain}.json"
    if selection.exists():raise RuntimeError("Forecast dataset frozen by canonical selection; preserve its exact sensor/target files")


def _provenance_sources(root,extra_sources=()):
    root=Path(root);sources=set()
    for package in ["robot_model_sdk","indoor_perception_lab"]:
        directory=root/"packages"/package/"src"/package
        sources.update(str(path.relative_to(root)) for path in directory.rglob("*") if path.is_file() and path.suffix in {".py",".json"})
    for relative in ["portfolio","tools/portfolio_cli.py","tools/forecast_v2.py",*extra_sources]:
        if (root/relative).is_file():sources.add(relative)
    return sorted(sources)


def _runtime_provenance(root,run,extra_sources=()):
    """Snapshot executed local sources and runtime without external writes."""
    from importlib.metadata import version
    root=Path(root);run=Path(run);sources=_provenance_sources(root,extra_sources);capture_started=datetime.datetime.now(datetime.timezone.utc).isoformat();contents={relative:(root/relative).read_bytes() for relative in sources};hashes={p:hashlib.sha256(value).hexdigest() for p,value in contents.items()};archive=run/"source.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for relative in sources:z.writestr(relative,contents[relative])
    try:commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    return {"source_archive":str(archive.relative_to(root)),"source_archive_sha256":sha(archive),"source_hashes":hashes,"source_snapshot_scope":"complete SDK and Lab Python/JSON source sets including initializers; root launcher/portfolio_cli/forecast tool; no data/env/artifacts","source_snapshot_complete":True,"source_capture_started_utc":capture_started,"source_capture_finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"git_head":commit,"git_working_sources_snapshotted":True,"hardware":{"platform":platform.platform(),"architecture":platform.machine(),"cpu_count":os.cpu_count(),"execution_device":"cpu","tensor_threads":3},"runtime_versions":{name:version(name) for name in ["torch","numpy","opencv-python-headless"]},"python":sys.version}


def baselines_v2(history):
    h=np.asarray(history,np.float64)
    if h.ndim==2:h=h[None]
    origin=h[:,-1:,:];times=np.linspace(-1,0,11);weights=(times-times.mean())/np.sum((times-times.mean())**2)
    v_last=(h[:,-1,:]-h[:,-2,:])/.1;v_ls=np.sum(h*weights[None,:,None],axis=1)
    return {"constant_position":np.repeat(origin,20,axis=1),"constant_velocity_last2":origin+v_last[:,None,:]*FUTURE_TIMES[None,:,None],"constant_velocity_LS":origin+v_ls[:,None,:]*FUTURE_TIMES[None,:,None]}


def _save_motion(root,domain,samples,metadata):
    _dataset_writable(root,domain)
    if not samples:raise ValueError("No observed-history forecast windows; cannot train a checkpoint")
    out=Path(root)/f"data/v2/forecast/{domain}";out.mkdir(parents=True,exist_ok=True)
    h=np.stack([s.pop("history") for s in samples]);y=np.stack([s.pop("future") for s in samples]);np.savez_compressed(out/"sensor_histories.npz",history=h);np.savez_compressed(out/"target_futures.npz",future=y)
    index=out/"index.jsonl";index.write_text("".join(json.dumps(s,separators=(",",":"))+"\n" for s in samples))
    metadata.update(schema_version=2,history_sha256=sha(out/"sensor_histories.npz"),future_sha256=sha(out/"target_futures.npz"),index_sha256=sha(index),
                    counts={part:sum(s["split"]==part for s in samples) for part in ["train","val","test"]},history_points=11,history_span_s=1.,future_points=20,resample_hz=10,
                    target_use="trainingloss/evaluatoronly; notpredictorfeatures",test_predictions_computed=False)
    json_write(out/"manifest.json",metadata);split_motion_files(out,h,y,samples,metadata);return out


def split_motion_files(out,h,y,rows,metadata):
    out=Path(out);files={}
    for part in ["train","val","test"]:
        ids=np.array([i for i,r in enumerate(rows) if r["split"]==part],int);hp=out/f"sensor_histories-{part}.npz";yp=out/f"target_futures-{part}.npz";rp=out/f"index-{part}.jsonl"
        np.savez_compressed(hp,history=h[ids]);np.savez_compressed(yp,future=y[ids]);rp.write_text("".join(json.dumps(rows[i],separators=(",",":"))+"\n" for i in ids))
        files[part]={"history_file":hp.name,"history_sha256":sha(hp),"future_file":yp.name,"future_sha256":sha(yp),"index_file":rp.name,"index_sha256":sha(rp),"windows":len(ids),"scene_groups":sorted({rows[i]["scene_id"] for i in ids})}
    metadata.update(split_files=files,training_target_access="trainvalsplitfilesonly; unsplitfilesnotreadbytrainer");json_write(out/"manifest.json",metadata)


def load_forecast_data(root,domain,splits=("train","val")):
    out=Path(root)/f"data/v2/forecast/{domain}";meta=json.loads((out/"manifest.json").read_text())
    if "split_files" not in meta:raise ValueError("V2 requires split-specifictargetfiles; migratefirst")
    splits=(splits,) if isinstance(splits,str) else splits;histories=[];futures=[];rows=[]
    for part in splits:
        info=meta["split_files"][part]
        for file,key in [(info["history_file"],"history_sha256"),(info["future_file"],"future_sha256"),(info["index_file"],"index_sha256")]:
            if sha(out/file)!=info[key]:raise ValueError("V2 forecast split checksum mismatch")
        histories.append(np.load(out/info["history_file"])["history"]);futures.append(np.load(out/info["future_file"])["future"]);rows.extend(json.loads(s) for s in (out/info["index_file"]).read_text().splitlines())
    h=np.concatenate(histories);y=np.concatenate(futures)
    if h.shape!=(len(rows),11,2) or y.shape!=(len(rows),20,2) or not np.isfinite(h).all() or not np.isfinite(y).all():raise ValueError("V2 forecast arrays invalid")
    groups={part:{r["scene_id"] for r in rows if r["split"]==part} for part in ["train","val","test"]}
    if any(groups[a]&groups[b] for a,b in [("train","val"),("train","test"),("val","test")]):raise ValueError("Forecast scene leakage")
    return h,y,rows,meta,out


def build_fixture_v2(root=None,counts=(72,24,24)):
    from robot_model_sdk.models import FixtureAdapter
    root=Path(root) if root is not None else root_path();_dataset_writable(root,"fixture");adapter=FixtureAdapter();samples=[];sensor_hash=hashlib.sha256();seed_groups={};examples=root/"data/v2/fixture/sensor_examples";examples.mkdir(parents=True,exist_ok=True)
    for split,count,base in zip(["train","val","test"],counts,[40000,50000,60000]):
        seed_groups[split]=list(range(base,base+count))
        for seed in seed_groups[split]:
            params=fixture_scene_parameters(seed);tracker=Tracker(max_age_s=.2,iou_threshold=.15);past={};future={"block-0":[],"block-1":[]};anchors=[]
            for i in range(81):
                frame,labels=render_fixture_frame(params,i);t=i/10;sensor_hash.update(frame.rgb.tobytes());sensor_hash.update(frame.depth.tobytes());sensor_hash.update(frame.T_world_camera.tobytes())
                if seed==base and i in [0,20,40,60,80]:np.savez_compressed(examples/f"{split}-{seed}-{i:03d}.npz",rgb=frame.rgb,depth=frame.depth,T_world_camera=frame.T_world_camera)
                for target in labels:future[target["object_id"]].append((t,np.array(target["position_world_m"])[[0,2]]))
                measured=sensor_world_positions(frame,adapter);tracks=tracker.update([d for d,p in measured],frame.timestamp_ns)
                for track in tracks:
                    if not track.observed or not measured:continue
                    _,position=max([(bbox_iou(track.bbox,d.bbox),p) for d,p in measured],key=lambda x:x[0]);past.setdefault(track.id,[]).append((t,position.copy()))
                if i<10 or i>60 or i%2:continue
                pairs=sorted([(bbox_iou(tr.bbox,label["bbox_render_target_only"]),tr,label) for tr in tracks if tr.observed for label in labels],key=lambda p:p[0],reverse=True);used_t=set();used_g=set()
                for overlap,track,target in pairs:
                    if overlap<.3 or track.id in used_t or target["object_id"] in used_g:continue
                    used_t.add(track.id);used_g.add(target["object_id"]);v=past[track.id];h=resample_history([p[0] for p in v],[p[1] for p in v],t,.11)
                    if h is not None:anchors.append((t,track.id,target["object_id"],h))
            for t,tid,gtid,h in anchors:
                vals=future[gtid];y=interpolate_labels([v[0] for v in vals],[v[1] for v in vals],t+FUTURE_TIMES,.11)
                if y is not None:samples.append({"sample_id":f"fixture-v2/{seed}/{t:.1f}/{tid}","scene_id":f"fixture-v2-seed{seed}","split":split,"anchor_time_s":t,"history_source":"RGB_CC_registereddepth_knownpose","tracker_id":tid,"target_id_evaluator_only":gtid,"history":h,"future":y})
    adapter.close();return _save_motion(root,"fixture",samples,{"domain":"synthetic independentlyseeded calibratedcoloredblocks","coordinate_system":"world_XZ_planar","units":"metres","history_source":"RGB_CC_registereddepth_knownpose","seed_groups":seed_groups,"source_sensor_sha256":sensor_hash.hexdigest(),"source_renderer_sha256":sha(Path(__file__).with_name("motion_data.py")),"baseline_methods":["constant_position","constant_velocity_last2","constant_velocity_LS"],"knownpose_notSLAM":True})


def _jaad_frontend_identity(root,pack,threshold=None):
    from .tracking_v2 import TRACKING_PRESETS
    root=Path(root);pack=Path(pack);detection_path=root/"models/v2/selection-detection.json";tracking_path=root/"models/v2/selection-tracking.json"
    if not detection_path.exists() or not tracking_path.exists():raise ValueError("JAAD forecast requires frozen detector and validation-selected tracker locks")
    d=json.loads(detection_path.read_text());t=json.loads(tracking_path.read_text());weight_sha=sha(pack/"weights.pt");manifest_sha=sha(pack/"manifest.json");index_sha=sha(root/"data/v2/jaad/index.jsonl")
    if d.get("task")!="detection" or d.get("selection_split")!="val" or d.get("test_used_for_selection") is not False or d.get("dataset_split_hash")!=index_sha:raise ValueError("JAAD forecast detector selection must be validation-only on the frozen JAAD index")
    selected=d["selected"]
    if (root/selected["pack"]).resolve()!=pack.resolve() or selected["weights_sha256"]!=weight_sha or selected["manifest_sha256"]!=manifest_sha:raise ValueError("JAAD forecast teacher must be the exact canonical selected detector")
    if t.get("selection_split")!="val" or t.get("test_used_for_selection") is not False or t["detector_weights_sha256"]!=weight_sha or t["detector_manifest_sha256"]!=manifest_sha or t["validation_index_sha256"]!=index_sha:raise ValueError("JAAD forecast tracker lock teacher/source mismatch")
    if t.get("tracker_source_sha256")!=sha(Path(__file__).with_name("tracking_v2.py")):raise ValueError("Validation-selected tracker source changed after lock")
    configuration=t["selected"]["configuration"];params=t["selected"]["parameters"]
    if configuration not in TRACKING_PRESETS or params!=TRACKING_PRESETS[configuration]:raise ValueError("Selected tracker parameters/source preset changed")
    low=params["low_score"]
    if threshold is not None and threshold!=low:raise ValueError("JAAD score cutoff is frozen selected tracker low_score; independent threshold tuning forbidden")
    params={**params,"max_age_s":1.};parameter_sha=hashlib.sha256(json.dumps(params,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"detector_selection_sha256":sha(detection_path),"tracker_selection_sha256":sha(tracking_path),"tracker_configuration":configuration,"tracker_parameters":params,"tracker_parameters_sha256":parameter_sha,"tracker_source_sha256":sha(Path(__file__).with_name("tracking_v2.py")),"detector_sha256":weight_sha,"detector_manifest_sha256":manifest_sha,"source_index_sha256":index_sha,"threshold":low}


def build_jaad_forecast_v2(detector_pack,threshold=None,device="cpu",root=None):
    """Fixed final detector sensor histories; GT only eligibility/anchor/future join."""
    import cv2,torch
    from robot_model_sdk import SensorFrameV1,Detection
    from robot_model_sdk.models import load_adapter
    from .data_v2 import read_index_v2
    from .motion_data import _label_tracks
    from .tracking_v2 import make_tracker
    root=Path(root) if root is not None else root_path();_dataset_writable(root,"jaad");pack=Path(detector_pack);pack=pack if pack.is_absolute() else root/pack
    if not (root/"reports/v2/data/jaad-validation.json").exists():raise ValueError("Frozen JAAD data validation required")
    validation=json.loads((root/"reports/v2/data/jaad-validation.json").read_text())
    if not validation.get("primary_ready"):raise ValueError("JAAD source not frozen aftercopy audit")
    frontend=_jaad_frontend_identity(root,pack,threshold);threshold=frontend["threshold"];torch.set_num_threads(3);cv2.setNumThreads(1);adapter=load_adapter(pack,device=device);weight_sha=frontend["detector_sha256"]
    rows=read_index_v2("jaad",root=root);clips={r["scene_id"]:r for r in rows};cache=root/f"data/v2/forecast/jaad/detections-{weight_sha[:12]}-threshold{threshold:g}.jsonl";cache.parent.mkdir(parents=True,exist_ok=True)
    cache_metadata=cache.with_suffix(".manifest.json");cache_identity={**frontend,"sample_rate_hz":10,"device":device}
    if cache.exists():
        cached=json.loads(cache_metadata.read_text())
        if cached["identity"]!=cache_identity or cached["cache_sha256"]!=sha(cache):raise ValueError("Fixed sensor cache identity/checksum mismatch")
    if not cache.exists():
        partial=cache.with_suffix(".part")
        with partial.open("w") as f:
            for clip,row in sorted(clips.items()):
                cap=cv2.VideoCapture(str(root/row["video_path"]));fps=float(cap.get(cv2.CAP_PROP_FPS));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));chosen=set(round(t*fps) for t in np.arange(0,(n-1)/fps+1e-9,.1));count=0
                for i in range(n):
                    ok,bgr=cap.read()
                    if not ok:raise ValueError("Forecast actualJAAD sensor decode failed")
                    if i not in chosen:continue
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);frame=SensorFrameV1(f"jaad-v2/{clip}/{i:06d}",round(i/fps*1e9),rgb);detections=[d for d in adapter.predict(frame).detections if d.class_id==1 and d.score>=threshold]
                    record={"scene_id":clip,"split":row["split"],"frame_index":i,"timestamp":i/fps,"fps":fps,"width":rgb.shape[1],"height":rgb.shape[0],"RGB_decoded_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),"detections":[{"bbox":d.bbox,"score":d.score,"class_id":d.class_id} for d in detections]};f.write(json.dumps(record,separators=(",",":"))+"\n");count+=1
                cap.release();f.flush();print(json.dumps({"event":"forecast_real_sensor_clip","clip":clip,"split":row["split"],"frames":count}),flush=True)
        partial.replace(cache);json_write(cache_metadata,{"identity":cache_identity,"cache_sha256":sha(cache),"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"targets_opened_during_sensor_inference":False})
    adapter.close();observations={}
    for line in cache.read_text().splitlines():
        r=json.loads(line);observations.setdefault(r["scene_id"],[]).append(r)
    samples=[];denominator={s:0 for s in ["train","val","test"]};matched={s:0 for s in denominator};past_gaps={s:0 for s in denominator};source_hashes={}
    for clip,row in sorted(clips.items()):
        annotation_path=root/row["targets_path"];a=json.loads(annotation_path.read_text());labels=_label_tracks(a);source_hashes[clip]=sha(annotation_path);tracker=make_tracker(frontend["tracker_configuration"]);history={}
        for j,r in enumerate(observations[clip]):
            t=r["timestamp"];tracks=tracker.update([Detection(**d) for d in r["detections"]],round(t*1e9))
            for track in tracks:
                if track.observed:
                    b=track.bbox;history.setdefault(track.id,[]).append((t,[(b[0]+b[2])/2/a["width"],(b[1]+b[3])/2/a["height"]]))
            if j%2 or t<1:continue
            eligible={}
            # All annotatedeligible GT windows are counted before anydetectedpast.
            for tid,values in labels.items():
                times=[v[0] for v in values];positions=[v[1] for v in values]
                target=interpolate_labels(times,positions,t+FUTURE_TIMES,1.1/a["fps"]);past=interpolate_labels(times,positions,t+np.linspace(-1,0,11),1.1/a["fps"])
                if target is not None and past is not None:eligible[tid]=target
            denominator[row["split"]]+=len(eligible)
            candidates=[gt for gt in a["frames"].get(str(r["frame_index"]),[]) if gt["track_id"] in eligible and not gt["ignore"] and not gt["iscrowd"]]
            pairs=sorted([(bbox_iou(track.bbox,gt["bbox_xyxy"]),track,gt) for track in tracks if track.observed for gt in candidates],key=lambda p:p[0],reverse=True);used_t=set();used_gt=set()
            for iou,track,gt in pairs:
                if iou<.3 or track.id in used_t or gt["track_id"] in used_gt:continue
                used_t.add(track.id);used_gt.add(gt["track_id"]);v=history[track.id];past=resample_history([p[0] for p in v],[p[1] for p in v],t)
                if past is None:past_gaps[row["split"]]+=1;continue
                matched[row["split"]]+=1;samples.append({"sample_id":f"jaad-v2/{clip}/{r['frame_index']}/{track.id}","scene_id":clip,"split":row["split"],"anchor_time_s":t,"anchor_frame":r["frame_index"],"history_source":"final_detector_RGB_trackedpast","tracker_id":track.id,"target_id_evaluator_only":gt["track_id"],"history":past,"future":eligible[gt["track_id"]]})
    coverage={s:{"all_annotated_eligible_windows":denominator[s],"matched_observed_history_windows":matched[s],"coverage_fraction":matched[s]/denominator[s] if denominator[s] else None,"matched_anchor_insufficientpast":past_gaps[s]} for s in denominator}
    return _save_motion(root,"jaad",samples,{"domain":"JAADv2 real imageplane","coordinate_system":"image_xy_divided_by_width_height","units":"normalized_image","history_source":"final_detector_RGB_selectedtrackerpast","detector_pack":str(pack.relative_to(root)),**frontend,"detector_threshold":threshold,"detector_device":device,"sensor_cache_sha256":sha(cache),"source_detection_index_sha256":sha(root/"data/v2/jaad/index.jsonl"),"annotation_sha256":source_hashes,"coverage":coverage,"test_predictions_scope":"future forecast positions only; fixed selected detector supplies test sensor observations","fixed_frontend_test_sensor_predictions_computed":True,"eligibility_definition":"continuousnonignoredpersonGTboxes over1s history+2s future; no detection/match prerequisite","anchor_join":"one-to-one IoU>=.3 evaluatoronly; tracker receivesnoGTIDs","limitations":["image-plane includesegomotion; notmetres","conditionalADE mustbe reportedwith coverage","overlappingwindows correlated; clipgroups notoriginalrecordings"]})


def train_forecast_v2(domain="fixture",seed=42,lr=.001,max_steps=6000,epochs=30,resume=None,preset="primary",stop_steps=None):
    import torch
    from robot_model_sdk.quality_models import residual_motion_network
    root=root_path();canonical_path=root/f"models/v2/selection-forecast-{domain}.json"
    if preset=="primary" and canonical_path.exists() and not json.loads(canonical_path.read_text()).get("test_closed",True):raise RuntimeError("Forecast primary training frozen after canonical testopen; no overwrite/config additions")
    seed_all(seed);h,y,rows,meta,dataset=load_forecast_data(root,domain);train=np.array([i for i,r in enumerate(rows) if r["split"]=="train"]);val=np.array([i for i,r in enumerate(rows) if r["split"]=="val"])
    if not 1<=max_steps<=6000 or not 1<=epochs<=30 or min(len(train),len(val))==0:raise ValueError("V2 training limits orsplit invalid")
    if preset=="primary" and (seed not in {42,43,44} or lr not in {.001,.0003,.0001}):raise ValueError("Primary forecast uses the frozen three learning rates and seeds42/43/44")
    scale=train_scale(h[train]);origin=h[:,-1:,:];x=torch.from_numpy((h-origin)/scale);target=torch.from_numpy((y-origin)/scale)
    model=residual_motion_network(history=11,future=20,hidden=64);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler_parameters={"mode":"min","factor":.5,"patience":2,"threshold":1e-4,"threshold_mode":"rel","cooldown":0,"min_lr":1e-6,"eps":1e-8};scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,**scheduler_parameters);sampler=torch.Generator().manual_seed(seed+400000)
    config={"domain":domain,"seed":seed,"lr":lr,"epochs":epochs,"max_steps":max_steps,"batch_size":2,"optimizer":"AdamW","weight_decay":1e-4,"patience":5,"minimum_epochs":3,"scheduler":scheduler_parameters,"architecture":"residual_cv_mlp_2x64","network":{"history":11,"future":20,"hidden":64},"preset":preset,"initial_checkpoint_eligible":False,"blinding":"splittrainvalonly","validation_interval_steps":200,"safe_resume_weights_only":True}
    resume_state=None;dataset_sha=sha(dataset/"manifest.json")
    if resume:
        resume_state=torch.load(resume,map_location="cpu",weights_only=True)
        if resume_state["config"]!=config or resume_state["dataset_manifest_sha256"]!=dataset_sha or not np.array_equal(resume_state["normalization_scale"],scale):raise ValueError("Strictresume configuration/data/normalization mismatch")
    runid=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+f"-{domain}-seed{seed}-"+uuid.uuid4().hex[:6];run=root/"reports/v2/forecast/runs"/runid;run.mkdir(parents=True);log=run/"events.jsonl"
    record={"schema_version":2,"run_id":runid,"config":config,"dataset_manifest_sha256":sha(dataset/"manifest.json"),"command":shlex.join([sys.executable,*sys.argv]),"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":"running","test_access":False,"source_sha256":sha(__file__),"SDK_source_sha256":sha(root/"packages/robot_model_sdk/src/robot_model_sdk/quality_models.py"),**_runtime_provenance(root,run)};json_write(run/"run.json",record)
    def event(**fields):
        with log.open("a") as f:f.write(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields},allow_nan=False)+"\n")
    step=0;epoch=0;cursor=0;order=None;best=float("inf");best_state=None;best_step=0;bad=0
    if resume:
        cp=resume_state
        model.load_state_dict(cp["model"]);optimizer.load_state_dict(cp["optimizer"]);scheduler.load_state_dict(cp["scheduler"]);sampler.set_state(cp["sampler_rng"]);torch.set_rng_state(cp["torch_rng"])
        nr=cp["numpy_rng"];np.random.set_state((nr[0],np.array(nr[1],np.uint32),nr[2],nr[3],nr[4]));random.setstate(cp["python_rng"])
        step=cp["step"];epoch=cp["epoch"];cursor=cp["sampler_cursor"];order=None if cp["sampler_order"] is None else cp["sampler_order"].numpy();best=cp["best_val_ADE2"];best_state=cp["best_model"];best_step=cp["best_step"];bad=cp["early_stop_bad"]
        event(event="resume",checkpoint_sha256=sha(resume),step=step,restored=["model","optimizer","scheduler","sampler_rng","sampler_order","sampler_cursor","torch_rng","numpy_rng","python_rng","early_stop_bad"])
    def validate():
        model.eval()
        with torch.inference_mode():pred=model(x[val]).numpy()*scale+origin[val]
        return displacement_metrics(pred,y[val],[rows[i]["scene_id"] for i in val])
    def save_checkpoint():
        nr=np.random.get_state();safe_nr=(nr[0],nr[1].tolist(),int(nr[2]),int(nr[3]),float(nr[4]))
        torch.save({"config":config,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"sampler_rng":sampler.get_state(),"sampler_order":None if order is None else torch.from_numpy(order.copy()),"sampler_cursor":cursor,"torch_rng":torch.get_rng_state(),"numpy_rng":safe_nr,"python_rng":random.getstate(),"early_stop_bad":bad,"best_val_ADE2":best,"best_model":best_state,"best_step":best_step,"step":step,"epoch":epoch,"normalization_scale":torch.from_numpy(scale.copy()),"dataset_manifest_sha256":record["dataset_manifest_sha256"]},run/"checkpoint.pt")
    event(event="initial_validation_diagnostic",metrics=validate(),eligible=False);start=time.time();loss_sum=0.;seen=0;end_step=min(max_steps,stop_steps) if stop_steps is not None else max_steps
    while step<end_step and epoch<epochs and not(epoch>=3 and bad>=5):
        if order is None:order=train[torch.randperm(len(train),generator=sampler).numpy()];cursor=0
        ids=order[cursor:cursor+2];model.train();optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(x[ids]),target[ids])
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite residualloss")
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
        if not torch.isfinite(norm):raise FloatingPointError("Nonfinite residualgradient")
        optimizer.step();step+=1;cursor+=len(ids);loss_sum+=float(loss.detach())*len(ids);seen+=len(ids);boundary=cursor==len(order)
        if boundary:epoch+=1;order=None;cursor=0
        if step==30 or step%200==0 or boundary or step==end_step:
            import psutil
            rss=psutil.Process().memory_info().rss
            if rss>12*1024**3:raise MemoryError("Forecast training process exceeded12GiB")
            event(event="resource_measurement",step=step,rss_bytes=rss,elapsed_s=time.time()-start)
        if step%200==0 or boundary or step==end_step:
            vm=validate();score=vm["horizons"]["2.0"]["ADE"]
            if score<best-1e-9:best=score;best_step=step;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};bad=0
            else:bad+=1
            scheduler.step(score);event(event="measurement",step=step,epoch=epoch,train_loss=loss_sum/max(seen,1),validation=vm,lr=optimizer.param_groups[0]["lr"],bad_count=bad,elapsed_s=time.time()-start);save_checkpoint();loss_sum=0.;seen=0
    save_checkpoint()
    if best_state is None or best_step<=0:raise RuntimeError("No actualtrained selectablecheckpoint")
    model.load_state_dict(best_state);pack=root/"models/v2"/f"forecast-{domain}-seed{seed}-lr{lr:g}-{preset}";pack.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict()},pack/"weights.pt")
    manifest={"schema_version":1,"protocol_id":"quality-v2","preprocessing_id":"motion_center_scale_v2","task":"forecast","architecture":"residual_cv_mlp_2x64","network":{"history":11,"future":20,"hidden":64},"architecture_parameters":{"history":11,"future":20,"hidden":64},"domain":meta["domain"],"units":meta["units"],"coordinate_system":meta["coordinate_system"],"history_source":meta["history_source"],"native_weights":"weights.pt","sha256":sha(pack/"weights.pt"),
            "preprocess":{"center_last":True,"scale":scale.tolist(),"predicts_residual_cv":True,"normalization_fit_split":"train","history_span_s":1.,"sample_rate_hz":10},"postprocess":{"unscale":True,"relative_to_last":True,"future_points":20,"future_times_s":FUTURE_TIMES.tolist()},"run_id":runid,"seed":seed,"best_step":best_step,"completed_steps":step,"config":config,"dataset_manifest_sha256":record["dataset_manifest_sha256"],"weights_terms":"ourtrainedMLP; originaldataexcluded","experiment_protocol":{"path":"configs/v2/protocol.json","sha256":sha(root/"configs/v2/protocol.json"),"trainval_target_access_only":True},"backend_validation":{"architecture_probe_report":"reports/v2/backend-models.json","architecture_probe_sha256":sha(root/"reports/v2/backend-models.json"),"scope":"generic CPU/MPS architecture forward/backward, fallback0; actual trained-weight execution recorded separately"}};json_write(pack/"manifest.json",manifest)
    metrics={"best_validation_ADE2":best,"best_step":best_step,"completed_steps":step,"completed_epochs":epoch,"epochs_started":epoch+(1 if cursor else 0),"partial_epoch_cursor":cursor,"partial_epoch_total_windows":len(order) if order is not None else 0,"test_targets_loaded":False,"test_evaluated":False,"pack":str(pack.relative_to(root)),"run":str(run.relative_to(root)),"elapsed_s":time.time()-start};json_write(run/"metrics.json",metrics);record.update(status="completed",finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),best_step=best_step,completed_steps=step);json_write(run/"run.json",record);print(json.dumps({"event":"forecast_v2_training_done",**metrics}),flush=True);return str(pack)


def _completed_primary(root,domain,seed,lr,dataset_sha):
    root=Path(root);pack=root/f"models/v2/forecast-{domain}-seed{seed}-lr{lr:g}-primary"
    if not pack.exists():return None
    try:
        m=json.loads((pack/"manifest.json").read_text());config=m["config"];run=root/"reports/v2/forecast/runs"/m["run_id"];record=json.loads((run/"run.json").read_text());metrics=json.loads((run/"metrics.json").read_text());steps=m["completed_steps"];best=m["best_step"]
        valid=(m["seed"]==config["seed"]==seed and config["domain"]==domain and config["preset"]=="primary" and config["lr"]==lr and config["epochs"]==30 and config["max_steps"]==6000 and m["architecture"]=="residual_cv_mlp_2x64" and m["network"]=={"history":11,"future":20,"hidden":64} and m["dataset_manifest_sha256"]==record["dataset_manifest_sha256"]==dataset_sha and record["config"]==config and record["status"]=="completed" and 0<best<=steps<=6000 and metrics["best_step"]==best and metrics["completed_steps"]==steps and np.isfinite(metrics["best_validation_ADE2"]) and m["sha256"]==sha(pack/"weights.pt"))
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as error:raise ValueError(f"Existing primary forecast artifact incomplete/malformed; retained:{pack}") from error
    if not valid:raise ValueError(f"Existing primary forecast artifact does not match fixed completed trial; retained:{pack}")
    return pack


def select_forecast_v2(domain="fixture"):
    root=root_path();canonical_path=root/f"models/v2/selection-forecast-{domain}.json"
    dataset_sha=sha(root/f"data/v2/forecast/{domain}/manifest.json")
    if canonical_path.exists():
        canonical=json.loads(canonical_path.read_text())
        if canonical["dataset_manifest_sha256"]!=dataset_sha:raise ValueError("Frozen forecast selection dataset changed")
        for binding in [canonical["selected"],*canonical["repeated_seeds"]]:
            pack=root/binding["pack"]
            if binding["checkpoint_sha256"]!=sha(pack/"weights.pt") or binding["manifest_sha256"]!=sha(pack/"manifest.json"):raise ValueError("Frozen forecast selected artifact changed")
            complete=_completed_primary(root,domain,binding["seed"],canonical["selected"]["lr"],dataset_sha)
            if complete is None or complete.resolve()!=pack.resolve():raise ValueError("Locked forecast trial is not exact completed primary artifact")
        present={b["seed"] for b in canonical["repeated_seeds"]}
        if not canonical.get("test_closed",True) and present!={43,44}:raise RuntimeError("Opened forecast selection cannot add or retrain missing repeats")
        for seed in [43,44]:
            if seed in present:continue
            pack=_completed_primary(root,domain,seed,canonical["selected"]["lr"],dataset_sha)
            if pack is None:pack=Path(train_forecast_v2(domain,seed,canonical["selected"]["lr"]))
            canonical["repeated_seeds"].append({"seed":seed,"pack":str(pack.relative_to(root)),"checkpoint_sha256":sha(pack/"weights.pt"),"manifest_sha256":sha(pack/"manifest.json"),"dataset_manifest_sha256":dataset_sha});json_write(canonical_path,canonical)
        return canonical
    configs=[]
    for lr in [.001,.0003,.0001]:
        pack=_completed_primary(root,domain,42,lr,dataset_sha)
        if pack is None:pack=Path(train_forecast_v2(domain,42,lr))
        m=json.loads((pack/"manifest.json").read_text());run=json.loads((root/"reports/v2/forecast/runs"/m["run_id"]/"metrics.json").read_text());configs.append({"lr":lr,"pack":str(pack.relative_to(root)),"validation_ADE2":run["best_validation_ADE2"],"best_step":m["best_step"]})
    selected=min(configs,key=lambda c:c["validation_ADE2"]);selected.update(seed=42,checkpoint_sha256=sha(root/selected["pack"]/"weights.pt"),manifest_sha256=sha(root/selected["pack"]/"manifest.json"));dataset_sha=sha(root/f"data/v2/forecast/{domain}/manifest.json")
    record={"schema_version":2,"domain":domain,"selection_rule":"validationonlyADE2seed42; threefixedLR; trainedcheckpointsstep>0","configs":configs,"selected":selected,"dataset_manifest_sha256":dataset_sha,"locked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"test_closed":True,"repeated_seeds":[]};path=root/"models/v2"/f"selection-forecast-{domain}.json";json_write(path,record)
    for seed in [43,44]:
        pack=_completed_primary(root,domain,seed,selected["lr"],dataset_sha)
        if pack is None:pack=Path(train_forecast_v2(domain,seed,selected["lr"]))
        record["repeated_seeds"].append({"seed":seed,"pack":str(pack.relative_to(root)),"checkpoint_sha256":sha(pack/"weights.pt"),"manifest_sha256":sha(pack/"manifest.json"),"dataset_manifest_sha256":dataset_sha});json_write(path,record)
    return record


def evaluate_forecast_v2(pack,domain="fixture",split="val",selection_lock=None):
    from robot_model_sdk.models import load_adapter
    import torch
    torch.set_num_threads(3)
    root=root_path();pack=Path(pack);pack=pack if pack.is_absolute() else root/pack;started=datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.time();canonical_sha=None
    if split=="test":
        if selection_lock is None:raise PermissionError("V2 forecasttestclosed until explicitcheckpointselectionlock")
        canonical=json.loads((root/f"models/v2/selection-forecast-{domain}.json").read_text())
        if canonical.get("test_closed",True):raise PermissionError("Canonical forecast selection remains testclosed")
        lock=json.loads(Path(selection_lock).read_text()) if not isinstance(selection_lock,dict) else selection_lock
        if not isinstance(lock,dict):raise ValueError("Forecast selection lock must be an object")
        if "locks" in lock:
            matches=[entry for entry in lock["locks"] if (root/entry["pack"]).resolve()==pack.resolve()]
            if len(matches)!=1:raise ValueError("Forecast lock collection must bind this exact pack once")
            lock=matches[0]
        if lock.get("checkpoint_sha256")!=sha(pack/"weights.pt") or lock.get("domain")!=domain:raise ValueError("Forecasttestlock pack/domain mismatch")
        allowed=[canonical["selected"],*canonical["repeated_seeds"]];binding=next((b for b in allowed if (root/b["pack"]).resolve()==pack.resolve()),None)
        if binding is None or binding["checkpoint_sha256"]!=sha(pack/"weights.pt"):raise ValueError("Unselected forecast checkpoint cannot opentest")
        if binding["manifest_sha256"]!=sha(pack/"manifest.json"):raise ValueError("Forecast selectedmanifest changed")
        current_dataset_sha=sha(root/f"data/v2/forecast/{domain}/manifest.json")
        m=json.loads((pack/"manifest.json").read_text())
        if current_dataset_sha!=canonical["dataset_manifest_sha256"] or m.get("dataset_manifest_sha256")!=current_dataset_sha:raise ValueError("Forecast lock dataset fingerprint changed")
        configuration=m.get("config",{});seed=m.get("seed");best_step=m.get("best_step",0)
        if not isinstance(configuration,dict) or seed!=binding["seed"] or configuration.get("seed")!=binding["seed"] or seed not in {42,43,44} or configuration.get("domain")!=domain or configuration.get("preset")!="primary" or configuration.get("lr")!=canonical["selected"]["lr"] or not isinstance(best_step,int) or best_step<=0:raise ValueError("Forecast lock seed/config/step invalid")
        canonical_sha=sha(root/f"models/v2/selection-forecast-{domain}.json")
    runid=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+f"-evaluate-{domain}-{split}-"+uuid.uuid4().hex[:6];run=root/"reports/v2/forecast/evaluations"/runid;run.mkdir(parents=True);provenance=_runtime_provenance(root,run);provenance["source_capture_phase"]="before target loading/prediction/evaluation"
    h,y,rows,meta,dataset=load_forecast_data(root,domain,(split,));ids=np.arange(len(rows));hist=h;target=y;adapter=load_adapter(pack)
    if adapter.metadata["units"]!=meta["units"] or adapter.metadata["coordinate_system"]!=meta["coordinate_system"]:raise ValueError("Forecastunitsmismatch")
    preds=baselines_v2(hist);preds["residual_MLP"]=adapter.predict_history(hist);adapter.close()
    result={"split":split,"units":meta["units"],"history_source":meta["history_source"],"identical_observed_histories":True,"checkpoint_sha256":sha(pack/"weights.pt"),"metrics":{name:displacement_metrics(p,target,[rows[i]["scene_id"] for i in ids]) for name,p in preds.items()},"coverage":meta.get("coverage"),"evaluation_run_id":runid}
    after={p:sha(root/p) for p in _provenance_sources(root)};unchanged=after==provenance["source_hashes"]
    m=json.loads((pack/"manifest.json").read_text());record={"schema_version":2,"run_id":runid,"task":"forecast_evaluation","domain":domain,"split":split,"seed":m["seed"],"pack":str(pack.relative_to(root)),"training_run_id":m["run_id"],"dataset_manifest_sha256":sha(dataset/"manifest.json"),"checkpoint_sha256":sha(pack/"weights.pt"),"model_manifest_sha256":sha(pack/"manifest.json"),"canonical_selection_sha256":canonical_sha,"command":shlex.join([sys.executable,*sys.argv]),"api_invocation":{"function":"evaluate_forecast_v2","domain":domain,"split":split,"pack":str(pack.relative_to(root)),"selection_lock":str(selection_lock) if not isinstance(selection_lock,dict) else selection_lock},"started_utc":started,"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"elapsed_s":time.time()-start,"windows":len(rows),"scene_count":len({r["scene_id"] for r in rows}),"status":"completed" if unchanged else "source_changed_during_execution","source_hashes_after_execution":after,"source_unchanged_during_execution":unchanged,**provenance}
    json_write(run/"metrics.json",result);json_write(run/"run.json",record);return result


def resume_audit_v2(domain="fixture"):
    import torch
    root=root_path();full=Path(train_forecast_v2(domain,137,.001,max_steps=800,preset="resume-audit"));full_manifest=json.loads((full/"manifest.json").read_text());full_run=root/"reports/v2/forecast/runs"/full_manifest["run_id"]
    original=torch.load(full_run/"checkpoint.pt",weights_only=True)
    partial=Path(train_forecast_v2(domain,137,.001,max_steps=800,preset="resume-audit",stop_steps=200));pm=json.loads((partial/"manifest.json").read_text());partial_run=root/"reports/v2/forecast/runs"/pm["run_id"]
    resumed=Path(train_forecast_v2(domain,137,.001,max_steps=800,preset="resume-audit",resume=partial_run/"checkpoint.pt"));rm=json.loads((resumed/"manifest.json").read_text());resumed_run=root/"reports/v2/forecast/runs"/rm["run_id"];final=torch.load(resumed_run/"checkpoint.pt",weights_only=True)
    def same(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    keys=["model","best_model","optimizer","scheduler","sampler_rng","sampler_order","sampler_cursor","torch_rng","numpy_rng","python_rng","early_stop_bad","best_step","step","epoch"]
    checks={k:same(original[k],final[k]) for k in keys};result={"domain":domain,"seed":137,"role":"CPU resume mechanics only; no modelselection or testprediction","interruption_step":200,"final_step":800,"safe_weights_only_loading":True,"checks":checks,"all_bitwise_equal":all(checks.values()),"uninterrupted_run":str(full_run.relative_to(root)),"partial_run":str(partial_run.relative_to(root)),"resumed_run":str(resumed_run.relative_to(root))}
    json_write(root/f"reports/v2/forecast/{domain}-resume-audit.json",result)
    if not result["all_bitwise_equal"]:raise AssertionError("V2 resumed CPU state divergence")
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["build-fixture","build-jaad","train","select","evaluate","resume-audit"]);p.add_argument("--domain",default="fixture");p.add_argument("--pack");p.add_argument("--threshold",type=float);p.add_argument("--device",default="cpu");p.add_argument("--split",default="val");p.add_argument("--selection-lock");p.add_argument("--seed",type=int,default=42);p.add_argument("--lr",type=float,default=.001);p.add_argument("--epochs",type=int,default=30);p.add_argument("--max-steps",type=int,default=6000);p.add_argument("--configuration",default="primary");p.add_argument("--resume");a=p.parse_args()
    if a.action=="build-fixture":print(build_fixture_v2(),flush=True)
    elif a.action=="build-jaad":print(build_jaad_forecast_v2(a.pack,a.threshold,a.device),flush=True)
    elif a.action=="train":print(train_forecast_v2(a.domain,a.seed,a.lr,a.max_steps,a.epochs,a.resume,a.configuration),flush=True)
    elif a.action=="select":print(json.dumps(select_forecast_v2(a.domain)),flush=True)
    elif a.action=="resume-audit":print(json.dumps(resume_audit_v2(a.domain)),flush=True)
    else:print(json.dumps(evaluate_forecast_v2(a.pack,a.domain,a.split,a.selection_lock)),flush=True)


if __name__=="__main__":main()
