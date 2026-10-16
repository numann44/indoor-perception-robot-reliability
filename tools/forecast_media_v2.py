#!/usr/bin/env python3
"""Sensor-only JAAD validation chapter assets for the existing demo assembler.

Every10Hz sensor frame reaches PerceptionPipeline. PNG panels are sampled3Hz;
playback is governed by actual source timestamps, never inference wall time.
No annotation file or future-target array is read by this producer.
"""
from __future__ import annotations
import argparse,collections,datetime,hashlib,json,os,platform,shlex,shutil,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from robot_model_sdk import SensorFrameV1
from indoor_perception_lab.data_v2 import read_index_v2
from indoor_perception_lab.experiments import root_path,json_write,sha
from indoor_perception_lab.forecast_v2 import _jaad_frontend_identity,_runtime_provenance,baselines_v2
from indoor_perception_lab.motion_data import resample_history
from indoor_perception_lab.pipeline import PerceptionPipeline
from indoor_perception_lab.tracking_v2 import make_tracker


def select_validation_clip(root,clip=None,duration=15.):
    if duration<=0:raise ValueError("Positive source duration required")
    clips={r["scene_id"]:r for r in read_index_v2("jaad","val",root)};inventory=[]
    for name,row in sorted(clips.items()):
        capture=cv2.VideoCapture(str(Path(root)/row["video_path"]));fps=float(capture.get(cv2.CAP_PROP_FPS));count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release()
        if fps<=0 or count<=0:raise ValueError("Invalid source video metadata")
        inventory.append({"clip":name,"video_path":row["video_path"],"fps":fps,"frames":count,"duration_s":count/fps})
    if clip is not None:
        candidates=[r for r in inventory if r["clip"]==clip]
        if len(candidates)!=1:raise ValueError("Chapter must use an official retained validation clip")
        chosen=candidates[0]
    else:chosen=min(inventory,key=lambda r:(-r["duration_s"],r["clip"]))
    return {**chosen,"chapter_duration_s":min(duration,chosen["duration_s"]),"selection_rule":"longest retained validation source duration; clip name tie-break; no labels/predictions"},inventory


def render_panel(rgb,result,ls):
    image=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(1280,720),interpolation=cv2.INTER_AREA);scale=np.array([1280/rgb.shape[1],720/rgb.shape[0]])
    for track in result["tracks"]:
        if not track["observed"]:continue  # Keep lost-track counts/raw JSON; avoid overlapping ghost labels.
        box=np.asarray(track["bbox"])*np.tile(scale,2);x0,y0,x1,y1=map(int,box);color=(220,210,60) if track["observed"] else (145,145,145);cv2.rectangle(image,(x0,y0),(x1,y1),color,2);cv2.putText(image,track["id"]+("" if track["observed"] else " unobserved"),(max(0,x0),max(102,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
        origin=np.array([(track["bbox"][0]+track["bbox"][2])/2,(track["bbox"][1]+track["bbox"][3])/2])
        for future,color in [(ls.get(track["id"]),(220,110,210)),(track["forecasts"].get("MLP2x64"),(100,245,80))]:
            if not future:continue
            points=np.array([origin,*[future[key] for key in sorted(future,key=float)]],float)*scale
            if not np.isfinite(points).all():raise ValueError("Nonfinite displayed forecast")
            cv2.polylines(image,[np.round(points).astype(np.int32)],False,color,2,cv2.LINE_AA)
            for point in points[1:]:cv2.circle(image,tuple(np.round(point).astype(int)),4,color,-1,cv2.LINE_AA)
    overlay=image.copy();cv2.rectangle(overlay,(0,0),(1280,93),(15,24,31),-1);cv2.rectangle(overlay,(0,657),(1280,720),(15,24,31),-1);image=cv2.addWeighted(overlay,.88,image,.12,0)
    counts=collections.Counter(t["learned_forecast_status"] for t in result["tracks"])
    cv2.putText(image,"JAAD | RECORDED RGB | LEARNED IMAGE FORECAST",(18,31),cv2.FONT_HERSHEY_SIMPLEX,.83,(240,247,250),2,cv2.LINE_AA)
    cv2.putText(image,f"{result['sample_id']}  source {result['timestamp_ns']/1e9:.2f}s  |  available {counts['available']}  history<1s/gaps {counts['insufficient_causal_history']}  unobserved {counts['unavailable_unobserved_track']}",(18,60),cv2.FONT_HERSHEY_SIMPLEX,.46,(195,213,225),1,cv2.LINE_AA)
    cv2.putText(image,"Green: trained residual MLP  |  Magenta: LS-CV on the same resampled observed past",(18,682),cv2.FONT_HERSHEY_SIMPLEX,.55,(205,231,234),1,cv2.LINE_AA)
    cv2.putText(image,"1s causal history; 0.5 / 1 / 2s forecasts; model normalized XY displayed in pixels; RGB-only; no target inputs",(18,706),cv2.FONT_HERSHEY_SIMPLEX,.47,(162,185,201),1,cv2.LINE_AA)
    return image


def produce(detector,forecaster,device="mps",clip=None,duration=15.,output="reports/v2/forecast/media/jaad-validation"):
    root=root_path();out=(root/output).resolve()
    if not out.is_relative_to(root/"reports/v2/forecast/media"):raise ValueError("Chapter writes must remain in owned project media directory")
    if out.exists() and (out/"chapter-manifest.json").exists():raise FileExistsError("Completed chapter preserved; choose another output for a new run")
    detector=Path(detector);detector=detector if detector.is_absolute() else root/detector;forecaster=Path(forecaster);forecaster=forecaster if forecaster.is_absolute() else root/forecaster;identity=_jaad_frontend_identity(root,detector)
    selection=json.loads((root/"models/v2/selection-forecast-jaad.json").read_text());binding=selection["selected"]
    if selection.get("test_closed",True) or (root/binding["pack"]).resolve()!=forecaster.resolve() or binding["checkpoint_sha256"]!=sha(forecaster/"weights.pt") or binding["manifest_sha256"]!=sha(forecaster/"manifest.json"):raise ValueError("Chapter requires the frozen primary seed42 forecast after authorized heldout evaluation")
    source,inventory=select_validation_clip(root,clip,duration);out.mkdir(parents=True,exist_ok=True)
    for name in ["panels","results","panel-overlays"]:(out/name).mkdir(exist_ok=True)
    torch.set_num_threads(3);cv2.setNumThreads(1);started=datetime.datetime.now(datetime.timezone.utc).isoformat();wall=time.monotonic();pipe=PerceptionPipeline(detector=detector,device=device,score_threshold=identity["threshold"],use_depth=False,forecaster=forecaster,tracker=make_tracker(identity["tracker_configuration"]))
    fps=source["fps"];sensor_indices=sorted({min(source["frames"]-1,round(t*fps)) for t in np.arange(0,source["chapter_duration_s"]-1e-9,.1)});panel_slots={round(k*10/3) for k in range(int(np.ceil(source["chapter_duration_s"]*3)))};capture=cv2.VideoCapture(str(root/source["video_path"]));chosen=set(sensor_indices);entries=[];stream_hash=hashlib.sha256();status_counts=collections.Counter();processed=0;available_frames=0;first_available=None
    with (out/"processed-frames.jsonl").open("w") as stream:
        try:
            for i in range(sensor_indices[-1]+1):
                ok,bgr=capture.read()
                if not ok:raise ValueError("Source decode failed before requested sensor span")
                if i not in chosen:continue
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);timestamp_ns=round(i/fps*1e9);frame=SensorFrameV1(f"jaad-v2/{source['clip']}/{i:06d}",timestamp_ns,rgb);result=pipe.process(frame);result_path=out/"results"/f"{processed:05d}.json";json_write(result_path,result);digest=hashlib.sha256(rgb.tobytes()).hexdigest();stream_hash.update(digest.encode());statuses=collections.Counter(t["learned_forecast_status"] for t in result["tracks"]);status_counts.update(statuses)
                if statuses["available"]:
                    available_frames+=1
                    if first_available is None:first_available=processed
                row={"sample_id":frame.sample_id,"timestamp_ns":timestamp_ns,"frame_index":i,"time_s":i/fps,"rgb_decoded_sha256":digest,"result_ref":str(result_path.relative_to(out)),"depth_available":False,"pose_source":None,"forecast_status_counts":dict(statuses)};stream.write(json.dumps(row)+"\n")
                if processed in panel_slots:
                    ls={}
                    for track in pipe.tracker.tracks:
                        if not track.observed or not track.history:continue
                        values=track.history;history=resample_history([h[0] for h in values],[h[1] for h in values],timestamp_ns/1e9)
                        if history is not None:
                            prediction=baselines_v2(history)["constant_velocity_LS"][0];ls[track.id]={str(h):prediction[round(h*10)-1].tolist() for h in [.5,1,2]}
                    panel=out/"panels"/f"{processed:05d}.png";cv2.imwrite(str(panel),render_panel(rgb,result,ls));json_write(out/"panel-overlays"/f"{processed:05d}.json",{"LS_CV_sensor_past_only":ls,"raw_pipeline_result":row["result_ref"]});entries.append({**row,"image":str(panel.relative_to(out))})
                processed+=1
        finally:capture.release();pipe.close()
    if processed!=len(sensor_indices):raise AssertionError("Not all requested source sensor frames processed")
    manifest={"schema_version":1,"chapter_id":"jaad-validation-forecast","label":"JAAD recorded pedestrian RGB: learned image-coordinate forecast","source_duration_s":source["chapter_duration_s"],"playback_speed":1.,"frames":entries,"processed_sensor_frames":processed,"all_source_frames_processed":True,"clock_policy":"source video frame_index/FPS timestamps; all selected10Hz inputs processed; ~3FPS visualization sample-and-hold; no wall-time pacing, loops or source-duration padding","source_metadata":{"dataset":"JAADv2","partition":"validation","clip":source["clip"],"original_video":source["video_path"],"video_sha256":sha(root/source["video_path"]),"source_fps":fps,"source_original_frames":source["frames"],"selected_original_frame_indices":sensor_indices,"sensor_sampling_hz":10,"selection_rule":source["selection_rule"],"validation_clip_inventory":inventory,"decoded_RGB_sha256_chain":stream_hash.hexdigest(),"annotation_inputs":False,"future_target_inputs":False},"model_bindings":{"detector":str(detector.relative_to(root)),"forecaster":str(forecaster.relative_to(root)),"forecaster_weights_sha256":sha(forecaster/"weights.pt"),"forecaster_manifest_sha256":sha(forecaster/"manifest.json"),**identity},"forecast_domain":{"model_units":"normalized_image","display_units":"pixels","coordinates":"image XY divided by original width/height in model; converted back to original pixel coordinates by PerceptionPipeline","world_motion_available":False},"summary":{"frames_with_learned_forecast":available_frames,"first_available_processed_frame":first_available,"track_status_occurrences":dict(status_counts),"panels":len(entries)},"formal_performance_measurement":False}
    schema=json.loads((root/"reports/v2/robotics/chapter-manifest-schema.json").read_text())
    try:
        import jsonschema
        jsonschema.validate(manifest,schema)
    except ImportError:
        if any(k not in manifest for k in schema["required"]):raise ValueError("Chapter schema required field missing")
        for row in entries:
            if any(k not in row for k in schema["properties"]["frames"]["items"]["required"]):raise ValueError("Chapter frame schema required field missing")
    json_write(out/"chapter-manifest.json",manifest);record={"schema_version":2,"task":"JAAD_validation_sensor_only_forecast_chapter","started_utc":started,"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"elapsed_wall_s":time.monotonic()-wall,"command":shlex.join([sys.executable,*sys.argv]),"device":device,"MPS_fallback":os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),"processed_sensor_frames":processed,"annotations_accessed":False,"future_targets_accessed":False,"chapter_manifest_sha256":sha(out/"chapter-manifest.json"),"results_index_sha256":sha(out/"processed-frames.jsonl"),"producer_sha256":sha(__file__),"pipeline_source_sha256":sha(root/"packages/indoor_perception_lab/src/indoor_perception_lab/pipeline.py"),**_runtime_provenance(root,out,extra_sources=("tools/forecast_media_v2.py","packages/indoor_perception_lab/src/indoor_perception_lab/pipeline.py"))};record["hardware"]["execution_device"]=device;json_write(out/"run.json",record);viewer_wrapper(output);print(json.dumps({"chapter_manifest":str((out/"chapter-manifest.json").relative_to(root)),"duration_s":manifest["source_duration_s"],"processed_sensor_frames":processed,"panels":len(entries),**manifest["summary"]}),flush=True);return manifest


def rerender(output):
    """Presentation-only revision from saved pipeline outputs; zero NN calls."""
    root=root_path();out=(root/output).resolve()
    if not out.is_relative_to(root/"reports/v2/forecast/media"):raise ValueError("Owned media directory required")
    manifest=json.loads((out/"chapter-manifest.json").read_text());original=out/"panels-initial"
    if original.exists():raise FileExistsError("Initial panels already preserved; do not overwrite revision history")
    shutil.copytree(out/"panels",original);frames={r["frame_index"]:r for r in manifest["frames"]};capture=cv2.VideoCapture(str(root/manifest["source_metadata"]["original_video"]));count=0
    try:
        for i in range(max(frames)+1):
            ok,bgr=capture.read()
            if not ok:raise ValueError("Rerender source decode failed")
            if i not in frames:continue
            row=frames[i];result=json.loads((out/row["result_ref"]).read_text());slot=Path(row["image"]).stem;overlay=json.loads((out/"panel-overlays"/f"{slot}.json").read_text());rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=row["rgb_decoded_sha256"]:raise ValueError("Rerender source RGB changed")
            cv2.imwrite(str(out/row["image"]),render_panel(rgb,result,overlay["LS_CV_sensor_past_only"]));count+=1
    finally:capture.release()
    json_write(out/"render-revision.json",{"reason":"Observed boxes only for legibility; all lost-track counts and original pipeline JSON preserved","NN_calls":0,"source_RGB_checksums_verified":True,"panels":count,"original_panels":"panels-initial","renderer_sha256":sha(__file__),"command":shlex.join([sys.executable,*sys.argv]),"finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"panel_sha256":{r["image"]:sha(out/r["image"]) for r in manifest["frames"]}})
    shutil.copyfile(__file__,out/"renderer-revision-source.py");viewer_wrapper(output);return count


def viewer_wrapper(output):
    """Verify executed proof and emit consumer fields without rewriting raw proof."""
    root=root_path();out=(root/output).resolve()
    if not out.is_relative_to(root/"reports/v2/forecast/media"):raise ValueError("Owned media directory required")
    original=out/"chapter-manifest.json";original_sha=sha(original);manifest=json.loads(original.read_text());record=json.loads((out/"run.json").read_text());stream=out/"processed-frames.jsonl";rows=[json.loads(s) for s in stream.read_text().splitlines()]
    required=["chapter_id","label","source_duration_s","processed_sensor_frames","clock_policy","all_source_frames_processed","playback_speed","frames"]
    if any(k not in manifest for k in required):raise ValueError("Raw chapter consumer field missing")
    if original_sha!=record["chapter_manifest_sha256"] or sha(stream)!=record["results_index_sha256"]:raise ValueError("Original executed chapter/index proof changed")
    if len(rows)!=record["processed_sensor_frames"] or len(rows)!=manifest["processed_sensor_frames"] or not manifest["all_source_frames_processed"] or manifest["playback_speed"]!=1.:raise ValueError("Executed sensor frame count/clock proof inconsistent")
    if record["annotations_accessed"] or record["future_targets_accessed"]:raise ValueError("Sensor-only chapter input proof required")
    video=root/manifest["source_metadata"]["original_video"]
    if sha(video)!=manifest["source_metadata"]["video_sha256"]:raise ValueError("Source video changed")
    result_sha={};previous=-1
    for row in rows:
        path=(out/row["result_ref"]).resolve()
        if not path.is_relative_to(out):raise ValueError("Result reference escapes chapter")
        result=json.loads(path.read_text())
        if result["timestamp_ns"]!=row["timestamp_ns"] or result["sample_id"]!=row["sample_id"] or row["timestamp_ns"]<=previous:raise ValueError("Executed result/sample clock mismatch")
        previous=row["timestamp_ns"];result_sha[row["result_ref"]]=sha(path)
    revision=json.loads((out/"render-revision.json").read_text()) if (out/"render-revision.json").exists() else None;panel_sha={};previous=-1.
    for frame in manifest["frames"]:
        if any(k not in frame for k in ["time_s","image","result_ref","timestamp_ns","depth_available","pose_source"]):raise ValueError("Chapter frame consumer field missing")
        path=(out/frame["image"]).resolve()
        if not path.is_relative_to(out) or cv2.imread(str(path)) is None or frame["result_ref"] not in result_sha:raise ValueError("Chapter image/result cannot be verified")
        if not previous<frame["time_s"]<=manifest["source_duration_s"]:raise ValueError("Chapter panel clock not increasing/in range")
        previous=frame["time_s"];panel_sha[frame["image"]]=sha(path)
        if revision and revision["panel_sha256"][frame["image"]]!=panel_sha[frame["image"]]:raise ValueError("Panel revision checksum changed")
    wrapper={**manifest,"status":"verified","model_role":"Selected D3/T2 perception and learned residual-MLP image forecasting from causal RGB observations; normalized XY outputs displayed in pixels. LS-CV shares the same resampled observed past. World coordinates and robot control are unavailable in this RGB chapter.","source_original_manifest":"chapter-manifest.json","source_original_manifest_sha256":original_sha,"verification":{"processed_sensor_frames":len(rows),"raw_results_verified":len(result_sha),"panels_decoded":len(panel_sha),"source_video_sha256_verified":True,"processed_index_sha256":sha(stream),"result_sha256":result_sha,"panel_sha256":panel_sha,"run_proof_sha256":sha(out/"run.json"),"render_revision_sha256":sha(out/"render-revision.json") if revision else None,"NN_calls_for_normalization":0,"verified_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}}
    target=out/"viewer-chapter-manifest.json";json_write(target,wrapper)
    if sha(original)!=original_sha:raise AssertionError("Portable original proof mutated")
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument("--detector");p.add_argument("--forecaster");p.add_argument("--rerender",action="store_true");p.add_argument("--normalize",action="store_true");p.add_argument("--device",default="mps");p.add_argument("--clip");p.add_argument("--duration",type=float,default=15.);p.add_argument("--output",default="reports/v2/forecast/media/jaad-validation");a=p.parse_args()
    if a.normalize:print(json.dumps({"viewer_manifest":str(viewer_wrapper(a.output)),"NN_calls":0}));return
    if a.rerender:print(json.dumps({"rerendered_panels":rerender(a.output),"NN_calls":0}));return
    if not a.detector or not a.forecaster:p.error("--detector and --forecaster required for sensor inference")
    produce(a.detector,a.forecaster,a.device,a.clip,a.duration,a.output)


if __name__=="__main__":main()
