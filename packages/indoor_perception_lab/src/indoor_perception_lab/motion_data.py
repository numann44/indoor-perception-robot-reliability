"""Motion datasets with a hard separation between sensor histories and targets.

All predictor arrays are two-dimensional: JAAD image x/y divided by image
width/height, or calibrated fixture world x/z in metres. Ground-truth identities
are used only to join an anchor to its future label, never by the tracker.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time, shutil, shlex, sys
from pathlib import Path
import numpy as np
from robot_model_sdk import Calibration, SensorFrameV1
from robot_model_sdk.models import FixtureAdapter, load_adapter
from .experiments import root_path, sha, json_write
from .geometry import robust_object_position, transform_points
from .tracking import Tracker, bbox_iou

HISTORY_TIMES = np.linspace(-1., 0., 11)
FUTURE_TIMES = np.arange(1, 21, dtype=np.float64) / 10.
HORIZONS = (.5, 1., 2.)


def _disk_guard(root):
    root=Path(root);used=sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink());free=shutil.disk_usage(root).free
    if used>=8_000_000_000 or free<15_000_000_000:raise RuntimeError("Project8GB/free15GB guard exceeded")
    return {"project_bytes":used,"free_disk_bytes":free}


def resample_history(times, positions, anchor_time, max_gap=.21):
    """Return eleven causal 10 Hz observations; reject gaps/extrapolation.

    Future observations are explicitly rejected rather than silently included.
    No interpolation crosses a missing-observation gap greater than max_gap.
    """
    t = np.asarray(times, np.float64); p = np.asarray(positions, np.float64)
    if t.ndim != 1 or p.shape != (len(t), 2):
        raise ValueError("History must have aligned timestamps and Nx2 positions")
    if len(t) < 2:return None
    if not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(np.diff(t) <= 0):
        raise ValueError("History must be finite and strictly increasing")
    if t[-1] > anchor_time + 1e-8:
        raise ValueError("Future observation in predictor history")
    grid = anchor_time + HISTORY_TIMES
    if t[0] > grid[0] + 1e-8 or t[-1] < grid[-1] - 1e-8:
        return None
    relevant = t[(t >= grid[0] - max_gap) & (t <= anchor_time)]
    if np.any(np.diff(relevant) > max_gap + 1e-8): return None
    return np.stack([np.interp(grid, t, p[:, k]) for k in range(2)], 1).astype(np.float32)


def interpolate_labels(times, positions, grid, max_gap):
    t = np.asarray(times, np.float64); p = np.asarray(positions, np.float64)
    if len(t) < 2 or t[0] > grid[0] + 1e-8 or t[-1] < grid[-1] - 1e-8: return None
    low=max(0, np.searchsorted(t, grid[0], side="right")-1)
    high=min(len(t), np.searchsorted(t, grid[-1], side="left")+1)
    if np.any(np.diff(t[low:high]) > max_gap): return None
    return np.stack([np.interp(grid, t, p[:, k]) for k in range(2)], 1).astype(np.float32)


def _save_dataset(root, name, samples, metadata):
    metadata["budget_at_save"]=_disk_guard(root)
    out = Path(root)/"reports/motion/datasets"/name; out.mkdir(parents=True, exist_ok=True)
    history=np.stack([s.pop("history") for s in samples]); future=np.stack([s.pop("future") for s in samples])
    np.savez_compressed(out/"predictor_histories.npz", history=history)
    np.savez_compressed(out/"evaluator_futures.npz", future=future)
    with (out/"samples.jsonl").open("w") as f:
        for s in samples: f.write(json.dumps(s, allow_nan=False)+"\n")
    metadata.update(schema_version=1, name=name, count=len(samples), split_counts={s:sum(x["split"]==s for x in samples) for s in ["train","val","test"]},
                    predictor_sha256=sha(out/"predictor_histories.npz"), target_sha256=sha(out/"evaluator_futures.npz"), index_sha256=sha(out/"samples.jsonl"),
                    protocol={"history_span_s":1.,"sample_rate_hz":10,"history_points":11,"future_points":20,"horizons_s":list(HORIZONS),
                              "anchor_stride_s":.2,"future_use":"loss/evaluator only","anchor_join":"ground truth identity is evaluator metadata only"})
    groups={s:sorted({x["scene_id"] for x in samples if x["split"]==s}) for s in ["train","val","test"]}
    if any(set(groups[a])&set(groups[b]) for a,b in [("train","val"),("train","test"),("val","test")]): raise ValueError("Scene leakage")
    metadata["scene_groups"]=groups; json_write(out/"manifest.json",metadata)
    return out


def load_motion_dataset(path):
    path=Path(path); meta=json.loads((path/"manifest.json").read_text())
    for filename,key in [("predictor_histories.npz","predictor_sha256"),("evaluator_futures.npz","target_sha256"),("samples.jsonl","index_sha256")]:
        if sha(path/filename)!=meta[key]: raise ValueError("Motion dataset checksum mismatch")
    rows=[json.loads(s) for s in (path/"samples.jsonl").read_text().splitlines() if s]
    h=np.load(path/"predictor_histories.npz")["history"]; y=np.load(path/"evaluator_futures.npz")["future"]
    if h.shape != (len(rows),11,2) or y.shape != (len(rows),20,2): raise ValueError("Motion array dimensions")
    if not np.isfinite(h).all() or not np.isfinite(y).all(): raise ValueError("Nonfinite motion data")
    return h,y,rows,meta


def jaad_sensor_cache(root, detector="models/detection-trained", threshold=.1):
    """Decode real videos, infer persons and save observations without labels."""
    import cv2, torch
    torch.set_num_threads(3); cv2.setNumThreads(1)
    root=Path(root);budget=_disk_guard(root);stem=f"jaad_sensor_observations_{Path(detector).name}_threshold{threshold:g}"
    output=root/"reports/motion/datasets"/f"{stem}.jsonl"
    manifest=root/"reports/motion/datasets"/f"{stem}.manifest.json"
    pack=root/detector; weight_hash=sha(pack/"weights.pt"); index_hash=sha(root/"data/jaad/index.jsonl")
    if output.exists() and manifest.exists():
        m=json.loads(manifest.read_text())
        if m["detector_sha256"]==weight_hash and m["index_sha256"]==index_hash and m["threshold"]==threshold and m["observations_sha256"]==sha(output): return output
        raise ValueError("Existing JAAD observation cache has different provenance")
    adapter=load_adapter(pack,device="cpu")
    # Index metadata is whitelisted. targets_path is never passed into inference.
    all_rows=[json.loads(s) for s in (root/"data/jaad/index.jsonl").read_text().splitlines()]
    clips={r["sequence_id"]:{k:r[k] for k in ["sequence_id","split","video_path","fps"]} for r in all_rows}
    started=time.time(); counts={}; rgb_hash=hashlib.sha256()
    with output.with_suffix(".partial").open("w") as f:
        for clip, info in sorted(clips.items()):
            cap=cv2.VideoCapture(str(root/info["video_path"])); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS))
            if abs(fps-info["fps"])>.001: raise ValueError("Decoded/metadata FPS mismatch")
            indices=sorted(set(round(t*fps) for t in np.arange(0,(total-1)/fps+1e-9,.1))); selected=set(indices); count=0
            for i in range(total):
                ok,bgr=cap.read()
                if not ok: raise ValueError("JAAD video decode failed")
                if i not in selected: continue
                rgb=cv2.cvtColor(cv2.resize(bgr,(640,360),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                frame=SensorFrameV1(f"{clip}/{i}",round(i/fps*1e9),rgb)
                out=adapter.predict(frame); ds=[d for d in out.detections if d.class_id==1 and d.score>=threshold]
                digest=hashlib.sha256(rgb.tobytes()).hexdigest();rgb_hash.update(digest.encode())
                rec={"scene_id":clip,"split":info["split"],"frame_index":i,"timestamp":i/fps,"fps":fps,"rgb_sha256":digest,
                     "width":640,"height":360,"detections":[{"bbox":d.bbox,"score":d.score,"class_id":d.class_id} for d in ds]}
                f.write(json.dumps(rec,allow_nan=False)+"\n");count+=1
            cap.release();counts[clip]=count;f.flush()
            print(json.dumps({"event":"jaad_sensor_clip","clip":clip,"frames":count,"elapsed_s":round(time.time()-started,2)}),flush=True)
    output.with_suffix(".partial").replace(output);adapter.close()
    json_write(manifest,{"schema_version":1,"detector_pack":detector,"detector_sha256":weight_hash,"detector_manifest_sha256":sha(pack/"manifest.json"),
                         "index_sha256":index_hash,"observations_sha256":sha(output),"decoded_rgb_sha256_chain":rgb_hash.hexdigest(),"threshold":threshold,"rate_hz":10,
                         "clip_counts":counts,"elapsed_s":time.time()-started,"sensor_only":True,"targets_opened_by_inference":False,
                         "video_sha256":{clip:sha(root/info["video_path"]) for clip,info in clips.items()},"command":shlex.join([sys.executable,*sys.argv]),"budget_start":budget,"budget_end":_disk_guard(root)})
    return output


def _label_tracks(annotation):
    tracks={};fps=annotation["fps"]
    for key, targets in annotation["frames"].items():
        for t in targets:
            b=t["bbox_xyxy"]
            if t["ignore"] or t["iscrowd"] or t["occlusion"]==2 or not np.isfinite(b).all(): continue
            if b[2]<=b[0] or b[3]<=b[1]: continue
            tracks.setdefault(t["track_id"],[]).append((int(key)/fps,np.array([(b[0]+b[2])/2/annotation["width"],(b[1]+b[3])/2/annotation["height"]]),b))
    return {key:sorted(vals,key=lambda v:v[0]) for key,vals in tracks.items()}


def build_jaad(root, observed=True, detector="models/detection-trained", threshold=.1):
    from robot_model_sdk.contracts import Detection
    root=Path(root); cache=jaad_sensor_cache(root,detector,threshold) if observed else None
    samples=[]; rejected={"history_gap":0,"unmatched_anchor":0,"future_missing_or_occluded":0}; source_hashes={}
    indexed=[json.loads(s) for s in (root/"data/jaad/index.jsonl").read_text().splitlines()]
    clips={r["scene_id"]:r for r in indexed}; cache_rows={}
    if observed:
        for line in cache.read_text().splitlines():
            r=json.loads(line);cache_rows.setdefault(r["scene_id"],[]).append(r)
    for clip,info in sorted(clips.items()):
        annotation_path=root/info["targets_path"]; a=json.loads(annotation_path.read_text()); labels=_label_tracks(a);source_hashes[clip]=sha(annotation_path)
        if observed:
            tracker=Tracker(max_age_s=.3,iou_threshold=.15); histories={}
            for j,r in enumerate(cache_rows[clip]):
                ts=r["timestamp"]; tracks=tracker.update([Detection(**d) for d in r["detections"]],round(ts*1e9))
                for track in tracks:
                    if track.observed:
                        b=track.bbox;histories.setdefault(track.id,[]).append((ts,[(b[0]+b[2])/2/640,(b[1]+b[3])/2/360]))
                if j%2 or ts<1:continue
                valid=[t for t in a["frames"].get(str(r["frame_index"]),[]) if not t["ignore"] and not t["iscrowd"] and t["occlusion"]!=2]
                # One-to-one anchor joins cannot give one GT person to two tracks.
                pairs=sorted([(bbox_iou(track.bbox,t["bbox_xyxy"]),track,t) for track in tracks if track.observed for t in valid],key=lambda v:v[0],reverse=True)
                matched_tracks=set();matched_gt=set()
                for overlap,track,t in pairs:
                    if overlap<.3 or track.id in matched_tracks or t["track_id"] in matched_gt:continue
                    matched_tracks.add(track.id);matched_gt.add(t["track_id"])
                    hist=histories[track.id];h=resample_history([v[0] for v in hist],[v[1] for v in hist],ts)
                    if h is None:rejected["history_gap"]+=1;continue
                    vals=labels[t["track_id"]];y=interpolate_labels([v[0] for v in vals],[v[1] for v in vals],ts+FUTURE_TIMES,1.1/a["fps"])
                    if y is None:rejected["future_missing_or_occluded"]+=1;continue
                    samples.append({"sample_id":f"{clip}/{r['frame_index']}/{track.id}","scene_id":clip,"split":info["split"],"anchor_time_s":ts,
                                    "anchor_frame":r["frame_index"],"history_source":"detected_and_tracked_RGB","tracker_id":track.id,"target_id_evaluator_only":t["track_id"],"history":h,"future":y})
                rejected["unmatched_anchor"]+=sum(t.observed and t.id not in matched_tracks for t in tracks)
        else:
            # Explicitly secondary oracle experiment, never called sensor evidence.
            for tid,vals in labels.items():
                times=[v[0] for v in vals];pos=[v[1] for v in vals]
                for ts in np.arange(1.,max(times)-2.+1e-9,.2):
                    old=[i for i,t in enumerate(times) if t<=ts+1e-9]
                    if not old:continue
                    # Pick an actual frame as anchor; preserve actual clip FPS.
                    ts=times[old[-1]];old=[i for i,t in enumerate(times) if t<=ts+1e-9]
                    h=resample_history(np.array(times)[old],np.array(pos)[old],ts,max_gap=1.1/a["fps"])
                    y=interpolate_labels(times,pos,ts+FUTURE_TIMES,1.1/a["fps"])
                    if h is None or y is None:continue
                    samples.append({"sample_id":f"oracle/{clip}/{round(ts*a['fps'])}/{tid}","scene_id":clip,"split":info["split"],"anchor_time_s":ts,
                                    "history_source":"human_annotated_past_ORACLE","target_id_evaluator_only":tid,"history":h,"future":y})
    name="jaad-observed-image" if observed else "jaad-oracle-image"
    return _save_dataset(root,name,samples,{"domain":"JAAD real image-plane","coordinate_system":"image_xy_divided_by_width_height","units":"normalized_image",
            "history_source":"detected_and_tracked_RGB" if observed else "human_annotated_past_ORACLE","end_to_end_evidence":observed,"annotation_hashes":source_hashes,
            "jaad_index_sha256":sha(root/"data/jaad/index.jsonl"),"observations_sha256":sha(cache) if cache else None,"rejections":rejected,
            "detector_pack":detector if observed else None,"detector_threshold":threshold if observed else None,"detector_sha256":sha(root/detector/"weights.pt") if observed else None,
            "tracker":{"algorithm":"SDK IoU association + Kalman filter","iou_threshold":.15,"max_age_s":.3,"GT_id_input":False},
            "limitations":["clip-disjoint; original-recording groups unavailable","egocentric image motion includes camera motion","overlapping windows are correlated","GT anchor IoU selection conditions evaluation on detected labeled people"]})


def fixture_scene_parameters(seed):
    rng=np.random.default_rng(seed)
    return {"seed":seed,"phase":rng.uniform(-np.pi,np.pi,2),"frequency":rng.uniform(.25,.8,2),"amplitude":rng.uniform(.18,.55,2),
            "depth":rng.uniform(3.1,4.5,2),"drift":rng.uniform(-.05,.05,2),"yaw":float(rng.uniform(-.07,.07)),"camera_x":float(rng.uniform(-.08,.08))}


def fixture_world_positions(params,t):
    result=[]
    for k in range(2):
        phase=params["phase"][k];f=params["frequency"][k];a=params["amplitude"][k]
        x=(-.75 if k==0 else .75)+a*np.sin(f*t+phase)+params["drift"][k]*t
        z=params["depth"][k]+.22*np.sin(.75*f*t+phase+.4)
        result.append(np.array([x,.05,z],np.float64))
    return result


def render_fixture_frame(params,i,fps=10):
    """Render actual calibrated RGB and optical-Z depth, return targets separately."""
    w,h=160,120;t=i/fps;c=Calibration(140,140,w/2,h/2,w,h)
    yaw=params["yaw"]+.015*np.sin(.3*t);co,si=np.cos(yaw),np.sin(yaw)
    T=np.eye(4);T[:3,:3]=[[co,0,si],[0,1,0],[-si,0,co]];T[:3,3]=[params["camera_x"]+.03*np.sin(.2*t),0,0]
    rgb=np.full((h,w,3),[35,40,50],np.uint8);depth=np.full((h,w),6.,np.float32);labels=[]
    for k,pworld in enumerate(fixture_world_positions(params,t)):
        pcamera=(pworld-T[:3,3])@T[:3,:3]
        x,y,z=pcamera;u=c.fx*x/z+c.cx;v=c.fy*y/z+c.cy;ww=c.fx*.35/z;hh=c.fy*.65/z
        x0,x1=max(0,int(round(u-ww/2))),min(w,int(round(u+ww/2)))
        y0,y1=max(0,int(round(v-hh/2))),min(h,int(round(v+hh/2)))
        if x1>x0 and y1>y0:
            nearer=z<depth[y0:y1,x0:x1];rgb[y0:y1,x0:x1][nearer]=[230,50,40] if k==0 else [35,210,55];depth[y0:y1,x0:x1][nearer]=z
        labels.append({"object_id":f"block-{k}","position_world_m":pworld.tolist(),"bbox_render_target_only":[x0,y0,x1,y1]})
    return SensorFrameV1(f"fixture-seed{params['seed']}/{i}",round(t*1e9),rgb,depth,c,T,"synthetic-known-pose"),labels


def sensor_world_positions(frame,adapter):
    """Connected-components detections + depth/pose geometry, no renderer labels."""
    output=[]
    for detection in adapter.predict(frame).detections:
        measurement=robust_object_position(frame.depth,frame.calibration,detection.bbox)
        if measurement:
            p=transform_points([measurement["position_camera_m"]],frame.T_world_camera)[0]
            output.append((detection,p[[0,2]]))
    return output


def build_fixture(root, counts=(48,12,12), duration_s=8):
    root=Path(root);samples=[];source_hash=hashlib.sha256();adapter=FixtureAdapter();seed_groups={};sensor_dir=root/"reports/motion/datasets/fixture-sensor-examples";sensor_dir.mkdir(parents=True,exist_ok=True)
    for split,count,base in zip(["train","val","test"],counts,[10000,20000,30000]):
        seed_groups[split]=list(range(base,base+count))
        for seed in seed_groups[split]:
            params=fixture_scene_parameters(seed);tracker=Tracker(max_age_s=.2,iou_threshold=.15);histories={};labels_by_id={f"block-{k}":[] for k in range(2)};observed=[]
            for i in range(duration_s*10+1):
                frame,labels=render_fixture_frame(params,i);t=i/10
                source_hash.update(frame.rgb.tobytes());source_hash.update(frame.depth.tobytes());source_hash.update(frame.T_world_camera.tobytes())
                if seed==base and i in [0,20,40,60,80]:np.savez_compressed(sensor_dir/f"{split}-{seed}-{i:03}.npz",rgb=frame.rgb,depth=frame.depth,T_world_camera=frame.T_world_camera,calibration=np.array([140,140,80,60,160,120]))
                for target in labels:labels_by_id[target["object_id"]].append((t,np.array(target["position_world_m"])[[0,2]]))
                measured=sensor_world_positions(frame,adapter);tracks=tracker.update([d for d,p in measured],frame.timestamp_ns)
                by_detection={tuple(d.bbox):p for d,p in measured}
                for track in tracks:
                    if not track.observed:continue
                    # Measurement/track association uses bbox geometry only.
                    candidates=[(bbox_iou(track.bbox,d.bbox),p) for d,p in measured]
                    if not candidates:continue
                    _,p=max(candidates,key=lambda x:x[0]);histories.setdefault(track.id,[]).append((t,p.copy()))
                if i<10 or i>duration_s*10-20 or i%2:continue
                pairs=sorted([(bbox_iou(track.bbox,target["bbox_render_target_only"]),track,target) for track in tracks if track.observed for target in labels],key=lambda x:x[0],reverse=True)
                used_tracks=set();used_labels=set()
                for iou,track,target in pairs:
                    if iou<.3 or track.id in used_tracks or target["object_id"] in used_labels:continue
                    used_tracks.add(track.id);used_labels.add(target["object_id"])
                    hist=histories[track.id];h=resample_history([p[0] for p in hist],[p[1] for p in hist],t,max_gap=.11)
                    if h is not None:observed.append((t,track.id,target["object_id"],h))
            for t,tid,gtid,h in observed:
                vals=labels_by_id[gtid];y=interpolate_labels([p[0] for p in vals],[p[1] for p in vals],t+FUTURE_TIMES,.11)
                if y is not None:samples.append({"sample_id":f"fixture/{seed}/{t:.1f}/{tid}","scene_id":f"fixture-seed{seed}","split":split,"anchor_time_s":t,
                                                "history_source":"RGB_connected_components_plus_registered_depth_and_pose","tracker_id":tid,"target_id_evaluator_only":gtid,"history":h,"future":y})
    adapter.close()
    return _save_dataset(root,"fixture-observed-world",samples,{"domain":"synthetic calibrated independently seeded colored block scenes","coordinate_system":"world_XZ_planar","units":"metres",
            "history_source":"RGB_connected_components_plus_registered_depth_and_pose","scene_seed_groups":seed_groups,"sensor_stream_sha256":source_hash.hexdigest(),
            "sensor_frontend":"SDK FixtureAdapter, Tracker, robust_object_position, transform_points","depth_convention":"optical Z metres","pose_source":"synthetic-known-pose",
            "calibration":{"fx":140,"fy":140,"cx":80,"cy":60,"width":160,"height":120},"end_to_end_evidence":True,
            "limitations":["synthetic colored planar blocks; no person/domain-realism claim","provided exact camera poses; not pose estimation","quantized pixel-centre depth introduces small measurement error","overlapping windows within each independently seeded scene"]})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=["fixture","jaad","oracle","all"],default="all");args=parser.parse_args();root=root_path()
    if args.dataset in ["fixture","all"]:print(build_fixture(root),flush=True)
    if args.dataset in ["jaad","all"]:print(build_jaad(root,True),flush=True)
    if args.dataset in ["oracle","all"]:print(build_jaad(root,False),flush=True)


if __name__=="__main__":main()
