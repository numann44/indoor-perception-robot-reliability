from __future__ import annotations
import json,time,hashlib,datetime,platform,uuid
from pathlib import Path
import numpy as np
from robot_model_sdk import load_adapter,PerceptionResultV1
from .tracking import Tracker
from .geometry import SpatialMap,robust_object_position,transform_points
from .prediction import baseline_predictions,operational_behavior

class PerceptionPipeline:
    def __init__(self,detector="fixture",segmenter=None,device="cpu",score_threshold=.3,tracking=True,use_depth=True,forecaster=None,tracker=None):
        self.detector=load_adapter(detector,device=device) if isinstance(detector,(str,Path)) else detector
        self.segmenter=load_adapter(segmenter,device=device) if isinstance(segmenter,(str,Path)) else segmenter
        self.forecaster=load_adapter(forecaster,device=device) if isinstance(forecaster,(str,Path)) else forecaster
        self.tracker=tracker if tracker is not None else Tracker();self.spatial_map=SpatialMap();self.score_threshold=score_threshold;self.tracking=tracking;self.use_depth=use_depth;self.ranges={};self.timestamp_ns=None;self.last_semantic=None
    def process(self,frame):
        start=time.perf_counter()
        if self.timestamp_ns is not None and frame.timestamp_ns<=self.timestamp_ns:return {"schema_version":1,"sample_id":frame.sample_id,"timestamp_ns":frame.timestamp_ns,"status":"rejected_timestamp","tracks":[],"latency_ms":0.0}
        self.timestamp_ns=frame.timestamp_ns
        det=self.detector.predict(frame);ds=[d for d in det.detections if d.score>=self.score_threshold]
        # Learned COCO/person-head packs restrict track class; fixture uses its own block class1.
        if getattr(self.detector,"task","")=="detection":ds=[d for d in ds if d.class_id==1]
        seg=self.segmenter.predict(frame) if self.segmenter else None
        self.last_semantic=seg.semantic if seg else None
        tracks=self.tracker.update(ds,frame.timestamp_ns) if self.tracking else []
        live_ids={t.id for t in tracks};self.ranges={k:v for k,v in self.ranges.items() if k in live_ids}
        out=[];ts=frame.timestamp_ns/1e9
        for t in tracks:
            position=None;unit="pixels";coords="image";point=[(t.bbox[0]+t.bbox[2])/2,(t.bbox[1]+t.bbox[3])/2]
            if self.use_depth and t.observed and frame.depth is not None and frame.calibration:
                position=robust_object_position(frame.depth,frame.calibration,t.bbox)
                if position and frame.T_world_camera is not None:
                    point=transform_points([position["position_camera_m"]],frame.T_world_camera)[0].tolist();unit="metres";coords="world";position["position_world_m"]=point
                elif position:position["world_motion_available"]=False
            if t.observed:
                if t.history and t.history[-1][2]!=coords:t.history=[]
                t.history.append((ts,point,coords));t.history=t.history[-30:]
            hist=[(h[0],h[1]) for h in t.history]
            if not t.observed and t.history:
                coords=t.history[-1][2];unit="metres" if coords=="world" else "pixels"
            if position:self.ranges.setdefault(t.id,[]).append((ts,position["surface_range_m"]));self.ranges[t.id]=self.ranges[t.id][-30:]
            state=operational_behavior(hist,self.ranges.get(t.id)) if unit=="metres" and t.observed else ("unknown" if unit=="metres" else "unavailable_without_world_coordinates")
            forecasts=baseline_predictions(hist,reference_time=ts)
            forecast_status='not_configured' if self.forecaster is None else 'insufficient_causal_history' if t.observed else 'unavailable_unobserved_track'
            if self.forecaster and t.observed and len(hist)>=2:
                metadata=self.forecaster.metadata;mode=metadata['coordinate_system'];positions=None
                if coords=='world' and mode=='world_XZ_planar' and frame.pose_source=='synthetic-known-pose':positions=np.asarray([h[1] for h in hist])[:,[0,2]]
                elif coords=='image' and 'image' in mode.lower() and frame.sample_id.startswith(('jaad/','jaad-v2/')):positions=np.asarray([h[1] for h in hist])/np.array([frame.rgb.shape[1],frame.rgb.shape[0]])
                if positions is None:forecast_status='unavailable_domain_or_coordinates'
                else:
                    learned=self.forecaster.predict_timed_history([h[0] for h in hist],positions,ts);forecast_status='insufficient_causal_history' if learned is None else 'available'
                    if learned is not None:
                        if coords=='world':points=[[float(p[0]),float(hist[-1][1][1]),float(p[1])] for p in learned]
                        else:points=(learned*np.array([frame.rgb.shape[1],frame.rgb.shape[0]])).tolist()
                        forecasts['MLP2x64']={str(h):points[int(round(h*10))-1] for h in [.5,1,2]}
            out.append({"id":t.id,"class_id":t.class_id,"bbox":t.bbox,"score":t.score,"observed":t.observed,"confirmed":t.hits>=2,"position":position,"coordinate_frame":coords,"unit":unit,"forecasts":forecasts,"learned_forecast_status":forecast_status,"behavior":state})
        probabilities=None
        if seg and seg.logits is not None and self.use_depth and frame.depth is not None and frame.calibration is not None and frame.T_world_camera is not None:
            import cv2
            logits=seg.logits
            if logits.shape[-2:]!=frame.rgb.shape[:2]:logits=np.stack([cv2.resize(z,(frame.rgb.shape[1],frame.rgb.shape[0]),interpolation=cv2.INTER_LINEAR) for z in logits])
            exp=np.exp(logits-logits.max(0,keepdims=True));probabilities=exp/exp.sum(0,keepdims=True)
        map_status=self.spatial_map.update(frame,seg.semantic if seg else None,[d.bbox for d in ds],probabilities) if self.use_depth else {"updated":False,"reason":"depth_ablation_disabled"}
        return PerceptionResultV1(sample_id=frame.sample_id,timestamp_ns=frame.timestamp_ns,status="ok" if frame.depth is not None else "rgb_only",pose_source=frame.pose_source,detections=[d.to_json() for d in ds],tracks=out,semantic_shape=list(seg.semantic.shape) if seg and seg.semantic is not None else None,map=map_status,latency_ms=(time.perf_counter()-start)*1e3,detector_timings_ms=self.detector.last_timings_ms).to_json()
    def close(self):
        self.detector.close()
        if self.segmenter:self.segmenter.close()
        if self.forecaster:self.forecaster.close()

def replay(frames,output_dir,detector="fixture",segmenter=None,device="cpu",tracking=True,use_depth=True,save_every=20,forecaster=None,score_threshold=.3):
    import cv2
    from .visualization import render_frame
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);pipe=PerceptionPipeline(detector,segmenter,device,tracking=tracking,use_depth=use_depth,forecaster=forecaster,score_threshold=score_threshold);rows=[]
    metadata={'seed':42,'device':device,'tracking':tracking,'use_depth':use_depth,'score_threshold':score_threshold,'detector':getattr(pipe.detector,'metadata',{}),'segmenter':getattr(pipe.segmenter,'metadata',None),'forecaster':getattr(pipe.forecaster,'metadata',None),'save_every':save_every,'evaluation_labels_accessed':False}
    from .experiments import Run,root_path,sha
    try:root=root_path()
    except RuntimeError:root=None
    run=Run('perception-replay',metadata) if root else None
    run_id=run.id if run else datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-replay-'+uuid.uuid4().hex[:6]
    sensor_hash=hashlib.sha256();source_indices={}
    with (output_dir/"messages.jsonl").open("w") as stream:
        for i,frame in enumerate(frames):
            result=pipe.process(frame)
            sensor_hash.update(json.dumps(frame.to_json(),sort_keys=True).encode());sensor_hash.update(frame.rgb.tobytes())
            if frame.depth is not None:sensor_hash.update(frame.depth.tobytes())
            if root:
                dataset=frame.sample_id.split('/')[0];index=root/'data'/dataset/'index.jsonl'
                if dataset not in source_indices and index.exists():source_indices[dataset]=sha(index)
            if pipe.last_semantic is not None:
                mask_path=output_dir/f"semantic-{i:05d}.png";cv2.imwrite(str(mask_path),pipe.last_semantic);result["semantic_mask_path"]=str(mask_path.name)
            stream.write(json.dumps(result,allow_nan=False)+"\n");rows.append(result)
            if i%save_every==0:
                vis=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR)
                for t in result.get("tracks",[]):
                    x0,y0,x1,y1=map(int,t["bbox"]);cv2.rectangle(vis,(x0,y0),(x1,y1),(30,220,255),1);cv2.putText(vis,t["id"],(x0,max(10,y0-3)),cv2.FONT_HERSHEY_SIMPLEX,.25,(255,255,255),1)
                cv2.putText(vis,f"{frame.timestamp_ns/1e9:.2f}s {result['status']}",(3,12),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
                cv2.imwrite(str(output_dir/f"frame-{i:05d}.png"),vis)
                render_frame(frame,result,pipe.last_semantic,pipe.spatial_map,output_dir/f"panel-{i:05d}.png")
    (output_dir/"spatial-map.json").write_text(json.dumps(pipe.spatial_map.to_json()))
    summary={"frames":len(rows),"track_ids":len({t['id'] for row in rows for t in row.get('tracks',[])}),"latency_p50_ms":float(np.median([x['latency_ms'] for x in rows])) if rows else None,"status":"PARTIAL","domain":"synthetic fixture" if str(detector)=="fixture" else "recorded real RGB","function_completed":True,"recorded_duration_s":(rows[-1]["timestamp_ns"]-rows[0]["timestamp_ns"])/1e9 if len(rows)>1 else 0,"saved_panels":len(list(output_dir.glob("panel-*.png")))}
    summary.update(run_id=run_id,input_stream_sha256=sensor_hash.hexdigest(),source_index_sha256=source_indices,messages_sha256=sha(output_dir/'messages.jsonl'))
    if run:
        run.event(event='replay_complete',input_stream_sha256=sensor_hash.hexdigest(),output_dir=str(output_dir),source_indices=source_indices);run.finish(summary);provenance={**run.record,'config':metadata,'output_dir':str(output_dir),'input_stream_sha256':sensor_hash.hexdigest(),'source_index_sha256':source_indices}
    else:
        provenance={'run_id':run_id,'config':metadata,'input_stream_sha256':sensor_hash.hexdigest(),'hardware':{'os':platform.platform(),'architecture':platform.machine()},'code_sha256':sha(Path(__file__)),'commit':None,'scope':'portable replay outside portfolio root; no repository commit available','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (output_dir/'run.json').write_text(json.dumps(provenance,indent=2,allow_nan=False))
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2));pipe.close();return summary
