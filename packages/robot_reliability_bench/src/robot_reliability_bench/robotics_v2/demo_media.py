"""Actual replay panels and chapter manifests for the root-owned demo viewer."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import zlib
import numpy as np
from .openloris import frame_from_reference
from .remote_model import RemoteModelSession
from .simulator import MuJoCoFrameSource, evaluator_state
from .controller import ObservedController, observed_clearance
from .provenance import context, finish


def semantic_mask(result):
    encoded=result.get('semantic')
    if not encoded:return None
    shape=encoded['shape'];raw=zlib.decompress(base64.b64decode(encoded['zlib_base64'],validate=True))
    if encoded['dtype']!='uint8' or len(shape)!=2 or len(raw)!=int(np.prod(shape)):
        raise ValueError('Invalid semantic-mask IPC result')
    return np.frombuffer(raw,dtype=np.uint8).reshape(shape)


def save_panel(frame,result,path,label,source_time,command=None):
    import cv2
    image=frame.rgb.copy();mask=semantic_mask(result)
    if mask is not None:
        if mask.shape!=image.shape[:2]:raise ValueError('Semantic mask must use original image pixels')
        palette=np.random.default_rng(7).integers(30,220,(256,3),dtype=np.uint8);palette[255]=0
        image=np.rint(.72*image+.28*palette[mask]).astype(np.uint8)
    original_height,original_width=image.shape[:2]
    display_scale=max(1.,640/original_width)
    if display_scale>1.:image=cv2.resize(image,(640,round(original_height*display_scale)),interpolation=cv2.INTER_LINEAR)
    output=result.get('result',{})
    tracks=output.get('tracks')
    items=tracks if tracks is not None else output.get('detections',[])
    for item in items:
        if item.get('observed',True) is False:continue
        box=np.rint(np.asarray(item['bbox'])*display_scale).astype(int)
        x1,y1,x2,y2=box
        cv2.rectangle(image,(x1,y1),(x2,y2),(40,230,80),2)
        caption=f"{item.get('id','class '+str(item.get('class_id','?')))} score={item.get('score',0):.2f}"
        position=item.get('position')
        if position:caption+=f" range={position.get('surface_range_m',0):.2f}m"
        cv2.putText(image,caption,(max(0,x1),max(13,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.35,(240,240,240),1)
    cv2.rectangle(image,(0,0),(image.shape[1],48),(20,25,30),-1)
    short_label='OpenLORIS recorded sensors' if label.startswith('OpenLORIS') else label.replace('MuJoCo controlled simulation:','MuJoCo:')
    cv2.putText(image,f'{short_label} | source {source_time:.2f}s',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1)
    caption=f"RGB-D={frame.depth is not None} pose={frame.pose_source or 'missing'} RPC={result.get('actual_rpc_elapsed_ms',result.get('adapter_ms',0)):.1f}ms"
    cv2.putText(image,caption,(8,38),cv2.FONT_HERSHEY_SIMPLEX,.36,(240,240,240),1)
    if command:
        cv2.putText(image,f"command v={command.linear_mps:.2f} w={command.angular_rps:.2f}: {command.reason}",
                    (8,image.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,230,100),1)
    if frame.depth is not None:
        depth=frame.depth;valid=np.isfinite(depth)&(depth>0)
        scaled=np.clip(depth/5*255,0,255).astype(np.uint8);depth_color=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)
        depth_color[~valid]=0
        small=cv2.resize(depth_color,(image.shape[1]//4,image.shape[0]//4),interpolation=cv2.INTER_NEAREST)
        image[-small.shape[0]-20:-20,-small.shape[1]:]=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):raise IOError('Could not write replay panel')


def make_chapter(root,output,model_pack,*,source='openloris',semantic_pack=None,tracker='T2',
                 frame_refs=None,seconds=23.,scenario='depth_loss',seed=42,panel_fps=3.,
                 quiescent=False,backend='native',device='cpu',threads=1,tracker_lock=None,semantic_backend='native'):
    if str(model_pack)!='fixture' and not quiescent:raise ValueError('Selected-model media generation requires training-quiescent attestation')
    if panel_fps<=0 or panel_fps>5:raise ValueError('Panel sampling must be in(0,5]fps')
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    if str(model_pack)!='fixture':
        lock_path=Path(tracker_lock or root/'models/v2/selection-tracking.json')
        locked=json.loads(lock_path.read_text())
        from .matched_control import bind_tracker_lock
        binding=bind_tracker_lock(lock_path,tracker,locked['selected']['parameters']['high_score'])
        tracker_lock=lock_path
    else:binding=None
    model_role='fixture perception contract probe; no learned model' if str(model_pack)=='fixture' else 'learned detection/tracking/semantic overlays; simulation command is geometric, not a control-contribution claim'
    label='OpenLORIS recorded physical robot RGB-D + wheel odometry' if source=='openloris' else f'MuJoCo controlled simulation: {scenario}'
    provenance=context({'source':source,'scenario':scenario if source=='mujoco' else None,'seed':seed,
        'model_pack':str(model_pack),'semantic_pack':str(semantic_pack),'semantic_backend':semantic_backend,'detector_backend':backend,'device':device,'threads':threads,'tracker':tracker,'panel_fps':panel_fps,
        'playback_speed':1.,'model_role':model_role,
        'GT_to_inference':False,'canonical_tracker_lock':binding},root)
    simulator=None;controller=None;first=None;last_panel=-float('inf');panels=[];processed=[]
    if source=='openloris':
        refs=Path(frame_refs or root/'reports/v2/robotics/openloris-full-10hz/sensor-frames.jsonl')
        reference_rows=[json.loads(line) for line in refs.read_text().splitlines()]
        frames=((frame_from_reference(row),row) for row in reference_rows)
        provenance['sensor_reference_sha256']=hashlib.sha256(refs.read_bytes()).hexdigest()
        source_metadata={'bag':str(root/'data/v2/openloris/office1-1.bag'),'bag_duration_s':26.998946667,
                         'clock':'original recorded sensor timestamps; no retimestamping for this offline replay'}
    elif source=='mujoco':
        simulator=MuJoCoFrameSource(scenario,seed);controller=ObservedController()
        simulator.current_odom=simulator.odom.update(simulator.data.qpos[simulator.wheel_qpos])
        def simulated_frames():
            for _ in range(round(seconds/.1)):
                t=float(simulator.data.time)
                yield simulator.capture(depth_missing=scenario=='depth_loss' and 3<=t<5),None
        frames=simulated_frames();source_metadata=simulator.metadata()
    else:raise ValueError('Unknown demo chapter source')
    try:
        with RemoteModelSession(root,model_pack,output/'inference',backend=backend,device=device,
                                threads=threads,tracker=tracker,semantic_pack=semantic_pack,tracker_lock=tracker_lock,semantic_backend=semantic_backend) as model:
            for i,(frame,reference) in enumerate(frames):
                if first is None:first=frame.timestamp_ns
                source_time=(frame.timestamp_ns-first)/1e9
                start=time.perf_counter();result=model.predict(frame)
                result['actual_rpc_elapsed_ms']=(time.perf_counter()-start)*1000
                result['latency_scope']='Observed replay RPC including IPC; not formal quiescent repeated benchmark statistics'
                command=None
                if simulator is not None:
                    geometry=observed_clearance(frame)
                    command=controller.command(frame,simulator.current_odom,geometry,frame.timestamp_ns,0.)
                    simulator.step(command)
                    result['control']={'command':command.to_json(),'role':'geometric policy; learned outputs are displayed shadow observations'}
                    result['evaluator']=evaluator_state(simulator)
                result_path=output/'results'/f'{i:05d}.json';result_path.parent.mkdir(exist_ok=True)
                result_path.write_text(json.dumps(result,indent=2)+'\n')
                row={'time_s':source_time,'sample_id':frame.sample_id,'timestamp_ns':frame.timestamp_ns,
                     'result_ref':os.path.relpath(result_path,output),'result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),'depth_available':frame.depth is not None,
                     'pose_source':frame.pose_source,'calibration':frame.calibration.to_json(),
                     'source_rgb_ref':reference.get('rgb_path') if reference else None,
                     'source_depth_ref':reference.get('depth_path') if reference else None}
                processed.append(row)
                if source_time-last_panel>=1/panel_fps-1e-8:
                    panel=output/'panels'/f'{i:05d}.png'
                    save_panel(frame,result,panel,label,source_time,command)
                    panels.append({**row,'image':os.path.relpath(panel,output),'image_sha256':hashlib.sha256(panel.read_bytes()).hexdigest()})
                    last_panel=source_time
    finally:
        if simulator is not None:simulator.close()
    if not processed:raise ValueError('No actual sensor frames processed')
    source_duration=processed[-1]['time_s']+(0.1 if source=='mujoco' else 0.)
    (output/'processed-frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in processed))
    manifest={'schema_version':1,'chapter_id':source if source=='openloris' else f'mujoco-{scenario}',
        'label':label,'source_duration_s':source_duration,'playback_speed':1.,'frames':panels,
        'source_metadata':source_metadata,'processed_sensor_frames':len(processed),'panel_frames':len(panels),
        'all_source_frames_processed':True,'panel_fps_requested':panel_fps,
        'clock_policy':'chapter time_s rebased from its own source timestamp; chapters do not share a sensor clock',
        'model_role':model_role,
        'forecaster':'disabled for worldXY / recorded wheel-odom domain',
        'provenance':finish(provenance),'status':'verified'}
    (output/'chapter-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
