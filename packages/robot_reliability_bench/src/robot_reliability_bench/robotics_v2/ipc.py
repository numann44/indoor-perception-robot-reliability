"""Length-framed, bounded, localhost Unix-domain sensor-only inference IPC."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import socket
import struct
import time
import zlib
import numpy as np
from robot_model_sdk import Calibration,SensorFrameV1,load_adapter

MAX_MESSAGE=16*1024**2
SENSOR_FIELDS={'sample_id','timestamp_ns','rgb','depth','calibration','T_world_camera','pose_source'}

def array_wire(array):
    if array is None:return None
    a=np.ascontiguousarray(array)
    return {'shape':list(a.shape),'dtype':a.dtype.str,'bytes':base64.b64encode(a.tobytes()).decode()}

def array_read(value,kind):
    if value is None:return None
    if set(value)!={'shape','dtype','bytes'}:raise ValueError('Unknown array fields')
    shape=value['shape'];dtype=np.dtype(value['dtype'])
    if len(shape) not in (2,3) or any(not isinstance(x,int) or x<=0 or x>4096 for x in shape):raise ValueError('Invalid array shape')
    if kind=='rgb' and (dtype!=np.dtype('uint8') or len(shape)!=3 or shape[2]!=3):raise ValueError('RGB must be uint8 HWC3')
    if kind=='depth' and (dtype.kind!='f' or dtype.itemsize!=4 or len(shape)!=2):raise ValueError('Depth must be float32 camera-Z meters')
    raw=base64.b64decode(value['bytes'],validate=True)
    if len(raw)!=int(np.prod(shape))*dtype.itemsize:raise ValueError('Array byte/shape mismatch')
    return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()

def frame_wire(frame):
    return {'sample_id':frame.sample_id,'timestamp_ns':frame.timestamp_ns,'rgb':array_wire(frame.rgb),
            'depth':array_wire(frame.depth),'calibration':frame.calibration.to_json() if frame.calibration else None,
            'T_world_camera':frame.T_world_camera.tolist() if frame.T_world_camera is not None else None,
            'pose_source':frame.pose_source}

def frame_read(value):
    if set(value)!=SENSOR_FIELDS:raise ValueError('Only whitelisted sensor fields permitted; target/truth fields rejected')
    c=Calibration(**value['calibration']) if value['calibration'] else None
    return SensorFrameV1(value['sample_id'],value['timestamp_ns'],array_read(value['rgb'],'rgb'),array_read(value['depth'],'depth'),c,
                         np.asarray(value['T_world_camera'],dtype=float) if value['T_world_camera'] is not None else None,value['pose_source'])

def receive(sock):
    def exact(n):
        chunks=[];total=0
        while total<n:
            part=sock.recv(n-total)
            if not part:raise ConnectionError('IPC peer disconnected mid-message')
            chunks.append(part);total+=len(part)
        return b''.join(chunks)
    length=struct.unpack('!I',exact(4))[0]
    if length>MAX_MESSAGE:raise ValueError('IPC message exceeds bounded byte limit')
    return json.loads(exact(length))

def send(sock,value):
    raw=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>MAX_MESSAGE:raise ValueError('IPC message exceeds bounded byte limit')
    sock.sendall(struct.pack('!I',len(raw))+raw)

def predict_remote(socket_path,frame,timeout=5.):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(timeout);client.connect(str(socket_path));send(client,{'protocol':1,'frame':frame_wire(frame)})
        return receive(client)

def serve(socket_path,model_pack='fixture',backend='native',device='cpu',delay_ms=0.,log=None,threads=1,pipeline_semantic=None,pipeline_tracker='T2',pipeline_tracking=False,tracker_lock=None,semantic_backend='native'):
    import cv2
    cv2.setNumThreads(threads)
    if model_pack!='fixture' and backend=='native':
        import torch
        torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    pipeline=None
    if pipeline_semantic is not None or pipeline_tracking:
        from indoor_perception_lab.pipeline import PerceptionPipeline
        from indoor_perception_lab.tracking_v2 import make_tracker, TRACKING_PRESETS
        parameters=TRACKING_PRESETS[pipeline_tracker]
        if tracker_lock is not None:
            from .matched_control import bind_tracker_lock
            lock=bind_tracker_lock(tracker_lock,pipeline_tracker,parameters['high_score'])
            import hashlib,inspect
            tracker_source=Path(inspect.getfile(make_tracker))
            if lock['tracker_source_sha256']!=hashlib.sha256(tracker_source.read_bytes()).hexdigest():
                raise ValueError('Canonical tracker lock source differs from inference implementation')
            if lock['detector_manifest_sha256']!=hashlib.sha256((Path(model_pack)/'manifest.json').read_bytes()).hexdigest():
                raise ValueError('Tracker lock detector manifest differs from active model')
            if {k:v for k,v in lock['tracking'].items() if k!='preset'}!=parameters:
                raise ValueError('Tracker lock parameters differ from frozen implementation')
        detector_adapter=load_adapter(model_pack,backend=backend,device=device)
        semantic_adapter=load_adapter(pipeline_semantic,backend=semantic_backend,device='cpu' if semantic_backend=='onnx' else device) if pipeline_semantic else None
        pipeline=PerceptionPipeline(detector=detector_adapter,segmenter=semantic_adapter,device=device,score_threshold=parameters['low_score'],
                                    tracking=True,use_depth=True,forecaster=None,tracker=make_tracker(pipeline_tracker))
        adapter=None
    else:adapter=load_adapter(model_pack,backend=backend,device=device)
    path=Path(socket_path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.unlink()
    stream=Path(log).open('a') if log else None
    def event(value):
        if stream:stream.write(json.dumps({'unix':time.time(),**value})+'\n');stream.flush()
    event({'event':'loaded','pid':os.getpid(),'model_pack':str(model_pack),'backend':backend,'device':device,'threads':threads,'controlled_delay_ms':delay_ms,'pipeline_enabled':pipeline is not None,'pipeline_tracker':pipeline_tracker if pipeline is not None else None,'semantic_backend':semantic_backend,'cv2_threads':cv2.getNumThreads()})
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(str(path));path.chmod(0o600);server.listen(2);server.settimeout(1.)
            while True:
                try:connection,_=server.accept()
                except socket.timeout:continue
                with connection:
                    connection.settimeout(5.)
                    try:
                        request=receive(connection)
                        if set(request)!={'protocol','frame'} or request['protocol']!=1:raise ValueError('Unknown IPC protocol')
                        frame=frame_read(request['frame']);start=time.perf_counter()
                        if delay_ms:time.sleep(delay_ms/1000)
                        output=pipeline.process(frame) if pipeline is not None else adapter.predict(frame);elapsed=(time.perf_counter()-start)*1000
                        from .controller import observed_clearance
                        geometry=observed_clearance(frame,camera_height_m=float(frame.T_world_camera[2,3]) if frame.T_world_camera is not None else .30)
                        result={'observed_geometry':geometry,'status':'ok','sample_id':frame.sample_id,'timestamp_ns':frame.timestamp_ns,'result':output.to_json() if hasattr(output,'to_json') else output,'pipeline_enabled':pipeline is not None,'forecasting_domain_policy':'learned forecasting disabled for ROS odometry/world-XY; no world-XZ checkpoint fed',
                                'adapter_ms':elapsed,'last_timings_ms':getattr(adapter,'last_timings_ms',{})}
                        semantic_mask=pipeline.last_semantic if pipeline is not None else getattr(output,'semantic',None)
                        if semantic_mask is not None:
                            mask=np.asarray(semantic_mask,dtype=np.uint8)
                            result['semantic']={'shape':list(mask.shape),'dtype':'uint8','zlib_base64':base64.b64encode(zlib.compress(mask.tobytes())).decode()}
                        send(connection,result);event({'event':'prediction','sample_id':frame.sample_id,'elapsed_ms':elapsed})
                    except Exception as exc:
                        event({'event':'error','type':type(exc).__name__,'error':str(exc)})
                        try:send(connection,{'status':'error','error':str(exc)})
                        except (BrokenPipeError,ConnectionError):pass
    finally:
        if adapter is not None:adapter.close()
        if pipeline is not None:pipeline.close()
        path.unlink(missing_ok=True)
        if stream:stream.close()
