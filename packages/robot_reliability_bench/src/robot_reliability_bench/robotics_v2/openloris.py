"""OpenLORIS ROS1 bag sensors decoded without a ROS1 installation.

Uses CameraInfo + observed static TF for registration, and recorded odometry
for primary map pose. Ground-truth topics and files are never inference inputs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from robot_model_sdk import Calibration,SensorFrameV1
from .geometry import interpolate_pose,nearest_one_to_one,register_depth,resolve_static_transform,rigid_transform,voxel_points


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def field(obj,name):return getattr(obj,name,getattr(obj,name.upper(),None))
def stamp_ns(header):
    s=header.stamp
    return int(s.sec)*10**9+int(getattr(s,'nanosec',getattr(s,'nsec',0)))
def message_transform(transform):
    t=transform.translation;q=transform.rotation
    return rigid_transform((t.x,t.y,t.z),(q.x,q.y,q.z,q.w))
def forbidden(topic):return any(v in topic.lower() for v in ('ground_truth','groundtruth','/gt','/mocap','/vicon'))

def choose_topics(connections):
    rgb=[c for c in connections if c.msgtype.endswith('/Image') and ('color' in c.topic.lower() or 'rgb' in c.topic.lower()) and 'depth' not in c.topic.lower() and not forbidden(c.topic)]
    depth=[c for c in connections if c.msgtype.endswith('/Image') and 'depth' in c.topic.lower() and not forbidden(c.topic)]
    odom=[c for c in connections if c.msgtype.endswith('/Odometry') and not forbidden(c.topic)]
    rgb.sort(key=lambda c:(not ('/color/' in c.topic),c.topic));depth.sort(key=lambda c:('aligned' not in c.topic,c.topic))
    odom.sort(key=lambda c:(c.topic!='/odom','wheel' not in c.topic.lower(),'t265' in c.topic.lower(),c.topic))
    return {'rgb':rgb[0].topic if rgb else None,'depth':depth[0].topic if depth else None,
            'odometry':odom[0].topic if odom else None,
            'camera_info':[c.topic for c in connections if c.msgtype.endswith('/CameraInfo') and not forbidden(c.topic)],
            'extrinsics':[c.topic for c in connections if c.msgtype.endswith('/TFMessage') and not forbidden(c.topic)],
            'imu':[c.topic for c in connections if c.msgtype.endswith('/Imu') and not forbidden(c.topic)],
            'laser':[c.topic for c in connections if c.msgtype.endswith('/LaserScan') and not forbidden(c.topic)],
            'excluded_ground_truth_topics':[c.topic for c in connections if forbidden(c.topic)]}


def inventory_bag(path,output):
    from rosbags.highlevel import AnyReader
    path=Path(path);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with AnyReader([path]) as reader:
        topics=[{'topic':c.topic,'message_type':c.msgtype,'message_count':c.msgcount} for c in reader.connections]
        selection=choose_topics(reader.connections);infos=[];edges=[];seen=set()
        selected=[c for c in reader.connections if c.topic in selection['camera_info']+selection['extrinsics']]
        tf_count=0
        for c,t,raw in reader.messages(connections=selected):
            if c.topic in selection['camera_info']:
                if c.topic in seen:continue
                m=reader.deserialize(raw,c.msgtype);seen.add(c.topic)
                infos.append({'topic':c.topic,'timestamp_ns':stamp_ns(m.header),'frame_id':m.header.frame_id,
                              'width':m.width,'height':m.height,'K':np.asarray(field(m,'k')).tolist(),
                              'D':np.asarray(field(m,'d')).tolist(),'distortion_model':m.distortion_model})
            else:
                if tf_count>=100:continue
                m=reader.deserialize(raw,c.msgtype);tf_count+=1
                for tr in m.transforms:
                    key=(tr.header.frame_id,tr.child_frame_id)
                    if key in seen:continue
                    seen.add(key);edges.append({'topic':c.topic,'parent':key[0],'child':key[1],
                                                'T_parent_child':message_transform(tr.transform).tolist(),
                                                'static_topic':c.topic.endswith('tf_static')})
            if len(infos)>=len(selection['camera_info']) and tf_count>=100:break
        result={'status':'inventoried','bag':str(path.resolve()),'size_bytes':path.stat().st_size,
                'sha256':digest(path),'start_ns':reader.start_time,'end_ns':reader.end_time,
                'duration_s':(reader.end_time-reader.start_time)/1e9,'topics':topics,'selection':selection,
                'CameraInfo':infos,'extrinsic_edges':edges,
                'ground_truth_policy':'Excluded by topic whitelist; evaluator labels never enter SensorFrame',
                'odom_source':'Recorded Odometry selected by topic priority; not ground-truth pose'}
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def decode_image(message):
    """Respect encoding, endianness and row padding; corrupt bodies fail closed."""
    encoding=message.encoding.lower();shape=(int(message.height),int(message.width))
    table={'rgb8':('u1',3),'bgr8':('u1',3),'rgba8':('u1',4),'bgra8':('u1',4),
           'mono8':('u1',1),'mono16':('u2',1),'16uc1':('u2',1),'32fc1':('f4',1)}
    if encoding not in table:raise ValueError(f'Unsupported image encoding {encoding}')
    kind,channels=table[encoding];dtype=np.dtype(('>' if message.is_bigendian else '<')+kind)
    rowbytes=shape[1]*channels*dtype.itemsize;step=int(message.step)
    body=np.asarray(message.data,dtype=np.uint8).tobytes()
    if step<rowbytes or len(body)!=shape[0]*step:raise ValueError('Corrupt sensor image row/data size')
    rows=np.frombuffer(body,dtype=np.uint8).reshape(shape[0],step)[:,:rowbytes].copy()
    out=rows.reshape(-1).view(dtype).reshape(*shape,channels) if channels>1 else rows.reshape(-1).view(dtype).reshape(shape)
    out=out.astype(dtype.newbyteorder('='),copy=False)
    if encoding=='bgr8':out=out[:,:,::-1]
    if encoding in ('rgba8','bgra8'):
        out=out[:,:,:3]
        if encoding=='bgra8':out=out[:,:,::-1]
    if encoding=='mono8':out=np.repeat(out[:,:,None],3,axis=2)
    return np.ascontiguousarray(out)

class OpenLorisFrameSource:
    def __init__(self,path,maximum_frames=100,depth_u16_scale=.001,maximum_width=640,minimum_rgb_period_ns=0):
        self.path=Path(path);self.maximum_frames=maximum_frames;self.scale=depth_u16_scale;self.maximum_width=maximum_width
        self.statistics={};self._closed=False;self.minimum_rgb_period_ns=minimum_rgb_period_ns
    def close(self):self._closed=True
    def metadata(self):return dict(self.statistics)
    def __iter__(self):
        if self._closed:return
        import cv2
        from rosbags.highlevel import AnyReader
        rgb=[];depth=[];odom=[];infos={};edges=[];tf_samples={};rejections={}
        with AnyReader([self.path]) as reader:
            selection=choose_topics(reader.connections)
            if not selection['rgb']:raise ValueError('No selected RGB sensor topic')
            wanted=set([selection['rgb'],selection['depth'],selection['odometry'],*selection['camera_info'],*selection['extrinsics']])
            conns=[c for c in reader.connections if c.topic in wanted and not forbidden(c.topic)]
            last_rgb_bag=None
            for c,bagtime,raw in reader.messages(connections=conns):
                if last_rgb_bag is not None and bagtime>last_rgb_bag+300_000_000:break
                m=reader.deserialize(raw,c.msgtype)
                if c.topic==selection['rgb']:
                    if len(rgb)<self.maximum_frames and (not rgb or stamp_ns(m.header)-rgb[-1][0]>=self.minimum_rgb_period_ns):
                        rgb.append((stamp_ns(m.header),m));
                        if len(rgb)==self.maximum_frames:last_rgb_bag=bagtime
                elif c.topic==selection['depth']:depth.append((stamp_ns(m.header),m))
                elif c.topic==selection['odometry']:
                    t=stamp_ns(m.header);p=m.pose.pose.position;q=m.pose.pose.orientation
                    if not odom or t>odom[-1][0]:odom.append((t,rigid_transform((p.x,p.y,p.z),(q.x,q.y,q.z,q.w)),m.header.frame_id,m.child_frame_id))
                elif c.topic in selection['camera_info']:
                    # Keep all calibration frames; choose latest causally observed calibration.
                    infos.setdefault(m.header.frame_id.lstrip('/'),[]).append((stamp_ns(m.header),m))
                else:
                    for tr in m.transforms:
                        parent=tr.header.frame_id.lstrip('/');child=tr.child_frame_id.lstrip('/')
                        if forbidden('/'+parent) or forbidden('/'+child):continue
                        # Dynamic world/base transforms are not sensor mounting calibration.
                        if not c.topic.endswith('tf_static') and parent.lower() in ('map','odom','world'):continue
                        T=message_transform(tr.transform);key=(parent,child)
                        previous=tf_samples.get(key)
                        if previous is not None and not np.allclose(previous,T,atol=1e-5):
                            rejections['changing_extrinsic']=rejections.get('changing_extrinsic',0)+1;continue
                        tf_samples[key]=T
            edges=[(p,c,T) for (p,c),T in tf_samples.items()]
        rgb.sort(key=lambda r:r[0]);depth.sort(key=lambda r:r[0]);matches=nearest_one_to_one([r[0] for r in rgb],[d[0] for d in depth])
        def calibration(frame_id,stamp):
            candidates=infos.get(frame_id.lstrip('/'),[])
            observed=[(t,m) for t,m in candidates if t<=stamp]
            return max(observed,key=lambda p:p[0])[1] if observed else None
        self.statistics={'source':'OpenLORIS recorded physical robot sensors','bag':str(self.path.resolve()),
                         'selection':selection,'ground_truth_input':False,'pose_source':'recorded_wheel_odometry',
                         'depth_u16_scale_m':self.scale,'depth_u16_scale_basis':'explicit RealSense/OpenLORIS millimeter configuration, never fitted to labels',
                         'registration':'CameraInfo distortion rectification + static sensor TF + metric Z buffer',
                         'rgb_depth_tolerance_ms':20,'odometry_max_interpolation_gap_ms':100,
                         'minimum_rgb_period_ns':self.minimum_rgb_period_ns,'frames_requested':self.maximum_frames,'frames_yielded':0,'depth_missing':0,'pose_missing':0,
                         'rejections':rejections,'calibration_frames':{k:len(v) for k,v in infos.items()},
                         'extrinsic_edge_count':len(edges),'odom_samples':len(odom),'coordinate_frame':'Recorded ROS odometry frame; optical X right Y down Z forward'}
        for i,((t,rm),di) in enumerate(zip(rgb,matches)):
            if self._closed:break
            ci=calibration(rm.header.frame_id,t)
            if ci is None:
                rejections['missing_rgb_calibration']=rejections.get('missing_rgb_calibration',0)+1;continue
            Kr=np.asarray(field(ci,'k'),dtype=float).reshape(3,3);Dr=np.asarray(field(ci,'d'),dtype=float)
            if ci.distortion_model not in ('plumb_bob','rational_polynomial','Brown Conrady',''):
                raise ValueError(f'Unsupported RGB distortion: {ci.distortion_model}')
            image=decode_image(rm)
            if image.ndim!=3 or image.dtype!=np.uint8:raise ValueError('RGB source encoding is not RGB8-compatible')
            image=cv2.undistort(image,Kr,Dr,newCameraMatrix=Kr)
            registered=None
            if di is not None:
                dm=depth[di][1];dc=calibration(dm.header.frame_id,depth[di][0])
                if dc is None and dm.header.frame_id==rm.header.frame_id:dc=ci
                transform=resolve_static_transform(edges,rm.header.frame_id,dm.header.frame_id)
                if dc is not None and transform is not None:
                    raw=decode_image(dm)
                    if raw.ndim!=2:raise ValueError('Depth must be scalar image')
                    metric=raw.astype(np.float32)*(self.scale if raw.dtype==np.uint16 else 1.)
                    registered=register_depth(metric,np.asarray(field(dc,'k')).reshape(3,3),Kr,transform,image.shape,field(dc,'d'))
                else:rejections['missing_depth_calibration_or_extrinsic']=rejections.get('missing_depth_calibration_or_extrinsic',0)+1
            T_world_camera=None
            if odom:
                observed_pose=interpolate_pose([p[0] for p in odom],[p[1] for p in odom],t)
                T_body_camera=resolve_static_transform(edges,odom[0][3],rm.header.frame_id)
                if observed_pose is not None and T_body_camera is not None:T_world_camera=observed_pose@T_body_camera
            if image.shape[1]>self.maximum_width:
                scale=self.maximum_width/image.shape[1];new=(self.maximum_width,round(image.shape[0]*scale))
                image=cv2.resize(image,new,interpolation=cv2.INTER_AREA)
                if registered is not None:registered=cv2.resize(registered,new,interpolation=cv2.INTER_NEAREST_EXACT)
                sx=new[0]/rm.width;sy=new[1]/rm.height
                Kr=Kr.copy();Kr[0,0]*=sx;Kr[1,1]*=sy
                Kr[0,2]=(Kr[0,2]+.5)*sx-.5;Kr[1,2]=(Kr[1,2]+.5)*sy-.5
            h,w=image.shape[:2];cal=Calibration(Kr[0,0],Kr[1,1],Kr[0,2],Kr[1,2],w,h,1.,'z')
            self.statistics['frames_yielded']+=1;self.statistics['depth_missing']+=registered is None;self.statistics['pose_missing']+=T_world_camera is None
            yield SensorFrameV1(f'openloris-office1-1-{i:06d}',int(t),image,registered,cal,T_world_camera,
                                'recorded_wheel_odometry' if T_world_camera is not None else None)


def replay_bag(path,output,frames=100,period_ms=0.,save_refs=False):
    import cv2
    output=Path(output);output.mkdir(parents=True,exist_ok=True);source=OpenLorisFrameSource(path,frames,minimum_rgb_period_ns=int(period_ms*1e6))
    occupied=set();rows=[];replay_rgb=[];replay_depth=[];replay_K=[];replay_pose=[];start=time.time()
    from .provenance import context,finish
    provenance=context({'bag':str(Path(path).resolve()),'frames':frames,'rgb_period_ms':period_ms,'save_refs':save_refs,'registration_tolerance_ms':20,'maximum_odometry_gap_ms':100,'depth_scale_m':.001,'voxel_size_m':.1})
    try:
        for i,frame in enumerate(source):
            cells=voxel_points(frame)
            replay_rgb.append(frame.rgb);replay_depth.append(frame.depth if frame.depth is not None else np.zeros(frame.rgb.shape[:2],dtype=np.float32))
            c=frame.calibration;replay_K.append(np.array([[c.fx,0,c.cx],[0,c.fy,c.cy],[0,0,1.]]))
            replay_pose.append(frame.T_world_camera if frame.T_world_camera is not None else np.full((4,4),np.nan))
            occupied.update(map(tuple,cells.tolist()))
            rows.append({**frame.to_json(),'occupied_cells_current_frame':len(cells),'occupied_observation_union':len(occupied),
                         'valid_depth_fraction':float(np.mean(frame.depth>0)) if frame.depth is not None else None})
            if i<5 or save_refs:
                cv2.imwrite(str(output/f'rgb-{i:03d}.png'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                if frame.depth is not None:np.savez_compressed(output/f'depth-{i:03d}.npz',depth=frame.depth)
                rows[-1].update({'rgb_path':str((output/f'rgb-{i:03d}.png').resolve()),'depth_path':str((output/f'depth-{i:03d}.npz').resolve()) if frame.depth is not None else None})
    finally:source.close()
    if replay_rgb:
        np.savez_compressed(output/'replay-sensors.npz',rgb=np.stack(replay_rgb),depth=np.stack(replay_depth),K=np.stack(replay_K),T_world_camera=np.stack(replay_pose),timestamp_ns=np.asarray([r['timestamp_ns'] for r in rows],dtype=np.int64))
    np.savez_compressed(output/'observed-voxel-cells.npz',cells=np.asarray(sorted(occupied),dtype=np.int32).reshape(-1,3))
    (output/'sensor-frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result={'provenance':finish(provenance),'status':'verified' if rows else 'unavailable','frames':len(rows),'sensor_clock_duration_s':(rows[-1]['timestamp_ns']-rows[0]['timestamp_ns'])/1e9 if len(rows)>1 else 0.,'metadata':source.metadata(),
            'elapsed_seconds':time.time()-start,'voxel_size_m':.1,'occupied_observation_cells':len(occupied),
            'mapping_label':'recorded-odometry-assisted observed geometry',
            'unknown_cells':'remain unknown; no free-space claim','dynamic_filter':'unavailable in geometric-only union; union is an observation history, not a persistent static occupancy map',
            'accuracy_metrics':{'status':'unavailable','reason':'No independent occupancy/depth/pose labels supplied to this sensor-only replay'},
            'sensor_results':str(output/'sensor-frames.jsonl'),'ROS_replay_sensor_archive':str(output/'replay-sensors.npz'),'resizing_intrinsics':'separate x/y scaling with OpenCV half-pixel centers; registered depth nearest-exact'}
    (output/'run.json').write_text(json.dumps(result,indent=2)+'\n');return result


def frame_from_reference(row):
    import cv2
    rgb=cv2.imread(row['rgb_path'])
    if rgb is None:raise ValueError('Missing RGB reference')
    depth=np.load(row['depth_path'])['depth'] if row.get('depth_path') else None
    return SensorFrameV1(row['sample_id'],row['timestamp_ns'],cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB),depth,
                         Calibration(**row['calibration']),np.asarray(row['T_world_camera']) if row.get('T_world_camera') is not None else None,row.get('pose_source'))
