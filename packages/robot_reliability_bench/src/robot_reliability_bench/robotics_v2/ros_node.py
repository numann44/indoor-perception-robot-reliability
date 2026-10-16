"""Separate ROS processes; this module needs ROS interfaces, not the ML runtime.

Executed by path to avoid importing the SDK into the ROS Python environment.
Inference runs through a bounded, whitelisted Unix-domain sensor protocol.
"""
import argparse
import array
import base64
from collections import deque
import json
import hashlib
from pathlib import Path
import socket
import struct
import threading
import time

PREFIX='/portfolio_v2'

def timestamp(message):return message.header.stamp.sec*10**9+message.header.stamp.nanosec

def rpc(path,wire):
    raw=json.dumps({'protocol':1,'frame':wire},separators=(',',':')).encode()
    if len(raw)>16*1024**2:raise ValueError('Bounded IPC input exceeded')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(5);client.connect(path);client.sendall(struct.pack('!I',len(raw))+raw)
        def exact(n):
            chunks=[];count=0
            while count<n:
                value=client.recv(n-count)
                if not value:raise ConnectionError('Inference IPC disconnected')
                chunks.append(value);count+=len(value)
            return b''.join(chunks)
        length=struct.unpack('!I',exact(4))[0]
        if length>16*1024**2:raise ValueError('Bounded IPC output exceeded')
        return json.loads(exact(length))

def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image,CameraInfo
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('publisher','inference','consumer'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--duration',type=float,default=15);p.add_argument('--publisher-duration',type=float,default=10)
    p.add_argument('--fps',type=float,default=50);p.add_argument('--sensor');p.add_argument('--frame-refs');p.add_argument('--socket');p.add_argument('--faults',action='store_true');args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    log=(output/f'{args.role}-events.jsonl').open('w');event_lock=threading.Lock()
    def event(value):
        with event_lock:log.write(json.dumps({'monotonic_ns':time.monotonic_ns(),**value})+'\n');log.flush()
    rclpy.init();start=time.monotonic()
    # DDS receive history is separate from the strictly two-frame worker queue.
    # Recorded RGB-D messages need bounded transport history while fragments arrive.
    sensor_history=8 if args.frame_refs else 2
    class Publisher(Node):
        def __init__(self):
            super().__init__('portfolio_v2_recorded_publisher')
            import numpy as np
            data=np.load(args.sensor);self.rgb=data['rgb'];self.depth=data['depth'];self.K=data['K'] if 'K' in data else None;self.poses=data['T_world_camera'] if 'T_world_camera' in data else None;self.n=0;self.previous_stamp=None;self.disconnected=False;self.first_publish_monotonic=None;self.clock_offset_ns=None
            self.references=[json.loads(line) for line in Path(args.frame_refs).read_text().splitlines() if line.strip()] if args.frame_refs else None
            self.reference_sha256=__import__('hashlib').sha256(Path(args.frame_refs).read_bytes()).hexdigest() if args.frame_refs else None
            if self.references is not None and(len(self.references)!=len(self.rgb) or self.rgb.ndim!=4):raise ValueError('Recorded references/archive frame counts differ')
            self.rgbpub=self.create_publisher(Image,PREFIX+'/rgb',sensor_history);self.depthpub=self.create_publisher(Image,PREFIX+'/depth',sensor_history)
            self.calpub=self.create_publisher(CameraInfo,PREFIX+'/camera_info',sensor_history)
            self.odompub=self.create_publisher(Odometry,PREFIX+'/odom',sensor_history)
            self.metapub=self.create_publisher(String,PREFIX+'/source_metadata',sensor_history)
            self.timer=self.create_timer(.005 if self.references is not None else 1/args.fps,self.tick)
        def tick(self):
            if any(pub.get_subscription_count()==0 for pub in (self.rgbpub,self.depthpub,self.calpub,self.odompub,self.metapub)):
                if time.monotonic()-start>60:raise RuntimeError('No actual DDS RGB subscriber discovered within60s')
                return
            if self.first_publish_monotonic is None:
                self.first_publish_monotonic=time.monotonic();event({'event':'DDS_subscriber_discovered','RGB_subscription_count':self.rgbpub.get_subscription_count()})
            if (self.references is not None and self.n>=len(self.references)) or (self.references is None and time.monotonic()-self.first_publish_monotonic>args.publisher_duration):self.timer.cancel();self.disconnected=True;event({'event':'publisher_disconnect','scope':'publisher process exits and rclpy/DDS endpoints shut down immediately after this callback'});return
            now=self.get_clock().now().nanoseconds
            stamp=now;reference=self.references[self.n] if self.references is not None else None
            if reference is not None:
                original=int(reference['timestamp_ns']);first_original=int(self.references[0]['timestamp_ns'])
                if time.monotonic()<self.first_publish_monotonic+(original-first_original)/1e9:return
                if self.clock_offset_ns is None:self.clock_offset_ns=now-first_original
                stamp=original+self.clock_offset_ns
            if args.faults and self.n==20:stamp=self.previous_stamp
            if args.faults and self.n==30:stamp=now-5*10**9
            idx=self.n if reference is not None else(self.n%len(self.rgb) if self.rgb.ndim==4 else 0)
            image=self.rgb[idx] if self.rgb.ndim==4 else self.rgb
            depth_array=self.depth[idx] if self.depth.ndim==3 else self.depth
            h,w=image.shape[:2]
            rgb=Image();rgb.header.stamp.sec=stamp//10**9;rgb.header.stamp.nanosec=stamp%10**9;rgb.header.frame_id='camera_rgb_optical'
            rgb.height=h;rgb.width=w;rgb.encoding='rgb8';rgb.step=w*3;rgb.data=array.array('B',image.tobytes())
            depth=Image();depth.header=rgb.header;depth.height=h;depth.width=w;depth.encoding='32FC1';depth.step=w*4;depth.data=array.array('B',depth_array.astype('<f4').tobytes())
            ci=CameraInfo();ci.header=rgb.header;ci.height=h;ci.width=w
            import math
            f=.5*h/math.tan(math.radians(55)/2);ci.k=(self.K[idx] if self.K.ndim==3 else self.K).reshape(-1).tolist() if self.K is not None else [f,0.,(w-1)/2,0.,f,(h-1)/2,0.,0.,1.]
            self.calpub.publish(ci)
            source_metadata={'timestamp_ns':stamp,'original_timestamp_ns':int(reference['timestamp_ns']) if reference else stamp,'sample_id':reference['sample_id'] if reference else f'ros-{stamp}','source_sequence':self.n,'sensor_reference_sha256':self.reference_sha256,'pose_expected':bool(reference and reference.get('T_world_camera') is not None),'depth_expected':bool(reference.get('depth_available')) if reference else True}
            self.metapub.publish(String(data=json.dumps(source_metadata)))
            if self.poses is not None:
                import numpy as np
                T=self.poses[idx] if self.poses.ndim==3 else self.poses
                if np.isfinite(T).all():
                    R=T[:3,:3];q=np.empty(4);tr=float(np.trace(R))
                    if tr>0:
                        scale=(tr+1)**.5*2;q[:]=[(R[2,1]-R[1,2])/scale,(R[0,2]-R[2,0])/scale,(R[1,0]-R[0,1])/scale,.25*scale]
                    else:
                        j=int(np.argmax(np.diag(R)));a=(j+1)%3;b=(j+2)%3;scale=(1+R[j,j]-R[a,a]-R[b,b])**.5*2
                        q[j]=.25*scale;q[a]=(R[a,j]+R[j,a])/scale;q[b]=(R[b,j]+R[j,b])/scale;q[3]=(R[b,a]-R[a,b])/scale
                    od=Odometry();od.header=rgb.header;od.header.frame_id='recorded_wheel_odom';od.child_frame_id='camera_rgb_optical'
                    od.pose.pose.position.x=float(T[0,3]);od.pose.pose.position.y=float(T[1,3]);od.pose.pose.position.z=float(T[2,3])
                    od.pose.pose.orientation.x=float(q[0]);od.pose.pose.orientation.y=float(q[1]);od.pose.pose.orientation.z=float(q[2]);od.pose.pose.orientation.w=float(q[3]);self.odompub.publish(od)
            if source_metadata['depth_expected'] and not (args.faults and 40<=self.n<45):self.depthpub.publish(depth)
            self.rgbpub.publish(rgb);self.previous_stamp=stamp;self.n+=1
            event({'event':'published','sequence':self.n,'timestamp_ns':stamp,'depth_published':source_metadata['depth_expected'] and not(args.faults and 40<=self.n-1<45),'source_metadata':source_metadata,'publication_clock_lag_ms':(now-stamp)/1e6,'RGB_payload_sha256':hashlib.sha256(bytes(rgb.data)).hexdigest(),'depth_payload_sha256':hashlib.sha256(bytes(depth.data)).hexdigest()})
    class Inference(Node):
        def __init__(self):
            super().__init__('portfolio_v2_inference_subscriber')
            self.results=self.create_publisher(String,PREFIX+'/result',2)
            self.depths={};self.pending={};self.poses={};self.source_meta={};self.cal=None;self.queue=deque();self.lock=threading.Lock();self.stop=False;self.seen=set();self.last_processed=-1
            self.counters={'received_rgb':0,'duplicate':0,'out_of_order':0,'oldest_dropped':0,'processed':0,'inference_error':0,'missing_depth':0,'calibration_missing':0,'shutdown_pending':0,'inflight_abandoned':0,'source_metadata_missing':0,'recorded_metadata_processed':0};self.max_queue=0
            self.create_subscription(String,PREFIX+'/source_metadata',self.on_source_metadata,sensor_history)
            self.create_subscription(Odometry,PREFIX+'/odom',self.on_odom,sensor_history)
            self.create_subscription(CameraInfo,PREFIX+'/camera_info',self.on_calibration,sensor_history)
            self.create_subscription(Image,PREFIX+'/depth',self.on_depth,sensor_history);self.create_subscription(Image,PREFIX+'/rgb',self.on_rgb,sensor_history)
            self.create_timer(.02,self.flush_pending)
            self.thread=threading.Thread(target=self.worker,daemon=True);self.thread.start()
        def on_source_metadata(self,m):
            value=json.loads(m.data)
            if set(value)!={'timestamp_ns','original_timestamp_ns','sample_id','source_sequence','sensor_reference_sha256','pose_expected','depth_expected'}:raise ValueError('Non-sensor fields in source metadata rejected')
            event({'event':'received_source_metadata','timestamp_ns':value['timestamp_ns'],'source_metadata':value})
            self.source_meta[value['timestamp_ns']]=value
            while len(self.source_meta)>8:self.source_meta.pop(min(self.source_meta))
            self.flush_pending()
        def on_calibration(self,m):
            self.cal=m;event({'event':'received_calibration','timestamp_ns':timestamp(m),'K':list(m.k),'width':m.width,'height':m.height})
        def on_odom(self,m):
            event({'event':'received_odometry','timestamp_ns':timestamp(m)})
            import numpy as np
            p=m.pose.pose.position;q=m.pose.pose.orientation;x,y,z,w=q.x,q.y,q.z,q.w
            T=np.eye(4);T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];T[:3,3]=[p.x,p.y,p.z]
            self.poses[timestamp(m)]=T.tolist()
            while len(self.poses)>8:self.poses.pop(min(self.poses))
        def on_depth(self,m):
            event({'event':'received_depth','timestamp_ns':timestamp(m),'payload_sha256':hashlib.sha256(bytes(m.data)).hexdigest(),'encoding':m.encoding,'width':m.width,'height':m.height})
            self.depths[timestamp(m)]=m
            while len(self.depths)>8:self.depths.pop(min(self.depths))
            self.flush_pending()
        def on_rgb(self,m):
            self.counters['received_rgb']+=1;t=timestamp(m)
            event({'event':'received_rgb','timestamp_ns':t,'payload_sha256':hashlib.sha256(bytes(m.data)).hexdigest(),'encoding':m.encoding,'width':m.width,'height':m.height})
            if t in self.seen:self.counters['duplicate']+=1;event({'event':'frame_drop','reason':'duplicate','timestamp_ns':t});return
            self.seen.add(t)
            # Bound duplicate history rather than retaining every session frame.
            if len(self.seen)>1024:self.seen=set(sorted(self.seen)[-512:])
            if t<=self.last_processed:self.counters['out_of_order']+=1;event({'event':'frame_drop','reason':'out_of_order','timestamp_ns':t});return
            self.pending[t]=(time.monotonic(),m)
            self.flush_pending()
        def flush_pending(self):
            for t,(received,m) in list(self.pending.items()):
                ds=self.depths.get(t);meta=self.source_meta.get(t)
                arrival_grace=.2 if args.frame_refs else .021
                if args.frame_refs and (meta is None or(meta['pose_expected'] and t not in self.poses)) and time.monotonic()-received<arrival_grace and len(self.pending)<=2:continue
                if ds is None and time.monotonic()-received<arrival_grace and len(self.pending)<=2:continue
                self.pending.pop(t,None)
                if self.cal is None:self.counters['calibration_missing']+=1;event({'event':'frame_drop','reason':'calibration_missing','timestamp_ns':t});continue
                k=self.cal.k
                calibration={'fx':k[0],'fy':k[4],'cx':k[2],'cy':k[5],'width':m.width,'height':m.height,'depth_scale':1.,'depth_convention':'z'}
                def body(im,dtype,shape):return {'shape':shape,'dtype':dtype,'bytes':base64.b64encode(bytes(im.data)).decode()}
                wire={'sample_id':meta['sample_id'] if meta else f'ros-{t}','timestamp_ns':t,'rgb':body(m,'|u1',[m.height,m.width,3]),
                      'depth':body(ds,'<f4',[ds.height,ds.width]) if ds is not None else None,'calibration':calibration,
                      'T_world_camera':self.poses.get(t),'pose_source':'recorded_wheel_odometry' if t in self.poses else None}
                if ds is None:self.counters['missing_depth']+=1
                if args.frame_refs and meta is None:self.counters['source_metadata_missing']+=1
                with self.lock:
                    if len(self.queue)==2:
                        old,_,old_meta=self.queue.popleft();self.counters['oldest_dropped']+=1;event({'event':'frame_drop','reason':'oldest_dropped','timestamp_ns':old['timestamp_ns'],'sample_id':old['sample_id'],'source_metadata':old_meta})
                    self.queue.append((wire,time.monotonic_ns(),meta));self.max_queue=max(self.max_queue,len(self.queue))
        def worker(self):
            while not self.stop:
                with self.lock:item=self.queue.popleft() if self.queue else None
                if item is None:time.sleep(.002);continue
                wire,enqueued,source_metadata=item
                if wire['timestamp_ns']<=self.last_processed:self.counters['out_of_order']+=1;event({'event':'frame_drop','reason':'out_of_order','timestamp_ns':wire['timestamp_ns'],'sample_id':wire['sample_id']});continue
                began=time.monotonic_ns()
                try:
                    result=rpc(args.socket,wire)
                    if result.get('status')!='ok':raise RuntimeError(result.get('error','backend failure'))
                    result.update({'source_metadata':source_metadata,'clock_policy':'Original sensor clock preserved separately; recorded publication uses a single constant offset into ROS publication clock' if args.frame_refs else 'Synthetic stress clock is ROS publication time','queue_wait_ms':(began-enqueued)/1e6,'subscriber_to_result_ms':(time.monotonic_ns()-enqueued)/1e6,
                                   'depth_available':wire['depth'] is not None,'counters':dict(self.counters),'max_queue_observed':self.max_queue})
                    self.results.publish(String(data=json.dumps(result)));self.last_processed=wire['timestamp_ns'];self.counters['processed']+=1
                    if args.frame_refs and source_metadata is not None:self.counters['recorded_metadata_processed']+=1
                    event({'event':'result','sample_id':result['sample_id'],'timestamp_ns':result['timestamp_ns'],'queue_wait_ms':result['queue_wait_ms']})
                except Exception as exc:self.counters['inference_error']+=1;event({'event':'inference_error','error':str(exc),'timestamp_ns':wire['timestamp_ns'],'sample_id':wire['sample_id']})
        def finish(self):
            self.stop=True
            with self.lock:
                self.counters['shutdown_pending']+=len(self.queue)
                for wire,_,metadata in self.queue:event({'event':'frame_drop','reason':'shutdown_pending','timestamp_ns':wire['timestamp_ns'],'sample_id':wire['sample_id'],'source_metadata':metadata})
                self.queue.clear()
            self.thread.join(timeout=5.)
            if self.thread.is_alive():self.counters['inflight_abandoned']+=1
            (output/'inference-summary.json').write_text(json.dumps({'counters':self.counters,'max_queue_observed':self.max_queue,'inflight_worker_limit':1,'worker_alive_at_close':self.thread.is_alive()},indent=2)+'\n')
    class Consumer(Node):
        def __init__(self):
            super().__init__('portfolio_v2_result_consumer');self.latest=None;self.last_received=None;self.processed=0;self.stops=0;self.last_state=None
            self.cmd=self.create_publisher(Twist,PREFIX+'/cmd_vel',2)
            self.create_subscription(String,PREFIX+'/result',self.on_result,2)
            self.create_timer(.05,self.control)
            self.writer=None
            try:
                import rosbag2_py
                self.writer=rosbag2_py.SequentialWriter();self.writer.open(rosbag2_py.StorageOptions(uri=str(output/'rosbag'),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
                for name,kind in ((PREFIX+'/result','std_msgs/msg/String'),(PREFIX+'/cmd_vel','geometry_msgs/msg/Twist')):
                    self.writer.create_topic(rosbag2_py.TopicMetadata(id=0,name=name,type=kind,serialization_format='cdr'))
                event({'event':'rosbag_open','storage':'mcap'})
            except Exception as exc:event({'event':'rosbag_error','error':str(exc)})
        def record(self,topic,message):
            if self.writer:
                from rclpy.serialization import serialize_message
                self.writer.write(topic,serialize_message(message),self.get_clock().now().nanoseconds)
        def on_result(self,message):
            self.latest=json.loads(message.data);self.last_received=time.monotonic();self.processed+=1
            self.record(PREFIX+'/result',message)
            event({'event':'consumed','timestamp_ns':self.latest['timestamp_ns'],'age_ms':(self.get_clock().now().nanoseconds-self.latest['timestamp_ns'])/1e6})
        def control(self):
            now=self.get_clock().now().nanoseconds
            age=(now-self.latest['timestamp_ns'])/1e6 if self.latest else None
            geometry=self.latest.get('observed_geometry') if self.latest else None
            reason='no_result' if not self.latest else ('future_timestamp' if age<0 else ('stale_result' if age>300 else ('missing_depth' if not self.latest['depth_available'] else 'observed_result_available')))
            if reason=='observed_result_available' and (geometry is None or not geometry.get('ground_validated')):reason='ground_geometry_unavailable'
            if reason=='observed_result_available' and geometry['forward_m']<.8:reason='obstacle_safety_stop'
            cmd=Twist()
            if reason=='observed_result_available':cmd.linear.x=.1
            else:self.stops+=1
            self.cmd.publish(cmd);self.record(PREFIX+'/cmd_vel',cmd)
            event({'event':'control_tick','reason':reason,'result_age_ms':age,'linear_mps':cmd.linear.x,'angular_rps':cmd.angular.z,'source_timestamp_ns':self.latest['timestamp_ns'] if self.latest else None,'control_timestamp_ns':now})
            if reason!=self.last_state:event({'event':'control_state','reason':reason,'result_age_ms':age,'linear_mps':cmd.linear.x});self.last_state=reason
        def finish(self):
            (output/'consumer-summary.json').write_text(json.dumps({'processed':self.processed,'safe_stop_ticks':self.stops,'final_reason':self.last_state,'rosbag_written':self.writer is not None,'controller_period_s':.05,'stale_limit_ms':300,'command_topic':PREFIX+'/cmd_vel','physical_actuator_connected':False},indent=2)+'\n')
            self.writer=None
    from rclpy.executors import SingleThreadedExecutor
    node={'publisher':Publisher,'inference':Inference,'consumer':Consumer}[args.role]()
    executor=SingleThreadedExecutor();executor.add_node(node)
    start=time.monotonic()
    event({'event':'started','role':args.role,'RMW_IMPLEMENTATION':rclpy.get_rmw_implementation_identifier(),'DDS_sensor_history':sensor_history})
    try:
        while args.role=='publisher' or time.monotonic()-start<args.duration:
            executor.spin_once(timeout_sec=.02)
            if args.role=='publisher' and node.disconnected:break
    finally:
        if hasattr(node,'finish'):node.finish()
        executor.remove_node(node);executor.shutdown(timeout_sec=5.);node.destroy_node();rclpy.shutdown();event({'event':'finished','scope':'ROS node destroyed and DDS endpoints shut down'});log.close()
if __name__=='__main__':main()
