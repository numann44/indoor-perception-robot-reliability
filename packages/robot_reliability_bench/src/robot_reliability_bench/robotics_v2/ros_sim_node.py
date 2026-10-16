"""Actual DDS live-simulator source/inference/controller, isolated ROS Python."""
import argparse,array,base64,hashlib,json,math,signal,socket,struct,threading,time
from collections import deque
from pathlib import Path
import numpy as np

PREFIX='/portfolio_sim_v2'

def rpc(path,request):
    raw=json.dumps(request,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>16*1024**2:raise ValueError('Bounded IPC input exceeded')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as peer:
        peer.settimeout(5.);peer.connect(path);peer.sendall(struct.pack('!I',len(raw))+raw)
        def exact(n):
            data=b''
            while len(data)<n:
                value=peer.recv(n-len(data))
                if not value:raise ConnectionError('IPC disconnected')
                data+=value
            return data
        n=struct.unpack('!I',exact(4))[0]
        if n>16*1024**2:raise ValueError('Bounded IPC output exceeded')
        value=json.loads(exact(n))
        if value.get('status')!='ok':raise RuntimeError(value.get('error','IPC failed'))
        return value

def stamp(message):return message.header.stamp.sec*10**9+message.header.stamp.nanosec

def wire_array(message,dtype,shape):return {'shape':shape,'dtype':dtype,'bytes':base64.b64encode(bytes(message.data)).decode()}

def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import Image,CameraInfo
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    p=argparse.ArgumentParser();p.add_argument('--role',required=True,choices=('source','inference','consumer'));p.add_argument('--tag',required=True);p.add_argument('--output',required=True)
    p.add_argument('--duration',type=float,default=30.);p.add_argument('--physics-socket',required=True);p.add_argument('--ml-socket',required=True);p.add_argument('--faults',action='store_true');args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);log=(output/f'{args.tag}-events.jsonl').open('w');write_lock=threading.Lock();closing=False
    def event(value):
        with write_lock:log.write(json.dumps({'monotonic_ns':time.monotonic_ns(),**value})+'\n');log.flush()
    def requested_shutdown(*_):
        nonlocal closing
        closing=True
    signal.signal(signal.SIGTERM,requested_shutdown)
    # Our SIGTERM handler closes worker/writer before DDS shutdown. The default
    # rclpy handler would shut its context down before our cleanup executes.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);began=time.monotonic()
    class Source(Node):
        def __init__(self):
            super().__init__('portfolio_sim_source');self.latest=None;self.received_commands=0;self.seen_sequence=-1;self.first=None;self.published=0;self.done=False
            self.rgb=self.create_publisher(Image,PREFIX+'/rgb',8);self.depth=self.create_publisher(Image,PREFIX+'/depth',8);self.cal=self.create_publisher(CameraInfo,PREFIX+'/camera_info',8);self.odom=self.create_publisher(Odometry,PREFIX+'/wheel_odom',8);self.meta=self.create_publisher(String,PREFIX+'/sensor_status',8)
            self.create_subscription(String,PREFIX+'/control_command',self.command,8);self.create_timer(.1,self.tick)
        def command(self,message):
            packet=json.loads(message.data)
            if set(packet)!={'command','consumer_sequence','consumer_instance'}:raise ValueError('Unknown control envelope')
            # Timestamp is the liveness authority; counters may restart with a new process.
            if self.latest and packet['command']['timestamp_ns']<=self.latest['command']['timestamp_ns']:
                event({'event':'control_rejected','reason':'old_or_duplicate_command'});return
            self.latest=packet;self.received_commands+=1
            event({'event':'DDS_control_received','envelope':packet,'envelope_sha256':hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest()})
        def tick(self):
            if any(pub.get_subscription_count()<1 for pub in (self.rgb,self.depth,self.cal,self.odom,self.meta)):
                if time.monotonic()-began>60:raise RuntimeError('DDS sensor discovery timeout')
                return
            if self.first is None:self.first=time.monotonic();event({'event':'DDS_live_loop_started'})
            if time.monotonic()-self.first>=args.duration:self.done=True;return
            sensors=rpc(args.physics_socket,{'op':'sensor_tick','command_envelope':self.latest,'now_ns':self.get_clock().now().nanoseconds})
            frame=sensors['frame'];now=sensors['capture_begin_ns'];frame['timestamp_ns']=now
            def image(wire,encoding):
                result=Image();result.header.stamp.sec=now//10**9;result.header.stamp.nanosec=now%10**9;result.header.frame_id='camera_optical'
                result.height=wire['shape'][0];result.width=wire['shape'][1];result.encoding=encoding;result.step=result.width*(3 if encoding=='rgb8' else 4);result.data=array.array('B',base64.b64decode(wire['bytes']));return result
            rgb=image(frame['rgb'],'rgb8');c=frame['calibration'];ci=CameraInfo();ci.header=rgb.header;ci.width=c['width'];ci.height=c['height'];ci.k=[c['fx'],0.,c['cx'],0.,c['fy'],c['cy'],0.,0.,1.]
            xy=sensors['wheel_odometry'];odom=Odometry();odom.header=rgb.header;odom.header.frame_id='wheel_odom';odom.child_frame_id='base_link';odom.pose.pose.position.x=xy[0];odom.pose.pose.position.y=xy[1];odom.pose.pose.orientation.z=math.sin(xy[2]/2);odom.pose.pose.orientation.w=math.cos(xy[2]/2)
            meta={'timestamp_ns':now,'sample_id':frame['sample_id'],'depth_available':frame['depth'] is not None,'simulation_timestamp_ns':sensors['simulation_timestamp_ns'],'simulation_time_s':sensors['simulation_timestamp_ns']/1e9,
                  'capture_begin_ns':sensors['capture_begin_ns'],'capture_end_ns':sensors['capture_end_ns'],'T_encoder_base_camera':sensors['T_encoder_base_camera']}
            self.cal.publish(ci);self.odom.publish(odom);self.meta.publish(String(data=json.dumps(meta)))
            if frame['depth'] is not None:self.depth.publish(image(frame['depth'],'32FC1'))
            self.rgb.publish(rgb);self.published+=1
            event({'event':'sensor_published','timestamp_ns':now,'publication_timestamp_ns':self.get_clock().now().nanoseconds,'capture_begin_ns':meta['capture_begin_ns'],'capture_end_ns':meta['capture_end_ns'],
                   'sample_id':frame['sample_id'],'depth_available':meta['depth_available'],'simulation_time_s':meta['simulation_time_s']})
        def finish(self):
            (output/f'{args.tag}-summary.json').write_text(json.dumps({'sensor_frames_published':self.published,'DDS_commands_received':self.received_commands,'physical_actuator_connected':False},indent=2)+'\n')
    class Inference(Node):
        def __init__(self):
            super().__init__('portfolio_sim_'+args.tag.replace('-','_'));self.output=self.create_publisher(String,PREFIX+'/result',8)
            self.depths={};self.pending={};self.meta={};self.poses={};self.cals={};self.queue=deque();self.lock=threading.Lock();self.stop=False;self.last=-1;self.seen=set();self.max_queue=0
            self.counts={'received_rgb':0,'processed':0,'oldest_dropped':0,'old_message':0,'duplicate':0,'inference_error':0,'sync_missing':0,'shutdown_pending':0,'inflight_abandoned':0}
            self.create_subscription(Image,PREFIX+'/rgb',self.rgb,8);self.create_subscription(Image,PREFIX+'/depth',lambda m:self.cache(self.depths,stamp(m),m),8)
            self.create_subscription(CameraInfo,PREFIX+'/camera_info',lambda m:self.cache(self.cals,stamp(m),m),8);self.create_subscription(Odometry,PREFIX+'/wheel_odom',lambda m:self.cache(self.poses,stamp(m),m),8)
            self.create_subscription(String,PREFIX+'/sensor_status',self.status,8);self.create_timer(.01,self.flush)
            self.thread=threading.Thread(target=self.worker,daemon=True);self.thread.start()
        def cache(self,dictionary,t,value):
            dictionary[t]=value
            while len(dictionary)>8:dictionary.pop(min(dictionary))
            self.flush()
        def status(self,m):
            value=json.loads(m.data)
            if set(value)!={'timestamp_ns','sample_id','depth_available','simulation_timestamp_ns','simulation_time_s','capture_begin_ns','capture_end_ns','T_encoder_base_camera'}:raise ValueError('Non-sensor metadata rejected')
            if value['timestamp_ns']!=value['capture_begin_ns'] or value['capture_end_ns']<value['capture_begin_ns']:raise ValueError('Capture clock invariant failed')
            self.cache(self.meta,value['timestamp_ns'],value)
        def rgb(self,m):
            t=stamp(m);self.counts['received_rgb']+=1;event({'event':'RGB_received','timestamp_ns':t})
            if t in self.seen:self.counts['duplicate']+=1;event({'event':'frame_drop','reason':'duplicate','timestamp_ns':t});return
            self.seen.add(t)
            if len(self.seen)>1024:self.seen=set(sorted(self.seen)[-512:])
            if t<=self.last:self.counts['old_message']+=1;event({'event':'frame_drop','reason':'old_message','timestamp_ns':t});return
            self.pending[t]=(time.monotonic(),m);self.flush()
        def flush(self):
            for t,(received,rgb) in list(self.pending.items()):
                meta=self.meta.get(t);pose=self.poses.get(t);ci=self.cals.get(t);ds=self.depths.get(t)
                complete=meta is not None and pose is not None and ci is not None and(ds is not None or not meta['depth_available'])
                if not complete and time.monotonic()-received<.2 and len(self.pending)<=2:continue
                self.pending.pop(t)
                if not complete:self.counts['sync_missing']+=1;event({'event':'frame_drop','reason':'sync_missing','timestamp_ns':t});continue
                q=pose.pose.pose.orientation;p=pose.pose.pose.position;yaw=math.atan2(2*q.w*q.z,1-2*q.z*q.z);c,s=math.cos(yaw),math.sin(yaw)
                mount=np.asarray(meta['T_encoder_base_camera'],float)
                if mount.shape!=(4,4) or not np.isfinite(mount).all():raise ValueError('Invalid declared camera mount')
                Tbase=np.eye(4);Tbase[:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]]);Tbase[:3,3]=[p.x,p.y,0.]
                T=Tbase@mount
                wire={'sample_id':meta['sample_id'],'timestamp_ns':t,'rgb':wire_array(rgb,'|u1',[rgb.height,rgb.width,3]),'depth':wire_array(ds,'<f4',[ds.height,ds.width]) if ds is not None else None,
                      'calibration':{'fx':ci.k[0],'fy':ci.k[4],'cx':ci.k[2],'cy':ci.k[5],'width':rgb.width,'height':rgb.height,'depth_scale':1.,'depth_convention':'z'},'T_world_camera':T.tolist(),'pose_source':'wheel_encoder_odometry'}
                with self.lock:
                    if len(self.queue)==2:
                        old=self.queue.popleft();self.counts['oldest_dropped']+=1;event({'event':'frame_drop','reason':'oldest_dropped','timestamp_ns':old[0]['timestamp_ns']})
                    self.queue.append((wire,[p.x,p.y,yaw],meta,time.monotonic_ns()));self.max_queue=max(self.max_queue,len(self.queue))
        def worker(self):
            while not self.stop:
                with self.lock:item=self.queue.popleft() if self.queue else None
                if item is None:time.sleep(.002);continue
                wire,odom,meta,enqueued=item;worker_begin=time.monotonic_ns()
                if wire['timestamp_ns']<=self.last:self.counts['old_message']+=1;continue
                try:
                    if args.faults and 8<=meta['simulation_time_s']<10:time.sleep(.4)
                    result=rpc(args.ml_socket,{'protocol':1,'frame':wire})
                    if self.stop:self.counts['inflight_abandoned']+=1;event({'event':'frame_drop','reason':'inflight_abandoned','timestamp_ns':wire['timestamp_ns']});continue
                    result.update({'control_sensor_frame':wire,'control_wheel_odometry':odom,'source_status':meta,'queue_wait_ms':(worker_begin-enqueued)/1e6,'worker_total_ms':(time.monotonic_ns()-worker_begin)/1e6})
                    self.output.publish(String(data=json.dumps(result,separators=(',',':'))));self.last=wire['timestamp_ns'];self.counts['processed']+=1;event({'event':'result_published','timestamp_ns':self.last,'sample_id':wire['sample_id']})
                except Exception as exc:self.counts['inference_error']+=1;event({'event':'inference_error','error':str(exc),'timestamp_ns':wire['timestamp_ns']})
        def finish(self):
            self.stop=True
            with self.lock:
                self.counts['shutdown_pending']+=len(self.queue)
                for row in self.queue:event({'event':'frame_drop','reason':'shutdown_pending','timestamp_ns':row[0]['timestamp_ns']})
                self.queue.clear()
            self.thread.join(timeout=5.)
            (output/f'{args.tag}-summary.json').write_text(json.dumps({'counts':self.counts,'max_queue':self.max_queue,'inflight_limit':1,'worker_alive_at_close':self.thread.is_alive()},indent=2)+'\n')
    class Consumer(Node):
        def __init__(self):
            super().__init__('portfolio_sim_'+args.tag.replace('-','_'));self.result=None;self.sensor_status=None;self.sequence=0;self.results=0
            self.output=self.create_publisher(String,PREFIX+'/control_command',8)
            import rosbag2_py
            self.writer=rosbag2_py.SequentialWriter();self.writer.open(rosbag2_py.StorageOptions(uri=str(output/(args.tag+'-mcap')),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
            from rclpy.serialization import serialize_message
            self.serialize=serialize_message
            for topic,kind in [('rgb','sensor_msgs/msg/Image'),('depth','sensor_msgs/msg/Image'),('camera_info','sensor_msgs/msg/CameraInfo'),('wheel_odom','nav_msgs/msg/Odometry'),('sensor_status','std_msgs/msg/String'),('result','std_msgs/msg/String'),('control_command','std_msgs/msg/String')]:
                self.writer.create_topic(rosbag2_py.TopicMetadata(id=0,name=PREFIX+'/'+topic,type=kind,serialization_format='cdr'))
            for topic,kind in [('rgb',Image),('depth',Image),('camera_info',CameraInfo),('wheel_odom',Odometry)]:
                self.create_subscription(kind,PREFIX+'/'+topic,lambda m,t=topic:self.record(t,m),8)
            self.create_subscription(String,PREFIX+'/sensor_status',self.status,8);self.create_subscription(String,PREFIX+'/result',self.receive,8);self.create_timer(.05,self.control)
        def record(self,name,message):self.writer.write(PREFIX+'/'+name,self.serialize(message),self.get_clock().now().nanoseconds)
        def status(self,message):self.sensor_status=json.loads(message.data);self.record('sensor_status',message)
        def receive(self,message):
            self.result=json.loads(message.data);self.results+=1;self.record('result',message);event({'event':'result_consumed','timestamp_ns':self.result['timestamp_ns']})
        def control(self):
            now=self.get_clock().now().nanoseconds
            if self.result is None:
                command={'schema_version':1,'timestamp_ns':now,'linear_mps':0.,'angular_rps':0.,'state':'stopped','reason':'no_perception','source_sample_id':None,'observed_result_age_ms':None,'safety_stop':True}
            else:
                command=rpc(args.physics_socket,{'op':'policy','frame':self.result['control_sensor_frame'],'result':{k:v for k,v in self.result.items() if k not in ('control_sensor_frame','control_wheel_odometry','source_status')},
                    'wheel_odometry':self.result['control_wheel_odometry'],'current_sensor_status':self.sensor_status,'now_ns':now})['command']
            packet={'command':command,'consumer_sequence':self.sequence,'consumer_instance':args.tag};message=String(data=json.dumps(packet,separators=(',',':')))
            self.output.publish(message);self.record('control_command',message);self.sequence+=1;event({'event':'control_published','envelope':packet})
        def finish(self):
            (output/f'{args.tag}-summary.json').write_text(json.dumps({'results_consumed':self.results,'commands_published':self.sequence,'MCAP_topics':[PREFIX+'/'+t for t in ('rgb','depth','camera_info','wheel_odom','sensor_status','result','control_command')]},indent=2)+'\n');self.writer=None
    node={'source':Source,'inference':Inference,'consumer':Consumer}[args.role]();executor=SingleThreadedExecutor();executor.add_node(node)
    event({'event':'started','role':args.role,'tag':args.tag,'rmw':rclpy.get_rmw_implementation_identifier()})
    try:
        while not closing and(args.role=='source' or time.monotonic()-began<args.duration+30):
            executor.spin_once(timeout_sec=.02)
            if args.role=='source' and node.done:break
    finally:
        node.finish();executor.remove_node(node);executor.shutdown(timeout_sec=5.);node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        event({'event':'finished','DDS_endpoints_destroyed':True});log.close()
if __name__=='__main__':main()
