"""Five isolated processes: physics/ML plus live DDS source/worker/consumer."""
from __future__ import annotations
import hashlib,json,os,subprocess,time,uuid
from pathlib import Path
from .provenance import context,finish
from .ros_sim_node import rpc


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def run_closed_loop(root,output,duration=30.,scenario='clear',seed=42,faults=False,model_pack='fixture',
                    backend='native',device='cpu',threads=1,semantic_pack=None,tracker_lock=None,
                    tracker='T2',semantic_backend='native',active_person=False,quiescent=False):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    if (model_pack!='fixture' or semantic_pack) and not quiescent:raise ValueError('Selected-neural ROS-MuJoCo loop requires root NN quiescence')
    if faults and duration<35:raise ValueError('Fault/recovery proof requires at least35s')
    active_high=None
    if active_person:
        if not tracker_lock:raise ValueError('Canonical tracker lock required for active person guard')
        value=json.loads(Path(tracker_lock).read_text());active_high=value['selected']['parameters']['high_score']
        from .matched_control import bind_tracker_lock
        bind_tracker_lock(tracker_lock,tracker,active_high)
    key=uuid.uuid4().hex[:6];physics_socket=f'cache/v2/robotics/phys-{key}.sock';ml_socket=f'cache/v2/robotics/mdl-{key}.sock'
    pixi=root/'cache/v2/robotics/bin/pixi';manifest=root/'environments/ros-v2-jazzy/pixi.toml';node=root/'packages/robot_reliability_bench/src/robot_reliability_bench/robotics_v2/ros_sim_node.py'
    env=os.environ.copy();env.update({'ROS_DOMAIN_ID':'147','ROS_LOCALHOST_ONLY':'1','RMW_IMPLEMENTATION':'rmw_cyclonedds_cpp','PYTORCH_ENABLE_MPS_FALLBACK':'0','MUJOCO_GL':'cgl','ROBOT_MODEL_THREADS':str(threads),'PIXI_CACHE_DIR':str(root/'cache/v2/robotics/pixi-cache'),'RATTLER_CACHE_DIR':str(root/'cache/v2/robotics/pixi-cache')})
    provenance=context({'mode':'live_ROS_MuJoCo_closed_loop','duration_s':duration,'scenario':scenario,'seed':seed,'faults':faults,
        'model_pack':str(model_pack),'semantic_pack':str(semantic_pack),'device':device,'threads':threads,'active_person':active_person,
        'controller_policy':'Unchanged ObservedController; .3m/s .6rad/s .8m forward guard; actuator250ms command timeout and300ms effective perception age',
        'physical_actuation':False,'truth_to_inference':False,'clock':'Simulation ns preserved separately; physics capture-start system timestamp retained through DDS/IPC for all ages; render/send never refresh old observations',
        'fault_schedule':{'depth_loss_sim_s':[3,5],'slow_worker_sim_s':[8,10],'inference_process_exit_wall_s':6,'inference_restart_wall_s':7.5,'consumer_exit_wall_s':12,'consumer_restart_wall_s':13.5}},root)
    jobs={};handles=[];commands=[];fault_log=[]
    def launch(label,command):
        f=(output/(label+'-process.log')).open('w');handles.append(f)
        p=subprocess.Popen([str(c) for c in command],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT);jobs[label]=p;commands.append({'tag':label,'command':[str(c) for c in command],'pid':p.pid});return p
    def wait_ready(path,process,timeout=60):
        deadline=time.monotonic()+timeout
        while not path.exists():
            if process.poll() is not None:raise RuntimeError('Bridge child failed before ready; inspect unique process log')
            if time.monotonic()>deadline:raise TimeoutError('Bridge child startup timeout')
            time.sleep(.05)
    def launch_ros(role,tag):
        command=[pixi,'run','--manifest-path',manifest,'python',node,'--role',role,'--tag',tag,'--output',output,'--duration',duration,'--physics-socket',physics_socket,'--ml-socket',ml_socket]
        if faults:command+=['--faults']
        p=launch(tag,command);wait_ready(output/(tag+'-events.jsonl'),p)
        deadline=time.monotonic()+60
        while not any(e['event']=='started' for e in events(output/(tag+'-events.jsonl'))):
            if p.poll() is not None:raise RuntimeError('ROS node exited before DDS readiness: '+tag)
            if time.monotonic()>deadline:raise TimeoutError('ROS node readiness timeout: '+tag)
            time.sleep(.05)
        return p
    def stop_ros(tag):
        p=jobs[tag];before=time.monotonic_ns();p.terminate();p.wait(timeout=10)
        fault_log.append({'event':'actual_DDS_process_shutdown','tag':tag,'requested_monotonic_ns':before,'exited_monotonic_ns':time.monotonic_ns(),'exit_code':p.returncode})
    try:
        pc=[root/'environments/robotics-v2/bin/python',root/'tools/robotics_v2.py','serve-simulator','--socket',physics_socket,'--output',output/'physics','--scenario',scenario,'--seed',seed]
        if faults:pc+=['--faults']
        if active_high is not None:pc+=['--active-high-score',active_high]
        physics=launch('physics',pc);wait_ready(root/physics_socket,physics)
        mc=[root/'environments/main/bin/python',root/'tools/robotics_v2.py','serve','--socket',ml_socket,'--model-pack',model_pack,'--backend',backend,'--device',device,'--threads',threads,'--log',output/'ML-events.jsonl']
        if semantic_pack:mc+=['--pipeline-semantic',semantic_pack,'--semantic-backend',semantic_backend]
        if tracker_lock:mc+=['--tracker-lock',tracker_lock,'--pipeline-tracking','--pipeline-tracker',tracker]
        ml=launch('ML',mc);wait_ready(root/ml_socket,ml)
        launch_ros('consumer','consumer-0');launch_ros('inference','inference-0');source=launch_ros('source','source')
        first=None;done=set();deadline=time.monotonic()+duration+70
        while source.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Live source exceeded bounded duration')
            source_events=events(output/'source-events.jsonl');begin=next((e for e in source_events if e['event']=='DDS_live_loop_started'),None)
            if begin and first is None:first=begin['monotonic_ns']/1e9
            elapsed=time.monotonic()-first if first is not None else 0.
            if faults:
                for when,action,tag,role in [(6.,'stop','inference-0','inference'),(7.5,'start','inference-1','inference'),(12.,'stop','consumer-0','consumer'),(13.5,'start','consumer-1','consumer')]:
                    if elapsed>=when and tag not in done:
                        if action=='stop':stop_ros(tag)
                        else:launch_ros(role,tag);fault_log.append({'event':'DDS_process_restarted','tag':tag,'monotonic_ns':time.monotonic_ns()})
                        done.add(tag)
            for tag,p in jobs.items():
                if tag in ('source','physics','ML') or tag in ('inference-0','consumer-0') and tag in done:continue
                if p.poll() is not None:raise RuntimeError('Unexpected bridge ROS child exit: '+tag)
            time.sleep(.05)
        for tag in list(jobs):
            if tag.startswith(('consumer','inference')) and jobs[tag].poll() is None:stop_ros(tag)
        rpc(physics_socket,{'op':'finish'});physics.wait(timeout=10)
        truth=json.loads((output/'physics/physics-summary.json').read_text());trace=events(output/'physics/physics-evaluator.jsonl')
        source_events=events(output/'source-events.jsonl');controls=[e for e in source_events if e['event']=='DDS_control_received']
        received_packets={json.dumps(e['envelope'],sort_keys=True) for e in controls}
        applied=[e for e in trace if e['command_admitted'] and(abs(e['applied_command']['linear_mps'])>0 or abs(e['applied_command']['angular_rps'])>0)]
        consumers=[e for file in output.glob('consumer-*-events.jsonl') for e in events(file)]
        control_events=[e for e in consumers if e['event']=='control_published'];stale=[e for e in control_events if (e['envelope']['command'].get('observed_result_age_ms') or 0)>300]
        summaries=[json.loads(p.read_text()) for p in output.glob('inference-*-summary.json')]
        from ruamel.yaml import YAML
        mcap_topic_counts={}
        for metadata in output.glob('consumer-*-mcap/metadata.yaml'):
            bag=YAML(typ='safe').load(metadata.read_text())['rosbag2_bagfile_information']
            for row in bag['topics_with_message_count']:
                topic=row['topic_metadata']['name'];mcap_topic_counts[topic]=mcap_topic_counts.get(topic,0)+row['message_count']
        expected_topics={'/portfolio_sim_v2/'+t for t in ('rgb','depth','camera_info','wheel_odom','sensor_status','result','control_command')}
        sensors=[e for e in source_events if e['event']=='sensor_published']
        all_inference_events=[e for p in output.glob('inference-*-events.jsonl') for e in events(p)]
        published_ids={e['timestamp_ns'] for e in sensors};received_ids={e['timestamp_ns'] for e in all_inference_events if e['event']=='RGB_received'}
        processed_ids={e['timestamp_ns'] for e in all_inference_events if e['event']=='result_published'}
        consumed_ids={e['timestamp_ns'] for e in consumers if e['event']=='result_consumed'}
        drops={}
        for e in all_inference_events:
            if e['event']=='frame_drop':drops[e['reason']]=drops.get(e['reason'],0)+1
        accounting={'published_unique_RGB_frames':len(published_ids),'subscriber_received_unique_RGB_frames':len(received_ids),
            'not_observed_by_RGB_subscribers_including_actual_disconnect_gaps':len(published_ids-received_ids),
            'published_results_unique':len(processed_ids),'consumed_results_unique':len(consumed_ids),
            'published_results_not_consumed_including_actual_disconnect_gaps':len(processed_ids-consumed_ids),
            'worker_drop_reasons':drops,'worker_accounted_received':sum(s['counts']['processed']+sum(v for k,v in s['counts'].items() if k not in ('received_rgb','processed')) for s in summaries),
            'scope':'Exact observed counts; DDS delivery/discovery/shutdown and deliberate disconnect gaps remain explicit and are not reassigned to queue drops'}
        faults_metrics={}
        if faults:
            depth_begin=next((e for e in sensors if not e['depth_available']),None)
            depth_stop=next((e for e in trace if e['applied_command']['reason']=='actuator_missing_depth'),None)
            faults_metrics['depth_loss_to_next_applied_zero_ms']=(depth_stop['monotonic_ns']-depth_begin['monotonic_ns'])/1e6 if depth_begin and depth_stop else None
            for f in fault_log:
                if f['event']=='actual_DDS_process_shutdown' and f['tag'] in ('inference-0','consumer-0'):
                    rows=[e for e in trace if e['monotonic_ns']>=f['requested_monotonic_ns']]
                    target='actuator_stale_perception' if f['tag']=='inference-0' else 'actuator_command_timeout'
                    first_zero=next((e for e in rows if e['applied_command']['reason']==target),None)
                    faults_metrics[f['tag']+'_shutdown_request_to_applied_zero_ms']=(first_zero['monotonic_ns']-f['requested_monotonic_ns'])/1e6 if first_zero else None
            actual_stale=[e for e in trace if e['requested_DDS_command'] and
                ((e['requested_DDS_command']['command'].get('observed_result_age_ms') or 0)+(e['applied_command']['timestamp_ns']-e['requested_DDS_command']['command']['timestamp_ns'])/1e6)>300]
            faults_metrics['stale_applied_zero_control_steps']=sum(abs(e['applied_command']['linear_mps'])==0 and abs(e['applied_command']['angular_rps'])==0 for e in actual_stale)
            faults_metrics['scope']='Physics samples at100ms control ticks;300ms is maximum observation age before next-loop zero, not elapsed time since process disconnect'
        checks={'actual_DDS_commands_received':len(controls)>0,'actual_nonzero_commands_applied':len(applied)>0,
            'applied_commands_match_received_DDS':all(json.dumps(e['requested_DDS_command'],sort_keys=True) in received_packets for e in applied),
            'robot_reached_goal_without_collision':truth['goal_success'],'actual_goal_progress':truth['maximum_goal_progress_m']>3.5,
            'no_unexpected_every5ms_contacts':truth['every5ms_contact_control_steps']==0,
            'all_applied_velocity_limits':all(abs(e['applied_command']['linear_mps'])<=.3 and abs(e['applied_command']['angular_rps'])<=.6 for e in trace),
            'worker_queue_capacity_two':bool(summaries) and all(s['max_queue']<=2 for s in summaries),
            'MCAP_has_sensor_result_control_topics':all(mcap_topic_counts.get(topic,0)>0 for topic in expected_topics),
            'capture_age_preserved_through_publication':bool(sensors) and all(e['timestamp_ns']==e['capture_begin_ns']<=e['capture_end_ns']<=e['publication_timestamp_ns'] for e in sensors),
            'no_physical_actuator':truth['physical_actuator_connected'] is False,
            'source_and_physics_clean_exit':source.returncode==0 and physics.returncode==0,
            'all_ROS_processes_shutdown_cleanly':all(jobs[t].returncode==0 and any(e['event']=='finished' and e['DDS_endpoints_destroyed'] for e in events(output/(t+'-events.jsonl'))) for t in jobs if t.startswith(('source','consumer','inference'))),
            'worker_received_frames_fully_accounted':accounting['worker_accounted_received']==sum(s['counts']['received_rgb'] for s in summaries),
            'every_observed_stale_control_is_zero':all(e['envelope']['command']['linear_mps']==0 and e['envelope']['command']['angular_rps']==0 for e in stale)}
        if faults:
            def reason_rows(name):return [e for e in trace if e['applied_command']['reason']==name]
            checks.update({'depth_loss_caused_zero_applied_commands':bool(reason_rows('actuator_missing_depth')),
                'slow_inference_caused_stale_stops':any(e['applied_command']['reason'] in ('stale_perception','actuator_stale_perception') for e in trace),
                'actual_inference_DDS_disconnect_and_restart':{'inference-0','inference-1'}<=done,
                'actual_consumer_DDS_disconnect_and_restart':{'consumer-0','consumer-1'}<=done,
                'command_timeout_caused_zero_applied_commands':bool(reason_rows('actuator_command_timeout')),
                'movement_recovered_after_faults':any(e['monotonic_ns']>max(f['monotonic_ns'] for f in fault_log if f['event']=='DDS_process_restarted') and abs(e['applied_command']['linear_mps'])>0 for e in applied),
                'slow_worker_oldest_drop_counted':sum(s['counts']['oldest_dropped'] for s in summaries)>0})
        report={'status':'verified' if all(checks.values()) else 'failed','checks':checks,'provenance':finish(provenance),
            'physics_evaluator_summary':truth,'commands':commands,'fault_events':fault_log,
            'received_DDS_control_messages':len(controls),'nonzero_DDS_commands_applied_physics_steps':len(applied),
            'inference_process_summaries':summaries,'control_commands_published':len(control_events),'observed_stale_zero_controls':len(stale),'MCAP_topic_message_counts':mcap_topic_counts,
            'message_accounting':accounting,'fault_stop_metrics':faults_metrics,
            'capture_to_publication_ms':{'maximum':max((e['publication_timestamp_ns']-e['capture_begin_ns'])/1e6 for e in sensors),'median':sorted((e['publication_timestamp_ns']-e['capture_begin_ns'])/1e6 for e in sensors)[len(sensors)//2]} if sensors else None,
            'NN_performance_claim':False,'selected_neural_model_executed':model_pack!='fixture','model_role':'controlled fixture integration proof' if model_pack=='fixture' else 'selected neural closed-loop rollout; not standalone model latency benchmark',
            'transport':'Live MuJoCo sensor generation -> real DDS RGB/depth/CameraInfo/wheel-odom -> queue2 inference worker -> DDS result -> consumer -> timestamped DDS ControlCommandV1 -> actual simulated wheel actuation',
            'ROS_domain_id':147,'namespace':'/portfolio_sim_v2','RMW_IMPLEMENTATION':'rmw_cyclonedds_cpp','local_only':True,'physical_actuation':False,
            'MCAP_scope':'All seven sensor/status/result/control topics actually received by each consumer instance; disconnect gaps remain gaps',
            'clocks':'Simulation time separately retained; physics capture-start/system timestamp survives render/IPC/DDS and controls result/command age. Fixed100ms physics advancement per sensor tick; OS/render/DDS wall delays measured, no hard real-time claim.'}
        (output/'run.json').write_text(json.dumps(report,indent=2)+'\n');return {k:v for k,v in report.items() if k not in ('commands','inference_process_summaries')}
    finally:
        for p in jobs.values():
            if p.poll() is None:p.terminate()
        for p in jobs.values():
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in handles:f.close()
        (root/physics_socket).unlink(missing_ok=True);(root/ml_socket).unlink(missing_ok=True)
