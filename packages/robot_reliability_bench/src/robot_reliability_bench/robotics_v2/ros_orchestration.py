"""Launch actual DDS processes plus an isolated local ML inference server."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import numpy as np
import psutil


def run_ros(root,output,duration=20.,publisher_duration=15.,fps=50.,model_pack='fixture',delay_ms=40.,backend='native',device='cpu',threads=1,sensor=None,pipeline_semantic=None,pipeline_tracker='T2',tracker_lock=None,semantic_backend='native',frame_refs=None):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    references=None
    if frame_refs:
        frame_refs=Path(frame_refs).resolve();references=[json.loads(line) for line in frame_refs.read_text().splitlines() if line.strip()]
        if not references or any(row.get('pose_source') not in (None,'recorded_wheel_odometry','wheel_encoder_odometry') or set(row)&{'targets_path','labels','ground_truth','annotation_path'} for row in references):
            raise ValueError('Recorded ROS source must contain only non-GT sensor references')
        publisher_duration=(references[-1]['timestamp_ns']-references[0]['timestamp_ns'])/1e9
        duration=max(duration,publisher_duration+8.)
        if sensor is None:sensor=frame_refs.parent/'replay-sensors.npz'
        with np.load(sensor) as archive:
            if len(archive['rgb'])!=len(references) or not np.array_equal(archive['timestamp_ns'],[row['timestamp_ns'] for row in references]):raise ValueError('Recorded sensor archive does not match exact reference timestamps/count')
    if duration<=publisher_duration+.5:raise ValueError('Allow >0.5 s for actual publisher disconnect watchdog')
    socket=f'cache/v2/robotics/rpc-{uuid.uuid4().hex[:6]}.sock'
    pixi=root/'cache/v2/robotics/bin/pixi';manifest=root/'environments/ros-v2-jazzy/pixi.toml'
    node=root/'packages/robot_reliability_bench/src/robot_reliability_bench/robotics_v2/ros_node.py'
    sensor=Path(sensor) if sensor else root/'reports/v2/robotics/mujoco-smoke/sensor.npz'
    environment=os.environ.copy();environment.update({'ROS_DOMAIN_ID':'146','ROS_LOCALHOST_ONLY':'1','RMW_IMPLEMENTATION':'rmw_cyclonedds_cpp',
        'PIXI_CACHE_DIR':str(root/'cache/v2/robotics/pixi-cache'),'RATTLER_CACHE_DIR':str(root/'cache/v2/robotics/pixi-cache'),
        'ROBOT_MODEL_THREADS':str(threads),'PYTORCH_ENABLE_MPS_FALLBACK':'0'})
    jobs=[];files=[];commands=[];memory=[];start=time.monotonic()
    from .provenance import context,finish
    provenance=context({'duration_s':duration,'publisher_duration_s':publisher_duration,'fps':fps,'model_pack':str(model_pack),'pipeline_semantic':str(pipeline_semantic) if pipeline_semantic else None,'pipeline_tracker':pipeline_tracker,'semantic_backend':semantic_backend,'tracker_lock':str(tracker_lock) if tracker_lock else None,'backend':backend,'device':device,'threads':threads,'delay_ms':delay_ms,'mode':'recorded' if frame_refs else 'fixture-stress','frame_refs_sha256':hashlib.sha256(frame_refs.read_bytes()).hexdigest() if frame_refs else None,'pipeline_semantic':str(pipeline_semantic) if pipeline_semantic else None,'pipeline_tracker':pipeline_tracker,'semantic_backend':semantic_backend,'tracker_lock':str(tracker_lock) if tracker_lock else None},root)
    def launch(label,command):
        log=(output/f'{label}-process.log').open('w');files.append(log)
        process=subprocess.Popen([str(c) for c in command],cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((label,process));commands.append({'role':label,'command':[str(c) for c in command],'pid':process.pid});return process
    server_command=[root/'environments/main/bin/python',root/'tools/robotics_v2.py','serve','--socket',socket,'--model-pack',model_pack,'--backend',backend,'--device',device,'--threads',threads,'--delay-ms',delay_ms,'--log',output/'ipc-server.jsonl']
    if pipeline_semantic:server_command+=['--pipeline-semantic',str(pipeline_semantic),'--pipeline-tracker',pipeline_tracker,'--semantic-backend',semantic_backend]
    if tracker_lock:server_command+=['--tracker-lock',str(tracker_lock),'--pipeline-tracking']
    server=launch('inference-ml',server_command)
    try:
        deadline=time.monotonic()+60
        while not (root/socket).exists():
            if server.poll() is not None:raise RuntimeError('ML inference server failed; see process log')
            if time.monotonic()>deadline:raise TimeoutError('Inference server did not initialize')
            time.sleep(.05)
        for role in ('consumer','inference'):
            role_command=[pixi,'run','--manifest-path',manifest,'python',node,'--role',role,'--output',output,'--duration',duration+3,'--socket',socket]
            if frame_refs:role_command+=['--frame-refs',frame_refs]
            launch(role,role_command)
        ready_deadline=time.monotonic()+60.
        while True:
            ready=[]
            for role in ('consumer','inference'):
                path=output/f'{role}-events.jsonl'
                ready.append(path.exists() and any(json.loads(line).get('event')=='started' for line in path.read_text().splitlines() if line.strip()))
            if all(ready):break
            if any(p.poll() is not None for role,p in jobs if role in ('consumer','inference')):
                raise RuntimeError('ROS subscriber/consumer failed before readiness')
            if time.monotonic()>ready_deadline:raise TimeoutError('ROS subscriber/consumer not ready within60s')
            time.sleep(.05)
        publisher_command=[pixi,'run','--manifest-path',manifest,'python',node,'--role','publisher','--output',output,'--duration',duration,'--publisher-duration',publisher_duration,'--fps',fps,'--sensor',sensor]
        if frame_refs:publisher_command+=['--frame-refs',frame_refs]
        else:publisher_command+=['--faults']
        publisher=launch('publisher',publisher_command)
        timeout=time.monotonic()+duration+65
        while any(p.poll() is None for name,p in jobs if name!='inference-ml'):
            if time.monotonic()>timeout:raise TimeoutError('ROS child process exceeded session duration')
            snapshot={'elapsed_wall_s':time.monotonic()-start,'process_rss_bytes':{}}
            for name,process in jobs:
                if process.poll() is not None:continue
                try:
                    parent=psutil.Process(process.pid);tree=[parent,*parent.children(recursive=True)];detail=[]
                    for proc in tree:
                        try:detail.append({'pid':proc.pid,'name':proc.name(),'rss_bytes':proc.memory_info().rss})
                        except psutil.NoSuchProcess:pass
                    snapshot['process_rss_bytes'][name]=sum(p['rss_bytes'] for p in detail)
                    snapshot.setdefault('process_tree',{})[name]=detail
                except psutil.NoSuchProcess:pass
            memory.append(snapshot);time.sleep(1.)
        exits={name:p.returncode for name,p in jobs if name!='inference-ml'}
        inference=json.loads((output/'inference-summary.json').read_text())
        consumer=json.loads((output/'consumer-summary.json').read_text())
        events=[json.loads(x) for x in (output/'consumer-events.jsonl').read_text().splitlines()]
        controls=[e for e in events if e.get('event')=='control_tick']
        if not controls:controls=[e for e in events if e.get('event')=='control_state']
        pub_events=[json.loads(x) for x in (output/'publisher-events.jsonl').read_text().splitlines()]
        disconnect=next((e['monotonic_ns'] for e in pub_events if e.get('event')=='publisher_disconnect'),None)
        post=[e for e in controls if disconnect is not None and e['monotonic_ns']>=disconnect and e.get('reason')=='stale_result']
        stop_delay=(post[0]['monotonic_ns']-disconnect)/1e6 if post else None
        stopped_post=[e for e in controls if disconnect is not None and e['monotonic_ns']>=disconnect and e.get('linear_mps')==0 and e.get('angular_rps',0)==0]
        actual_zero_delay=(stopped_post[0]['monotonic_ns']-disconnect)/1e6 if stopped_post else None
        publisher_finished=next((e['monotonic_ns'] for e in pub_events if e.get('event')=='finished'),None)
        fault_reasons={'stale_result','missing_depth','future_timestamp','no_result','ground_geometry_unavailable','obstacle_safety_stop'}
        faults=[e for e in controls if e.get('reason') in fault_reasons]
        tick_periods=np.diff([e['monotonic_ns'] for e in controls])/1e6
        stale_ticks=[e for e in controls if e.get('result_age_ms') is not None and e['result_age_ms']>300]
        loop_observation={'nominal_period_ms':50,'actual_period_p50_ms':float(np.median(tick_periods)) if len(tick_periods) else None,'actual_period_p95_ms':float(np.percentile(tick_periods,95)) if len(tick_periods) else None,'actual_period_max_ms':float(np.max(tick_periods)) if len(tick_periods) else None,'age_over300ms_ticks':len(stale_ticks),'all_age_over300ms_ticks_zero':all(e['linear_mps']==0 and e.get('angular_rps',0)==0 for e in stale_ticks),'scope':'Actual consumer callback intervals and observed age predicate; disconnect elapsed time is separate. No hard real-time scheduler or physical stopping guarantee.'}
        published_count=sum(e.get('event')=='published' for e in pub_events)
        message_accounting={'publisher_RGB':published_count,'subscriber_RGB':inference['counters']['received_rgb'],
            'unobserved_RGB_transport_or_discovery':published_count-inference['counters']['received_rgb'],
            'worker_results':inference['counters']['processed'],'consumer_results':consumer['processed'],
            'unconsumed_results_transport_or_lifecycle':inference['counters']['processed']-consumer['processed'],
            'unknown_cause_policy':'Counts are measured from publisher/subscriber/worker/consumer logs; transport/discovery/lifecycle causes cannot be distinguished and are not invented',
            'received_RGB_fully_accounted':inference['counters']['received_rgb']==sum(inference['counters'].get(k,0) for k in ('duplicate','out_of_order','oldest_dropped','processed','inference_error','calibration_missing','shutdown_pending','inflight_abandoned'))}
        checks={'all_ROS_process_exit_codes_zero':all(v==0 for v in exits.values()),'actual_results_consumed':consumer['processed']>0,
                'queue_capacity_two':inference['max_queue_observed']<=2,'duplicate_counted':inference['counters']['duplicate']>0,
                'out_of_order_counted':inference['counters']['out_of_order']>0,'oldest_drop_observed':inference['counters']['oldest_dropped']>0,
                'ROSbag_written':consumer['rosbag_written'],'publisher_disconnect_stops':consumer['final_reason']=='stale_result',
                'no_physical_actuator':consumer['physical_actuator_connected'] is False,'received_frame_accounting_exact':message_accounting['received_RGB_fully_accounted'],'worker_stopped_at_close':not inference['worker_alive_at_close'],'all_fault_control_ticks_zero':bool(faults) and all(e['linear_mps']==0 and e.get('angular_rps',0)==0 for e in faults),'publisher_process_disconnected':publisher_finished is not None and disconnect is not None and publisher_finished-disconnect<1_000_000_000}
        transport_validation=None
        if frame_refs:
            from .ros_validation import validate_recorded_transport
            subscriber_events=[json.loads(line) for line in (output/'inference-events.jsonl').read_text().splitlines() if line.strip()]
            transport_validation=validate_recorded_transport(references,pub_events,subscriber_events)
            checks.update(transport_validation['checks'])
            (output/'recorded-transport-validation.json').write_text(json.dumps(transport_validation,indent=2)+'\n')
            for key in ('duplicate_counted','out_of_order_counted','oldest_drop_observed'):checks.pop(key)
            checks['all_selected_source_frames_published']=published_count==len(references)
            checks['original_metadata_preserved_for_processed_frames']=inference['counters']['recorded_metadata_processed']==inference['counters']['processed'] and inference['counters']['source_metadata_missing']==0
        result={'provenance':finish(provenance),'status':'verified' if all(checks.values()) else 'failed','checks':checks,'commands':commands,'exit_codes':exits,
                'duration_requested_s':duration,'elapsed_wall_s':time.monotonic()-start,'publisher_duration_s':publisher_duration,
                'publisher_disconnect_to_stop_ms':stop_delay,'publisher_disconnect_to_first_zero_command_ms':actual_zero_delay,'publisher_disconnect_to_process_shutdown_ms':(publisher_finished-disconnect)/1e6 if publisher_finished is not None and disconnect is not None else None,'disconnect_stop_semantics':'Publisher process actually exits/shuts down DDS endpoints. Consumer checks source timestamp age on each nominal50ms loop and always publishes zero for age>300ms/missingdepth/futuretimestamp; OS scheduling and queue latency are measured, not assumed. First zero command may precede first stale-state transition.',
                'source_sensor':str(sensor.resolve()),'source_sensor_sha256':hashlib.sha256(sensor.read_bytes()).hexdigest(),
                'model_pack':str(model_pack),'pipeline_semantic':str(pipeline_semantic) if pipeline_semantic else None,'pipeline_tracker':pipeline_tracker,'semantic_backend':semantic_backend,'tracker_lock':str(tracker_lock) if tracker_lock else None,'backend':backend,'device':device,'threads':threads,'controlled_backend_delay_ms':delay_ms,
                'inference':inference,'consumer':consumer,'message_accounting':message_accounting,'control_loop_observation':loop_observation,'memory_samples':memory,
                'transport':'Actual separate ROS2 Jazzy publisher/subscriber/consumer processes with local DDS; JSON Unix socket crosses independent ML Python runtime',
                'recorded_transport_validation':transport_validation,'source_frame_refs':str(frame_refs) if frame_refs else None,'source_frame_refs_sha256':hashlib.sha256(frame_refs.read_bytes()).hexdigest() if frame_refs else None,'source_unique_frames':len(references) if references else None,'source_clock_duration_s':publisher_duration if references else None,
                'sensor_clock_policy':('Recorded frames published once at original relative sensor intervals. A single constant clock offset maps original timestamps into ROS clock; source_metadata topic and every result preserve original sample/timestamp/reference SHA.' if frame_refs else 'Source RGB-D/pose arrays are cyclic replay inputs retimestamped to ROS wall-clock for real-time queue/watchdog stress; source archive retains original timestamps. Cycling is not a continuous mapping-quality experiment.'),
                'RMW_IMPLEMENTATION':environment['RMW_IMPLEMENTATION'],'DDS_sensor_history':8 if frame_refs else 2,'DDS_history_is_worker_queue':False,'ROS_domain_id':146,'local_only':True,'memory_definition':'Sampled RSS for each launched role and all descendants (Pixi wrapper + ROS Python when distinct); shared library pages can be counted in more than one role. Sampling 1s, not instantaneous allocation peak.', 'NN_label_access':False,'physical_robot_validation':False,
                'ROSbag_scope':'Actual result and cmd_vel topics recorded in MCAP; RGB-D provenance is the hashed replay sensor file, not a claim that the output bag records all input sensors'}
        (output/'run.json').write_text(json.dumps(result,indent=2)+'\n')
        (output/'memory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in memory))
        return {k:v for k,v in result.items() if k not in ('commands','memory_samples','inference','consumer')}
    finally:
        for _,p in reversed(jobs):
            if p.poll() is None:p.terminate()
        for _,p in jobs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for file in files:file.close()
        (root/socket).unlink(missing_ok=True)
