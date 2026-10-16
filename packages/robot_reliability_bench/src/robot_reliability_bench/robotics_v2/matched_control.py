"""Paired geometry/active learned-person experiments on predeclared seeds."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .controller import ObservedController, observed_clearance
from .learned_control import POLICY, learned_person_observation, constrain_command
from .remote_model import RemoteModelSession
from .simulator import MuJoCoFrameSource, SCENARIOS, evaluator_state
from .provenance import context, finish


def bind_tracker_lock(lock_path, tracker, high_score):
    """Require exact canonical tracking values, not arbitrary CLI thresholds."""
    path=Path(lock_path)
    value=json.loads(path.read_text())
    candidates=[value]
    for key in ('canonical','canonical_config','selected','selection','config'):
        if isinstance(value.get(key),dict):candidates.append(value[key])
    selected=value.get('selected',{})
    tracking=({'preset':selected['configuration'],**selected['parameters']}
              if 'configuration' in selected and 'parameters' in selected
              else next((c['tracking'] for c in candidates if isinstance(c.get('tracking'),dict)),None))
    if tracking is None or tracking.get('high_score') != high_score:
        raise ValueError('Canonical lock must provide tracking.high_score matching the bound value')
    preset=tracking.get('preset',tracking.get('configuration',tracking.get('name',tracking.get('id'))))
    if preset != tracker:
        raise ValueError('Canonical lock tracking preset does not match selected tracker')
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'tracking':tracking,'detector_weights_sha256':value.get('detector_weights_sha256'),
            'detector_manifest_sha256':value.get('detector_manifest_sha256'),'tracker_source_sha256':value.get('tracker_source_sha256')}


def run_matched_control(root, output, model_pack, tracker, high_score, tracker_lock,
                        *, quiescent=False, backend='native', device='cpu', threads=1):
    if not quiescent:
        raise ValueError('Actual selected-model simulation requires explicit training-quiescent attestation')
    root,output=Path(root),Path(output)
    frozen=json.loads((root/'configs/v2/robotics-active-person.json').read_text())
    binding=bind_tracker_lock(tracker_lock,tracker,high_score)
    controller_hash=hashlib.sha256((Path(__file__).parent/'controller.py').read_bytes()).hexdigest()
    if controller_hash != frozen['controller_sha256']:
        raise ValueError('Geometric controller changed after active-policy freeze')
    output.mkdir(parents=True,exist_ok=True)
    manifest=Path(model_pack)/'manifest.json'
    pack_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    weights=Path(model_pack)/'weights.pt'
    if binding['detector_weights_sha256']!=hashlib.sha256(weights.read_bytes()).hexdigest():
        raise ValueError('Selected tracker lock is bound to different detector weights')
    if binding['detector_manifest_sha256'] != pack_hash:
        raise ValueError('Selected tracker lock is bound to a different detector manifest')
    protocol={**frozen,'high_score':high_score,'tracker_lock':binding,'model_pack':str(model_pack),
              'model_manifest_sha256':pack_hash,'backend':backend,'device':device,'threads':threads,
              'physics_contact_sampling':'every5ms physics step; minimum clearance sampled every100ms control step'}
    (output/'bound-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    provenance=context(protocol,root)
    rows=[]
    for condition in frozen['conditions']:
        active=condition=='geometry_plus_active_person_constraint'
        for scenario in SCENARIOS:
            for seed in range(200,210):
                episode=output/condition/f'{scenario}-seed{seed}'
                episode.mkdir(parents=True,exist_ok=True)
                source=MuJoCoFrameSource(scenario,seed)
                controller=ObservedController()
                pending=[];latest=None;latest_stamp=None;learned_pending=[];learned_latest=None
                trace=[];minclear=float('inf');contacts=0;reach=False;fault_started=None
                fault_last=False;stop_delays=[];awaiting_recovery=False;recovered=False
                try:
                    # Preserve the frozen geometric protocol's initial capture/noise draw.
                    source.capture()
                    source.current_odom=source.odom.update(source.data.qpos[source.wheel_qpos])
                    with RemoteModelSession(root,model_pack,episode/'inference',backend=backend,
                                            device=device,threads=threads,tracker=tracker,tracker_lock=tracker_lock) as model:
                        for step in range(round(frozen['episode_seconds']/.1)):
                            t=float(source.data.time);now=int(round(t*1e9))
                            missing=scenario=='depth_loss' and 3<=t<5
                            controlled_delay=.4 if scenario=='slow_inference' and 3<=t<5 else 0.
                            frame=source.capture(depth_missing=missing)
                            odom_input=source.current_odom.copy()
                            observed=observed_clearance(frame)
                            if observed is not None:pending.append((t+controlled_delay,now,observed))
                            ready=[p for p in pending if p[0]<=t+1e-8]
                            pending=[p for p in pending if p[0]>t+1e-8]
                            if ready:
                                newest=max(ready,key=lambda p:p[1])
                                if latest_stamp is None or newest[1]>latest_stamp:latest_stamp=newest[1];latest=newest[2]
                            age=(now-latest_stamp)/1e6 if latest_stamp is not None else 1e9
                            start=time.perf_counter();result=model.predict(frame)
                            rpc_s=time.perf_counter()-start
                            obs=learned_person_observation(frame,result,high_score)
                            learned_pending.append((t+rpc_s+controlled_delay,obs))
                            available=[p for p in learned_pending if p[0]<=t+1e-8]
                            learned_pending=[p for p in learned_pending if p[0]>t+1e-8]
                            if available:
                                newest=max(available,key=lambda p:p[1]['timestamp_ns'])[1]
                                if learned_latest is None or newest['timestamp_ns']>learned_latest['timestamp_ns']:learned_latest=newest
                            geometric=controller.command(frame,odom_input,latest,now,age)
                            command,changed=constrain_command(geometric,frame,learned_latest,now) if active else(geometric,False)
                            fault=missing or age>300 or(active and(command.reason=='learned_constraint_stale_result'))
                            if fault and not fault_last:fault_started=t;awaiting_recovery=True
                            if fault and command.safety_stop and fault_started is not None:
                                stop_delays.append(t-fault_started);fault_started=None
                            if awaiting_recovery and not fault and command.linear_mps>0:recovered=True;awaiting_recovery=False
                            fault_last=fault
                            source.step(command)
                            truth=evaluator_state(source)
                            contacts+=bool(truth['collisions']);minclear=min(minclear,truth['minimum_clearance_m'])
                            reach=truth['true_goal_distance_m']<.35
                            trace.append({'time_s':t,'sample_id':frame.sample_id,'controller_input_odom_xyyaw':odom_input.tolist(),
                                'depth_available':not missing,'geometric_result_age_ms':age,'observed_clearance':latest,
                                'geometric_command':geometric.to_json(),'command':command.to_json(),'command_changed':changed,
                                'learned_inference_result':result,'learned_sensor_observation':obs,
                                'learned_observation_used':learned_latest,'actual_rpc_elapsed_ms':rpc_s*1000,
                                'causal_delay_scope':'observed wall-clock RPC elapsed shifted into simulation clock; not formal timing benchmark',
                                'evaluator':truth})
                            if reach:break
                    path=episode/'trace.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in trace))
                    rows.append({'condition':condition,'scenario':scenario,'seed':seed,'success':bool(reach and contacts==0),
                        'goal_reached':reach,'contact_control_steps':contacts,'elapsed_sim_s':float(source.data.time),
                        'minimum_clearance_m':minclear,'max_fault_stop_delay_s':max(stop_delays,default=None),
                        'recovery_observed':recovered,'final_goal_distance_m':truth['true_goal_distance_m'],
                        'eligible_learned_observations':sum(len(r['learned_sensor_observation']['eligible_tracks']) for r in trace),
                        'hazard_observations':sum(r['learned_sensor_observation']['hazard'] for r in trace),
                        'actual_command_changes':sum(r['command_changed'] for r in trace),
                        'hazard_command_changes':sum(r['command_changed'] and r['command']['reason']=='learned_person_safety_stop' for r in trace),
                        'stale_command_changes':sum(r['command_changed'] and r['command']['reason']=='learned_constraint_stale_result' for r in trace),
                        'trace':str(path),'metadata':source.metadata(),'inference_process_pid':model.process.pid})
                    print(json.dumps({k:rows[-1][k] for k in ('condition','scenario','seed','success','actual_command_changes')}),flush=True)
                finally:source.close()
    geometric={(r['scenario'],r['seed']):r for r in rows if r['condition']==frozen['conditions'][0]}
    paired=[]
    for row in rows:
        if row['condition']!=frozen['conditions'][1]:continue
        before=geometric[row['scenario'],row['seed']]
        paired.append({'scenario':row['scenario'],'seed':row['seed'],
            'geometric_success':before['success'],'active_success':row['success'],
            'completion_time_difference_s':row['elapsed_sim_s']-before['elapsed_sim_s'],
            'minimum_clearance_difference_m':row['minimum_clearance_m']-before['minimum_clearance_m'],
            'active_command_changes':row['actual_command_changes'],'hazard_command_changes':row['hazard_command_changes'],
            'active_contact_steps':row['contact_control_steps'],'geometric_contact_steps':before['contact_control_steps']})
    summary={'provenance':finish(provenance),'status':'executed','protocol':protocol,'episodes':len(rows),
        'rows':rows,'paired':paired,'contribution_scope':'Measured additive constraint effects; zero eligible/hazard/command-change outcomes remain valid results, not evidence of improvement',
        'physical_robot_validation':False,'primitive_actor_OOD':True,'GT_to_controller':False}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
