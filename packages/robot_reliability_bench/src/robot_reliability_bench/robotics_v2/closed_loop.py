"""Live simulated actuation and sensor-only policy RPC behind actual DDS.

Physics truth is written solely to an evaluator file. It is never returned by
sensor/control IPC and never enters the policy, inference worker or DDS topics.
"""
from __future__ import annotations
import hashlib,json,os,socket,time
from pathlib import Path
import numpy as np
from robot_model_sdk import ControlCommandV1
from .ipc import frame_wire,frame_read,receive,send
from .controller import ObservedController
from .simulator import MuJoCoFrameSource,evaluator_state

COMMAND_KEYS={'schema_version','timestamp_ns','linear_mps','angular_rps','state','reason','source_sample_id','observed_result_age_ms','safety_stop'}


def command_read(value):
    if set(value)!=COMMAND_KEYS or value['schema_version']!=1:raise ValueError('Exact ControlCommandV1 fields required')
    command=ControlCommandV1(**{k:v for k,v in value.items() if k!='schema_version'})
    if abs(command.linear_mps)>.3 or abs(command.angular_rps)>.6:raise ValueError('Frozen velocity limits exceeded')
    return command


def zero(now,reason,sample_id=None):return ControlCommandV1(int(now),0.,0.,'stopped',reason,sample_id,safety_stop=True)


def nominal_camera_mount(source):
    """Known mount expressed in the planar encoder frame, not a truth pose."""
    mj=source.mujoco;model=source.model
    camera=mj.mj_name2id(model,mj.mjtObj.mjOBJ_CAMERA,'rgbd')
    rotation=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    expected=np.array([.20,0.,.16])
    if not np.allclose(model.cam_pos[camera],expected,atol=1e-12):raise ValueError('Frozen known camera mount differs')
    # The encoder frame lies on the ground plane; the XML base origin is .14m.
    if not np.allclose(model.body_pos[source.robot_body],[0.,0.,.14],atol=1e-12):raise ValueError('Frozen encoder/base reference differs')
    T=np.eye(4);T[:3,:3]=rotation;T[:3,3]=expected+[0.,0.,.14]
    return T


def admit_command(envelope,now_ns,depth_available=True,command_timeout_ms=250.):
    """Independent actuator watchdog: no positive velocity is synthesized."""
    if not depth_available:return zero(now_ns,'actuator_missing_depth'),False
    if envelope is None:return zero(now_ns,'actuator_no_command'),False
    try:command=command_read(envelope['command'])
    except (ValueError,TypeError,KeyError):return zero(now_ns,'actuator_invalid_command'),False
    age=(now_ns-command.timestamp_ns)/1e6
    if age<0:return zero(now_ns,'actuator_future_command'),False
    if age>command_timeout_ms:return zero(now_ns,'actuator_command_timeout'),False
    if command.observed_result_age_ms is not None and command.observed_result_age_ms+age>300:
        return zero(now_ns,'actuator_stale_perception'),False
    return command,True


def sensor_policy(controller,request,active_high_score=None):
    allowed={'op','frame','result','wheel_odometry','current_sensor_status','now_ns'}
    if set(request)!=allowed:raise ValueError('Only observed sensor/result fields accepted by policy')
    frame=frame_read(request['frame']);now=int(request['now_ns']);result=request['result']
    if result['sample_id']!=frame.sample_id or result['timestamp_ns']!=frame.timestamp_ns:raise ValueError('Policy frame/result identity mismatch')
    age=(now-frame.timestamp_ns)/1e6;status=request['current_sensor_status']
    if status is not None and status['timestamp_ns']>=frame.timestamp_ns and not status['depth_available']:
        return zero(now,'missing_depth',status['sample_id'])
    if age<0:return zero(now,'future_result',frame.sample_id)
    odom=np.asarray(request['wheel_odometry'],float)
    if odom.shape!=(3,) or not np.isfinite(odom).all():return zero(now,'missing_wheel_odometry',frame.sample_id)
    command=controller.command(frame,odom,result.get('observed_geometry'),now,age)
    if active_high_score is not None:
        from .learned_control import learned_person_observation,constrain_command
        observation=learned_person_observation(frame,result,active_high_score)
        command,_=constrain_command(command,frame,observation,now)
    return command


def serve_physics(path,output,scenario='clear',seed=42,faults=False,active_high_score=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);source=MuJoCoFrameSource(scenario,seed)
    controller=ObservedController();source.current_odom=source.odom.update(source.data.qpos[source.wheel_qpos])
    log=(output/'physics-evaluator.jsonl').open('w');latest_depth_available=True;steps=0;commands_applied=0;contact_steps=0;minclear=float('inf');maximum_goal_progress=0.
    mount=nominal_camera_mount(source);metadata={**source.metadata(),'T_encoder_base_camera':mount.tolist(),
        'camera_mount_source':'Unchanged XML local camera .20/0/.16 plus .14m nominal planar base height; encoder frame ground Z-up'}
    # Startup evaluator only: compare known calibration to actual rendered-camera
    # geometry before motion. Neither this truth matrix nor its points enter IPC.
    cam=source.mujoco.mj_name2id(source.model,source.mujoco.mjtObj.mjOBJ_CAMERA,'rgbd')
    actual=np.eye(4);actual[:3,:3]=source.data.cam_xmat[cam].reshape(3,3)@np.diag([1.,-1.,-1.]);actual[:3,3]=source.data.cam_xpos[cam]
    points=np.array([[0.,0.,.5,1.],[.2,.1,1.,1.],[-.4,.2,3.,1.],[.5,-.3,5.,1.]])
    projected=(np.linalg.inv(mount)@actual@points.T).T
    c=source.calibration
    pixels=lambda q:np.column_stack((c.fx*q[:,0]/q[:,2]+c.cx,c.fy*q[:,1]/q[:,2]+c.cy))
    proof={'status':'verified' if np.allclose(actual,mount,atol=1e-10) else 'failed','known_mount':mount.tolist(),
        'actual_initial_camera_evaluator_only':actual.tolist(),'maximum_coordinate_error_m':float(np.max(np.abs(projected-points))),
        'maximum_projection_error_pixels':float(np.max(np.abs(pixels(projected)-pixels(points)))),
        'point_depths_m':points[:,2].tolist(),'scope':'Initial pose/mount/pixel projection; nominal planar encoder approximation during motion, no truth correction',
        'rendered_depth_acceptance_ref':'reports/v2/robotics/platform-warnings/macgl-depth-precision.json'}
    (output/'known-mount-acceptance.json').write_text(json.dumps(proof,indent=2)+'\n')
    if proof['status']!='verified':source.close();raise ValueError('Known sensor mount geometry does not match frozen simulator')
    socketpath=Path(path);socketpath.parent.mkdir(parents=True,exist_ok=True)
    if socketpath.exists():socketpath.unlink()
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(str(socketpath));socketpath.chmod(0o600);server.listen(4);server.settimeout(1.)
            while True:
                try:peer,_=server.accept()
                except socket.timeout:continue
                with peer:
                    peer.settimeout(5.)
                    try:
                        request=receive(peer);op=request.get('op')
                        if op=='policy':
                            command=sensor_policy(controller,request,active_high_score)
                            send(peer,{'status':'ok','command':command.to_json()});continue
                        if op=='finish':
                            if set(request)!={'op'}:raise ValueError('Unexpected finish fields')
                            send(peer,{'status':'ok'});break
                        if set(request)!={'op','command_envelope','now_ns'} or op!='sensor_tick':raise ValueError('Exact sensor-tick protocol required')
                        now=int(request['now_ns'])
                        if steps:
                            command,accepted=admit_command(request['command_envelope'],now,latest_depth_available)
                            source.step(command,dt=.1);commands_applied+=accepted
                            truth=evaluator_state(source);contact_steps+=bool(truth['collisions']);minclear=min(minclear,truth['minimum_clearance_m'])
                            maximum_goal_progress=max(maximum_goal_progress,4-truth['true_goal_distance_m'])
                            log.write(json.dumps({'event':'physics_step','monotonic_ns':time.monotonic_ns(),'simulation_time_s':float(source.data.time),
                                'requested_DDS_command':request['command_envelope'],'command_admitted':accepted,'applied_command':command.to_json(),
                                'evaluator':truth})+'\n');log.flush()
                        t=float(source.data.time);missing=bool(faults and 3<=t<5)
                        capture_begin_ns=time.time_ns();frame=source.capture(depth_missing=missing);capture_end_ns=time.time_ns()
                        latest_depth_available=frame.depth is not None;steps+=1
                        # Encoder estimate only; private physics truth never leaves this process.
                        send(peer,{'status':'ok','frame':frame_wire(frame),'wheel_odometry':source.current_odom.tolist(),
                                   'simulation_timestamp_ns':frame.timestamp_ns,'sensor_fault':'controlled_depth_loss' if missing else None,
                                   'capture_begin_ns':capture_begin_ns,'capture_end_ns':capture_end_ns,'T_encoder_base_camera':mount.tolist()})
                    except Exception as exc:
                        send(peer,{'status':'error','error':str(exc)})
    finally:
        truth=evaluator_state(source)
        summary={'sensor_ticks':steps,'admitted_DDS_command_steps':commands_applied,'every5ms_contact_control_steps':contact_steps,
                 'minimum_clearance_m':minclear if np.isfinite(minclear) else None,'maximum_goal_progress_m':maximum_goal_progress,
                 'final_evaluator':truth,'goal_reached':truth['true_goal_distance_m']<.35,'goal_success':truth['true_goal_distance_m']<.35 and contact_steps==0,
                 'metadata':metadata,'contact_sampling':'Existing unchanged MuJoCoFrameSource.step captures every5ms physics step',
                 'controller_sha256':hashlib.sha256(Path(__file__).with_name('controller.py').read_bytes()).hexdigest(),
                 'simulator_sha256':hashlib.sha256(Path(__file__).with_name('simulator.py').read_bytes()).hexdigest(),
                 'physical_actuator_connected':False,'truth_to_inference_or_policy':False,'sensor_noise':'Existing frozen encoder/depth model'}
        (output/'physics-summary.json').write_text(json.dumps(summary,indent=2)+'\n');log.close();source.close();socketpath.unlink(missing_ok=True)
