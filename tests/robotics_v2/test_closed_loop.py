import numpy as np
import pytest
from robot_model_sdk import Calibration,SensorFrameV1,ControlCommandV1
from robot_reliability_bench.robotics_v2.closed_loop import admit_command,command_read,sensor_policy,nominal_camera_mount
from robot_reliability_bench.robotics_v2.controller import ObservedController
from robot_reliability_bench.robotics_v2.ipc import frame_wire


def moving(stamp=1_000_000_000,age=50.):
    return {'command':ControlCommandV1(stamp,.3,0.,'moving','goal_tracking','observed',age).to_json(),'consumer_sequence':1,'consumer_instance':'test'}


def test_actuator_applies_exact_received_command_and_never_synthesizes_motion():
    envelope=moving();command,accepted=admit_command(envelope,1_100_000_000)
    assert accepted and command.to_json()==envelope['command']
    command,accepted=admit_command(None,1_100_000_000)
    assert not accepted and command.safety_stop and command.linear_mps==command.angular_rps==0


@pytest.mark.parametrize('now,depth,age,reason',[(1_300_000_001,True,50.,'actuator_command_timeout'),(900_000_000,True,50.,'actuator_future_command'),(1_100_000_000,False,50.,'actuator_missing_depth'),(1_100_000_000,True,201.,'actuator_stale_perception')])
def test_actuator_depth_freshness_and_process_disconnect_watchdogs(now,depth,age,reason):
    command,accepted=admit_command(moving(age=age),now,depth)
    assert not accepted and command.reason==reason and command.linear_mps==command.angular_rps==0


def test_out_of_limits_or_unknown_command_fields_fail_closed():
    envelope=moving();envelope['command']['linear_mps']=.31
    command,accepted=admit_command(envelope,1_100_000_000)
    assert not accepted and command.reason=='actuator_invalid_command'
    envelope=moving();envelope['command']['future_actor_position']=[1,2]
    with pytest.raises(ValueError):command_read(envelope['command'])


def request():
    f=SensorFrameV1('observed',1_000_000_000,np.zeros((4,4,3),np.uint8),np.full((4,4),2.,np.float32),Calibration(2,2,1.5,1.5,4,4))
    return {'op':'policy','frame':frame_wire(f),'result':{'sample_id':f.sample_id,'timestamp_ns':f.timestamp_ns,'observed_geometry':{'forward_m':4.,'left_m':4.,'right_m':4.,'valid_fraction':1.,'ground_validated':True}},'wheel_odometry':[0,0,0],'current_sensor_status':None,'now_ns':1_100_000_000}


def test_policy_sensor_only_boundary_and_latest_depth_fault():
    value=request();command=sensor_policy(ObservedController(),value)
    assert command.linear_mps>0
    value['current_sensor_status']={'timestamp_ns':1_100_000_000,'sample_id':'new-depth-loss','depth_available':False}
    command=sensor_policy(ObservedController(),value)
    assert command.reason=='missing_depth' and command.safety_stop
    value['actor_ground_truth']=[2,0]
    with pytest.raises(ValueError):sensor_policy(ObservedController(),value)


def test_policy_age_keeps_capture_clock_and_identity():
    value=request();value['now_ns']=1_300_000_001
    command=sensor_policy(ObservedController(),value)
    assert command.reason=='stale_perception' and command.safety_stop
    value['result']['sample_id']='different'
    with pytest.raises(ValueError):sensor_policy(ObservedController(),value)


def test_declared_extrinsics_match_actual_frozen_initial_renderer_camera():
    from robot_reliability_bench.robotics_v2.simulator import MuJoCoFrameSource
    source=MuJoCoFrameSource('clear',42,sensor_noise=False)
    try:
        declared=nominal_camera_mount(source);f=source.capture()
        cam=source.mujoco.mj_name2id(source.model,source.mujoco.mjtObj.mjOBJ_CAMERA,'rgbd')
        actual=np.eye(4);actual[:3,:3]=source.data.cam_xmat[cam].reshape(3,3)@np.diag([1.,-1.,-1.]);actual[:3,3]=source.data.cam_xpos[cam]
        assert np.allclose(declared,f.T_world_camera,atol=1e-12) and np.allclose(actual,declared,atol=1e-10)
    finally:source.close()
