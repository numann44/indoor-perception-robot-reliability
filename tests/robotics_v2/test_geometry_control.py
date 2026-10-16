from types import SimpleNamespace
import numpy as np
import pytest
from robot_model_sdk import Calibration,SensorFrameV1
from robot_reliability_bench.robotics_v2.geometry import nearest_one_to_one,interpolate_pose,register_depth,resolve_static_transform,rigid_transform
from robot_reliability_bench.robotics_v2.controller import ObservedController,WheelOdometry
from robot_reliability_bench.robotics_v2.openloris import decode_image,choose_topics
from robot_reliability_bench.robotics_v2.ipc import frame_read,frame_wire,array_read


def frame():return SensorFrameV1('sensor-only',1,np.zeros((4,4,3),dtype=np.uint8),np.full((4,4),2.,dtype=np.float32),Calibration(2,2,1.5,1.5,4,4))

def test_registration_zbuffer_and_extrinsic():
    K=np.array([[2.,0,1],[0,2.,1],[0,0,1]])
    depth=np.full((3,3),2.,dtype=np.float32);T=np.eye(4);T[0,3]=1.
    result=register_depth(depth,K,K,T,(3,3))
    assert np.all(result[:,0]==0) and np.all(result[:,1:]==2)

def test_timestamp_pairing_is_unique_and_includes_boundary():
    assert nearest_one_to_one([0,20_000_000,60_000_001],[20_000_000,40_000_000])==[0,1,None]
    with pytest.raises(ValueError):nearest_one_to_one([2,1],[1])

def test_pose_interpolation_refuses_long_gaps():
    a=np.eye(4);b=rigid_transform([1,0,0],[0,0,np.sin(np.pi/4),np.cos(np.pi/4)])
    mid=interpolate_pose([0,100_000_000],[a,b],50_000_000)
    assert np.allclose(mid[:3,3],[.5,0,0])
    assert np.allclose(mid[:3,:3]@np.array([1,0,0]),[np.sqrt(.5),np.sqrt(.5),0])
    assert interpolate_pose([0,100_000_001],[a,b],50_000_000) is None

def test_extrinsic_chain_and_inverse():
    a=np.eye(4);a[0,3]=1;b=np.eye(4);b[1,3]=2
    assert np.allclose(resolve_static_transform([('base','camera',a),('camera','depth',b)],'depth','base'),np.linalg.inv(a@b))

def test_control_fail_closed_missing_depth_and_old_result():
    f=frame();p={'valid_fraction':1.,'forward_m':4.,'left_m':4.,'right_m':4.}
    c=ObservedController()
    assert c.command(f,[0,0,0],p,1,301).safety_stop
    f.depth=None;cmd=c.command(f,[0,0,0],p,1,0)
    assert cmd.safety_stop and cmd.linear_mps==cmd.angular_rps==0 and cmd.reason=='missing_depth'

def test_control_is_not_always_stopped_and_limits():
    f=frame();p={'valid_fraction':1.,'forward_m':4.,'left_m':4.,'right_m':4.}
    cmd=ObservedController().command(f,[0,0,0],p,1,0)
    assert 0<cmd.linear_mps<=.3 and abs(cmd.angular_rps)<=.6
    with pytest.raises(ValueError):ObservedController(stop_distance_m=.79)
    p['forward_m']=.5
    cmd=ObservedController().command(f,[0,0,0],p,1,0)
    assert cmd.linear_mps==0

def test_wheel_odometry_uses_encoders_only():
    o=WheelOdometry();o.update([0,0]);value=o.update([1,1])
    assert np.allclose(value,[.075,0,0])

def test_padded_bigendian_depth_decode():
    a=np.array([[1,256],[1024,9]],dtype='>u2')
    rows=np.concatenate((a.view('u1').reshape(2,4),np.zeros((2,2),dtype=np.uint8)),axis=1)
    m=SimpleNamespace(encoding='16UC1',height=2,width=2,step=6,is_bigendian=True,data=rows.reshape(-1))
    assert np.array_equal(decode_image(m),a)
    m.step=3
    with pytest.raises(ValueError):decode_image(m)

def test_ipc_sensor_whitelist_forbids_truth_and_targets():
    f=frame();wire=frame_wire(f);reconstructed=frame_read(wire)
    assert np.array_equal(reconstructed.depth,f.depth)
    wire['ground_truth_boxes']=[]
    with pytest.raises(ValueError):frame_read(wire)

def test_topic_selection_excludes_gt_pose():
    connections=[SimpleNamespace(topic='/ground_truth/odom',msgtype='nav_msgs/msg/Odometry'),SimpleNamespace(topic='/odom',msgtype='nav_msgs/msg/Odometry')]
    selected=choose_topics(connections)
    assert selected['odometry']=='/odom' and selected['excluded_ground_truth_topics']==['/ground_truth/odom']


def test_mujoco_actual_framesource_lifecycle_and_observed_pose_contract():
    from robot_reliability_bench.robotics_v2.simulator import MuJoCoFrameSource
    source=MuJoCoFrameSource('clear',42,sensor_noise=False)
    try:
        stream=iter(source);a=next(stream);b=next(stream)
        assert isinstance(a,SensorFrameV1) and b.timestamp_ns>a.timestamp_ns
        assert a.depth.shape==a.rgb.shape[:2] and a.calibration.depth_convention=='z'
        assert a.pose_source=='wheel_encoder_odometry'
        assert source.metadata()['source']=='MuJoCo controlled simulation'
        assert 'actor_motion' not in source.metadata()
    finally:source.close()
    assert list(source)==[]
    source.close()


def test_actual_openloris_source_is_sensor_only_and_close_prevents_IO():
    from pathlib import Path
    from robot_reliability_bench.robotics_v2.openloris import OpenLorisFrameSource
    root=Path(__file__).resolve().parents[2];bag=root/'data/v2/openloris/office1-1.bag'
    if not bag.exists():pytest.skip('Actual local bag not present; integration acceptance runs it when fetched')
    source=OpenLorisFrameSource(bag,maximum_frames=2)
    try:
        frames=list(source)
        assert len(frames)==2 and all(isinstance(f,SensorFrameV1) for f in frames)
        assert all(f.depth is not None and f.calibration.depth_scale==1. for f in frames)
        assert source.metadata()['ground_truth_input'] is False
        assert source.metadata()['selection']['excluded_ground_truth_topics']==['/gt']
    finally:source.close()
    assert list(source)==[]
