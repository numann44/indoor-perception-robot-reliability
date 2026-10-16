import numpy as np
import pytest
from robot_model_sdk import Calibration, SensorFrameV1, ControlCommandV1
from robot_reliability_bench.robotics_v2.learned_control import learned_person_observation, constrain_command
from robot_reliability_bench.robotics_v2.odometry_evaluation import first_pose_errors
from robot_reliability_bench.robotics_v2.matched_control import bind_tracker_lock, run_matched_control


def sample(depth=.7):
    return SensorFrameV1('sensor', 100_000_000, np.zeros((10,10,3),dtype=np.uint8),
                         np.full((10,10),depth,dtype=np.float32),Calibration(10,10,4.5,4.5,10,10))


def result(frame,score=.4,observed=True,box=(3,3,6,6)):
    return {'sample_id':frame.sample_id,'timestamp_ns':frame.timestamp_ns,
            'result':{'tracks':[{'id':'person_1_suffix_2','score':score,'class_id':1,'observed':observed,'bbox':box}]}}


def command(frame):
    return ControlCommandV1(frame.timestamp_ns,.2,.1,'moving','geometric',source_sample_id=frame.sample_id)


def test_active_person_rule_preserves_suffix_and_stops_both_axes():
    frame=sample();obs=learned_person_observation(frame,result(frame),.4)
    assert obs['eligible_tracks'][0]['track_id']=='person_1_suffix_2'
    stopped,changed=constrain_command(command(frame),frame,obs,frame.timestamp_ns)
    assert changed and stopped.safety_stop and stopped.linear_mps==stopped.angular_rps==0


@pytest.mark.parametrize('score,observed,box',[(.39,True,(3,3,6,6)),(.4,False,(3,3,6,6)),(.4,True,(3,3,5,5))])
def test_low_predicted_or_insufficient_depth_not_eligible(score,observed,box):
    frame=sample();obs=learned_person_observation(frame,result(frame,score,observed,box),.4)
    assert not obs['hazard'] and not obs['eligible_tracks']
    geometric=command(frame);out,changed=constrain_command(geometric,frame,obs,frame.timestamp_ns)
    assert out is geometric and not changed


def test_stale_and_missing_depth_stop_never_relaxes_geometric_stop():
    frame=sample(2.);obs=learned_person_observation(frame,result(frame),.3)
    stopped,_=constrain_command(command(frame),frame,obs,frame.timestamp_ns+300_000_001)
    assert stopped.safety_stop
    frame.depth=None
    stopped,_=constrain_command(command(frame),frame,obs,frame.timestamp_ns)
    assert stopped.safety_stop


def test_mismatched_frame_rejected_before_depth_access():
    frame=sample();r=result(frame);r['sample_id']='different'
    with pytest.raises(ValueError,match='same sensor frame'):learned_person_observation(frame,r,.4)


def test_first_pose_alignment_preserves_scale_drift():
    truth=np.repeat(np.eye(4)[None],3,axis=0);truth[:,0,3]=[0,1,2]
    recorded=truth.copy();recorded[:,0,3]=[10,11.1,12.2]
    alignment,aligned,error,rot=first_pose_errors(recorded,truth)
    assert alignment[0,3]==-10
    assert np.allclose(error,[0,.1,.2]) and np.allclose(rot,0)


def test_canonical_root_tracker_lock_is_required(tmp_path):
    import json
    p=tmp_path/'lock.json'
    p.write_text(json.dumps({'selected':{'configuration':'T3','parameters':{'kind':'two_stage','high_score':.3,'low_score':.05,'iou_threshold':.1}},'detector_weights_sha256':'weights','detector_manifest_sha256':'manifest'}))
    bound=bind_tracker_lock(p,'T3',.3)
    assert bound['tracking']['low_score']==.05 and bound['detector_manifest_sha256']=='manifest'
    with pytest.raises(ValueError):bind_tracker_lock(p,'T3',.4)
    with pytest.raises(ValueError):bind_tracker_lock(p,'T2',.3)


def test_active_run_cannot_start_during_training(tmp_path):
    with pytest.raises(ValueError,match='quiescent'):
        run_matched_control(tmp_path,tmp_path/'out','missing','T3',.3,'missing')
