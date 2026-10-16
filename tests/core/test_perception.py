import json
from pathlib import Path
import numpy as np
import pytest
from robot_model_sdk import Calibration,SensorFrameV1,Detection,fixture_frames,fixture_frame,load_adapter
from indoor_perception_lab.geometry import depth_to_z,project_depth,transform_points,robust_object_position,SpatialMap,geometric_plane_mask
from indoor_perception_lab.synchronization import match_timestamps,interpolate_pose
from indoor_perception_lab.tracking import Tracker
from indoor_perception_lab.pipeline import PerceptionPipeline
from indoor_perception_lab.prediction import baseline_predictions,operational_behavior
from indoor_perception_lab.datasets import read_index,sensor_from_row


def test_depth_scale_and_pinhole_center():
    c=Calibration(2,2,1,1,3,3,5000);depth=np.full((3,3),5000,np.uint16);p,vu=project_depth(depth,c);np.testing.assert_allclose(p[(vu==[1,1]).all(1)][0],[0,0,1]);assert np.isnan(depth_to_z(np.zeros((3,3)),c)).all()

def test_radial_not_planar_depth():
    c=Calibration(1,1,0,0,2,2,1,'radial');z=depth_to_z(np.full((2,2),np.sqrt(3)),c);assert z[1,1]==pytest.approx(1);assert z[0,0]==pytest.approx(np.sqrt(3))

def test_transform_round_trip():
    T=np.eye(4);T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];T[:3,3]=[2,3,4];p=np.array([[1.,2,3],[3,4,5]]);np.testing.assert_allclose(transform_points(transform_points(p,T),np.linalg.inv(T)),p)

def test_timestamp_tolerance_unique_and_rejections():
    assert match_timestamps([0,30_000_000,90_000_000],[10_000_000,31_000_000],20_000_000)==[(0,0),(1,1),(2,None)]
    with pytest.raises(ValueError):match_timestamps([1,1],[1])

def test_pose_interpolation_and_missing_gap():
    a=np.eye(4);b=np.eye(4);b[0,3]=2;assert interpolate_pose(50_000_000,[(0,a),(100_000_000,b)])[0,3]==pytest.approx(1);assert interpolate_pose(150_000_000,[(0,a),(300_000_000,b)]) is None

def test_tracker_short_and_long_occlusion():
    tr=Tracker();d=Detection([10,10,30,30],.9,1);id0=tr.update([d],0)[0].id;assert tr.update([],400_000_000)[0].observed is False;assert tr.update([d],800_000_000)[0].id==id0;assert tr.update([],2_000_000_000)==[];assert tr.update([d],2_100_000_000)[0].id!=id0

def test_tracker_association_uses_space_not_input_order():
    tr=Tracker();a=Detection([0,0,20,20],1,1);b=Detection([100,0,120,20],1,1);tracks=tr.update([a,b],0);ids=[t.id for t in tracks];tracks=tr.update([b,a],100_000_000);assert [t.id for t in tracks]==ids;assert tracks[0].bbox[0]<50

def test_duplicate_late_timestamp_do_not_advance_tracker():
    tr=Tracker();d=Detection([0,0,10,10],1,1);tr.update([d],100);hits=tr.tracks[0].hits;tr.update([d],100);tr.update([d],99);assert tr.tracks[0].hits==hits;assert tr.rejections=={'duplicate':1,'out_of_order':1}

def test_empty_and_unregistered_sensor_rejected():
    with pytest.raises(ValueError):SensorFrameV1('bad',0,np.zeros((0,0,3),np.uint8))
    with pytest.raises(ValueError):SensorFrameV1('bad',0,np.zeros((3,3,3),np.uint8),np.ones((2,2)))
    with pytest.raises(TypeError):SensorFrameV1('oracle',0,np.zeros((3,3,3),np.uint8),ground_truth_boxes=[])

def test_fixture_prediction_reads_pixels_only():
    f=fixture_frame(0);adapter=load_adapter('fixture');assert len(adapter.predict(f).detections)==2;f.rgb[:]=0;assert adapter.predict(f).detections==[]

def test_pipeline_deterministic_without_timing_fields():
    def run():
        p=PerceptionPipeline();out=[]
        for f in fixture_frames(25):
            v=p.process(f);v.pop('latency_ms');v.pop('detector_timings_ms');out.append(v)
        return out
    assert run()==run()

def test_map_missing_pose_keeps_unknown_and_probability_fusion():
    f=fixture_frame(0);m=SpatialMap();f.T_world_camera=None;assert m.update(f)['updated'] is False;assert m.cells=={};f.T_world_camera=np.eye(4);semantic=np.zeros(f.rgb.shape[:2],np.uint8);probs=np.stack([np.full((12,16),.2),np.full((12,16),.8)]);m.update(f,semantic,probabilities=probs);row=m.to_json()['cells'][0];assert row['semantic_class']==1;assert row['semantic_probability_average'][1]==pytest.approx(.8);assert 'unobserved' in m.to_json()['representation']

def test_geometric_floor_known_fixture_is_not_semantics():
    f=fixture_frame(0);mask,info=geometric_plane_mask(f.depth,f.calibration);assert info['available'];assert np.sum(mask==1)>20;assert set(np.unique(mask))<={0,1,255}

def test_operational_behavior_persistence_and_deadband():
    ts=np.linspace(0,.5,6);stationary=[(t,[0,0]) for t in ts];moving=[(t,[.4*t,0]) for t in ts];dead=[(t,[.15*t,0]) for t in ts];assert operational_behavior(stationary)=='stationary';assert operational_behavior(moving)=='moving';assert operational_behavior(moving,[(t,5-.4*t) for t in ts])=='approaching';assert operational_behavior(dead)=='unknown'

def test_motion_baseline_units_and_horizon():
    out=baseline_predictions([(0,[1,2]),(1,[2,2])]);assert out['constant_velocity']['2']==[4,2];assert out['constant_position']['2']==[2,2]

def test_split_overlap_rejected(tmp_path):
    folder=tmp_path/'data/x';folder.mkdir(parents=True);(folder/'index.jsonl').write_text('\n'.join(json.dumps({'scene_id':'same','split':s}) for s in ['train','test']));
    with pytest.raises(ValueError):read_index(tmp_path,'x')

def test_target_paths_never_opened_by_sensor_loader(tmp_path):
    import cv2
    image=tmp_path/'rgb.png';cv2.imwrite(str(image),np.zeros((8,8,3),np.uint8));row={'rgb_path':'rgb.png','sample_id':'safe','timestamp':0,'targets_path':'DOES_NOT_EXIST.json','mask_path':'DOES_NOT_EXIST.png'};frame=sensor_from_row(tmp_path,row);assert frame.rgb.shape==(8,8,3)

def test_json_contracts_no_oracle_keys():
    import jsonschema
    import robot_model_sdk
    folder=Path(robot_model_sdk.__file__).parent/'schemas';frame=fixture_frame(0);jsonschema.validate(frame.to_json(),json.loads((folder/'sensor-frame-v1.json').read_text()));out=PerceptionPipeline().process(frame);jsonschema.validate(out,json.loads((folder/'perception-result-v1.json').read_text()));assert not any('ground_truth' in k for k in frame.to_json())


def test_occluded_world_track_preserves_units_and_forecast_reference():
    p=PerceptionPipeline();p.process(fixture_frame(0));p.process(fixture_frame(1));f=fixture_frame(5);f.rgb[:]=0;row=p.process(f)['tracks'][0];assert not row['observed'];assert row['unit']=='metres';assert row['coordinate_frame']=='world';assert row['behavior']=='unknown'
    out=baseline_predictions([(0,[0,0]),(.1,[.1,0])],reference_time=.5);assert out['constant_velocity']['0.5'][0]==pytest.approx(1.0)

def test_crossing_tracks_use_predicted_motion():
    tr=Tracker();initial=None
    for i in range(21):
        a=Detection([20+2*i,0,40+2*i,20],1,1);b=Detection([60-2*i,10,80-2*i,30],1,1)
        tracks=tr.update([a,b] if i%2 else [b,a],i*100_000_000)
        if i==0:initial={round(t.bbox[1]):t.id for t in tracks}
    tracks_by_id={t.id:t for t in tracks}
    assert tracks_by_id[initial[0]].bbox[0]>tracks_by_id[initial[10]].bbox[0]

def test_dynamic_regions_do_not_enter_surface_map():
    f=fixture_frame(0);m=SpatialMap();m.update(f,dynamic_boxes=[[0,0,160,120]])
    assert m.cells=={};assert 'unknown' in m.to_json()['representation']

def test_mapping_explicitly_names_supplied_ground_truth_pose():
    f=fixture_frame(0);f.pose_source='provided_ground_truth';m=SpatialMap()
    assert m.update(f)['mapping_mode']=='ground-truth-pose-assisted mapping'
    assert m.to_json()['mapping_mode']=='ground-truth-pose-assisted mapping'

def test_unsupported_detector_unfreeze_cannot_be_silently_recorded():
    from indoor_perception_lab.training import train
    with pytest.raises(ValueError,match='semantic-only'):train('detection',unfreeze_last=True)
