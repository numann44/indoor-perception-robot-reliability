import numpy as np
import pytest
from indoor_perception_lab.tracking_v2 import make_tracker,make_legacy_tracker
from robot_model_sdk import Detection
from indoor_perception_lab.tracking_v2 import QualityTracker

def d(x,score=.9,cls=1):return Detection([x,0,x+10,20],score,cls)

def test_low_score_can_continue_but_cannot_create_track():
    t=QualityTracker();assert not t.update([d(0,.2)],0)
    first=t.update([d(0)],100000000)[0].id
    tracks=t.update([d(.5,.2)],200000000);assert tracks[0].id==first and tracks[0].observed

def test_expiration_order_and_class_gating():
    t=QualityTracker();first=t.update([d(0)],0)[0].id
    assert t.update([],100000000)[0].observed is False
    assert t.update([d(0)],500000000)[0].id==first
    assert t.update([d(0)],1600000001)[0].id!=first
    prior=t.timestamp_ns;t.update([],100000000);assert t.timestamp_ns==prior and t.rejections['out_of_order']==1

def test_invalid_edges_cannot_steal_valid_assignment(monkeypatch):
    from indoor_perception_lab import tracking_v2
    t=QualityTracker(iou_threshold=.85);t.update([d(0),d(20)],0)
    # Only first edge is admissible. Ungated global assignment prefers two
    # inadmissible cross edges; masking before Hungarian retains track1.
    table={(0,0):.86,(0,20):.84,(20,0):.80,(20,20):.01}
    monkeypatch.setattr(tracking_v2,'bbox_iou',lambda a,b:table[(round(a[0]),round(b[0]))])
    before=t.tracks[0].id;tracks=t.update([d(0),d(20)],100000000)
    assert next(x for x in tracks if x.id==before).observed
    assert np.linalg.eigvalsh(tracks[0].covariance).min()>0

@pytest.mark.parametrize('configuration',['T1','T2','T3'])
def test_deployed_preset_keeps_edge_which_legacy_hungarian_loses(monkeypatch,configuration):
    from indoor_perception_lab import tracking_v2,tracking
    t=make_tracker(configuration);threshold=t.iou_threshold
    table={(0,0):threshold+.01,(0,20):threshold-.01,(20,0):threshold-.05,(20,20):.001}
    overlap=lambda a,b:table[(round(a[0]),round(b[0]))]
    monkeypatch.setattr(tracking_v2,'bbox_iou',overlap);monkeypatch.setattr(tracking,'bbox_iou',overlap)
    first=t.update([d(0),d(20)],0)[0].id
    assert next(x for x in t.update([d(0),d(20)],100000000) if x.id==first).observed
    old=make_legacy_tracker();old.iou_threshold=threshold;prior=old.update([d(0),d(20)],0)[0].id
    assert not next(x for x in old.update([d(0),d(20)],100000000) if x.id==prior).observed


@pytest.mark.parametrize('configuration',['T1','T2','T3'])
def test_causal_crossing_and_short_occlusion_keep_known_motion_ids(configuration):
    tracker=make_tracker(configuration);initial=tracker.update([d(10),d(51)],0)
    left_id,right_id=[track.id for track in initial]
    for i in range(1,23):
        # Two oppositely moving people cross with nonidentical boxes. Detection
        # order alternates, so the test cannot succeed by trusting input order.
        detections=[d(10+2*i),d(51-2*i)]
        if 13<=i<=15:detections=detections[1:]  # 0.4s return gap, below 1s TTL.
        elif i%2:detections.reverse()
        tracks=tracker.update(detections,i*100000000)
        assert {track.id for track in tracks}=={left_id,right_id}
        if 13<=i<=15:assert not next(track for track in tracks if track.id==left_id).observed
    positions={track.id:track.bbox[0] for track in tracks}
    assert abs(positions[left_id]-54)<1
    assert abs(positions[right_id]-7)<1
