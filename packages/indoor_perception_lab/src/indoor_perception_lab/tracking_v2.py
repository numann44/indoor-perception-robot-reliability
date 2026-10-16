"""Causal high/low confidence association with gates applied before assignment."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracking import Track,measurement,bbox_iou

TRACKING_PRESETS={
 'T1':{'kind':'single_gated','high_score':.3,'low_score':.3,'iou_threshold':.15},
 'T2':{'kind':'two_stage','high_score':.4,'low_score':.1,'iou_threshold':.15},
 'T3':{'kind':'two_stage','high_score':.3,'low_score':.05,'iou_threshold':.1},
}

class QualityTracker:
    def __init__(self,high_score=.4,low_score=.1,iou_threshold=.15,max_age_s=1.):
        if not 0<=low_score<=high_score<=1:raise ValueError('Score thresholds must be ordered')
        self.high_score=high_score;self.low_score=low_score;self.iou_threshold=iou_threshold;self.max_age_ns=int(max_age_s*1e9)
        self.tracks=[];self.next_id=1;self.timestamp_ns=None;self.rejections={'duplicate':0,'out_of_order':0}
    def update(self,detections,timestamp_ns):
        if self.timestamp_ns is not None and timestamp_ns<=self.timestamp_ns:
            self.rejections['duplicate' if timestamp_ns==self.timestamp_ns else 'out_of_order']+=1;return self.tracks
        dt=(timestamp_ns-self.timestamp_ns)/1e9 if self.timestamp_ns is not None else 0.;self.timestamp_ns=timestamp_ns
        self.tracks=[t for t in self.tracks if timestamp_ns-t.last_detection_ns<=self.max_age_ns]
        F=np.eye(8);F[:4,4:]=np.eye(4)*dt;H=np.concatenate([np.eye(4),np.zeros((4,4))],1)
        for t in self.tracks:t.state=F@t.state;t.covariance=F@t.covariance@F.T+np.eye(8)*max(dt,.001);t.observed=False
        ds=[d for d in detections if d.score>=self.low_score and np.isfinite(d.bbox).all() and d.bbox[2]>d.bbox[0] and d.bbox[3]>d.bbox[1]]
        high=[i for i,d in enumerate(ds) if d.score>=self.high_score];low=[i for i,d in enumerate(ds) if d.score<self.high_score]
        remaining=set(range(len(self.tracks)));matched=set()
        for candidates in (high,low):
            track_indices=sorted(remaining)
            if not track_indices or not candidates:continue
            cost=np.full((len(track_indices),len(candidates)),1e6)
            for a,i in enumerate(track_indices):
                for b,j in enumerate(candidates):
                    t=self.tracks[i];d=ds[j];iou=bbox_iou(t.bbox,d.bbox)
                    if t.class_id==d.class_id and iou>=self.iou_threshold:cost[a,b]=1-iou
            rows,cols=linear_sum_assignment(cost)
            for a,b in zip(rows,cols):
                if cost[a,b]>=1e6:continue
                i=track_indices[a];j=candidates[b];t=self.tracks[i];d=ds[j]
                R=np.eye(4)*2;K=t.covariance@H.T@np.linalg.inv(H@t.covariance@H.T+R);t.state+=K@(measurement(d.bbox)-H@t.state)
                I=np.eye(8)-K@H;t.covariance=I@t.covariance@I.T+K@R@K.T
                t.last_detection_ns=timestamp_ns;t.hits+=1;t.observed=True;t.score=d.score;remaining.remove(i);matched.add(j)
        for j in high:
            if j not in matched:
                d=ds[j];self.tracks.append(Track(f'track-{self.next_id:06d}',d.class_id,np.r_[measurement(d.bbox),np.zeros(4)],np.diag([4]*4+[100]*4),timestamp_ns,score=d.score));self.next_id+=1
        return self.tracks

def make_tracker(configuration='T2'):
    p=TRACKING_PRESETS[configuration]
    return QualityTracker(p['high_score'],p['low_score'],p['iou_threshold'])

def make_legacy_tracker():
    """Immutable v1 algorithm for common-detector historical comparison only."""
    from .tracking import Tracker
    return Tracker(max_age_s=1.,iou_threshold=.15)
