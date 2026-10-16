from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from scipy.optimize import linear_sum_assignment

def bbox_iou(a,b):
    x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]);inter=max(0,x1-x0)*max(0,y1-y0)
    area=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/area if area>0 else 0.0

def measurement(b):return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2,max(1,b[2]-b[0]),max(1,b[3]-b[1])],float)
@dataclass
class Track:
    id:str
    class_id:int
    state:np.ndarray
    covariance:np.ndarray
    last_detection_ns:int
    hits:int=1
    observed:bool=True
    score:float=1.0
    history:list=field(default_factory=list)
    @property
    def bbox(self):
        cx,cy,w,h=self.state[:4];w=max(1,w);h=max(1,h);return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

class Tracker:
    def __init__(self,max_age_s=1.0,iou_threshold=.15):self.tracks=[];self.next_id=1;self.timestamp_ns=None;self.max_age_ns=int(max_age_s*1e9);self.iou_threshold=iou_threshold;self.rejections={"duplicate":0,"out_of_order":0}
    def update(self,detections,timestamp_ns):
        if self.timestamp_ns is not None and timestamp_ns<=self.timestamp_ns:
            self.rejections["duplicate" if timestamp_ns==self.timestamp_ns else "out_of_order"]+=1;return self.tracks
        dt=(timestamp_ns-self.timestamp_ns)/1e9 if self.timestamp_ns is not None else 0;self.timestamp_ns=timestamp_ns
        self.tracks=[t for t in self.tracks if timestamp_ns-t.last_detection_ns<=self.max_age_ns]
        F=np.eye(8);F[:4,4:]=np.eye(4)*dt;H=np.concatenate([np.eye(4),np.zeros((4,4))],axis=1)
        for t in self.tracks:t.state=F@t.state;t.covariance=F@t.covariance@F.T+np.eye(8)*max(dt,.001);t.observed=False
        ds=[d for d in detections if np.isfinite(d.bbox).all() and d.bbox[2]>d.bbox[0] and d.bbox[3]>d.bbox[1]]
        matched=set()
        if self.tracks and ds:
            cost=np.array([[1-bbox_iou(t.bbox,d.bbox) if t.class_id==d.class_id else 1e6 for d in ds] for t in self.tracks]);rows,cols=linear_sum_assignment(cost)
            for row,col in zip(rows,cols):
                if cost[row,col]>1-self.iou_threshold:continue
                t=self.tracks[row];d=ds[col];z=measurement(d.bbox);S=H@t.covariance@H.T+np.eye(4)*2;K=t.covariance@H.T@np.linalg.inv(S);t.state+=K@(z-H@t.state);t.covariance=(np.eye(8)-K@H)@t.covariance;t.last_detection_ns=timestamp_ns;t.hits+=1;t.observed=True;t.score=d.score;matched.add(col)
        for j,d in enumerate(ds):
            if j not in matched:
                t=Track(f"track-{self.next_id:06d}",d.class_id,np.r_[measurement(d.bbox),np.zeros(4)],np.diag([4]*4+[100]*4),timestamp_ns,score=d.score);self.next_id+=1;self.tracks.append(t)
        return self.tracks
