"""Causal coordinate history contract, independent of training/evaluation labels."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TrajectoryHistoryV1:
    times_s: np.ndarray
    positions: np.ndarray
    coordinate_system: str
    units: str
    def __post_init__(self):
        t=np.asarray(self.times_s);p=np.asarray(self.positions)
        if t.ndim!=1 or p.shape!=(len(t),2) or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(np.diff(t)<=0):raise ValueError('Finite increasing times and aligned Nx2 positions required')
        if self.units not in {'metres','normalized_image'}:raise ValueError('Explicit supported history units required')

def resample_causal(times,positions,anchor_time=None,max_gap=.21):
    t=np.asarray(times,float);p=np.asarray(positions,float)
    if t.ndim!=1 or p.shape!=(len(t),2) or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(np.diff(t)<=0):raise ValueError('Invalid trajectory history')
    anchor=float(t[-1]) if anchor_time is None else float(anchor_time)
    if t[-1]>anchor+1e-8:raise ValueError('Future observation cannot enter inference history')
    grid=anchor+np.linspace(-1,0,11)
    if t[0]>grid[0]+1e-8 or t[-1]<anchor-1e-8:return None
    relevant=t[(t>=grid[0]-max_gap)&(t<=anchor)]
    if np.any(np.diff(relevant)>max_gap+1e-8):return None
    return np.stack([np.interp(grid,t,p[:,i]) for i in range(2)],1).astype(np.float32)

def prepare_motion(history,metadata):
    h=np.asarray(history,np.float32);single=h.ndim==2
    if single:h=h[None]
    if h.ndim!=3 or h.shape[1:]!=(11,2) or not np.isfinite(h).all():raise ValueError('History must be finite Nx11x2')
    settings=metadata.get('preprocess',{});scale=np.asarray(settings.get('scale',[1,1]),np.float32);origin=h[:,-1:,:] if settings.get('center_last') else np.zeros((len(h),1,2),np.float32)
    return (h-origin)/scale,origin,scale,single

def finish_motion(output,origin,scale,single,metadata):
    y=np.asarray(output);settings=metadata.get('postprocess',{})
    if settings.get('unscale'):y=y*scale
    if settings.get('relative_to_last'):y=y+origin
    return y[0] if single else y
