import numpy as np

def baseline_predictions(history,horizons=(.5,1,2),reference_time=None):
    if not history:return {}
    times=np.asarray([p[0] for p in history],float);pos=np.asarray([p[1] for p in history],float);vel=np.zeros(pos.shape[1])
    if len(pos)>=2 and times[-1]>times[-2]:vel=(pos[-1]-pos[-2])/(times[-1]-times[-2])
    elapsed=max(0,(reference_time if reference_time is not None else times[-1])-times[-1])
    return {"constant_position":{str(h):pos[-1].tolist() for h in horizons},"constant_velocity":{str(h):(pos[-1]+(elapsed+h)*vel).tolist() for h in horizons}}

def operational_behavior(history,ranges=None):
    if len(history)<2 or history[-1][0]-history[0][0]<.5:return "unknown"
    recent=[h for h in history if h[0]>=history[-1][0]-.5]
    if len(recent)<2:return "unknown"
    speeds=[np.linalg.norm(np.asarray(b[1])-a[1])/(b[0]-a[0]) for a,b in zip(recent,recent[1:]) if b[0]>a[0]]
    if not speeds:return "unknown"
    if all(s<.1 for s in speeds):return "stationary"
    if not all(s>=.2 for s in speeds):return "unknown"
    if ranges and len(ranges)>=2:
        rr=[h for h in ranges if h[0]>=ranges[-1][0]-.5]
        rates=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(rr,rr[1:]) if b[0]>a[0]]
        if rates and all(v<-.2 for v in rates):return "approaching"
    return "moving"
