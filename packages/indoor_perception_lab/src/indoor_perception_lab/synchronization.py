import numpy as np

def match_timestamps(rgb_ns,depth_ns,tolerance_ns=20_000_000):
    if any(b<=a for a,b in zip(rgb_ns,rgb_ns[1:])) or any(b<=a for a,b in zip(depth_ns,depth_ns[1:])):raise ValueError("Sensor timestamps must be strictly increasing")
    used=set();out=[]
    for i,t in enumerate(rgb_ns):
        candidates=[j for j,d in enumerate(depth_ns) if j not in used and abs(d-t)<=tolerance_ns]
        if not candidates:out.append((i,None));continue
        j=min(candidates,key=lambda j:(abs(depth_ns[j]-t),depth_ns[j]));used.add(j);out.append((i,j))
    return out

def interpolate_pose(timestamp_ns,poses,max_gap_ns=100_000_000):
    from scipy.spatial.transform import Rotation,Slerp
    times=np.asarray([p[0] for p in poses],np.int64)
    if len(times)==0 or timestamp_ns<times[0] or timestamp_ns>times[-1]:return None
    idx=int(np.searchsorted(times,timestamp_ns))
    if idx<len(times) and times[idx]==timestamp_ns:return np.asarray(poses[idx][1]).copy()
    if idx==0 or idx==len(times) or times[idx]-times[idx-1]>max_gap_ns:return None
    a,b=poses[idx-1],poses[idx];alpha=(timestamp_ns-a[0])/(b[0]-a[0]);T=np.eye(4);T[:3,3]=(1-alpha)*np.asarray(a[1])[:3,3]+alpha*np.asarray(b[1])[:3,3]
    T[:3,:3]=Slerp([0,1],Rotation.from_matrix([np.asarray(a[1])[:3,:3],np.asarray(b[1])[:3,:3]]))([alpha]).as_matrix()[0];return T
