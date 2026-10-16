"""RGB-D registration and observed odometry interpolation in declared frames."""
from __future__ import annotations
from bisect import bisect_left
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def rigid_transform(xyz, xyzw):
    q=np.asarray(xyzw,dtype=float)
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q),1,atol=1e-3):
        raise ValueError('Invalid unit quaternion')
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(q).as_matrix();T[:3,3]=xyz
    if not np.isfinite(T).all():raise ValueError('Nonfinite pose')
    return T


def nearest_one_to_one(rgb_ns, depth_ns, tolerance_ns=20_000_000):
    """Deterministic nearest unused depth match; no pairing outside tolerance."""
    rgb_ns=list(rgb_ns);depth_ns=list(depth_ns)
    if rgb_ns!=sorted(rgb_ns) or depth_ns!=sorted(depth_ns):raise ValueError('Timestamps must be sorted')
    used=set();matches=[]
    for t in rgb_ns:
        lo=bisect_left(depth_ns,t-tolerance_ns);hi=bisect_left(depth_ns,t+tolerance_ns+1)
        choices=[i for i in range(lo,hi) if i not in used]
        selected=min(choices,key=lambda i:(abs(depth_ns[i]-t),depth_ns[i],i)) if choices else None
        if selected is not None:used.add(selected)
        matches.append(selected)
    return matches


def interpolate_pose(stamps, transforms, timestamp_ns, maximum_gap_ns=100_000_000):
    stamps=np.asarray(stamps,dtype=np.int64)
    if not len(stamps) or np.any(np.diff(stamps)<=0):raise ValueError('Unique sorted odometry stamps required')
    i=int(np.searchsorted(stamps,timestamp_ns))
    if i<len(stamps) and stamps[i]==timestamp_ns:return np.asarray(transforms[i]).copy()
    if i==0 or i==len(stamps):return None
    gap=int(stamps[i]-stamps[i-1])
    if gap>maximum_gap_ns:return None
    a=(timestamp_ns-int(stamps[i-1]))/gap
    before=np.asarray(transforms[i-1]);after=np.asarray(transforms[i])
    out=np.eye(4);out[:3,3]=before[:3,3]*(1-a)+after[:3,3]*a
    out[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(np.stack((before[:3,:3],after[:3,:3]))))([a]).as_matrix()[0]
    return out


def resolve_static_transform(edges, parent, child):
    """Return T_parent_child using sensor extrinsic edges, including inverses."""
    parent=parent.lstrip('/');child=child.lstrip('/')
    if parent==child:return np.eye(4)
    graph={}
    for p,c,T in edges:
        p=p.lstrip('/');c=c.lstrip('/')
        graph.setdefault(p,[]).append((c,np.asarray(T)))
        graph.setdefault(c,[]).append((p,np.linalg.inv(T)))
    frontier=[(parent,np.eye(4))];seen={parent}
    while frontier:
        name,T=frontier.pop(0)
        for nxt,edge in graph.get(name,[]):
            if nxt in seen:continue
            combined=T@edge
            if nxt==child:return combined
            seen.add(nxt);frontier.append((nxt,combined))
    return None


def register_depth(depth_m, K_depth, K_rgb, T_rgb_depth, rgb_shape, distortion_depth=None):
    """Project metric camera-Z depth through calibrated extrinsics, nearest Z buffer.

    RGB is rectified to K_rgb first. Invalid/zero/range points remain unknown,
    never free. The input is axial camera-Z; radial data is rejected by callers.
    """
    import cv2
    depth=np.asarray(depth_m,dtype=np.float32);h,w=depth.shape
    u,v=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    pixels=np.stack((u,v),axis=-1).reshape(-1,1,2)
    Kd=np.asarray(K_depth,dtype=float);Kr=np.asarray(K_rgb,dtype=float)
    D=np.asarray(distortion_depth if distortion_depth is not None else [],dtype=float)
    rays=cv2.undistortPoints(pixels,Kd,D).reshape(-1,2) if len(D) else np.stack(((u-Kd[0,2])/Kd[0,0],(v-Kd[1,2])/Kd[1,1]),axis=-1).reshape(-1,2)
    z=depth.reshape(-1)
    valid=np.isfinite(z)&(z>0)&(z<=10)
    p=np.column_stack((rays[valid]*z[valid,None],z[valid]))
    T=np.asarray(T_rgb_depth,dtype=float)
    if T.shape!=(4,4):raise ValueError('4x4 depth-to-RGB transform required')
    p=p@T[:3,:3].T+T[:3,3]
    valid=np.isfinite(p).all(axis=1)&(p[:,2]>0)
    p=p[valid];z=p[:,2]
    u=np.rint(Kr[0,0]*p[:,0]/z+Kr[0,2]).astype(np.int64)
    v=np.rint(Kr[1,1]*p[:,1]/z+Kr[1,2]).astype(np.int64)
    rh,rw=rgb_shape[:2];valid=(u>=0)&(u<rw)&(v>=0)&(v<rh)
    flat=np.full(rh*rw,np.inf,dtype=np.float32)
    np.minimum.at(flat,v[valid]*rw+u[valid],z[valid])
    flat[~np.isfinite(flat)]=0
    return flat.reshape(rh,rw)


def voxel_points(frame, voxel_size_m=.1, stride=4):
    """Observed metric occupied cells only; unknown space is not marked free."""
    if frame.depth is None or frame.calibration is None or frame.T_world_camera is None:return np.empty((0,3),dtype=np.int64)
    c=frame.calibration
    if c.depth_convention!='z' or c.depth_scale!=1:raise ValueError('Expected registered metric camera-Z depth')
    v,u=np.mgrid[0:frame.rgb.shape[0]:stride,0:frame.rgb.shape[1]:stride]
    z=frame.depth[v,u];valid=np.isfinite(z)&(z>0)&(z<8)
    p=np.stack(((u-c.cx)*z/c.fx,(v-c.cy)*z/c.fy,z),axis=-1)[valid]
    T=frame.T_world_camera;p=p@T[:3,:3].T+T[:3,3]
    return np.unique(np.floor(p/voxel_size_m).astype(np.int64),axis=0)
