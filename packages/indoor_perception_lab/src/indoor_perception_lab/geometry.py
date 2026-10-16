from __future__ import annotations
import numpy as np

def depth_to_z(depth,calibration):
    z=np.asarray(depth,dtype=np.float64)/calibration.depth_scale
    if calibration.depth_convention=="radial":
        v,u=np.indices(z.shape);z=z/np.sqrt(1+((u-calibration.cx)/calibration.fx)**2+((v-calibration.cy)/calibration.fy)**2)
    return np.where(np.isfinite(z)&(z>0),z,np.nan)

def project_depth(depth,calibration,stride=1):
    z=depth_to_z(depth,calibration);v,u=np.indices(z.shape);keep=np.isfinite(z)&(z<=8)
    keep &= (v%stride==0)&(u%stride==0)
    p=np.stack([(u-calibration.cx)*z/calibration.fx,(v-calibration.cy)*z/calibration.fy,z],-1)
    return p[keep],np.stack([v[keep],u[keep]],-1)

def transform_points(points,T):
    return np.asarray(points)@np.asarray(T)[:3,:3].T+np.asarray(T)[:3,3]

def robust_object_position(depth,calibration,bbox):
    z=depth_to_z(depth,calibration);h,w=z.shape;x0,y0,x1,y1=bbox
    dx=(x1-x0)*.1;dy=(y1-y0)*.1
    u0,u1=max(0,int(x0+dx)),min(w,int(np.ceil(x1-dx)));v0,v1=max(0,int(y0+dy)),min(h,int(np.ceil(y1-dy)))
    if u1<=u0 or v1<=v0:return None
    roi=z[v0:v1,u0:u1];valid=np.isfinite(roi)&(roi<=8)
    if valid.sum()<9:return None
    vv,uu=np.indices(roi.shape);zz=roi[valid];median=float(np.median(zz));mad=float(np.median(np.abs(zz-median)))
    inlier=np.abs(zz-median)<=max(.05,3*mad);zz=zz[inlier];us=(uu[valid]+u0)[inlier];vs=(vv[valid]+v0)[inlier]
    xyz=np.median(np.stack([(us-calibration.cx)*zz/calibration.fx,(vs-calibration.cy)*zz/calibration.fy,zz],1),axis=0)
    return {"position_camera_m":xyz.tolist(),"surface_range_m":float(np.linalg.norm(xyz)),"depth_mad_m":mad,"valid_pixels":int(valid.sum()),"uncertainty":"robust spread diagnostic; not calibrated covariance"}

class SpatialMap:
    def __init__(self,voxel_size=.1,max_voxels=100000):self.voxel_size=voxel_size;self.max_voxels=max_voxels;self.cells={};self.frame_id=None;self.mapping_mode=None
    def update(self,frame,semantic=None,dynamic_boxes=(),probabilities=None,dynamic_semantic_classes=(30,)):
        if frame.depth is None or frame.calibration is None or frame.T_world_camera is None:return {"updated":False,"reason":"missing_depth_calibration_or_pose"}
        if frame.pose_source=='provided_ground_truth':self.mapping_mode='ground-truth-pose-assisted mapping'
        elif frame.pose_source in {'recorded_wheel_odometry','robot_odometry','simulated_wheel_odometry'}:self.mapping_mode='odometry-assisted mapping'
        else:self.mapping_mode='known-pose-assisted mapping'
        points,vu=project_depth(frame.depth,frame.calibration,stride=6)
        keep=np.ones(len(points),bool)
        if semantic is not None:
            if semantic.shape!=frame.depth.shape:raise ValueError('Registered semantic/depth shapes must agree')
            keep &= ~np.isin(semantic[vu[:,0],vu[:,1]],dynamic_semantic_classes)
        for x0,y0,x1,y1 in dynamic_boxes:keep &= ~((vu[:,1]>=x0)&(vu[:,1]<x1)&(vu[:,0]>=y0)&(vu[:,0]<y1))
        points=transform_points(points[keep],frame.T_world_camera);vu=vu[keep];keys=np.floor(points/self.voxel_size).astype(int)
        labels=semantic[vu[:,0],vu[:,1]] if semantic is not None else np.full(len(keys),255)
        for pixel_index,(k,label) in enumerate(zip(keys,labels)):
            key=tuple(int(v) for v in k);cell=self.cells.setdefault(key,{"votes":{},"observations":0,"last_timestamp_ns":0})
            cell["observations"]+=1;cell["last_timestamp_ns"]=frame.timestamp_ns
            if int(label)!=255:cell["votes"][int(label)]=cell["votes"].get(int(label),0)+1
            if probabilities is not None:
                vv,uu=vu[pixel_index];ph,pw=probabilities.shape[-2:];v=min(ph-1,int(vv*ph/frame.rgb.shape[0]));u=min(pw-1,int(uu*pw/frame.rgb.shape[1]));prob=probabilities[:,v,u]
                cell.setdefault("probability_sums",np.zeros(len(prob),np.float64));cell["probability_sums"]+=prob
        if len(self.cells)>self.max_voxels:
            for k in sorted(self.cells,key=lambda k:self.cells[k]["last_timestamp_ns"])[:len(self.cells)-self.max_voxels]:del self.cells[k]
        self.frame_id="world";return {"updated":True,"observed_surface_voxels":len(self.cells),"pose_source":frame.pose_source,"mapping_mode":self.mapping_mode}
    def to_json(self):
        return {"representation":"observed surface occupancy; unobserved cells unknown","mapping_mode":self.mapping_mode,"voxel_size_m":self.voxel_size,"frame_id":self.frame_id,"cells":[{"index":list(k),"occupied":True,"observations":v["observations"],"semantic_class":int(np.argmax(v["probability_sums"])) if "probability_sums" in v else (max(v["votes"],key=v["votes"].get) if v["votes"] else None),"semantic_votes":v["votes"],"semantic_probability_average":(v["probability_sums"]/v["probability_sums"].sum()).tolist() if "probability_sums" in v else None,"last_timestamp_ns":v["last_timestamp_ns"]} for k,v in sorted(self.cells.items())],"uncertainty":"unobserved is unknown; semantic vote fractions are uncalibrated"}


def geometric_plane_mask(depth,calibration,seed=42,threshold_m=.05):
    """Dominant upright-camera ground-plane heuristic; not learned semantics."""
    points,_=project_depth(depth,calibration,stride=4);rng=np.random.default_rng(seed);best=None;best_count=0
    if len(points)<3:return np.full(depth.shape,255,np.uint8),{"available":False}
    for _ in range(100):
        p=points[rng.choice(len(points),3,replace=False)];normal=np.cross(p[1]-p[0],p[2]-p[0]);norm=np.linalg.norm(normal)
        if norm<1e-8:continue
        normal/=norm
        if abs(normal[1])<.9:continue
        offset=-np.dot(normal,p[0]);inlier=np.abs(points@normal+offset)<threshold_m
        if inlier.sum()>best_count:best=(normal,offset);best_count=int(inlier.sum())
    mask=np.full(depth.shape,255,np.uint8)
    if best is None or best_count<20:return mask,{"available":False,"reason":"no upright ground plane"}
    xyz,vu=project_depth(depth,calibration);inside=np.abs(xyz@best[0]+best[1])<threshold_m;mask[vu[:,0],vu[:,1]]=inside.astype(np.uint8)
    return mask,{"available":True,"normal_camera":best[0].tolist(),"offset_m":float(best[1]),"inliers":best_count,"assumption":"camera optical y approximately gravity; no calibrated semantic claim"}
