"""Reactive policy: rendered depth and wheel odometry, no simulator state."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from robot_model_sdk import ControlCommandV1

@dataclass
class WheelOdometry:
    radius_m:float=.075
    track_m:float=.32
    x:float=0.
    y:float=0.
    yaw:float=0.
    previous:np.ndarray|None=None
    def update(self,angles):
        angles=np.asarray(angles,dtype=float)
        if self.previous is not None:
            left,right=(angles-self.previous)*self.radius_m
            ds=(left+right)/2;dyaw=(right-left)/self.track_m
            self.x+=ds*math.cos(self.yaw+dyaw/2);self.y+=ds*math.sin(self.yaw+dyaw/2)
            self.yaw=wrap(self.yaw+dyaw)
        self.previous=angles.copy()
        return np.array((self.x,self.y,self.yaw))

def wrap(angle):return (angle+math.pi)%(2*math.pi)-math.pi

def observed_clearance(frame,camera_height_m=.30,robot_half_width_m=.23):
    if frame.depth is None or frame.calibration is None:return None
    c=frame.calibration
    if c.depth_convention!='z' or c.depth_scale!=1:raise ValueError('Metric camera-Z depth required')
    depth=np.asarray(frame.depth);v,u=np.mgrid[:depth.shape[0],:depth.shape[1]]
    x=(u-c.cx)*depth/c.fx;y=(v-c.cy)*depth/c.fy
    valid=np.isfinite(depth)&(depth>.05)&(depth<8)
    points=np.stack((x,y,depth),axis=-1)
    # Validate the ground from observed points and the declared camera/gravity
    # mount; no floor geometry or simulator pose is queried by this function.
    candidates=points[valid&(v>depth.shape[0]*.6)&(np.abs(y-camera_height_m)<.09)]
    if len(candidates)>1500:candidates=candidates[np.linspace(0,len(candidates)-1,1500,dtype=int)]
    best=None;best_count=0;rng=np.random.default_rng(42)
    if len(candidates)>=30:
        for _ in range(24):
            triple=candidates[rng.choice(len(candidates),3,replace=False)]
            normal=np.cross(triple[1]-triple[0],triple[2]-triple[0]);norm=np.linalg.norm(normal)
            if norm<1e-8:continue
            normal=normal/norm
            if normal[1]<0:normal=-normal
            d=-float(normal@triple[0])
            if normal[1]<math.cos(math.radians(15)) or abs(-d-camera_height_m)>.09:continue
            residual=np.abs(candidates@normal+d);count=int(np.sum(residual<.015))
            if count>best_count:best_count=count;best=(normal,d)
    ground_validated=best is not None and best_count>=max(30,len(candidates)*.6)
    if ground_validated:
        normal,d=best;height=-(points@normal+d)
    else:height=camera_height_m-y
    obstacles=valid&(height>.08)&(height<1.3)
    corridor=obstacles&(np.abs(x)<robot_half_width_m)
    distance=float(np.quantile(depth[corridor],.05)) if np.any(corridor) else 8.
    sectors=[]
    for lo,hi in ((-.55,-.18),(-.18,.18),(.18,.55)):
        bearing=np.arctan2(-x,depth)
        mask=obstacles&(bearing>=lo)&(bearing<hi)
        sectors.append(float(np.quantile(depth[mask],.1)) if np.any(mask) else 8.)
    return {'forward_m':distance,'right_m':sectors[0],'center_m':sectors[1],'left_m':sectors[2],
            'valid_fraction':float(valid.mean()),'obstacle_pixels':int(obstacles.sum()),'ground_validated':bool(ground_validated),'ground_inliers':best_count,'ground_candidates':len(candidates),'ground_plane_camera':np.append(best[0],best[1]).tolist() if ground_validated else None}

class ObservedController:
    def __init__(self,goal=(4.,0.),stop_distance_m=.8,max_v=.3,max_omega=.6):
        if stop_distance_m<.8 or max_v>.3 or max_omega>.6:raise ValueError('Safety limits exceeded')
        self.goal=np.asarray(goal,dtype=float);self.stop_distance_m=stop_distance_m
        self.max_v=max_v;self.max_omega=max_omega;self.avoid_direction=0;self.avoid_start_yaw=None
        self.bypass_until=None
    def command(self,frame,odom,perception,now_ns,result_age_ms):
        reason='goal_tracking';v=0.;omega=0.;stop=False
        if frame.depth is None:
            reason='missing_depth';stop=True
        elif perception is None or not np.isfinite(result_age_ms) or result_age_ms>300:
            reason='stale_perception';stop=True
        elif perception.get('ground_validated') is False:
            reason='ground_geometry_unavailable';stop=True
        elif perception['valid_fraction']<.05:
            reason='insufficient_depth';stop=True
        else:
            odom=np.asarray(odom);delta=self.goal-odom[:2];goal_distance=float(np.linalg.norm(delta))
            heading=wrap(math.atan2(delta[1],delta[0])-float(odom[2]))
            forward=perception['forward_m']
            if goal_distance<.1:
                reason='observed_goal_reached'
            elif self.avoid_direction:
                turned=abs(wrap(float(odom[2])-self.avoid_start_yaw))
                if turned<.75:
                    omega=self.avoid_direction*self.max_omega;reason='avoid_turn'
                else:
                    self.avoid_direction=0;self.bypass_until=(odom[:2].copy(),float(odom[2]));reason='avoid_bypass'
            elif forward<self.stop_distance_m+.25:
                self.avoid_direction=1 if perception['left_m']>=perception['right_m'] else -1
                self.avoid_start_yaw=float(odom[2]);omega=self.avoid_direction*self.max_omega
                reason='obstacle_stop_and_turn';stop=True
            if reason=='avoid_bypass' or (reason=='goal_tracking' and self.bypass_until is not None):
                origin,yaw=self.bypass_until
                if float(np.linalg.norm(odom[:2]-origin))<.95:
                    reason='avoid_bypass';v=min(.22,self.max_v);omega=float(np.clip(1.5*wrap(yaw-float(odom[2])),-self.max_omega,self.max_omega))
                else:self.bypass_until=None
            if reason=='goal_tracking':
                omega=float(np.clip(1.5*heading,-self.max_omega,self.max_omega))
                v=min(self.max_v,.3*goal_distance)*max(0.,math.cos(heading)) if abs(heading)<.7 else 0.
            if forward<self.stop_distance_m:
                v=0.;stop=True
                if reason=='goal_tracking':reason='obstacle_safety_stop'
        return ControlCommandV1(timestamp_ns=int(now_ns),linear_mps=float(v),angular_rps=float(omega),
            state='stopped' if abs(v)<1e-8 and abs(omega)<1e-8 else 'moving',reason=reason,
            source_sample_id=frame.sample_id,observed_result_age_ms=float(result_age_ms) if np.isfinite(result_age_ms) else None,safety_stop=stop and omega==0.)
