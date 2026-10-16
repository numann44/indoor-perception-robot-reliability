"""MuJoCo differential-drive experiments with a sensor-only control boundary.

Scene state drives physics/rendering and evaluator labels. The controller sees
RGB-D, fixed calibration and wheel-encoder odometry only. Learned outputs are
shadow observations unless an explicit matched policy experiment is requested.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from robot_model_sdk import Calibration, SensorFrameV1, ControlCommandV1
from .controller import ObservedController, WheelOdometry, observed_clearance

SCENARIOS=('clear','static_obstacle','crossing_actor','occlusion','depth_loss','slow_inference')


def scene_xml(scenario='clear',seed=42):
    if scenario not in SCENARIOS:raise ValueError('Unknown scenario')
    rng=np.random.default_rng(seed)
    offset=float(rng.uniform(-.06,.06))
    obstacle=''
    if scenario=='static_obstacle':
        obstacle=f'<geom name="obstacle" type="box" pos="1.9 {offset} .5" size=".18 .28 .5" rgba=".5 .45 .35 1"/>'
    if scenario=='occlusion':
        obstacle='<geom name="occluder" type="box" pos="2.2 -.58 .65" size=".1 .32 .65" rgba=".55 .55 .6 1"/>'
    actorpos='2 3 .5' if scenario not in ('crossing_actor','occlusion') else f'2 {1.2+offset} .5'
    return f'''<mujoco model="portfolio_observed_robot"><compiler angle="radian"/>
<option timestep=".005" gravity="0 0 -9.81" integrator="implicitfast"/>
<visual><global offwidth="320" offheight="240"/><quality shadowsize="1024"/></visual>
<default><geom friction="1 .01 .001" condim="4"/><joint damping=".05"/></default>
<worldbody><light pos="0 0 5" diffuse=".8 .8 .8"/>
<geom name="floor" type="plane" size="10 10 .1" friction=".02 .001 .0001" rgba=".7 .73 .77 1"/>
<geom name="left_wall" type="box" pos="4 2.5 .8" size="6 .05 .8" rgba=".8 .85 .9 1"/>
<geom name="right_wall" type="box" pos="4 -2.5 .8" size="6 .05 .8" rgba=".8 .85 .9 1"/>
<geom name="end_wall" type="box" pos="6 0 .8" size=".05 2.5 .8" rgba=".65 .72 .8 1"/>{obstacle}
<body name="actor" mocap="true" pos="{actorpos}"><geom name="actor_geom" type="capsule" size=".16 .35" rgba=".85 .25 .18 1"/></body>
<body name="robot" pos="0 0 .14"><freejoint name="base"/>
<geom name="base_geom" type="box" size=".2 .14 .065" mass="3" rgba=".2 .45 .65 1"/>
<body name="left_wheel" pos="0 .16 -.06"><joint name="left_joint" type="hinge" axis="0 1 0"/>
<geom name="left_wheel_geom" type="cylinder" size=".075 .025" euler="1.5707963 0 0" mass=".25" rgba=".12 .12 .12 1"/></body>
<body name="right_wheel" pos="0 -.16 -.06"><joint name="right_joint" type="hinge" axis="0 1 0"/>
<geom name="right_wheel_geom" type="cylinder" size=".075 .025" euler="1.5707963 0 0" mass=".25" rgba=".12 .12 .12 1"/></body>
<body name="rear_caster_body" pos="-.14 0 -.095"><joint name="rear_caster_ball" type="ball" damping=".00001"/><geom name="caster" type="sphere" size=".04" mass=".03" friction="1 .001 .0001" rgba=".2 .2 .2 1"/></body>
<body name="front_caster_body" pos=".14 0 -.095"><joint name="front_caster_ball" type="ball" damping=".00001"/><geom name="front_caster" type="sphere" size=".04" mass=".03" friction="1 .001 .0001" rgba=".2 .2 .2 1"/></body>
<camera name="rgbd" pos=".20 0 .16" xyaxes="0 -1 0 0 0 1" fovy="55"/>
<site name="imu" pos="0 0 0"/></body></worldbody>
<actuator><velocity name="left_motor" joint="left_joint" kv="1" ctrllimited="true" ctrlrange="-8 8"/>
<velocity name="right_motor" joint="right_joint" kv="1" ctrllimited="true" ctrlrange="-8 8"/></actuator>
<sensor><jointpos name="left_encoder" joint="left_joint"/><jointpos name="right_encoder" joint="right_joint"/>
<gyro name="gyro" site="imu"/><accelerometer name="accelerometer" site="imu"/></sensor></mujoco>'''

class MuJoCoFrameSource:
    def __init__(self,scenario='clear',seed=42,width=160,height=120,sensor_noise=True):
        os.environ.setdefault('MUJOCO_GL','cgl')
        import mujoco
        self.mujoco=mujoco;self.scenario=scenario;self.seed=seed;self.sensor_noise=sensor_noise
        self.sensor_rng=np.random.default_rng(seed+10000);motion_rng=np.random.default_rng(seed+20000)
        self.actor_start_y=1.25+float(motion_rng.uniform(-.1,.1));self.actor_speed=.32+float(motion_rng.uniform(-.025,.025))
        self.xml=scene_xml(scenario,seed);self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.data=mujoco.MjData(self.model);mujoco.mj_forward(self.model,self.data)
        self.renderer=mujoco.Renderer(self.model,height=height,width=width)
        f=.5*height/math.tan(math.radians(55)/2)
        self.calibration=Calibration(f,f,(width-1)/2,(height-1)/2,width,height,1.,'z')
        self.odom=WheelOdometry();self.counter=0;self.closed=False;self._command=ControlCommandV1(0,0.,0.,'stopped','awaiting_control');self.current_odom=np.zeros(3)
        self.wheel_qpos=[int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in ('left_joint','right_joint')]
        self.robot_body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'robot')
        self.actor_body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'actor')
        self.actor_mocap=int(self.model.body_mocapid[self.actor_body])
        self._evaluator_contacts=[]
        self._robot_geom_ids=set()
        for i in range(self.model.ngeom):
            body=int(self.model.geom_bodyid[i])
            while body>0 and body!=self.robot_body:body=int(self.model.body_parentid[body])
            if body==self.robot_body:self._robot_geom_ids.add(i)
    def metadata(self):
        return {'source':'MuJoCo controlled simulation','version':self.mujoco.__version__,'scenario':self.scenario,
                'seed':self.seed,'scene_sha256':hashlib.sha256(self.xml.encode()).hexdigest(),
                'rgb_depth':'same calibrated offscreen camera; depth is metric axial camera-Z',
                'pose_source':'wheel_encoder_odometry','coordinate_frame':'odom: X forward, Y left, Z up; camera: X right, Y down, Z forward',
                'ground_truth_boundary':'physics state and contacts are evaluator-only, not supplied to controller',
                'controlled_sensor_noise':{'encoder_sigma_rad':.0005 if self.sensor_noise else 0.,'depth_sigma_m':.001 if self.sensor_noise else 0.},
                'assumption':'planar encoder odometry with nominal fixed camera mount; slip is not corrected from simulator truth'}
    def step(self,command,dt=.1):
        v=float(command.linear_mps);omega=float(command.angular_rps)
        if abs(v)>.3+1e-9 or abs(omega)>.6+1e-9:raise ValueError('Controller limit violation')
        # First encoder is left (+Y); right is -Y. Positive yaw needs the right wheel faster.
        self.data.ctrl[:]=[(v-.16*omega)/.075,(v+.16*omega)/.075]
        self._evaluator_contacts=[]
        for _ in range(round(dt/self.model.opt.timestep)):
            t=float(self.data.time)
            if self.scenario=='crossing_actor':
                self.data.mocap_pos[self.actor_mocap]=[2.,self.actor_start_y-self.actor_speed*t,.5]
            elif self.scenario=='occlusion':
                self.data.mocap_pos[self.actor_mocap]=[2.3,-self.actor_start_y+self.actor_speed*max(0,t-2),.5]
            self.mujoco.mj_step(self.model,self.data)
            # Labels only: capture contacts at every5ms physics step, never send to controller.
            for contact in self.data.contact:
                a,b=int(contact.geom1),int(contact.geom2)
                if (a in self._robot_geom_ids)==(b in self._robot_geom_ids):continue
                other=b if a in self._robot_geom_ids else a
                name=self.mujoco.mj_id2name(self.model,self.mujoco.mjtObj.mjOBJ_GEOM,other)
                if name!='floor':self._evaluator_contacts.append({'geom':name,'penetration_m':max(0.,-float(contact.dist)),'physics_time_s':float(self.data.time)})
        angles=self.data.qpos[self.wheel_qpos].copy()
        if self.sensor_noise:angles+=self.sensor_rng.normal(0,.0005,size=2)
        self.current_odom=self.odom.update(angles)
    def capture(self,depth_missing=False):
        if self.closed:raise RuntimeError('FrameSource is closed')
        self.renderer.update_scene(self.data,camera='rgbd');self.renderer.disable_depth_rendering()
        rgb=self.renderer.render().copy();self.renderer.enable_depth_rendering();depth=self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        if self.sensor_noise:depth=np.maximum(0,depth+self.sensor_rng.normal(0,.001,size=depth.shape)).astype(np.float32)
        yaw=float(self.current_odom[2]);R=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1.]])
        T=np.eye(4);T[:3,:3]=R@np.array([[0,0,1],[-1,0,0],[0,-1,0.]])
        T[:3,3]=[self.current_odom[0]+.20*math.cos(yaw),self.current_odom[1]+.20*math.sin(yaw),.30]
        frame=SensorFrameV1(f'mujoco-{self.scenario}-{self.seed}-{self.counter:05d}',int(round(self.data.time*1e9)),rgb,
                            None if depth_missing else depth.astype(np.float32),self.calibration,T,'wheel_encoder_odometry')
        self.counter+=1;return frame
    def apply_control(self,command):
        self._command=command
    def __iter__(self):
        while not self.closed:
            yield self.capture()
            command=self._command
            now=int(round(self.data.time*1e9))
            if now-command.timestamp_ns>300_000_000:
                command=ControlCommandV1(now,0.,0.,'stopped','stale_control',safety_stop=True)
            self.step(command)
    def close(self):
        if not self.closed:self.renderer.close();self.closed=True


def evaluator_state(source):
    """Private evaluator: never passed to ObservedController."""
    m=source.model;d=source.data
    position=d.xpos[source.robot_body].copy();actor=d.xpos[source.actor_body].copy()
    clearances=[];robot_geom_ids=set()
    for i in range(m.ngeom):
        body=int(m.geom_bodyid[i])
        while body>0 and body!=source.robot_body:body=int(m.body_parentid[body])
        if body==source.robot_body:robot_geom_ids.add(i)
    collisions=list(source._evaluator_contacts)
    for contact in d.contact:
        a,b=int(contact.geom1),int(contact.geom2)
        if (a in robot_geom_ids) == (b in robot_geom_ids):continue
        other=b if a in robot_geom_ids else a
        name=source.mujoco.mj_id2name(m,source.mujoco.mjtObj.mjOBJ_GEOM,other)
        if name!='floor':collisions.append({'geom':name,'penetration_m':max(0.,-float(contact.dist))})
    for i in range(m.ngeom):
        name=source.mujoco.mj_id2name(m,source.mujoco.mjtObj.mjOBJ_GEOM,i)
        if i in robot_geom_ids or name=='floor':continue
        for robot_geom in robot_geom_ids:
            distance=float(source.mujoco.mj_geomDistance(m,d,robot_geom,i,10.,np.zeros(6)))
            clearances.append(distance)
    return {'position_xyz':position.tolist(),'actor_xyz':actor.tolist(),'collisions':collisions,
            'contact_sampling':'every5ms physics step since prior control tick; endpoint contacts also retained','minimum_clearance_m':min(clearances,default=None),'clearance_definition':'MuJoCo exact signed geomDistance, excluding floor, over all robot-vs-obstacle geometry pairs','true_goal_distance_m':float(np.linalg.norm(position[:2]-np.array((4.,0.))))}


def geometry_smoke(output):
    import cv2
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=MuJoCoFrameSource('static_obstacle',sensor_noise=False)
    try:
        frame=source.capture();camera=source.mujoco.mj_name2id(source.model,source.mujoco.mjtObj.mjOBJ_CAMERA,'rgbd')
        # Evaluator-only true extrinsic validates the rendered depth contract.
        R=source.data.cam_xmat[camera].reshape(3,3)@np.diag([1.,-1.,-1.]);origin=source.data.cam_xpos[camera]
        h,w=frame.depth.shape;u=(w-1)/2;v=(h-1)/2;c=frame.calibration
        z=float(np.median(frame.depth[h//2-2:h//2+2,w//2-2:w//2+2]));point=R@np.array([(u-c.cx)*z/c.fx,(v-c.cy)*z/c.fy,z])+origin
        expected_x=1.9-.18;error=abs(float(point[0])-expected_x)
        trials=[]
        gid=source.mujoco.mj_name2id(source.model,source.mujoco.mjtObj.mjOBJ_GEOM,'obstacle')
        old=float(source.model.geom_pos[gid,0])
        for distance in (.5,1.,2.,3.,5.):
            source.model.geom_pos[gid,0]=.20+.18+distance
            source.mujoco.mj_forward(source.model,source.data)
            observed=source.capture();measured=float(np.median(observed.depth[h//2-2:h//2+2,w//2-2:w//2+2]))
            off_axis=float(observed.depth[h//2,w//2+5]);ray_norm=math.sqrt(1+((w//2+5-c.cx)/c.fx)**2+((h//2-c.cy)/c.fy)**2)
            trials.append({'expected_axial_m':distance,'measured_center_m':measured,'absolute_error_m':abs(measured-distance),
                           'off_axis_axial_m':off_axis,'off_axis_expected_radial_m':distance*ray_norm,
                           'radial_z_difference_m':distance*(ray_norm-1)})
        source.model.geom_pos[gid,0]=old;source.mujoco.mj_forward(source.model,source.data)
        rgbpath=output/'rgb.png';cv2.imwrite(str(rgbpath),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
        depthvis=np.clip(frame.depth/5*255,0,255).astype(np.uint8);cv2.imwrite(str(output/'depth.png'),depthvis)
        np.savez_compressed(output/'sensor.npz',rgb=frame.rgb,depth=frame.depth)
        result={'status':'verified' if error<.02 else 'failed','metadata':source.metadata(),
                'depth_center_m':z,'expected_plane_x_m':expected_x,'backprojected_world_xyz':point.tolist(),
                'plane_error_m':error,'distance_trials':trials,'all_range_errors_under_2cm':all(t['absolute_error_m']<.02 for t in trials),'off_axis_depth_is_axial':all(abs(t['off_axis_axial_m']-t['expected_axial_m'])<.002 for t in trials),'OpenGL_warning':'ARB_clip_control unavailable for ZEROFAR on macOS; the 0.5/1/2/3/5 m rendered planes are quantitatively checked rather than assuming finite depth is correct','depth_valid_fraction':float(np.mean(np.isfinite(frame.depth)&(frame.depth>0))),
                'calibration':c.to_json(),'rgb':str(rgbpath),'offscreen':True,
                'rendering_contract':'MuJoCo renderer metric axial depth, not raw OpenGL normalized depth'}
        (output/'run.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['status']!='verified' or not result['all_range_errors_under_2cm'] or not result['off_axis_depth_is_axial']:raise RuntimeError('Rendered depth geometry failed')
        return result
    finally:source.close()


def run_scenarios(output,seeds=10,seconds=35,scenarios='all',model_pack=None,seed_start=100):
    import cv2
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    selected=SCENARIOS if scenarios=='all' else tuple(scenarios.split(','))
    if any(s not in SCENARIOS for s in selected):raise ValueError('Unsupported scenario')
    rows=[];start=time.time()
    from .provenance import context,finish
    provenance=context({'seeds':list(range(seed_start,seed_start+seeds)),'episode_seconds':seconds,'scenarios':list(selected),'model_pack':model_pack,'policy':'observed RGB-D + wheel odometry','stop_distance_m':.8,'maximum_v':.3,'maximum_omega':.6,'result_age_stop_ms':300,'observed_goal_radius_m':.1,'evaluator_goal_radius_m':.35,'pilot_seed_excluded':42,'encoder_noise_sigma_rad':.0005,'depth_noise_sigma_m':.001})
    adapter=None
    if model_pack:
        from .remote_model import RemoteModelSession
        adapter=RemoteModelSession(Path(__file__).parents[5],model_pack,output/'shadow-inference').__enter__()
    try:
        for scenario in selected:
            for seed in range(seed_start,seed_start+seeds):
                source=MuJoCoFrameSource(scenario,seed);controller=ObservedController()
                trace=[];pending=[];latest=None;latest_stamp=None;collision_steps=0;minclearance=float('inf')
                stop_delays=[];recovered=False;fault_last=False;fault_started=None;awaiting_recovery=False;reach=False;shadow=[]
                try:
                    frame=source.capture()
                    # Initialize encoder readings before motion; observed odometry starts at zero.
                    source.current_odom=source.odom.update(source.data.qpos[source.wheel_qpos])
                    for step in range(round(seconds/.1)):
                        t=float(source.data.time);now_ns=int(round(t*1e9))
                        missing=scenario=='depth_loss' and 3<=t<5
                        delay=.4 if scenario=='slow_inference' and 3<=t<5 else 0.
                        frame=source.capture(depth_missing=missing)
                        observation=observed_clearance(frame)
                        if observation is not None:pending.append((t+delay,now_ns,observation))
                        ready=[p for p in pending if p[0]<=t+1e-8];pending=[p for p in pending if p[0]>t+1e-8]
                        if ready:
                            newest=max(ready,key=lambda p:p[1])
                            if latest_stamp is None or newest[1]>latest_stamp:latest_stamp=newest[1];latest=newest[2]
                        age=(now_ns-latest_stamp)/1e6 if latest_stamp is not None else 1e9
                        command=controller.command(frame,source.current_odom,latest,now_ns,age)
                        fault=missing or age>300
                        if fault and not fault_last:fault_started=t;awaiting_recovery=True
                        if fault and command.safety_stop and fault_started is not None:
                            stop_delays.append(t-fault_started);fault_started=None
                        if awaiting_recovery and not fault and command.linear_mps>0:recovered=True;awaiting_recovery=False
                        fault_last=fault
                        # Learned predictions are shadow-only: never change this geometric safety command.
                        if adapter is not None and step%10==0:
                            ts=time.perf_counter();pred=adapter.predict(frame)
                            shadow.append({'sample_id':frame.sample_id,'model_output':pred,
                                           'elapsed_ms':(time.perf_counter()-ts)*1000,'role':'shadow; no control influence'})
                        source.step(command)
                        truth=evaluator_state(source)
                        collision_steps+=bool(truth['collisions']);minclearance=min(minclearance,truth['minimum_clearance_m'])
                        reach=truth['true_goal_distance_m']<.35
                        trace.append({'time_s':t,'sample_id':frame.sample_id,'odom_xyyaw':source.current_odom.tolist(),
                                      'depth_available':frame.depth is not None,'result_age_ms':age,
                                      'observed_clearance':latest,'command':command.to_json(),
                                      'evaluator':truth})
                        if seed==seed_start and step in (0,30,40,50):
                            cv2.imwrite(str(output/f'{scenario}-{step:04d}.png'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                        if reach:break
                    name=f'{scenario}-seed{seed}'
                    (output/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
                    if shadow:(output/f'{name}-shadow.json').write_text(json.dumps(shadow,indent=2)+'\n')
                    row={'scenario':scenario,'seed':seed,'goal_reached':reach,'collision_steps':collision_steps,
                         'minimum_clearance_m':minclearance,'elapsed_sim_s':float(source.data.time),
                         'final_goal_distance_m':truth['true_goal_distance_m'],
                         'observed_goal_distance_m':float(np.linalg.norm(source.current_odom[:2]-controller.goal)),
                         'final_odometry_position_error_m':float(np.linalg.norm(source.current_odom[:2]-np.asarray(truth['position_xyz'][:2]))),
                         'safety_stop_count':sum(r['command']['safety_stop'] for r in trace),
                         'max_fault_stop_delay_s':max(stop_delays,default=None),'recovery_observed':recovered,
                         'success':bool(reach and collision_steps==0),'trace':str(output/f'{name}.jsonl'),
                         'metadata':source.metadata(),'evaluator_scene_params':{'actor_initial_y_m':source.actor_start_y,'actor_speed_mps':source.actor_speed},'learned_role':'shadow only' if adapter else 'unavailable'}
                    rows.append(row)
                    print(json.dumps({'scenario':scenario,'seed':seed,'success':row['success'],'goal':reach,'collisions':collision_steps}),flush=True)
                finally:source.close()
    finally:
        if adapter is not None:adapter.close()
    summary={'provenance':finish(provenance),'status':'executed','source':'controlled MuJoCo simulation, not physical robot validation',
             'rows':rows,'episodes':len(rows),'elapsed_wall_s':time.time()-start,
             'successes':sum(r['success'] for r in rows),'collision_episodes':sum(r['collision_steps']>0 for r in rows),
             'controller_observations':['RGB-D camera','known intrinsics/mount','wheel encoders','observed perception timestamp','requested goal'],
             'truth_access':'only evaluator and scene physics/rendering; never controller',
             'slow_inference_scope':'causal 400 ms virtual-time sensor-result delay; actual wall-clock backend stress is a separate ROS/IPC experiment',
             'limits':{'linear_mps':.3,'angular_rps':.6,'minimum_stop_distance_m':.8,'result_age_stop_ms':300,'control_period_s':.1},
             'contact_sampling':'every5ms physics step; minimumclearance sampled at100ms control ticks',
             'success_definition':'true goal radius <0.35 m AND no obstacle contact; stationary episodes never count as success',
             'model_pack':model_pack,'learned_control_contribution':'not established; shadow outputs do not affect geometric policy'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return {k:v for k,v in summary.items() if k!='rows'}
