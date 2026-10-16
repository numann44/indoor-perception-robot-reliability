from pathlib import Path
import numpy as np

def render_frame(frame,result,semantic,spatial_map,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from .geometry import geometric_plane_mask
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');fig.patch.set_facecolor('#f5f7fa');ax=axes[0,0];ax.imshow(frame.rgb)
    display_tracks=sorted(result.get('tracks',[]),key=lambda t:t['score'],reverse=True)
    for i,t in enumerate(display_tracks):
        x0,y0,x1,y1=t['bbox'];color=plt.get_cmap('tab10')(int(t['id'].split('-')[-1])%10);ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color=color,lw=1))
        if i<8:ax.text(x0,max(0,y0-3),f"{t['id']} class{t['class_id']} {t['score']:.2f}",fontsize=7,color='white',clip_on=True,bbox={'facecolor':color,'alpha':.8,'pad':1})
        if i<8 and t.get('coordinate_frame')=='image' and t.get('confirmed'):
            for method,style in [('constant_velocity','--'),('MLP2x64','-')]:
                future=t.get('forecasts',{}).get(method,{});points=[[(x0+x1)/2,(y0+y1)/2]]+[future[k] for k in ['0.5','1','2'] if k in future]
                if len(points)>1:points=np.asarray(points);ax.plot(points[:,0],points[:,1],style,color=color,lw=1,marker='.',ms=3)
    ax.set_xlim(-.5,frame.rgb.shape[1]-.5);ax.set_ylim(frame.rgb.shape[0]-.5,-.5)
    ax.set_title('Sensor RGB + tracks (top8 ID/forecast labels)');ax.axis('off');ax=axes[0,1]
    if semantic is not None:ax.imshow(semantic,cmap=plt.get_cmap('turbo',40),vmin=-.5,vmax=39.5);ax.set_title('Learned NYU40 semantic output (0–39)')
    elif frame.depth is not None and frame.calibration:
        mask,info=geometric_plane_mask(frame.depth,frame.calibration);display=np.where(mask==255,np.nan,mask);ax.imshow(display,cmap='viridis',vmin=0,vmax=1);ax.set_title('Geometric plane heuristic; separate from semantics')
    else:ax.text(.5,.5,'Semantic model / valid depth unavailable',ha='center');ax.set_title('Sensor availability')
    ax.axis('off');ax=axes[1,0]
    cells=spatial_map.cells
    # A world X-Z slice need not be horizontal (robot odometry often has Z up).
    # Use the documented optical camera projection until a ground frame is verified.
    camera_from_world=np.linalg.inv(frame.T_world_camera) if frame.T_world_camera is not None else None
    def camera_points(world):
        values=np.asarray(world,float)
        return values@camera_from_world[:3,:3].T+camera_from_world[:3,3]
    if cells and camera_from_world is not None:
        points=camera_points((np.asarray(list(cells),float)+.5)*spatial_map.voxel_size);ax.scatter(points[:,0],points[:,2],s=2,alpha=.25,color='#5b687b')
    for i,t in enumerate(result.get('tracks',[])):
        p=t.get('position') or {};point=p.get('position_world_m');camera_point=p.get('position_camera_m');color=plt.get_cmap('tab10')(int(t['id'].split('-')[-1])%10)
        if camera_point:ax.scatter(camera_point[0],camera_point[2],s=35,color=color)
        if point and camera_from_world is not None:
            future=t.get('forecasts',{}).get('constant_velocity',{});values=[point]+[future[k] for k in ['0.5','1','2'] if k in future]
            if len(values)>1:values=camera_points(values);ax.plot(values[:,0],values[:,2],'--o',ms=3,color=color)
            learned=t.get('forecasts',{}).get('MLP2x64',{});lp=[point]+[learned[k] for k in ['0.5','1','2'] if k in learned]
            if len(lp)>1:lp=camera_points(lp);ax.plot(lp[:,0],lp[:,2],'-x',ms=4,color=color)
    ax.set_xlabel('camera X (m)');ax.set_ylabel('camera optical Z (m)');ax.set_title('Camera X-Z projection · ground frame unverified');ax.grid(alpha=.2)
    if not cells or camera_from_world is None:ax.text(.5,.5,'World map unavailable',transform=ax.transAxes,ha='center')
    ax=axes[1,1];ax.axis('off');lines=[f"Frame: {frame.sample_id}",f"Status: {result['status']}",f"Pipeline: {result['latency_ms']:.2f} ms (drawing excluded)",f"Pose: {frame.pose_source or 'unavailable'}",f"Observed surface voxels: {len(cells)}",'Forecast horizons: 0.5 / 1 / 2 seconds','Unobserved cells stay unknown','Local research; no physical robot actuation']
    if result.get('control_command'):
        command=result['control_command'];lines.append(f"Sim control: {command['linear_mps']:.2f} m/s / {command['angular_rps']:.2f} rad/s");lines.append('Control reason: '+command['reason'])
    for t in display_tracks[:6]:
        p=t.get('position') or {};distance=p.get('surface_range_m');lines.append(f"{t['id']}: {distance:.2f} m | {t['behavior']}" if distance is not None else f"{t['id']}: pixel motion | range unavailable")
    if len(display_tracks)>6:lines.append(f"Details: top6 of{len(display_tracks)} tracks shown")
    ax.text(0,1,'\n'.join(lines),va='top',fontsize=10,family='monospace')
    if frame.sample_id.startswith('tum/'):attribution='TUM RGB-D · J. Sturm et al. · CC BY 4.0 · cvg.cit.tum.de/data/datasets/rgbd-dataset'
    elif frame.sample_id.startswith(('jaad/','jaad-v2/')):attribution='JAAD · Rasouli et al. · CC BY 4.0 · github.com/ykotseruba/JAAD'
    elif frame.sample_id.startswith('openloris'):attribution='OpenLORIS-Scene · real mobile-robot recording · local research · dataset redistribution terms unresolved'
    else:attribution='Controlled synthetic scene · independent candidate research · no physical robot actuation'
    fig.suptitle(attribution,fontsize=9,color='#425269');fig.savefig(path,dpi=120);plt.close(fig)
