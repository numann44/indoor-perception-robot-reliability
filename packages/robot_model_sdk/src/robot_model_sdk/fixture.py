"""Physical pinhole fixture sensors. Ground truth is never passed to adapters."""
import numpy as np
from .contracts import Calibration,SensorFrameV1

def fixture_objects(t):
    # Opaque front-facing surfaces; RGB and depth share the same z-buffer.
    return [{"id":"block-a","position":[-0.8+0.8*np.sin(.2*t),0.0,3.2-0.2*np.sin(.15*t)],"size":[0.42,0.70],"color":[230,50,40]},
            {"id":"block-b","position":[0.9-0.8*np.sin(.14*t),0.12,4.1],"size":[0.38,0.60],"color":[35,210,55]}]

def fixture_frame(i,fps=10,width=160,height=120,variant="default",objects=None):
    t=i/fps;c=Calibration(140,140,(width-1)/2,(height-1)/2,width,height)
    rgb=np.full((height,width,3),[35,40,50],np.uint8);depth=np.full((height,width),6.0,np.float32)
    # Floor plane y=0.7, valid only for positive camera-y rays.
    ys=(np.arange(height)-c.cy)/c.fy
    for v in range(height):
        if ys[v]>0:
            z=0.7/ys[v]
            if z<6:depth[v]=z;rgb[v]=[100,110,120]
    for ob in fixture_objects(t) if objects is None else objects:
        if variant=="occlusion" and ob["id"]=="block-a" and 12<=i<=16:continue
        if variant=="entry_exit" and ob["id"]=="block-b" and i<10:continue
        x,y,z=ob["position"];ow,oh=ob["size"]
        u0=int(np.floor(c.fx*(x-ow/2)/z+c.cx));u1=int(np.ceil(c.fx*(x+ow/2)/z+c.cx))
        v0=int(np.floor(c.fy*(y-oh/2)/z+c.cy));v1=int(np.ceil(c.fy*(y+oh/2)/z+c.cy))
        u0,u1=max(0,u0),min(width,u1);v0,v1=max(0,v0),min(height,v1)
        if u1<=u0 or v1<=v0:continue
        region=depth[v0:v1,u0:u1];visible=region>z
        region[visible]=z;rgb[v0:v1,u0:u1][visible]=ob["color"]
    return SensorFrameV1(f"fixture-{variant}-{i:05d}",int(round(t*1e9)),rgb,depth,c,np.eye(4),"synthetic-known-pose")

def fixture_frames(n=120,fps=10):
    for i in range(n):yield fixture_frame(i,fps)
