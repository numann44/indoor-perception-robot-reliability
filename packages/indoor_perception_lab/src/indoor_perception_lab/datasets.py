from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from robot_model_sdk import SensorFrameV1,Calibration

def read_index(root,dataset,split=None,index_path=None):
    p=Path(index_path) if index_path else Path(root)/"data"/dataset/"index.jsonl"
    rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if split is not None:rows=[r for r in rows if r["split"]==split]
    groups={s:{r["scene_id"] for r in rows if r["split"]==s} for s in {r["split"] for r in rows}}
    for s,gs in groups.items():
        for t,gt in groups.items():
            if s!=t and gs & gt:raise ValueError("Train/validation/test scene overlap")
    return rows

def sensor_from_row(root,row):
    # Deliberately whitelist sensor fields: target paths/labels are never read.
    import cv2
    from scipy.spatial.transform import Rotation
    root=Path(root);img=cv2.imread(str(root/row["rgb_path"]))
    if img is None:raise ValueError("Unreadable RGB sensor")
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);depth=None;c=None;T=None
    if row.get("calibration_path"):
        meta=json.loads((root/row["calibration_path"]).read_text());keys={k:meta[k] for k in ["fx","fy","cx","cy","width","height","depth_scale"]};keys["depth_convention"]="z";c=Calibration(**keys)
        if row.get("depth_path"):depth=cv2.imread(str(root/row["depth_path"]),cv2.IMREAD_UNCHANGED)
        pose=row.get("pose_xyz_xyzw")
        if pose is not None:T=np.eye(4);T[:3,3]=pose[:3];T[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix()
    t=row.get("timestamp")
    if t is None:t=row.get("frame_index",0)/10
    return SensorFrameV1(row["sample_id"],int(round(t*1e9)),rgb,depth,c,T,row.get("pose_source"))

def jaad_targets(root,row):
    obj=json.loads((Path(root)/row["targets_path"]).read_text());return obj["frames"].get(str(row["frame_index"]),[])

class SemanticDataset:
    def __init__(self,root,rows,augment=False):self.root=Path(root);self.rows=rows;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import torch,cv2
        row=self.rows[i];rgb=cv2.cvtColor(cv2.imread(str(self.root/row["rgb_path"])),cv2.COLOR_BGR2RGB);mask=cv2.imread(str(self.root/row["mask_path"]),cv2.IMREAD_UNCHANGED)
        rgb=cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA);mask=cv2.resize(mask,(320,240),interpolation=cv2.INTER_NEAREST)
        if self.augment and np.random.rand()<.5:rgb=rgb[:,::-1].copy();mask=mask[:,::-1].copy()
        x=rgb.astype(np.float32)/255;x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(mask.astype(np.int64))

class DetectionDataset:
    def __init__(self,root,rows,augment=False):self.root=Path(root);self.rows=rows;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import torch,cv2
        row=self.rows[i];rgb=cv2.cvtColor(cv2.imread(str(self.root/row["rgb_path"])),cv2.COLOR_BGR2RGB);targets=jaad_targets(self.root,row);boxes=np.array([t["bbox_xyxy"] for t in targets if not t["ignore"]],np.float32).reshape(-1,4)
        if self.augment and np.random.rand()<.5:
            rgb=rgb[:,::-1].copy();old=boxes.copy();boxes[:,0]=rgb.shape[1]-old[:,2];boxes[:,2]=rgb.shape[1]-old[:,0]
        box=torch.as_tensor(boxes);return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,{"boxes":box,"labels":torch.ones(len(boxes),dtype=torch.int64)}

def jaad_video_frames(root,sequence,sample_rate_hz=10):
    """Whitelisted recorded video sensor stream; never open targets or identity labels."""
    import cv2
    from robot_model_sdk import SensorFrameV1
    rows=read_index(root,'jaad');row=next((r for r in rows if r['sequence_id']==sequence),None)
    if row is None:raise ValueError('JAAD sequence not in the frozen subset')
    cap=cv2.VideoCapture(str(Path(root)/row['video_path']));fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or count<1:raise ValueError('Invalid video sensor stream')
    selected=set(round(t*fps) for t in np.arange(0,(count-1)/fps+1e-9,1/sample_rate_hz))
    try:
        for index in range(count):
            ok,bgr=cap.read()
            if not ok:raise ValueError(f'Video decode failed at {index}')
            if index not in selected:continue
            h,w=bgr.shape[:2];bgr=cv2.resize(bgr,(640,round(h*640/w)));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            yield SensorFrameV1(f'jaad/{sequence}/{index:06d}',int(round(index/fps*1e9)),rgb)
    finally:cap.release()
