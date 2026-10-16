from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
from .contracts import Detection,ModelOutput
from .manifest import validate_manifest
from .history import prepare_motion,finish_motion,resample_causal

class FixtureAdapter:
    backend="native";task="detection"
    def __init__(self):self.last_timings_ms={};self.metadata={"domain":"synthetic colored planar blocks","learned":False}
    @classmethod
    def load(cls,model_spec='fixture',device='cpu'):return cls()
    def predict(self,frame):
        import cv2
        start=time.perf_counter();rgb=frame.rgb
        mask=(((rgb[:,:,0]>160)&(rgb[:,:,1]<100))|((rgb[:,:,1]>150)&(rgb[:,:,0]<100))).astype(np.uint8)
        t=time.perf_counter();n,l,stats,cent=cv2.connectedComponentsWithStats(mask)
        ds=[Detection([float(x),float(y),float(x+w),float(y+h)],1.0,1) for x,y,w,h,area in stats[1:] if area>=4]
        end=time.perf_counter();self.last_timings_ms={"preprocess":(t-start)*1e3,"model":(end-t)*1e3,"postprocess":0.0}
        return ModelOutput(detections=ds)
    def close(self):pass

def semantic_network(freeze_encoder=True):
    import torch
    from torchvision.models import mobilenet_v3_large
    class SemanticNet(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=mobilenet_v3_large(weights=None).features
            self.decoder=torch.nn.Sequential(torch.nn.Conv2d(960,64,1,bias=False),torch.nn.BatchNorm2d(64),torch.nn.ReLU(),torch.nn.Conv2d(64,40,1))
            self.freeze_encoder=freeze_encoder
            if freeze_encoder:
                for p in self.encoder.parameters():p.requires_grad=False
        def train(self,mode=True):
            super().train(mode)
            if self.freeze_encoder:self.encoder.eval()
            return self
        def forward(self,x):
            return torch.nn.functional.interpolate(self.decoder(self.encoder(x)),size=x.shape[-2:],mode="bilinear",align_corners=False)
    return SemanticNet()

def motion_network(history=11,future=20,hidden=64):
    import torch
    return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(history*2,hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,hidden),torch.nn.ReLU(),torch.nn.Linear(hidden,future*2),torch.nn.Unflatten(1,(future,2)))

def _sync(device):
    if str(device).startswith("mps"):
        import torch;torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        import torch;torch.cuda.synchronize()

class TorchAdapter:
    backend="native"
    @classmethod
    def load(cls,model_spec,device='cpu'):return cls(model_spec,device)
    def __init__(self,pack,device="cpu"):
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        self.pack=Path(pack);self.metadata=validate_manifest(self.pack,'native');self.task=self.metadata["task"];self.device=device;self.last_timings_ms={}
        if (self.metadata.get('protocol_id')=='quality-v2' or self.metadata.get('architecture')=='residual_cv_mlp_2x64') and str(device).startswith('mps'):
            import os
            if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('quality-v2 requires MPS CPU fallback to be disabled')
        wf=self.pack/self.metadata["native_weights"]
        if self.metadata.get("sha256") and hashlib.sha256(wf.read_bytes()).hexdigest()!=self.metadata["sha256"]:raise ValueError("Checkpoint hash mismatch")
        if self.task=="detection":
            if self.metadata['architecture'].endswith('quality_v2'):
                from .quality_models import detection_model
                self.model=detection_model(self.metadata['architecture'],num_classes=self.metadata.get('num_classes',2))
            else:self.model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=self.metadata.get("num_classes",91))
        elif self.task=="semantic":
            if self.metadata['architecture']=='mobilenet_v3_multiscale_nyu40':
                from .quality_models import MultiScaleSemanticNet
                self.model=MultiScaleSemanticNet()
            else:self.model=semantic_network()
        elif self.task=="forecast":
            if self.metadata['architecture']=='residual_cv_mlp_2x64':
                from .quality_models import residual_motion_network
                self.model=residual_motion_network(**self.metadata.get('network',{}))
            else:self.model=motion_network(**self.metadata.get("network",{}))
        else:raise ValueError("Unsupported model task")
        obj=torch.load(wf,map_location="cpu",weights_only=True)
        self.model.load_state_dict(obj["model"] if "model" in obj else obj)
        self.model.to(device).eval()
    def predict(self,frame):
        import torch,cv2
        a=time.perf_counter();rgb=frame.rgb;h,w=rgb.shape[:2]
        if self.task=="forecast":raise ValueError("Forecast model requires predict_history; an RGB frame is insufficient")
        if self.task=="semantic":
            ih,iw=self.metadata.get("input_size",[240,320])
            if self.metadata.get('preprocessing_id')=='quality_rgb_v2':
                from .preprocessing import quality_rgb
                x=torch.from_numpy(quality_rgb(rgb,(ih,iw),self.metadata['mean'],self.metadata['std'])).unsqueeze(0).to(self.device)
            else:
                x=cv2.resize(rgb,(iw,ih)).astype(np.float32)/255
                mean=np.array(self.metadata.get("mean",[.485,.456,.406]),np.float32);std=np.array(self.metadata.get("std",[.229,.224,.225]),np.float32)
                x=torch.from_numpy(((x-mean)/std).transpose(2,0,1).copy()).unsqueeze(0).to(self.device)
        else:x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float().to(self.device)/255
        _sync(self.device);b=time.perf_counter()
        with torch.inference_mode():y=self.model(x if self.task=="semantic" else [x])
        _sync(self.device);c=time.perf_counter()
        if self.task=="semantic":
            logits=y[0].cpu().numpy()
            if self.metadata.get('postprocess')=='resize_logits_then_argmax':
                from .preprocessing import quality_logits_to_mask
                semantic=quality_logits_to_mask(logits,h,w)
            else:semantic=cv2.resize(logits.argmax(0).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
            out=ModelOutput(semantic=semantic,logits=logits)
        else:
            d=y[0];boxes=d["boxes"].cpu().numpy();scores=d["scores"].cpu().numpy();labels=d["labels"].cpu().numpy()
            out=ModelOutput(detections=[Detection(b.tolist(),float(s),int(l)) for b,s,l in zip(boxes,scores,labels)])
        end=time.perf_counter();self.last_timings_ms={"preprocess":(b-a)*1e3,"model":(c-b)*1e3,"postprocess":(end-c)*1e3}
        return out
    def predict_history(self,history):
        import torch
        if self.task!='forecast':raise ValueError('History inference requires a forecast model')
        x,origin,scale,single=prepare_motion(history,self.metadata);start=time.perf_counter();x=torch.as_tensor(x,device=self.device);_sync(self.device)
        with torch.inference_mode():y=self.model(x).cpu().numpy()
        _sync(self.device);self.last_timings_ms={'total':(time.perf_counter()-start)*1000}
        return finish_motion(y,origin,scale,single,self.metadata)
    def predict_timed_history(self,times,positions,anchor_time=None):
        history=resample_causal(times,positions,anchor_time)
        return None if history is None else self.predict_history(history)
    def close(self):del self.model

class OnnxAdapter:
    backend="onnx"
    @classmethod
    def load(cls,model_spec,device='cpu'):return cls(model_spec,device)
    def __init__(self,pack,device="cpu"):
        import onnxruntime as ort
        if device!='cpu':raise ValueError('This ONNX package is verified only with CPUExecutionProvider')
        self.pack=Path(pack);self.metadata=validate_manifest(self.pack,'onnx');self.task=self.metadata["task"];self.last_timings_ms={}
        import os
        threads=int(os.environ.get('ROBOT_MODEL_THREADS','4'))
        if threads<1:raise ValueError('ROBOT_MODEL_THREADS must be positive')
        opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(self.pack/self.metadata.get("onnx","model.onnx")),sess_options=opts,providers=["CPUExecutionProvider"])
        self.metadata={**self.metadata,'runtime_intra_op_threads':self.session.get_session_options().intra_op_num_threads,'runtime_inter_op_threads':self.session.get_session_options().inter_op_num_threads,'runtime_providers':self.session.get_providers()}
    def predict(self,frame):
        import cv2
        if self.task!="semantic":raise ValueError("Only semantic frame ONNX adapter is implemented; forecast uses predict_history")
        a=time.perf_counter();h,w=frame.rgb.shape[:2];ih,iw=self.metadata.get("input_size",[240,320]);x=cv2.resize(frame.rgb,(iw,ih)).astype(np.float32)/255
        mean=np.array(self.metadata.get("mean",[.485,.456,.406]),np.float32);std=np.array(self.metadata.get("std",[.229,.224,.225]),np.float32)
        if self.metadata.get('preprocessing_id')=='quality_rgb_v2':
            from .preprocessing import quality_rgb
            x=quality_rgb(frame.rgb,(ih,iw),mean,std)[None]
        else:x=((x-mean)/std).transpose(2,0,1)[None].copy()
        b=time.perf_counter();logits=self.session.run(None,{self.session.get_inputs()[0].name:x})[0][0];c=time.perf_counter()
        if self.metadata.get('postprocess')=='resize_logits_then_argmax':
            from .preprocessing import quality_logits_to_mask
            semantic=quality_logits_to_mask(logits,h,w)
        else:semantic=cv2.resize(logits.argmax(0).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
        out=ModelOutput(semantic=semantic,logits=logits);end=time.perf_counter();self.last_timings_ms={"preprocess":(b-a)*1e3,"model":(c-b)*1e3,"postprocess":(end-c)*1e3};return out
    def predict_history(self,history):
        if self.task!='forecast':raise ValueError('History inference requires a forecast model')
        x,origin,scale,single=prepare_motion(history,self.metadata)
        return finish_motion(self.session.run(None,{self.session.get_inputs()[0].name:x})[0],origin,scale,single,self.metadata)
    def predict_timed_history(self,times,positions,anchor_time=None):
        history=resample_causal(times,positions,anchor_time)
        return None if history is None else self.predict_history(history)
    def close(self):del self.session

def load_adapter(model_spec="fixture",backend="native",device="cpu"):
    if str(model_spec)=="fixture":
        if backend!="native":raise ValueError("Fixture has no ONNX model")
        return FixtureAdapter()
    pack=Path(model_spec)
    if pack.is_file():pack=pack.parent
    validate_manifest(pack,backend)
    return OnnxAdapter(pack,device) if backend=="onnx" else TorchAdapter(pack,device)
