from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Iterator
import numpy as np

@dataclass(frozen=True)
class Calibration:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 1.0
    depth_convention: str = "z"
    def __post_init__(self):
        if not np.isfinite([self.fx,self.fy,self.cx,self.cy,self.depth_scale]).all() or min(self.fx,self.fy,self.depth_scale,self.width,self.height)<=0:
            raise ValueError("Invalid calibration")
        if self.depth_convention not in {"z","radial"}: raise ValueError("Unknown depth convention")
    def to_json(self): return dict(self.__dict__)

@dataclass
class SensorFrameV1:
    sample_id: str
    timestamp_ns: int
    rgb: np.ndarray
    depth: np.ndarray | None = None
    calibration: Calibration | None = None
    T_world_camera: np.ndarray | None = None
    pose_source: str | None = None
    def __post_init__(self):
        if not isinstance(self.timestamp_ns,int) or self.timestamp_ns<0: raise ValueError("Invalid nanosecond timestamp")
        if self.rgb.ndim!=3 or self.rgb.shape[2]!=3 or self.rgb.dtype!=np.uint8: raise ValueError("RGB must be uint8 HWC")
        h,w=self.rgb.shape[:2]
        if h<=0 or w<=0:raise ValueError("Empty RGB sensor")
        if self.depth is not None and self.depth.shape!=(h,w): raise ValueError("Depth must be registered to RGB")
        if self.calibration and (self.calibration.height,self.calibration.width)!=(h,w): raise ValueError("Calibration/image size mismatch")
        if self.T_world_camera is not None:
            T=np.asarray(self.T_world_camera)
            if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]): raise ValueError("Invalid rigid pose")
            if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-3): raise ValueError("Non-rigid pose")
    def to_json(self):
        return {"schema_version":1,"sample_id":self.sample_id,"timestamp_ns":self.timestamp_ns,"rgb_shape":list(self.rgb.shape),"depth_available":self.depth is not None,"calibration":self.calibration.to_json() if self.calibration else None,"T_world_camera":self.T_world_camera.tolist() if self.T_world_camera is not None else None,"pose_source":self.pose_source}

@dataclass
class Detection:
    bbox: list[float]
    score: float
    class_id: int
    def to_json(self): return {"bbox":[float(x) for x in self.bbox],"score":float(self.score),"class_id":int(self.class_id)}

@dataclass
class ModelOutput:
    detections: list[Detection] = field(default_factory=list)
    semantic: np.ndarray | None = None
    logits: np.ndarray | None = None
    forecast: np.ndarray | None = None
    def to_json(self):
        return {"schema_version":1,"detections":[d.to_json() for d in self.detections],"semantic_shape":list(self.semantic.shape) if self.semantic is not None else None,"forecast":self.forecast.tolist() if self.forecast is not None else None}

class ModelAdapter(Protocol):
    backend: str
    task: str
    last_timings_ms: dict[str,float]
    @classmethod
    def load(cls,model_spec:str,device:str="cpu") -> "ModelAdapter": ...
    def predict(self, frame: SensorFrameV1) -> ModelOutput: ...
    def close(self) -> None: ...

class FrameSource(Protocol):
    """Sensor-only stream; evaluator labels must live outside this interface."""
    def __iter__(self) -> Iterator[SensorFrameV1]: ...
    def close(self) -> None: ...
    def metadata(self) -> dict: ...

@dataclass(frozen=True)
class ControlCommandV1:
    timestamp_ns: int
    linear_mps: float
    angular_rps: float
    state: str
    reason: str
    source_sample_id: str | None = None
    observed_result_age_ms: float | None = None
    safety_stop: bool = False

    def __post_init__(self):
        if not isinstance(self.timestamp_ns,int) or self.timestamp_ns<0:
            raise ValueError('Control timestamp must be nonnegative integer nanoseconds')
        if not np.isfinite([self.linear_mps,self.angular_rps]).all():
            raise ValueError('Control speeds must be finite')
        if self.observed_result_age_ms is not None and (not np.isfinite(self.observed_result_age_ms) or self.observed_result_age_ms<0):
            raise ValueError('Result age must be finite and nonnegative')
        if self.safety_stop and (self.linear_mps!=0 or self.angular_rps!=0):
            raise ValueError('Safety stop cannot command movement')
        if not self.state or not self.reason:
            raise ValueError('Control state and reason are required')

    def to_json(self):
        return {'schema_version':1,**self.__dict__}

@dataclass
class PerceptionResultV1:
    sample_id: str
    timestamp_ns: int
    status: str
    tracks: list[dict]
    latency_ms: float
    detections: list[dict] = field(default_factory=list)
    semantic_shape: list[int] | None = None
    pose_source: str | None = None
    map: dict = field(default_factory=dict)
    detector_timings_ms: dict[str,float] = field(default_factory=dict)
    def __post_init__(self):
        if self.timestamp_ns<0 or not np.isfinite(self.latency_ms) or self.latency_ms<0:raise ValueError('Invalid result timestamp or latency')
        for track in self.tracks:
            if track.get('unit') not in {'pixels','metres'} or track.get('coordinate_frame') not in {'image','world'}:raise ValueError('Track must declare coordinates and units')
    def to_json(self):return {'schema_version':1,**self.__dict__}
