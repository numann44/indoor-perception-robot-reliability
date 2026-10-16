"""Dataset FrameSource with an explicit sensor-only field whitelist."""
from pathlib import Path
from .quality_evaluation import image_frame

class RGBDatasetSource:
    def __init__(self,root,rows):
        self.root=Path(root);self._closed=False
        allowed=('sample_id','rgb_path','timestamp','timestamp_ns','dataset')
        self.rows=[{k:r[k] for k in allowed if k in r} for r in rows]
    def __iter__(self):
        if self._closed:raise RuntimeError('FrameSource has been closed')
        for row in self.rows:
            if self._closed:break
            yield image_frame(self.root,row)
    def close(self):self._closed=True
    def metadata(self):
        return {'source':'RGBDatasetSource','datasets':sorted({r.get('dataset','unknown') for r in self.rows}),'frames':len(self.rows),'modalities':['rgb'],'annotation_access':False,'coordinate_system':'camera_optical','schema_version':1}
