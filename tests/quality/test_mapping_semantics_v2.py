import numpy as np
from robot_model_sdk import SensorFrameV1,Calibration,ModelOutput
from robot_model_sdk.preprocessing import quality_logits_to_mask
from indoor_perception_lab.geometry import SpatialMap
from indoor_perception_lab.pipeline import PerceptionPipeline

def sensor():
    return SensorFrameV1('fixture/sensor',1,np.zeros((18,18,3),np.uint8),np.full((18,18),2.,np.float32),Calibration(18,18,8.5,8.5,18,18),np.eye(4),'synthetic-known-pose')

def test_semantic_person_never_written_as_static_occupancy():
    f=sensor();m=SpatialMap();status=m.update(f,np.full((18,18),30,np.uint8))
    assert status['updated'] and not m.cells
    assert m.to_json()['representation']=='observed surface occupancy; unobserved cells unknown'

def test_probability_fusion_uses_original_pixel_logit_interpolation():
    f=sensor();logits=np.array([[[1,1],[-9,-9]],[[0,0],[1,1]]],np.float32);mask=quality_logits_to_mask(logits,18,18)
    class Detector:
        task='detection';last_timings_ms={};metadata={}
        def predict(self,frame):return ModelOutput(detections=[])
        def close(self):pass
    class Segmenter:
        def predict(self,frame):return ModelOutput(semantic=mask,logits=logits)
        def close(self):pass
    p=PerceptionPipeline(Detector(),Segmenter());p.process(f)
    for u in [0,6,12]:
        key=tuple(np.floor(np.array([(u-8.5)*2/18,(6-8.5)*2/18,2])/.1).astype(int))
        cell=p.spatial_map.cells[key]
        assert mask[6,u]==1 and np.argmax(cell['probability_sums'])==1
    p.close()
