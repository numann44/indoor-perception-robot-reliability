import cv2
import numpy as np
import pytest
from robot_model_sdk import Detection, ModelOutput
from indoor_perception_lab import data_v2
from indoor_perception_lab.quality_evaluation import evaluate_detection


def test_unsupported_coco_size_groups_are_unavailable(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path/'rgb.png'), np.zeros((80,80,3),np.uint8))
    row={'rgb_path':'rgb.png','sample_id':'oracle','scene_id':'oracle','dataset':'jaad'}
    monkeypatch.setattr(data_v2,'targets_for_row',lambda root,row:[{'bbox_xyxy':[0,0,64,64]}])
    class Perfect:
        def predict(self,frame):return ModelOutput(detections=[Detection([0,0,64,64],1.,1)])
    result,_=evaluate_detection(tmp_path,[row],Perfect())
    assert result['AP']==pytest.approx(1.)
    assert result['AP_medium']==pytest.approx(1.)
    assert result['AP_small'] is None and result['AP_large'] is None
    assert result['metric_status']['AP_small']=='unavailable'


def test_v2_mps_adapter_rejects_enabled_cpu_fallback(monkeypatch):
    from robot_model_sdk import models
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    monkeypatch.setattr(models,'validate_manifest',lambda *args:{'task':'semantic','protocol_id':'quality-v2'})
    with pytest.raises(ValueError,match='fallback'):
        models.TorchAdapter('/unused-oracle','mps')
