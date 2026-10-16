import cv2
import numpy as np
import pytest
from indoor_perception_lab.sensor_sources_v2 import RGBDatasetSource

def test_dataset_source_annotation_paths_never_opened(tmp_path):
    cv2.imwrite(str(tmp_path/'rgb.png'),np.zeros((4,6,3),np.uint8))
    row={'sample_id':'image1','rgb_path':'rgb.png','timestamp_ns':12,'dataset':'nyu','mask_path':'FORBIDDEN','targets_path':'FORBIDDEN','ground_truth_boxes':[1],'future_trajectory':[2]}
    source=RGBDatasetSource(tmp_path,[row]);frame=next(iter(source))
    assert frame.timestamp_ns==12 and frame.rgb.shape==(4,6,3)
    assert not any('ground_truth' in k or k in {'mask_path','targets_path','future_trajectory'} for k in source.rows[0])
    assert not source.metadata()['annotation_access']
    source.close()
    with pytest.raises(RuntimeError):next(iter(source))
