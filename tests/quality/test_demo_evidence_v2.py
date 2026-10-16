"""Controlled fixtures test replay evidence admission, not model or demo quality."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import pytest


def controlled_chapter(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('demo_evidence_oracle',Path(__file__).resolve().parents[2]/'tools/demo_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    media=tmp_path/'media';media.mkdir()
    image=media/'panel.png';cv2.imwrite(str(image),np.zeros((8,8,3),np.uint8))
    result=media/'result.json';result.write_text('{}')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    frame={'time_s':0.,'timestamp_ns':1540123456789012345,'sample_id':'controlled0','image':'panel.png','image_sha256':digest(image),'result_ref':'result.json','result_sha256':digest(result),'depth_available':False,'pose_source':None}
    c={'status':'verified','all_source_frames_processed':True,'playback_speed':1.,'chapter_id':'controlled','label':'Admission test only','source_duration_s':60.,'frames':[frame],'model_role':'Controlled admission fixture; no neural execution','processed_sensor_frames':1,'clock_policy':'controlled'}
    path=media/'chapter.json';path.write_text(json.dumps(c))
    return module,path,c


def test_validated_viewer_preserves_integer_sensor_clock(tmp_path,monkeypatch):
    module,path,_=controlled_chapter(tmp_path,monkeypatch)
    out=tmp_path/'demo/v2';module.build([path],out)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['chapters'][0]['frames'][0]['timestamp_ns']=='1540123456789012345'
    assert manifest['source_duration_s']==60.


@pytest.mark.parametrize('fault',['changed-image','missing-hash','nonzero-first-clock','out-of-range-clock'])
def test_replay_rejects_changed_or_invalid_evidence(tmp_path,monkeypatch,fault):
    module,path,c=controlled_chapter(tmp_path,monkeypatch)
    c=copy.deepcopy(c)
    if fault=='changed-image':(path.parent/'panel.png').write_bytes(b'changed')
    elif fault=='missing-hash':del c['frames'][0]['image_sha256']
    elif fault=='nonzero-first-clock':c['frames'][0]['time_s']=1.
    else:c['frames'].append({**c['frames'][0],'time_s':61.,'sample_id':'controlled1'})
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):module.build([path],tmp_path/'demo/v2')
