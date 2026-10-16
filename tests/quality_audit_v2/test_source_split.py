"""Reject mistaken NYU source split metadata before sensor/target loading."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat


path=Path(__file__).resolve().parents[2]/'tools/quality_audit_v2.py'
spec=importlib.util.spec_from_file_location('source_split_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def official_split(root):
    p=root/'data/v2/nyu/source/splits.mat';p.parent.mkdir(parents=True)
    savemat(p,{'trainNdxs':np.array([[1],[2]]),'testNdxs':np.array([[3]])})


def test_train_val_both_derive_from_official_training_indices(tmp_path):
    official_split(tmp_path)
    rows=[{'dataset':'nyu','sample_id':'synthetic/1','split':'train','source_index_1based':1},
          {'dataset':'nyu','sample_id':'synthetic/2','split':'val','source_index_1based':2}]
    assert audit.source_training_membership(tmp_path,rows)['checked']==2


def test_mislabeled_source_test_index_rejected_before_missing_image_path(tmp_path):
    official_split(tmp_path)
    row={'dataset':'nyu','sample_id':'synthetic/not-train','split':'train','source_index_1based':3,
         'rgb_path':'MUST_NOT_OPEN.png','mask_path':'MUST_NOT_OPEN_MASK.png'}
    with pytest.raises(ValueError,match='official training source index'):
        audit.source_training_membership(tmp_path,[row])


def test_detection_source_audit_serializes_scaled_numpy_box_counters(tmp_path):
    rows,_,_=audit.make_oracle_fixture(tmp_path,tmp_path/'oracle-data')
    index=tmp_path/'index.jsonl';index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out=tmp_path/'audit-output'
    result=audit.source_audit(tmp_path,index,out,kind='detection',panels=2)
    saved=json.loads((out/'source-audit.json').read_text())
    assert result['status']=='passed'
    assert saved['detection_scale_diagnostic']['boxes']==16
    assert all(isinstance(x,int) for x in saved['detection_scale_diagnostic'].values())
