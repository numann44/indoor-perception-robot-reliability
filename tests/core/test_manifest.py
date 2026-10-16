import hashlib,json
import pytest
from robot_model_sdk.manifest import validate_manifest,checked_artifact

def test_hash_and_path_boundaries(tmp_path):
    artifact=tmp_path/'model.onnx';artifact.write_bytes(b'identity')
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert checked_artifact(tmp_path,'model.onnx',digest)==artifact
    with pytest.raises(ValueError,match='hash mismatch'):checked_artifact(tmp_path,'model.onnx','0'*64)
    with pytest.raises(ValueError,match='inside'):checked_artifact(tmp_path,'../outside','0'*64)

def test_preprocessing_contract_rejects_zero_std(tmp_path):
    manifest={'schema_version':1,'task':'semantic','architecture':'mobilenet','domain':'test','class_names':['class']*40,'input_size':[240,320],'mean':[.5]*3,'std':[0,.2,.2]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='normalization'):validate_manifest(tmp_path)
