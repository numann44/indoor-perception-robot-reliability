import hashlib,json
import pytest
from robot_model_sdk.classes import NYU40_NAMES
from robot_model_sdk.manifest import validate_manifest,checked_artifact


def test_semantic_contract_rejects_class_permutation_and_wrong_preprocessing(tmp_path):
    artifact=tmp_path/'weights.pt';artifact.write_bytes(b'manifest-only-oracle')
    manifest={'schema_version':1,'protocol_id':'quality-v2','task':'semantic','architecture':'mobilenet_v3_multiscale_nyu40','domain':'oracle',
              'class_names':list(NYU40_NAMES),'input_size':[240,320],'mean':[.485,.456,.406],'std':[.229,.224,.225],
              'preprocessing_id':'quality_rgb_v2','postprocess':'resize_logits_then_argmax','native_weights':'weights.pt','sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
    def write(): (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    write();assert validate_manifest(tmp_path)['class_names'][30]=='person'
    manifest['class_names'][0],manifest['class_names'][1]=manifest['class_names'][1],manifest['class_names'][0];write()
    with pytest.raises(ValueError,match='class order'):validate_manifest(tmp_path)
    manifest['class_names']=list(NYU40_NAMES);manifest['preprocessing_id']='rgb_bgr_oracle';write()
    with pytest.raises(ValueError,match='preprocessing'):validate_manifest(tmp_path)
    manifest['preprocessing_id']='quality_rgb_v2';manifest['mean']=[.5,.5,.5];write()
    with pytest.raises(ValueError,match='normalization'):validate_manifest(tmp_path)


def test_missing_artifact_path_is_rejected_as_validation_error(tmp_path):
    with pytest.raises(ValueError,match='path'):checked_artifact(tmp_path,None,'0'*64)
