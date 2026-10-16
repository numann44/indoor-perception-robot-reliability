"""Closed-test selection guard checks without opening any real test data."""
import hashlib
import json
from pathlib import Path
import pytest
from indoor_perception_lab import quality_release, data_v2
from indoor_perception_lab.experiments import NYU40
from indoor_perception_lab.quality_training import PRESETS


def digest(value):
    return hashlib.sha256(value).hexdigest()


def pack(root,name,*,run_id,seed=42,configuration='S2',diagnostic=False,protocol='quality-v2'):
    p=root/'models/v2'/name;p.mkdir(parents=True)
    weights=('not-a-real-model:'+name).encode();(p/'weights.pt').write_bytes(weights)
    idx=root/'data/v2/nyu/index.jsonl';idx.parent.mkdir(parents=True,exist_ok=True)
    if not idx.exists():idx.write_text('{"sample_id":"audit/train","split":"train","scene_id":"audit-train"}\n')
    m={'task':'semantic','protocol_id':protocol,'run_id':run_id,'seed':seed,'configuration':configuration,
       'schema_version':1,'native_weights':'weights.pt','domain':'synthetic seal audit only',
       'class_names':NYU40,'input_size':[240,320],'mean':[.485,.456,.406],'std':[.229,.224,.225],
       'preprocessing_id':'quality_rgb_v2','postprocess':'resize_logits_then_argmax',
       'architecture':PRESETS[configuration]['architecture'],'steps':8,'selected_checkpoint_step':8,'best_epoch':0,
       'dataset_split_hash':digest(idx.read_bytes()),'sha256':digest(weights),'selection_value':.5,
       'selection_metric':'train_diagnostic' if diagnostic else 'validation_mIoU',
       'architecture_config':{'encoder_unfreeze':'last3'},'overfit_diagnostic':diagnostic}
    (p/'manifest.json').write_text(json.dumps(m));return p,m


def seal(root,winner,manifest):
    candidates=[{'configuration':k,'run_id':f'run-{k}','pack':f'models/v2/semantic-{k}-seed42',
                 'weights_sha256':manifest['sha256'],'manifest_sha256':digest((winner/'manifest.json').read_bytes()),
                 'dataset_split_hash':manifest['dataset_split_hash'],'validation_score':.5} for k in ['S1','S2','S3']]
    selected={**candidates[1],'pack':str(winner.relative_to(root)),'run_id':manifest['run_id']}
    lock={'protocol_id':'quality-v2','task':'semantic','selected':selected,'candidates':candidates,
          'seed_repeats':[42,43,44],'selection_split':'val','test_used_for_selection':False,
          'frozen_index_sha256':manifest['dataset_split_hash'],'dataset_split_hash':manifest['dataset_split_hash']}
    (root/'models/v2/selection-semantic.json').write_text(json.dumps(lock))


def forbidden_rows(*args,**kwargs):
    raise AssertionError('Unauthorized request reached test row loader')


def assert_blocked_before_test_loader(monkeypatch,tmp_path,candidate):
    winner,meta=pack(tmp_path,'semantic-S2-seed42',run_id='run-S2')
    seal(tmp_path,winner,meta)
    monkeypatch.setattr(quality_release,'root_path',lambda:tmp_path)
    monkeypatch.setattr(data_v2,'read_index_v2',forbidden_rows)
    with pytest.raises(ValueError):
        quality_release.evaluate_quality('semantic',candidate,'nyu',split='test')


def test_unselected_validation_candidate_cannot_access_new_test(monkeypatch,tmp_path):
    p,_=pack(tmp_path,'semantic-S1-seed42',run_id='run-S1',configuration='S1')
    assert_blocked_before_test_loader(monkeypatch,tmp_path,p)


def test_overfit_seed_prefix_does_not_unlock_new_test(monkeypatch,tmp_path):
    p,_=pack(tmp_path,'semantic-S2-seed9916-overfit',run_id='overfit',seed=9916,diagnostic=True)
    assert_blocked_before_test_loader(monkeypatch,tmp_path,p)


def test_unlisted_non_v2_pack_does_not_unlock_new_test(monkeypatch,tmp_path):
    p,_=pack(tmp_path,'unlisted-legacy-model',run_id='unlisted',protocol='legacy')
    assert_blocked_before_test_loader(monkeypatch,tmp_path,p)


def test_selection_does_not_trust_mismatched_weight_hash(monkeypatch,tmp_path):
    for key in ['S1','S2','S3']:
        p,m=pack(tmp_path,f'semantic-{key}-seed42',run_id=f'run-{key}',configuration=key)
        if key=='S2':(p/'weights.pt').write_bytes(b'changed-weights')
    monkeypatch.setattr(quality_release,'root_path',lambda:tmp_path)
    with pytest.raises(ValueError):quality_release.select_quality('semantic')


class ClosedTargetsWouldBeRead(Exception):
    """Sentinel boundary only: no actual target files are opened."""


def authorized_boundary(*args,**kwargs):
    raise ClosedTargetsWouldBeRead('Authorized artifact reached closed-target boundary; stopped before reading')


@pytest.mark.parametrize('seed',[42,43,44])
def test_selected_allowed_seed_reaches_only_sentinel(monkeypatch,tmp_path,seed):
    winner,m=pack(tmp_path,'semantic-S2-seed42',run_id='run-S2')
    seal(tmp_path,winner,m)
    candidate=winner if seed==42 else pack(tmp_path,f'semantic-S2-seed{seed}',run_id=f'run-S2-{seed}',seed=seed)[0]
    monkeypatch.setattr(quality_release,'root_path',lambda:tmp_path)
    monkeypatch.setattr(data_v2,'read_index_v2',authorized_boundary)
    with pytest.raises(ClosedTargetsWouldBeRead):
        quality_release.evaluate_quality('semantic',candidate,'nyu',split='test')


def test_explicit_frozen_baseline_reaches_only_sentinel(monkeypatch,tmp_path):
    winner,m=pack(tmp_path,'semantic-S2-seed42',run_id='run-S2');seal(tmp_path,winner,m)
    baseline,b=pack(tmp_path,'frozen-v1-semantic',run_id='baseline',protocol='legacy')
    path=tmp_path/'reports/v2/baseline-allowlist.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'baselines':[{'task':'semantic','pack':str(baseline.relative_to(tmp_path)),
                    'weights_sha256':b['sha256'],'manifest_sha256':digest((baseline/'manifest.json').read_bytes())}]}))
    monkeypatch.setattr(quality_release,'root_path',lambda:tmp_path)
    monkeypatch.setattr(data_v2,'read_index_v2',authorized_boundary)
    with pytest.raises(ClosedTargetsWouldBeRead):
        quality_release.evaluate_quality('semantic',baseline,'nyu',split='test')


def test_changed_index_blocks_before_target_boundary(monkeypatch,tmp_path):
    winner,m=pack(tmp_path,'semantic-S2-seed42',run_id='run-S2');seal(tmp_path,winner,m)
    (tmp_path/'data/v2/nyu/index.jsonl').write_text('changed index metadata')
    monkeypatch.setattr(quality_release,'root_path',lambda:tmp_path)
    monkeypatch.setattr(data_v2,'read_index_v2',forbidden_rows)
    with pytest.raises(ValueError,match='Dataset changed'):
        quality_release.evaluate_quality('semantic',winner,'nyu',split='test')


@pytest.mark.parametrize('changed_field',['class_names','preprocessing_id','postprocess'])
def test_semantic_identity_rejects_changed_class_or_operator_contract(tmp_path,changed_field):
    # Verify semantic meaning even when the fake artifact itself has a valid hash.
    from robot_model_sdk.manifest import validate_manifest
    p,m=pack(tmp_path,'semantic-S2-seed42',run_id='contract-oracle')
    if changed_field=='class_names':m[changed_field]=list(reversed(m[changed_field]))
    else:m[changed_field]='incorrect-operator'
    (p/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='quality-v2'):
        validate_manifest(p)


def test_missing_native_artifact_is_a_clean_validation_error(tmp_path):
    from robot_model_sdk.manifest import validate_manifest
    p,m=pack(tmp_path,'semantic-S2-seed42',run_id='missing-artifact')
    m.pop('native_weights');(p/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='nonempty relative path'):
        validate_manifest(p)


def test_interrupted_repeat_cannot_reach_closed_targets(monkeypatch,tmp_path):
    winner,m=pack(tmp_path,'semantic-S2-seed42',run_id='run-S2');seal(tmp_path,winner,m)
    p,m=pack(tmp_path,'semantic-S2-seed43',run_id='run-S2-43',seed=43)
    m['interrupted_for_resume_test']=True;(p/'manifest.json').write_text(json.dumps(m))
    monkeypatch.setattr(quality_release,'root_path',lambda:tmp_path)
    monkeypatch.setattr(data_v2,'read_index_v2',forbidden_rows)
    with pytest.raises(ValueError):
        quality_release.evaluate_quality('semantic',p,'nyu',split='test')
