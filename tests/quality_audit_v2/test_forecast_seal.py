"""Forecast release guard audit using fake artifacts and sentinel-only targets."""
import hashlib
import json
from pathlib import Path

import pytest
from indoor_perception_lab import forecast_v2


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def synthetic_selection(root,closed=False):
    data=root/'data/v2/forecast/fixture/manifest.json'
    write_json(data,{'domain':'synthetic sentinel fixture','units':'metres'})
    dataset_hash=digest(data.read_bytes());bindings=[]
    for seed in (42,43,44):
        path=root/f'models/v2/forecast-fixture-seed{seed}-lr0.001-primary'
        path.mkdir(parents=True);(path/'weights.pt').write_bytes(f'fake-for-seed{seed}'.encode())
        manifest={'seed':seed,'run_id':f'synthetic-seed{seed}','config':{'domain':'fixture','seed':seed,'preset':'primary','lr':.001},
                  'best_step':10,'dataset_manifest_sha256':dataset_hash}
        write_json(path/'manifest.json',manifest)
        bindings.append({'seed':seed,'pack':str(path.relative_to(root)),
                         'checkpoint_sha256':digest((path/'weights.pt').read_bytes()),
                         'manifest_sha256':digest((path/'manifest.json').read_bytes()),
                         'dataset_manifest_sha256':dataset_hash,'lr':.001})
    lock={'domain':'fixture','test_closed':closed,'selected':bindings[0],
          'repeated_seeds':bindings[1:],'dataset_manifest_sha256':dataset_hash}
    write_json(root/'models/v2/selection-forecast-fixture.json',lock)
    return lock,root/bindings[0]['pack']


class AuthorizedTargetBoundary(Exception):
    """Authorized call reached loader; no real arrays or files were opened."""


def authorized_sentinel(*args,**kwargs):
    raise AuthorizedTargetBoundary('Sentinel stops before reading targets')


def forbidden_sentinel(*args,**kwargs):
    raise AssertionError('Unauthorized artifact reached closed target loader')


def request(root,pack):
    return {'domain':'fixture','checkpoint_sha256':digest((pack/'weights.pt').read_bytes())}


@pytest.mark.parametrize('seed',[42,43,44])
def test_forecast_canonical_allowed_seeds_reach_only_sentinel(monkeypatch,tmp_path,seed):
    lock,_=synthetic_selection(tmp_path)
    binding=next(b for b in [lock['selected'],*lock['repeated_seeds']] if b['seed']==seed)
    candidate=tmp_path/binding['pack']
    monkeypatch.setattr(forecast_v2,'root_path',lambda:tmp_path)
    monkeypatch.setattr(forecast_v2,'load_forecast_data',authorized_sentinel)
    with pytest.raises(AuthorizedTargetBoundary):
        forecast_v2.evaluate_forecast_v2(candidate,'fixture','test',request(tmp_path,candidate))


def test_forecast_self_supplied_hash_cannot_open_canonical_closed_test(monkeypatch,tmp_path):
    _,candidate=synthetic_selection(tmp_path,closed=True)
    monkeypatch.setattr(forecast_v2,'root_path',lambda:tmp_path)
    monkeypatch.setattr(forecast_v2,'load_forecast_data',forbidden_sentinel)
    with pytest.raises(PermissionError,match='testclosed'):
        forecast_v2.evaluate_forecast_v2(candidate,'fixture','test',request(tmp_path,candidate))


@pytest.mark.parametrize('change',['unselected','dataset','weights','diagnostic_seed','step_zero','winning_lr'])
def test_forecast_invalid_identity_blocks_before_targets(monkeypatch,tmp_path,change):
    _,candidate=synthetic_selection(tmp_path)
    manifest_path=candidate/'manifest.json';m=json.loads(manifest_path.read_text())
    if change=='unselected':
        import shutil
        other=tmp_path/'models/v2/forecast-fixture-seed42-lr0.0003-primary'
        shutil.copytree(candidate,other);candidate=other
    elif change=='dataset':
        (tmp_path/'data/v2/forecast/fixture/manifest.json').write_text('changed dataset')
    elif change=='weights':
        (candidate/'weights.pt').write_bytes(b'changed trained artifact')
    elif change=='diagnostic_seed':
        m['seed']=9916;write_json(manifest_path,m)
    elif change=='step_zero':
        m['best_step']=0;write_json(manifest_path,m)
    elif change=='winning_lr':
        m['config']['lr']=.0003;write_json(manifest_path,m)
    monkeypatch.setattr(forecast_v2,'root_path',lambda:tmp_path)
    monkeypatch.setattr(forecast_v2,'load_forecast_data',forbidden_sentinel)
    with pytest.raises(ValueError):
        forecast_v2.evaluate_forecast_v2(candidate,'fixture','test',request(tmp_path,candidate))


@pytest.mark.parametrize('change',['manifest_hash','seed_binding','domain_binding','config_seed_binding'])
def test_forecast_manifest_and_semantic_identity_are_bound(monkeypatch,tmp_path,change):
    lock,candidate=synthetic_selection(tmp_path)
    path=candidate/'manifest.json';m=json.loads(path.read_text())
    if change=='manifest_hash':m['unrecorded_edit']='tampered metadata'
    elif change=='seed_binding':m['seed']=43
    elif change=='domain_binding':m['config']['domain']='jaad'
    else:m['config']['seed']=43
    write_json(path,m)
    if change!='manifest_hash':
        # A matching content hash is insufficient when semantic identity is wrong.
        lock['selected']['manifest_sha256']=digest(path.read_bytes())
        write_json(tmp_path/'models/v2/selection-forecast-fixture.json',lock)
    monkeypatch.setattr(forecast_v2,'root_path',lambda:tmp_path)
    monkeypatch.setattr(forecast_v2,'load_forecast_data',forbidden_sentinel)
    with pytest.raises(ValueError):
        forecast_v2.evaluate_forecast_v2(candidate,'fixture','test',request(tmp_path,candidate))
