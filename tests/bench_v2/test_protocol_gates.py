import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_calibration_uses_128_unique_train_images_and_never_val_test():
    export=module('export_v2','tools/export_v2.py')
    rows=[{'sample_id':f'train{i}','split':'train'} for i in range(200)]+[{'sample_id':'heldout','split':'test'}]*1000
    selected=export.calibration_selection(rows)
    assert len(selected)==128 and len({r['sample_id'] for r in selected})==128
    assert all(r['split']=='train' for r in selected)
    with pytest.raises(ValueError):export.calibration_selection(rows[:100])


def test_int8_default_blocks_accuracy_loss_and_worse_p95():
    export=module('export_v2','tools/export_v2.py')
    assert not export.int8_default_gate(.5,.48,20,10)['eligible']
    assert not export.int8_default_gate(.5,.499,20,21)['eligible']
    assert not export.int8_default_gate(.5,.499)['eligible']
    assert export.int8_default_gate(.5,.499,20,19)['eligible']


def test_formal_matrix_requires_quiescence_before_model_access(tmp_path):
    bench=module('benchmark_v2','tools/benchmark_v2.py')
    with pytest.raises(ValueError,match='quiescent'):
        bench.matrix(SimpleNamespace(output=str(tmp_path),quiescent=False))


def test_quality_accuracy_test_labels_remain_locked(tmp_path):
    from robot_reliability_bench.robotics_v2.quality_accuracy import evaluate
    with pytest.raises(ValueError,match='locked'):
        evaluate('missing-pack','missing-index',tmp_path/'out.json',split='test')


def test_schema_mutation_invalidates_complete_runtime_fingerprint(tmp_path,monkeypatch):
    import robot_model_sdk,robot_reliability_bench
    sdk=tmp_path/'sdk';sdk.mkdir();benchdir=tmp_path/'bench';benchdir.mkdir()
    (sdk/'__init__.py').write_text('');(benchdir/'__init__.py').write_text('')
    schema=sdk/'sensor.json';schema.write_text('{"version":1}')
    monkeypatch.setattr(robot_model_sdk,'__file__',str(sdk/'__init__.py'))
    monkeypatch.setattr(robot_reliability_bench,'__file__',str(benchdir/'__init__.py'))
    bench=module('benchmark_v2','tools/benchmark_v2.py');before=bench.runtime_sources()
    assert 'robot_model_sdk/sensor.json' in before
    schema.write_text('{"version":2}')
    assert bench.runtime_sources()!=before
