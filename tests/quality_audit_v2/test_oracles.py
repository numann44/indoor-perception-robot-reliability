"""Meaningful known-answer evaluator and closed-test guard checks."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quality_audit_v2',ROOT/'tools/quality_audit_v2.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_closed_test_guard_fails_before_target_read():
    with pytest.raises(ValueError,match='Closed-test'):
        audit.require_train_val([{'sample_id':'closed/test','split':'test','mask_path':'must-never-open'}])
    with pytest.raises(ValueError,match='Closed-test'):
        audit.read_rgb(ROOT,{'sample_id':'closed/outside','split':'outside_holdout','rgb_path':'must-never-open'})


def test_stratified_selection_is_fixed_and_has_both_partitions():
    rows=[{'sample_id':f'{split}/{scene}/{i}','split':split,'scene_id':scene,'frame_index':i}
          for split in ['train','val'] for scene in ['a','b'] for i in range(3)]
    chosen=audit.stratified_first(rows,4)
    assert {r['split'] for r in chosen}=={'train','val'}
    assert all(r['frame_index']==0 for r in chosen)
    assert chosen==audit.stratified_first(list(reversed(rows)),4)


def test_bilinear_hand_reference_discriminates_argmax_resize():
    logits=np.array([[[9,0],[9,0]],[[0,1],[0,1]]],float)
    mask=audit.reference_bilinear_mask(logits,4,4)
    assert np.array_equal(mask,np.array([[0,0,0,1]]*4,np.uint8))
    assert not np.array_equal(mask,np.repeat(np.repeat(logits.argmax(0),2,0),2,1))


def test_legacy_evaluator_known_answers(tmp_path):
    result=audit.oracle_checks(tmp_path,tmp_path/'oracle',evaluator='legacy')
    assert result['passed'],result['checks']
    assert result['variants']['perfect']['lab_detection']['AP']==pytest.approx(1.)
    assert result['variants']['wrong']['lab_semantic']['mIoU']==pytest.approx(0.)


def test_v2_evaluator_known_answers(tmp_path):
    pytest.importorskip('indoor_perception_lab.quality_evaluation')
    result=audit.oracle_checks(tmp_path,tmp_path/'oracle',evaluator='v2')
    assert result['passed'],result['checks']


def test_sdk_preprocess_known_answers():
    pytest.importorskip('robot_model_sdk.preprocessing')
    result=audit.preprocessing_checks()
    assert result['passed'],result['checks']


def test_many_training_scenes_do_not_starve_validation_panels():
    rows=[{'sample_id':f'{split}/{scene}/{i}','split':split,'scene_id':scene,'frame_index':i}
          for split,scenes in [('train',[f't{n}' for n in range(20)]),('val',[f'v{n}' for n in range(20)])]
          for scene in scenes for i in range(2)]
    chosen=audit.stratified_first(rows,16)
    assert sum(r['split']=='train' for r in chosen)==8
    assert sum(r['split']=='val' for r in chosen)==8
    assert len({r['scene_id'] for r in chosen})==16


def test_ignore_regions_mask_background_without_removing_positive_supervision():
    result=audit.ignore_gradient_checks()
    assert result['passed'],result['checks']


def test_scene_count_imbalance_still_balances_split_panels():
    rows=[{'sample_id':f'{split}/{scene}/{i}','split':split,'scene_id':scene,'frame_index':i}
          for split,scenes in [('train',[f't{n}' for n in range(20)]),('val',[f'v{n}' for n in range(3)])]
          for scene in scenes for i in range(10)]
    chosen=audit.stratified_first(rows,16)
    assert sum(r['split']=='train' for r in chosen)==8
    assert sum(r['split']=='val' for r in chosen)==8
    assert len({r['scene_id'] for r in chosen if r['split']=='val'})==3
