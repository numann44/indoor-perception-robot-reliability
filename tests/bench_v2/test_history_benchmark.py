from types import SimpleNamespace
import hashlib,json
import numpy as np
import pytest
from robot_reliability_bench.robotics_v2.history_benchmark import history_inputs,ControlledCV,worker,matrix


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def data(tmp_path):
    h=tmp_path/'sensor.npz';np.savez_compressed(h,history=np.zeros((2,11,2),np.float32))
    rows=tmp_path/'index.jsonl';rows.write_text('\n'.join(json.dumps({'sample_id':f'causal-{i}-suffix_12','split':'val','anchor_time_s':1.,'target_id_evaluator_only':'must_not_be_input'}) for i in range(2)))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'domain':'test','units':'metres','coordinate_system':'world_XZ_planar','history_source':'observed_past','split_files':{'val':{'history_file':h.name,'history_sha256':sha(h),'index_file':rows.name,'index_sha256':sha(rows),'future_file':'must_never_open.npz'}}}))
    return manifest


def test_only_sensor_history_archive_opened_and_evaluator_fields_removed(tmp_path,monkeypatch):
    manifest=data(tmp_path);opened=[];real=np.load
    def tracked(p,*a,**k):opened.append(str(p));return real(p,*a,**k)
    monkeypatch.setattr(np,'load',tracked)
    histories,source,_=history_inputs(manifest)
    assert len(opened)==1 and opened[0].endswith('sensor.npz')
    assert histories[0]['sample_id']=='causal-0-suffix_12' and 'target_id_evaluator_only' not in histories[0]
    assert source['labels_or_futures_opened'] is False


def test_future_array_and_future_time_are_rejected(tmp_path):
    manifest=data(tmp_path);meta=json.loads(manifest.read_text());np.savez_compressed(tmp_path/'sensor.npz',history=np.zeros((2,11,2)),future=np.zeros((2,20,2)))
    meta['split_files']['val']['history_sha256']=sha(tmp_path/'sensor.npz');manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='only history'):history_inputs(manifest)
    with pytest.raises(ValueError,match='Future'):ControlledCV().predict_timed_history(np.linspace(0,1.1,11),np.zeros((11,2)),1.)


def test_actual_controlled_worker_counts100_and_separate_ram_without_neural_claim():
    result=worker({'model_pack':'history-fixture','threads':1,'data_manifest':None,'split':'val','device':'cpu','warmup':10,'samples':100,'repeat':0})
    assert result['status']=='verified' and result['causal_adapter_latency']['count']==100
    assert result['unique_timed_windows']==100 and result['memory_pass']['predictions']==100
    assert result['memory_pass']['separate_from_timing'] and not result['label_or_future_input']
    assert result['model_forward_latency']['status']=='unavailable'


def test_real_history_timing_requires_quiescence_before_pack_or_source_access():
    with pytest.raises(ValueError,match='quiescence'):
        matrix(SimpleNamespace(model_pack='missing',quiescent=False),'/missing/tool.py')


def test_model_history_coordinate_and_unit_mismatch_fails_closed(monkeypatch):
    import robot_model_sdk
    from robot_reliability_bench.robotics_v2.history_benchmark import load_history_adapter
    class Adapter:
        task='forecast';metadata={'units':'metres','coordinate_system':'world_XZ_planar'};closed=False
        def close(self):self.closed=True
    adapter=Adapter();monkeypatch.setattr(robot_model_sdk,'load_adapter',lambda *a,**k:adapter)
    with pytest.raises(ValueError,match='coordinate frame or units'):
        load_history_adapter('not-opened-pack',{'units':'normalized_image','coordinate_system':'image_xy_divided_by_width_height'},'cpu')
    assert adapter.closed
