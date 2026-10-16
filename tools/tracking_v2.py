"""Validation-only tracker choice, followed by sealed external identity evaluation."""
from __future__ import annotations
import argparse,json,hashlib,time,datetime
from pathlib import Path
import numpy as np
import cv2
from robot_model_sdk import SensorFrameV1,load_adapter,Detection
from indoor_perception_lab.experiments import root_path,json_write,sha
from indoor_perception_lab.data_v2 import read_index_v2,targets_for_row
from indoor_perception_lab.tracking_v2 import TRACKING_PRESETS,make_tracker,make_legacy_tracker
from indoor_perception_lab.quality_release import ensure_test_seal
from indoor_perception_lab.quality_training import QualityRun
from robot_reliability_bench.metrics import tracking_metrics


def sensor_stream(rows):
    root=root_path();by_clip={}
    for row in rows:by_clip.setdefault(row['sequence_id'],row)
    for clip,row in sorted(by_clip.items()):
        cap=cv2.VideoCapture(str(root/row['video_path']));fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps<=0:raise ValueError('Invalid video FPS')
        chosen={int(round(t*fps)) for t in np.arange(0,count/fps,.1)}
        try:
            for i in range(count):
                ok,bgr=cap.read()
                if not ok:raise ValueError('Recorded video decode failed')
                if i not in chosen:continue
                sampled={**row,'frame_index':i,'sample_id':f'jaad-v2/{clip}/{i:06d}','timestamp':i/fps}
                yield sampled,SensorFrameV1(sampled['sample_id'],int(round(i/fps*1e9)),cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        finally:cap.release()


def build_validation_cache(pack,device='mps'):
    root=root_path();rows=read_index_v2('jaad','val');out=root/'reports/v2/tracking';out.mkdir(parents=True,exist_ok=True);adapter=load_adapter(pack,device=device);items=[]
    try:
        for row,frame in sensor_stream(rows):
            detections=adapter.predict(frame).detections
            items.append({'sample_id':frame.sample_id,'sequence_id':row['sequence_id'],'timestamp_ns':frame.timestamp_ns,'detections':[d.to_json() for d in detections if d.class_id==1],'targets':[{'bbox':t['bbox_xyxy'],'track_id':row['sequence_id']+'/'+str(t['track_id']),'class_id':1,'ignore':t.get('ignore',False)} for t in targets_for_row(root,row)]})
    finally:adapter.close()
    record={'split':'val','sampling_hz':10,'model_pack':str(Path(pack).relative_to(root)),'weights_sha256':sha(Path(pack)/'weights.pt'),'index_sha256':sha(root/'data/v2/jaad/index.jsonl'),'items':items,'test_used':False}
    json_write(out/'validation-cache.json',record);return record


def evaluate_cache(items,configuration):
    params={'low_score':.3} if configuration=='legacy-v1' else TRACKING_PRESETS[configuration];trackers={};preds={};targets={}
    for item in items:
        clip=item['sequence_id']
        if clip not in trackers:trackers[clip]=make_legacy_tracker() if configuration=='legacy-v1' else make_tracker(configuration)
        tracker=trackers[clip]
        ds=[Detection(d['bbox'],d['score'],d['class_id']) for d in item['detections'] if d['score']>=params['low_score']]
        tracks=tracker.update(ds,item['timestamp_ns']);sid=item['sample_id']
        preds[sid]=[{'bbox':t.bbox,'track_id':clip+'/'+t.id,'class_id':1} for t in tracks if t.observed]
        targets[sid]=item['targets']
    return tracking_metrics(preds,targets),preds


def select_tracking(pack,device='mps'):
    root=root_path();model_lock=json.loads((root/'models/v2/selection-detection.json').read_text())
    ensure_test_seal('detection',pack,'jaad')
    if Path(pack).resolve()!=(root/model_lock['selected']['pack']).resolve() or sha(Path(pack)/'weights.pt')!=model_lock['selected']['weights_sha256'] or sha(Path(pack)/'manifest.json')!=model_lock['selected']['manifest_sha256']:raise ValueError('Tracker validation must use exact selected fixed detector')
    previous=root/'models/v2/selection-tracking.json'
    if previous.exists():
        frozen=json.loads(previous.read_text())
        if frozen['detector_weights_sha256']!=sha(Path(pack)/'weights.pt') or frozen['detector_manifest_sha256']!=sha(Path(pack)/'manifest.json') or frozen['validation_index_sha256']!=sha(root/'data/v2/jaad/index.jsonl') or frozen['tracker_source_sha256']!=sha(root/'packages/indoor_perception_lab/src/indoor_perception_lab/tracking_v2.py'):raise ValueError('Frozen tracker selection inputs changed')
        return frozen
    run=QualityRun('quality-v2-tracking-select',{'protocol_id':'quality-v2','device':device,'seed':42,'selection_split':'val','model_pack':str(Path(pack).relative_to(root)),'checkpoint_sha256':sha(Path(pack)/'weights.pt')},root/'data/v2/jaad/index.jsonl')
    record=build_validation_cache(pack,device);configs=[]
    for name in TRACKING_PRESETS:
        result,pred=evaluate_cache(record['items'],name);configs.append({'configuration':name,'parameters':TRACKING_PRESETS[name],'metrics':result})
    selected=max(configs,key=lambda c:(c['metrics'].get('IDF1',0),-c['metrics'].get('ID_switch',0),-list(TRACKING_PRESETS).index(c['configuration'])))
    lock={'protocol_id':'quality-v2','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':run.id,'selection_split':'val','test_used_for_selection':False,'detector_weights_sha256':record['weights_sha256'],'detector_manifest_sha256':sha(Path(pack)/'manifest.json'),'validation_index_sha256':record['index_sha256'],'validation_cache_sha256':sha(root/'reports/v2/tracking/validation-cache.json'),'tracker_source_sha256':sha(root/'packages/indoor_perception_lab/src/indoor_perception_lab/tracking_v2.py'),'configs':configs,'selected':selected}
    json_write(root/'models/v2/selection-tracking.json',lock);json_write(root/'reports/v2/tracking/selection.json',lock);run.finish(lock);return lock


def evaluate_mot(pack,device='mps'):
    from indoor_perception_lab.quality_evaluation import image_frame
    root=root_path();ensure_test_seal('detection',pack,'mot17');lock=json.loads((root/'models/v2/selection-tracking.json').read_text())
    if lock['detector_weights_sha256']!=sha(Path(pack)/'weights.pt') or lock['detector_manifest_sha256']!=sha(Path(pack)/'manifest.json') or lock['tracker_source_sha256']!=sha(root/'packages/indoor_perception_lab/src/indoor_perception_lab/tracking_v2.py'):raise ValueError('Tracking lock/model/source differs')
    run=QualityRun('quality-v2-tracking-evaluate',{'protocol_id':'quality-v2','device':device,'seed':42,'dataset':'mot17','model_pack':str(Path(pack).relative_to(root)),'checkpoint_sha256':sha(Path(pack)/'weights.pt'),'tracking_lock_sha256':sha(root/'models/v2/selection-tracking.json')},root/'data/v2/mot17/index.jsonl')
    rows=read_index_v2('mot17','outside_holdout');adapter=load_adapter(pack,device=device);items=[]
    try:
        for row in rows:
            frame=image_frame(root,row);items.append({'sample_id':frame.sample_id,'sequence_id':row['sequence_id'],'timestamp_ns':frame.timestamp_ns,'detections':[d.to_json() for d in adapter.predict(frame).detections if d.class_id==1],'targets':[{'bbox':t['bbox_xyxy'],'track_id':str(t['track_id']),'class_id':1,'ignore':t.get('ignore',False)} for t in targets_for_row(root,row)]})
    finally:adapter.close()
    results={}
    for key in dict.fromkeys(['legacy-v1','T1',lock['selected']['configuration']]):
        result,preds=evaluate_cache(items,key);results[key]=result;json_write(root/f'reports/v2/tracking/mot17-{key}-predictions.json',preds)
    report={'protocol_id':'quality-v2','run_id':run.id,'scope':'all900MOT17-11-SDPframes atoriginal30Hz; 1outsideheldoutscene','evaluator':'local independently tested identity evaluator, not official MOTChallenge submission','iou_threshold':.5,'selected_configuration':lock['selected']['configuration'],'common_detector':str(Path(pack).relative_to(root)),'weights_sha256':sha(Path(pack)/'weights.pt'),'index_sha256':sha(root/'data/v2/mot17/index.jsonl'),'results':results,'selection_was_validation_only':True}
    json_write(root/'reports/v2/tracking/mot17-heldout.json',report);run.finish(report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','mot']);p.add_argument('--pack',required=True);p.add_argument('--device',default='mps');p.add_argument('--wait-selection',action='store_true');a=p.parse_args();root=root_path()
    if a.wait_selection:
        deadline=datetime.datetime.fromisoformat(json.loads((root/'configs/v2/protocol.json').read_text())['deadline_utc'].replace('Z','+00:00'))
        while not (root/'models/v2/selection-detection.json').exists():
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Approved implementation deadline reached')
            time.sleep(10)
    pack=root/json.loads((root/'models/v2/selection-detection.json').read_text())['selected']['pack'] if a.pack=='selected' else root/a.pack
    result=select_tracking(pack,a.device) if a.action=='select' else evaluate_mot(pack,a.device)
    if a.action=='select':
        summary={'run_id':result['run_id'],'selected':result['selected']['configuration'],'candidates':[{k:c[k] for k in ['configuration','parameters']}|{'IDF1':c['metrics']['IDF1'],'ID_switch':c['metrics']['ID_switch']} for c in result['configs']],'evidence':'reports/v2/tracking/selection.json'}
    else:summary={'run_id':result['run_id'],'selected':result['selected_configuration'],'results':{k:{name:v.get(name) for name in ['IDF1','ID_switch','missed_detections','false_positive']} for k,v in result['results'].items()},'evidence':'reports/v2/tracking/mot17-heldout.json'}
    print(json.dumps(summary))
