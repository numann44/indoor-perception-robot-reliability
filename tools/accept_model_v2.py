"""Trained-artifact, annotation-free CPU/MPS inference acceptance on validation RGB."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import numpy as np
import torch
from robot_model_sdk import load_adapter
from indoor_perception_lab.experiments import root_path,sha,json_write
from indoor_perception_lab.data_v2 import read_index_v2
from indoor_perception_lab.quality_evaluation import image_frame
from indoor_perception_lab.quality_training import PRESETS,manifest_for

def accept(pack,limit=8):
    root=root_path();pack=Path(pack).resolve();m=json.loads((pack/'manifest.json').read_text())
    if m.get('overfit_diagnostic') or m.get('selected_checkpoint_step',0)<=0:raise ValueError('A primary trained perception artifact is required')
    lock=root/f"models/v2/selection-{m['task']}.json"
    if lock.exists() and (root/json.loads(lock.read_text())['selected']['pack']).resolve()==pack:raise ValueError('Artifact metadata must be accepted before the immutable selection lock')
    rows=read_index_v2('nyu' if m['task']=='semantic' else 'jaad','val',root=root)[:limit]
    frames=[image_frame(root,r) for r in rows]
    torch.set_num_threads(4)
    records=[];cpu_outputs=None
    for device in ['cpu','mps']:
        adapter=None;start=time.perf_counter();record={'device':device,'fallback_enabled':False,'annotation_access':False}
        try:
            if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
            adapter=load_adapter(pack,device=device);outputs=[adapter.predict(f) for f in frames]
            if device=='mps':torch.mps.synchronize()
            if m['task']=='semantic':
                if any(o.semantic.shape!=f.rgb.shape[:2] or not np.isfinite(o.logits).all() for f,o in zip(frames,outputs)):raise ValueError('Invalid semantic shape/logits')
                if device=='cpu':cpu_outputs=outputs
                else:
                    record['max_abs_logit_difference_vs_cpu']=max(float(np.max(np.abs(a.logits-b.logits))) for a,b in zip(cpu_outputs,outputs))
                    record['pixel_agreement_vs_cpu']=float(np.mean([np.mean(a.semantic==b.semantic) for a,b in zip(cpu_outputs,outputs)]))
            else:
                if any(not np.isfinite([*d.bbox,d.score]).all() for o in outputs for d in o.detections):raise ValueError('Nonfinite detection output')
                record['detection_counts']=[len(o.detections) for o in outputs]
                if device=='cpu':cpu_outputs=outputs
            record.update(verified=True,images=len(frames),seconds=time.perf_counter()-start)
        except Exception as exc:record.update(verified=False,error=repr(exc),seconds=time.perf_counter()-start)
        finally:
            if adapter is not None:adapter.close()
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        records.append(record)
    if not records[0]['verified']:raise RuntimeError('Mandatory trained CPU inference acceptance failed')
    evidence=root/f"reports/v2/artifact-backends/{pack.name}.json"
    report={'protocol_id':'quality-v2','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_sha256':sha(pack/'weights.pt'),'dataset_split_hash':m['dataset_split_hash'],'sample_ids':[f.sample_id for f in frames],'scope':'trained artifact inference; architecture backward/update proof is separately bound','devices':records}
    json_write(evidence,report)
    m['trained_backend_evidence']={'path':str(evidence.relative_to(root)),'sha256':sha(evidence),'verified_devices':[r['device'] for r in records if r['verified']]}
    m['architecture_parameters']=manifest_for(PRESETS[m['configuration']],m['seed'],m['run_id'])['architecture_parameters']
    m.setdefault('architecture_backend_evidence',{'path':'reports/v2/backend-models.json','sha256':sha(root/'reports/v2/backend-models.json')})
    json_write(pack/'manifest.json',m)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',required=True);p.add_argument('--limit',type=int,default=8);a=p.parse_args()
    print(json.dumps(accept(root_path()/a.pack,a.limit)))
