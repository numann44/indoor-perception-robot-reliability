from __future__ import annotations
import json,shutil,datetime
from pathlib import Path
import numpy as np
from robot_model_sdk import load_adapter
from .experiments import Run,root_path,json_write,sha,save_pack,NYU40,set_seed
from .datasets import read_index,SemanticDataset
from .evaluation import semantic_evaluate,detection_evaluate

def prepare_detection_baseline():
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
    root=root_path();out=root/'models/detection-baseline'
    if (out/'manifest.json').exists():return out
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None);model.load_state_dict(torch.load(root/'cache/torch/ssdlite320_mobilenet_v3_large_coco-a79551df.pth',weights_only=True,map_location='cpu'))
    return save_pack(root,'detection-baseline',model,{'task':'detection','architecture':'ssdlite320_mobilenet_v3_large','num_classes':91,'class_names':SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.meta['categories'],'domain':'COCO pretrained official baseline','weight_source':'https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth','redistribution':'not included in shareable source archive'})

def select_model(task):
    root=root_path();lock=root/'models'/f'selection-{task}.json'
    if lock.exists():return json.loads(lock.read_text())
    candidates=[]
    for p in (root/'models').glob('*/manifest.json'):
        m=json.loads(p.read_text())
        if m.get('task')==task and m.get('seed')==42 and 'selection_value' in m:candidates.append((float(m['selection_value']),p.parent,m))
    if not candidates:raise ValueError('No completed seed42 validation-selected training candidates')
    _,best,m=max(candidates,key=lambda x:(x[0],str(x[1])));name='segmentation-trained' if task=='semantic' else 'detection-trained';alias=root/'models'/name
    shutil.copytree(best,alias,dirs_exist_ok=True)
    record={'task':task,'selection_metric':m['selection_metric'],'selected_pack':str(alias.relative_to(root)),'source_pack':str(best.relative_to(root)),'selected_sha256':m['sha256'],'selected_seed':42,'selected_validation_score':m['selection_value'],'selection_locked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_consulted':False,'candidates':[{'pack':str(p.relative_to(root)),'validation_score':score} for score,p,_ in candidates]}
    json_write(lock,record)
    if task=='semantic':shutil.copytree(best.parent/(best.name+'-initial'),root/'models/segmentation-initial',dirs_exist_ok=True)
    else:prepare_detection_baseline()
    return record

def evaluate_release(task,pack,split='test',index_path=None,device='cpu'):
    root=root_path();pack=Path(pack);m=json.loads((pack/'manifest.json').read_text());dataset='hypersim' if task=='semantic' else 'jaad';idx=Path(index_path) if index_path else root/'data'/dataset/'index.jsonl';rows=read_index(root,dataset,index_path=idx);chosen=[r for r in rows if r['split']==split]
    if not chosen:raise ValueError('No requested held-out split; metric unavailable')
    selection=json.loads((root/'models'/f'selection-{task}.json').read_text())
    if m['sha256']!=selection['selected_sha256']:raise ValueError('Model is not the validation-frozen primary selection')
    run=Run(task+'-evaluation',{'seed':m.get('seed'),'split':split,'model_pack':str(pack),'checkpoint_sha256':m['sha256'],'selection_lock':selection,'test_used_for_selection':False},idx)
    adapter=load_adapter(pack,device=device)
    if task=='semantic':
        trained=semantic_evaluate(root,chosen,adapter);train_rows=[r for r in rows if r['split']=='train'];train=semantic_evaluate(root,train_rows,adapter);adapter.close();initial=load_adapter(root/'models/segmentation-initial',device=device);baseline=semantic_evaluate(root,chosen,initial);initial.close();counts=np.zeros(40,np.int64)
        for _,mask in SemanticDataset(root,train_rows):counts+=np.bincount(mask.numpy()[mask.numpy()!=255],minlength=40)
        majority=semantic_evaluate(root,chosen,majority_class=int(counts.argmax()));metrics={'trained':trained,'untrained_head':baseline,'train_majority':majority,'train':train,'test_minus_baseline_mIoU':trained['mIoU']-baseline['mIoU'],'domain_limit':'Hypersim synthetic only; no real-indoor semantic accuracy claim'}
    else:
        trained,predictions=detection_evaluate(root,chosen,adapter);adapter.close();baseline_adapter=load_adapter(prepare_detection_baseline(),device=device);baseline,base_predictions=detection_evaluate(root,chosen,baseline_adapter);baseline_adapter.close();metrics={'trained':trained,'pretrained_baseline':baseline,'AP_delta':trained['AP']-baseline['AP'],'domain_limit':'JAAD predominantly traffic scenes; no general indoor person accuracy claim'};json_write(run.dir/'trained-predictions.json',predictions);json_write(run.dir/'baseline-predictions.json',base_predictions)
    run.finish(metrics);json_write(root/'reports'/f'{task}-heldout.json',{'run_id':run.id,'split':split,**metrics});return {'run_id':run.id,'report':str((root/'reports'/f'{task}-heldout.json').relative_to(root)),**({'trained_mIoU':metrics['trained']['mIoU'],'baseline_mIoU':metrics['untrained_head']['mIoU']} if task=='semantic' else {'trained_AP':metrics['trained']['AP'],'baseline_AP':metrics['pretrained_baseline']['AP']})}
