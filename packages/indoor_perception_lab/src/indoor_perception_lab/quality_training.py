"""Auditable quality-v2 neural training with train/validation-only selection."""
from __future__ import annotations
import json,os,random,time,math,shutil,copy
from pathlib import Path
import numpy as np
import torch
from robot_model_sdk import load_adapter
from robot_model_sdk.models import semantic_network
from robot_model_sdk.quality_models import MultiScaleSemanticNet,detection_model
from robot_model_sdk.preprocessing import quality_rgb,IMAGENET_MEAN,IMAGENET_STD
from .quality_evaluation import evaluate_semantic,evaluate_detection
from .experiments import Run,root_path,sha,json_write,save_pack,NYU40,set_seed

PRESETS={
 'S1':{'task':'semantic','architecture':'mobilenet_v3_large_frozen_64decoder','encoder_lr':0.,'head_lr':.001,'unfreeze':'none'},
 'S2':{'task':'semantic','architecture':'mobilenet_v3_multiscale_nyu40','encoder_lr':.0001,'head_lr':.001,'unfreeze':'last3'},
 'S3':{'task':'semantic','architecture':'mobilenet_v3_multiscale_nyu40','encoder_lr':.00003,'head_lr':.0003,'unfreeze':'all'},
 'D1':{'task':'detection','architecture':'ssdlite320_quality_v2','encoder_lr':0.,'head_lr':.0003,'unfreeze':'none'},
 'D2':{'task':'detection','architecture':'ssdlite320_quality_v2','encoder_lr':.0001,'head_lr':.0003,'unfreeze':'last'},
 'D3':{'task':'detection','architecture':'fasterrcnn_mobilenet_v3_large_320_fpn_quality_v2','encoder_lr':.0001,'head_lr':.0003,'unfreeze':'last3'},
}

class QualityRun(Run):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);destination=self.root/'reports/v2/runs'/self.id;destination.parent.mkdir(parents=True,exist_ok=True);self.dir.rename(destination);self.dir=destination;self.log=self.dir/'events.jsonl'

class QualityDataset:
    def __init__(self,root,rows,task,augment=True):self.root=Path(root);self.rows=rows;self.task=task;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        import cv2
        from .data_v2 import targets_for_row
        row=self.rows[i]
        if row['split'] not in {'train','val'}:raise ValueError('Training loader refuses test targets')
        image=cv2.imread(str(self.root/row['rgb_path']))
        if image is None:raise ValueError('Unreadable train RGB')
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
        if self.task=='semantic':
            target=cv2.imread(str(self.root/row['mask_path']),cv2.IMREAD_UNCHANGED)
            if target is None or target.shape!=(h,w):raise ValueError('Training image/target mismatch')
            target=cv2.resize(target,(320,240),interpolation=cv2.INTER_NEAREST)
        else:
            annotations=targets_for_row(self.root,row)
            boxes=np.asarray([a['bbox_xyxy'] for a in annotations if not a.get('ignore',False)],np.float32).reshape(-1,4)
            ignored=np.asarray([a['bbox_xyxy'] for a in annotations if a.get('ignore',False)],np.float32).reshape(-1,4)
            for b in [boxes,ignored]:
                if len(b) and (np.any(b[:,2:]<=b[:,:2]) or not np.isfinite(b).all()):raise ValueError('Invalid detection target')
        if self.augment:
            if np.random.rand()<.5:
                rgb=rgb[:,::-1].copy()
                if self.task=='semantic':target=target[:,::-1].copy()
                else:
                    for b in [boxes,ignored]:
                        old=b.copy();b[:,0]=w-old[:,2];b[:,2]=w-old[:,0]
            if np.random.rand()<.5:
                contrast=np.random.uniform(.8,1.2);brightness=np.random.uniform(-20,20)
                rgb=np.clip((rgb.astype(np.float32)-127.5)*contrast+127.5+brightness,0,255).astype(np.uint8)
            if np.random.rand()<.1:rgb=cv2.GaussianBlur(rgb,(3,3),.6)
        if self.task=='semantic':return torch.from_numpy(quality_rgb(rgb)),torch.from_numpy(target.astype(np.int64))
        return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,{'boxes':torch.from_numpy(boxes),'labels':torch.ones(len(boxes),dtype=torch.int64),'ignore_boxes':torch.from_numpy(ignored)}

def build_train_model(preset,root):
    if preset['task']=='semantic':
        model=semantic_network() if preset['architecture']=='mobilenet_v3_large_frozen_64decoder' else MultiScaleSemanticNet()
        initial=root/'cache/torch/mobilenet_v3_large-8738ca79.pth'
        weights=torch.load(initial,map_location='cpu',weights_only=True)
        model.encoder.load_state_dict({k.removeprefix('features.'):v for k,v in weights.items() if k.startswith('features.')})
        encoder=model.encoder
        for p in encoder.parameters():p.requires_grad=False
        if preset['unfreeze']=='last3':
            for block in encoder[-3:]:
                for p in block.parameters():p.requires_grad=True
        elif preset['unfreeze']=='all':
            for p in encoder.parameters():p.requires_grad=True
    else:
        filename='fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth' if preset['unfreeze']=='last3' else 'ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
        initial=root/'cache/torch'/filename;model=detection_model(preset['architecture'],initial)
        encoder=model.backbone
        for p in encoder.parameters():p.requires_grad=False
        if preset['unfreeze']=='last':
            for p in model.backbone.features[-1].parameters():p.requires_grad=True
        elif preset['unfreeze']=='last3':
            for block in list(model.backbone.body.children())[-3:]:
                for p in block.parameters():p.requires_grad=True
            for p in model.backbone.fpn.parameters():p.requires_grad=True
    enc_ids={id(p) for p in encoder.parameters()}
    groups=[]
    ep=[p for p in model.parameters() if p.requires_grad and id(p) in enc_ids]
    hp=[p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    if ep:groups.append({'params':ep,'lr':preset['encoder_lr'],'group':'encoder'})
    if hp:groups.append({'params':hp,'lr':preset['head_lr'],'group':'head'})
    return model,groups,initial

def manifest_for(preset,seed,runid):
    common={'task':preset['task'],'architecture':preset['architecture'],'protocol_id':'quality-v2','run_id':runid,'seed':seed,'domain':'NYU Depth V2 real indoor' if preset['task']=='semantic' else 'JAAD real person detection','architecture_config':{'encoder_unfreeze':preset['unfreeze'],'encoder_lr':preset['encoder_lr'],'head_lr':preset['head_lr']},'contribution':'candidate portfolio with disclosed code assistance','training_target_access':'train and validation only'}
    if preset['task']=='semantic':common.update(class_names=NYU40,input_size=[240,320],mean=list(IMAGENET_MEAN),std=list(IMAGENET_STD),preprocessing_id='quality_rgb_v2',postprocess='resize_logits_then_argmax',label_mapping='NYU40 source1..40 -> model0..39;255ignore')
    else:common.update(num_classes=2,class_names=['background','person'],preprocessing_id='torchvision_detection_transform',ignore_policy='background candidates with ignored-region IOA>0.5 excluded from sampling; positive person matches retained')
    common['architecture_parameters']={'input_size':[240,320],'output_classes':40,'decoder_channels':64,'feature_indices':[3,6,12,16],'feature_strides':[4,8,16,32],'groupnorm_groups':8} if preset['architecture']=='mobilenet_v3_multiscale_nyu40' else {'input_size':[240,320],'output_classes':40,'decoder_channels':64} if preset['task']=='semantic' else {'minimum_image_side':320,'output_classes':2,'class_order':['background','person']}
    return common

def checkpoint_rng():
    ns=np.random.get_state()
    return {'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':ns[1].tolist(),'numpy_pos':ns[2],'numpy_has_gauss':ns[3],'numpy_cached_gaussian':ns[4],'mps_rng':torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}

def restore_rng(ck,device):
    torch.set_rng_state(ck['torch_rng'].cpu());random.setstate(ck['python_rng']);np.random.set_state(('MT19937',np.array(ck['numpy_rng'],dtype=np.uint32),ck['numpy_pos'],ck['numpy_has_gauss'],ck['numpy_cached_gaussian']))
    if device=='mps' and ck.get('mps_rng') is not None:torch.mps.set_rng_state(ck['mps_rng'].cpu())

def select_device(preset,requested):
    if requested!='auto':return requested
    configuration=next(k for k,v in PRESETS.items() if v==preset)
    specific=root_path()/f'reports/v2/backend-presets/{configuration}.json'
    if specific.exists():
        checks=json.loads(specific.read_text())['results']
        return 'mps' if any(r['device']=='mps' and r['verified'] for r in checks) else 'cpu'
    name='semantic_legacy' if preset['architecture']=='mobilenet_v3_large_frozen_64decoder' else 'semantic_multiscale' if preset['task']=='semantic' else 'fasterrcnn' if preset['unfreeze']=='last3' else 'ssdlite'
    result=json.loads((root_path()/'reports/v2/backend-models.json').read_text())
    return 'mps' if any(r['architecture']==name and r['device']=='mps' and r['verified'] for r in result['results']) else 'cpu'

def train_quality(configuration,seed=42,device='auto',epochs=30,max_steps=6000,resume=None,overfit=False,index_path=None,stop_after_epochs=None):
    import cv2,psutil
    from .data_v2 import read_index_v2
    root=root_path();preset=PRESETS[configuration];task=preset['task'];device=select_device(preset,device);set_seed(seed)
    if device=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Silent MPS fallback must be disabled')
    lock=root/f'models/v2/selection-{task}.json'
    if not overfit and lock.exists():
        candidate=root/f'models/v2/{task}-{configuration}-seed{seed}'
        frozen=json.loads(lock.read_text())
        if candidate.resolve()==(root/frozen['selected']['pack']).resolve():raise ValueError('Refusing to overwrite the immutable selected artifact')
        if (candidate/'manifest.json').exists():
            existing=json.loads((candidate/'manifest.json').read_text())
            if existing.get('selected_checkpoint_step',0)>0 and not existing.get('interrupted_for_resume_test',False):raise ValueError('Refusing to overwrite a completed seed repeat; preserve the frozen experiment')
    if not (1<=epochs<=30 and 1<=max_steps<=6000):raise ValueError('quality-v2 requires1..30epochs and1..6000steps')
    dataset='nyu' if task=='semantic' else 'jaad';idx=Path(index_path) if index_path else root/f'data/v2/{dataset}/index.jsonl'
    rows=[json.loads(x) for x in idx.read_text().splitlines() if x.strip()];train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    if not train or (not val and not overfit):raise ValueError('Nonempty train and validation required for primary runs')
    groups={s:{r['scene_id'] for r in rows if r['split']==s} for s in ('train','val','test')}
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Split scene overlap')
    if overfit:train=train[:16];val=train
    source_datasets=sorted({r['dataset'] for r in train})
    config={'protocol_id':'quality-v2','configuration':configuration,'seed':seed,'device':device,'epochs':epochs,'max_steps':max_steps,'batch_size':2,'optimizer':'AdamW','patience':5,'minimum_epochs':3,'overfit_diagnostic':overfit,'selection_split':'train_diagnostic' if overfit else 'val','test_used_for_selection':False,'preset':preset,'train_samples':len(train),'validation_samples':len(val),'index_path':str(idx.relative_to(root)),'source_datasets':source_datasets}
    run=QualityRun('quality-v2-'+task+('-overfit' if overfit else '-train'),config,idx);shutil.copyfile(idx,run.dir/'split-index.jsonl');model,groups_params,initial=build_train_model(preset,root);model.to(device)
    meta=manifest_for(preset,seed,run.id);meta.update(initialization_sha256=sha(initial),configuration=configuration,dataset_split_hash=sha(idx),overfit_diagnostic=overfit,interrupted_for_resume_test=stop_after_epochs is not None,training_backend=device,mps_fallback_enabled=False,architecture_backend_evidence={'path':'reports/v2/backend-models.json','sha256':sha(root/'reports/v2/backend-models.json')});name=f'v2/{task}-{configuration}-seed{seed}'+('-overfit' if overfit else '')
    meta['source_datasets']=source_datasets
    specific=root/f'reports/v2/backend-presets/{configuration}.json'
    if specific.exists():meta['configuration_backend_evidence']={'path':str(specific.relative_to(root)),'sha256':sha(specific)}
    if source_datasets==['hypersim']:meta['domain']='Hypersim synthetic indoor diagnostic'
    ds=QualityDataset(root,train,task,not overfit);generator=torch.Generator().manual_seed(seed)
    loader=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,num_workers=0,generator=generator,drop_last=True,collate_fn=(lambda b:tuple(zip(*b))) if task=='detection' else None)
    optimizer=torch.optim.AdamW(groups_params,weight_decay=1e-4)
    def schedule(step):return min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*min(step,max_steps)/max_steps)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    criterion=None;majority=None
    if task=='semantic':
        counts=np.zeros(40,np.int64)
        for row in train:
            mask=cv2.imread(str(root/row['mask_path']),cv2.IMREAD_UNCHANGED);counts+=np.bincount(mask[mask!=255],minlength=40)
        w=np.minimum(10,np.sqrt(counts.sum()/np.maximum(counts,1)));w[counts==0]=0;w/=w[w>0].mean();criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device),ignore_index=255);majority=int(counts.argmax());run.event(event='train_label_support',counts=counts.tolist(),weights=w.tolist(),majority_class=majority)
    step=0;epoch0=0;best=-float('inf');stale=0;best_state=None;best_epoch=None;best_step=None;initial_validation=None;initial_validation_seconds=0.
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=True)
        if ck['dataset_split_hash']!=sha(idx) or ck['config']['configuration']!=configuration or ck['config']['device']!=device:raise ValueError('Resume requires same frozen split/configuration/backend')
        for key in ('seed','epochs','max_steps','preset','batch_size','overfit_diagnostic'):
            if ck['config'].get(key)!=config[key]:raise ValueError('Resume immutable configuration differs: '+key)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.to(device)
        step=ck['step'];epoch0=ck['next_epoch'];best=ck['best'];best_state=ck['best_model'];best_epoch=ck['best_epoch'];best_step=ck.get('best_step');stale=ck['stale'];generator.set_state(ck['sampler_generator_state'].cpu());restore_rng(ck,device);initial_validation=ck['initial_validation'];run.event(event='resume',checkpoint=str(resume),step=step)
    else:
        pack=save_pack(root,name+'-initial',model,meta);adapter=load_adapter(pack,device=device);initial_start=time.perf_counter()
        initial_validation=evaluate_semantic(root,val,adapter) if task=='semantic' else evaluate_detection(root,val,adapter)[0];initial_validation_seconds=time.perf_counter()-initial_start;adapter.close();run.event(event='initial_validation',metrics=initial_validation,evaluation_seconds=initial_validation_seconds)
    train_time=0.;validation_time=0.;data_time=0.;history=[];started=time.perf_counter()
    try:
        for epoch in range(epoch0,epochs):
            if step>=max_steps or (epoch>=3 and stale>=5):break
            model.train()
            for m in model.modules():
                if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and (m in set(model.encoder.modules()) if task=='semantic' else m in set(model.backbone.modules())):m.eval()
            losses=[];last=time.perf_counter()
            for batch in loader:
                data_time+=time.perf_counter()-last;optimizer.zero_grad(set_to_none=True);a=time.perf_counter()
                if task=='semantic':
                    x,y=batch;loss=criterion(model(x.to(device)),y.to(device))
                else:
                    images,targets=batch;loss=sum(model([x.to(device) for x in images],[{k:v.to(device) for k,v in t.items()} for t in targets]).values())
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                if not torch.isfinite(gradient_norm):raise FloatingPointError('Nonfinite training gradient')
                optimizer.step();scheduler.step()
                if device=='mps':torch.mps.synchronize()
                train_time+=time.perf_counter()-a;step+=1;losses.append(float(loss.detach().cpu()))
                rss=psutil.Process().memory_info().rss
                if rss>12*1024**3:raise MemoryError('Training process exceeded12GiB')
                if device=='mps' and torch.mps.driver_allocated_memory()>10*1024**3:raise MemoryError('MPS driver allocation exceeded10GiB')
                run.event(event='training_step',step=step,epoch=epoch,loss=losses[-1],training_s=train_time,data_s=data_time,rss_bytes=rss,lrs=scheduler.get_last_lr())
                if step==100:
                    remaining_train=(train_time+data_time)/100*(max_steps-100);estimated_evaluations=min(epochs,math.ceil(max_steps/len(loader)))
                    run.event(event='throughput_100',training_s=train_time,data_s=data_time,steps_per_second=100/(train_time+data_time),estimated_remaining_training_s=remaining_train,observed_validation_pass_s=initial_validation_seconds,estimated_validation_s=initial_validation_seconds*estimated_evaluations,estimated_remaining_total_s=remaining_train+initial_validation_seconds*estimated_evaluations,estimate_excludes_checkpoint_io=True)
                last=time.perf_counter()
                if step>=max_steps:break
            model.eval();pack=save_pack(root,name,model,meta);adapter=load_adapter(pack,device=device);a=time.perf_counter()
            metrics=evaluate_semantic(root,val,adapter) if task=='semantic' else evaluate_detection(root,val,adapter)[0];validation_time+=time.perf_counter()-a;adapter.close();score=metrics['mIoU' if task=='semantic' else 'AP']
            if score is None or not math.isfinite(score):raise ValueError('No evaluable validation metric')
            if score>best+1e-6:best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;best_step=step;stale=0
            else:stale+=1
            record={'epoch':epoch,'step':step,'loss':float(np.mean(losses)),'validation_score':score,'elapsed_s':time.perf_counter()-started};history.append(record);run.event(event='validation',**record,metrics=metrics)
            ck={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'next_epoch':epoch+1,'step':step,'best':best,'best_model':best_state,'best_epoch':best_epoch,'best_step':best_step,'stale':stale,'config':config,'dataset_split_hash':sha(idx),'sampler_generator_state':generator.get_state(),'initial_validation':initial_validation,**checkpoint_rng()}
            torch.save(ck,run.dir/'checkpoint.pt');print(json.dumps({'run_id':run.id,**record}),flush=True)
            if stop_after_epochs is not None and len(history)>=stop_after_epochs:break
        if best_state is None:raise ValueError('No trained checkpoint produced')
        model.load_state_dict(best_state);pack=save_pack(root,name,model,{**meta,'selection_value':best,'selection_metric':'train_diagnostic' if overfit else 'validation_mIoU' if task=='semantic' else 'validation_AP','best_epoch':best_epoch,'selected_checkpoint_step':best_step,'steps':step,'dataset_split_hash':sha(idx),'run_dir':str(run.dir.relative_to(root)),'majority_class':majority})
        run.finish({'model_pack':str(pack.relative_to(root)),'initial_validation':initial_validation,'best_validation_score':best,'best_epoch':best_epoch,'steps':step,'training_seconds':train_time,'data_seconds':data_time,'validation_seconds':validation_time,'curve':history,'checkpoint_sha256':sha(run.dir/'checkpoint.pt'),'test_evaluated':False,'interrupted_for_resume_test':stop_after_epochs is not None})
        return pack,run.dir
    except Exception as exc:
        import traceback
        run.event(event='failure',error=repr(exc),traceback=traceback.format_exc());run.finish({'error':repr(exc)},status='failed');raise
