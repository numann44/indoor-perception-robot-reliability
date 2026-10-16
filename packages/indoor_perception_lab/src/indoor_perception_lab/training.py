from __future__ import annotations
import copy,json,time,traceback,random,shutil
from pathlib import Path
import numpy as np
from robot_model_sdk import semantic_network,load_adapter
from .datasets import read_index,SemanticDataset,DetectionDataset,jaad_targets
from .evaluation import semantic_evaluate,detection_evaluate
from .experiments import Run,set_seed,root_path,save_pack,sha,json_write,NYU40

def train(task="semantic",seed=42,lr=.001,epochs=15,max_steps=600,device="cpu",resume=None,index_path=None,unfreeze_last=False):
    if task=='detection' and unfreeze_last:raise ValueError('Final-encoder unfreezing is a semantic-only preset; validated detector presets train heads with the backbone frozen')
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    root=root_path();set_seed(seed);dataset="hypersim" if task=="semantic" else "jaad";idx=Path(index_path) if index_path else root/"data"/dataset/"index.jsonl";rows=read_index(root,dataset,index_path=idx);tr=[r for r in rows if r["split"]=="train"];val=[r for r in rows if r["split"]=="val"]
    ignored_training_rows=0
    if task=="detection":
        clean=[row for row in tr if not any(t["ignore"] for t in jaad_targets(root,row))];ignored_training_rows=len(tr)-len(clean);tr=clean
    if not tr or not val:raise ValueError("Training and validation scene split required")
    config={"task":task,"seed":seed,"lr":lr,"epochs":epochs,"max_steps":max_steps,"batch_size":2,"patience":3,"device":device,"hypothesis":"A frozen lightweight encoder with supervised head tuning can improve held-out task performance; improvement is not guaranteed","selection_split":"val","test_used_for_selection":False,"unfreeze_last":unfreeze_last,"training_images":len(tr),"training_clips_or_scenes":len({r["scene_id"] for r in tr}),"excluded_train_images_with_ignored_regions":ignored_training_rows,"crowd_training_policy":"exclude entire training frame containing ignored region; no false background-negative supervision" if task=="detection" else None}
    run=Run(task+"-train",config,idx);shutil.copyfile(idx,run.dir/"split-index.jsonl");idx=run.dir/"split-index.jsonl";generator=torch.Generator().manual_seed(seed);packname=f"{task}-seed{seed}-lr{lr:g}"+("-unfreeze" if unfreeze_last else "");optimizer=None
    try:
        if task=="semantic":
            model=semantic_network();pre=torch.load(root/"cache/torch/mobilenet_v3_large-8738ca79.pth",map_location="cpu",weights_only=True);encoder={k.removeprefix("features."):v for k,v in pre.items() if k.startswith("features.")};model.encoder.load_state_dict(encoder)
            if unfreeze_last:
                for block in model.encoder[-3:]:
                    for parameter in block.parameters():parameter.requires_grad=True
            ds=SemanticDataset(root,tr,True)
            counts=np.zeros(40)
            for _,mask in SemanticDataset(root,tr):counts+=np.bincount(mask.numpy()[mask.numpy()!=255],minlength=40)
            weights=np.where(counts>0,np.minimum(10,np.sqrt(counts.sum()/np.maximum(counts,1))),0);weights/=np.mean(weights[weights>0]);criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),ignore_index=255)
            metadata={"task":"semantic","architecture":"mobilenet_v3_large_frozen_64decoder","class_names":NYU40,"input_size":[240,320],"mean":[.485,.456,.406],"std":[.229,.224,.225],"domain":"Hypersim synthetic indoor","label_mapping":"source NYU40 1..40 to output0..39;255ignore","run_id":run.id}
            loader=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,num_workers=0,generator=generator)
        else:
            model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None);model.load_state_dict(torch.load(root/"cache/torch/ssdlite320_mobilenet_v3_large_coco-a79551df.pth",map_location="cpu",weights_only=True))
            for p in model.backbone.parameters():p.requires_grad=False
            head=model.head.classification_head
            for block in head.module_list:
                old=block[-1];anchors=old.out_channels//91;new=torch.nn.Conv2d(old.in_channels,anchors*2,1)
                with torch.no_grad():new.weight.copy_(old.weight.reshape(anchors,91,old.in_channels,1,1)[:,[0,1]].reshape_as(new.weight));new.bias.copy_(old.bias.reshape(anchors,91)[:,[0,1]].reshape_as(new.bias))
                block[-1]=new
            head.num_columns=2;ds=DetectionDataset(root,tr,True);loader=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,num_workers=0,drop_last=True,generator=generator,collate_fn=lambda batch:tuple(zip(*batch)))
            metadata={"task":"detection","architecture":"ssdlite320_mobilenet_v3_large","num_classes":2,"class_names":["background","person"],"domain":"JAAD real traffic","run_id":run.id,"contribution":"person/background classification head initialized from COCO rows; encoder frozen; classification and box heads fine-tuned"}
        model.to(device);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4);epoch0=0;step=0;best=-float('inf');stale=0;best_epoch=None;best_state=None
        if resume:
            ck=torch.load(resume,map_location=device,weights_only=True);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);epoch0=ck['next_epoch'];step=ck['step'];best=ck['best'];best_state=ck['best_model'];torch.set_rng_state(ck['torch_rng']);np.random.set_state(('MT19937',np.array(ck['numpy_rng'],dtype=np.uint32),ck['numpy_pos'],ck['numpy_has_gauss'],ck['numpy_cached_gaussian']));
            if ck['dataset_split_hash']!=sha(idx):raise ValueError('Resume dataset split differs from checkpoint')
            generator.set_state(ck['sampler_generator_state'].cpu());random.setstate(ck['python_rng']);best_epoch=ck.get('best_epoch');stale=ck.get('stale',0)
            if device=='mps' and ck.get('mps_rng') is not None:torch.mps.set_rng_state(ck['mps_rng'])
            run.event(event='resume',checkpoint=str(resume),step=step)
            shutil.copyfile(resume,run.dir/'checkpoint.pt')
        start=time.perf_counter();train_logs=[]
        # Baseline validation only; no test labels are consulted during selection.
        if resume:
            # Do not overwrite the true untrained baseline or advance restored RNG
            # by constructing a disposable randomly initialized inference model.
            baseline=ck['initial_validation']
        else:
            baseline_pack=save_pack(root,packname+'-initial',model,metadata);adapter=load_adapter(baseline_pack,device=device)
            baseline=semantic_evaluate(root,val,adapter) if task=='semantic' else detection_evaluate(root,val,adapter)[0];adapter.close()
        run.event(event='initial_validation',metrics=baseline,reused_checkpoint=bool(resume))
        for epoch in range(epoch0,epochs):
            if step>=max_steps or stale>=3:break
            model.train()
            if task=='detection':model.backbone.eval()
            elif unfreeze_last:
                for block in model.encoder[-3:]:block.train()
            losses=[]
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                if task=='semantic':x,y=batch;x=x.to(device);y=y.to(device);loss=criterion(model(x),y)
                else:
                    images,targets=batch;images=[x.to(device) for x in images];targets=[{k:v.to(device) for k,v in t.items()} for t in targets];loss=sum(model(images,targets).values())
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                import psutil
                rss=psutil.Process().memory_info().rss
                if rss>12*1024**3:raise MemoryError("Per-experiment CPU RSS exceeded12GiB; resume with smaller model/device")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();step+=1;losses.append(float(loss.detach().cpu()));run.event(event='training_step',epoch=epoch,step=step,loss=losses[-1],elapsed_s=time.perf_counter()-start,rss_bytes=rss)
                if step==30:run.event(event='throughput_pilot',steps=30,elapsed_s=time.perf_counter()-start,estimated_remaining_s=(time.perf_counter()-start)/30*(max_steps-30))
                if step>=max_steps:break
            model.eval();temp=save_pack(root,packname,model,metadata);adapter=load_adapter(temp,device=device);metrics=semantic_evaluate(root,val,adapter) if task=='semantic' else detection_evaluate(root,val,adapter)[0];adapter.close();score=metrics['mIoU' if task=='semantic' else 'AP'];run.event(event='validation',epoch=epoch,step=step,metrics=metrics,mean_training_loss=float(np.mean(losses)))
            if score>best+1e-6:best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
            else:stale+=1
            ns=np.random.get_state();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'next_epoch':epoch+1,'step':step,'best':best,'best_model':best_state,'torch_rng':torch.get_rng_state(),'numpy_rng':ns[1].tolist(),'numpy_pos':ns[2],'numpy_has_gauss':ns[3],'numpy_cached_gaussian':ns[4],'seed':seed,'config':config,'dataset_split_hash':sha(idx),'sampler_generator_state':generator.get_state(),'python_rng':random.getstate(),'mps_rng':torch.mps.get_rng_state() if device=='mps' else None,'best_epoch':best_epoch,'stale':stale,'initial_validation':baseline,'checkpoint_boundary':'end-of-epoch or explicit max-step stop'},run.dir/'checkpoint.pt')
            train_logs.append({'epoch':epoch,'step':step,'loss':float(np.mean(losses)),'validation_score':score})
            print(json.dumps({'run_id':run.id,'epoch':epoch,'step':step,'loss':train_logs[-1]['loss'],'validation_score':score}),flush=True)
            if stale>=3 or step>=max_steps:break
        model.load_state_dict(best_state);pack=save_pack(root,packname,model,{**metadata,'selection_metric':'validation_mIoU' if task=='semantic' else 'validation_AP','selection_value':best,'best_epoch':best_epoch,'checkpoint_run_id':run.id,'seed':seed,'training_split_hash':sha(idx)})
        run.finish({'initial_validation':baseline,'best_validation_score':best,'best_epoch':best_epoch,'steps':step,'epochs_run':len(train_logs),'training_curve':train_logs,'model_pack':str(pack.relative_to(root)),'checkpoint_sha256':sha(run.dir/'checkpoint.pt'),'test_evaluated':False});return str(pack)
    except Exception as exc:
        run.event(event='failure',error=repr(exc),traceback=traceback.format_exc());run.finish({'error':repr(exc)},status='failed');raise
