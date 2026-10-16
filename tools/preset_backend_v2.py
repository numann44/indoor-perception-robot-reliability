"""Probe the exact encoder/head trainability used by each fixed preset."""
from pathlib import Path
import argparse,copy,datetime,json,os,time,traceback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import torch
from indoor_perception_lab.quality_training import PRESETS,build_train_model
from indoor_perception_lab.experiments import root_path,sha,json_write,set_seed
root=root_path()
for key,folder in [('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('TORCH_EXTENSIONS_DIR','torch-extensions'),('TRITON_CACHE_DIR','triton')]:os.environ[key]=str(root/'cache'/folder)
p=argparse.ArgumentParser();p.add_argument('--configuration',choices=list(PRESETS),required=True);a=p.parse_args();preset=PRESETS[a.configuration]
set_seed(741);model,groups,initial=build_train_model(preset,root);base=copy.deepcopy(model.state_dict());results=[]
x=torch.rand(2,3,240,320);labels=torch.randint(0,40,(2,240,320));targets=[{'boxes':torch.tensor([[40.,30.,150.,200.]]),'labels':torch.tensor([1]),'ignore_boxes':torch.tensor([[180.,20.,300.,220.]])} for _ in range(2)]
for device in ['cpu','mps']:
    record={'configuration':a.configuration,'device':device,'fallback_enabled':False};start=time.perf_counter()
    try:
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
        set_seed(741);model.load_state_dict(base);model.to(device).train()
        for m in model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        before={name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x.to(device)),labels.to(device)) if preset['task']=='semantic' else sum(model([i.to(device) for i in x],[{k:v.to(device) for k,v in t.items()} for t in targets]).values())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite exact-preset loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite exact-preset gradient')
            optimizer.step()
        updated={name:not torch.equal(before[name],p.detach().cpu()) for name,p in model.named_parameters() if name in before}
        enc_ids={id(p) for p in (model.encoder if preset['task']=='semantic' else model.backbone).parameters()}
        record['updated_encoder_parameters']=sum(updated[n] for n,p in model.named_parameters() if n in updated and id(p) in enc_ids)
        record['updated_head_parameters']=sum(updated[n] for n,p in model.named_parameters() if n in updated and id(p) not in enc_ids)
        if not record['updated_head_parameters'] or (preset['encoder_lr']>0 and not record['updated_encoder_parameters']):raise ValueError('Requested trainable group did not update')
        model.eval()
        with torch.inference_mode():output=model(x[:1].to(device)) if preset['task']=='semantic' else model([x[0].to(device)])
        if device=='mps':torch.mps.synchronize()
        record.update(verified=True,optimizer_steps=2,final_loss=float(loss.detach().cpu()),inference_verified=True,seconds=time.perf_counter()-start)
    except Exception as exc:record.update(verified=False,error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start)
    finally:
        model.to('cpu')
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    results.append(record);print(json.dumps(record),flush=True)
path=root/f'reports/v2/backend-presets/{a.configuration}.json'
json_write(path,{'configuration':a.configuration,'preset':preset,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'initialization_sha256':sha(initial),'sources':{f:sha(root/f) for f in ['packages/robot_model_sdk/src/robot_model_sdk/quality_models.py','packages/indoor_perception_lab/src/indoor_perception_lab/quality_training.py']},'results':results})
raise SystemExit(0 if results[0]['verified'] else 1)
