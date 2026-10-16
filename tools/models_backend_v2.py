"""Bounded architecture-specific CPU/MPS backward/update/inference acceptance."""
from pathlib import Path
import os,json,time,traceback,copy
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import torch
from robot_model_sdk.models import semantic_network
from robot_model_sdk.quality_models import MultiScaleSemanticNet,detection_model,residual_motion_network
ROOT=Path(os.environ['PORTFOLIO_ROOT'])
torch.set_num_threads(4);torch.set_num_interop_threads(1)
results=[]
for name in ['semantic_legacy','semantic_multiscale','ssdlite','fasterrcnn','forecast']:
    torch.manual_seed(42)
    if name=='semantic_legacy':model=semantic_network()
    elif name=='semantic_multiscale':model=MultiScaleSemanticNet()
    elif name=='forecast':model=residual_motion_network()
    else:model=detection_model('ssdlite320_quality_v2' if name=='ssdlite' else 'fasterrcnn_mobilenet_v3_large_320_fpn_quality_v2')
    base=copy.deepcopy(model.state_dict())
    if name in ['ssdlite','fasterrcnn']:
        for p in model.backbone.parameters():p.requires_grad=False
    images=torch.rand(2,3,240,320);labels=torch.randint(0,40,(2,240,320));history=torch.rand(2,11,2);future=torch.rand(2,20,2)
    targets=[{'boxes':torch.tensor([[40.,30.,150.,200.]]),'labels':torch.tensor([1]),'ignore_boxes':torch.tensor([[180.,20.,300.,220.]])} for _ in range(2)]
    cpu_loss=None
    for device in ['cpu','mps']:
        record={'architecture':name,'device':device,'fallback_enabled':False};start=time.perf_counter()
        try:
            if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
            model.load_state_dict(base);model.to(device).train()
            for module in model.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
            before={k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                if name in ['ssdlite','fasterrcnn']:
                    output=model([x.to(device) for x in images],[{k:v.to(device) for k,v in t.items()} for t in targets]);loss=sum(output.values())
                elif name=='forecast':loss=torch.nn.functional.smooth_l1_loss(model(history.to(device)),future.to(device))
                else:loss=torch.nn.functional.cross_entropy(model(images.to(device)),labels.to(device))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                if step==0:
                    record['initial_loss']=float(loss.detach().cpu())
                    if device=='cpu':cpu_loss=record['initial_loss']
                    elif abs(cpu_loss-record['initial_loss'])>max(.001,abs(cpu_loss)*.005):raise ValueError('CPU/MPS initial loss difference >0.5%')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradients')
                opt.step()
            record['updated_parameters']=sum(not torch.equal(before[k],p.detach().cpu()) for k,p in model.named_parameters() if k in before)
            if record['updated_parameters']==0:raise ValueError('No trainable parameter updated')
            model.eval()
            with torch.inference_mode():
                output=model([images[0].to(device)]) if name in ['ssdlite','fasterrcnn'] else model(history[:1].to(device) if name=='forecast' else images[:1].to(device))
            if device=='mps':torch.mps.synchronize()
            record.update(verified=True,forward_backward_optimizer_steps=2,inference_verified=True,seconds=time.perf_counter()-start)
        except Exception as exc:record.update(verified=False,error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start)
        finally:
            model.to('cpu')
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        results.append(record);print(json.dumps(record),flush=True)
    del model
(ROOT/'reports/v2/backend-models.json').write_text(json.dumps({'fallback_enabled':False,'scope':'architecture-specific, initialized probe; trained artifact acceptance is separate','results':results},indent=2))
if not all(r['verified'] for r in results if r['device']=='cpu'):raise SystemExit(1)
