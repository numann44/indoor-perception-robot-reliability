#!/usr/bin/env python3
"""Qualified v2 backend proofs and independent fresh-process timing sweeps."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def runtime_sources():
    import robot_model_sdk
    import robot_reliability_bench
    source={}
    for module in (robot_model_sdk,robot_reliability_bench):
        base=Path(module.__file__).parent
        for path in base.rglob('*'):
            if not path.is_file() or path.suffix not in ('.py','.json'):continue
            source[f'{module.__name__}/{path.relative_to(base)}']=hashlib.sha256(path.read_bytes()).hexdigest()
    return source

def qualify_worker(pack,device,output,input_manifest=None):
    import numpy as np
    import torch
    import cv2
    from robot_model_sdk import load_adapter,fixture_frames
    from robot_reliability_bench.inputs import load_frames
    from robot_reliability_bench.runner import model_identity
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    before=model_identity(pack);started=time.time();adapter=None
    result={'model':before,'device':device,'backend':'native','controlled_backend_stability_only':True,
            'label_access':False,'pid':os.getpid(),'fallback_enabled':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='1'}
    result['runtime_source_files_sha256']=runtime_sources()
    try:
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
        frame=load_frames(input_manifest,count=1,split='val')[0][0] if input_manifest else next(iter(fixture_frames(n=1)))
        adapter=load_adapter(pack,device=device)
        prediction=adapter.predict(frame)
        if prediction.logits is not None and not np.isfinite(prediction.logits).all():raise RuntimeError('Nonfinite semantic inference')
        model=adapter.model
        if adapter.task=='semantic':
            metadata=adapter.metadata;size=metadata.get('input_size',[240,320])
            if metadata.get('preprocessing_id')=='quality_rgb_v2':
                from robot_model_sdk.preprocessing import quality_rgb
                x=quality_rgb(frame.rgb,tuple(size),metadata['mean'],metadata['std'])
            else:
                x=cv2.resize(frame.rgb,(size[1],size[0])).astype(np.float32)/255
                x=((x-np.array(metadata['mean'],np.float32))/np.array(metadata['std'],np.float32)).transpose(2,0,1).copy()
            tensor=torch.from_numpy(x).unsqueeze(0).to(device)
            model.eval();loss=model(tensor).square().mean()
        elif adapter.task=='detection':
            model.train()
            for module in model.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
            tensor=torch.from_numpy(frame.rgb.transpose(2,0,1).copy()).float().to(device)/255
            h,w=frame.rgb.shape[:2]
            # Artificial target exists only for backend backward stability, not
            # data training, validation, inference or task-accuracy claims.
            target={'boxes':torch.tensor([[.2*w,.2*h,.8*w,.8*h]],device=device),'labels':torch.tensor([1],dtype=torch.int64,device=device)}
            losses=model([tensor],[target]);loss=sum(losses.values())
        else:raise ValueError('Frame backend qualification is detection/semantic only')
        if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite backward objective')
        loss.backward();gradients=[p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):raise RuntimeError('Missing/nonfinite gradients')
        if device=='mps':torch.mps.synchronize()
        result.update({'status':'verified','forward_backward_passed':True,'finite_gradient_tensors':len(gradients),
                       'inference_output':prediction.to_json(),'objective':float(loss.detach().cpu()),'actual_device':str(next(model.parameters()).device)})
    except Exception as exc:result.update({'status':'unavailable','forward_backward_passed':False,'error':str(exc),'traceback':traceback.format_exc()})
    finally:
        if adapter is not None:adapter.close()
    result.update({'elapsed_seconds':time.time()-started,'finished_unix':time.time(),'source_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if model_identity(pack)['sha256']!=before['sha256']:result.update({'status':'failed','error':'Model pack changed during backend proof'})
    save(output,result);return 0 if result['status']=='verified' else 1

def qualify(pack,output,input_manifest=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);records=[]
    for device in ('cpu','mps'):
        path=output/f'{device}.json';command=[sys.executable,str(Path(__file__).resolve()),'_qualify','--model-pack',pack,'--device',device,'--output',str(path)]
        if input_manifest:command.extend(['--input-manifest',input_manifest])
        environment=os.environ.copy();environment['PYTORCH_ENABLE_MPS_FALLBACK']='0'
        with (output/f'{device}.log').open('w') as log:
            try:r=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=120);code=r.returncode
            except subprocess.TimeoutExpired:code=124
        proof=json.loads(path.read_text()) if path.exists() else {'status':'unavailable','error':'Worker timed out/failed before result'}
        records.append({'device':device,'exit_code':code,'command':command,'proof':proof})
    result={'status':'verified' if records[0]['proof']['status']=='verified' else 'failed','results':records,'MPS_policy':'Used only when this exact checkpoint pack passes CPU/MPS forward and backward without fallback'}
    save(output/'qualification.json',result);return result

def matrix(args):
    from robot_reliability_bench.runner import run_benchmark,model_identity
    from robot_reliability_bench.corruptions import CONDITIONS
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if not args.quiescent:raise ValueError('Formal timing requires an explicit training-quiescent attestation')
    if args.split=='test' and not args.test_unlocked:raise ValueError('Held-out test remains locked until model selection is frozen')
    proof=json.loads(Path(args.qualification).read_text())
    identity=model_identity(args.model_pack)
    cpu=next(r for r in proof['results'] if r['device']=='cpu')['proof']
    if cpu.get('status')!='verified' or cpu['model']['sha256']!=identity['sha256']:raise ValueError('CPU qualification is absent or stale for this model pack')
    if any(r['proof'].get('fallback_enabled') for r in proof['results']):raise ValueError('CPU fallback is not qualified MPS execution')
    if cpu.get('source_tool_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Backend proof tool source differs from timing tool')
    source_hashes=runtime_sources()
    if cpu.get('runtime_source_files_sha256')!=source_hashes:
        raise ValueError('Backend proof runtime source differs from timing runtime')
    selected=[(args.model_pack,'native','cpu',t) for t in (1,2,4)]
    mps=next((r['proof'] for r in proof['results'] if r['device']=='mps'),{})
    if mps.get('status')=='verified' and mps['model']['sha256']==identity['sha256']:
        if mps.get('runtime_source_files_sha256')!=source_hashes:raise ValueError('MPS qualification runtime source differs from timing runtime')
        selected.append((args.model_pack,'native','mps',1))
    for pack in (args.onnx_fp32,args.onnx_int8):
        if pack:
            native_manifest=json.loads((Path(args.model_pack)/'manifest.json').read_text())
            derived_manifest=json.loads((Path(pack)/'manifest.json').read_text())
            if derived_manifest.get('parent_native_sha256')!=native_manifest['sha256']:
                raise ValueError('ONNX timing pack is not derived from the qualified native checkpoint')
            selected.extend((pack,'onnx','cpu',t) for t in (1,2,4))
    rows=[]
    for pack,backend,device,threads in selected:
        # Full trained corruptions at predeclared native-CPU4 and ONNX-CPU4;
        # other thread/device clean sweeps are not described as full matrices.
        conditions=CONDITIONS if args.conditions=='all' and device=='cpu' and threads==4 else ['clean']
        label=f'{Path(pack).name}-{backend}-{device}-threads{threads}'
        os.environ['ROBOT_MODEL_THREADS']=str(threads)
        result=run_benchmark(pack,output/label,backend=backend,device=device,samples=100,warmup=10,repeats=3,threads=threads,
                             input_manifest=args.input_manifest,split=args.split,conditions=conditions,timeout_s=1800)
        rows.append({'label':label,'summary':str(Path(result['output_directory'])/'summary.json'),'status':result['status'],'model_pack':pack,'backend':backend,'device':device,'threads':threads,'conditions':conditions})
        if result['status']!='verified':raise RuntimeError(f'Actual benchmark failed: {label}; inspect its summary/logs')
        save(output/'matrix-index.json',{'status':'running','runs':rows,'runtime_source_files_sha256':source_hashes,'source_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    result={'status':'verified','runs':rows,'training_quiescent_attested':True,'test_unlock_attested':args.test_unlocked,
            'qualification':str(Path(args.qualification).resolve()),'source_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['runtime_source_files_sha256']=source_hashes
    save(output/'matrix-index.json',result);return result

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for verb in ('qualify','_qualify'):
        a=sub.add_parser(verb);a.add_argument('--model-pack',required=True);a.add_argument('--output',required=True);a.add_argument('--input-manifest');a.add_argument('--device',default='cpu',choices=('cpu','mps'))
    m=sub.add_parser('matrix');m.add_argument('--model-pack',required=True);m.add_argument('--onnx-fp32');m.add_argument('--onnx-int8');m.add_argument('--input-manifest',required=True);m.add_argument('--qualification',required=True);m.add_argument('--output',required=True);m.add_argument('--split',default='val',choices=('val','test'));m.add_argument('--conditions',default='all',choices=('all','clean'));m.add_argument('--quiescent',action='store_true');m.add_argument('--test-unlocked',action='store_true')
    a=sub.add_parser('accuracy');a.add_argument('--model-pack',required=True);a.add_argument('--input-manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--task',choices=('detection','semantic'),default='semantic');a.add_argument('--backend',default='native',choices=('native','onnx'));a.add_argument('--device',default='cpu',choices=('cpu','mps'));a.add_argument('--split',default='val',choices=('val','test'));a.add_argument('--test-unlocked',action='store_true');a.add_argument('--conditions',default='all',choices=('all','clean'))
    for verb in ('history-qualify','_history-qualify'):
        h=sub.add_parser(verb);h.add_argument('--model-pack',required=True);h.add_argument('--data-manifest');h.add_argument('--output',required=True);h.add_argument('--device',default='cpu',choices=('cpu','mps'))
    h=sub.add_parser('history-matrix');h.add_argument('--model-pack',required=True);h.add_argument('--data-manifest');h.add_argument('--qualification',required=True);h.add_argument('--output',required=True);h.add_argument('--split',default='val',choices=('val','test'));h.add_argument('--quiescent',action='store_true');h.add_argument('--test-unlocked',action='store_true')
    h=sub.add_parser('_history-worker');h.add_argument('--config',required=True);h.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command.startswith('history-') or args.command.startswith('_history-'):
        from robot_reliability_bench.robotics_v2 import history_benchmark as history
        tool=Path(__file__).resolve()
        if args.command=='_history-worker':
            result=history.worker(json.loads(Path(args.config).read_text()));history.save(args.output,result)
        elif args.command=='_history-qualify':
            result=history.qualification_worker(args.model_pack,args.device,args.data_manifest,args.output)
        elif args.command=='history-qualify':result=history.qualify(args.model_pack,args.data_manifest,args.output,tool)
        else:result=history.matrix(args,tool)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sample_measurements','output_digests','neural_forward_measurements','runtime_source_files_sha256','results')}));return 0 if result['status']=='verified' else 1
    if args.command=='accuracy':
        from robot_reliability_bench.robotics_v2 import quality_accuracy
        from robot_reliability_bench.robotics_v2.provenance import context
        # Noneditable installations cannot infer the project from __file__.
        # Bind provenance to this launcher's root; evaluator/model math is unchanged.
        def project_context(config):
            proof=context(config,root=ROOT)
            expected_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            if not proof.get('source_sha256') or proof.get('git_commit')!=expected_commit:
                raise ValueError('Accuracy provenance must bind the nonempty project source tree and current project Git identity')
            proof['installed_runtime_source_files_sha256']=runtime_sources()
            return proof
        quality_accuracy.context=project_context
        result=quality_accuracy.evaluate(args.model_pack,args.input_manifest,args.output,args.task,args.backend,args.device,args.split,args.test_unlocked,None if args.conditions=='all' else ['clean'])
        print(json.dumps({'status':result['status'],'images':result['images'],'output':args.output}));return 0
    if args.command=='_qualify':return qualify_worker(args.model_pack,args.device,args.output,args.input_manifest)
    result=qualify(args.model_pack,args.output,args.input_manifest) if args.command=='qualify' else matrix(args)
    print(json.dumps(result));return 0 if result['status']=='verified' else 1
if __name__=='__main__':raise SystemExit(main())
