"""Independent causal-history timing; no future/target arrays are opened.

The controlled CV fixture verifies orchestration only. Learned measurements use
immutable forecast packs, a qualified device and explicit NN quiescence.
"""
from __future__ import annotations
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4
import numpy as np
import psutil
from robot_model_sdk.history import resample_causal,prepare_motion,finish_motion
from robot_reliability_bench.runner import model_identity,_configure_threads,_summary,deployed_size

FIXTURE='history-fixture'
INDEX_FIELDS=('sample_id','scene_id','split','anchor_time_s','history_source','tracker_id')


def file_sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def package_sources():
    import robot_model_sdk,robot_reliability_bench
    out={}
    for module in (robot_model_sdk,robot_reliability_bench):
        base=Path(module.__file__).parent
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json'):
                out[f'{module.__name__}/{path.relative_to(base)}']=file_sha(path)
    return out


def identity(pack):
    if str(pack)==FIXTURE:
        return {'spec':FIXTURE,'kind':'controlled_numpy_CV_fixture','sha256':file_sha(__file__),
                'total_artifact_bytes':0,'deployed_model_bytes':None,'neural_model':False}
    return {**model_identity(pack),'neural_model':True}


def history_inputs(manifest=None,split='val',limit=120):
    """Read sensor-only NPZ; strip evaluator identity fields from index rows."""
    began=time.perf_counter()
    if manifest is None:
        rng=np.random.default_rng(42);samples=[]
        for i in range(limit):
            anchor=1.+i*.1;t=anchor+np.linspace(-1,0,11);v=rng.uniform(-.3,.3,2)
            p=(rng.uniform(-1,1,2)+(t-anchor)[:,None]*v+np.sin((t-anchor)[:,None]*3)*.01).astype(np.float32)
            samples.append({'sample_id':f'controlled-history-{i:06d}','times_s':t,'positions':p,'anchor_time_s':anchor})
        source={'kind':'controlled_array_fixture','domain':'controlled causal planar histories','units':'metres',
                'coordinate_system':'world_XZ_planar','history_source':'Generated finite past positions; no sensor frontend performance claim','seed':42,
                'source_available_windows':limit,'selected_windows':limit,'labels_or_futures_opened':False}
    else:
        path=Path(manifest).resolve();meta=json.loads(path.read_text())
        selected=meta['split_files'][split]
        # Only sensor history and its index are admitted. Target filenames are not resolved.
        history=path.parent/selected['history_file'];index=path.parent/selected['index_file']
        for file,key in ((history,'history_sha256'),(index,'index_sha256')):
            if file_sha(file)!=selected[key]:raise ValueError('Causal-history source checksum mismatch')
        with np.load(history,allow_pickle=False) as archive:
            if set(archive.files)!={'history'}:raise ValueError('Sensor history archive must contain only history, never future/targets')
            h=archive['history']
        rows=[]
        for line in index.read_text().splitlines():
            raw=json.loads(line)
            rows.append({k:raw[k] for k in INDEX_FIELDS if k in raw})
        if h.shape!=(len(rows),11,2) or not np.isfinite(h).all():raise ValueError('Finite Nx11x2 sensor histories required')
        if any(r.get('split')!=split for r in rows) or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('History split/identity overlap')
        samples=[]
        for row,positions in zip(rows[:limit],h[:limit]):
            anchor=float(row['anchor_time_s'])
            samples.append({'sample_id':row['sample_id'],'times_s':anchor+np.linspace(-1,0,11),
                            'positions':positions,'anchor_time_s':anchor})
        source={'kind':'observed_sensor_history','manifest':str(path),'manifest_sha256':file_sha(path),
                'sensor_history':str(history),'sensor_history_sha256':file_sha(history),
                'sensor_index':str(index),'sensor_index_sha256':file_sha(index),'index_fields_consumed':list(INDEX_FIELDS),
                'domain':meta['domain'],'units':meta['units'],'coordinate_system':meta['coordinate_system'],
                'history_source':meta['history_source'],'source_available_windows':len(rows),'selected_windows':len(samples),
                'selection':'First up to120 unique windows in frozen split index order; repeats cycle only if fewer than100 selected',
                'labels_or_futures_opened':False,'upstream_scope':'Conditional observed-history windows; this timing excludes detector/tracker and does not establish forecast accuracy or coverage'}
    if not samples:raise ValueError('No causal histories available')
    for row in samples:
        # Anchor and increasing source clock are checked before any adapter call.
        if not np.isfinite(row['anchor_time_s']) or resample_causal(row['times_s'],row['positions'],row['anchor_time_s']) is None:
            raise ValueError('Incomplete/invalid causal history')
    return samples,source,(time.perf_counter()-began)*1000


class ControlledCV:
    task='forecast';backend='native'
    metadata={'units':'metres','coordinate_system':'world_XZ_planar'}
    def predict_timed_history(self,times,positions,anchor_time=None):
        h=resample_causal(times,positions,anchor_time)
        if h is None:return None
        t=np.linspace(-1,0,11);weights=(t-t.mean())/np.sum((t-t.mean())**2)
        velocity=np.sum(h*weights[:,None],axis=0)
        return (h[-1]+np.arange(1,21)[:,None]*.1*velocity).astype(np.float32)
    def close(self):pass


def load_history_adapter(pack,source,device):
    if str(pack)==FIXTURE:
        if device!='cpu':raise ValueError('Controlled history fixture is CPU only')
        return ControlledCV()
    from robot_model_sdk import load_adapter
    adapter=load_adapter(pack,device=device)
    if adapter.task!='forecast' or adapter.metadata['units']!=source['units'] or adapter.metadata['coordinate_system']!=source['coordinate_system']:
        adapter.close();raise ValueError('History/model task, coordinate frame or units differ')
    if adapter.metadata.get('dataset_manifest_sha256')!=source.get('manifest_sha256'):
        adapter.close();raise ValueError('History/model frozen dataset identity differs')
    return adapter


def _sync(device):
    if device=='mps':
        import torch
        torch.mps.synchronize()


def _predict(adapter,row):
    out=adapter.predict_timed_history(row['times_s'],row['positions'],row['anchor_time_s'])
    if out is None or np.asarray(out).shape!=(20,2) or not np.isfinite(out).all():raise ValueError('Finite20x2 forecast required')
    return np.asarray(out)


def worker(config):
    neural=config['model_pack']!=FIXTURE
    _configure_threads(config['threads'],native_neural=neural)
    samples,source,io_ms=history_inputs(config.get('data_manifest'),config['split'])
    before=identity(config['model_pack']);started=time.perf_counter();adapter=load_history_adapter(config['model_pack'],source,config['device'])
    _sync(config['device']);load_ms=(time.perf_counter()-started)*1000
    process=psutil.Process();result={'pid':os.getpid(),'repeat':config['repeat'],'source':source,'input_decode_total_ms':io_ms,
        'adapter_load_ms':load_ms,'requested_threads':config['threads'],'device':config['device'],
        'input_fields':['times_s','positions','anchor_time_s'],'label_or_future_input':False,
        'causal_adapter_scope':'Causal1s resampling, normalization, synchronized inference, CPU output and inverse coordinate transform',
        'model_forward_scope':'Separate synchronized neural-forward pass with resampling, normalization and tensor transfer outside clock'}
    try:
        first=time.perf_counter();first_output=_predict(adapter,samples[0]);_sync(config['device'])
        result['cold_first_prediction_ms']=(time.perf_counter()-first)*1000
        result['process_birth_to_first_prediction_ms']=(time.time()-process.create_time())*1000
        for i in range(config['warmup']):_predict(adapter,samples[i%len(samples)]);_sync(config['device'])
        measured=[];digests=[];unique=set()
        for i in range(config['samples']):
            row=samples[i%len(samples)];_sync(config['device']);total=time.perf_counter();began=time.perf_counter()
            out=_predict(adapter,row);_sync(config['device']);elapsed=(time.perf_counter()-began)*1000
            serialized=json.dumps(out.tolist(),allow_nan=False,separators=(',',':'))
            measured.append({'sample_id':row['sample_id'],'causal_adapter_ms':elapsed,'end_to_end_ms':(time.perf_counter()-total)*1000})
            digests.append(sha256(serialized.encode()).hexdigest());unique.add(row['sample_id'])
        result['sample_measurements']=measured;result['output_digests']=digests;result['unique_timed_windows']=len(unique)
        result['causal_adapter_latency']=_summary([r['causal_adapter_ms'] for r in measured])
        result['end_to_end_latency']=_summary([r['end_to_end_ms'] for r in measured])
        result['neural_forward_measurements']=[];result['forward_pass_adapter_parity']=None
        if neural:
            import torch
            result['framework_threads']={'torch_intraop':torch.get_num_threads(),'torch_interop':torch.get_num_interop_threads()}
            def tensor_for(row):
                h=resample_causal(row['times_s'],row['positions'],row['anchor_time_s'])
                x,origin,scale,single=prepare_motion(h,adapter.metadata)
                return torch.as_tensor(x,device=config['device']),origin,scale,single
            x,origin,scale,single=tensor_for(samples[0])
            with torch.inference_mode():raw=adapter.model(x).cpu().numpy()
            decoded=finish_motion(raw,origin,scale,single,adapter.metadata)
            result['forward_pass_adapter_parity']=bool(np.allclose(first_output,decoded,rtol=1e-4,atol=1e-5))
            if not result['forward_pass_adapter_parity']:raise ValueError('Isolated neural-forward pass differs from adapter output')
            for i in range(config['warmup']):
                x,_,_,_=tensor_for(samples[i%len(samples)])
                with torch.inference_mode():adapter.model(x)
                _sync(config['device'])
            for i in range(config['samples']):
                row=samples[i%len(samples)];x,_,_,_=tensor_for(row);_sync(config['device']);began=time.perf_counter()
                with torch.inference_mode():adapter.model(x)
                _sync(config['device']);result['neural_forward_measurements'].append({'sample_id':row['sample_id'],'model_ms':(time.perf_counter()-began)*1000})
        else:result['framework_threads']={'torch_intraop':None,'torch_interop':None,'fixture':'No neural network'}
        result['model_forward_latency']=_summary([r['model_ms'] for r in result['neural_forward_measurements']])
        rss=[process.memory_info().rss];stop=threading.Event()
        def monitor():
            while not stop.wait(.002):rss.append(process.memory_info().rss)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:
            for i in range(100):_predict(adapter,samples[i%len(samples)]);_sync(config['device']);rss.append(process.memory_info().rss)
        finally:stop.set();thread.join()
        result['memory_pass']={'separate_from_timing':True,'predictions':100,'rss_samples':len(rss),'sample_interval_ms':2,
            'rss_peak_observed_bytes':max(rss),'rss_end_bytes':rss[-1],'definition':'Sampled process RSS, not instantaneous allocation/device peak'}
        result['model_identity_unchanged']=identity(config['model_pack'])['sha256']==before['sha256']
        result['status']='verified' if result['model_identity_unchanged'] else 'failed'
        return result
    finally:adapter.close()


def qualification_worker(pack,device,manifest,output):
    result={'device':device,'model':identity(pack),'runtime_source_files_sha256':package_sources(),
            'label_or_future_input':False,'fallback_enabled':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='1'}
    adapter=None
    try:
        _configure_threads(1,native_neural=str(pack)!=FIXTURE)
        if result['fallback_enabled']:raise ValueError('Fallback-enabled runtime is not qualified MPS')
        samples,source,_=history_inputs(manifest,limit=1);adapter=load_history_adapter(pack,source,device)
        out=_predict(adapter,samples[0]);result.update({'source':source,'finite_inference':bool(np.isfinite(out).all())})
        if str(pack)!=FIXTURE:
            import torch
            h=resample_causal(samples[0]['times_s'],samples[0]['positions'],samples[0]['anchor_time_s'])
            x,_,_,_=prepare_motion(h,adapter.metadata);tensor=torch.as_tensor(x,device=device);loss=adapter.model(tensor).square().mean()
            if not torch.isfinite(loss).item():raise ValueError('Nonfinite diagnostic objective')
            loss.backward();grad=[p.grad for p in adapter.model.parameters() if p.grad is not None]
            if not grad or any(not torch.isfinite(g).all().item() for g in grad):raise ValueError('Missing/nonfinite gradients')
            _sync(device);result.update({'forward_backward_passed':True,'finite_gradient_tensors':len(grad),'actual_device':str(next(adapter.model.parameters()).device),'diagnostic_target_input':False})
        else:result.update({'forward_backward_passed':None,'controlled_fixture_only':True})
        result['status']='verified'
    except Exception as exc:result.update({'status':'unavailable','error':str(exc)})
    finally:
        if adapter is not None:adapter.close()
    if identity(pack)['sha256']!=result['model']['sha256']:result.update({'status':'failed','error':'Model pack changed during qualification'})
    save(output,result);return result


def qualify(pack,manifest,output,tool):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
    for device in ('cpu','mps'):
        path=output/f'{device}.json';command=[sys.executable,str(tool),'_history-qualify','--model-pack',str(pack),'--device',device,'--output',str(path)]
        if manifest:command+=['--data-manifest',str(manifest)]
        env=os.environ.copy();env['PYTORCH_ENABLE_MPS_FALLBACK']='0'
        with (output/f'{device}.log').open('w') as log:
            try:run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120);code=run.returncode
            except subprocess.TimeoutExpired:code=124
        proof=json.loads(path.read_text()) if path.exists() else {'status':'unavailable','error':'Qualification worker failed or timed out'}
        rows.append({'device':device,'exit_code':code,'command':command,'proof':proof})
    result={'status':'verified' if rows[0]['proof']['status']=='verified' else 'failed','results':rows,
            'source_tool_sha256':file_sha(tool),'neural_timing_executed':False,'qualification_scope':'Artificial squared-output backward stability only; no targets/futures/training/accuracy'}
    save(output/'qualification.json',result);return result


def run_measurement(pack,manifest,split,device,threads,output,tool):
    output=Path(output)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]);output.mkdir(parents=True,exist_ok=False)
    config={'model_pack':str(pack),'data_manifest':str(manifest) if manifest else None,'split':split,'device':device,
            'threads':threads,'batch_size':1,'samples':100,'warmup':10,'repeats':3}
    from .provenance import context,finish
    provenance=context(config,Path(tool).resolve().parents[1])
    records=[];commands=[];model=identity(pack);sources=package_sources()
    for repeat in range(3):
        conf=output/f'repeat-{repeat}-config.json';raw=output/f'repeat-{repeat}.json';save(conf,{**config,'repeat':repeat})
        command=[sys.executable,str(tool),'_history-worker','--config',str(conf),'--output',str(raw)]
        env=os.environ.copy();env.update({'OMP_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':str(threads),'MKL_NUM_THREADS':str(threads),'ROBOT_MODEL_THREADS':str(threads),'PYTORCH_ENABLE_MPS_FALLBACK':'0'})
        with (output/f'repeat-{repeat}.log').open('w') as log:
            try:run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120);code=run.returncode
            except subprocess.TimeoutExpired:code=124
        commands.append({'command':command,'exit_code':code})
        if code!=0 or not raw.exists():raise RuntimeError(f'History worker failed: {output}/repeat-{repeat}.log')
        records.append(json.loads(raw.read_text()))
    if len({r['pid'] for r in records})!=3:raise RuntimeError('Distinct three-process repeat evidence missing')
    values=[m for r in records for m in r['sample_measurements']];forward=[m['model_ms'] for r in records for m in r['neural_forward_measurements']]
    summary={'status':'verified' if all(r['status']=='verified' for r in records) else 'failed','run_id':output.name,
        'provenance':finish(provenance),'config':config,'config_sha256':sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'model':model,
        'source':records[0]['source'],'runtime_source_files_sha256':sources,'source_tool_sha256':file_sha(tool),
        'causal_adapter_latency':_summary([m['causal_adapter_ms'] for m in values]),'end_to_end_latency':_summary([m['end_to_end_ms'] for m in values]),
        'model_forward_latency':_summary(forward),'predicted_histories':len(values),'unique_timed_windows':records[0]['unique_timed_windows'],
        'throughput_fps':1000/np.mean([m['end_to_end_ms'] for m in values]),'repeat_pids':[r['pid'] for r in records],
        'repeat_evidence':[{'pid':r['pid'],'adapter_load_ms':r['adapter_load_ms'],'cold_first_prediction_ms':r['cold_first_prediction_ms'],
            'process_birth_to_first_prediction_ms':r['process_birth_to_first_prediction_ms'],'input_decode_total_ms':r['input_decode_total_ms'],'memory_pass':r['memory_pass'],'framework_threads':r['framework_threads']} for r in records],
        'commands':commands,'neural_performance_claim':str(pack)!=FIXTURE,'label_or_future_input':False,
        'model_identity_unchanged':identity(pack)['sha256']==model['sha256'],'runtime_sources_unchanged':package_sources()==sources,
        'timing_scopes':{'adapter':records[0]['causal_adapter_scope'],'forward':records[0]['model_forward_scope'],'end_to_end':'Adapter plus forecast JSON serialization; disk reads and upstream detector/tracker excluded'}}
    if str(pack)!=FIXTURE:
        metadata=json.loads((Path(pack)/'manifest.json').read_text());summary['deployed_size']=deployed_size(Path(pack),Path(pack)/metadata['native_weights'],'native')
    else:summary['deployed_size']={'bytes':None,'status':'unavailable','reason':'Controlled CV fixture has no learned artifact'}
    if not summary['model_identity_unchanged'] or not summary['runtime_sources_unchanged']:summary['status']='failed'
    save(output/'summary.json',summary)
    with (output/'measurements.csv').open('w') as stream:
        stream.write('repeat,pid,sample_id,causal_adapter_ms,end_to_end_ms\n')
        for r in records:
            for m in r['sample_measurements']:stream.write(f"{r['repeat']},{r['pid']},{m['sample_id']},{m['causal_adapter_ms']},{m['end_to_end_ms']}\n")
    return {'status':summary['status'],'summary':str(output/'summary.json'),'device':device,'threads':threads}


def matrix(args,tool):
    pack=args.model_pack
    if pack!=FIXTURE and not args.quiescent:raise ValueError('Real history model timing requires explicit NN quiescence')
    if args.split=='test' and not args.test_unlocked:raise ValueError('Held-out history source remains locked')
    if pack!=FIXTURE and not args.data_manifest:raise ValueError('Real forecast timing requires a frozen sensor-history manifest')
    proof=json.loads(Path(args.qualification).read_text());model=identity(pack);sources=package_sources()
    if proof.get('source_tool_sha256')!=file_sha(tool):raise ValueError('History qualifier/tool source changed')
    def accepted(device):
        record=next((r['proof'] for r in proof['results'] if r['device']==device),{})
        return record.get('status')=='verified' and record['model']['sha256']==model['sha256'] and record.get('runtime_source_files_sha256')==sources and not record.get('fallback_enabled')
    if not accepted('cpu'):raise ValueError('Exact CPU model/source history qualification required')
    options=[('cpu',t) for t in (1,2,4)]
    if accepted('mps'):options.append(('mps',1))
    rows=[]
    for device,threads in options:
        row=run_measurement(pack,args.data_manifest,args.split,device,threads,Path(args.output)/f'{device}-threads{threads}',tool);rows.append(row)
        save(Path(args.output)/'history-matrix-index.json',{'status':'running','runs':rows})
        if row['status']!='verified':raise RuntimeError('History timing source/model changed')
    result={'status':'verified','runs':rows,'model':model,'runtime_source_files_sha256':sources,'source_tool_sha256':file_sha(tool),
            'quiescence_attested':args.quiescent,'neural_performance_claim':pack!=FIXTURE,'label_or_future_input':False,
            'qualification':str(Path(args.qualification).resolve()),'qualification_sha256':file_sha(args.qualification)}
    save(Path(args.output)/'history-matrix-index.json',result);return result
