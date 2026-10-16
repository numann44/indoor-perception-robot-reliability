#!/usr/bin/env python3
"""Frozen-pack semantic export; 128 TRAIN calibration images, validation gates.

Held-out labels are never opened by this command. Test accuracy is a separate
locked-model evaluator, run only after the primary selection and backend settings
are frozen. Source native pack and v1 artifacts remain read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def calibration_selection(rows,count=128):
    import numpy as np
    train=[r for r in rows if r.get('split')=='train']
    if len(train)<count:raise ValueError(f'Require {count} unique TRAIN rows; got {len(train)}')
    picked=[train[i] for i in np.linspace(0,len(train)-1,count,dtype=int)]
    if len({r['sample_id'] for r in picked})!=count:raise ValueError('Calibration IDs overlap')
    return picked

def int8_default_gate(fp32_miou,int8_miou,fp32_p95=None,int8_p95=None):
    loss=float(fp32_miou)-float(int8_miou)
    accurate=loss<=.01
    timed=fp32_p95 is not None and int8_p95 is not None
    faster=timed and float(int8_p95)<=float(fp32_p95)
    return {'eligible':bool(accurate and faster),'validation_miou_loss':loss,'loss_limit':.01,
            'accuracy_gate_passed':accurate,'latency_gate_measured':timed,'p95_gate_passed':bool(faster),
            'policy':'Default INT8 is blocked by >0.01 validation mIoU loss, any worse validation p95, or absent timing evidence'}

def gate_from_matrix(matrix_path,validation_path,output):
    """Join validation accuracy to predeclared CPU4 clean timing; no test data."""
    matrix_path,validation_path=Path(matrix_path),Path(validation_path)
    matrix=json.loads(matrix_path.read_text());validation=json.loads(validation_path.read_text())
    if matrix.get('status')!='verified' or validation.get('split')!='validation':
        raise ValueError('Verified matrix and validation-only accuracy evidence required')
    summaries={}
    for precision,packname in [('fp32','semantic-onnx-fp32'),('int8','semantic-onnx-int8')]:
        row=next((r for r in matrix['runs'] if Path(r['model_pack']).name==packname and r['backend']=='onnx' and r['device']=='cpu' and r['threads']==4),None)
        if row is None:raise ValueError('Predeclared ONNX CPU4 timing run missing')
        path=Path(row['summary']);summary=json.loads(path.read_text())
        if summary['status']!='verified' or summary['config'].get('split')!='val':
            raise ValueError('INT8 default may only use validation timing')
        if summary['model']['sha256']!=validation['model_pack_identities'][precision]['sha256']:
            raise ValueError('Timing and validation accuracy refer to different model pack bytes')
        if summary['split_or_input_sha256']!=digest(Path(validation['source']['manifest'])):
            raise ValueError('Timing and accuracy refer to different validation index bytes')
        clean=summary['conditions']['clean']
        if clean['process_repeats']!=3 or clean['predicted_frames']<300:
            raise ValueError('Require3fresh processes and100clean measurements each')
        summaries[precision]={'path':str(path),'sha256':digest(path),'run_id':summary['run_id'],
                              'model':summary['model'],'adapter_p95_ms':clean['adapter_latency']['p95_ms']}
    gate=int8_default_gate(validation['metrics']['fp32']['mIoU'],validation['metrics']['int8']['mIoU'],
                           summaries['fp32']['adapter_p95_ms'],summaries['int8']['adapter_p95_ms'])
    result={'status':'verified','default_INT8_eligible':gate['eligible'],'gate':gate,'split':'validation',
            'timing_selection':'Predeclared ONNX CPU4 clean condition; no fastest-thread or heldout selection',
            'validation_accuracy':{'path':str(validation_path),'sha256':digest(validation_path)},
            'matrix':{'path':str(matrix_path),'sha256':digest(matrix_path)},'timings':summaries,
            'deployment_effect':'Decision evidence only; model manifests and weights are unchanged',
            'test_data_used':False}
    save(output,result);return result

def experiment(args):
    import cv2
    import numpy as np
    import torch
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader,CalibrationMethod,QuantFormat,QuantType,quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from robot_model_sdk import load_adapter
    from robot_model_sdk.preprocessing import quality_rgb,quality_logits_to_mask
    from robot_reliability_bench.inputs import load_frames,_path
    from robot_reliability_bench.metrics import semantic_metrics
    from robot_reliability_bench.runner import model_identity
    from robot_reliability_bench.robotics_v2.provenance import context,finish
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(2)
    source=Path(args.model_pack).resolve();index=Path(args.input_manifest).resolve()
    before=model_identity(source);meta=json.loads((source/'manifest.json').read_text())
    if meta['task']!='semantic':raise ValueError('Semantic model pack required')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-export-'+uuid.uuid4().hex[:6]
    output=Path(args.output).resolve()/run_id;output.mkdir(parents=True,exist_ok=True)
    provenance=context({'model_pack':str(source),'model_pack_sha256':before['sha256'],'input_index_sha256':digest(index),'calibration_images':128,'calibration_split':'train','validation_only':True,'opset':18,'rtol':1e-4,'atol':1e-5},ROOT)
    rows=[json.loads(s) for s in index.read_text().splitlines() if s.strip()]
    picked=calibration_selection(rows)
    calibration_index=output/'calibration-sensor-inputs.jsonl'
    calibration_index.write_text(''.join(json.dumps({'sample_id':r['sample_id'],'split':'train','rgb_path':str(_path(index,r['rgb_path'])),'timestamp':r.get('timestamp')})+'\n' for r in picked))
    calibration,_,_=load_frames(calibration_index,count=128,split='train')
    validation,_,valsource=load_frames(index,count=sum(r.get('split') in ('val','validation') for r in rows),split='val')
    if not validation:raise ValueError('Independent validation split required')
    validation_ids=[frame.sample_id for frame in validation]
    if len(set(validation_ids))!=len(validation_ids):raise ValueError('Duplicate validation sample IDs')
    # Labels are explicitly evaluation-only. Calibration never uses this map.
    valrows={r['sample_id']:r for r in rows if r.get('split') in ('val','validation')}
    if len(valrows)!=len(validation_ids) or set(valrows)!=set(validation_ids):raise ValueError('Validation frame/target IDs do not match exactly')
    targets={};target_hashes={}
    for frame in validation:
        row=valrows[frame.sample_id];name=row.get('mask_path') or row.get('targets_path') or row.get('semantic_path')
        if name is None:raise ValueError('Validation semantic mask path absent')
        path=_path(index,name);mask=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim!=2 or mask.shape!=frame.rgb.shape[:2] or not np.isin(mask,np.r_[np.arange(40),255]).all():raise ValueError(f'Invalid NYU40 validation mask {path}')
        targets[frame.sample_id]=mask;target_hashes[str(path)]=digest(path)
    def preprocess(frame):
        size=meta.get('input_size',[240,320]);mean=meta.get('mean',[.485,.456,.406]);std=meta.get('std',[.229,.224,.225])
        if meta.get('preprocessing_id')=='quality_rgb_v2':return quality_rgb(frame.rgb,size,mean,std)[None]
        rgb=cv2.resize(frame.rgb,(size[1],size[0])).astype(np.float32)/255
        return np.ascontiguousarray(((rgb-np.array(mean,np.float32))/np.array(std,np.float32)).transpose(2,0,1)[None])
    calibration_manifest=[{'sample_id':r['sample_id'],'split':'train','rgb_path':str(_path(index,r['rgb_path'])),'rgb_sha256':digest(_path(index,r['rgb_path']))} for r in picked]
    save(output/'calibration-manifest.json',calibration_manifest)
    fpdir=ROOT/'models/v2/semantic-onnx-fp32';qdir=ROOT/'models/v2/semantic-onnx-int8'
    def clone(destination,precision):
        destination.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/meta['native_weights'],destination/meta['native_weights'])
        derived={**meta,'onnx':'model.onnx','onnx_precision':precision,'precision':precision,'native_precision':'fp32',
                 'parent_pack':str(source),'parent_native_sha256':meta['sha256'],'deployment_default':False,
                 'calibration_manifest_sha256':digest(output/'calibration-manifest.json')}
        return derived
    fpmeta=clone(fpdir,'fp32');qmeta=clone(qdir,'int8')
    native=load_adapter(source,device='cpu')
    if meta['task']!='semantic':raise ValueError('Semantic adapter required')
    try:
        native.model.eval();example=torch.from_numpy(preprocess(calibration[0]))
        torch.onnx.export(native.model,example,fpdir/'model.onnx',input_names=['rgb'],output_names=['logits'],opset_version=18,dynamo=False,external_data=False)
        fpmeta['onnx_sha256']=digest(fpdir/'model.onnx');save(fpdir/'manifest.json',fpmeta)
        optimized=output/'calibration-preprocessed.onnx'
        quant_pre_process(str(fpdir/'model.onnx'),str(optimized),skip_symbolic_shape=True,skip_onnx_shape=False)
        class Reader(CalibrationDataReader):
            def __init__(self):self.i=0
            def get_next(self):
                if self.i==len(calibration):return None
                frame=calibration[self.i];self.i+=1;return {'rgb':preprocess(frame)}
            def rewind(self):self.i=0
        quantize_static(str(optimized),str(qdir/'model.onnx'),Reader(),quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,
                        calibrate_method=CalibrationMethod.MinMax)
        qmeta['onnx_sha256']=digest(qdir/'model.onnx');save(qdir/'manifest.json',qmeta)
        fp=load_adapter(fpdir,backend='onnx');quant=load_adapter(qdir,backend='onnx')
        try:
            native_predictions=[];fp_predictions=[];quant_predictions=[];truth=[];parity=[]
            for frame in validation:
                a=native.predict(frame);b=fp.predict(frame);q=quant.predict(frame)
                close=bool(np.allclose(a.logits,b.logits,rtol=1e-4,atol=1e-5))
                parity.append({'sample_id':frame.sample_id,'allclose':close,'maximum_absolute_logit_error':float(np.max(np.abs(a.logits-b.logits))),
                               'full_semantic_mask_pixel_agreement':float(np.mean(a.semantic==b.semantic))})
                native_predictions.append(a.semantic);fp_predictions.append(b.semantic);quant_predictions.append(q.semantic);truth.append(targets[frame.sample_id])
            # The metric API keys masks by sample ID so missing/extra predictions
            # remain detectable; preserve the same validation frame ordering here.
            truth_by_id={frame.sample_id:mask for frame,mask in zip(validation,truth,strict=True)}
            metrics={name:semantic_metrics({frame.sample_id:mask for frame,mask in zip(validation,predictions,strict=True)},truth_by_id,40) for name,predictions in [('native',native_predictions),('fp32',fp_predictions),('int8',quant_predictions)]}
            gate=int8_default_gate(metrics['fp32']['mIoU'],metrics['int8']['mIoU'])
            save(output/'fp32-parity.json',{'rtol':1e-4,'atol':1e-5,'all_validation_frames_passed':all(r['allclose'] for r in parity),'frames':parity})
            save(output/'validation.json',{'split':'validation','images':len(validation),'source':valsource,'target_files_sha256':target_hashes,'model_pack_identities':{'native':before,'fp32':model_identity(fpdir),'int8':model_identity(qdir)},'metrics':metrics,'int8_default_gate':gate})
            result={'status':'verified' if all(r['allclose'] for r in parity) else 'failed','provenance':finish(provenance),
                    'primary_pack_unchanged':model_identity(source)['sha256']==before['sha256'],'calibration_count':len(calibration),
                    'validation_count':len(validation),'test_labels_opened':False,'model_packs':{'fp32':str(fpdir),'int8':str(qdir)},
                    'deployed_bytes':{'fp32':(fpdir/'model.onnx').stat().st_size,'int8':(qdir/'model.onnx').stat().st_size},
                    'int8_default_gate':gate,'heldout_policy':'No test reads; selected source architecture and calibration were fixed before validation evaluation'}
            save(output/'run.json',result);save(Path(args.output)/'latest.json',{'run_id':run_id,'result':str(output/'run.json')})
            if not result['primary_pack_unchanged']:raise RuntimeError('Primary model pack changed during export')
            if result['status']!='verified':raise RuntimeError('FP32 validation parity failed')
            return result
        finally:fp.close();quant.close()
    finally:native.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-pack');p.add_argument('--input-manifest');p.add_argument('--output',default=str(ROOT/'reports/v2/optimization'))
    p.add_argument('--gate-from-matrix');p.add_argument('--validation-result')
    args=p.parse_args()
    if args.gate_from_matrix:
        if not args.validation_result:p.error('--validation-result is required with --gate-from-matrix')
        result=gate_from_matrix(args.gate_from_matrix,args.validation_result,Path(args.output)/'int8-default-decision.json')
    else:
        if not args.model_pack or not args.input_manifest:p.error('--model-pack and --input-manifest are required for export')
        result=experiment(args)
    print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
