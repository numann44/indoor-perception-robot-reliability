"""Standalone lab CLI: training/evaluation with a local data root; RGB-only inference."""
from pathlib import Path
import argparse,json,sys

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('demo');q.add_argument('--fixture',action='store_true');q.add_argument('--dataset',choices=['tum','jaad'],default='tum');q.add_argument('--sequence');q.add_argument('--frames',type=int,default=700);q.add_argument('--output',type=Path,required=True);q.add_argument('--detector',default='fixture');q.add_argument('--segmenter');q.add_argument('--forecaster');q.add_argument('--device',default='cpu')
    q=sub.add_parser('infer');q.add_argument('--model-pack',type=Path,required=True);q.add_argument('--rgb',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--device',default='cpu')
    for command in ['train','evaluate']:
        q=sub.add_parser(command);q.add_argument('--task',choices=['semantic','detection','forecast'],required=True);q.add_argument('--domain',default='fixture');q.add_argument('--device',default='cpu')
        q.add_argument('--protocol',choices=['v1','quality-v2'],default='v1')
        if command=='train':q.add_argument('--seed',type=int,default=42);q.add_argument('--lr',type=float,default=.001);q.add_argument('--epochs',type=int);q.add_argument('--max-steps',type=int);q.add_argument('--configuration',choices=['S1','S2','S3','D1','D2','D3','F1','F2','F3'])
        else:q.add_argument('--model-pack',type=Path,required=True);q.add_argument('--split',choices=['train','val','test'],default='test');q.add_argument('--dataset',choices=['nyu','jaad','mot17','hypersim']);q.add_argument('--selection-lock',type=Path);q.add_argument('--baseline',choices=['model','train-majority'],default='model')
    a=p.parse_args()
    if a.command=='infer':
        import numpy as np,torch
        from PIL import Image
        from robot_model_sdk import SensorFrameV1,load_adapter
        torch.set_num_threads(4);frame=SensorFrameV1(a.rgb.name,0,np.array(Image.open(a.rgb).convert('RGB')));adapter=load_adapter(a.model_pack,device=a.device);out=adapter.predict(frame);a.output.parent.mkdir(parents=True,exist_ok=True);result=out.to_json()
        if out.semantic is not None:
            mask=a.output.with_suffix('.mask.png');Image.fromarray(out.semantic).save(mask);result['semantic_mask_path']=str(mask)
        a.output.write_text(json.dumps(result));adapter.close();print(json.dumps({'output':str(a.output),'task':adapter.task}));return 0
    if a.command=='demo':
        from robot_model_sdk import fixture_frames
        from .pipeline import replay
        if a.frames<1:raise ValueError('Positive frame count required')
        if a.fixture:frames=fixture_frames(a.frames)
        else:
            from .experiments import root_path
            from .datasets import read_index,sensor_from_row
            root=root_path();rows=read_index(root,a.dataset);sequence=a.sequence or rows[0]['sequence_id'];frames=(sensor_from_row(root,r) for r in [r for r in rows if r['sequence_id']==sequence][:a.frames])
        if a.detector!='fixture' or a.segmenter or a.forecaster:
            import torch;torch.set_num_threads(4)
        print(json.dumps(replay(frames,a.output,a.detector,a.segmenter,a.device,forecaster=a.forecaster)));return 0
    if a.protocol=='quality-v2':
        if a.task=='forecast':
            from .forecast_v2 import train_forecast_v2,evaluate_forecast_v2
            if a.command=='train':
                if a.device!='cpu':raise ValueError('quality-v2 forecast training supports CPU; MPS inference is separately validated')
                if a.configuration and not a.configuration.startswith('F'):raise ValueError('Forecast requires F1/F2/F3')
                lr={'F1':.001,'F2':.0003,'F3':.0001}.get(a.configuration,a.lr)
                if a.configuration and any(token=='--lr' or token.startswith('--lr=') for token in sys.argv[2:]) and a.lr!=lr:raise ValueError('Explicit learning rate disagrees with the frozen forecasting configuration')
                result=str(train_forecast_v2(a.domain,a.seed,lr,epochs=30 if a.epochs is None else a.epochs,max_steps=6000 if a.max_steps is None else a.max_steps))
            else:
                if a.device!='cpu' or a.baseline!='model' or a.dataset is not None:raise ValueError('Forecast evaluation supports CPU/model/domain options only')
                result=evaluate_forecast_v2(a.model_pack,a.domain,a.split,a.selection_lock)
        elif a.command=='train':
            from .quality_training import train_quality,PRESETS
            if any(token=='--domain' or token.startswith('--domain=') for token in sys.argv[2:]):raise ValueError('Perception dataset is fixed by --task; --domain belongs to forecasting')
            if any(token=='--lr' or token.startswith('--lr=') for token in sys.argv[2:]):raise ValueError('quality-v2 perception learning rates are fixed by --configuration')
            configuration=a.configuration or ('S1' if a.task=='semantic' else 'D1')
            if configuration not in PRESETS or PRESETS[configuration]['task']!=a.task:raise ValueError('Configuration does not match task')
            pack,run=train_quality(configuration,a.seed,a.device,30 if a.epochs is None else a.epochs,6000 if a.max_steps is None else a.max_steps);result={'model_pack':str(pack),'run':str(run)}
        else:
            from .quality_release import evaluate_quality
            if any(token=='--domain' or token.startswith('--domain=') for token in sys.argv[2:]):raise ValueError('Perception evaluation uses --dataset; --domain belongs to forecasting')
            if a.selection_lock is not None:raise ValueError('Perception evaluation uses the canonical task selection lock')
            result=evaluate_quality(a.task,a.model_pack,a.dataset or ('nyu' if a.task=='semantic' else 'jaad'),a.split,a.device,a.baseline)
        print(json.dumps({'result':result}));return 0
    if a.command=='train':
        if a.epochs is None:a.epochs=15
        if a.max_steps is None:a.max_steps=600
        if a.task=='forecast':
            from .motion_training import train_motion
            result=train_motion(a.domain,a.seed,a.lr,a.max_steps,a.epochs)
        else:
            from .training import train
            result=train(a.task,a.seed,a.lr,a.epochs,a.max_steps,a.device)
    else:
        if a.task=='forecast':
            from .motion_training import evaluate_motion
            result=evaluate_motion(a.model_pack,a.domain,a.split)
        else:
            import torch;torch.set_num_threads(4)
            from .release import evaluate_release
            result=evaluate_release(a.task,a.model_pack,a.split,device=a.device)
    print(json.dumps({'result':result}));return 0

if __name__=='__main__':raise SystemExit(main())
