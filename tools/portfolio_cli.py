from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def explicit_option(flag):
    return any(token==flag or token.startswith(flag+'=') for token in sys.argv[2:])

def main():
    # These verbs own their parsers; dispatch flags before argparse.REMAINDER.
    if len(sys.argv)>1 and sys.argv[1] in {'simulate','ros2'}:
        action=sys.argv[1];args=sys.argv[2:]
        if args[:1]==['--']:args=args[1:]
        if action=='simulate' and '--active' in args:
            action='simulate-active';args=[x for x in args if x!='--active']
        return subprocess.call([sys.executable,str(ROOT/'tools/robotics_v2.py'),action,*args])
    p=argparse.ArgumentParser(description="Independent robotics research portfolio; local only")
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("bootstrap");q=sub.add_parser("fetch");q.add_argument("--datasets",nargs="+",default=["jaad","hypersim","tum"])
    q=sub.add_parser("train");q.add_argument("--task",choices=["semantic","detection","forecast"],required=True);q.add_argument("--seed",type=int,default=42);q.add_argument("--lr",type=float,default=.001);q.add_argument("--epochs",type=int);q.add_argument("--max-steps",type=int);q.add_argument("--device",default="cpu");q.add_argument("--index");q.add_argument("--checkpoint");q.add_argument("--unfreeze-last",action="store_true");q.add_argument("--domain",choices=["jaad","fixture"],default="jaad")
    q=sub.add_parser("demo");q.add_argument("--fixture",action="store_true");q.add_argument("--dataset",choices=["tum","jaad"],default="tum");q.add_argument("--sequence");q.add_argument("--frames",type=int,default=700);q.add_argument("--stride",type=int,default=1);q.add_argument("--output");q.add_argument("--chapters",nargs='+');q.add_argument("--detector",default="models/detection-trained");q.add_argument("--segmenter");q.add_argument("--forecaster");q.add_argument('--score-threshold',type=float,default=.3);q.add_argument("--device",default="cpu")
    q=sub.add_parser("evaluate");q.add_argument("--task",choices=["semantic","detection","forecast"],required=True);q.add_argument("--model-pack",required=True);q.add_argument("--split",choices=["train","val","test"],default="test");q.add_argument("--index");q.add_argument("--device",default="cpu");q.add_argument("--domain",default="jaad")
    q=sub.add_parser("benchmark");q.add_argument("args",nargs=argparse.REMAINDER)
    q=sub.add_parser("verify");q.add_argument("--quick",action="store_true")
    q=sub.add_parser("resume");q.add_argument("--checkpoint",required=True)
    q=sub.add_parser("select");q.add_argument("--task",choices=["semantic","detection","forecast"],required=True);q.add_argument('--domain',choices=['jaad','fixture'],default='jaad')
    for name in ('bootstrap','train','evaluate','demo','select','fetch','verify','benchmark'):
        parser=sub.choices[name];parser.add_argument('--protocol',choices=['v1','quality-v2'],default='v1')
    sub.choices['train'].add_argument('--configuration',choices=['S1','S2','S3','D1','D2','D3','F1','F2','F3'])
    sub.choices['train'].add_argument('--overfit',action='store_true')
    sub.choices['train'].add_argument('--stop-after-epochs',type=int)
    sub.choices['verify'].add_argument('--suite',choices=['core','data','models','ros2','simulation','release','all'],default='all')
    sub.choices['evaluate'].add_argument('--dataset',choices=['nyu','jaad','mot17','hypersim'])
    sub.choices['evaluate'].add_argument('--selection-lock')
    sub.choices['evaluate'].add_argument('--baseline',choices=['model','train-majority'],default='model')
    q=sub.add_parser('simulate');q.add_argument('args',nargs=argparse.REMAINDER)
    q=sub.add_parser('ros2');q.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if getattr(a,'protocol',None)=='quality-v2':
        if a.command=='bootstrap':return subprocess.call([sys.executable,str(ROOT/'tools/bootstrap_v2.py')])
        if a.command=='fetch':
            datasets=['nyu','jaad','mot17','openloris'] if not explicit_option('--datasets') else a.datasets
            if any(dataset not in {'nyu','jaad','mot17','openloris'} for dataset in datasets):raise ValueError('quality-v2 fetch supports nyu, jaad, mot17 and openloris')
            for dataset in datasets:
                code=subprocess.call([sys.executable,str(ROOT/'tools/data_v2.py'),dataset])
                if code:return code
            return 0
        if a.command=='train':
            if a.task=='forecast':
                from indoor_perception_lab.forecast_v2 import train_forecast_v2
                if a.device!='cpu' or a.overfit or a.stop_after_epochs is not None:raise ValueError('quality-v2 forecast training supports CPU primary runs only; overfit/epoch interruption flags are unsupported')
                if a.index is not None or a.unfreeze_last:raise ValueError('Forecast uses the frozen domain dataset and has no encoder to unfreeze')
                if a.configuration and not a.configuration.startswith('F'):raise ValueError('Forecast requires F1/F2/F3 configuration')
                lr={'F1':.001,'F2':.0003,'F3':.0001}.get(a.configuration,a.lr)
                if a.configuration and explicit_option('--lr') and a.lr!=lr:raise ValueError('Explicit learning rate disagrees with the frozen forecasting configuration')
                pack=train_forecast_v2(a.domain,a.seed,lr,epochs=30 if a.epochs is None else a.epochs,max_steps=6000 if a.max_steps is None else a.max_steps,resume=ROOT/a.checkpoint if a.checkpoint else None)
                print(json.dumps({'model_pack':str(pack)}));return 0
            from indoor_perception_lab.quality_training import train_quality,PRESETS
            if explicit_option('--domain'):raise ValueError('Perception training uses the frozen NYU/JAAD dataset selected by --task; --domain belongs to forecasting')
            if explicit_option('--lr') or a.unfreeze_last:raise ValueError('quality-v2 perception learning rates and trainable encoder blocks are fixed by --configuration')
            configuration=a.configuration or ('S1' if a.task=='semantic' else 'D1')
            if configuration not in PRESETS or PRESETS[configuration]['task']!=a.task:raise ValueError('Configuration does not match task')
            pack,run=train_quality(configuration,a.seed,a.device,30 if a.epochs is None else a.epochs,6000 if a.max_steps is None else a.max_steps,a.checkpoint,a.overfit,ROOT/a.index if a.index else None,a.stop_after_epochs)
            print(json.dumps({'model_pack':str(pack),'run':str(run)}));return 0
        if a.command in {'select','evaluate'}:
            if a.task=='forecast':
                from indoor_perception_lab.forecast_v2 import select_forecast_v2,evaluate_forecast_v2
                if a.command=='evaluate' and (a.device!='cpu' or a.baseline!='model' or a.dataset is not None or a.index is not None):raise ValueError('Forecast evaluation supports CPU/model/domain options; perception index/dataset/majority options are incompatible')
                result=select_forecast_v2(a.domain) if a.command=='select' else evaluate_forecast_v2(ROOT/a.model_pack,a.domain,a.split,ROOT/a.selection_lock if a.selection_lock else None)
                print(json.dumps(result));return 0
            from indoor_perception_lab.quality_release import select_quality,evaluate_quality
            if a.command=='select' and explicit_option('--domain'):raise ValueError('Perception selection uses the fixed task dataset; --domain belongs to forecasting')
            if a.command=='evaluate' and (a.index is not None or a.selection_lock is not None):raise ValueError('Perception evaluation enforces the frozen dataset index and canonical task selection lock')
            if a.command=='evaluate' and explicit_option('--domain'):raise ValueError('Perception evaluation uses --dataset; --domain belongs to forecasting')
            result=select_quality(a.task) if a.command=='select' else evaluate_quality(a.task,ROOT/a.model_pack,a.dataset or ('nyu' if a.task=='semantic' else 'jaad'),a.split,a.device,a.baseline)
            print(json.dumps(result));return 0
        if a.command=='verify':
            if a.quick:raise ValueError('quality-v2 verification uses --suite; --quick belongs to v1')
            return subprocess.call([sys.executable,str(ROOT/'tools/verify_v2.py'),'--suite',a.suite])
        if a.command=='demo':
            if any(explicit_option(flag) for flag in ['--fixture','--dataset','--sequence','--frames','--stride','--detector','--segmenter','--forecaster','--score-threshold','--device']):raise ValueError('quality-v2 demo consumes verified --chapters and --output; inference flags belong to chapter producers or v1 demo')
            args=['--output',a.output or 'demo/v2']
            if a.chapters:args+=['--chapters',*a.chapters]
            return subprocess.call([sys.executable,str(ROOT/'tools/demo_v2.py'),*args])
        if a.command=='benchmark':
            # The fresh Bench wheel environment proves that Lab is unnecessary.
            # Bootstrap-only users retain the installed development runtime path.
            bench_python=ROOT/'environments/v2-bench/bin/python'
            runtime=str(bench_python) if bench_python.is_file() else sys.executable
            environment=os.environ.copy();environment.pop('PYTHONPATH',None)
            return subprocess.call([runtime,str(ROOT/'tools/benchmark_v2.py'),*a.args],env=environment)
    if a.command=="train":
        if a.epochs is None:a.epochs=15
        if a.max_steps is None:a.max_steps=600
    if a.command=="bootstrap":return subprocess.call([sys.executable,str(ROOT/"tools/bootstrap.py")])
    if a.command=="fetch":
        rc=subprocess.call([sys.executable,str(ROOT/"tools/data_fetch.py"),"--datasets",*a.datasets]);
        if rc:return rc
        rc=subprocess.call([sys.executable,str(ROOT/"tools/data_validate.py"),"--datasets",*a.datasets])
        return rc or subprocess.call([sys.executable,str(ROOT/'tools/fetch_weights.py')])
    if a.command=="train":
        if a.task=="forecast":
            from indoor_perception_lab.motion_training import train_motion
            result=train_motion(a.domain,seed=a.seed,lr=a.lr,max_steps=a.max_steps,epochs=a.epochs,resume=ROOT/a.checkpoint if a.checkpoint else None)
        else:
            from indoor_perception_lab.training import train
            result=train(a.task,seed=a.seed,lr=a.lr,epochs=a.epochs,max_steps=a.max_steps,device=a.device,resume=a.checkpoint,index_path=ROOT/a.index if a.index else None,unfreeze_last=a.unfreeze_last)
        print(json.dumps({"model_pack":result}));return 0
    if a.command=="select":
        from indoor_perception_lab.release import select_model
        print(json.dumps(select_model(a.task)));return 0
    if a.command=="demo":
        if a.frames<1 or a.stride<1:raise ValueError('Positive frame count and stride required')
        if a.detector!='fixture' or a.forecaster or a.segmenter:
            import torch;torch.set_num_threads(4)
        from robot_model_sdk import fixture_frames
        from indoor_perception_lab.datasets import read_index,sensor_from_row
        from indoor_perception_lab.pipeline import replay
        if a.fixture:frames=fixture_frames(a.frames);detector="fixture"
        else:
            rows=read_index(ROOT,a.dataset)
            if a.sequence:rows=[r for r in rows if r["sequence_id"]==a.sequence]
            else:rows=[r for r in rows if r["sequence_id"]==rows[0]["sequence_id"]]
            rows=rows[::a.stride][:a.frames]
            if a.dataset=='jaad':
                from indoor_perception_lab.datasets import jaad_video_frames
                import itertools
                frames=itertools.islice(jaad_video_frames(ROOT,rows[0]['sequence_id']),0,a.frames,a.stride)
            else:frames=(sensor_from_row(ROOT,r) for r in rows)
            detector=ROOT/a.detector
        print(json.dumps(replay(frames,ROOT/(a.output or 'demo/fixture'),detector,ROOT/a.segmenter if a.segmenter else None,a.device,forecaster=ROOT/a.forecaster if a.forecaster else None,score_threshold=a.score_threshold)));return 0
    if a.command=="evaluate":
        import torch;torch.set_num_threads(4)
        if a.task=="forecast":
            from indoor_perception_lab.motion_training import evaluate_motion
            result=evaluate_motion(ROOT/a.model_pack,a.domain,a.split)
        else:
            from indoor_perception_lab.release import evaluate_release
            result=evaluate_release(a.task,ROOT/a.model_pack,a.split,ROOT/a.index if a.index else None,a.device)
        print(json.dumps(result));return 0
    if a.command=="benchmark":return subprocess.call([sys.executable,"-m","robot_reliability_bench.cli",*a.args])
    if a.command=="verify":return subprocess.call([sys.executable,str(ROOT/"tools/verify.py"),*( ["--quick"] if a.quick else [])])
    if a.command=="resume":
        checkpoint=(ROOT/a.checkpoint).resolve()
        if not checkpoint.is_relative_to(ROOT/'reports'):raise ValueError('Resume only accepts this project generated report checkpoints')
        record=json.loads((checkpoint.parent/'run.json').read_text())
        if record['config'].get('architecture')=='residual_cv_mlp_2x64':
            from indoor_perception_lab.forecast_v2 import train_forecast_v2
            c=record['config'];pack=train_forecast_v2(c['domain'],c['seed'],c['lr'],c['max_steps'],c['epochs'],resume=checkpoint,preset=c['preset']);print(json.dumps({'model_pack':str(pack)}));return 0
        if record['config'].get('protocol_id')=='quality-v2' and record['task'].startswith('quality-v2-'):
            from indoor_perception_lab.quality_training import train_quality
            c=record['config'];pack,run=train_quality(c['configuration'],c['seed'],c['device'],c['epochs'],c['max_steps'],checkpoint,c['overfit_diagnostic'],checkpoint.parent/'split-index.jsonl');print(json.dumps({'model_pack':str(pack),'run':str(run)}));return 0
        if record['task']=='motion-train':
            from indoor_perception_lab.motion_training import train_model
            c=record['config'];pack,_,_=train_model(ROOT,ROOT/record['dataset_path'],c['lr'],c['seed'],epochs=c['epochs'],max_steps=c['max_steps'],resume=checkpoint,patience=c['patience']);print(pack);return 0
        import torch
        from indoor_perception_lab.training import train
        ck=torch.load(ROOT/a.checkpoint,map_location="cpu",weights_only=True);c=ck["config"];print(train(c["task"],c["seed"],c["lr"],c["epochs"],c["max_steps"],c["device"],ROOT/a.checkpoint,(ROOT/a.checkpoint).parent/"split-index.jsonl",c.get("unfreeze_last",False)));return 0
if __name__=="__main__":raise SystemExit(main())
