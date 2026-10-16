"""Freeze artifact identities and validation choices before loading fresh targets."""
from pathlib import Path
import json,datetime
from robot_model_sdk import load_adapter
from robot_model_sdk.manifest import validate_manifest
from .experiments import root_path,sha,json_write
from .quality_evaluation import evaluate_semantic,evaluate_detection
from .quality_training import QualityRun,PRESETS


def _trained_manifest(pack,task,configuration,seed,index_hash):
    m=validate_manifest(pack,'native')
    expected_metric='validation_mIoU' if task=='semantic' else 'validation_AP'
    if m.get('protocol_id')!='quality-v2' or m.get('task')!=task:raise ValueError('Wrong trained protocol/task')
    if m.get('configuration')!=configuration or m.get('seed')!=seed or seed not in (42,43,44):raise ValueError('Selected configuration and allowed seed required')
    if m.get('overfit_diagnostic',False) or m.get('interrupted_for_resume_test',False) or m.get('selection_metric')!=expected_metric:raise ValueError('Diagnostics/initialization cannot be promoted as trained models')
    if m.get('steps',0)<1 or m.get('selected_checkpoint_step',0)<1:raise ValueError('Positive trained checkpoint step required')
    if m.get('dataset_split_hash')!=index_hash:raise ValueError('Trained model differs from frozen data index')
    if m['architecture']!=PRESETS[configuration]['architecture']:raise ValueError('Unexpected selected architecture')
    if pack.name!=f'{task}-{configuration}-seed{seed}':raise ValueError('Exact selected artifact identity required')
    return m


def _index_hash(root,task):return sha(root/f'data/v2/{"nyu" if task=="semantic" else "jaad"}/index.jsonl')


def select_quality(task):
    root=root_path();expected=['S1','S2','S3'] if task=='semantic' else ['D1','D2','D3'];candidates=[]
    lock_path=root/f'models/v2/selection-{task}.json'
    if lock_path.exists():
        record=json.loads(lock_path.read_text());ensure_test_seal(task,root/record['selected']['pack'],'nyu' if task=='semantic' else 'jaad');return record
    index_hash=_index_hash(root,task)
    for key in expected:
        pack=root/f'models/v2/{task}-{key}-seed42'
        if not (pack/'manifest.json').exists():raise ValueError('Complete predefined validation runs before selection: missing '+key)
        m=_trained_manifest(pack,task,key,42,index_hash)
        candidates.append({'configuration':key,'validation_score':m['selection_value'],'pack':str(pack.relative_to(root)),'weights_sha256':m['sha256'],'manifest_sha256':sha(pack/'manifest.json'),'run_id':m['run_id'],'dataset_split_hash':index_hash})
    winner=max(candidates,key=lambda c:(c['validation_score'],-expected.index(c['configuration'])))
    initial=root/(winner['pack']+'-initial')
    baseline=validate_manifest(initial,'native')
    if baseline.get('seed')!=42 or baseline.get('overfit_diagnostic',False):raise ValueError('Initial baseline must be primary non-diagnostic seed42')
    initial_record={'task':task,'pack':str(initial.relative_to(root)),'weights_sha256':baseline['sha256'],'manifest_sha256':sha(initial/'manifest.json'),'role':'untrained_head_pretrained_encoder' if task=='semantic' else 'pretrained_initialization'}
    record={'protocol_id':'quality-v2','task':task,'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_split':'val','test_used_for_selection':False,'dataset_split_hash':index_hash,'candidates':candidates,'selected':winner,'seed_repeats':[42,43,44],'additional_baselines':[initial_record]}
    json_write(lock_path,record);json_write(root/f'reports/v2/selection-{task}.json',record);return record


def ensure_test_seal(task,pack,dataset):
    root=root_path();pack=Path(pack).resolve();lock=root/f'models/v2/selection-{task}.json'
    if not lock.exists():raise ValueError('Fresh targets require frozen trained model selection')
    selection=json.loads(lock.read_text());index_hash=_index_hash(root,task)
    if selection.get('dataset_split_hash')!=index_hash:raise ValueError('Dataset changed after model selection')
    selected=(root/selection['selected']['pack']).resolve()
    if sha(selected/'weights.pt')!=selection['selected']['weights_sha256'] or sha(selected/'manifest.json')!=selection['selected']['manifest_sha256']:raise ValueError('Selected artifact changed after lock')
    if dataset=='mot17' and not (root/'models/v2/selection-tracking.json').exists():raise ValueError('MOT external holdout also requires tracker lock')
    m=json.loads((pack/'manifest.json').read_text())
    baseline_file=root/'reports/v2/baseline-allowlist.json'
    allowed=json.loads(baseline_file.read_text())['baselines'] if baseline_file.exists() else []
    allowed+=selection.get('additional_baselines',[])
    for b in allowed:
        if b['task']==task and (root/b['pack']).resolve()==pack:
            validate_manifest(pack,'native')
            if sha(pack/m['native_weights'])!=b['weights_sha256'] or sha(pack/'manifest.json')!=b['manifest_sha256']:raise ValueError('Baseline identity changed')
            return selection
    key=selection['selected']['configuration'];seed=m.get('seed')
    _trained_manifest(pack,task,key,seed,index_hash)
    if seed==42 and pack!=selected:raise ValueError('Primary model must be exact locked artifact')
    return selection


def evaluate_quality(task,pack,dataset,split='test',device='cpu',baseline='model'):
    from .data_v2 import read_index_v2
    if dataset not in ({'nyu','hypersim'} if task=='semantic' else {'jaad','mot17'} if task=='detection' else set()):raise ValueError('Unsupported task/dataset evaluation pair')
    root=root_path();pack=Path(pack);manifest=json.loads((pack/'manifest.json').read_text())
    if split in {'test','outside_holdout'} or dataset=='mot17':ensure_test_seal(task,pack,dataset)
    if dataset=='hypersim':
        from .datasets import read_index
        rows=read_index(root,'hypersim',split=split);idx=root/'data/hypersim/index.jsonl'
    else:
        rows=read_index_v2(dataset,split='outside_holdout' if dataset=='mot17' else split,root=root);idx=root/f'data/v2/{dataset}/index.jsonl'
    if not rows:raise ValueError('No evaluation rows')
    if baseline not in {'model','train-majority'} or baseline=='train-majority' and task!='semantic':raise ValueError('Training-majority baseline is semantic only')
    majority=None
    if baseline=='train-majority':
        majority=manifest.get('majority_class')
        if not isinstance(majority,int) or not 0<=majority<40:raise ValueError('Missing training-derived majority class')
        events=[json.loads(line) for line in (root/manifest['run_dir']/'events.jsonl').read_text().splitlines()]
        supports=[e for e in events if e.get('event')=='train_label_support']
        if len(supports)!=1 or supports[0]['majority_class']!=majority:raise ValueError('Majority class must match recorded training-only label support')
    config={'protocol_id':'quality-v2','task':task,'dataset':dataset,'split':split,'device':device,'baseline':baseline,'majority_class':majority,'model_pack':str(pack.relative_to(root)),'checkpoint_sha256':sha(pack/'weights.pt'),'seed':manifest.get('seed')}
    run=QualityRun('quality-v2-'+task+'-evaluate',config,idx);adapter=None
    try:
        if baseline=='model':adapter=load_adapter(pack,device=device)
        if task=='semantic':result=evaluate_semantic(root,rows,adapter,majority_class=majority,save_predictions=run.dir/'semantic-predictions' if split=='test' else None)
        else:result,preds=evaluate_detection(root,rows,adapter);json_write(run.dir/'predictions.json',preds)
        result.update(run_id=run.id,baseline=baseline,majority_class=majority,model_pack=config['model_pack'],checkpoint_sha256=config['checkpoint_sha256'],split_index_sha256=sha(idx));run.finish(result)
        suffix='-train-majority' if baseline=='train-majority' else ''
        json_write(root/f'reports/v2/{task}-{dataset}-{split}-{pack.name}{suffix}.json',result)
        return {k:v for k,v in result.items() if k in {'run_id','AP','AP50','AR100','mIoU','pixel_accuracy','images','scenes','model_pack'}}
    except Exception as exc:
        run.finish({'error':repr(exc)},status='failed');raise
    finally:
        if adapter is not None:adapter.close()
