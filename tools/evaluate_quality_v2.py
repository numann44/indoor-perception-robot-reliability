"""Run the predefined, sealed held-out comparisons; never select on test scores."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from indoor_perception_lab.experiments import root_path,sha,json_write
from indoor_perception_lab.quality_release import ensure_test_seal
from run_quality_v2 import execute
ROOT=root_path()


def evaluation(task,dataset,pack,device,baseline='model'):
    pack=Path(pack);ensure_test_seal(task,pack,dataset)
    suffix='-train-majority' if baseline=='train-majority' else ''
    output=ROOT/f'reports/v2/{task}-{dataset}-test-{pack.name}{suffix}.json'
    index=ROOT/(f'data/v2/{dataset}/index.jsonl' if dataset!='hypersim' else 'data/hypersim/index.jsonl')
    if output.exists():
        report=json.loads(output.read_text());run=json.loads((ROOT/'reports/v2/runs'/report['run_id']/'run.json').read_text())
        if report.get('checkpoint_sha256')!=sha(pack/'weights.pt') or report.get('split_index_sha256')!=sha(index) or run['status']!='completed' or report.get('baseline','model')!=baseline:raise ValueError('Existing held-out evidence identity mismatch')
        return str(output.relative_to(ROOT))
    name=f'evaluate-{dataset}-{pack.name}{suffix}'
    execute(name,[ROOT/'portfolio','evaluate','--protocol','quality-v2','--task',task,'--dataset',dataset,'--model-pack',str(pack.relative_to(ROOT)),'--split','test','--device',device,'--baseline',baseline])
    return str(output.relative_to(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['semantic','detection'],required=True);p.add_argument('--device',default='mps');a=p.parse_args()
    lock=json.loads((ROOT/f'models/v2/selection-{a.task}.json').read_text());selected=lock['selected'];configuration=selected['configuration'];reports=[]
    # Repeat checkpoints must exist before opening primary fresh targets.
    for seed in (42,43,44):ensure_test_seal(a.task,ROOT/f'models/v2/{a.task}-{configuration}-seed{seed}','nyu' if a.task=='semantic' else 'jaad')
    if a.task=='semantic':
        for seed in (42,43,44):reports.append(evaluation(a.task,'nyu',ROOT/f'models/v2/semantic-{configuration}-seed{seed}',a.device))
        for pack in ['models/segmentation-trained','models/segmentation-initial',lock['additional_baselines'][0]['pack']]:reports.append(evaluation(a.task,'nyu',ROOT/pack,a.device))
        reports.append(evaluation(a.task,'nyu',ROOT/selected['pack'],'cpu','train-majority'))
        for pack in [selected['pack'],'models/segmentation-trained']:reports.append(evaluation(a.task,'hypersim',ROOT/pack,a.device))
    else:
        for seed in (42,43,44):reports.append(evaluation(a.task,'jaad',ROOT/f'models/v2/detection-{configuration}-seed{seed}',a.device))
        for pack in ['models/detection-baseline','models/detection-trained',lock['additional_baselines'][0]['pack']]:reports.append(evaluation(a.task,'jaad',ROOT/pack,a.device))
        for seed in (42,43,44):reports.append(evaluation(a.task,'mot17',ROOT/f'models/v2/detection-{configuration}-seed{seed}',a.device))
        reports.append(evaluation(a.task,'mot17',ROOT/'models/detection-baseline',a.device))
        existing=ROOT/'reports/v2/tracking/mot17-heldout.json'
        if existing.exists():
            report=json.loads(existing.read_text());run=json.loads((ROOT/'reports/v2/runs'/report['run_id']/'run.json').read_text())
            if report['weights_sha256']!=selected['weights_sha256'] or report['index_sha256']!=sha(ROOT/'data/v2/mot17/index.jsonl') or report['selected_configuration']!=json.loads((ROOT/'models/v2/selection-tracking.json').read_text())['selected']['configuration'] or run['status']!='completed':raise ValueError('Existing tracker measurement identity mismatch')
        else:execute('evaluate-mot17-trackers',[sys.executable,ROOT/'tools/tracking_v2.py','mot','--pack',selected['pack'],'--device',a.device])
        reports.append('reports/v2/tracking/mot17-heldout.json')
    json_write(ROOT/f'reports/v2/{a.task}-heldout-complete.json',{'protocol_id':'quality-v2','primary_seed':42,'seed_selection_on_test':False,'selection_lock_sha256':sha(ROOT/f'models/v2/selection-{a.task}.json'),'reports':[{'path':path,'sha256':sha(ROOT/path)} for path in reports]})
    print(json.dumps({'task':a.task,'comparisons':len(reports),'status':'completed'}))


if __name__=='__main__':main()
