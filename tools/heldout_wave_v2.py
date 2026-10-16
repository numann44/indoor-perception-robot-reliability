"""Serialize frozen perception holdouts after both complete training waves."""
from __future__ import annotations
import datetime
import json
import sys
import time
from run_quality_v2 import ROOT,execute
from indoor_perception_lab.experiments import json_write,sha


def main():
    deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'configs/v2/protocol.json').read_text())['deadline_utc'].replace('Z','+00:00'))
    bindings={}
    for task in ('detection','semantic'):
        path=ROOT/f'reports/v2/{task}-training-complete.json'
        print(json.dumps({'waiting_for':str(path.relative_to(ROOT)),'test_targets_read':False}),flush=True)
        while not path.exists():
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Approved implementation deadline reached before holdout admission')
            time.sleep(10)
        result=json.loads(path.read_text())
        if result['seeds']!=[42,43,44] or result['selection']['test_used_for_selection'] or result['selection']['task']!=task:raise ValueError('Incomplete or invalid family completion evidence')
        lock=ROOT/f'models/v2/selection-{task}.json'
        if result['selection']!=json.loads(lock.read_text()):raise ValueError('Training completion differs from frozen selection lock')
        bindings[task]={'completion':str(path.relative_to(ROOT)),'completion_sha256':sha(path),'selection_lock_sha256':sha(lock)}
    # Each task driver separately checks all exact trained repeat identities before
    # admitting fresh target files. No training/timing worker is started here.
    for task in ('detection','semantic'):
        execute('heldout-wave-'+task,[sys.executable,ROOT/'tools/evaluate_quality_v2.py','--task',task,'--device','mps'])
    execute('quality-final-measured-report',[sys.executable,ROOT/'tools/report_quality_v2.py','--final'])
    json_write(ROOT/'reports/v2/perception-heldout-complete.json',{'status':'verified','primary_seed':42,'seed_selection_on_test':False,'family_bindings':bindings,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'measurement_completion':{task:sha(ROOT/f'reports/v2/{task}-heldout-complete.json') for task in ('detection','semantic')},'quality_summary_sha256':sha(ROOT/'reports/v2/quality-summary.json'),'quality_passes_are_separate_from_command_exits':True})


if __name__=='__main__':main()
