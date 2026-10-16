"""Serial predefined training/validation selection; never opens fresh test targets."""
from __future__ import annotations
import argparse,datetime,json,os,shlex,subprocess,sys,time,uuid
from pathlib import Path
from indoor_perception_lab.experiments import root_path,sha,json_write
from indoor_perception_lab.quality_training import PRESETS
from indoor_perception_lab.quality_release import _trained_manifest,select_quality
ROOT=root_path()

def execute(name,args):
    invocation=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    log=ROOT/f'reports/v2/commands/{name}-{invocation}.log';log.parent.mkdir(parents=True,exist_ok=True)
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.perf_counter()
    with log.open('w') as f:rc=subprocess.call(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    record={'name':name,'argv':[str(x) for x in args],'command':shlex.join([str(x) for x in args]),'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.perf_counter()-clock,'exit_code':rc,'verified':rc==0,'log':str(log.relative_to(ROOT)),'log_sha256':sha(log)}
    with (ROOT/'reports/v2/verified-commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    if rc:raise RuntimeError(name+' failed; see '+str(log))

def train_one(configuration,seed,device):
    task=PRESETS[configuration]['task'];idx=ROOT/f'data/v2/{"nyu" if task=="semantic" else "jaad"}/index.jsonl';pack=ROOT/f'models/v2/{task}-{configuration}-seed{seed}'
    valid=False
    if (pack/'manifest.json').exists():
        try:_trained_manifest(pack,task,configuration,seed,sha(idx));valid=True
        except (ValueError,KeyError):pass
    if not valid:execute(f'train-{configuration}-seed{seed}',[ROOT/'portfolio','train','--protocol','quality-v2','--task',task,'--configuration',configuration,'--seed',str(seed),'--device',device,'--epochs','30','--max-steps','6000'])
    m=json.loads((pack/'manifest.json').read_text())
    if 'trained_backend_evidence' not in m:execute(f'accept-{configuration}-seed{seed}',[sys.executable,ROOT/'tools/accept_model_v2.py','--pack',str(pack.relative_to(ROOT))])
    return pack

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['semantic','detection'],required=True);p.add_argument('--device',default='auto');p.add_argument('--wait-run');p.add_argument('--wait-complete');a=p.parse_args()
    if a.wait_complete:
        marker=(ROOT/a.wait_complete).resolve()
        if marker not in {ROOT/'reports/v2/semantic-training-complete.json',ROOT/'reports/v2/detection-training-complete.json'}:raise ValueError('Only predefined family completion evidence can be awaited')
        deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'configs/v2/protocol.json').read_text())['deadline_utc'].replace('Z','+00:00'))
        while not marker.exists():
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Approved implementation deadline reached')
            time.sleep(10)
        completion=json.loads(marker.read_text())
        if completion['seeds']!=[42,43,44] or completion['selection']['test_used_for_selection']:raise ValueError('Invalid awaited family completion evidence')
    if a.wait_run:
        pending=(ROOT/a.wait_run).resolve()
        if not pending.is_relative_to(ROOT/'reports/v2/runs'):raise ValueError('Only a generated v2 run can be awaited')
        deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'configs/v2/protocol.json').read_text())['deadline_utc'].replace('Z','+00:00'))
        while True:
            state=json.loads((pending/'run.json').read_text())
            if state['status']!='running':break
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Approved implementation deadline reached')
            time.sleep(10)
        if state['status']!='completed':raise RuntimeError('Awaited training failed; inspect saved run')
    family=['S1','S2','S3'] if a.task=='semantic' else ['D1','D2','D3']
    for c in family:train_one(c,42,a.device)
    selected=select_quality(a.task)
    for seed in [43,44]:train_one(selected['selected']['configuration'],seed,a.device)
    json_write(ROOT/f'reports/v2/{a.task}-training-complete.json',{'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':selected,'seeds':[42,43,44],'fresh_test_evaluated':False})
    print(json.dumps({'task':a.task,'selected':selected['selected'],'repeats_complete':True}),flush=True)

if __name__=='__main__':main()
