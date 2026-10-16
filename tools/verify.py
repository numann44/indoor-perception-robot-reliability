"""Acceptance audit uses real subprocess exits and saved measurement identities."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');args=parser.parse_args()
env=os.environ.copy();env.update(PORTFOLIO_ROOT=str(root),PIP_CACHE_DIR=str(root/'cache/pip'),XDG_CACHE_HOME=str(root/'cache'),TORCH_HOME=str(root/'cache/torch'),TMPDIR=str(root/'cache/tmp'),MPLCONFIGDIR=str(root/'cache/matplotlib'),PYTHONDONTWRITEBYTECODE='1')
report={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'running','checks':{}}
def execute(name,argv):
    path=root/'reports'/f'verify-{name}.log';start=time.perf_counter()
    with path.open('w') as log:rc=subprocess.call(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    report['checks'][name]={'exit_code':rc,'verified':rc==0,'seconds':time.perf_counter()-start,'argv':argv,'log':str(path.relative_to(root)),'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if rc:raise RuntimeError(f'{name} failed; see {path}')
try:
    execute('tests',[sys.executable,'-m','pytest','tests','-q','--basetemp='+str(root/'cache/pytest-acceptance')])
    execute('pip-check',[sys.executable,'-m','pip','check'])
    if not args.quick:execute('data',[sys.executable,str(root/'tools/data_validate.py'),'--datasets','jaad','hypersim','tum'])
    from robot_model_sdk.manifest import validate_manifest
    identities={}
    for name,backend in [('segmentation-trained','native'),('detection-trained','native'),('segmentation-onnx','onnx'),('segmentation-int8','onnx')]:
        m=validate_manifest(root/'models'/name,backend);identities[name]=m['onnx_sha256'] if backend=='onnx' else m['sha256']
    report['checks']['model-identities']={'verified':True,'sha256':identities}
    if not args.quick:
        import numpy as np,torch
        from robot_model_sdk import load_adapter
        from indoor_perception_lab.datasets import read_index,sensor_from_row
        torch.set_num_threads(4);native=load_adapter(root/'models/segmentation-trained');onnx=load_adapter(root/'models/segmentation-onnx','onnx');rows=read_index(root,'hypersim',split='val');exact=0
        for row in rows:
            frame=sensor_from_row(root,row);a=native.predict(frame);b=onnx.predict(frame);np.testing.assert_allclose(a.logits,b.logits,rtol=1e-4,atol=1e-5);exact+=int(np.array_equal(a.semantic,b.semantic))
        native.close();onnx.close();report['checks']['export-parity']={'verified':True,'validation_images':len(rows),'exact_semantic_masks':exact,'rtol':1e-4,'atol':1e-5}
    ledger=root/'reports/verified-commands.jsonl';commands=[json.loads(line) for line in ledger.read_text().splitlines()];bad=[];verified=0
    for c in commands:
        log=root/c['log']
        if c['verified'] and (not log.exists() or hashlib.sha256(log.read_bytes()).hexdigest()!=c['log_sha256']):bad.append(c['name'])
        verified+=int(c['verified'])
    if bad:raise ValueError('Command evidence hash changed: '+','.join(bad))
    report['checks']['command-evidence']={'verified':True,'successful_commands':verified,'failed_commands_retained':sum(not c['verified'] for c in commands)}
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink());free=shutil.disk_usage(root).free
    downloads=[json.loads(s) for s in (root/'reports/data/download_ledger.jsonl').read_text().splitlines()];download_bytes=downloads[-1]['cumulative_bytes'] if downloads else 0
    if size>=8_000_000_000 or download_bytes+1_000_000_000>=8_000_000_000 or free<15_000_000_000:raise RuntimeError('Resource budget failed')
    report['checks']['budget']={'verified':True,'project_bytes':size,'free_bytes':free,'recorded_download_bytes':download_bytes,'unmetered_initial_dependency_reserve_bytes':1_000_000_000,'limit_bytes':8_000_000_000}
    report['status']='passed'
except Exception as exc:
    report['status']='failed';report['error']=repr(exc)
finally:
    report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(root/'reports/acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'report':'reports/acceptance.json'}))
raise SystemExit(0 if report['status']=='passed' else 1)
