"""Run requested verification suites and retain real exits, hashes and failures."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,os,shutil,subprocess,sys,tarfile,time,uuid
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--suite',choices=['core','data','models','ros2','simulation','release','all'],default='all');a=p.parse_args()
out=ROOT/'reports/v2/verification'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+a.suite+'-'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
report={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'suite':a.suite,'checks':{},'scope':'software verification; quality targets are assessed separately'}
env=os.environ.copy();env.update(PORTFOLIO_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONPATH=os.pathsep.join(str(ROOT/'packages'/n/'src') for n in ['robot_model_sdk','indoor_perception_lab','robot_reliability_bench']))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def execute(name,argv):
    log=out/(name+'.log');start=time.perf_counter()
    with log.open('w') as f:code=subprocess.call([str(x) for x in argv],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    report['checks'][name]={'verified':code==0,'exit_code':code,'argv':[str(x) for x in argv],'seconds':time.perf_counter()-start,'log':str(log.relative_to(ROOT)),'log_sha256':sha(log)}
def pytest(name,paths,python=sys.executable):
    execute(name,[python,'-m','pytest',*paths,'-q','--basetemp',out/(name+'-tmp')])
try:
    source_paths=[]
    for dirname in ('packages','tools','tests','configs'):
        for path in (ROOT/dirname).rglob('*'):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.py','.json','.toml','.lock'} and not any(part in {'build','__pycache__'} or part.endswith('.egg-info') for part in path.relative_to(ROOT/dirname).parts):source_paths.append(path)
    source_paths.extend([ROOT/'portfolio',ROOT/'requirements.lock'])
    source_bytes={str(path.relative_to(ROOT)):path.read_bytes() for path in sorted(set(source_paths))}
    source_before={name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()}
    archive_path=out/'source-before-verification.tar.gz'
    with tarfile.open(archive_path,'w:gz') as archive:
        for name,data in source_bytes.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    report['source_before_execution']=source_before
    report['source_archive']={'path':str(archive_path.relative_to(ROOT)),'sha256':sha(archive_path),'captured_before_checks':True}
    report['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    suite=a.suite
    if suite in ['core','all']:
        pytest('main-regression',['tests','--ignore=tests/robotics_v2'] if suite=='all' else ['tests/core','tests/bench']);execute('pip-check',[sys.executable,'-m','pip','check'])
    # The all-suite main regression already includes these paths once.
    if suite=='data':pytest('data',['tests/data_v2']);pytest('evaluation-oracles',['tests/quality_audit_v2'])
    if suite=='models':pytest('models',['tests/quality','tests/motion_v2'])
    if suite in ['ros2','simulation','all']:
        pytest('robot-contracts',['tests/robotics_v2'],ROOT/'environments/robotics-v2/bin/python')
    if suite in ['ros2','all']:
        execute('ros-runtime',[ROOT/'cache/v2/robotics/bin/pixi','run','--manifest-path',ROOT/'environments/ros-v2-jazzy/pixi.toml','python','-c','import rclpy; from sensor_msgs.msg import Image, CameraInfo; from nav_msgs.msg import Odometry; import rosbag2_py; rclpy.init(); n=rclpy.create_node("quality_v2_runtime_verify"); print(n.get_name()); n.destroy_node(); rclpy.shutdown()'])
    if suite in ['simulation','all']:execute('simulation-geometry',[sys.executable,ROOT/'tools/robotics_v2.py','mujoco-smoke','--output',str(out.relative_to(ROOT)/'mujoco-geometry')])
    if suite in ['release','all']:
        from indoor_perception_lab.quality_release import ensure_test_seal
        from robot_model_sdk.manifest import validate_manifest
        identities={}
        for task,dataset in [('semantic','nyu'),('detection','jaad')]:
            lock=json.loads((ROOT/f'models/v2/selection-{task}.json').read_text());pack=ROOT/lock['selected']['pack'];ensure_test_seal(task,pack,dataset);m=validate_manifest(pack);identities[task]=m['sha256']
        report['checks']['selected-model-identities']={'verified':True,'weights_sha256':identities}
        commands=ROOT/'reports/v2/verified-commands.jsonl'
        for line in commands.read_text().splitlines():
            c=json.loads(line)
            if c['verified'] and (c['exit_code']!=0 or sha(ROOT/c['log'])!=c['log_sha256']):raise ValueError('Command evidence changed: '+c['name'])
        report['checks']['command-evidence']={'verified':True,'records':len(commands.read_text().splitlines())}
        independent_path=ROOT/'reports/v2/independent-install/run.json'
        independent=json.loads(independent_path.read_text())
        if independent.get('status')!='passed' or independent.get('source_unchanged_during_execution') is not True or independent.get('selected_models_unchanged_during_execution') is not True:
            raise ValueError('Fresh noneditable package acceptance is absent or incomplete')
        if independent['source_before_execution']!=independent['source_after_execution'] or independent['selected_model_files_before_execution']!=independent['selected_models_after_execution']:
            raise ValueError('Independent-install before/after identity maps differ')
        if sha(ROOT/independent['source_archive']['path'])!=independent['source_archive']['sha256']:
            raise ValueError('Independent-install pre-execution archive changed')
        for name,value in independent['source_after_execution'].items():
            if sha(ROOT/name)!=value:raise ValueError('Independent-install source changed: '+name)
        for name,value in independent['selected_models_after_execution'].items():
            if sha(ROOT/name)!=value:raise ValueError('Independent-install selected model changed: '+name)
        for check in independent['checks']:
            if check.get('exit_code')!=0 or check.get('verified') is not True or sha(ROOT/check['log'])!=check['log_sha256']:raise ValueError('Independent-install command proof changed')
        for binding in independent['wheels'].values():
            if sha(ROOT/binding['wheel'])!=binding['sha256']:raise ValueError('Independent wheel changed')
        report['checks']['independent-packages']={'verified':True,'path':str(independent_path.relative_to(ROOT)),'sha256':sha(independent_path),'roles':['sdk','lab-without-bench','bench-without-lab']}
    ledger=ROOT/'reports/v2/data/download_ledger.jsonl'
    lines=[json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
    total=lines[-1]['cumulative_bytes'] if lines else 0
    size=sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file() and not x.is_symlink());free=shutil.disk_usage(ROOT).free
    report['checks']['budget']={'verified':size<40_000_000_000 and total<40_000_000_000 and free>=15_000_000_000,'project_bytes':size,'recorded_cumulative_download_bytes':total,'free_bytes':free}
    preserved=json.loads((ROOT/'reports/v2/v1-preservation.json').read_text())
    changed=[x['path'] for x in preserved['files'] if sha(ROOT/x['path'])!=x['sha256']]
    report['checks']['v1-artifacts-preserved']={'verified':not changed,'files':len(preserved['files']),'changed':changed,'baseline_commit':preserved['baseline_commit']}
    source_after={name:sha(ROOT/name) for name in source_before}
    report['source_after_execution']=source_after
    report['source_unchanged_during_execution']=source_before==source_after
    report['checks']['current-source-binding']={'verified':source_before==source_after,'files':len(source_before),'archive_sha256':sha(archive_path)}
except Exception as exc:report['error']=repr(exc)
report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();report['status']='passed' if 'error' not in report and all(c['verified'] for c in report['checks'].values()) else 'failed'
(out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'suite':a.suite,'report':str((out/'run.json').relative_to(ROOT))}));raise SystemExit(0 if report['status']=='passed' else 1)
