"""Reconstruct isolated pinned runtimes while preserving all v1 installation logs."""
from pathlib import Path
import datetime,json,os,subprocess,sys,hashlib,uuid
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/v2/bootstrap'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]);REPORT.mkdir(parents=True,exist_ok=False)
env=os.environ.copy()
env.pop('PYTHONPATH',None)  # Inspect installed distributions, not development source metadata.
env.update(PORTFOLIO_ROOT=str(ROOT),PIP_CACHE_DIR=str(ROOT/'cache/pip'),TMPDIR=str(ROOT/'cache/tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),TORCH_HOME=str(ROOT/'cache/torch'),PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
for name in ['pip','tmp','wheels']:(ROOT/'cache'/name).mkdir(parents=True,exist_ok=True)
records=[]
def run(name,argv):
    log=REPORT/(name+'.log');start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log.open('w') as out:code=subprocess.call([str(x) for x in argv],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
    records.append({'name':name,'argv':[str(x) for x in argv],'exit_code':code,'verified':code==0,'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'log':str(log.relative_to(ROOT)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
    (REPORT/'run.json').write_text(json.dumps({'checks':records,'global_changes':False},indent=2)+'\n')
    if code:raise SystemExit(code)
python=ROOT/'environments/main/bin/python'
if not python.exists():run('main-venv',[sys.executable,'-m','venv',python.parent.parent])
# The accepted wheel cache is complete; fail closed instead of an unmetered new download.
run('main-pinned-dependencies',[python,'-m','pip','install','--no-index','--find-links',ROOT/'cache/wheels','--require-hashes','-r',ROOT/'requirements.lock'])
for name in ['robot_model_sdk','indoor_perception_lab','robot_reliability_bench']:
    run('package-'+name,[python,'-m','pip','install','--no-deps','--no-build-isolation','-e',ROOT/'packages'/name])
run('main-pip-check',[python,'-m','pip','check'])
robot_python=ROOT/'environments/robotics-v2/bin/python'
robot_lock=ROOT/'reports/v2/robotics/robotics-requirements.lock.txt'
if not robot_python.exists():run('robot-venv',[python,'-m','venv',robot_python.parent.parent])
if not robot_lock.exists():raise SystemExit('Robotics hash lock is missing; acquisition/bootstrap-python must first produce its evidence')
run('robot-pinned-dependencies',[robot_python,'-m','pip','install','--no-index','--find-links',ROOT/'cache/v2/robotics/wheels','--require-hashes','-r',robot_lock])
run('robot-pip-check',[robot_python,'-m','pip','check'])
pixi=ROOT/'cache/v2/robotics/bin/pixi';ros_manifest=ROOT/'environments/ros-v2-jazzy/pixi.toml'
if not pixi.exists() or not ros_manifest.with_name('pixi.lock').exists():raise SystemExit('Project-local locked ROS runtime is not present')
env['PIXI_CACHE_DIR']=str(ROOT/'cache/v2/robotics/pixi-cache')
env['RATTLER_CACHE_DIR']=env['PIXI_CACHE_DIR']
run('ros-locked-install',[pixi,'install','--locked','--manifest-path',ros_manifest])
print(json.dumps({'status':'verified','scope':'isolated development runtimes; clean noneditable package audit is separate','report':str((REPORT/'run.json').relative_to(ROOT))}))
