from pathlib import Path
import os,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1]
if sys.version_info<(3,12):raise SystemExit('Python3.12+ required; set PORTFOLIO_PYTHON to an existing supported interpreter')
for p in ['pip','tmp','torch','matplotlib','wheels']:(ROOT/'cache'/p).mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(PIP_CACHE_DIR=str(ROOT/'cache/pip'),TMPDIR=str(ROOT/'cache/tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),TORCH_HOME=str(ROOT/'cache/torch'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),PYTHONDONTWRITEBYTECODE='1')
python=ROOT/'environments/main/bin/python'
if not python.exists():subprocess.run([sys.executable,'-m','venv',str(ROOT/'environments/main')],check=True,env=env)
with (ROOT/'reports/bootstrap-reproduction.log').open('w') as log:
    args=[str(python),'-m','pip','install','--require-hashes','-r',str(ROOT/'requirements.lock'),'--find-links',str(ROOT/'cache/wheels')]
    if any((ROOT/'cache/wheels').glob('*.whl')):args.append('--no-index')
    subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    for name in ['robot_model_sdk','robot_reliability_bench','indoor_perception_lab']:subprocess.run([str(python),'-m','pip','install','--no-deps','--no-build-isolation','-e',str(ROOT/'packages'/name)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    subprocess.run([str(python),'-m','pip','check'],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
print(json.dumps({'runtime_ready':True,'platform_lock':'macOSARM64Python3.12','log':'reports/bootstrap-reproduction.log'}))
