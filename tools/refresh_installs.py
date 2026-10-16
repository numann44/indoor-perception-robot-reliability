"""Refresh built wheels after source freeze and verify noneditable installed bytes."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];env=os.environ.copy();env.update(PIP_CACHE_DIR=str(root/'cache/pip'),TMPDIR=str(root/'cache/tmp'),XDG_CACHE_HOME=str(root/'cache'),MPLCONFIGDIR=str(root/'cache/matplotlib'),TORCH_HOME=str(root/'cache/torch'),PYTHONDONTWRITEBYTECODE='1')
wh=root/'artifacts/wheels';names=['robot_model_sdk','indoor_perception_lab','robot_reliability_bench']
subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wh),*[str(root/'packages'/n) for n in names]],check=True,env=env)
report={'wheel_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in wh.glob('*.whl')},'environments':{}}
for name in ['lab','bench']:
    py=root/'environments'/name/'bin/python';packages=['robot_model_sdk','indoor_perception_lab' if name=='lab' else 'robot_reliability_bench'];matches=0
    for package in packages:
        wheel=next(wh.glob(package+'-*.whl'));subprocess.run([str(py),'-m','pip','install','--no-deps','--force-reinstall',str(wheel)],check=True,env=env)
        location=Path(subprocess.check_output([str(py),'-c',f'import {package};print({package}.__path__[0])'],text=True,env=env).strip())
        for source in (root/'packages'/package/'src'/package).rglob('*'):
            if not source.is_file() or source.suffix not in {'.py','.json'}:continue
            target=location/source.relative_to(root/'packages'/package/'src'/package)
            if not target.exists() or target.read_bytes()!=source.read_bytes():raise AssertionError(f'Installed source differs: {target}')
            matches+=1
    subprocess.run([str(py),'-m','pip','check'],check=True,env=env)
    if name=='bench':subprocess.run([str(py),'-c',"import importlib.util;assert importlib.util.find_spec('indoor_perception_lab') is None"],check=True,env=env)
    report['environments'][name]={'installed_source_files_equal':matches,'editable':False,'pip_check':True,'lab_dependency_absent':name=='bench'}
(root/'reports/independent-source-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
