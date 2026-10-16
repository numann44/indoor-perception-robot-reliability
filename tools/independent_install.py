"""Build two clean wheel-only project environments from the existing hash lock."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
env=os.environ.copy();env.update(PIP_CACHE_DIR=str(root/'cache/pip'),TMPDIR=str(root/'cache/tmp'),XDG_CACHE_HOME=str(root/'cache'),TORCH_HOME=str(root/'cache/torch'),MPLCONFIGDIR=str(root/'cache/matplotlib'),PYTHONDONTWRITEBYTECODE='1',PORTFOLIO_ROOT=str(root))
lines=(root/'requirements.lock').read_text().splitlines()
exclude_common={'onnx','onnx-ir','onnxscript','ml_dtypes','safetensors','PyYAML'}
locks={}
for name,exclude in [('lab',exclude_common|{'onnxruntime','flatbuffers','protobuf'}),('bench',exclude_common|{'h5py'})]:
    lock=root/'configs'/f'{name}-requirements.lock';lock.write_text('\n'.join(l for l in lines if l.split('==')[0] not in exclude)+'\n');locks[name]=lock
wheelhouse=root/'artifacts/wheels';wheelhouse.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheelhouse),*[str(root/'packages'/n) for n in ['robot_model_sdk','indoor_perception_lab','robot_reliability_bench']]],check=True,env=env)
wheel_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in wheelhouse.glob('*.whl')}
record={'wheel_sha256':wheel_hashes,'installation':'freshvenv noneditable builtwheel; offline hash-locked dependencies','environments':{}}
for name in ['lab','bench']:
    destination=root/'environments'/name
    if destination.exists():raise RuntimeError(f'Refusing to overwrite existing environment {destination}')
    if shutil.disk_usage(root).free<15_000_000_000:raise RuntimeError('Disk reserve below15GB')
    subprocess.run([sys.executable,'-m','venv',str(destination)],check=True,env=env)
    py=destination/'bin/python'
    subprocess.run([str(py),'-m','pip','install','--no-index','--find-links',str(root/'cache/wheels'),'--require-hashes','-r',str(locks[name])],check=True,env=env)
    packages=['robot_model_sdk','indoor_perception_lab' if name=='lab' else 'robot_reliability_bench']
    for package in packages:
        wheel=next(wheelhouse.glob(package+'-*.whl'));subprocess.run([str(py),'-m','pip','install','--no-deps',str(wheel)],check=True,env=env)
    subprocess.run([str(py),'-m','pip','check'],check=True,env=env)
    listing=subprocess.check_output([str(py),'-m','pip','list','--format=json'],text=True,env=env);installed=json.loads(listing)
    if name=='bench' and any(i['name']=='indoor-perception-lab' for i in installed):raise AssertionError('Bench independence violated')
    record['environments'][name]={'python':str(py.relative_to(root)),'packages':installed,'lock_sha256':hashlib.sha256(locks[name].read_bytes()).hexdigest(),'pip_check':True,'indoor_perception_lab_installed':name=='lab'}
    (root/'reports/independent-install.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'verified_installation':True,'environments':list(record['environments']),'wheel_sha256':wheel_hashes}))
