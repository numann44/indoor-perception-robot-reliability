"""Fresh noneditable v2 wheels, source-byte audit, and isolated package inference."""
from pathlib import Path
import datetime,hashlib,io,json,os,shutil,subprocess,sys,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
# Fail before building/logging if a fixed audit already exists. Preserve evidence.
for protected in ['reports/v2/independent-install','artifacts/v2/wheels','environments/v2-sdk','environments/v2-lab','environments/v2-bench']:
    if (ROOT/protected).exists():raise SystemExit('Independent-install evidence already exists; refusing to overwrite: '+protected)
out=ROOT/'reports/v2/independent-install';out.mkdir(parents=True,exist_ok=True)
wheelhouse=ROOT/'artifacts/v2/wheels';wheelhouse.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PIP_CACHE_DIR=str(ROOT/'cache/pip'),TMPDIR=str(ROOT/'cache/tmp'),PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'installation':'freshvenv; offline hash-locked dependencies; noneditable0.2wheels; isolated-python-I','checks':[]}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
protocol=json.loads((ROOT/'configs/v2/protocol.json').read_text())
def budget():
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    free=shutil.disk_usage(ROOT).free
    ledger=[json.loads(line) for line in (ROOT/'reports/v2/data/download_ledger.jsonl').read_text().splitlines() if line.strip()]
    downloaded=ledger[-1]['cumulative_bytes'] if ledger else 0
    result={'project_bytes':size,'free_bytes':free,'cumulative_download_bytes':downloaded}
    if size>=protocol['project_limit_bytes'] or free<protocol['minimum_free_bytes'] or downloaded>=protocol['download_limit_bytes']:
        raise RuntimeError('Independent-install budget exceeded: '+json.dumps(result))
    return result
def run(name,argv):
    before=budget()
    log=out/(name+'.log')
    with log.open('w') as f:rc=subprocess.call([str(x) for x in argv],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    after=budget()
    record['checks'].append({'name':name,'argv':[str(x) for x in argv],'exit_code':rc,'verified':rc==0,'log':str(log.relative_to(ROOT)),'log_sha256':sha(log),'budget_before':before,'budget_after':after})
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    if rc:raise RuntimeError(name+' failed')
try:
    packages=['robot_model_sdk','indoor_perception_lab','robot_reliability_bench']
    source_paths=[p for name in packages for p in (ROOT/'packages'/name).rglob('*') if p.is_file() and not p.is_symlink() and (p.suffix in {'.py','.json','.toml'} or p.name in {'LICENSE','README.md'}) and not any(part in {'build','__pycache__'} or part.endswith('.egg-info') for part in p.relative_to(ROOT/'packages'/name).parts)]
    source_paths+=[Path(__file__).resolve(),ROOT/'requirements.lock',ROOT/'configs/v2/protocol.json']
    source_bytes={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(set(source_paths))}
    source_before={name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()}
    snapshot=out/'source-before-install.tar.gz'
    with tarfile.open(snapshot,'w:gz') as archive:
        for name,data in source_bytes.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    record['source_before_execution']=source_before
    record['source_archive']={'path':str(snapshot.relative_to(ROOT)),'sha256':sha(snapshot),'captured_before_wheel_build':True}
    record['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    model_paths=[];model_bindings={}
    for task in ('semantic','detection','forecast-fixture','forecast-jaad'):
        lock_path=ROOT/f'models/v2/selection-{task}.json'
        selection=json.loads(lock_path.read_text());selected=selection['selected'];pack=ROOT/selected['pack']
        manifest=json.loads((pack/'manifest.json').read_text());weights=pack/manifest['native_weights']
        expected_weight=selected.get('weights_sha256',selected.get('checkpoint_sha256'))
        if sha(weights)!=expected_weight or sha(pack/'manifest.json')!=selected['manifest_sha256'] or manifest.get('seed')!=42:
            raise ValueError('Selected primary model differs from lock: '+task)
        model_paths.extend([lock_path,pack/'manifest.json',weights])
        model_bindings[task]={'selection':str(lock_path.relative_to(ROOT)),'selection_sha256':sha(lock_path),'pack':selected['pack'],'manifest_sha256':sha(pack/'manifest.json'),'weights_sha256':sha(weights)}
    models_before={str(p.relative_to(ROOT)):sha(p) for p in model_paths}
    record['selected_models_before_execution']=model_bindings
    record['selected_model_files_before_execution']=models_before
    run('build-wheels',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',wheelhouse,*[ROOT/'packages'/n for n in packages]])
    bindings={}
    for name in packages:
        wheel=next(wheelhouse.glob(name+'-0.2.0-*.whl'));bindings[name]={'wheel':str(wheel.relative_to(ROOT)),'sha256':sha(wheel),'source_files':0}
        with zipfile.ZipFile(wheel) as z:
            expected_payload=set()
            for src in (ROOT/'packages'/name/'src'/name).rglob('*'):
                if src.is_file() and src.suffix in ['.py','.json']:
                    rel=str(src.relative_to(ROOT/'packages'/name/'src'))
                    expected_payload.add(rel)
                    if z.read(rel)!=src.read_bytes():raise ValueError('Wheel/source byte mismatch: '+rel)
                    bindings[name]['source_files']+=1
            actual_payload={rel for rel in z.namelist() if rel.startswith(name+'/') and Path(rel).suffix in {'.py','.json'}}
            if actual_payload!=expected_payload:raise ValueError('Wheel/source package payload set differs: '+json.dumps({'missing':sorted(expected_payload-actual_payload),'extra':sorted(actual_payload-expected_payload)}))
            bindings[name]['exact_source_payload_set_verified']=True
    record['wheels']=bindings
    lines=(ROOT/'requirements.lock').read_text().splitlines()
    common={'onnx','onnx-ir','onnxscript','ml_dtypes','safetensors','PyYAML'}
    sdk_base={'numpy','pillow','opencv-python-headless'}
    for role,exclude in [('sdk',set()),('lab',common|{'onnxruntime','flatbuffers','protobuf'}),('bench',common|{'h5py'})]:
        selected_lines=[l for l in lines if l.split('==')[0] in sdk_base] if role=='sdk' else [l for l in lines if l.split('==')[0] not in exclude]
        lock=ROOT/f'configs/v2/{role}-requirements.lock';lock.write_text('\n'.join(selected_lines)+'\n')
        dest=ROOT/'environments'/('v2-'+role)
        if dest.exists():raise RuntimeError('Fresh environment already exists; do not overwrite evidence: '+str(dest))
        run(role+'-venv',[sys.executable,'-m','venv',dest]);py=dest/'bin/python'
        run(role+'-dependencies',[py,'-m','pip','install','--no-index','--find-links',ROOT/'cache/wheels','--require-hashes','-r',lock])
        subset=['robot_model_sdk'] if role=='sdk' else ['robot_model_sdk','indoor_perception_lab' if role=='lab' else 'robot_reliability_bench']
        for name in subset:run(role+'-'+name,[py,'-m','pip','install','--no-deps',ROOT/bindings[name]['wheel']])
        run(role+'-pip-check',[py,'-m','pip','check'])
        code='''import importlib.metadata as md,importlib.util,json,pathlib,sys,numpy as np
import robot_model_sdk as sdk
assert md.version("robot-model-sdk")=="0.2.0"
assert pathlib.Path(sdk.__file__).is_relative_to(pathlib.Path(sys.prefix))
role=sys.argv[1];root=pathlib.Path(sys.argv[2])
import hashlib,zipfile
audit=json.loads((root/"reports/v2/independent-install/run.json").read_text())
installed_packages=["robot_model_sdk"] if role=="sdk" else ["robot_model_sdk","indoor_perception_lab" if role=="lab" else "robot_reliability_bench"]
for package in installed_packages:
 origin=pathlib.Path(importlib.util.find_spec(package).origin).parent
 wheel=root/audit["wheels"][package]["wheel"]
 with zipfile.ZipFile(wheel) as archive:
  for name in archive.namelist():
   if name.startswith(package+"/") and not name.endswith("/"):
    installed=origin/name.removeprefix(package+"/")
    assert installed.is_file() and installed.read_bytes()==archive.read(name), "Installed/wheel byte mismatch: "+name
if role=="sdk":
 assert importlib.util.find_spec("indoor_perception_lab") is None
 assert importlib.util.find_spec("robot_reliability_bench") is None
 assert importlib.util.find_spec("torch") is None
 f=next(iter(sdk.fixture_frames(2)));adapter=sdk.load_adapter("fixture")
 assert isinstance(adapter.predict(f),sdk.ModelOutput);adapter.close()
 from robot_model_sdk.manifest import validate_manifest
 for task in ("semantic","detection","forecast-fixture","forecast-jaad"):
  lock=json.loads((root/f"models/v2/selection-{task}.json").read_text())
  validate_manifest(root/lock["selected"]["pack"],"native")
 print(json.dumps({"role":role,"sdk_origin":sdk.__file__,"python":sys.executable,"torch_present":False,"fixture_and_manifest_admission":"verified"}))
 raise SystemExit(0)
name="robot_reliability_bench" if role=="bench" else "indoor_perception_lab"
assert md.version(name.replace("_","-"))=="0.2.0"
assert pathlib.Path(importlib.util.find_spec(name).origin).is_relative_to(pathlib.Path(sys.prefix))
if role=="bench":
 assert importlib.util.find_spec("indoor_perception_lab") is None
 from robot_reliability_bench.integration import replay_model
else:
 assert importlib.util.find_spec("robot_reliability_bench") is None
 from indoor_perception_lab.pipeline import PerceptionPipeline
f=next(iter(sdk.fixture_frames(2)))
from PIL import Image
Image.fromarray(f.rgb).save(root/f"reports/v2/independent-install/{role}-rgb.png")
fixture=sdk.load_adapter("fixture");assert isinstance(fixture.predict(f),sdk.ModelOutput);fixture.close()
for task in ("semantic","detection"):
 lock=json.loads((root/f"models/v2/selection-{task}.json").read_text());a=sdk.load_adapter(root/lock["selected"]["pack"],device="cpu");r=a.predict(f)
 assert isinstance(r,sdk.ModelOutput)
 if task=="semantic":assert r.semantic.shape==f.rgb.shape[:2]
 a.close()
for domain in ("fixture","jaad"):
 lock=json.loads((root/f"models/v2/selection-forecast-{domain}.json").read_text())
 a=sdk.load_adapter(root/lock["selected"]["pack"],device="cpu")
 times=np.linspace(0,1,11);positions=np.stack([times*.1,np.zeros(11)],axis=1).astype(np.float32)
 prediction=a.predict_timed_history(times,positions,anchor_time=1.0)
 assert prediction is not None and prediction.shape==(20,2) and np.isfinite(prediction).all()
 a.close()
print(json.dumps({"role":role,"sdk_origin":sdk.__file__,"python":sys.executable,"indoor_lab_present":importlib.util.find_spec("indoor_perception_lab") is not None,"native_inference":"verified"}))
'''
        run(role+'-isolated-inference',[py,'-I','-c',code,role,ROOT])
        if role=='sdk':continue
        # Exercise the installed public command, with no editable imports/PYTHONPATH.
        for task in ('semantic','detection'):
            selection=json.loads((ROOT/f'models/v2/selection-{task}.json').read_text())
            pack=ROOT/selection['selected']['pack']
            if role=='lab':
                run(role+'-cli-'+task,[dest/'bin/iplab','infer','--model-pack',pack,'--rgb',out/(role+'-rgb.png'),'--device','cpu','--output',out/(role+'-cli-'+task+'.json')])
            else:
                run(role+'-cli-'+task,[dest/'bin/robotbench','replay-model','--model-pack',pack,'--device','cpu','--mode','offline','--samples','3','--output',out/(role+'-cli-'+task)])
    source_after={name:sha(ROOT/name) for name in source_before}
    record['source_after_execution']=source_after
    record['source_unchanged_during_execution']=source_before==source_after
    if not record['source_unchanged_during_execution']:raise RuntimeError('Package source changed during independent installation/inference')
    models_after={name:sha(ROOT/name) for name in models_before}
    record['selected_models_after_execution']=models_after
    record['selected_models_unchanged_during_execution']=models_before==models_after
    if not record['selected_models_unchanged_during_execution']:raise RuntimeError('Selected model/lock changed during independent installation/inference')
    record.update(status='passed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),final_budget=budget())
except Exception as exc:record.update(status='failed',error=repr(exc),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'run.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'report':'reports/v2/independent-install/run.json'}));raise SystemExit(0 if record['status']=='passed' else 1)
