#!/usr/bin/env python3
"""Project-local robotics v2 runtime and experiment entry point."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/v2/robotics'
CACHE = ROOT / 'cache/v2/robotics'
PACKAGE = ROOT / 'packages/robot_reliability_bench/src'
SDK = ROOT / 'packages/robot_model_sdk/src'
for source in (PACKAGE, SDK):
    sys.path.insert(0, str(source))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2), b''):h.update(chunk)
    return h.hexdigest()

def env():
    value=os.environ.copy()
    value.update({'PIXI_CACHE_DIR':str(CACHE/'pixi-cache'), 'RATTLER_CACHE_DIR':str(CACHE/'pixi-cache'),
                  'PIP_CACHE_DIR':str(CACHE/'pip-cache'), 'TMPDIR':str(CACHE/'tmp'),
                  'PYTHONPATH':os.pathsep.join((str(PACKAGE),str(SDK))), 'MUJOCO_GL':'cgl','PYTORCH_ENABLE_MPS_FALLBACK':'0'})
    (CACHE/'tmp').mkdir(parents=True,exist_ok=True)
    return value

def run(command, log, timeout=2700):
    start=time.time()
    with log.open('w') as out:
        try:
            p=subprocess.run([str(x) for x in command], cwd=ROOT, env=env(), stdout=out,
                             stderr=subprocess.STDOUT, timeout=timeout)
            code=p.returncode
        except subprocess.TimeoutExpired:code=124
    return {'command':[str(x) for x in command], 'exit_code':code, 'started_unix':start,
            'finished_unix':time.time(), 'elapsed_seconds':time.time()-start, 'log':str(log)}

def runtime_python():return ROOT/'environments/robotics-v2/bin/python'

def bootstrap_python():
    REPORT.mkdir(parents=True,exist_ok=True)
    wheels=CACHE/'wheels'; wheels.mkdir(parents=True,exist_ok=True)
    record={'kind':'project-local-Python-robotics-runtime','attempts':[], 'sources':[
        'https://pypi.org/project/mujoco/3.15.0/', 'https://pypi.org/project/rosbags/0.11.7/',
        'https://mujoco.readthedocs.io/en/latest/python.html']}
    python=runtime_python()
    if not python.exists():
        result=run([ROOT/'environments/main/bin/python','-m','venv',python.parent.parent],REPORT/'python-venv.log',120)
        record['attempts'].append(result)
        if result['exit_code']:write(REPORT/'python-runtime.json',record);return result['exit_code']
    lockpath=REPORT/'robotics-requirements.lock.txt'
    if lockpath.exists() and list(wheels.glob('*.whl')):
        # Preserve first acquisition proof; revalidation never charges cached artifacts again.
        result=run([python,'-m','pip','install','--no-index','--find-links',wheels,'--require-hashes','-r',lockpath],REPORT/'python-revalidation-install.log',300)
        write(REPORT/'python-runtime-revalidation.json',{'status':'verified' if result['exit_code']==0 else 'failed','attempt':result,'original_acquisition':str(REPORT/'python-runtime.json'),'new_download_bytes':0,'lock_sha256':sha(lockpath),'python':str(python)})
        return result['exit_code']
    command=[python,'-m','pip','download','--only-binary=:all:','--find-links',ROOT/'cache/wheels',
             '--dest',wheels,'mujoco==3.15.0','rosbags==0.11.7','numpy==2.5.4',
             'opencv-python-headless==5.0.0.93','scipy==1.18.1','Pillow==12.3.0','psutil==7.2.2','pytest==9.1.1']
    result=run(command,REPORT/'python-download.log',600);record['attempts'].append(result)
    if result['exit_code']:write(REPORT/'python-runtime.json',record);return result['exit_code']
    import zipfile
    import re
    body_downloads=set(re.findall(r'Downloading ([A-Za-z0-9_.+\-]+\.whl) \(',Path(result['log']).read_text()))
    lock=[]; artifacts=[]
    for wheel in sorted(wheels.glob('*.whl')):
        with zipfile.ZipFile(wheel) as z:
            metadata=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        from email.parser import Parser
        parsed=Parser().parsestr(metadata)
        digest=sha(wheel)
        lock.append(f"{parsed['Name']}=={parsed['Version']} --hash=sha256:{digest}")
        old=ROOT/'cache/wheels'/wheel.name
        artifacts.append({'name':wheel.name,'size_bytes':wheel.stat().st_size,'sha256':digest,
                          'downloaded_new':wheel.name in body_downloads})
    lockpath=REPORT/'robotics-requirements.lock.txt';lockpath.write_text('\n'.join(lock)+'\n')
    result=run([python,'-m','pip','install','--no-index','--find-links',wheels,'--require-hashes','-r',lockpath],REPORT/'python-install.log',300)
    record['attempts'].append(result);record['artifacts']=artifacts
    record['new_download_bytes']=sum(a['size_bytes'] for a in artifacts if a['downloaded_new'])
    record['status']='verified' if not result['exit_code'] else 'failed'
    record['python']=str(python);record['lock_sha256']=sha(lockpath)
    write(REPORT/'python-runtime.json',record);return result['exit_code']

def bootstrap_ros(distro='jazzy',timeout=2700):
    REPORT.mkdir(parents=True,exist_ok=True);(CACHE/'bin').mkdir(parents=True,exist_ok=True)
    archive=CACHE/'pixi-v0.81.0-aarch64-apple-darwin.tar.gz';pixi=CACHE/'bin/pixi'
    url='https://github.com/prefix-dev/pixi/releases/download/v0.81.0/pixi-aarch64-apple-darwin.tar.gz'
    expected='f4e32ea91970d4e11739488817979a5f2c6ebbb9cedb0d6dea74b2b790b272dc'
    if not archive.exists():
        if shutil.disk_usage(ROOT).free<15*1024**3+100*1024**2:raise RuntimeError('Disk reserve <15 GiB')
        with urllib.request.urlopen(url,timeout=60) as response,archive.open('wb') as out:
            for chunk in iter(lambda:response.read(1024**2), b''):out.write(chunk)
    if sha(archive)!=expected:raise RuntimeError('Pixi SHA-256 mismatch')
    if not pixi.exists():
        with tarfile.open(archive) as tf:
            member=next(m for m in tf.getmembers() if m.name=='pixi' or m.name.endswith('/pixi'))
            with tf.extractfile(member) as src,pixi.open('wb') as out:shutil.copyfileobj(src,out)
        pixi.chmod(0o755)
    workspace=ROOT/f'environments/ros-v2-{distro}';workspace.mkdir(exist_ok=True)
    manifest=workspace/'pixi.toml'
    if not manifest.exists():
        manifest.write_text(f'''[workspace]\nname = "robotics-v2-{distro}"\nchannels = ["https://prefix.dev/robostack-{distro}", "https://prefix.dev/conda-forge"]\nplatforms = ["osx-arm64"]\n\n[dependencies]\nros-{distro}-ros-base = "*"\nros-{distro}-sensor-msgs = "*"\nros-{distro}-nav-msgs = "*"\nros-{distro}-geometry-msgs = "*"\nros-{distro}-rosbag2-py = "*"\nros-{distro}-rosbag2-storage-mcap = "*"\n''')
    result=run([pixi,'install','--manifest-path',manifest],REPORT/f'ros-{distro}-install.log',timeout)
    record={'distro':distro,'status':'installed-unverified' if not result['exit_code'] else 'failed',
            'attempt':result, 'pixi_url':url, 'pixi_sha256':expected, 'pixi_download_bytes':archive.stat().st_size,
            'project_local_binary':str(pixi), 'workspace':str(workspace),
            'sources':['https://robostack.github.io/GettingStarted.html','https://robostack.github.io/jazzy.html'],
            'cache':str(CACHE/'pixi-cache'), 'global_modifications':False}
    if (workspace/'pixi.lock').exists():record['lock_sha256']=sha(workspace/'pixi.lock')
    write(REPORT/f'ros-{distro}-runtime.json',record)
    print(json.dumps(record));return result['exit_code']

def main():
    os.chdir(ROOT)
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('bootstrap-python')
    b=sub.add_parser('bootstrap-ros');b.add_argument('--distro',default='jazzy',choices=('jazzy','humble'));b.add_argument('--timeout',type=int,default=2700)
    s=sub.add_parser('simulate');s.add_argument('--output',default='reports/v2/robotics/simulation');s.add_argument('--seeds',type=int,default=10);s.add_argument('--seed-start',type=int,default=100);s.add_argument('--seconds',type=float,default=35);s.add_argument('--model-pack');s.add_argument('--scenarios',default='all')
    dm=sub.add_parser('media-chapter');dm.add_argument('--output',required=True);dm.add_argument('--source',required=True,choices=('openloris','mujoco'));dm.add_argument('--model-pack',required=True);dm.add_argument('--semantic-pack');dm.add_argument('--semantic-backend',default='native',choices=('native','onnx'));dm.add_argument('--tracker',default='T2',choices=('T1','T2','T3'));dm.add_argument('--scenario',default='depth_loss');dm.add_argument('--seconds',type=float,default=23.);dm.add_argument('--seed',type=int,default=42);dm.add_argument('--panel-fps',type=float,default=3.);dm.add_argument('--frame-refs');dm.add_argument('--tracker-lock');dm.add_argument('--quiescent',action='store_true');dm.add_argument('--backend',default='native');dm.add_argument('--device',default='cpu');dm.add_argument('--threads',type=int,default=1)
    ac=sub.add_parser('simulate-active');ac.add_argument('--output',default='reports/v2/robotics/active-person-comparison');ac.add_argument('--model-pack',required=True);ac.add_argument('--tracker-lock',default='models/v2/selection-tracking.json');ac.add_argument('--tracker',required=True,choices=('T1','T2','T3'));ac.add_argument('--high-score',type=float,required=True);ac.add_argument('--quiescent',action='store_true');ac.add_argument('--backend',default='native');ac.add_argument('--device',default='cpu');ac.add_argument('--threads',type=int,default=1)
    m=sub.add_parser('mujoco-smoke');m.add_argument('--output',default='reports/v2/robotics/mujoco-smoke')
    i=sub.add_parser('inventory');i.add_argument('--bag',default='data/v2/openloris/office1-1.bag');i.add_argument('--output',default='reports/v2/robotics/openloris-inventory.json')
    ev=sub.add_parser('odometry-evaluate');ev.add_argument('--bag',default='data/v2/openloris/office1-1.bag');ev.add_argument('--frames',default='reports/v2/robotics/openloris-full-10hz/sensor-frames.jsonl');ev.add_argument('--inventory',default='reports/v2/robotics/openloris-inventory.json');ev.add_argument('--output',default='reports/v2/robotics/openloris-odometry-evaluation')
    r=sub.add_parser('openloris');r.add_argument('--bag',default='data/v2/openloris/office1-1.bag');r.add_argument('--output',default='reports/v2/robotics/openloris');r.add_argument('--frames',type=int,default=100);r.add_argument('--period-ms',type=float,default=0.);r.add_argument('--save-refs',action='store_true')
    server=sub.add_parser('serve');server.add_argument('--socket',required=True);server.add_argument('--model-pack',default='fixture');server.add_argument('--backend',default='native');server.add_argument('--device',default='cpu');server.add_argument('--delay-ms',type=float,default=0);server.add_argument('--threads',type=int,default=1);server.add_argument('--log');server.add_argument('--pipeline-semantic');server.add_argument('--semantic-backend',default='native',choices=('native','onnx'));server.add_argument('--tracker-lock');server.add_argument('--pipeline-tracking',action='store_true');server.add_argument('--pipeline-tracker',default='T2',choices=('T1','T2','T3'))
    ros=sub.add_parser('ros2');ros.add_argument('--output',default='reports/v2/robotics/ros2-pilot');ros.add_argument('--duration',type=float,default=20);ros.add_argument('--publisher-duration',type=float,default=15);ros.add_argument('--fps',type=float,default=50);ros.add_argument('--model-pack',default='fixture');ros.add_argument('--backend',default='native');ros.add_argument('--device',default='cpu');ros.add_argument('--delay-ms',type=float,default=40);ros.add_argument('--threads',type=int,default=1);ros.add_argument('--sensor');ros.add_argument('--frame-refs');ros.add_argument('--pipeline-semantic');ros.add_argument('--semantic-backend',default='native',choices=('native','onnx'));ros.add_argument('--pipeline-tracker',default='T2',choices=('T1','T2','T3'));ros.add_argument('--tracker-lock')
    phys=sub.add_parser('serve-simulator');phys.add_argument('--socket',required=True);phys.add_argument('--output',required=True);phys.add_argument('--scenario',default='clear');phys.add_argument('--seed',type=int,default=42);phys.add_argument('--faults',action='store_true');phys.add_argument('--active-high-score',type=float)
    ros.add_argument('--simulated',action='store_true');ros.add_argument('--sim-faults',action='store_true');ros.add_argument('--scenario',default='clear');ros.add_argument('--seed',type=int,default=42);ros.add_argument('--sim-active-person',action='store_true');ros.add_argument('--quiescent',action='store_true')
    args=p.parse_args()
    if args.command in ('simulate','mujoco-smoke','inventory','openloris','ros2','odometry-evaluate','simulate-active','media-chapter','serve-simulator') and Path(sys.executable).parent!=runtime_python().parent:
        os.execve(str(runtime_python()),[str(runtime_python()),str(Path(__file__).resolve()),*sys.argv[1:]],env())
    if args.command=='serve-simulator':
        from robot_reliability_bench.robotics_v2.closed_loop import serve_physics
        serve_physics(args.socket,ROOT/args.output,args.scenario,args.seed,args.faults,args.active_high_score);return 0
    if args.command=='serve':
        from robot_reliability_bench.robotics_v2.ipc import serve
        serve(args.socket,args.model_pack,args.backend,args.device,args.delay_ms,args.log,args.threads,args.pipeline_semantic,args.pipeline_tracker,args.pipeline_tracking,args.tracker_lock,args.semantic_backend);return 0
    if args.command=='ros2' and args.simulated:
        from robot_reliability_bench.robotics_v2.ros_sim_orchestration import run_closed_loop
        if args.frame_refs or args.sensor:raise ValueError('Live simulated mode uses actual sensors from the running physics process')
        result=run_closed_loop(ROOT,ROOT/args.output,args.duration,args.scenario,args.seed,args.sim_faults,args.model_pack,args.backend,args.device,args.threads,ROOT/args.pipeline_semantic if args.pipeline_semantic else None,ROOT/args.tracker_lock if args.tracker_lock else None,args.pipeline_tracker,args.semantic_backend,args.sim_active_person,args.quiescent)
        print(json.dumps(result));return 0 if result['status']=='verified' else 1
    if args.command=='ros2':
        from robot_reliability_bench.robotics_v2.ros_orchestration import run_ros
        result=run_ros(ROOT,ROOT/args.output,args.duration,args.publisher_duration,args.fps,args.model_pack,args.delay_ms,args.backend,args.device,args.threads,args.sensor,ROOT/args.pipeline_semantic if args.pipeline_semantic else None,args.pipeline_tracker,ROOT/args.tracker_lock if args.tracker_lock else None,args.semantic_backend,ROOT/args.frame_refs if args.frame_refs else None)
        print(json.dumps(result));return 0 if result['status']=='verified' else 1
    if args.command=='bootstrap-python':return bootstrap_python()
    if args.command=='bootstrap-ros':return bootstrap_ros(args.distro,args.timeout)
    if args.command=='mujoco-smoke':
        from robot_reliability_bench.robotics_v2.simulator import geometry_smoke
        print(json.dumps(geometry_smoke(ROOT/args.output)));return 0
    if args.command=='media-chapter':
        from robot_reliability_bench.robotics_v2.demo_media import make_chapter
        r=make_chapter(ROOT,ROOT/args.output,'fixture' if args.model_pack=='fixture' else ROOT/args.model_pack,source=args.source,semantic_pack=ROOT/args.semantic_pack if args.semantic_pack else None,tracker=args.tracker,frame_refs=ROOT/args.frame_refs if args.frame_refs else None,seconds=args.seconds,scenario=args.scenario,seed=args.seed,panel_fps=args.panel_fps,quiescent=args.quiescent,backend=args.backend,device=args.device,threads=args.threads,tracker_lock=ROOT/args.tracker_lock if args.tracker_lock else None,semantic_backend=args.semantic_backend)
        print(json.dumps({k:v for k,v in r.items() if k!='frames'}));return 0
    if args.command=='simulate-active':
        from robot_reliability_bench.robotics_v2.matched_control import run_matched_control
        r=run_matched_control(ROOT,ROOT/args.output,ROOT/args.model_pack,args.tracker,args.high_score,ROOT/args.tracker_lock,quiescent=args.quiescent,backend=args.backend,device=args.device,threads=args.threads)
        print(json.dumps({k:v for k,v in r.items() if k not in ('rows','paired')}));return 0
    if args.command=='simulate':
        from robot_reliability_bench.robotics_v2.simulator import run_scenarios
        print(json.dumps(run_scenarios(ROOT/args.output,args.seeds,args.seconds,args.scenarios,args.model_pack,args.seed_start)));return 0
    if args.command=='odometry-evaluate':
        from robot_reliability_bench.robotics_v2.odometry_evaluation import evaluate_odometry
        print(json.dumps(evaluate_odometry(ROOT/args.bag,ROOT/args.frames,ROOT/args.inventory,ROOT/args.output)));return 0
    if args.command in ('inventory','openloris'):
        from robot_reliability_bench.robotics_v2.openloris import inventory_bag, replay_bag
        result=inventory_bag(ROOT/args.bag,ROOT/args.output) if args.command=='inventory' else replay_bag(ROOT/args.bag,ROOT/args.output,args.frames,args.period_ms,args.save_refs)
        print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
