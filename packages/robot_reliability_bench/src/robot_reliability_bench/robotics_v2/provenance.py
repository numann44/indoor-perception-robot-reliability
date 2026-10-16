"""Run identity linked to exact configuration and source bytes."""
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid


def context(config,root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[5]
    source={}
    for directory in (root/'packages/robot_reliability_bench/src/robot_reliability_bench/robotics_v2',root/'packages/robot_model_sdk/src/robot_model_sdk',root/'packages/indoor_perception_lab/src/indoor_perception_lab'):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in {'.py','.json'}:source[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (root/'tools/robotics_v2.py',root/'tools/benchmark_v2.py',root/'tools/export_v2.py'):
        if path.is_file():source[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    canonical=json.dumps(config,sort_keys=True,separators=(',',':')).encode()
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    except subprocess.CalledProcessError:commit=None
    versions={}
    for name in ('mujoco','rosbags','numpy','opencv-python-headless','psutil','scipy','torch','torchvision','onnxruntime'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    import psutil
    hardware={'platform':platform.platform(),'machine':platform.machine(),'cpu_logical_cores':psutil.cpu_count(),'cpu_physical_cores':psutil.cpu_count(logical=False),'RAM_total_bytes':psutil.virtual_memory().total}
    try:dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip())
    except subprocess.CalledProcessError:dirty=None
    return {'run_id':datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8],
            'started_unix':time.time(),'command':[sys.executable,*sys.argv], 'config':config,
            'config_sha256':hashlib.sha256(canonical).hexdigest(),'source_sha256':source,'source_snapshot_scope':'Project SDK/Bench/Lab Python and JSON/schema source trees plus robotics/benchmark/export tools; independent model timing separately binds installed package sources','git_commit':commit,
            'python':sys.version,'hardware':hardware,'git_tracked_files_dirty':dirty,'dependencies':versions}


def finish(record):return {**record,'finished_unix':time.time(),'elapsed_seconds':time.time()-record['started_unix']}
