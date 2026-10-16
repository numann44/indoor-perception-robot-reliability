from __future__ import annotations
import datetime,hashlib,json,os,platform,random,subprocess,time,uuid,shlex,tarfile,importlib.metadata
from pathlib import Path
import numpy as np

def root_path():
    p=Path(os.environ.get("PORTFOLIO_ROOT",Path(__file__).resolve().parents[4])).resolve()
    if not (p/"PROJECT_ROOT").exists():raise RuntimeError("Set PORTFOLIO_ROOT to the isolated portfolio root")
    return p

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def json_write(path,obj):
    path=Path(path);payload=json.dumps(obj,indent=2,allow_nan=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(payload);temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)

def set_seed(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4);torch.set_num_interop_threads(1) if torch.get_num_interop_threads()!=1 else None

def source_hash(root):
    h=hashlib.sha256()
    for folder in ["packages","tools","configs"]:
        for p in sorted((root/folder).rglob("*")):
            if p.is_file() and p.suffix in {".py",".json",".toml"}:h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()

class Run:
    def __init__(self,task,config,dataset_index=None):
        self.root=root_path();self.id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+task+"-"+uuid.uuid4().hex[:6];self.dir=self.root/"reports/runs"/self.id;self.dir.mkdir(parents=True)
        try:commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=self.root,text=True,stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:commit="uncommitted-initial-work"
        self.record={"run_id":self.id,"task":task,"config":config,"seed":config.get("seed"),"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":"running","commit":commit,"source_snapshot_sha256":source_hash(self.root),"dataset_index_sha256":sha(dataset_index) if dataset_index else None,"command":shlex.join([__import__('sys').executable,*__import__('sys').argv]),"hardware":{"os":platform.system(),"os_version":platform.mac_ver()[0],"architecture":platform.machine(),"cpu_count":os.cpu_count()},"log":"events.jsonl"}
        try:self.record['git_dirty']=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=self.root,text=True).strip())
        except subprocess.CalledProcessError:self.record['git_dirty']=None
        self.record['runtime_versions']={}
        for name in ['torch','torchvision','numpy','opencv-python-headless','onnxruntime']:
            try:self.record['runtime_versions'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:self.record['runtime_versions'][name]=None
        with tarfile.open(self.dir/'source-snapshot.tar.gz','w:gz') as archive:
            for folder in ['packages','tools','configs']:
                for p in sorted((self.root/folder).rglob('*')):
                    if p.is_file() and not p.is_symlink() and p.suffix in {'.py','.json','.toml'}:archive.add(p,arcname=str(p.relative_to(self.root)))
            archive.add(self.root/'requirements.lock',arcname='requirements.lock')
        self.record['source_archive_sha256']=sha(self.dir/'source-snapshot.tar.gz')
        self.log=self.dir/"events.jsonl";self.log.touch();json_write(self.dir/"run.json",self.record)
    def event(self,**fields):
        with self.log.open("a") as f:f.write(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields},allow_nan=False)+"\n")
    def finish(self,metrics,status="completed"):
        self.record.update(status=status,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metrics_file="metrics.json");json_write(self.dir/"metrics.json",metrics);json_write(self.dir/"run.json",self.record)

from robot_model_sdk.classes import NYU40_NAMES
NYU40=list(NYU40_NAMES)

def save_pack(root,name,model,metadata):
    import torch
    out=root/"models"/name;out.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict()},out/"weights.pt")
    metadata={"schema_version":1,"native_weights":"weights.pt","sha256":sha(out/"weights.pt"),**metadata};json_write(out/"manifest.json",metadata);return out
