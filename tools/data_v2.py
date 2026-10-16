#!/usr/bin/env python3
"""V2 bounded acquisition; v1 files are read-only and the quota is explicit."""
from __future__ import annotations
import argparse, concurrent.futures, datetime, fcntl, hashlib, importlib.util, json, shutil, sys, tarfile, zipfile,threading,time,urllib.error,io,os,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OPENLORIS_BAG_BYTES=1_245_178_774
spec=importlib.util.spec_from_file_location("portfolio_v1_fetch",ROOT/"tools/data_fetch.py")
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


class BudgetV2(legacy.Budget):
    def __init__(self,root=ROOT):
        super().__init__(root,limit=38_000_000_000,project_limit=40_000_000_000,min_free=15_000_000_000)
        self.ledger=self.root/"reports/v2/data/download_ledger.jsonl";self.ledger.parent.mkdir(parents=True,exist_ok=True)
        old=self.root/"reports/data/download_ledger.jsonl"
        with self.ledger.open("a+",encoding="utf-8") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if self._last_total(f)==0:
                lines=old.read_text().splitlines() if old.exists() else []
                total=json.loads(lines[-1])["cumulative_bytes"] if lines else 0
                event={"time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"bytes":total,"cumulative_bytes":total,
                       "url":"local-v1-ledger","purpose":"carry_forward_all_v1_downloads_once","attempt":1}
                f.seek(0,2);f.write(json.dumps(event,separators=(",",":"))+"\n");f.flush()
    def check_storage(self,force=False):
        # One shared scan, not one expensive recursive stat walk per HTTP thread.
        # Free space is checked on every call. A complete project scan is cached
        # for60s across processes and uses scandir without following symlinks.
        free=shutil.disk_usage(self.root).free
        if free<self.min_free:raise RuntimeError("V2 free disk below15GB")
        snapshot=self.root/"reports/v2/data/storage-snapshot.json";lock=snapshot.with_suffix(".lock")
        now=time.time()
        if not force and snapshot.exists():
            old=json.loads(snapshot.read_text())
            if now-old["time_unix"]<60:
                if old["project_bytes"]>=self.project_limit:raise RuntimeError("V2 project40GBquota exceeded")
                return
        with lock.open("a+") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if not force and snapshot.exists():
                old=json.loads(snapshot.read_text())
                if now-old["time_unix"]<60:
                    if old["project_bytes"]>=self.project_limit:raise RuntimeError("V2 project40GBquota exceeded")
                    return
            size=0;stack=[self.root]
            while stack:
                directory=stack.pop()
                try:
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            try:
                                if entry.is_symlink():continue
                                if entry.is_dir(follow_symlinks=False):stack.append(entry.path)
                                elif entry.is_file(follow_symlinks=False):size+=entry.stat(follow_symlinks=False).st_size
                            except FileNotFoundError:pass
                except FileNotFoundError:pass
            legacy.write_json(snapshot,{"time_unix":time.time(),"project_bytes":size,"free_disk_bytes":free,"limit_bytes":self.project_limit,"symlinks_followed":False})
            if size>=self.project_limit:raise RuntimeError(f"V2 projectquota exceeded:{size}")


class _GatedResponse:
    def __init__(self,response,gate):self.response=response;self.gate=gate;self.closed=False
    def __getattr__(self,name):return getattr(self.response,name)
    def __enter__(self):return self
    def close(self):
        if not self.closed:self.response.close();self.gate.release();self.closed=True
    def __exit__(self,*args):self.close()


class HTTPV2(legacy.HTTP):
    def __init__(self,budget):super().__init__(budget);self.gate8=threading.BoundedSemaphore(8);self.gate4=threading.BoundedSemaphore(4);self.throttled=False
    def head(self,url):
        headers=super().head(url);headers["Content-Length"]=str(self.resource_sizes[url]);return headers
    def response(self,url,start=None,end=None):
        for attempt in range(6):
            gate=self.gate4 if self.throttled else self.gate8;gate.acquire()
            try:return _GatedResponse(super().response(url,start,end),gate)
            except urllib.error.HTTPError as e:
                gate.release()
                if e.code not in {429,503} or attempt==5:raise
                self.throttled=True;delay=min(30,max(float(e.headers.get("Retry-After",0) or 0),2**attempt));print(f"HTTPbackoff {e.code} delay{delay}s reduced4workers {url}",flush=True);time.sleep(delay)
            except Exception:gate.release();raise


def http_v2():return HTTPV2(BudgetV2())


def _project_file(relative):
    path=ROOT/relative
    if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError("Acquisition artifact outside project")
    return path


def _verify_artifact(relative,expected_sha=None,expected_bytes=None,checked=None):
    """Admission is local integrity checking, never a replacement acquisition."""
    path=_project_file(relative)
    if not path.is_file():raise ValueError(f"Existing acquisition artifact missing: {relative}")
    size=path.stat().st_size
    if expected_bytes is not None and size!=expected_bytes:raise ValueError(f"Existing acquisition artifact size mismatch: {relative}")
    prior=(checked or {}).get(str(relative))
    digest=prior["sha256"] if prior else legacy.sha256(path)
    if expected_sha is not None and digest!=expected_sha:raise ValueError(f"Existing acquisition artifact checksum mismatch: {relative}")
    record={"path":str(relative),"bytes":size,"sha256":digest}
    if checked is not None:checked[str(relative)]=record
    return record


def _preserve_report(path):
    """Copy exact bytes once, addressed by hash; do not rewrite the source."""
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    destination=ROOT/"reports/v2/data/immutable"/f"{path.stem}-{digest}.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not _exclusive_bytes(destination,raw) and destination.read_bytes()!=raw:raise ValueError("Immutable acquisition snapshot changed")
    return {"path":str(path.relative_to(ROOT)),"sha256":digest,"immutable_copy":str(destination.relative_to(ROOT))}


def _exclusive_bytes(path,raw):
    """Publish complete bytes atomically, with no replace-existing operation."""
    path.parent.mkdir(parents=True,exist_ok=True)
    stage=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with stage.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        try:os.link(stage,path)
        except FileExistsError:return False
        return True
    finally:stage.unlink(missing_ok=True)


def _unique_recheck(dataset,record):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path=ROOT/"reports/v2/data/rechecks"/f"{dataset}-{stamp}-{uuid.uuid4().hex[:8]}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(record,indent=2)+"\n").encode()
    if not _exclusive_bytes(path,raw):raise ValueError("Unique acquisition recheck collision")
    return path


def _write_acquisition_once(dataset,record):
    """Initial evidence is immutable; later execution evidence has a new path."""
    path=ROOT/f"reports/v2/data/{dataset}-download.json";path.parent.mkdir(parents=True,exist_ok=True)
    if not _exclusive_bytes(path,(json.dumps(record,indent=2)+"\n").encode()):
        binding=_preserve_report(path)
        return _unique_recheck(dataset,{"schema_version":2,"recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"role":"acquisition-repeat-preserving-prior-report","prior_acquisition_report":binding,"current_execution":record,"historical_report_replaced":False})
    _preserve_report(path)
    return path


def _frozen_index_admission(dataset,checked):
    """Bind frozen indices and check source/target files without decoding labels."""
    report_path=ROOT/f"reports/v2/data/{dataset}-validation.json"
    index_path=ROOT/f"data/v2/{dataset}/index.jsonl"
    if not report_path.exists() or not index_path.exists():return None
    report=json.loads(report_path.read_text())
    if dataset in {"nyu","jaad"} and not report.get("primary_ready"):raise ValueError(f"Existing {dataset} index has not been frozen; preserve it for investigation")
    _verify_artifact(str(index_path.relative_to(ROOT)),report["index_sha256"],checked=checked)
    frozen=ROOT/f"reports/v2/data/{dataset}-frozen-index-{report['index_sha256'][:12]}.jsonl"
    if frozen.exists():_verify_artifact(str(frozen.relative_to(ROOT)),report["index_sha256"],checked=checked)
    rows=[json.loads(s) for s in index_path.read_text().splitlines() if s.strip()]
    observed={split:sum(row["split"]==split for row in rows) for split in report["counts"]}
    if observed!=report["counts"]:raise ValueError(f"Frozen {dataset} split counts differ from validation")
    for row in rows:
        for key in ("rgb_path","mask_path","raw_label_path","targets_path","video_path"):
            if key in row and not _project_file(row[key]).is_file():raise ValueError(f"Frozen {dataset} referenced artifact missing: {row[key]}")
    if dataset=="jaad":
        for item in report["files"]:
            for directory,suffix,field in [("clips",".mp4","video_sha256"),("annotations",".xml","annotation_sha256"),("targets",".json","target_sha256")]:
                _verify_artifact(f"data/v2/jaad/{directory}/{item['clip']}{suffix}",item[field],checked=checked)
    if dataset=="mot17":
        base="data/v2/mot17/MOT17-11-SDP"
        _verify_artifact(base+"/gt/gt.txt",report["gt_sha256"],checked=checked)
        _verify_artifact(base+"/targets.json",report["target_sha256"],checked=checked)
    return {"validation_report":_preserve_report(report_path),"index_sha256":report["index_sha256"],"rows":len(rows),"counts":report["counts"],"referenced_files_exist":True,"index_and_validation_not_rewritten":True}


def _finish_admission(dataset,report_path,checked,index=None,decoder_run=False):
    BudgetV2(ROOT).check_storage()
    binding=_preserve_report(report_path)
    proof={"schema_version":2,"recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"role":"existing-file-admission","acquisition_report":binding,"checked_artifacts":list(checked.values()),"frozen_index":index,"http_requests":0,"new_download_bytes":0,"historical_report_replaced":False,"decoder_run":decoder_run,"integrity_basis":"stored acquisition SHA256 and byte counts; frozen source/index fingerprints; no prediction or target-value evaluation"}
    path=_unique_recheck(dataset,proof)
    print(f"{dataset} READY existing local artifacts; admission={path.relative_to(ROOT)}; network_requests=0",flush=True)
    return path


def fetch_nyu():
    report_path=ROOT/"reports/v2/data/nyu-download.json";source=ROOT/"data/v2/nyu/source"
    if report_path.exists()!=(source/"nyu_depth_v2_labeled.mat").exists():raise ValueError("Existing NYU source/report pair is incomplete; preserve acquisition evidence for investigation")
    if report_path.exists() and (source/"nyu_depth_v2_labeled.mat").exists():
        report=json.loads(report_path.read_text());checked={}
        if report["path"]!="data/v2/nyu/source/nyu_depth_v2_labeled.mat" or report["bytes"]!=2_972_037_809:raise ValueError("NYU acquisition identity changed")
        _verify_artifact(report["path"],report["sha256"],report["bytes"],checked)
        validation_path=ROOT/"reports/v2/data/nyu-validation.json";validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
        _verify_artifact("data/v2/nyu/source/splits.mat",validation.get("splits_sha256"),2626,checked)
        _verify_artifact("data/v2/nyu/source/classMapping40.mat","ff177194cce9a2163517f1e1e6f1c64c50987162f542dedddc04bbf6ebe6d851",checked=checked)
        frozen=_frozen_index_admission("nyu",checked)
        if frozen:return _finish_admission("nyu",report_path,checked,frozen)
        from indoor_perception_lab.data_v2 import decode_nyu
        print(decode_nyu(ROOT),flush=True)
        return _finish_admission("nyu",report_path,checked,_frozen_index_admission("nyu",checked),decoder_run=True)
    http=http_v2();out=ROOT/"data/v2/nyu/source";out.mkdir(parents=True,exist_ok=True)
    url="https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    head=http.head(url)
    if int(head["Content-Length"])!=2_972_037_809:raise ValueError("NYUv2 published archive size changed")
    # Tiny auxiliary files first; no downloaded code is executed.
    for name,source,expected in [
        ("splits.mat","https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",None),
        ("classMapping40.mat","https://raw.githubusercontent.com/ankurhanda/nyuv2-meta-data/master/classMapping40.mat","ff177194cce9a2163517f1e1e6f1c64c50987162f542dedddc04bbf6ebe6d851")]:
        file=out/name
        if not file.exists():file.write_bytes(http.read(source,max_bytes=200000,purpose="nyu_auxiliary"))
        if expected and legacy.sha256(file)!=expected:raise ValueError("NYU class mapping checksum mismatch")
    if (out/"splits.mat").stat().st_size!=2626:raise ValueError("NYU official split size changed")
    report=legacy.range_download(http,url,out/"nyu_depth_v2_labeled.mat",workers=8,chunk_bytes=16_000_000)
    if report["bytes"]!=2_972_037_809:raise ValueError("NYU downloaded size mismatch")
    _write_acquisition_once("nyu",report)
    from indoor_perception_lab.data_v2 import decode_nyu
    print(decode_nyu(ROOT),flush=True)


class HDFRangeReader(io.RawIOBase):
    """Read-only bounded fileobj for early HDF metadata/train-val preflight.

    Reuses only completed verified full-download chunks. Otherwise anonymous
    1MB ranges are cached locally and every byte is charged to the v2 ledger.
    """
    def __init__(self,http,url,size,cache,full_chunks):
        super().__init__();self.http=http;self.url=url;self.size=size;self.pos=0;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True);self.full_chunks=Path(full_chunks);self.block=1_000_000
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError("Negative HDF seek")
        return self.pos
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:return b""
        if n>8_000_000:raise ValueError("HDF request exceeds8MB per-read cap")
        pieces=[]
        while n:
            full_i=self.pos//16_000_000;full=self.full_chunks/f"{full_i:06d}.bin";proof=full.with_suffix(".json")
            if full.exists() and proof.exists():
                offset=self.pos%16_000_000;amount=min(n,full.stat().st_size-offset)
                with full.open("rb") as f:f.seek(offset);data=f.read(amount)
            else:
                start=(self.pos//self.block)*self.block;end=min(self.size,start+self.block)-1;p=self.cache/f"{start:012d}.bin";proof=p.with_suffix(".json")
                if not p.exists():
                    data=self.http.read(self.url,max_bytes=self.block,start=start,end=end,purpose="nyu_HDF_preflight_range");p.write_bytes(data);legacy.write_json(proof,{"start":start,"end":end,"sha256":legacy.sha256(p)})
                record=json.loads(proof.read_text())
                if legacy.sha256(p)!=record["sha256"]:raise ValueError("HDF preflight cache integrity failure")
                offset=self.pos-start;amount=min(n,end-start+1-offset)
                with p.open("rb") as f:f.seek(offset);data=f.read(amount)
            if len(data)!=amount:raise ValueError("HDF byte range cache truncated")
            pieces.append(data);n-=amount;self.pos+=amount
        return b"".join(pieces)


def preflight_nyu(train_only=False):
    http=http_v2();url="https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat";headers=http.head(url)
    size=int(headers["Content-Length"])
    if size!=2_972_037_809:raise ValueError("NYU range source size changed")
    source=ROOT/"data/v2/nyu/source";reader=HDFRangeReader(http,url,size,source/"preflight-range-cache",source/"nyu_depth_v2_labeled.mat.chunks")
    from indoor_perception_lab.data_v2 import decode_nyu_preflight
    print(decode_nyu_preflight(ROOT,reader,train_only),flush=True)


def selected_jaad():
    old=json.loads((ROOT/"configs/data_manifest.json").read_text())["jaad"];splits={};original=old["splits"]
    for part,extra in [("train",24),("val",4),("test",8)]:
        candidates=set((ROOT/f"data/jaad/official_splits/{part}.txt").read_text().splitlines())-set(original[part])
        ranked=sorted(candidates,key=lambda s:hashlib.sha256(s.encode()).hexdigest())
        splits[part]=(original[part] if part!="test" else [])+ranked[:extra]
    config={"schema_version":2,"source_commit":old["commit"],"archive_url":old["archive_url"],"splits":splits,"selection_rule":"retain originaltrain12/val4; appendhashSHA256(clipID)smallestunused24train/4val; fresh8testexcludeallv1test4","max_frames_per_clip":60,"decode_resize":None}
    path=ROOT/"configs/v2/data-jaad.json"
    if path.exists():
        if json.loads(path.read_text())!=config:raise ValueError("Frozen JAAD selection differs from deterministic source config")
    else:legacy.write_json(path,config)
    return config


def fetch_jaad():
    from indoor_perception_lab.data_v2 import decode_jaad_v2
    config=selected_jaad();report_path=ROOT/"reports/v2/data/jaad-download.json"
    if report_path.exists():
        report=json.loads(report_path.read_text());checked={}
        expected={(part,f"JAAD_clips/{clip}.mp4") for part,clips in config["splits"].items() for clip in clips}
        actual={(item["split"],item["member"]) for item in report}
        if actual!=expected or len(report)!=len(expected):raise ValueError("Existing JAAD acquisition report is incomplete or differs from frozen selection; preserve it for investigation")
        for item in report:
            _verify_artifact(item["path"],item["sha256"],item["bytes"],checked)
            _verify_artifact("data/v2/jaad/annotations/"+Path(item["path"]).stem+".xml",item["annotation_sha256"],checked=checked)
        frozen=_frozen_index_admission("jaad",checked)
        if frozen:return _finish_admission("jaad",report_path,checked,frozen)
        print(decode_jaad_v2(ROOT,config),flush=True)
        return _finish_admission("jaad",report_path,checked,_frozen_index_admission("jaad",checked),decoder_run=True)
    http=http_v2();remote=legacy.RemoteZipFile(config["archive_url"],http)
    with zipfile.ZipFile(remote) as z:index={info.filename:info for info in z.infolist()}
    out=ROOT/"data/v2/jaad";base=f"https://raw.githubusercontent.com/ykotseruba/JAAD/{config['source_commit']}/";records=[]
    for part,clips in config["splits"].items():
        def clip_job(clip):
            video=out/"clips"/f"{clip}.mp4";annotation=out/"annotations"/f"{clip}.xml";video.parent.mkdir(parents=True,exist_ok=True);annotation.parent.mkdir(parents=True,exist_ok=True)
            old_video=ROOT/"data/jaad/clips"/video.name;old_annotation=ROOT/"data/jaad/annotations"/annotation.name
            if old_video.exists() and not video.exists():shutil.copyfile(old_video,video)
            record=legacy.extract_member(remote,index[f"JAAD_clips/{clip}.mp4"],video)
            if not annotation.exists():
                if old_annotation.exists():shutil.copyfile(old_annotation,annotation)
                else:annotation.write_bytes(http.read(base+f"annotations/{clip}.xml",max_bytes=8_000_000,purpose="jaad_v2_annotation"))
            print(json.dumps({"event":"jaad_v2_payload","split":part,"clip":clip}),flush=True);return {**record,"split":part,"annotation_sha256":legacy.sha256(annotation)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records.extend(pool.map(clip_job,clips))
        _unique_recheck("jaad",{"role":"acquisition-progress","completed_splits":list(config["splits"])[:list(config["splits"]).index(part)+1],"files":records})
    _write_acquisition_once("jaad",records)
    print(decode_jaad_v2(ROOT,config),flush=True)


def fetch_openloris():
    out=ROOT/"data/v2/openloris/office1-1.bag";report_path=ROOT/"reports/v2/data/openloris-download.json"
    if out.exists():
        if not report_path.exists():raise ValueError("Existing OpenLORIS bag lacks acquisition report; investigate without re-downloading or inventing historical proof")
        report=json.loads(report_path.read_text());checked={}
        if report["member"]!="office1-1.bag" or report["path"]!="data/v2/openloris/office1-1.bag" or report["bytes"]!=OPENLORIS_BAG_BYTES:raise ValueError("OpenLORIS acquisition identity changed")
        _verify_artifact(report["path"],report["bag_sha256"],report["bytes"],checked)
        with out.open("rb") as f:
            if f.read(13)!=b"#ROSBAG V2.0\n":raise ValueError("Existing OpenLORIS file is not ROS1 bagv2")
        return _finish_admission("openloris",report_path,checked)
    if report_path.exists():raise ValueError("OpenLORIS acquisition report exists but its bag is missing; preserve evidence for investigation")
    http=http_v2();url="https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/rosbag/office1-1_7-rosbag.tar";headers=http.head(url)
    raw=http.read(url,max_bytes=512,start=0,end=511,purpose="openloris_tar_header");member=tarfile.TarInfo.frombuf(raw,"utf-8","strict")
    if member.name!="office1-1.bag" or member.size!=OPENLORIS_BAG_BYTES or not member.isfile():raise ValueError("OpenLORIS first member identity/size mismatch")
    out=ROOT/"data/v2/openloris/office1-1.bag";out.parent.mkdir(parents=True,exist_ok=True);chunks=out.parent/"office1-1.bag.chunks";chunks.mkdir(exist_ok=True);block=16_000_000
    jobs=[(i,start,min(member.size,start+block)) for i,start in enumerate(range(0,member.size,block))]
    def job(item):
        i,start,stop=item;p=chunks/f"{i:04d}.bin";proof=p.with_suffix(".json")
        if p.exists() and proof.exists():
            old=json.loads(proof.read_text())
            if p.stat().st_size==stop-start and legacy.sha256(p)==old["sha256"]:return old
            raise ValueError("OpenLORIS resumed chunk checksum mismatch")
        for attempt in range(1,4):
            part=p.with_suffix(".part")
            try:
                received=0
                with http.response(url,512+start,512+stop-1) as r,part.open("wb") as f:
                    while b:=r.read(1024*1024):http.budget.record(len(b),url,"openloris_selected_bag",attempt);f.write(b);received+=len(b)
                if received!=stop-start:raise OSError("Truncated OpenLORIS chunk")
                part.replace(p);record={"start":512+start,"end":512+stop-1,"bytes":received,"sha256":legacy.sha256(p)};legacy.write_json(proof,record);print(f"OpenLORIS selectedrange {i+1}/{len(jobs)}",flush=True);return record
            except OSError:
                if part.exists():part.unlink()
                if attempt==3:raise
    if not out.exists():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:proofs=list(pool.map(job,jobs))
        part=out.with_suffix(".bag.part")
        with part.open("wb") as dst:
            for i,_,_ in jobs:
                with (chunks/f"{i:04d}.bin").open("rb") as src:shutil.copyfileobj(src,dst,1024*1024)
        if part.stat().st_size!=member.size:raise ValueError("Selected bag assembled size mismatch")
        with part.open("rb") as f:
            if f.read(13)!=b"#ROSBAG V2.0\n":raise ValueError("Not ROS1 bagv2")
        part.replace(out)
    _write_acquisition_once("openloris",{"url":url,"resolved_url":headers["_resolved_url"].split("?",1)[0],"resolved_query_omitted":"anonymous temporary redirect query omitted from provenance","member":member.name,"archive_total_bytes":int(headers["Content-Length"]),"requested_start":512,"requested_end_inclusive":512+member.size-1,"bytes":member.size,"bag_sha256":legacy.sha256(out),"path":str(out.relative_to(ROOT)),"chunk_proofs":proofs,"rosbag_header_verified":True,"full_archive_downloaded":False})
    # Keep resumable chunk proofs until their complete report is durably saved.
    for p in chunks.iterdir():p.unlink()
    chunks.rmdir()
    print(f"OpenLORIS READY {out}",flush=True)


class MetadataZip(legacy.RemoteZipFile):
    def read(self,n=-1):
        n=self.size-self.offset if n is None or n<0 else min(n,self.size-self.offset)
        if n<=0:return b""
        b=self.http.read(self.url,max_bytes=20_000_000,start=self.offset,end=self.offset+n-1,purpose="v2_zip_metadata");self.offset+=len(b);return b


def fetch_mot17():
    from indoor_perception_lab.data_v2 import decode_mot17
    report_path=ROOT/"reports/v2/data/mot17-download.json"
    if report_path.exists():
        report=json.loads(report_path.read_text());checked={};files=report["files"]
        if report["selected_prefix"]!="MOT17/train/MOT17-11-SDP/" or len(files)!=902 or report["selected_member_count"]!=902:raise ValueError("Existing MOT17 acquisition selection differs or is incomplete")
        for item in files:_verify_artifact(item["path"],item["sha256"],item["bytes"],checked)
        frozen=_frozen_index_admission("mot17",checked)
        if frozen:return _finish_admission("mot17",report_path,checked,frozen)
        print(decode_mot17(ROOT),flush=True)
        return _finish_admission("mot17",report_path,checked,_frozen_index_admission("mot17",checked),decoder_run=True)
    url="https://motchallenge.net/data/MOT17.zip";http=http_v2();remote=MetadataZip(url,http)
    if remote.size!=5_860_214_001:raise ValueError("MOT17 archive published size changed")
    with zipfile.ZipFile(remote) as z:all_info=z.infolist()
    prefix="MOT17/train/MOT17-11-SDP/";chosen=[i for i in all_info if i.filename.startswith(prefix) and ("/img1/" in i.filename and i.filename.endswith(".jpg") or i.filename.endswith(("/gt/gt.txt","/seqinfo.ini")))]
    if sum("/img1/" in i.filename for i in chosen)!=900:raise ValueError("MOT17 selected sequence image count mismatch")
    if sum(i.compress_size for i in chosen if "/img1/" in i.filename)!=126_260_408:raise ValueError("MOT17 selected image byte census mismatch")
    out=ROOT/"data/v2/mot17/MOT17-11-SDP";records=[]
    def job(info):return legacy.extract_member(remote,info,out/info.filename[len(prefix):])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for j,record in enumerate(pool.map(job,chosen),1):
            records.append(record)
            if j%50==0:print(f"MOT17 selectedimages {j}/{len(chosen)}",flush=True)
    _write_acquisition_once("mot17",{"url":url,"archive_total_bytes":remote.size,"selected_prefix":prefix,"selected_member_count":len(chosen),"files":records,"full_archive_downloaded":False})
    print(decode_mot17(ROOT),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("dataset",choices=["nyu","nyu-preflight","nyu-preflight-train16","jaad","mot17","openloris","all"]);a=p.parse_args()
    if a.dataset=="nyu-preflight":preflight_nyu()
    if a.dataset=="nyu-preflight-train16":preflight_nyu(True)
    if a.dataset in ["nyu","all"]:fetch_nyu()
    if a.dataset in ["jaad","all"]:fetch_jaad()
    if a.dataset in ["mot17","all"]:fetch_mot17()
    if a.dataset in ["openloris","all"]:fetch_openloris()


if __name__=="__main__":main()
