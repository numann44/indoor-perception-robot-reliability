"""Bundle current local V2 code and small textual evidence, excluding dataset media."""
from pathlib import Path
import datetime,hashlib,json,subprocess,zipfile,shutil
from indoor_perception_lab.experiments import root_path,sha,json_write
ROOT=root_path()


def main():
    output=ROOT/'artifacts/portfolio-source-v2.zip';output.parent.mkdir(exist_ok=True)
    protocol=json.loads((ROOT/'configs/v2/protocol.json').read_text())
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    roots={'README.md','README_V1.md','FINAL_REPORT.md','FINAL_REPORT_V2.md','LICENSE','requirements.lock','portfolio','CHECKPOINT.md','RESUME.md','PROJECT_ROOT','.gitignore'}
    text={'.py','.json','.jsonl','.toml','.lock','.md','.txt','.csv','.log','.svg'}
    files={};omitted=[]
    for name in names:
        if not name:continue
        if name=='reports/v2/source-bundle.json':continue
        rel=Path(name);path=ROOT/rel
        if not path.is_file() or path.is_symlink():continue
        # Per-frame JSON can contain compressed semantic pixels; it is media
        # even though its filename is textual. Keep only aggregate/manifests.
        derived_pixels=(rel.parts[:2]==('reports','v2') and 'media' in rel.parts[2:] and 'results' in rel.parts[2:])
        # Raw tick traces and unpacked verification scratch/source copies remain
        # local. Their aggregate results and provenance belong in this review ZIP.
        report_copy_or_trace=(rel.parts[:2]==('reports','v2') and
                              (rel.name=='trace.jsonl' or 'clones' in rel.parts[2:] or
                               any(part.endswith('-temp') for part in rel.parts[2:])))
        if derived_pixels or report_copy_or_trace:
            omitted.append(name);continue
        source=rel.parts[0] in {'packages','tools','configs','tests','docs'}
        evidence=rel.parts[:2]==('reports','v2') and path.suffix in text and path.stat().st_size<=1_000_000
        if (source and path.suffix in text) or name in roots or evidence:
            files[name]=path
        elif rel.parts[:2]==('reports','v2'):omitted.append(name)
    for path in (ROOT/'models/v2').rglob('manifest.json'):files[str(path.relative_to(ROOT))]=path
    for path in (ROOT/'models/v2').glob('selection-*.json'):files[str(path.relative_to(ROOT))]=path
    required=['tools/portfolio_cli.py','packages/robot_model_sdk/src/robot_model_sdk/contracts.py','docs/v2/requirements.json']
    if not all(name in files for name in required):raise ValueError('Missing required current source/matrix files')
    total=sum(path.stat().st_size for path in files.values())
    if total>200_000_000:raise ValueError('Unexpected textual source bundle size')
    size=sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file() and not path.is_symlink())
    if size+total>=protocol['project_limit_bytes']:raise RuntimeError('Project budget cannot accommodate source bundle')
    temporary=output.with_suffix('.zip.tmp')
    snapshot={name:path.read_bytes() for name,path in sorted(files.items())}
    snapshot_hashes={name:hashlib.sha256(data).hexdigest() for name,data in snapshot.items()}
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in snapshot.items():archive.writestr('indoor-perception-portfolio-v2/'+name,data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError('Invalid source archive CRC')
        if any(Path(name).suffix in {'.pt','.pth','.onnx','.png','.jpg','.mp4','.bag','.hdf5','.npz'} for name in archive.namelist()):raise ValueError('Excluded artifact leaked into source bundle')
        for name,digest in snapshot_hashes.items():
            if hashlib.sha256(archive.read('indoor-perception-portfolio-v2/'+name)).hexdigest()!=digest:raise ValueError('ZIP member differs from source snapshot: '+name)
    if any(sha(files[name])!=digest for name,digest in snapshot_hashes.items()):
        temporary.unlink();raise RuntimeError('Source/evidence changed while creating the bundle; preserved prior bundle')
    if shutil.disk_usage(ROOT).free-temporary.stat().st_size<protocol['minimum_free_bytes']:temporary.unlink();raise RuntimeError('Free disk acceptance failed')
    temporary.replace(output)
    record={'protocol_id':'quality-v2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(output.relative_to(ROOT)),'bytes':output.stat().st_size,'sha256':sha(output),'source_files':len(files),'uncompressed_bytes':total,'files':{name:sha(path) for name,path in sorted(files.items())},'excluded':['datasets and dataset-derived media, including per-frame JSON containing compressed semantic pixels','weights/checkpoints','runtime environments/caches','Git history','per-run archived source snapshots and unpacked verification scratch copies','raw per-tick trace.jsonl files','textual reports over1MB'],'omitted_v2_evidence_files':omitted,'scope':'local source review bundle; exact full measurements, snapshots and media remain separately available in the local project; no upload or publication'}
    record['files']=snapshot_hashes
    record['archive_member_hashes_verified']=True
    record['source_unchanged_during_bundle']=True
    json_write(ROOT/'reports/v2/source-bundle.json',record)
    print(json.dumps({k:record[k] for k in ['path','bytes','sha256','source_files']}))


if __name__=='__main__':main()
