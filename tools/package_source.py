"""Create a local source/evidence bundle without datasets, weights or Git history."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'artifacts/portfolio-source.zip';out.parent.mkdir(exist_ok=True)
files=[]
for name in subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0'):
    if not name:continue
    path=Path(name)
    if path.parts[0] in {'data','models','cache','environments','.git','artifacts'} or path.suffix in {'.pt','.pth','.onnx','.mp4','.hdf5','.npz'} or str(path)=='reports/source-bundle.json':continue
    if (root/path).is_file():files.append(path)
temporary=out.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in files:archive.write(root/p,arcname='indoor-perception-portfolio/'+str(p))
with zipfile.ZipFile(temporary) as archive:
    assert archive.testzip() is None
    assert not any(p.endswith(('.pt','.pth','.onnx','.mp4','.hdf5','.npz')) or '/.git/' in p for p in archive.namelist())
temporary.replace(out)
size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
if size>=8_000_000_000:raise RuntimeError('Source bundle exceeded total project budget')
record={'path':str(out.relative_to(root)),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_files':len(files),'excluded':['datasets','model/checkpoint weights','raw motion arrays','environments','caches','Git history'],'license_note':'Source evidence includes attributed CC BY4.0 TUM/JAAD derived images; pretrained/derived weight redistribution is not assumed. Original Git history is excluded because it contains an excluded pilot checkpoint.','project_bytes_after_bundle':size}
(root/'reports/source-bundle.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
