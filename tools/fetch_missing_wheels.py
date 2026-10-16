from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request,zipfile,email
from data_fetch import Budget,HTTP,ROOT
PACKAGES=[("h5py",None),("pycocotools",None),("opencv-python-headless","5.0.0.93"),("onnx","1.23.2"),("onnxruntime","1.31.0"),("onnxscript",None),("onnx-ir",None),("ml-dtypes",None),("flatbuffers",None)]

def fetch(name,version):
    http=HTTP(Budget());url="https://pypi.org/pypi/"+name+(("/"+version) if version else "")+"/json";data=json.loads(http.read(url,max_bytes=5_000_000,purpose="dependency-metadata"))
    choices=[f for f in data["urls"] if f["filename"].endswith(".whl") and (("macosx" in f["filename"] and ("arm64" in f["filename"] or "universal2" in f["filename"]) and ("cp312" in f["filename"] or "abi3" in f["filename"])) or "py3-none-any" in f["filename"])]
    if not choices:raise RuntimeError("No compatible binary wheel "+name)
    f=min(choices,key=lambda f:f["size"]);out=ROOT/"cache/wheels"/f["filename"]
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest()==f["digests"]["sha256"]:print("reuse "+f["filename"],flush=True);return
    size=f["size"];chunk=1024*1024;parts=ROOT/"cache/tmp"/(f["filename"]+".parts");parts.mkdir(parents=True,exist_ok=True)
    def part(start):
        end=min(size,start+chunk)-1;p=parts/str(start)
        if p.exists() and p.stat().st_size==end-start+1:return p
        b=http.read(f["url"],max_bytes=chunk,start=start,end=end,purpose="dependency-wheel-range");p.write_bytes(b);return p
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(part,range(0,size,chunk)))
    with out.open("wb") as w:
        for start in range(0,size,chunk):w.write((parts/str(start)).read_bytes())
    if hashlib.sha256(out.read_bytes()).hexdigest()!=f["digests"]["sha256"]:raise ValueError("PyPI wheel SHA mismatch")
    for p in parts.iterdir():p.unlink()
    parts.rmdir();print("verified "+f["filename"]+" "+str(size),flush=True)
if __name__=="__main__":
    for n,v in PACKAGES:fetch(n,v)
