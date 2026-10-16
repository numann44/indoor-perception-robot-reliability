from pathlib import Path
import json,hashlib
from data_fetch import Budget,HTTP,ROOT
SOURCES={"ssdlite":"https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth","mobilenet":"https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"}
EXPECTED={"ssdlite":"a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2","mobilenet":"8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04"}
def main():
    http=HTTP(Budget());rows=[]
    for name,url in SOURCES.items():
        out=ROOT/"cache/torch"/Path(url).name;row=http.download(url,out,max_bytes=100_000_000)
        if row['sha256']!=EXPECTED[name]:raise ValueError('Pinned pretrained weight SHA-256 mismatch')
        rows.append({**row,"name":name,"url":url,"code_license":"TorchVision BSD-3-Clause; not automatically the weight license","use_scope":"local noncommercial academic evaluation of officially distributed weights","weight_terms":"upstream pretrained-model terms and COCO/ImageNet provenance; no separate blanket redistribution license asserted","redistribution":"excluded from shareable source archive; official fetch instructions retained"})
    (ROOT/"reports/model-sources.json").write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
if __name__=="__main__":main()
