"""Fixed-corruption task accuracy; targets stay outside inference frames."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from robot_model_sdk import load_adapter
from robot_reliability_bench.inputs import load_frames,_path
from robot_reliability_bench.runner import model_identity,_hash_file,_configure_threads
from robot_reliability_bench.evaluation import evaluate_corruptions
from robot_reliability_bench.model_evaluation import read_targets
from .provenance import context,finish


def evaluate(model_pack,input_manifest,output,task='semantic',backend='native',device='cpu',split='val',test_unlocked=False,conditions=None):
    if split=='test' and not test_unlocked:raise ValueError('Held-out test labels are locked until model selection is frozen')
    index=Path(input_manifest).resolve();rows=[json.loads(s) for s in index.read_text().splitlines() if s.strip()]
    accepted={'val','validation'} if split=='val' else {split};rows=[r for r in rows if r.get('split') in accepted]
    _configure_threads(2,native_neural=backend=='native')
    frames,_,source=load_frames(index,count=len(rows),split=split)
    if task=='detection':targets,target_hashes=read_targets(index,rows,task)
    elif task=='semantic':
        targets={};target_hashes={}
        for row in rows:
            name=row.get('mask_path') or row.get('semantic_path') or row.get('targets_path')
            if name is None:raise ValueError('Semantic target path absent')
            path=_path(index,name)
            with Image.open(path) as image:targets[row['sample_id']]=np.asarray(image)
            target_hashes[str(path)]=_hash_file(path)
    else:raise ValueError('Task must be detection or semantic')
    identity=model_identity(model_pack)
    provenance=context({'model':identity,'source_index_sha256':_hash_file(index),'task':task,'backend':backend,'device':device,'split':split,'conditions':conditions,'seed':42,'test_unlocked':test_unlocked})
    adapter=load_adapter(model_pack,backend=backend,device=device)
    try:metrics=evaluate_corruptions(frames,targets,adapter,task=task,seed=42,conditions=conditions)
    finally:adapter.close()
    if model_identity(model_pack)['sha256']!=identity['sha256']:raise RuntimeError('Model pack changed during fixed evaluation')
    result={'status':'verified','provenance':finish(provenance),'source':source,'images':len(frames),'split':split,
            'target_files_sha256':target_hashes,'inference_target_access':False,'evaluation':metrics}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
