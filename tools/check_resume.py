"""A functional interrupted-vs-uninterrupted AdamW/RNG/sampler continuation check."""
import json
from pathlib import Path
import torch
from indoor_perception_lab.training import train
from indoor_perception_lab.experiments import root_path,json_write,sha

root=root_path();seed=9901
pack=Path(train('semantic',seed=seed,epochs=2,max_steps=60))
manifest=json.loads((pack/'manifest.json').read_text())
fullpath=root/'reports/runs'/manifest['checkpoint_run_id']/'checkpoint.pt'
full=torch.load(fullpath,weights_only=True,map_location='cpu')
train('semantic',seed=seed,epochs=1,max_steps=60)
manifest=json.loads((pack/'manifest.json').read_text());firstpath=root/'reports/runs'/manifest['checkpoint_run_id']/'checkpoint.pt'
train('semantic',seed=seed,epochs=2,max_steps=60,resume=firstpath,index_path=firstpath.parent/'split-index.jsonl')
manifest=json.loads((pack/'manifest.json').read_text());resumedpath=root/'reports/runs'/manifest['checkpoint_run_id']/'checkpoint.pt'
resumed=torch.load(resumedpath,weights_only=True,map_location='cpu')
assert full['step']==resumed['step']==60
assert all(torch.equal(v,resumed['model'][k]) for k,v in full['model'].items()),'Resumed final parameters differ'
assert torch.equal(full['sampler_generator_state'],resumed['sampler_generator_state'])
assert torch.equal(full['torch_rng'],resumed['torch_rng'])
report={'verified':True,'steps':60,'parameter_equality':'bitwise','full_checkpoint':str(fullpath.relative_to(root)),'interrupted_checkpoint':str(firstpath.relative_to(root)),'resumed_checkpoint':str(resumedpath.relative_to(root)),'full_sha256':sha(fullpath),'resumed_sha256':sha(resumedpath),'limit':'Epoch boundary continuation; abrupt interruption replays work since previous checkpoint. Diagnostic seed9901 is excluded from model selection.'}
json_write(root/'reports/resume-continuation.json',report);print(json.dumps(report))
