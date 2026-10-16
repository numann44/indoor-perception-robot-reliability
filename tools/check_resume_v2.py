"""Actually interrupt/restart a v2 optimizer+scheduler+sampler CPU run."""
import os,json
from pathlib import Path
import torch
from indoor_perception_lab.quality_training import train_quality
from indoor_perception_lab.experiments import root_path,json_write,sha
root=root_path();kwargs={'configuration':'S1','seed':9972,'device':'cpu','epochs':3,'max_steps':24,'overfit':True,'index_path':root/'data/hypersim/index.jsonl'}
pack,full=train_quality(**kwargs);full_state=torch.load(full/'checkpoint.pt',map_location='cpu',weights_only=True)
pack,partial=train_quality(**kwargs,stop_after_epochs=1)
pack,resumed=train_quality(**kwargs,resume=partial/'checkpoint.pt');state=torch.load(resumed/'checkpoint.pt',map_location='cpu',weights_only=True)
def eq(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
    return a==b
checks={k:eq(full_state[k],state[k]) for k in ['model','best_model','optimizer','scheduler','torch_rng','python_rng','numpy_rng','sampler_generator_state','step','next_epoch','stale','best']}
result={'protocol_id':'quality-v2','device':'cpu','interrupted_after_steps':8,'completed_steps':24,'checks':checks,'all_passed':all(checks.values()),'full_run':str(full.relative_to(root)),'partial_run':str(partial.relative_to(root)),'resumed_run':str(resumed.relative_to(root)),'full_checkpoint_sha256':sha(full/'checkpoint.pt'),'resumed_checkpoint_sha256':sha(resumed/'checkpoint.pt'),'scope':'16trainRGB/masks only; not performance or holdout evidence'}
json_write(root/'reports/v2/resume-cpu.json',result);print(json.dumps(result));raise SystemExit(0 if result['all_passed'] else 1)
