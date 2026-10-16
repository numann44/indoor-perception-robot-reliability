"""Repeat the already validation-frozen configuration; never select using test scores."""
import json
from pathlib import Path
import numpy as np
from indoor_perception_lab.training import train
from indoor_perception_lab.datasets import read_index
from indoor_perception_lab.evaluation import semantic_evaluate
from indoor_perception_lab.experiments import root_path,Run,json_write
from robot_model_sdk import load_adapter

root=root_path()
lock=json.loads((root/'models/selection-semantic.json').read_text())
source=json.loads((root/lock['source_pack']/'manifest.json').read_text())
lr=float(Path(lock['source_pack']).name.split('-lr')[1].split('-')[0])
rows=read_index(root,'hypersim',split='test')
results=[]
for seed in (43,44):
    pack=Path(train('semantic',seed=seed,lr=lr,epochs=15,max_steps=450))
    manifest=json.loads((pack/'manifest.json').read_text())
    adapter=load_adapter(pack)
    metrics=semantic_evaluate(root,rows,adapter);adapter.close()
    results.append({'seed':seed,'pack':str(pack.relative_to(root)),'sha256':manifest['sha256'],'validation_mIoU':manifest['selection_value'],'heldout':metrics})
primary=json.loads((root/'reports/semantic-heldout.json').read_text())
scores=[primary['trained']['mIoU'],*[r['heldout']['mIoU'] for r in results]]
report={'selection_frozen_before_repeats':lock,'seed42_primary_run':primary['run_id'],'repeats':results,'heldout_mIoU_mean':float(np.mean(scores)),'heldout_mIoU_sample_std':float(np.std(scores,ddof=1)),'uncertainty_limit':'Three seeds and two test scenes; sample standard deviation is descriptive, not a calibrated confidence interval','test_used_for_selection':False}
run=Run('semantic-seed-repeat',{'seed':[42,43,44],'locked_configuration':str(root/lock['source_pack']),'test_used_for_selection':False},root/'data/hypersim/index.jsonl');run.finish(report)
json_write(root/'reports/semantic-seeds.json',{'run_id':run.id,**report})
print(json.dumps({'report':'reports/semantic-seeds.json','mean':report['heldout_mIoU_mean'],'std':report['heldout_mIoU_sample_std']}))
