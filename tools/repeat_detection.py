"""Repeat the frozen person fine-tuning configuration without held-out selection."""
import json
from pathlib import Path
import numpy as np
from indoor_perception_lab.training import train
from indoor_perception_lab.datasets import read_index
from indoor_perception_lab.evaluation import detection_evaluate
from indoor_perception_lab.experiments import root_path,Run,json_write
from robot_model_sdk import load_adapter

root=root_path();lock=json.loads((root/'models/selection-detection.json').read_text());lr=float(Path(lock['source_pack']).name.split('-lr')[1].split('-')[0]);rows=read_index(root,'jaad',split='test');results=[]
for seed in (43,44):
    pack=Path(train('detection',seed=seed,lr=lr,epochs=15,max_steps=600));manifest=json.loads((pack/'manifest.json').read_text());adapter=load_adapter(pack);metrics,_=detection_evaluate(root,rows,adapter);adapter.close()
    results.append({'seed':seed,'pack':str(pack.relative_to(root)),'sha256':manifest['sha256'],'validation_AP':manifest['selection_value'],'heldout':metrics})
primary=json.loads((root/'reports/detection-heldout.json').read_text());scores=[primary['trained']['AP'],*[r['heldout']['AP'] for r in results]]
report={'selection_frozen_before_repeats':lock,'seed42_primary_run':primary['run_id'],'repeats':results,'heldout_AP_mean':float(np.mean(scores)),'heldout_AP_sample_std':float(np.std(scores,ddof=1)),'uncertainty_limit':'Three seeds and four correlated test clips; descriptive sample SD is not a confidence interval','test_used_for_selection':False};run=Run('detection-seed-repeat',{'seed':[42,43,44],'locked_configuration':lock['source_pack']},root/'data/jaad/index.jsonl');run.finish(report);json_write(root/'reports/detection-seeds.json',{'run_id':run.id,**report});print(json.dumps({'report':'reports/detection-seeds.json','mean':report['heldout_AP_mean'],'std':report['heldout_AP_sample_std']}))
