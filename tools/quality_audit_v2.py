"""V2 evaluator and train/validation diagnostics; no model-selection test access.

Oracle outputs intentionally read labels to test evaluator arithmetic. They are
not model predictions and never count as model-quality evidence. Source audits
and panels reject test/outside-holdout rows before opening image/target files.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SPLITS = frozenset({'train', 'val'})


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_train_val(rows):
    for row in rows:
        if row.get('split') not in ALLOWED_SPLITS:
            raise ValueError(f"Closed-test diagnostic guard: {row.get('sample_id')} split={row.get('split')}")


def train_val_rows(index):
    # Reading split metadata is permitted; no rejected row path is opened.
    rows = [json.loads(x) for x in Path(index).read_text().splitlines() if x.strip()]
    selected = [r for r in rows if r.get('split') in ALLOWED_SPLITS]
    require_train_val(selected)
    if not selected:
        raise ValueError('No train/validation rows ready for audit')
    return selected


def source_training_membership(root,rows):
    """Check official split metadata before opening any NYU sensor/target path."""
    require_train_val(rows)
    nyu=[r for r in rows if r.get('dataset') in {'nyu','nyuv2'}]
    if not nyu:return {'checked':0,'scope':'No NYU source rows'}
    from scipy.io import loadmat
    split_path=Path(root)/'data/v2/nyu/source/splits.mat'
    source=loadmat(split_path);official_train=set(np.asarray(source['trainNdxs']).reshape(-1).astype(int))
    bad=[r['sample_id'] for r in nyu if r.get('source_index_1based') not in official_train]
    if bad:raise ValueError('NYU train/val row is not an official training source index: '+', '.join(bad))
    return {'checked':len(nyu),'official_split_sha256':sha_file(split_path),
            'scope':'Official source split indices only; no test sensor/label pixels read'}


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def source_context():
    def git(*args):
        r = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    paths = [Path(__file__), ROOT/'packages/indoor_perception_lab/src/indoor_perception_lab/evaluation.py',
             ROOT/'packages/indoor_perception_lab/src/indoor_perception_lab/quality_evaluation.py',
             ROOT/'packages/indoor_perception_lab/src/indoor_perception_lab/data_v2.py',
             ROOT/'packages/indoor_perception_lab/src/indoor_perception_lab/quality_training.py',
             ROOT/'packages/robot_model_sdk/src/robot_model_sdk/classes.py',
             ROOT/'packages/robot_model_sdk/src/robot_model_sdk/manifest.py',
             ROOT/'packages/robot_model_sdk/src/robot_model_sdk/preprocessing.py',
             ROOT/'packages/robot_model_sdk/src/robot_model_sdk/quality_models.py',
             ROOT/'packages/indoor_perception_lab/src/indoor_perception_lab/quality_release.py',
             ROOT/'packages/robot_reliability_bench/src/robot_reliability_bench/metrics.py']
    return {'commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
            'source_sha256': {str(p.relative_to(ROOT)): sha_file(p) for p in paths if p.exists()},
            'command': [sys.executable, *sys.argv], 'python': sys.version,
            'time_utc': datetime.now(timezone.utc).isoformat()}


def make_oracle_fixture(root, folder, count=8):
    """Write small, deliberately synthetic evaluator data with real RGB files."""
    from PIL import Image
    root, folder = Path(root), Path(folder); folder.mkdir(parents=True, exist_ok=True)
    rows, targets, masks = [], {}, {}
    for i in range(count):
        h, w = 64, 96
        rgb = np.full((h,w,3), [35,45,55], np.uint8)
        mask = np.zeros((h,w), np.uint8)
        boxes = [[8,8,24,48], [32,12,52,50], [60,28,92,58]]
        for j, box in enumerate(boxes):
            x0,y0,x1,y1 = box; rgb[y0:y1,x0:x1] = ([220,30,25] if j==0 else [25,200,50])
            mask[y0:y1,x0:x1] = 1 if j==0 else 2
        mask[:3,:5] = 255
        sample = f'audit-oracle/{i:04d}'
        image = folder/f'{i:04d}-rgb.png'; label = folder/f'{i:04d}-mask.png'; target = folder/f'{i:04d}-boxes.json'
        Image.fromarray(rgb).save(image); Image.fromarray(mask).save(label)
        objects = [{'bbox_xyxy': b, 'category_id':1, 'class_id':1, 'ignore':j==2,
                    'iscrowd':j==2, 'track_id':f'oracle-person-{j}b'} for j,b in enumerate(boxes)]
        write_json(target, {'frames': {'0': objects}})
        rows.append({'schema_version':2, 'dataset':'jaad', 'sample_id':sample, 'split':'val',
                     'scene_id':f'oracle-scene-{i}', 'sequence_id':f'oracle-scene-{i}', 'frame_index':0,
                     'timestamp':float(i), 'width':w, 'height':h, 'original_width':w, 'original_height':h,
                     'rgb_path':str(image.relative_to(root)), 'mask_path':str(label.relative_to(root)),
                     'targets_path':str(target.relative_to(root)), 'decode_transform':'synthetic PNG no resize'})
        targets[sample] = objects; masks[sample] = mask
    return rows, targets, masks


class OracleAdapter:
    """Evaluator-only oracle. Never use this as a learned or sensor-only model."""
    def __init__(self, targets, masks, variant='perfect'):
        self.targets, self.masks, self.variant = targets, masks, variant
        self.backend = 'evaluator-oracle'; self.task = 'diagnostic'
    def predict(self, frame):
        from robot_model_sdk import Detection, ModelOutput
        entries = self.targets[frame.sample_id]
        detections = [Detection(t['bbox_xyxy'], .99-j*.03, 1) for j,t in enumerate(entries)]
        mask = self.masks[frame.sample_id].copy(); valid = mask != 255; mask[~valid] = 0
        if self.variant == 'wrong':
            detections = [Detection([70,5,90,20], .99, 1)]
            mask[valid] = (mask[valid]+1) % 3
        elif self.variant == 'empty':
            detections = []; mask[:] = 0
        elif self.variant == 'extra_false_positive':
            detections.insert(0, Detection([70,5,90,20], 1.0, 1))
        return ModelOutput(detections=detections, semantic=mask)
    def close(self):
        pass


def oracle_checks(root, folder, *, evaluator='v2'):
    from robot_reliability_bench.metrics import detection_ap, semantic_metrics
    if evaluator == 'v2':
        module = importlib.import_module('indoor_perception_lab.quality_evaluation')
        det_eval, sem_eval = module.evaluate_detection, module.evaluate_semantic
    elif evaluator == 'legacy':
        module = importlib.import_module('indoor_perception_lab.evaluation')
        det_eval, sem_eval = module.detection_evaluate, module.semantic_evaluate
    else:
        raise ValueError('Evaluator must be v2 or legacy')
    rows, targets, masks = make_oracle_fixture(root, Path(folder)/'oracle-sensor-data')
    converted_targets = {sid:[{'bbox':t['bbox_xyxy'], 'class_id':1, 'ignore':t['ignore']} for t in ts]
                         for sid,ts in targets.items()}
    variants = {}
    for variant in ['perfect','wrong','empty','extra_false_positive']:
        adapter = OracleAdapter(targets, masks, variant)
        d, _ = det_eval(root, rows, adapter); s = sem_eval(root, rows, adapter=adapter)
        predictions, sem_predictions = {}, {}
        from indoor_perception_lab.datasets import sensor_from_row
        for row in rows:
            out = adapter.predict(sensor_from_row(root,row)); predictions[row['sample_id']] = [x.to_json() for x in out.detections]
            sem_predictions[row['sample_id']] = out.semantic
        independent = detection_ap(predictions,converted_targets,prefer_coco=False)
        official = detection_ap(predictions,converted_targets,prefer_coco=True)
        reference_semantic = semantic_metrics(sem_predictions,masks,num_classes=40)
        variants[variant] = {'lab_detection':d, 'lab_semantic':s, 'reference_detection':independent,
                             'official_reference_detection':official, 'reference_semantic':reference_semantic}
    p,w,e,f = (variants[k] for k in ['perfect','wrong','empty','extra_false_positive'])
    checks = {
        'perfect_boxes_AP_one': bool(np.isclose(p['lab_detection']['AP'],1.,atol=1e-8)),
        'perfect_mask_mIoU_one': bool(np.isclose(p['lab_semantic']['mIoU'],1.,atol=1e-8)),
        'perfect_mask_pixel_accuracy_one': bool(np.isclose(p['lab_semantic']['pixel_accuracy'],1.,atol=1e-8)),
        'wrong_boxes_AP_zero': bool(np.isclose(w['lab_detection']['AP'],0.,atol=1e-8)),
        'wrong_mask_mIoU_zero': bool(np.isclose(w['lab_semantic']['mIoU'],0.,atol=1e-8)),
        'empty_detection_AP_zero': bool(np.isclose(e['lab_detection']['AP'],0.,atol=1e-8)),
        'higher_ranked_false_positive_reduces_AP': bool(f['lab_detection']['AP'] < p['lab_detection']['AP']-.01),
        'crowd_prediction_does_not_reduce_perfect_AP': bool(np.isclose(p['lab_detection']['AP'],1.,atol=1e-8)),
        'independent_perfect_reference_one': bool(np.isclose(p['reference_detection']['AP'],1.,atol=1e-8)),
        'lab_matches_official_detection_all_variants': all(np.isclose(v['lab_detection']['AP'],v['official_reference_detection']['AP'],atol=1e-8) for v in variants.values()),
        'lab_matches_reference_semantic_all_variants': all(np.isclose(v['lab_semantic']['mIoU'],v['reference_semantic']['mIoU'],atol=1e-8) for v in variants.values()),
    }
    result = {'kind':'evaluator arithmetic oracle; no learned quality claim', 'evaluator':evaluator,
              'synthetic_images':len(rows), 'valid_detection_boxes':16, 'crowd_regions':8,
              'label_access':'Intentional evaluator-only oracle lookup; no real holdout used',
              'checks':checks, 'passed':all(checks.values()), 'variants':variants}
    write_json(Path(folder)/'oracle.json', result)
    return result


def stratified_first(rows, count=16):
    """Deterministic split-balanced round robin across scenes and frames."""
    from collections import deque
    require_train_val(rows)
    groups={}
    for row in sorted(rows,key=lambda r:(r['split'],r['scene_id'],r.get('frame_index',0),r['sample_id'])):
        groups.setdefault((row['split'],row['scene_id']),[]).append(row)
    queues={part:deque(key for key in sorted(groups) if key[0]==part) for part in sorted(ALLOWED_SPLITS)}
    selected=[]
    while any(queues.values()) and len(selected)<count:
        for queue in queues.values():
            if not queue:continue
            key=queue.popleft();selected.append(groups[key].pop(0))
            if groups[key]:queue.append(key)
            if len(selected)==count:break
    return selected


def read_rgb(root,row):
    from PIL import Image
    require_train_val([row])
    with Image.open(Path(root)/row['rgb_path']) as image:
        return np.asarray(image.convert('RGB'))


def reference_bilinear_mask(logits, height, width):
    """Independent explicit half-pixel bilinear reference, no Torch/OpenCV calls."""
    logits = np.asarray(logits,float)
    if logits.ndim == 4:logits=logits[0]
    classes,h,w = logits.shape; out=np.empty((classes,height,width),float)
    for y in range(height):
        sy=max(0.,min(h-1.,(y+.5)*h/height-.5)); y0=int(np.floor(sy)); y1=min(y0+1,h-1); wy=sy-y0
        for x in range(width):
            sx=max(0.,min(w-1.,(x+.5)*w/width-.5)); x0=int(np.floor(sx)); x1=min(x0+1,w-1); wx=sx-x0
            out[:,y,x] = ((1-wy)*((1-wx)*logits[:,y0,x0]+wx*logits[:,y0,x1]) + wy*((1-wx)*logits[:,y1,x0]+wx*logits[:,y1,x1]))
    return out.argmax(0).astype(np.uint8)


def preprocessing_checks():
    from robot_model_sdk.preprocessing import quality_rgb, quality_logits_to_mask
    rgb=np.zeros((13,19,3),np.uint8);rgb[:,:,0]=255
    x=quality_rgb(rgb,input_size=(7,11))
    x=x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    expected=np.array([(1-.485)/.229,(0-.456)/.224,(0-.406)/.225],np.float32)
    logits=np.full((40,2,2),-100.,np.float32);logits[0]=[[9,0],[9,0]];logits[1]=[[0,1],[0,1]]
    mask=quality_logits_to_mask(logits,4,4)
    mask=mask.detach().cpu().numpy() if hasattr(mask,'detach') else np.asarray(mask)
    reference=reference_bilinear_mask(logits,4,4)
    nearest=np.repeat(np.repeat(logits.argmax(0),2,0),2,1)
    checks={'RGB_not_BGR_normalization':bool(np.allclose(x,expected[:,None,None],rtol=0,atol=1e-6)),
            'CHW_shape':list(x.shape)==[3,7,11], 'float32_finite':x.dtype==np.float32 and bool(np.isfinite(x).all()),
            'logits_bilinear_before_argmax':bool(np.array_equal(mask,reference)),
            'discriminates_nearest_argmax_then_resize':bool(not np.array_equal(reference,nearest))}
    return {'checks':checks,'passed':all(checks.values()),'reference_mask':reference.tolist(),
            'argmax_then_nearest_mask':nearest.tolist(),'normalized_constant_red':x[:,0,0].tolist(),
            'scope':'Hand-computed constant RGB channel values and explicit independent bilinear oracle'}



def ignore_gradient_checks():
    """Measure actual SSD loss masking using controlled anchors, on CPU only."""
    import torch
    from types import SimpleNamespace
    from torchvision.models.detection._utils import BoxCoder
    from robot_model_sdk.quality_models import ignored_ssd_loss
    anchors=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.],[0.,20.,10.,30.]])
    owner=SimpleNamespace(box_coder=BoxCoder((10.,10.,5.,5.)),neg_to_pos_ratio=3.)
    cases={}
    for name,has_positive,ignored in [('one_ignored_background',True,[[20.,0.,30.,10.]]),('positive_inside_ignore',True,[[0.,0.,30.,10.]]),('background_only',False,[[20.,0.,30.,10.]]),('all_background_ignored',False,[[0.,0.,40.,40.]])]:
        logits=torch.tensor([[[0.,1.],[0.,1.],[0.,1.]]],requires_grad=True)
        bbox=torch.zeros((1,3,4),requires_grad=True)
        target={'boxes':anchors[:1].clone() if has_positive else torch.empty((0,4)),
                'labels':torch.tensor([1],dtype=torch.int64) if has_positive else torch.empty((0,),dtype=torch.int64),
                'ignore_boxes':torch.tensor(ignored)}
        matched=torch.tensor([0,-1,-1]) if has_positive else torch.tensor([-1,-1,-1])
        losses=ignored_ssd_loss(owner,[target],{'bbox_regression':bbox,'cls_logits':logits},[anchors],[matched])
        sum(losses.values()).backward()
        cases[name]={'losses':{k:float(v.detach()) for k,v in losses.items()},
                     'classification_grad_per_anchor':logits.grad.abs().sum(2)[0].tolist()}
    a,b,c=(cases[k] for k in ['one_ignored_background','positive_inside_ignore','background_only'])
    checks={'ignored_background_zero_gradient':a['classification_grad_per_anchor'][1]==0.,
            'eligible_background_nonzero_gradient':a['classification_grad_per_anchor'][2]>0.,
            'valid_positive_retained_inside_ignore':b['classification_grad_per_anchor'][0]>0.,
            'background_only_nonignored_has_gradient':any(v>0 for v in c['classification_grad_per_anchor']),
            'all_ignored_background_has_zero_gradient':all(v==0 for v in cases['all_background_ignored']['classification_grad_per_anchor'])}
    return {'passed':all(checks.values()),'checks':checks,'cases':cases,
            'background_only_has_learning_signal':any(v>0 for v in c['classification_grad_per_anchor']),
            'scope':'controlled CPU loss diagnostic, not learned person accuracy',
            'interpretation':'This version learns from nonignored background-only candidates and suppresses all-ignored backgrounds. Its real quality effect is not isolated.'}


def ontology_checks(root):
    """Compare official raw names with a separately sourced mapping; no labels."""
    from scipy.io import loadmat
    from indoor_perception_lab.experiments import NYU40
    root=Path(root);names_path=root/'data/v2/nyu/source/raw_class_names.json'
    mapping_path=root/'data/v2/nyu/source/classMapping40.mat'
    source=json.loads(names_path.read_text());mapping=loadmat(mapping_path)['mapClass'].reshape(-1)
    names=source['raw_names'];model_names=source['NYU40_names']
    aliases={'refrigerator':'refridgerator','nightstand':'night stand'}
    matched=[];unmatched=[]
    for model_id,name in enumerate(model_names,1):
        raw_name=aliases.get(name,name)
        ids=[i for i,n in enumerate(names) if n.strip().lower()==raw_name]
        if ids:
            matched.extend({'raw_id':i+1,'raw_name':names[i],'expected_source40':model_id,
                            'actual_source40':int(mapping[i]),'agree':int(mapping[i])==model_id} for i in ids)
        else:unmatched.append(name)
    checks={'894_raw_names_and_ids':len(names)==894 and source['raw_ids']==list(range(1,895)),
            'mapping_894_integer_source40':len(mapping)==894 and bool(np.all(mapping==mapping.astype(int))) and bool(np.all((mapping>=1)&(mapping<=40))),
            'official_name_metadata_mapping_agrees':bool(np.array_equal(mapping,np.asarray(source['mapClass40_source_ids']))),
            'SDK_class_order_agrees':list(NYU40)==model_names,
            '37_named_categories_with_two_documented_aliases':len(matched)==37 and all(x['agree'] for x in matched),
            'only_three_aggregate_names_absent_from_raw_names':unmatched==['otherstructure','otherfurniture','otherprop']}
    return {'passed':all(checks.values()),'checks':checks,'canonical_name_matches':matched,
            'aggregate_names':unmatched,'raw_names_sha256':sha_file(names_path),'mapping_sha256':sha_file(mapping_path),
            'scope':'Metadata only; official MAT raw names versus separate 894→40 mapping; no train, val or test label pixels opened',
            'limitation':'Named-class correspondence is checked; every rare raw-label aggregation is not independently reannotated.'}


def nyu_real_preprocessing_checks(root,rows):
    """Independent exact 2x block average/nearest label reference for real NYU."""
    from PIL import Image
    from indoor_perception_lab.quality_training import QualityDataset
    require_train_val(rows);records=[]
    for row in rows:
        if row.get('dataset') not in {'nyu','nyuv2'}:continue
        rgb=read_rgb(root,row)
        with Image.open(Path(root)/row['mask_path']) as im:mask=np.array(im)
        if rgb.shape!=(480,640,3) or mask.shape!=(480,640):raise ValueError('NYU preprocessing oracle requires source640x480')
        # OpenCV integer 2x area averaging uses half-up rounding of each 4-pixel block.
        down=((rgb.reshape(240,2,320,2,3).astype(np.uint16).sum(axis=(1,3))+2)//4).astype(np.float32)/255
        expected=((down-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)
        expected_mask=mask[::2,::2].astype(np.int64)
        x,y=QualityDataset(root,[row],'semantic',augment=False)[0]
        diff=float(np.max(np.abs(x.numpy()-expected)))
        records.append({'sample_id':row['sample_id'],'split':row['split'],'RGB_block_average_max_abs_error':diff,
                        'loader_target_nearest_exact':bool(np.array_equal(y.numpy(),expected_mask)),
                        'input_shape':list(x.shape),'target_shape':list(y.shape),'input_finite':bool(np.isfinite(x.numpy()).all())})
    passed=all(r['RGB_block_average_max_abs_error']<=1e-6 and r['loader_target_nearest_exact'] and r['input_finite'] for r in records)
    return {'passed':passed,'rows':len(records),'records':records,
            'scope':'No augmentation; actual train loader versus independent NumPy integer block-average normalization/nearest-target reference; train/val only; no neural forward or test input'}


def source_audit(root,index,folder,*,kind,panels=16):
    """Whole train/val structural audit; bounded source-origin checks/panels."""
    from PIL import Image, ImageDraw
    root,index,folder=Path(root),Path(index),Path(folder)
    rows=train_val_rows(index);membership=source_training_membership(root,rows); evidence=[];bad=[];hist=np.zeros(40,np.int64);groups={};tiny={'boxes':0,'ssd320_height_under4':0,'ssd320_area_under16':0};crowd=0
    split_hist={part:np.zeros(40,np.int64) for part in sorted(ALLOWED_SPLITS)};split_ignore={part:0 for part in sorted(ALLOWED_SPLITS)}
    selected=stratified_first(rows,panels);selected_ids={r['sample_id'] for r in selected};rendered=[]
    last_target_path=None;last_target_frames=None
    for row in rows:
        rgb=read_rgb(root,row);h,w=rgb.shape[:2]; record={'sample_id':row['sample_id'],'split':row['split'],'scene_id':row['scene_id'],'size':[w,h]}
        groups.setdefault(row['split'],set()).add(row['scene_id'])
        if (w,h)!=(row['width'],row['height']):bad.append({'sample_id':row['sample_id'],'reason':'index RGB shape mismatch'})
        if kind=='semantic':
            with Image.open(root/row['mask_path']) as im:mask=np.array(im)
            valid=(mask>=0)&(mask<40)&(mask!=255);record.update(mask_shape=list(mask.shape),ignore_pixels=int((~valid).sum()),valid_pixels=int(valid.sum()),classes=np.unique(mask[valid]).tolist())
            if mask.shape!=(h,w) or np.any((mask!=255)&(mask>=40)):bad.append({'sample_id':row['sample_id'],'reason':'mask shape/encoding'})
            pixels=np.bincount(mask[valid].astype(int),minlength=40);hist+=pixels;split_hist[row['split']]+=pixels;split_ignore[row['split']]+=record['ignore_pixels']
            if row.get('raw_label_path'):
                from scipy.io import loadmat
                with Image.open(root/row['raw_label_path']) as im:raw=np.array(im)
                mapping=loadmat(root/'data/v2/nyu/source/classMapping40.mat')['mapClass'].reshape(-1)
                mapped=np.zeros(raw.shape,np.int64);present=raw>0;mapped[present]=mapping[raw[present]-1];expected=np.where(mapped>0,mapped-1,255).astype(np.uint8)
                record['raw894_to40_exact']=bool(np.array_equal(mask,expected))
                if not record['raw894_to40_exact']:bad.append({'sample_id':row['sample_id'],'reason':'894→40 mapping disagreement'})
            if row['sample_id'] in selected_ids:
                colors=np.stack(((np.arange(40)*47)%255,(np.arange(40)*83)%255,(np.arange(40)*131)%255),1).astype(np.uint8)
                colored=np.full(rgb.shape,100,np.uint8);colored[valid]=colors[mask[valid]]
                panel=Image.new('RGB',(w*2,h+58),'white');panel.paste(Image.fromarray(rgb),(0,58));panel.paste(Image.fromarray(colored),(w,58))
                draw=ImageDraw.Draw(panel);draw.text((8,6),f"TRAIN/VAL SOURCE AUDIT | {row['sample_id']} | {row['split']}",fill='black');draw.text((8,26),'RGB source / target NYU40 mapping (gray=ignore); not model predictions',fill='black')
        else:
            target_path=root/row['targets_path']
            if target_path!=last_target_path:
                last_target_frames=json.loads(target_path.read_text())['frames'];last_target_path=target_path
            ts=last_target_frames.get(str(row['frame_index']),[]);record['targets']=len(ts);record['ignored']=sum(bool(t.get('ignore')) for t in ts);crowd+=record['ignored'];box_errors=[]
            for t in ts:
                b=np.asarray(t['bbox_xyxy'],float);bw,bh=b[2]-b[0],b[3]-b[1]
                if not np.isfinite(b).all() or bw<=0 or bh<=0 or b[0]<0 or b[1]<0 or b[2]>w or b[3]>h:box_errors.append(t.get('track_id'))
                if not t.get('ignore',False):
                    tiny['boxes']+=1; sx,sy=320/w,320/h; tiny['ssd320_height_under4']+=int(bh*sy<4); tiny['ssd320_area_under16']+=int((bw*bh*sx*sy)<16)
                if 'bbox_xyxy_original' in t:
                    expected=np.array(t['bbox_xyxy_original'],float)*float(row.get('bbox_scale',1.));expected[[0,2]]=np.clip(expected[[0,2]],0,w);expected[[1,3]]=np.clip(expected[[1,3]],0,h)
                    if not np.allclose(b,expected,rtol=0,atol=1e-5):box_errors.append('source scaling')
            record['bbox_errors']=box_errors
            if box_errors:bad.append({'sample_id':row['sample_id'],'reason':'invalid bbox/scaling','details':box_errors})
            if row['sample_id'] in selected_ids:
                panel=Image.fromarray(rgb);draw=ImageDraw.Draw(panel)
                for t in ts:
                    b=list(t['bbox_xyxy']);draw.rectangle(b,outline='orange' if t.get('ignore') else 'lime',width=max(2,w//400));draw.text((max(0,b[0]),max(0,b[1]-12)),str(t.get('track_id',''))[:36],fill='orange' if t.get('ignore') else 'lime')
                panel=panel.resize((min(w,1280),round(h*min(w,1280)/w)))
                canvas=Image.new('RGB',(panel.width,panel.height+52),'white');canvas.paste(panel,(0,52));draw=ImageDraw.Draw(canvas);draw.text((8,6),f"TRAIN/VAL SOURCE AUDIT | {row['sample_id']} | {row['split']}",fill='black');draw.text((8,24),'green=valid person | orange=crowd/full occlusion ignored; no model predictions',fill='black');panel=canvas
        if row['sample_id'] in selected_ids:
            out=folder/'panels'/f"{len(rendered):02d}-{row['split']}.png";out.parent.mkdir(parents=True,exist_ok=True);panel.save(out)
            rendered.append({'sample_id':row['sample_id'],'split':row['split'],'scene_id':row['scene_id'],'path':str(out.relative_to(root)),'sha256':sha_file(out),'rgb_sha256':sha_file(root/row['rgb_path']),'target_sha256':sha_file(root/row['mask_path'] if kind=='semantic' else root/row['targets_path'])})
        evidence.append(record)
    preprocessing=nyu_real_preprocessing_checks(root,selected) if kind=='semantic' else None
    if preprocessing is not None and not preprocessing['passed']:bad.append({'reason':'real-source preprocessing oracle disagreement'})
    result={'status':'passed' if not bad else 'failed','kind':kind,'index':str(index.relative_to(root)),'index_sha256':sha_file(index),'rows':len(rows),'source_training_membership':membership,'real_source_preprocessing':preprocessing,'splits':{p:sum(r['split']==p for r in rows) for p in sorted(ALLOWED_SPLITS)},'groups':{k:sorted(v) for k,v in groups.items()},'test_paths_opened':0,'errors':bad,'class_pixel_histogram':hist.tolist() if kind=='semantic' else None,'split_pixel_support':{p:{'class_pixel_histogram':split_hist[p].tolist(),'represented_classes':int((split_hist[p]>0).sum()),'valid_pixels':int(split_hist[p].sum()),'ignore_pixels':split_ignore[p]} for p in sorted(ALLOWED_SPLITS)} if kind=='semantic' else None,'detection_scale_diagnostic':tiny if kind=='detection' else None,'ignored_detection_regions':crowd if kind=='detection' else None,'panels':rendered,'row_evidence':evidence,'scope':'train/validation source targets only; no model quality or test selection'}
    write_json(folder/'source-audit.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluator',choices=['v2','legacy'],default='v2');p.add_argument('--oracles',action='store_true');p.add_argument('--preprocessing',action='store_true');p.add_argument('--ignore-gradients',action='store_true');p.add_argument('--ontology',action='store_true')
    p.add_argument('--index',type=Path);p.add_argument('--kind',choices=['semantic','detection']);p.add_argument('--panels',type=int,default=16);p.add_argument('--output',type=Path,default=ROOT/'reports/v2/audit')
    a=p.parse_args();rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-audit-'+uuid.uuid4().hex[:6];folder=(a.output if a.output.is_absolute() else ROOT/a.output)/rid;folder.mkdir(parents=True,exist_ok=True)
    result={'run_id':rid,**source_context(),'closed_test_policy':'No real test labels/predictions opened by diagnostic audit','checks':{}}
    if a.oracles:result['checks']['oracles']=oracle_checks(ROOT,folder,evaluator=a.evaluator)
    if a.preprocessing:result['checks']['preprocessing']=preprocessing_checks()
    if a.ignore_gradients:result['checks']['ignore_gradients']=ignore_gradient_checks()
    if a.ontology:result['checks']['ontology']=ontology_checks(ROOT)
    if a.index:
        if not a.kind:p.error('--kind required with --index')
        result['checks']['sources']=source_audit(ROOT,a.index if a.index.is_absolute() else ROOT/a.index,folder,kind=a.kind,panels=a.panels)
    if not result['checks']:p.error('Choose --oracles, --preprocessing, --ignore-gradients, --ontology and/or --index')
    result['passed']=all(v.get('passed',v.get('status')=='passed') for v in result['checks'].values());result['exit_code']=0 if result['passed'] else 1
    write_json(folder/'run.json',result);write_json(folder.parent/'latest.json',{'run':str(folder.relative_to(ROOT)),'passed':result['passed'],'exit_code':result['exit_code']})
    print(json.dumps({'run_id':rid,'passed':result['passed'],'exit_code':result['exit_code'],'run':str((folder/'run.json').relative_to(ROOT))}));return result['exit_code']

if __name__=='__main__':raise SystemExit(main())
