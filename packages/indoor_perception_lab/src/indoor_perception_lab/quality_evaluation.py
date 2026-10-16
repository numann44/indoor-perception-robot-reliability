"""Dataset-specific, full-resolution evaluation for the frozen quality-v2 protocol."""
from __future__ import annotations
import contextlib,io,json
from pathlib import Path
import numpy as np
from robot_model_sdk import SensorFrameV1
from .experiments import NYU40

def image_frame(root,row):
    import cv2
    rgb=cv2.imread(str(Path(root)/row['rgb_path']))
    if rgb is None:raise ValueError('Unreadable RGB: '+row['sample_id'])
    return SensorFrameV1(row['sample_id'],int(row.get('timestamp_ns',round((row.get('timestamp') or 0)*1e9))),cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))

def confusion_metrics(conf):
    diag=np.diag(conf);union=conf.sum(0)+conf.sum(1)-diag
    iou=np.divide(diag,union,out=np.full(len(diag),np.nan),where=union>0)
    return {'mIoU':float(np.nanmean(iou)) if np.isfinite(iou).any() else None,'pixel_accuracy':float(diag.sum()/conf.sum()) if conf.sum() else None,'per_class_IoU':{NYU40[i]:float(v) if np.isfinite(v) else None for i,v in enumerate(iou)},'pixel_support':conf.sum(1).tolist(),'predicted_support':conf.sum(0).tolist()}

def evaluate_semantic(root,rows,adapter=None,majority_class=None,save_predictions=None):
    import cv2
    conf=np.zeros((40,40),np.int64);scenes={};invalid=[];saved=[]
    if save_predictions is not None:Path(save_predictions).mkdir(parents=True,exist_ok=True)
    for sample_number,row in enumerate(rows):
        gt=cv2.imread(str(Path(root)/row['mask_path']),cv2.IMREAD_UNCHANGED)
        if gt is None or gt.ndim!=2:raise ValueError('Unreadable target mask')
        if adapter is None:
            if majority_class is None:raise ValueError('An adapter or training-majority class is required')
            pred=np.full(gt.shape,majority_class,np.uint8)
        else:pred=adapter.predict(image_frame(root,row)).semantic
        if pred.shape!=gt.shape:raise ValueError('Original-resolution prediction required')
        if not np.isin(gt,np.r_[np.arange(40),255]).all() or np.any(pred>=40):raise ValueError('Unexpected NYU40 label')
        if save_predictions is not None:
            from .experiments import sha
            path=Path(save_predictions)/f'{sample_number:05d}.png'
            if not cv2.imwrite(str(path),pred.astype(np.uint8)):raise IOError('Could not save held-out prediction mask')
            saved.append({'sample_id':row['sample_id'],'mask_path':str(path.relative_to(root)),'sha256':sha(path),'rgb_path':row['rgb_path']})
        keep=gt!=255
        if not keep.any():invalid.append(row['sample_id'])
        cm=np.bincount((gt[keep].astype(np.int64)*40+pred[keep]).ravel(),minlength=1600).reshape(40,40)
        conf+=cm;scenes.setdefault(row['scene_id'],np.zeros_like(conf));scenes[row['scene_id']]+=cm
    datasets=sorted({r['dataset'] for r in rows});domains=sorted({r.get('domain',r['dataset']) for r in rows})
    if saved:
        from .experiments import json_write
        json_write(Path(save_predictions)/'manifest.json',{'role':'frozen model evaluation outputs; not inference inputs','samples':saved})
    return {**confusion_metrics(conf),'protocol_id':'quality-v2','dataset':datasets,'domain':domains,'images':len(rows),'scenes':len(scenes),'images_with_evaluable_pixels':len(rows)-len(invalid),'all_ignore_images':invalid,'resolution':'original-image','ignore_index':255,'absent_class_policy':'unavailable only when target/prediction union both zero','confusion_matrix':conf.tolist(),'per_scene':{k:confusion_metrics(v) for k,v in scenes.items()},'prediction_manifest':str((Path(save_predictions)/'manifest.json').relative_to(root)) if saved else None}

def evaluate_detection(root,rows,adapter):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from .data_v2 import targets_for_row
    images=[];anns=[];preds=[];aid=1;per_scene={};support={'small':0,'medium':0,'large':0,'ignored':0}
    for i,row in enumerate(rows):
        frame=image_frame(root,row);h,w=frame.rgb.shape[:2];images.append({'id':i,'width':w,'height':h});per_scene[row['scene_id']]=per_scene.get(row['scene_id'],0)+1
        for target in targets_for_row(root,row):
            x0,y0,x1,y1=target['bbox_xyxy'];area=max(0,x1-x0)*max(0,y1-y0)
            if area<=0:raise ValueError('Degenerate target box')
            ignored=bool(target.get('ignore',False));support['ignored' if ignored else ('small' if area<1024 else 'medium' if area<9216 else 'large')]+=1
            anns.append({'id':aid,'image_id':i,'category_id':1,'bbox':[x0,y0,x1-x0,y1-y0],'area':area,'iscrowd':int(ignored)});aid+=1
        for d in adapter.predict(frame).detections:
            if d.class_id!=1:continue
            x0,y0,x1,y1=d.bbox
            if not np.isfinite([x0,y0,x1,y1,d.score]).all():raise ValueError('Nonfinite detection')
            if x1>x0 and y1>y0:preds.append({'image_id':i,'category_id':1,'bbox':[x0,y0,x1-x0,y1-y0],'score':d.score})
    gt=COCO();gt.dataset={'info':{},'images':images,'annotations':anns,'categories':[{'id':1,'name':'person'}]}
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        gt.createIndex()
        if preds:dt=gt.loadRes(preds)
        else:dt=COCO();dt.dataset={**gt.dataset,'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.catIds=[1];ev.evaluate();ev.accumulate();ev.summarize()
    # COCO uses -1 for strata without evaluable ground truth, not zero accuracy.
    metrics={name:float(ev.stats[i]) if ev.stats[i]>=0 else None for name,i in {'AP':0,'AP50':1,'AP75':2,'AP_small':3,'AP_medium':4,'AP_large':5,'AR100':8}.items()}
    return {**metrics,'metric_status':{name:'available' if value is not None else 'unavailable' for name,value in metrics.items()},'protocol_id':'quality-v2','dataset':sorted({r['dataset'] for r in rows}),'images':len(rows),'scenes':len(per_scene),'images_per_scene':per_scene,'annotations':len(anns),'predictions':len(preds),'target_support':support,'coordinate_unit':'original decoded image pixels','coco_summary':captured.getvalue()},preds
