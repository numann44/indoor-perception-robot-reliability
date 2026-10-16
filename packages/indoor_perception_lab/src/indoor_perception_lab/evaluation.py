from __future__ import annotations
import contextlib,io,json
from pathlib import Path
import numpy as np
from robot_model_sdk import load_adapter
from .datasets import sensor_from_row,jaad_targets
from .experiments import NYU40

def semantic_evaluate(root,rows,adapter=None,majority_class=None):
    import cv2
    conf=np.zeros((40,40),np.int64);per_scene={};all_ignore=[]
    for row in rows:
        gt=cv2.imread(str(Path(root)/row["mask_path"]),cv2.IMREAD_UNCHANGED)
        pred=adapter.predict(sensor_from_row(root,row)).semantic if adapter else np.full(gt.shape,majority_class,np.uint8)
        if pred.shape!=gt.shape:raise ValueError("Semantic prediction/target shape mismatch")
        valid=(gt!=255)&(gt<40)
        if not valid.any():all_ignore.append(row["sample_id"])
        cm=np.bincount((gt[valid].astype(int)*40+pred[valid]).ravel(),minlength=1600).reshape(40,40);conf+=cm;per_scene.setdefault(row["scene_id"],np.zeros((40,40),np.int64));per_scene[row["scene_id"]]+=cm
    def metrics(c):
        diag=np.diag(c);union=c.sum(0)+c.sum(1)-diag;ious=np.divide(diag,union,out=np.full(40,np.nan),where=union>0)
        return {"mIoU":float(np.nanmean(ious)) if np.isfinite(ious).any() else None,"pixel_accuracy":float(diag.sum()/c.sum()) if c.sum() else None,"per_class_IoU":{NYU40[i]:float(x) if np.isfinite(x) else None for i,x in enumerate(ious)},"pixel_support":c.sum(1).tolist()}
    return {**metrics(conf),"confusion_matrix":conf.tolist(),"per_scene":{s:metrics(c) for s,c in per_scene.items()},"images":len(rows),"images_with_evaluable_pixels":len(rows)-len(all_ignore),"all_ignore_images":all_ignore,"scenes":len(per_scene),"domain":"Hypersim synthetic indoor","ignore_index":255,"absent_class_policy":"IoU unavailable if prediction and target union both zero"}

def detection_evaluate(root,rows,adapter):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    anns=[];preds=[];images=[];aid=1;per_clip={}
    for i,row in enumerate(rows):
        frame=sensor_from_row(root,row);images.append({"id":i,"width":frame.rgb.shape[1],"height":frame.rgb.shape[0]})
        per_clip[row["scene_id"]]=per_clip.get(row["scene_id"],0)+1
        for t in jaad_targets(root,row):
            x0,y0,x1,y1=t["bbox_xyxy"];anns.append({"id":aid,"image_id":i,"category_id":1,"bbox":[x0,y0,x1-x0,y1-y0],"area":(x1-x0)*(y1-y0),"iscrowd":int(t["ignore"])});aid+=1
        for d in adapter.predict(frame).detections:
            if d.class_id!=1:continue
            x0,y0,x1,y1=d.bbox
            if x1>x0 and y1>y0:preds.append({"image_id":i,"category_id":1,"bbox":[x0,y0,x1-x0,y1-y0],"score":d.score})
    gt=COCO();gt.dataset={"info":{},"images":images,"annotations":anns,"categories":[{"id":1,"name":"person"}]}
    with contextlib.redirect_stdout(io.StringIO()) as capture:
        gt.createIndex()
        if preds:dt=gt.loadRes(preds)
        else:dt=COCO();dt.dataset={**gt.dataset,"annotations":[]};dt.createIndex()
        ev=COCOeval(gt,dt,"bbox");ev.params.catIds=[1];ev.evaluate();ev.accumulate();ev.summarize()
    return {"AP":float(ev.stats[0]),"AP50":float(ev.stats[1]),"AP75":float(ev.stats[2]),"AR100":float(ev.stats[8]),"images":len(rows),"clips":len(per_clip),"images_per_clip":per_clip,"annotations":len(anns),"predictions":len(preds),"protocol":"official COCOeval person only; group/full-occlusion regions ignored as crowd; maxDets100","coordinate_unit":"decoded-image pixels","domain":"JAAD real; predominantly traffic","coco_summary":capture.getvalue()},preds
