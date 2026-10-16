"""Shared v2 preprocessing: CPU image preparation is explicit on all backends."""
from __future__ import annotations
import numpy as np

IMAGENET_MEAN=(.485,.456,.406)
IMAGENET_STD=(.229,.224,.225)

def quality_rgb(rgb,input_size=(240,320),mean=IMAGENET_MEAN,std=IMAGENET_STD):
    import cv2
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected uint8 RGB HWC')
    ih,iw=map(int,input_size)
    if min(ih,iw)<=0:raise ValueError('Input dimensions must be positive')
    method=cv2.INTER_AREA if ih<=rgb.shape[0] and iw<=rgb.shape[1] else cv2.INTER_LINEAR
    resized=cv2.resize(rgb,(iw,ih),interpolation=method).astype(np.float32)/255
    mean=np.asarray(mean,np.float32);std=np.asarray(std,np.float32)
    if mean.shape!=(3,) or std.shape!=(3,) or not np.isfinite([mean,std]).all() or np.any(std<=0):raise ValueError('Invalid normalization')
    return np.ascontiguousarray(((resized-mean)/std).transpose(2,0,1))

def quality_logits_to_mask(logits,height,width):
    import cv2
    logits=np.asarray(logits,np.float32)
    if logits.ndim!=3 or not np.isfinite(logits).all():raise ValueError('Expected finite CxHxW logits')
    if logits.shape[-2:]!=(height,width):
        logits=cv2.resize(logits.transpose(1,2,0),(width,height),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
    return logits.argmax(0).astype(np.uint8)
