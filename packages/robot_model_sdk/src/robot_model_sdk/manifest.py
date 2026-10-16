"""Validate model identity and preprocessing before loading executable weights."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from .classes import NYU40_NAMES
from .preprocessing import IMAGENET_MEAN,IMAGENET_STD

def checked_artifact(pack,relative,expected_hash):
    if not isinstance(relative,str) or not relative:raise ValueError('Model artifact must declare a nonempty relative path')
    base=Path(pack).resolve();path=(base/relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():raise ValueError('Model artifact must be a file inside its package')
    if not isinstance(expected_hash,str) or not re.fullmatch('[0-9a-f]{64}',expected_hash):raise ValueError('Artifact SHA-256 is required')
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=expected_hash:raise ValueError('Checkpoint hash mismatch')
    return path

def validate_manifest(pack,backend='native'):
    m=json.loads((Path(pack)/'manifest.json').read_text())
    if m.get('schema_version')!=1:raise ValueError('Unknown model package schema')
    if m.get('task') not in {'detection','semantic','forecast'}:raise ValueError('Unsupported task')
    if backend not in {'native','onnx'}:raise ValueError('Unsupported inference backend')
    if not isinstance(m.get('architecture'),str) or not m.get('domain'):raise ValueError('Architecture and domain are required')
    if m['task'] in {'semantic','detection'}:
        classes=m.get('class_names')
        if not isinstance(classes,list) or not classes or not all(isinstance(c,str) for c in classes):raise ValueError('Class dictionary is required')
        if m['task']=='semantic':
            if len(classes)!=40:raise ValueError('Semantic network requires the NYU40 class dictionary')
            size=m.get('input_size');mean=m.get('mean');std=m.get('std')
            if not isinstance(size,list) or len(size)!=2 or any(not isinstance(x,int) or x<=0 for x in size):raise ValueError('Invalid input size')
            if np.shape(mean)!=(3,) or np.shape(std)!=(3,) or not np.isfinite([mean,std]).all() or min(std)<=0:raise ValueError('Invalid preprocessing normalization')
        elif len(classes)!=m.get('num_classes'):raise ValueError('Detection class count differs from dictionary')
    if m['task']=='forecast':
        settings=m.get('preprocess',{});scale=np.asarray(settings.get('scale',[1,1]),float)
        if scale.shape!=(2,) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid forecast normalization scale')
        if m.get('units') not in {'metres','normalized_image'} or not m.get('coordinate_system'):raise ValueError('Forecast coordinates and units required')
    if m.get('protocol_id')=='quality-v2':
        if m['task']=='semantic':
            if m['architecture'] not in {'mobilenet_v3_large_frozen_64decoder','mobilenet_v3_multiscale_nyu40'}:raise ValueError('Unsupported quality-v2 semantic architecture')
            if tuple(m['class_names'])!=NYU40_NAMES:raise ValueError('quality-v2 requires canonical NYU40 class order')
            if m.get('preprocessing_id')!='quality_rgb_v2' or m.get('postprocess')!='resize_logits_then_argmax':raise ValueError('quality-v2 semantic preprocessing/postprocessing contract mismatch')
            if m['input_size']!=[240,320] or tuple(m['mean'])!=IMAGENET_MEAN or tuple(m['std'])!=IMAGENET_STD:raise ValueError('quality-v2 semantic size/normalization differs from fixed preprocessing')
        elif m['task']=='detection':
            if m['architecture'] not in {'ssdlite320_quality_v2','fasterrcnn_mobilenet_v3_large_320_fpn_quality_v2'}:raise ValueError('Unsupported quality-v2 detection architecture')
            if m['class_names']!=['background','person'] or m.get('preprocessing_id')!='torchvision_detection_transform':raise ValueError('quality-v2 detection class/preprocessing contract mismatch')
        else:
            if m['architecture']!='residual_cv_mlp_2x64' or m.get('preprocessing_id')!='motion_center_scale_v2':raise ValueError('quality-v2 forecast architecture/preprocessing contract mismatch')
            if m.get('network')!={'history':11,'future':20,'hidden':64} or m['preprocess'].get('center_last') is not True or m.get('postprocess',{}).get('relative_to_last') is not True or m['postprocess'].get('unscale') is not True:raise ValueError('quality-v2 forecast history/centering/postprocessing contract mismatch')
    if backend=='native':checked_artifact(pack,m.get('native_weights',''),m.get('sha256'))
    else:
        checked_artifact(pack,m.get('onnx',''),m.get('onnx_sha256'))
        for item in m.get('onnx_external_files',[]):
            checked_artifact(pack,item['path'],item['sha256'])
    return m
