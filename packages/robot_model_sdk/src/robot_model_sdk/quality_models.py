"""Versioned v2 neural architectures and crowd-aware detection training.

Loss structure follows TorchVision BSD-3-Clause SSD implementation, with explicit
ignored-background masking. Initial weights are supplied as local verified files.
"""
from __future__ import annotations
import types
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import ssdlite320_mobilenet_v3_large,fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class MultiScaleSemanticNet(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=mobilenet_v3_large(weights=None).features
        self.indices=(3,6,12,16)
        self.lateral=nn.ModuleList([nn.Conv2d(c,64,1,bias=False) for c in (24,40,112,960)])
        self.fuse=nn.ModuleList([nn.Sequential(nn.Conv2d(64,64,3,padding=1,bias=False),nn.GroupNorm(8,64),nn.ReLU()) for _ in range(4)])
        self.classifier=nn.Conv2d(64,40,1)
    def train(self,mode=True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        return self
    def forward(self,x):
        size=x.shape[-2:];features=[]
        for i,block in enumerate(self.encoder):
            x=block(x)
            if i in self.indices:features.append(x)
        out=None
        for j in (3,2,1,0):
            z=self.lateral[j](features[j])
            if out is not None:z=z+F.interpolate(out,size=z.shape[-2:],mode='bilinear',align_corners=False)
            out=self.fuse[j](z)
        return F.interpolate(self.classifier(out),size=size,mode='bilinear',align_corners=False)

class ResidualMotionNet(nn.Module):
    def __init__(self,history=11,future=20,hidden=64):
        super().__init__();self.history=history;self.future=future
        self.residual=nn.Sequential(nn.Flatten(),nn.Linear(history*2,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,future*2),nn.Unflatten(1,(future,2)))
        # Initial output is a strong causal least-squares CV baseline.
        nn.init.zeros_(self.residual[5].weight);nn.init.zeros_(self.residual[5].bias)
        past=torch.linspace(-1,0,history);centered=past-past.mean()
        self.register_buffer('ls_weights',centered/(centered.square().sum()))
        self.register_buffer('future_times',torch.arange(1,future+1,dtype=torch.float32)*.1)
    def forward(self,x):
        velocity=(x*self.ls_weights[None,:,None]).sum(1)
        baseline=x[:,-1:, :]+velocity[:,None,:]*self.future_times[None,:,None]
        return baseline+self.residual(x)

def residual_motion_network(history=11,future=20,hidden=64):return ResidualMotionNet(history,future,hidden)

def intersection_over_area(candidates,ignored):
    if len(ignored)==0:return candidates.new_zeros((len(candidates),))
    lt=torch.maximum(candidates[:,None,:2],ignored[None,:,:2]);rb=torch.minimum(candidates[:,None,2:],ignored[None,:,2:])
    intersection=(rb-lt).clamp(min=0).prod(2)
    area=(candidates[:,2:]-candidates[:,:2]).clamp(min=0).prod(1).clamp(min=1e-8)
    return (intersection/area[:,None]).amax(1)

def ignored_ssd_loss(self,targets,head_outputs,anchors,matched_idxs):
    box_pred=head_outputs['bbox_regression'];logits=head_outputs['cls_logits'];regression=[];classes=[];ignore_masks=[];foreground_count=0
    for target,bp,lp,anchor,match in zip(targets,box_pred,logits,anchors,matched_idxs):
        positive=match>=0;indices=torch.where(positive)[0];matched=match[positive];foreground_count+=len(indices)
        encoded=self.box_coder.encode_single(target['boxes'][matched],anchor[positive])
        regression.append(F.smooth_l1_loss(bp[positive],encoded,reduction='sum'))
        labels=torch.zeros(len(anchor),dtype=torch.int64,device=anchor.device)
        labels[positive]=target['labels'][matched];classes.append(labels)
        excluded=intersection_over_area(anchor,target.get('ignore_boxes',anchor.new_empty((0,4))))>.5
        ignore_masks.append(excluded & ~positive)
    classes=torch.stack(classes);excluded=torch.stack(ignore_masks);positive=classes>0
    loss=F.cross_entropy(logits.flatten(0,1),classes.flatten(),reduction='none').view_as(classes)
    eligible=~(positive|excluded);rank_loss=loss.masked_fill(~eligible,-torch.inf)
    order=rank_loss.argsort(1,descending=True);ranks=order.argsort(1)
    # Empty but valid backgrounds still teach the person head to reject objects.
    counts=(self.neg_to_pos_ratio*positive.sum(1,keepdim=True)).clamp(min=3)
    negative=(ranks<counts)&eligible
    denominator=max(1,foreground_count)
    return {'bbox_regression':torch.stack(regression).sum()/denominator,'classification':(loss[positive].sum()+loss[negative].sum())/denominator}

def enable_ignore_regions(model):
    if hasattr(model,'neg_to_pos_ratio'):
        model.compute_loss=types.MethodType(ignored_ssd_loss,model)
        original=model.transform.forward
        def transformed(images,targets=None):
            result,processed=original(images,targets)
            if processed is not None:
                # Each image has its own resized shape; no batch-padding scaling.
                for i,(image,target,new) in enumerate(zip(images,targets,processed)):
                    if 'ignore_boxes' in target:
                        h,w=image.shape[-2:];nh,nw=result.image_sizes[i]
                        new['ignore_boxes']=target['ignore_boxes']*target['ignore_boxes'].new_tensor([nw/w,nh/h,nw/w,nh/h])
            return result,processed
        model.transform.forward=transformed
    else:
        # RPN and RoI samplers both exclude ignored background candidates.
        rpn_original=model.rpn.assign_targets_to_anchors
        def assign_anchors(anchors,targets):
            labels,boxes=rpn_original(anchors,targets)
            for a,t,l in zip(anchors,targets,labels):
                ignore=intersection_over_area(a,t.get('ignore_boxes',a.new_empty((0,4))))>.5
                l[ignore & (l==0)]=-1
            return labels,boxes
        model.rpn.assign_targets_to_anchors=assign_anchors
        rpn_loss_original=model.rpn.compute_loss
        def rpn_loss(objectness,pred_bbox_deltas,labels,regression_targets):
            if not any(torch.any(l>=0) for l in labels):return objectness.sum()*0,pred_bbox_deltas.sum()*0
            return rpn_loss_original(objectness,pred_bbox_deltas,labels,regression_targets)
        model.rpn.compute_loss=rpn_loss
        roi_original=model.roi_heads.select_training_samples
        roi_assign=model.roi_heads.assign_targets_to_proposals
        def roi_labels(proposals,boxes,labels):
            matched,assigned=roi_assign(proposals,boxes,labels)
            for p,l,ignored in zip(proposals,assigned,model.roi_heads._quality_ignore_boxes):
                mask=intersection_over_area(p,ignored)>.5;l[mask&(l==0)]=-1
            return matched,assigned
        def select_samples(proposals,targets):
            model.roi_heads._quality_ignore_boxes=[t.get('ignore_boxes',p.new_empty((0,4))) for p,t in zip(proposals,targets)]
            result=roi_original(proposals,targets)
            model.roi_heads._quality_all_samples_ignored=all(len(labels)==0 for labels in result[2])
            return result
        model.roi_heads.assign_targets_to_proposals=roi_labels;model.roi_heads.select_training_samples=select_samples
        roi_forward_original=model.roi_heads.forward
        def roi_forward(features,proposals,image_shapes,targets=None):
            result,losses=roi_forward_original(features,proposals,image_shapes,targets)
            if model.roi_heads.training and model.roi_heads._quality_all_samples_ignored:
                # TorchVision CE on an empty sampled batch is NaN. With no eligible
                # supervision, return a connected zero rather than invent negatives.
                zero=sum(f.sum()*0 for f in features.values())
                losses={'loss_classifier':zero,'loss_box_reg':zero}
            return result,losses
        model.roi_heads.forward=roi_forward
        original=model.transform.forward
        def transformed(images,targets=None):
            result,processed=original(images,targets)
            if processed is not None:
                for i,(image,target,new) in enumerate(zip(images,targets,processed)):
                    if 'ignore_boxes' in target:
                        h,w=image.shape[-2:];nh,nw=result.image_sizes[i]
                        new['ignore_boxes']=target['ignore_boxes']*target['ignore_boxes'].new_tensor([nw/w,nh/h,nw/w,nh/h])
            return result,processed
        model.transform.forward=transformed
    return model

def detection_model(architecture,weights_path=None,num_classes=2):
    if architecture=='fasterrcnn_mobilenet_v3_large_320_fpn_quality_v2':
        model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=None,weights_backbone=None,num_classes=91 if weights_path else num_classes)
        if weights_path:
            model.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
            old=model.roi_heads.box_predictor;new=FastRCNNPredictor(old.cls_score.in_features,2)
            with torch.no_grad():
                new.cls_score.weight.copy_(old.cls_score.weight[:2]);new.cls_score.bias.copy_(old.cls_score.bias[:2])
                new.bbox_pred.weight.copy_(old.bbox_pred.weight[:8]);new.bbox_pred.bias.copy_(old.bbox_pred.bias[:8])
            model.roi_heads.box_predictor=new
    else:
        model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=91 if weights_path else num_classes)
        if weights_path:
            model.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True));head=model.head.classification_head
            for block in head.module_list:
                old=block[-1];count=old.out_channels//91;new=nn.Conv2d(old.in_channels,count*2,1)
                with torch.no_grad():
                    new.weight.copy_(old.weight.reshape(count,91,old.in_channels,1,1)[:,[0,1]].reshape_as(new.weight));new.bias.copy_(old.bias.reshape(count,91)[:,[0,1]].reshape_as(new.bias))
                block[-1]=new
            head.num_columns=2
    return enable_ignore_regions(model)
