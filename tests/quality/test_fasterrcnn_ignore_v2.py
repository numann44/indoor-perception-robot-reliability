import torch
from robot_model_sdk.quality_models import detection_model

def test_frcnn_rpn_and_roi_ignore_preserve_real_positives():
    torch.set_num_threads(2)
    model=detection_model('fasterrcnn_mobilenet_v3_large_320_fpn_quality_v2')
    anchors=[torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.],[40.,40.,50.,50.]])]
    targets=[{'boxes':anchors[0][:1].clone(),'labels':torch.tensor([1]),'ignore_boxes':torch.tensor([[0.,0.,31.,31.]])}]
    labels,_=model.rpn.assign_targets_to_anchors(anchors,targets)
    torch.testing.assert_close(labels[0],torch.tensor([1.,-1.,0.]))
    scores=torch.tensor([0.,50.,0.],requires_grad=True)
    loss,_=model.rpn.compute_loss(scores,torch.zeros((3,4),requires_grad=True),labels,[torch.zeros((3,4))])
    loss.backward();assert scores.grad[1]==0 and scores.grad[0]!=0 and scores.grad[2]!=0
    proposals,_,sampled_labels,_=model.roi_heads.select_training_samples(anchors,targets)
    assert not torch.any(torch.all(proposals[0]==anchors[0][1],dim=1))
    assert torch.any(sampled_labels[0]==1) and torch.any(sampled_labels[0]==0)
    model.train();image=torch.zeros((3,640,1280));_,scaled=model.transform([image],targets)
    torch.testing.assert_close(scaled[0]['ignore_boxes'],targets[0]['ignore_boxes']*.5)

def test_frcnn_all_crowd_batch_has_finite_connected_zero_losses():
    torch.set_num_threads(2)
    model=detection_model('fasterrcnn_mobilenet_v3_large_320_fpn_quality_v2').train()
    boxes=[torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])]
    targets=[{'boxes':torch.empty((0,4)),'labels':torch.empty((0,),dtype=torch.int64),'ignore_boxes':torch.tensor([[0.,0.,320.,320.]])}]
    labels,_=model.rpn.assign_targets_to_anchors(boxes,targets)
    assert torch.all(labels[0]==-1)
    scores=torch.ones(2,requires_grad=True);deltas=torch.ones((2,4),requires_grad=True)
    rpn=model.rpn.compute_loss(scores,deltas,labels,[torch.zeros((2,4))]);sum(rpn).backward()
    assert all(torch.isfinite(x) and x==0 for x in rpn) and torch.all(scores.grad==0)
    features={'0':torch.randn(1,256,20,20,requires_grad=True),'1':torch.randn(1,256,10,10,requires_grad=True)}
    _,losses=model.roi_heads(features,boxes,[(320,320)],targets)
    assert all(torch.isfinite(x) and x==0 for x in losses.values())
    sum(losses.values()).backward();assert all(torch.isfinite(f.grad).all() and torch.all(f.grad==0) for f in features.values())
