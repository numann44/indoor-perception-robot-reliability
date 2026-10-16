import numpy as np
import torch
from robot_model_sdk.preprocessing import quality_rgb,quality_logits_to_mask
from robot_model_sdk.quality_models import ignored_ssd_loss,intersection_over_area,residual_motion_network,MultiScaleSemanticNet
from robot_model_sdk import ControlCommandV1

def test_residual_initializes_to_causal_ls_cv():
    model=residual_motion_network();t=torch.linspace(-1,0,11);h=torch.stack([t*.2,t*-.3],1)[None]
    result=model(h);ft=torch.arange(1,21)*.1
    torch.testing.assert_close(result[0,:,0],ft*.2);torch.testing.assert_close(result[0,:,1],ft*-.3)
    result.sum().backward();assert model.residual[5].weight.grad.abs().sum()>0

def test_semantic_multiscale_shape_and_bn_freezing():
    torch.set_num_threads(2);m=MultiScaleSemanticNet();m.train()
    assert all(not p.training for p in m.encoder.modules() if isinstance(p,torch.nn.BatchNorm2d))
    assert m(torch.randn(2,3,64,96)).shape==(2,40,64,96)

def test_logit_interpolation_precedes_argmax():
    z=np.array([[[1.,1.],[-9.,-9.]],[[0.,0.],[1.,1.]]],np.float32)
    p=quality_logits_to_mask(z,4,4)
    assert p.shape==(4,4) and np.all(p[1]==1)

def test_preprocessing_rgb_order_and_float_dtype():
    x=np.full((8,8,3),[255,0,0],np.uint8);z=quality_rgb(x,(4,4))
    assert z.dtype==np.float32 and z.flags.c_contiguous and z[0].mean()>z[1].mean()

def test_ignore_ioa_and_positive_precedence():
    boxes=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])
    ignored=torch.tensor([[0.,0.,10.,10.]])
    torch.testing.assert_close(intersection_over_area(boxes,ignored),torch.tensor([1.,0.]))
    from torchvision.models.detection._utils import BoxCoder
    class Dummy:box_coder=BoxCoder((10.,10.,5.,5.));neg_to_pos_ratio=3
    # Positive0, ignored negative1, clean negative2. Mask1 changes must not change loss.
    anchors=[torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.],[40.,40.,50.,50.]])]
    targets=[{'boxes':anchors[0][:1].clone(),'labels':torch.tensor([1]),'ignore_boxes':anchors[0][:2].clone()}]
    match=[torch.tensor([0,-1,-1])];box=torch.zeros((1,3,4),requires_grad=True)
    logits=torch.tensor([[[0.,1.],[0.,50.],[1.,0.]]],requires_grad=True)
    loss=ignored_ssd_loss(Dummy(),targets,{'bbox_regression':box,'cls_logits':logits},anchors,match)
    loss['classification'].backward();assert torch.all(logits.grad[0,1]==0) and logits.grad[0,0].abs().sum()>0
    assert torch.isfinite(loss['classification'])

def test_control_stop_validation():
    c=ControlCommandV1(0,0,0,'stopped','missing_depth',safety_stop=True);assert c.to_json()['schema_version']==1
    import pytest
    with pytest.raises(ValueError):ControlCommandV1(0,.1,0,'stopped','missing_depth',safety_stop=True)

def test_empty_image_has_background_loss_but_crowd_has_none():
    from torchvision.models.detection._utils import BoxCoder
    class Dummy:box_coder=BoxCoder((10.,10.,5.,5.));neg_to_pos_ratio=3
    anchors=[torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])]
    target={'boxes':torch.empty((0,4)),'labels':torch.empty((0,),dtype=torch.int64),'ignore_boxes':torch.empty((0,4))}
    logits=torch.tensor([[[0.,3.],[0.,2.]]],requires_grad=True);heads={'bbox_regression':torch.zeros((1,2,4),requires_grad=True),'cls_logits':logits}
    loss=ignored_ssd_loss(Dummy(),[target],heads,anchors,[torch.tensor([-1,-1])])['classification']
    assert loss>0;loss.backward();assert logits.grad.abs().sum()>0
    target['ignore_boxes']=torch.tensor([[-1.,-1.,31.,31.]])
    masked=ignored_ssd_loss(Dummy(),[target],heads,anchors,[torch.tensor([-1,-1])])['classification'];assert masked==0
