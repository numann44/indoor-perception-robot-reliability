# Data, models and redistribution

Public repository scope: original source code, metadata, aggregate metrics/charts and original MuJoCo controlled-scene panels. No third-party research RGB/depth pixels, annotations, pretrained weights, trained checkpoints or bags are redistributed here.

| Research source used locally | Role | Original source |
|---|---|---|
| NYU Depth V2 | Real indoor RGB-D/NYU40 segmentation | https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html |
| JAAD | Person boxes/IDs; image-domain motion | https://github.com/ykotseruba/JAAD |
| MOT17-11-SDP | External shopping-mall detection/tracking | https://motchallenge.net/data/MOT17/ |
| OpenLORIS-Scene | Real robot-recorded RGB-D, calibration and wheel odometry | https://lifelong-robotic-vision.github.io/dataset/scene.html |
| TUM RGB-D | Historical real RGB-D geometry/replay | https://cvg.cit.tum.de/data/datasets/rgbd-dataset |
| Hypersim | Historical synthetic indoor semantics | https://github.com/apple-aiml-research/ml-hypersim |

NYU, MOT and OpenLORIS dataset redistribution permission was not established in the recorded source review; those images and derivatives remain local. JAAD video attribution differs from annotation/code terms. TUM CC BY and Hypersim CC BY-SA conditions require their own handling. No blanket dependency-code license is assigned to model weights or datasets. Original sources and terms should be checked before any future redistribution.

The demo is generated from project-authored primitive MuJoCo geometry. It shows saved selected-model overlays; control in these media chapters is geometric, so the video is not evidence of learned-control causality. The public excerpt strips absolute local command paths and excludes target masks/IDs and private logs. Full measured evidence remains local.
