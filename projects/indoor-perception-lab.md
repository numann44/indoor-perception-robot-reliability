# Indoor Perception Lab

Python/PyTorch RGB-D perception research: detection, NYU40 semantics, Kalman/Hungarian tracking, causal residual motion prediction and 10 cm observed-surface maps.

- [Implementation](../packages/indoor_perception_lab) and [shared contracts](../packages/robot_model_sdk).
- MobileNetV3 multiscale semantic decoder: NYU test mIoU 0.27309 versus V1 0.05163.
- Validation-selected Faster R-CNN MobileNetV3-320 detector: fresh JAAD AP 0.14814 versus COCO 0.05545; external MOT17 AP50 0.49139 (COCO is higher).
- Residual 2×64 MLP forecasting improves conditional JAAD 2 s ADE versus least-squares CV; eligible-history coverage is only 36.56%. Image and metre domains use separate checkpoints.
- RGB/depth timing and geometry, invalid-depth handling, pose admission and target-free inference contracts are explicit.

Tracking IDF1 and ID-switch acceptance fail. Maps built from saved hard labels are not softmax posterior reconstruction or SLAM. Physical/embedded operation is unverified.

[Full numerical context](../docs/RESULTS.md) · [Demo](../demo/README.md) · [Install/reproduce](../docs/REPRODUCIBILITY.md) · [Contribution disclosure](../docs/CONTRIBUTIONS.md)
