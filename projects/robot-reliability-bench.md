# Robot ML Deployment & Reliability Bench

An independently installable Python benchmark application consuming the small model SDK; Indoor Perception Lab is not required for fixture/model adapter execution.

- [Implementation](../packages/robot_reliability_bench), [model SDK](../packages/robot_model_sdk), [simulation/ROS orchestration](../tools/robotics_v2.py).
- Bounded two-frame inference queue with individually counted evictions and duplicate/out-of-order admission guards.
- Actual recorded DDS replay: 261 OpenLORIS RGB-D inputs received, 133 results and 128 oldest drops; offline rendering processed all 261.
- Actual selected-neural ROS2→MuJoCo fault loop: 22/22 checks, goal progress 3.8786 m and zero every-5-ms contacts. Observed zero-command delays: depth loss 94.60 ms; inference DDS exit 274.46 ms; consumer timeout 364.98 ms. These are different faults, not a physical braking bound.
- 120 matched geometric/active simulation episodes: 57/60 and 59/60 goals, zero contacts. Primary occlusion fails 7/10; secondary 9/10 does not establish whole-delta semantic causality.
- Mac CPU/MPS and experimental ONNX profiles; separate cold start/RSS, corruption accuracy and source/model/split hash binding.

The 600 s and separate 5 Hz ROS frontdoors each retain failed 12/13 statuses; complementary saved-event chronology establishes scoped pressure/ordering evidence. Strict export/INT8 deployment acceptance fails and native remains default. Embedded power/temperature and physical robot validation are unavailable.

[Offline fixture](../docs/REPRODUCIBILITY.md) · [Demo](../demo/README.md) · [Results and limitations](../docs/RESULTS.md)
