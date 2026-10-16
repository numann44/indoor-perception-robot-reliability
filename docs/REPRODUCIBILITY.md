# Reproducibility and source-only boundaries

## Small fixture path

Python 3.12+ is required. Create a project-local environment, then install the SDK and independent Bench:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install ./packages/robot_model_sdk ./packages/robot_reliability_bench
.venv/bin/robotbench replay-test --output local-output/replay
```

For the PyTorch perception package:

```sh
.venv/bin/python -m pip install ./packages/indoor_perception_lab
```

Installing third-party packages may download wheels. Package metadata declares dependencies; the retained Mac experiment locks under configs/ record the original platform versions/hashes. This is not an all-platform dependency lock promise.

## Full learned-model results

Datasets, cached wheels, checkpoints, ROS/Pixi runtime and original experiment outputs are not bundled. The original `./portfolio bootstrap --protocol quality-v2` is a **cached-workspace restore**, not a network bootstrap for this export; it expects caches and acquired-data evidence absent here. Do not run full corpus-dependent verification expecting the historical test count from this source-only repository.

Training/evaluation source and frozen configs are supplied for review. Reproducing learned results requires lawful acquisition from the original sources, reconstruction of split/selection guards and the required checkpoints or a fresh registered training experiment. The historical hashes and omissions are documented in evidence/. No fresh training or evaluation is silently claimed.

## Simulation and ROS2

The saved demo runs entirely from supplied controlled-simulation panels/video without model weights. Live robotics needs MuJoCo, project-local ROS2/RoboStack and ML processes; the source is included, the runtime is not. RGB-D replay, simulation feedback and candidate-operated physical hardware are separate scopes.

## Public-export installation check

On 2026-10-16 the SDK and independent Bench were built and installed non-editably in a fresh local Python 3.12 venv. `pip check` and the replay command returned exit code 0, and all seven fixture checks passed. Imports were confirmed from installed site-packages. The exact dependency versions and output hashes are retained in [the installation evidence](../evidence/public-fresh-install.json); they describe this fixture check, not replacements for the historical ML experiment locks. The Lab and full model/corpus/ROS suites were not freshly installed or rerun for this publication.
