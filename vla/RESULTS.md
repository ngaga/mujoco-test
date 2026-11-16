# SO-101 pretrained SmolVLA: first independent MuJoCo result

**Date:** 2026-11-15. **Checkpoint:** [smolvla-so101-pickplace-v2](https://huggingface.co/ahmedsohail2003/smolvla-so101-pickplace-v2).

**Environment:** author's pinned `sim2cell PickPlaceEnv` MuJoCo SO-ARM100 scene, 6 joints, front + wrist RGB 224x224, 15Hz, 240-action cap, seed 123, red block -> blue tray. Published weights and LeRobot 0.6.0 pre-/postprocessors, **no EarlyBots training**.

## Results measured on GitHub Actions

| Evaluation | Success | Mean successful steps | Setup |
|---|---:|---:|---|
| EarlyBots, one-episode smoke | 1/1 | 103 | Ubuntu 24.04, CPU PyTorch 2.8, OSMesa |
| **EarlyBots, 20-episode independent evaluation** | **15/20 = 75%** | **107.9** | Ubuntu 24.04, CPU PyTorch 2.8, OSMesa |
| Original author's claim | 18/20 = 90% | Not independently checked | Author's training/eval environment |

The 20-episode run produced genuine model-generated six-joint commands applied to MuJoCo, then checked *block in tray and TCP retreated*. No scripted teacher was called, and no block poses were fed to the VLA.

**Failures (1-indexed):** episodes 8, 16, 17, 18, 20. Each reached the 240-action limit with the block outside the tray. All successful episodes completed in 84–168 action steps.

**Full artifact:** [GitHub Actions workflow #65](https://github.com/ngaga/mujoco-test/actions/runs/43766290378), artifact `so101-pretrained-vla-rollout`, file `so101_evaluation.json` with positions, step counts, wall time and verdict for all 20 runs.

One clear limitation is that these 20 cases are a small sample. The difference 15/20 versus 18/20 does not isolate a single cause: CPU vs author's GPU inference, dependency versions, stochastic seeds, or environment differences may play a role. We must not claim the checkpoint guarantees 90% in EarlyBots or that its SO-101 actions work on our custom XYZ palm robot.

## Practical development next steps

1. Inspect failed block XY configurations, VLA action traces and two-camera footage, with no retuning using the held-out evaluation set.
2. Record visual outputs of successful and failed VLA rollouts.
3. Try a smaller square soap bar **in the matching SO-101 workcell**, retaining real joint inputs, calibrated front/wrist cameras and actuator conventions. This may not work zero-shot; measure the failure modes.
4. Only then consider adapting the trained VLA from the SO-101 arm to a future articulated EarlyBots manipulator, with an explicit action-space mapping and independent safety controls.

The standard PR CI keeps a **single real pretrained VLA rollout** and its GIF as an integration smoke, rather than rerunning the longer 20-episode benchmark on every change.
