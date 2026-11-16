## Controlled geometry follow-up: yellow 24mm cube

The `vla/eval_yellow_cube.py` benchmark uses an object with **exactly the
same 24 × 24 × 24 mm box geometry, 20 g mass, friction, solver settings,
initial spawn distribution and orientation rule** as the original red cube.
The MuJoCo physics and SO-101 motor commands remain unchanged; only the
material changes to yellow and the pretrained VLA's prompt becomes
`Pick up the yellow soap and place it in the blue tray.`.

This is deliberately a **cube**, not a realistically proportioned soap bar.
It tests whether the performance drop seen with the 36 × 24 × 16 mm yellow
rectangle persists when the original geometry is restored. Changing color
and the language prompt together still prevents claiming a color-only effect.

From the repo root in `.venv-vla`:

```bash
mjpython vla/eval_yellow_cube.py --check-env
mjpython vla/eval_yellow_cube.py --episodes 20 --seed 123 --device mps
open vla_runs/yellow_cube_gifs/
```

Reports are `vla_runs/yellow_cube_report.md` and
`vla_runs/yellow_cube_evaluation.json`; actual two-camera MuJoCo GIFs of
the first success and failure are in `vla_runs/yellow_cube_gifs/`.
The `SO-101 yellow cube VLA transfer evaluation` GitHub workflow runs
20 independent pretrained model rollouts on Ubuntu CPU and uploads the
same report and authentic GIF recordings. This does **not** train or
fine-tune the model, and does not modify the original red-cube script.

Next: if geometry restoration improves success, instrument first contact
between the fingers and the box (gripper/object offset, contact IDs and
relative orientation) and test the rectangular soap with a local pose
correction. To isolate appearance from language, perform independent
red/yellow visual and red-block/yellow-soap instruction ablations.

---

## Zero-shot yellow rectangular mini-soap (EarlyBots experiment)

The source-of-truth runner is `vla/eval_soap.py`. This is a real MuJoCo
physics experiment driven by the pretrained SO-101 SmolVLA (no training,
no scripted teleoperation and no object pose in its policy input).

The original 24x24x24 mm red cube is replaced in a freshly compiled version
of the pinned sim2cell MJCF by a **36x24x16 mm, 20 g yellow rectangular mini-soap**.
The original finger friction and blue tray stay the same; front/wrist 224x224
RGB cameras, arm joints and action normalization remain unchanged. The new
language instruction is "Pick up the yellow soap and place it in the blue tray."
This is a mini-soap chosen to fit the SO-101 gripper, **not** a full-size
hotel bar. The original red-cube demonstration is unchanged.

Run from the repository root in the existing isolated `.venv-vla`:

```bash
source .venv-vla/bin/activate
mjpython vla/eval_soap.py --check-env
mjpython vla/eval_soap.py --episodes 20 --seed 123 --device mps
open vla_runs/soap_gifs/
```

Outputs: `vla_runs/soap_zero_shot.json` (every trial with final position),
`vla_runs/soap_report.md` (numeric summary), and several
`vla_runs/soap_gifs/episode_NN_success|failure.gif` recordings of
**both real MuJoCo cameras**. The first episode, first successful and first
failed episode are saved when available. Change to `--device cpu` if MPS
cannot execute a particular operator.

The dedicated `SO-101 yellow soap zero-shot VLA evaluation` GitHub Actions
workflow runs 20 CPU rollouts with the pinned pretrained checkpoint and uploads
the report and genuine RGB GIFs. We do not declare success until a real
rollout shows the soap physically deposited inside the tray and the arm retreats.
Even a positive result only establishes transfer to this small modified
SO-101 scene, not our floating-palm EarlyBots soap workstation.

---

For the independent EarlyBots result, see [RESULTS.md](RESULTS.md):
**15/20 physical pick-place successes** on Ubuntu CPU using the pretrained
model, versus the author's reported 18/20. All raw episode outcomes are
available in GitHub Actions workflow #65.

# Run the actual pretrained SmolVLA in MuJoCo SO-101

**This is the next stage after checkpoint downloading.** It uses the author's
**actual** SO-ARM100 pick-and-place environment, including a 5-DoF articulated
arm, sixth jaw actuator, 224×224 front RGB camera and wrist RGB camera.
SmolVLA reads both cameras, 6 motor joint positions and the English task
instruction, then sends six motor targets to MuJoCo every 1/15 second.

The purpose is to reproduce **one genuine pretrained VLA-controlled pickup**,
not merely load the weights. There is **no fine-tuning**. We are intentionally
using the original red cube/blue tray for this first proof, not yet the
yellow hotel soap or EarlyBots' floating XYZ palm.

## Reproduce on your Apple Silicon Mac

For a compact installation, run `bash vla/setup_so101.sh` once. This
bootstraps the pinned environment and LeRobot in `.venv-vla`. Alternatively,
perform the commands below manually.

From the EarlyBots repository root, set up a **separate** Python 3.12
environment (do not overwrite your current .venv for the hotel demo):

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv-vla
source .venv-vla/bin/activate
python -m pip install --upgrade pip

# Exact source version of the author's SO-ARM100 MuJoCo workcell and meshes
python -m pip install "git+https://github.com/ahmedsohail2003/workcell-partA-imitation.git@5822ac1dd508c92183fd8ecc99f91d6ec386549b"

# Same major/minor LeRobot version as the author used to train the checkpoint
python -m pip install "lerobot[smolvla]==0.6.0"
```

First check the real robot model and both cameras, **without downloading the
large model or running an AI policy**:

```bash
mjpython vla/run_so101.py --check-env
```

Then execute **one actual pretrained VLA-controlled simulation**:

```bash
mjpython vla/run_so101.py --episodes 1 --device mps \
  --gif vla_runs/so101_first_vla.gif
```

The default model ID is
[ahmedsohail2003/smolvla-so101-pickplace-v2](https://huggingface.co/ahmedsohail2003/smolvla-so101-pickplace-v2).
This downloads public pretrained weights once, if not already cached. If
you downloaded them with fetch_smolvla.py, use:

```bash
mjpython vla/run_so101.py --repo vla/checkpoints/smolvla-so101-pickplace-v2 \
  --episodes 1 --device mps --gif vla_runs/so101_first_vla.gif
```

If the current PyTorch/MPS combination rejects an operation, try
`--device cpu` (slower). The first command may download ~0.9 GB of
checkpoint weights plus its dependencies. No GPU training is necessary.

To compare with the author's evaluation (20 distinct spawns, seed 123,
240 steps at 15Hz, identical instruction, exact SO-101 motor conventions):

```bash
mjpython vla/run_so101.py --episodes 20 --seed 123 --device mps \
  --output vla_runs/so101_20episodes.json
```

The JSON report contains every rollout's actual MuJoCo success/steps,
initial and final block position and final TCP pose. Success is scored by the
environment's **real block-in-tray + gripper-retreated** condition, never by
the model's own text or by pretending its outputs are correct.

## What the tests establish

The EarlyBots PR CI includes a **one-episode real pretrained VLA evaluation**
on CPU in Ubuntu/Mesa. It downloads the actual checkpoint and uses the
original SO-101 environment, uploading a JSON artifact with success/failure.
This is a small integration trial (1 episode), not a statistically reliable
20-episode benchmark. The CI also verifies:
- Data contract: the model gets six joint states, two rendered RGB cameras
  and the exact task instruction, and returns six finite joint targets.
- A stub-policy closed loop actually sends commands to motor actuators.
- Ubuntu offscreen MuJoCo: installs the author's pinned complete scene
  and renders both cameras, checks six articulated actuators, and steps real
  physics with their control targets.

**An actual SmolVLA pick-place result must come from the GitHub JSON
artifact or an observed Mac rollout.** The 18/20 rate is the original
author's result, not an EarlyBots experiment. One episode is an integration
check, not a measured success rate.

## Next integration milestone

After the matching SO-101 workcell succeeds with the published checkpoint,
we can replace the cube with a *small, physically graspable* soap bar in
that same six-joint SO-101 action space, preserving the camera and
normalization assumptions initially. Our existing EarlyBots robot, which
uses a floating mocap palm, is NOT yet a compatible inference target.

---

# SmolVLA pretrained Pick & Place experiment

We will **reuse a downloaded pretrained VLA**, not train our own foundation
model. The first checkpoint is a smallish (~450M parameters, 907MB weights)
SmolVLA fine-tuned specifically for a MuJoCo SO-ARM100 / SO-101 arm.

**Model:** https://huggingface.co/ahmedsohail2003/smolvla-so101-pickplace-v2

**Author's reproducibility source:**
https://github.com/ahmedsohail2003/workcell-partB-vla

The author reports **18/20 successes** in their own matching SO-ARM100
MuJoCo pick-place work cell, with the English instruction:

> Pick up the red block and place it in the blue tray.

This is **not** an independent EarlyBots measurement; their base model
without fine-tuning scored 0/20 in that same evaluation. Treat it as an
interesting candidate, not evidence of zero-shot grasping of yellow soaps.

## Download on an Apple Silicon Mac

Keep this **separate** from the existing `.venv` using Python 3.14.
The LeRobot installation docs recommend Python 3.12.

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv-vla
source .venv-vla/bin/activate

python -m pip install --upgrade pip
python -m pip install "huggingface_hub>=0.28,<2"

# Inspect the model card and input / output features without downloading 907MB
python vla/fetch_smolvla.py --metadata

# Actually retrieve the trained weights and their normalization statistics
python vla/fetch_smolvla.py --download

# Optional next test: instantiate the real VLA via LeRobot
python -m pip install "lerobot[smolvla]"
python vla/fetch_smolvla.py --load
```

The model is cached under `vla/checkpoints/` and is ignored by Git.
No credentials are required for this public Apache-2.0 model.
The download requires ~1GB of disk space and internet access.
The `--load` operation **only loads** weights: no robot actions are
produced and no safety is claimed. An MPS-enabled Mac may support model
loading and inference, but we have not benchmarked its speed there.

## Why we cannot just plug the checkpoint into demo.py

Our current EarlyBots scene has one RGB camera, a dynamic **floating palm**
whose XYZ target is moved by a compliant weld, and two sliding jaws.
It lacks a real articulated robot arm and the joint-angle observations
the checkpoint expects.

The borrowed SO-101 VLA expects six joint/motor state dimensions plus two
camera views (front and wrist) with the same normalization used in training.
It emits six normalized motor/joint commands. Directly feeding those
numbers into our current XYZ palm would be geometrically invalid.

**The next integration step should therefore be an SO-ARM100 reference scene
with the trained camera, joint state and joint actuator conventions.** We
will first reproduce the author's pick-place case without fine-tuning,
then adapt to an EarlyBots soap scene using a compatible arm or a tested
action-space adapter. We should only consider transfer successful after
real MuJoCo physics pick-and-place rollouts, not just a load/inference probe.

The author's evaluation code currently contains a Windows-specific local
`sys.path` to its matching `sim2cell` work cell. It is not directly a
plug-and-play Mac/MuJoCo example. We should port the environment cleanly
rather than pretending it runs unmodified.

Reference code:
- https://github.com/ahmedsohail2003/workcell-partB-vla/blob/master/scripts/eval_smolvla.py
- https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100
- https://huggingface.co/docs/lerobot/installation
