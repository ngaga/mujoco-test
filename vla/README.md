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
