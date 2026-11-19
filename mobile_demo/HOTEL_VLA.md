# Hotel bathroom, full Stanford TidyBot: three physical cubes

Run from the repository root after PR #32 has been merged:

```bash
python -m mobile_demo.tidybot_full --fetch
mjpython -m mobile_demo.hotel_vla --check-model
mjpython -m mobile_demo.hotel_vla --policy scripted --viewer --max-steps 3500 --require-success
# Viewer at http://127.0.0.1:8080
```

Scene: **one red 24 mm cube on the floor**, **two red 24 mm cubes at different
orientations on the hotel vanity**, an actual recessed sink, and a **blue open bin
on the countertop**. Everything is MuJoCo contact-physical, with three free
joints. The official TidyBot 2023 model has the 7-DoF Kinova Gen3, Robotiq
2F-85, wrist camera, real links and collision meshes. The native low base
camera cannot see the elevated counter, so an elevated front camera and
non-colliding mast were added. The planar mobile base remains servo-driven,
not physically powered wheels.

`scripted` uses **privileged object xyz** plus numerical site Jacobian IK, and
is an engineering baseline only. No cube is teleported or welded. It is a
first, uncalibrated IK implementation, and physical grasps, table clearance,
final placement, and general reliability are **not yet validated**. Failed
pickup, timeout, unreachable IK and incorrect drop must remain visible in the
JSON report, not hidden as successes.

## Real VLA inference

```bash
# A *locally fine-tuned* checkpoint for this EXACT 11D robot is required.
mjpython -m mobile_demo.hotel_vla --policy smolvla \
  --checkpoint /path/to/hotel-trained-tidybot-checkpoint \
  --device mps --viewer --require-success
```

The VLA receives a natural-language task, two rendered 224x224 RGB views
(`hotel_front` and `wrist`) and 11 motor/joint state values. It predicts 11
**absolute native actuator targets** at 15 Hz in this order:

`joint_x`, `joint_y`, `joint_th`, `joint_1` through `joint_7`,
`fingers_actuator` (last command **0 to 255**, not a joint angle).

The state contains the first ten joint positions, then
`right_driver_joint` angle. The 11-action and 11-state conventions must be
normalized **as trained**: the correct number of dimensions is *not enough*.
At the checkpoint root place `hotel_action_contract.json` containing:

```json
{
  "actuators": ["joint_x", "joint_y", "joint_th", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "fingers_actuator"],
  "state_joints": ["joint_x", "joint_y", "joint_th", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "right_driver_joint"],
  "cameras": ["camera1", "camera2"],
  "control_hz": 15,
  "action_convention": "absolute_native_targets_si_gripper_0_255"
}
```

The connector intentionally rejects a missing/incompatible checkpoint. In
particular **`ahmedsohail2003/smolvla-so101-pickplace-v2` is a fixed 6D arm
model**, so using it for TidyBot's mobile 11D actuation would produce invalid
robot actions. This PR implements a real inference *path* but **does not
include trained 11D TidyBot weights**, data collection or a proven successful
VLA rollout. You must train/port a compatible checkpoint to execute that
policy. `--check-model` is structural, not a demonstration of VLA success.

## Validation

```bash
python -m unittest tests.test_hotel_vla -v
mjpython -m mobile_demo.hotel_vla --check-model
mjpython -m mobile_demo.hotel_vla --policy scripted --max-steps 3500 \
  --output mobile_runs/hotel_vla.json --require-success
```

Success is defined only by the **measured simulated centers of all three cubes
inside the bin**. A failed smoke run must not be described as solved.
