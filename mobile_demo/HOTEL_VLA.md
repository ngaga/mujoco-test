# Hotel bathroom, full Stanford TidyBot: three physical cubes

Run from the repository root after PR #32 has been merged:

```bash
python -m mobile_demo.tidybot_full --fetch
mjpython -m mobile_demo.hotel_vla --check-model
mjpython -m mobile_demo.hotel_vla --policy scripted --planner vertical --viewer --max-steps 3500
# Viewer at http://127.0.0.1:8080
```

Scene: **three red 24 mm cubes on the hotel vanity** with different orientations,
an actual recessed sink, and a **blue open bin on the countertop**. Everything is MuJoCo contact-physical, with three free
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
python -m unittest discover -s tests -p 'test_hotel_vla.py' -v
mjpython -m mobile_demo.hotel_vla --check-model
mjpython -m mobile_demo.hotel_vla --policy scripted --max-steps 3500 \
  --output mobile_runs/hotel_vla.json --require-success
```

Success is defined only if **all three cube centers are physically inside
the bin and their velocities are low**, after the controller has released
them. A failed smoke run must not be described as solved.

## Collision-checked hotel bin depositing (OMPL RRTConnect)

This follow-up adds two comparable modes to the same physical three-cube
scene. Both use **privileged** simulated object poses and a DLS IK teacher,
**not** a pretrained VLA. Existing SmolVLA inference action conventions
remain unchanged.

```bash
# OMPL Python wheel (not part of the previous install; supported platform required)
python -m pip install "ompl==2.0.1"

# Both modes produce detailed JSON measurements, no `--require-success`
mjpython -m mobile_demo.hotel_vla --policy scripted --planner vertical \
  --max-steps 3500 --output mobile_runs/hotel_vertical.json --viewer
mjpython -m mobile_demo.hotel_vla --policy scripted --planner ompl \
  --max-steps 3500 --planner-timeout 2.5 \
  --output mobile_runs/hotel_ompl.json --viewer

# Strict 3/3 benchmark; command exits nonzero for a failure:
mjpython -m mobile_demo.hotel_vla --policy scripted --planner ompl \
  --max-steps 3500 --no-hold-viewer --require-success
```

**What is planned:** seven actual Kinova joints (not merely a TCP A* search).
The controller first solves a downward-facing TCP IK pose at the center
of the bin, then checks the **entire arm plus a non-contact 24mm carried-cube
proxy** against MuJoCo hotel-bin walls, countertop, basin, cabinet and other
physical cubes. OMPL uses RRTConnect when a direct path is not collision-free.
Every returned segment is independently sampled at intervals of at most
0.045 rad/joint for another MuJoCo geometry-distance check. There is **no
fallback to an unchecked motion** when the planner fails.

The 7D planner holds the base at a fixed pose. The base still moves with
MuJoCo's existing position servos during navigation (not part of the OMPL
search). The pickup sequence is still scripted. The last place/retract
segments are divided into four Cartesian-height waypoints over the open bin
and certified against all arm/obstacle geometry, rather than letting the
joint solver take one large blind step through the rim.

**What gets measured:** the JSON report includes
`planner.mode`, `plans_succeeded`, `rrt_calls`, `rrt_solutions`,
`rejections`, `plan_methods`, `waypoints_generated`,
`bin_arm_contact_physics_steps`, `worst_bin_contact_penetration_m`,
`contact_examples`, and the real cube-in-bin + settled `success` result.
A collided arm in OMPL mode ends the rollout as an explicit failure.

`ompl==2.0.1` offers Python wheels for CPython 3.12 and macOS 15+ Apple
Silicon. Older macOS/Python installations may need a supported environment
or a locally built OMPL binding. `--planner vertical` remains available
without OMPL.

These clearances and ideal position-controlled base are **simulation-only**,
not proven safe around people. A successful plan alone does not count as
a successful grasp or physical cube release.


## Persistent recovery: all three cubes start on the vanity

The default scripted teacher now **keeps the episode running after failed grasps,
missed deposits, dropped cubes, unreachable IK, arm/bin contact interruptions
and RRTConnect planning rejections**. It reads the object's CURRENT MuJoCo
world pose on every new attempt, so a block knocked onto the floor stays on
the floor and can be approached there. We do not reset the scene or fake a
successful grasp. The helper `mobile_demo/hotel_recovery.py` records every
failed attempt and retries the same item once before rotating to another
undelivered cube. Delivered cubes are excluded from new picks.

Default `--retries -1` removes the old one-retry-per-cube cutoff, **but not
the global episode time limit**. `--max-steps 6000` (default) caps simulation
steps, and the report explicitly records missing cubes and a timeout instead
of claiming success. Use a finite number such as `--retries 8` if desired.
Some locations after a drop (e.g. under the vanity) may still be unreachable.

```bash
# Pull latest main after merge, with OMPL in the same Python environment
python -m pip install 'ompl==2.0.1'
python -m mobile_demo.tidybot_full --fetch
mjpython -m mobile_demo.hotel_vla --policy scripted --planner ompl \\
  --retries -1 --max-steps 6000 --viewer \\
  --output mobile_runs/hotel_retry.json

# Deliberately open the physical Robotiq during first transport, once:
mjpython -m mobile_demo.hotel_vla --policy scripted --planner ompl \\
  --inject-drop-once --retries -1 --max-steps 6000 \\
  --output mobile_runs/hotel_forced_drop.json
```

Read `recovery.retry_events`, `recovery.attempts_per_cube`,
`recovery.undelivered_cubes`, `injected_gripper_drop`, and `failure`
in the JSON. The actual physical cube-in-bin measurements still determine
`delivered` and `success`. The scripted teacher retains privileged poses;
this is **not** autonomous visual perception or a trained VLA.
