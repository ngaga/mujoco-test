# EarlyBots mobile cube collection (MuJoCo research prototype)

**New alternative: [SO-101 real articulated arm + moving gripper](SO101.md).** The original 2-DOF Cartesian arm remains the default; the SO-101 version uses 9 joint commands (3 base + 6 actual SO-101), pinned real mechanical assets, live TCP IK and a strict contact-physics delivery test. Run `python -m mobile_demo.so101 --check-model` after installing the extra requirements, then `python -m mobile_demo.so101 --cubes 1 --require-success`. This is a physical arm on a *still simplified* holonomic base, not full LeKiwi wheel physics and not mobile VLA inference.

**Camera prototype:** [RGB front + wrist pickup](RGB_PERCEPTION.md). Launch `mjpython -m mobile_demo.so101 --cubes 3 --perception rgb --viewer` to experiment with camera-derived local cube coordinates. The default `--perception oracle` remains the reference, and long-range navigation in RGB mode still uses simulator truth.


**Current status: physical scene + oracle teacher + prospective 7D VLA adapter.**
The `expert` is a privileged, deterministic controller. Its successful rollouts
are *not* VLA successes, autonomous vision, or proof of real-world robot safety.

The scene has a LeKiwi-*inspired* holonomic mobile base (three decorative omni
wheels, directly actuated planar X/Y/yaw joints), a simplified Cartesian 2-DOF
arm, two driven pinch jaws, 5 freely simulated red cubes and an open blue bin.
All cube motion after reset is due to actual MuJoCo contacts and gravity, not
teleportation, mocap object attachment or welds. This is **not** the authentic
LeKiwi SO-101 model, nor a validated omni-wheel contact simulation. For the
full LeKiwi MJCF see [Ekumen-OS/lekiwi](https://github.com/Ekumen-OS/lekiwi).

```bash
# In the repo root, with the existing MuJoCo 4 environment activated:
python -m mobile_demo.run --check-model
python -m mobile_demo.run --cubes 1 --smoke-test --max-steps 750
mjpython -m mobile_demo.run --cubes 5 --viewer
# Then open http://127.0.0.1:8080/ (MuJoCo Studio).
```

For headless Ubuntu rendering (`libosmesa6` required):

```bash
MUJOCO_GL=osmesa python -m mobile_demo.run --cubes 1 --smoke-test \
  --gif mobile_runs/demo.gif --output mobile_runs/result.json
```

For data collection (raw NumPy observations, NOT yet LeRobot dataset format):

```bash
mjpython -m mobile_demo.run --cubes 5 \
  --record mobile_runs/oracle_episodes.npz --output mobile_runs/result.json
```

## VLA interface

The policy observes `task` text, two **real rendered** RGB 224×224 cameras
(`front` and `wrist`), and proprioceptive `state` with seven joint values. It
must output *absolute SI target positions* for `base_x, base_y, base_yaw,
reach, lift, left_jaw, right_jaw` at 15 Hz. It never receives cube world poses.
A compatible, specially fine-tuned SmolVLA checkpoint can be used via:

```bash
mjpython -m mobile_demo.run --policy smolvla \
  --checkpoint /path/to/compatible-mobile-seven-dimensional-checkpoint --cubes 1
```

**No compatible trained checkpoint is provided by this PR.** The existing
`ahmedsohail2003/smolvla-so101-pickplace-v2` checkpoint has a fixed-arm action
space and cannot simply be reused for autonomous mobile manipulation. The
adapter checks dimensions and refuses to fake a 7D action by padding/truncation.
Future work: replace the Cartesian proxy and decorative wheels with Ekumen's
physical LeKiwi model, collect training episodes, convert them into a LeRobot
dataset, fine-tune the VLA, and compare held-out scene-level performance against
the privileged teacher. Validate all grasp/release failures honestly.

## Reporting

JSON logs explicitly include `policy`, `vla_inference_executed`,
`expert_uses_privileged_cube_poses`, `cubes_requested`, `cubes_in_bin`,
`expert_physical_lifts`, `failure` and the final object positions. The CI runs
one physically simulated cube attempt and uploads the actual GIF and JSON.

## Non-resetting recovery after a drop

In oracle `expert` mode, a failed lift, drop during travel or missed deposit
does not end the simulation on the first failure. The fingers open, the arm
physically retracts, and the teacher re-approaches the cube at its **current**
MuJoCo position. The scene is never reset between attempts. Default:
five additional retries per cube (six total attempts). After exhausting
attempts on a cube, it proceeds to the next and reports skipped objects.

```bash
mjpython -m mobile_demo.run --cubes 1 --viewer --max-retries 5
# Hold the final scene in the browser until Ctrl+C (default with --viewer).
# Add --no-hold-viewer to exit immediately instead.
mjpython -m mobile_demo.run --cubes 1 --inject-drop-once --max-retries 3 \
  --gif mobile_runs/retry.gif --output mobile_runs/retry.json
```

Reports now include `retries`, `retry_events`, `attempts_per_cube`,
`skipped_cubes`, and `failure`. The gripper can still lose the cube:
recovery behavior being tested does not imply successful placement.


## Small grip tuning experiment (2026-11-17)

The mobile physical gripper has a modest tune:
- Pad and finger-stop tangential friction **5.0 → 6.0** (MuJoCo dimensionless coefficients).
- Jaw actuator force limits **±15 → ±18 N**.
- Keep the original symmetric closing command **0.051 m** per jaw, corresponding to a 32 mm nominal free gap for the 48 mm cube. A first trial at 0.046 m dropped the cube earlier, so the width adjustment is not retained.

The jaw motor setting is still a **position target**, not a closed-loop grip-force controller.
Actual contact forces, cube compression and eventual delivery are not guaranteed
by these parameters alone. The existing recovery sequence and 7D expert/VLA
interface are unchanged. CI records end-to-end physical delivery separately
from syntax and mechanical regression tests.
