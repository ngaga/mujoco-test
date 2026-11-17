# EarlyBots mobile cube collection (MuJoCo research prototype)

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