# Mobile SO-101: first camera-driven grasp experiment

This is the next measured step toward an autonomous indoor mobile robot, NOT a
pretrained VLA or fully sensor-autonomous robot yet.

The existing (default) \`--perception oracle\` mode uses exact MuJoCo cube positions
throughout its scripted policy. The new \`--perception rgb\` mode:
- **still uses privileged cube positions for long-distance navigation** to
  place the base near each desired cube, for post-lift success detection and for
  evaluation; the bin center is a known landmark
- **uses only rendered front and wrist camera RGB pixels for local X/Y
  targeting while hovering and descending**, never the cube's ground-truth
  position for that target. A deterministic red connected-components detector
  finds separate red objects, and calibrated camera rays intersect an assumed
  tabletop-height plane at 18 mm. Expected grasp zone is relative to the
  commanded mobile base pose, and an image target must fall within 18 cm
- takes a valid front estimate by default; when both front and wrist detections
  agree within 25 mm, averages them; uses wrist alone if front misses
- decides whether to **close the jaws** from measured TCP versus the image-derived XY, never from ground-truth cube XY. Ground-truth position is used only for evaluation and for other explicitly privileged phases
- never silently falls back to oracle XYZ in local RGB grasp mode; after
  repeated failed detections the normal bounded recovery system triggers

This uses *real camera renders from MuJoCo*, not fake red XY coordinates, not
a language-trained model, and not YOLO. The camera calibration and tabletop
height are currently exact simulated measurements.

## Commands

Install physical arm dependencies and Mesa/OpenGL support first:

    python -m pip install -r requirements.txt -r mobile_demo/so101_requirements.txt

Use the reference oracle first:

    mjpython -m mobile_demo.so101 --perception oracle --cubes 3 --max-steps 4000 --viewer

Try the front+wrist camera driven local grasp:

    mjpython -m mobile_demo.so101 --perception rgb --cubes 3 --max-steps 4000 --viewer

Both print JSON with delivery, retries and timing. In RGB mode the report
includes the number of RGB frames actually rendered, how many returned
detections per camera, and average/max position error against the hidden
MuJoCo truth **only for offline evaluation**, never for grasp control.

## Two simulator bugs corrected

**Cube count:** previously five cube bodies were always instantiated and
rendered, even if the scripted policy was instructed to handle fewer.
Now \`--cubes N\` compiles only the first N bodies. It is not merely a hidden
visual toggle; inactive objects have no collision or mass in the simulation.

**Visible wheel colliding with the bin:** the base chassis had a collision
radius of 0.17 m, while the forward decorative omniwheel projected to 0.225 m.
The old nominal bin approach at X=0.56 m could therefore place the *visible*
wheel at X=0.785 m, over the bin's left wall X=0.75 m, even while the chassis
contact probe reported zero collisions. The new planner clamps the
base-centre goal to X<=0.50 m, leaving a nominal 25 mm wheel margin. We
also measure conservative visible-wheel-to-bin clearance on every MuJoCo
integration step, separate from actual MuJoCo robot/bin contact geometry.

Both modes and the baseline remain supported. The vision experiment is
evaluated alongside the oracle on identical deterministic scenes; robust
generalization is not claimed from a single successful run. Next stages:
self-localization from wheel odometry + LiDAR, RGB detection without oracle
long-range navigation, then a pretrained/fine-tuned VLA when data and
distribution are compatible.

## Experimental RGB cube navigation

Opt-in mode `--perception rgb-nav` finds red cubes directly in rendered front-camera pixels,
projects them onto the floor using known camera calibration, and drives the existing
planar base to an approach standoff of 39 cm. If no cube is seen, it follows
fixed room patrol waypoints, never prerecorded cube coordinates. The close-range
front/wrist RGB grasp stage is unchanged.

    mjpython -m mobile_demo.so101 --perception rgb-nav --cubes 1 --max-steps 2000 --viewer
    mjpython -m mobile_demo.so101 --perception rgb-nav --cubes 3 --max-steps 5500 --viewer

**Limitations and non-oracle boundary:**
- A privileged task supervisor associates an already-selected visual target
  with a cube ID **only after** the camera-based approach. It still checks
  grasp/transport outcomes, retries and bin occupancy with simulator truth.
  These cube IDs or poses never choose any driving XY command.
- Base XY comes from idealized simulator slide joints, not real wheel odometry,
  LiDAR, SLAM or omniwheel physics.
- Camera extrinsics and tabletop elevation are exact simulator parameters;
  red segmentation is not robust to changes of color or lighting.
- The bin landmark and room patrol points are predefined, not image-mapped.
- The strict CI gate now covers both 1-cube and 3-cube deterministic missions;
  passing them does **not** prove robustness across new poses or lighting.
- The arm grasp policy is classical IK/scripted control, **not a mobile VLA**.

Regression tests make cube_position() and in_bin() raise while executing an RGB
navigation step; see tests/test_rgb_navigation.py and the new CI workflow.
