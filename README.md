# EarlyBots · MuJoCo bathroom demo

First interactive hotel-bathroom restocking demo using **MuJoCo 4 WebViewer**.

A soap bar starts outside a hotel amenities tray containing a toothbrush and toothpaste. Click **Ranger le savon** to start a scripted pick-and-place sequence. The gripper and soap move in the 3D viewer; reset and pause are available from a local control page. An optional RGB visual servo now corrects the approach in XY while preserving the default scripted demo.

**Important:** The default demo remains a scripted trajectory with a kinematic soap attachment. The optional `--physical-grasp` mode uses motorized jaws and real contacts, but **neither mode is autonomous or safety-certified**.

## Actual SO-101 MuJoCo VLA evaluation (pretrained; no training)

**New experiment:** `vla/run_so101.py` uses the author's *matching*
SO-ARM100 articulated robot, six genuine motor actuators, two RGB cameras,
and the published SmolVLA v2 checkpoint. It runs closed-loop predictions
on the real MuJoCo scene and scores a physically completed cube-in-tray
task. No motion-script actions and no fine-tuning are used.

- First verify the environment: `mjpython vla/run_so101.py --check-env`
- Then attempt a true learned pick/place:
  `mjpython vla/run_so101.py --episodes 1 --device mps`
- For detailed setup, pinned third-party scene and sample GIF:
  see `vla/README.md`.

This is the SO-101 reference task with a **red cube and blue tray**, not
yet EarlyBots' hotel soap. CI tests the scene, image/state/action interfaces, and attempts one
real pretrained-model rollout on Ubuntu CPU with the ~907MB checkpoint.
The uploaded JSON contains the observed outcome. This single attempt
is not evidence of an 18/20 success rate; see the report rather than
assuming the grasp succeeded.

## Pretrained VLA: SmolVLA pick-and-place (experimental)

To avoid developing a VLA ourselves, we selected a real pretrained
**SmolVLA 450M** pick-and-place model already fine-tuned for a MuJoCo
SO-ARM100/101 arm:

- Model: https://huggingface.co/ahmedsohail2003/smolvla-so101-pickplace-v2
- Download/probe script: `vla/fetch_smolvla.py`
- Mac installation and honest compatibility details: `vla/README.md`

This model was *not trained by EarlyBots*. Its author reports 18/20
pick-place successes in their own matching MuJoCo robot cell, not in ours.
We can **download** its existing checkpoint and load it with LeRobot without
any training. We cannot yet claim to execute its actions in EarlyBots' custom
floating palm: the learned 6-joint SO-101 action space and two-camera
observation space differ from our current simulation. A proper SO-101
reference scene and validated actuator mapping are the next step.

## First visual-conditioned behavioral cloning policy (not yet VLA)

Our first **learned action policy** is now trained from successful MuJoCo
teacher demonstrations. The teacher runs the existing scripted physical
grasp and saves the **real RGB camera image, detected soap XY, 20ms motor
commands** (palm XYZ and jaw open/close), final free-soap pose and success
status. A saved compressed dataset and model can be inspected and reused.

    mjpython imitation.py both --train-demos 24 --test-demos 12 \
      --seed 2026 --save-images --output-dir imitation_runs

The learner is a small supervised **time-knot linear policy**. Training
fits its weights by least squares from successful demonstrations. Its inputs
are the initial soap location extracted **from RGB pixels** and elapsed task
time; its outputs directly command the palm and both finger actuators. On
evaluation, **no scripted Motion controller supplies actions**, and no
ground-truth soap coordinates are used for control.

The same held-out positions, with a different random seed from the training
positions, are run twice: original scripted teacher and learned student.
MuJoCo free-soap position and pad contact data are used only to score real
physics grasp, lifting and placement. The full results are written to
`imitation_runs/evaluation.json`. The dataset and policy are saved as
`teacher_dataset.npz` and `learned_policy.npz`, with optional initial
RGB frames in `teacher_images/`. GitHub Ubuntu/Mesa CI trains the model,
tests 12 held-out soap positions and uploads all artifacts.

**Limitations:** this is trajectory *distillation*, not an end-to-end VLA.
A human-written color detector extracts the image feature. The learned
policy uses task time and an initially observed XY, not a continual stream
of RGB images; if the soap moves unexpectedly after grasp it does not
automatically replan. No language input, general object detection, tactile
feedback, force-limited real arm or safety certification is implied. We
must measure held-out physical success before considering richer models.

## Camera-guided robustness and controlled domain randomization

The new `rgb_benchmark.py` tests **real RGB localization**, not MuJoCo
pose truth, for the first pickup, visual servo corrections and any retry.
It uses the calibrated oblique camera and yellow-soap segmentation currently
used by the live bathroom demo. The simulated soap pose remains available
**only to the outcome checker and evaluation metrics**, never to the RGB
planner.

Run on macOS with a graphical desktop (MuJoCo offscreen CGL requires one):

    mjpython rgb_benchmark.py --trials 100 --stress-trials 24 --seed 2026 \
      --output rgb_robustness_report.json

On Linux with OSMesa installed:

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python rgb_benchmark.py \
      --trials 100 --stress-trials 24 --seed 2026

The first 100 scenarios use the **exact same** positions and 50 forced
drops as `robustness.py` (the ground-truth reference). Each scenario is
run twice: one attempt without recovery and up to two attempts with recovery.
The additional 24 out-of-distribution (OOD) trials cover four categories,
six cases each: yaw ±20°, mass 0.18–0.42 kg, ellipsoidal shape, and mixed
yaw/mass/ellipsoid. Half deliberately release the soap during transport.
The sizes remain 150 × 100 × 44 mm. Shape and density are set when MuJoCo
compiles the model, so mass/inertia/collisions are physical rather than
visual-only changes. None of the experimental stress variations are claimed
to represent all real soaps.

The report contains *all attempts*, initial and retry image estimates,
image counts, camera localization errors (measured against truth **for
reporting only**), collision/placement success, failures and totals by
variation category. Failures are not filtered or silently replaced with
ground-truth coordinates. An undetected soap counts as failure.

On GitHub Actions, the Linux/Mesa job runs the full RGB 100+24-scenario
benchmark and uploads `rgb_robustness_report.json`. macOS CI checks
non-rendering model, geometry and scenario regression tests because the
hosted macOS runner lacks a usable offscreen CGL context. The benchmark
remains a simulation, not a real hardware or safety claim.

## Reproducible physical robustness benchmark

Use the headless physical MuJoCo model to run **100 seeded initial soap
positions** with exactly the same cases **with** and **without** automatic
recovery. Fifty of the 100 scenarios deliberately open the jaws during
the first transport to stress failure handling.

    python robustness.py --trials 100 --seed 2026 --output robustness_report.json

The JSON report includes every initial pose, whether a fault was injected,
task outcome, attempts used, actual final soap pose, failure reason, and
paired aggregate success rates (clean cases and forced-drop cases separated).
For reproducibility, running the same seed and model gives the same scenarios.

The benchmark deliberately uses **MuJoCo ground-truth XY for approach and
retry planning in both policies**, and the soap starts in a conservative,
reachable rectangle (X: -0.32 to -0.22 m, Y: -0.045 to +0.055 m).
Randomization occurs **only before the simulation begins**. There is
no soap teleportation during handling or regrasp.

These numbers measure **physics + scripted control + recovery**, not visual
perception or generality over arbitrary soap shapes and rotations. RGB
camera robustness is a different experiment. A comparison of recovery
policies on identical trials should not be interpreted as a production
robot reliability guarantee.

GitHub CI runs all 100 paired positions on Ubuntu and a shorter 8-position
cross-platform check on macOS; both upload a detailed JSON report artifact.
The CI tests enforce physically verified outcomes and bounded attempts,
but do not impose an invented success-rate threshold. Failures remain in
the report to guide improvements.

## Angled camera for easier observation and recovery

The RGB camera is now **offset from the back of the counter and tilted
about 26 degrees** rather than pointing straight down. The same named
`overhead` camera feeds visual servoing and the right-hand preview.
The calibrated 3D camera matrix is used for image-to-world projection; the
camera's old top-down position is no longer assumed.

During a failed-grasp recovery the gripper first **retracts to its home
position**, clear of the view of the dropped soap. Only after both the
soap has settled and the gripper is parked do we request a new RGB observation.
In visual mode we **do not** silently use MuJoCo's soap coordinates if
the camera cannot detect it. Failure now distinguishes out-of-reach,
unstable and visually undetected cases.

Camera calibration is tested for both scenes. Ubuntu CI uses Mesa software
rendering for an actual RGB-camera recovery test with a deliberately dropped
soap, while macOS CI checks the model, grasp physics and WebViewer. A real
browser rendering check is still useful on your Mac.

## Automatic recovery after a failed physical grasp

In `--physical-grasp` mode the robot now makes **at most two attempts**. If
its grasp supervisor detects a lost soap, it stops moving toward the tray,
opens the jaws, and waits until the freely simulated soap is stationary on
the reachable part of the countertop. It then re-plans approach from the
**current palm pose** to the **new soap location**, without resetting the
MuJoCo scene or copying soap coordinates into the physics state.

- With `--visual-servo`, retry localization comes from the overhead RGB
  camera. No invisible ground-truth fallback if the camera cannot see it.
- Without `--visual-servo`, simulation ground-truth XY is used as an
  explicitly documented development fallback.
- A soap that falls off the counter, keeps moving, or cannot be detected
  results in a terminal failure rather than an unsafe invented grasp.
- The interface reports **Tentative 1/2**, **Récupération du savon** and
  **Tentative 2/2**. A new **Run / Restart** still resets the full scene.

To demonstrate an actual failure and automatic retry in the browser, run:

    mjpython demo.py --physical-grasp --inject-drop-once

The optional fault injector opens the jaws during the first transport.
It is **only for testing** and never runs without that flag. The recovery
acceptance test is headless:

    python tests/recovery_smoke.py

This milestone handles one supported, flat-sided soap on a reachable
countertop. It is not a general grasp planner, does not handle arbitrary
orientations, and is not proof of safe real-world autonomous recovery.

## Task outcome verification (physical grasp)

In `--physical-grasp` mode, the robot now tracks real MuJoCo evidence:
the soap must rise at least 8 cm while both jaws contact it, remain elevated
during transport, and end inside the tray. The status panel shows **Prise
confirmée**, **Résultat** and a reason when the operation fails.

When pickup is lost, motion stops and the jaws open instead of continuing a
fake successful pick-and-place. The recovery controller makes one additional
attempt only if the soap settles in the supported reachable region.

The supervisor reads MuJoCo internal contact and object position data as a
simulation oracle, *not* a learned vision detector. Real-world confirmation
must eventually use onboard sensors (vision, fingers, force estimates).

## Experimental contact-only physical grasp

In the same unified workspace you already use, opt into the physical scene:

    mjpython demo.py --physical-grasp --visual-servo --soap-xy -0.21 0.09

In this mode a **dynamic palm follows a scripted mocap reference** through a compliant weld. The two jaws have MuJoCo sliding joints, mass and position motors;
they squeeze the freely simulated soap using real collision/contact forces.
**No soap position or velocity is assigned during the grasp or transport.**
The pinch is released by reopening the motors. This test uses a flat-sided
rectangular soap of about 300 g and 2 ms integration steps. Curved soaps,
pose variations and real robot arm control are not yet validated.

To test the contact physics without a browser:

    mjpython demo.py --physical-grasp --smoke-test

The acceptance test must observe simultaneous contacts between the soap and
both pads during pickup, at least 8 cm of soap lift, completion of the
transport, and the soap released inside the tray. This distinguishes real
frictional pickup from the old scripted teleport. The default invocation
without `--physical-grasp` retains the previous working demonstration.
For a developer-only parameter sweep, run `python tests/physical_sweep.py`.

This is **not** yet an autonomous physical manipulation policy: it uses a
prescribed palm motion, a task-specific camera detector and scripted
open/close timing. It is not an assessment of real-world robot safety.

## Single-tab robot workspace (recommended)

Run `mjpython demo.py --visual-servo --soap-xy -0.21 0.09`.
The browser now opens **http://127.0.0.1:8091/**, which includes:

- The live MuJoCo Studio 3D WebViewer embedded in the main work area.
- Always-visible **Run / Restart**, **Pause / Resume**, and **Reset** buttons above it.
- The camera RGB preview, detection status, and progress in a side panel.

You no longer need to switch tabs to restart the scripted sequence. The 3D
WebViewer at http://127.0.0.1:8080/ remains accessible in a separate tab.
If the embedded viewer is in spectator mode, close the old standalone 8080
viewer or use its role indicator to transfer control.

The combined page is a local convenience wrapper around Studio's WebViewer,
not a replacement for Studio's built-in simulation toolbar. It embeds the
viewer as an iframe and may need a normal browser (not a restricted embedded
browser). If a browser refuses the iframe, use the **Open Studio separately**
link and the 8091 controls until the embedding issue is resolved.

## Control the robot directly from the 3D viewer

The standalone 3D viewer at **http://127.0.0.1:8080/** can include a floating
**EarlyBots | Robot commands** panel, with **Run / Restart**, **Reset**,
and **Pause / Resume**. These are the controls for our scripted soap
pick-and-place, and they work without switching to another tab.
Use **Run / Restart** to restart the whole sequence from its initial state,
**Reset** to return the robot and soap to their start positions, and
**Pause** to stop or resume.

Studio's **own** Run/Reset toolbar is a separate built-in simulation feature
and does not manage our custom sequence. Use the *EarlyBots* panel instead.
If the panel is hidden, restore it via Studio's normal window layout controls.

The alternative control panel at **http://127.0.0.1:8091/** remains available
for the annotated RGB camera feed and diagnostics. Starting `mjpython demo.py`
opens the unified workspace on 8091 by default.

## macOS quick start

Clone the repository with: git clone git@github.com:ngaga/mujoco-test.git

Then in the mujoco-test directory, run:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    mjpython demo.py

Open the **unified workspace** at http://127.0.0.1:8091/ (opens automatically), which embeds the 3D scene and the Run / Restart buttons in one page. The standalone 3D viewer at http://127.0.0.1:8080/ remains available.

To update, run git pull, then source .venv/bin/activate and mjpython demo.py. Quit with Ctrl+C.

## Optional RGB visual-servo experiment

Run with an overhead MuJoCo camera and a yellow-soap detector:

    mjpython demo.py --visual-servo

To prove the grasp approach is **not** memorizing the soap's initial XY, move
it to an alternative starting location:

    mjpython demo.py --visual-servo --soap-xy -0.21 0.09

The camera renders RGB observations every 10 simulation ticks during approach
and descent, and at a slower rate elsewhere. The same RGB feed is displayed
in the local **control panel** at http://127.0.0.1:8091/ with a green crosshair
marking the detected soap. It is also available as a single PNG at
http://127.0.0.1:8091/api/camera.png after the renderer starts. The vision module segments the yellow soap, uses camera intrinsics
and an assumed soap-top surface height to estimate world XY, and smoothly corrects the
approach trajectory using a damped XY controller with bounded velocity and
acceleration. This avoids abrupt pose jumps when the camera target changes. The control panel's /api/status includes `vision_detected`
and `observed_soap_xy`. If the soap becomes occluded, the last target is kept.

**Limitations:** Visual feedback currently corrects **XY approach only**;
it does not solve grasp physics. The gripper is still a mocap body and holding
still imposes the soap position kinematically. Transport and release are
scripted. There is no visual confirmation of a successful grip or placement.
The yellow color threshold and calibrated surface height are task-specific.
The offscreen OpenGL renderer may require platform-specific setup, and should
be checked on a real Mac before relying on it. Run without `--visual-servo` to
keep the original working demo.

This is an incremental perception/servo prototype, **not autonomous grasping**.

## Diagnostics

    python -m unittest discover -s tests -v
    mjpython demo.py --check-model
    mjpython demo.py --smoke-test

The smoke test runs one scripted motion cycle and checks the resulting soap position without a browser. Alternative ports: mjpython demo.py --viewer-port 8082 --controls-port 8092.

## Files

- demo.py: simulation, MuJoCo Studio WebViewer integration, local HTTP command server
- motion.py: motion state machine and observable pick retargeting
- vision.py: RGB soap detection and calibrated pixel-to-world projection
- camera_preview.py: annotated PNG camera preview
- viewer_controls.py: custom Run / Restart, Reset and Pause/Resume buttons embedded in Studio
- models/bathroom.xml: original kinematic scene
- models/bathroom_physical.xml: actuated two-jaw contact scene
- physical_grasp.py: physically driven grip and contact acceptance test
- grasp_supervisor.py: failure detection and verified physical outcome
- recovery.py: two-attempt non-resetting recovery and replan logic
- tests/recovery_smoke.py: inject one drop and verify second pickup
- robustness.py: seeded ground-truth baseline benchmark
- rgb_benchmark.py: paired RGB-only localization plus shape/mass/yaw study
- imitation.py: collect successful RGB teacher demonstrations, train a small action policy and evaluate held-out grasping
- static/index.html: self-contained command page
- tests/test_motion.py: unit tests for the script

MuJoCo WebViewer uses unauthenticated HTTP and WebSocket connections and may listen on the local network. Only run this development demo on trusted networks; do not expose the port publicly.

## Pull requests and CI

All future changes should go to a feature branch and an **open PR**, not directly to `main`.

The workflow `.github/workflows/ci.yml` runs on every PR targeting `main` and every push to `main`. It checks **Ubuntu x64** and **macOS 15 on Apple Silicon** using Python 3.12:

1. Python syntax and the unit/regression tests (including synthetic RGB vision)
2. Compilation of the real MuJoCo MJCF file (`--check-model`)
3. A full physics cycle (`--smoke-test`)
4. Startup of the headless WebViewer and a real HTTP request to run the scripted movement

The end-to-end test catches failures in the server/viewer simulation loop, rather than only testing the standalone `motion.py`.

**Protect `main` on GitHub:** Repository Settings → Branches → Add branch protection rule (or Settings → Rules → Rulesets). Pattern: `main`. Enable **Require a pull request before merging** and **Require status checks to pass before merging**. After the first CI run registers the check names, select both `MuJoCo smoke (ubuntu-24.04)` and `MuJoCo smoke (macos-15)`. Do not add the connector to the bypass list.

GitHub Actions does not replace a real visual review on the Mac. In particular, the tests do not validate the browser's 3D rendering or robotic safety.
