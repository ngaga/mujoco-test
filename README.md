# EarlyBots · MuJoCo bathroom demo

First interactive hotel-bathroom restocking demo using **MuJoCo 4 WebViewer**.

A soap bar starts outside a hotel amenities tray containing a toothbrush and toothpaste. Click **Ranger le savon** to start a scripted pick-and-place sequence. The gripper and soap move in the 3D viewer; reset and pause are available from a local control page. An optional RGB visual servo now corrects the approach in XY while preserving the default scripted demo.

**Important:** This prototype uses a scripted trajectory and kinematic object attachment. It does **not** yet implement autonomous grasping, robot arm dynamics, visual perception, or any safety-critical control.

## macOS quick start

Clone the repository with: git clone git@github.com:ngaga/mujoco-test.git

Then in the mujoco-test directory, run:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    mjpython demo.py

Open the **3D viewer** at http://127.0.0.1:8080/ and the **control panel** at http://127.0.0.1:8091/ (the latter should open automatically). Click **Ranger le savon** and watch it move into the tray.

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
- models/bathroom.xml: scene definition
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
