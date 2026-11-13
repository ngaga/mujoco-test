# EarlyBots · MuJoCo bathroom demo

First interactive hotel-bathroom restocking demo using **MuJoCo 4 WebViewer**.

A soap bar starts outside a hotel amenities tray containing a toothbrush and toothpaste. Click **Ranger le savon** to start a scripted pick-and-place sequence. The gripper and soap move in the 3D viewer; reset and pause are available from a local control page.

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

## Diagnostics

    python -m unittest discover -s tests -v
    mjpython demo.py --check-model
    mjpython demo.py --smoke-test

The smoke test runs one scripted motion cycle and checks the resulting soap position without a browser. Alternative ports: mjpython demo.py --viewer-port 8082 --controls-port 8092.

## Files

- demo.py: simulation, MuJoCo Studio WebViewer integration, local HTTP command server
- motion.py: motion state machine
- models/bathroom.xml: scene definition
- static/index.html: self-contained command page
- tests/test_motion.py: unit tests for the script

MuJoCo WebViewer uses unauthenticated HTTP and WebSocket connections and may listen on the local network. Only run this development demo on trusted networks; do not expose the port publicly.