"""EarlyBots MuJoCo 4 WebViewer: scripted bathroom restocking demo."""
import argparse
import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
from motion import HOME, Motion, SOAP_OFFSET

ROOT = Path(__file__).resolve().parent


class State:
    """HTTP handlers only access thread-safe state and queue, not MjData."""

    def __init__(self, port):
        self.lock = threading.Lock()
        self.commands = queue.Queue()
        self.snapshot = dict(
            phase="Prêt", running=False, holding=False, paused=False,
            progress=0, soap_in_tray=False, viewer_url=f"http://127.0.0.1:{port}/"
        )

    def read(self):
        with self.lock:
            return self.snapshot.copy()

    def publish(self, **kwargs):
        with self.lock:
            self.snapshot.update(kwargs)


def handler_factory(state):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.respond(200, (ROOT / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self.respond(200, json.dumps(state.read()).encode(), "application/json")
            else:
                self.respond(404, b"Not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/command":
                self.respond(404, b"Not found", "text/plain")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 1024:
                    raise ValueError("Invalid length")
                cmd = json.loads(self.rfile.read(size))["action"]
                if cmd not in ("start", "pause", "reset"):
                    raise ValueError("Invalid action")
            except (ValueError, KeyError, TypeError):
                self.respond(400, b'{"error":"Bad command"}', "application/json")
                return
            state.commands.put(cmd)
            self.respond(202, b'{"queued":true}', "application/json")

        def log_message(self, *_args):
            pass

    return Handler


def reset(model, data, motion):
    mujoco.mj_resetData(model, data)
    motion.reset()
    data.mocap_pos[0] = HOME
    mujoco.mj_forward(model, data)


def step(model, data, motion, qadr, vadr):
    motion.advance(model.opt.timestep)
    data.mocap_pos[0] = motion.position
    if motion.holding:
        data.qpos[qadr:qadr + 3] = [
            a + b for a, b in zip(motion.position, SOAP_OFFSET)
        ]
        data.qvel[vadr:vadr + 6] = 0
    mujoco.mj_step(model, data)


def soap_in_tray(data, adr, holding):
    x, y, z = data.qpos[adr:adr + 3]
    return bool(not holding and 0.075 < x < 0.445 and -0.12 < y < 0.22 and 0.82 < z < 0.92)


def smoke_test(model):
    data = mujoco.MjData(model)
    motion = Motion()
    joint = model.joint("soap_free")
    reset(model, data, motion)
    motion.start()
    for _ in range(1500):
        step(model, data, motion, joint.qposadr, joint.dofadr)
    if not motion.finished or not soap_in_tray(data, joint.qposadr, motion.holding):
        raise RuntimeError("Smoke test: soap not in tray")
    print("OK: scripted action ended, soap is in tray.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--controls-port", type=int, default=8091)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "bathroom.xml"))
    print(f"Loaded: {model.nbody} bodies, {model.ngeom} geoms")
    if args.check_model:
        print("OK: MJCF compiled successfully.")
        return
    if args.smoke_test:
        smoke_test(model)
        return
    if args.viewer_port == args.controls_port:
        parser.error("Control and viewer ports must be different")

    from mujoco.studio import app as studio_app
    from mujoco.studio import launch_web, messages, plugin, sim, viewer_protocol

    class ViewerControls(plugin.SimPlugin):
        def __init__(self):
            self.paused = False
            self.speed = 100.0

        @messages.handler
        def on_step_control(self, snapshot: messages.StepControlSnapshot):
            self.paused = snapshot.pause_state != sim.PauseState.UNPAUSED
            self.speed = snapshot.speed

    original_model = model
    data = mujoco.MjData(model)
    motion = Motion()
    joint = model.joint("soap_free")
    qadr, vadr = joint.qposadr, joint.dofadr
    reset(model, data, motion)
    state = State(args.viewer_port)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.controls_port), handler_factory(state))
    except OSError as exc:
        parser.error(f"Cannot start command server on port {args.controls_port}: {exc}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    controls = ViewerControls()
    cfg = viewer_protocol.ViewerConfig(title="EarlyBots | Bathroom Demo", http_port=args.viewer_port)
    print(f"WebViewer 3D: http://127.0.0.1:{args.viewer_port}/")
    print(f"Control panel: http://127.0.0.1:{args.controls_port}/")
    print("Ctrl+C to stop. Demo uses a scripted grasp, not an autonomous one.")
    panel_paused = False
    try:
        with launch_web.launch(cfg, plugins=[studio_app.App(), controls]) as viewer:
            if not args.no_browser:
                webbrowser.open(f"http://127.0.0.1:{args.controls_port}/")
            while viewer.is_running():
                tick = time.perf_counter()
                model, data = viewer.sync(model, data)
                if model is None or data is None:
                    continue
                own = model is original_model
                while True:
                    try:
                        cmd = state.commands.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "pause":
                        panel_paused = not panel_paused
                    elif own and cmd == "reset":
                        reset(model, data, motion)
                        panel_paused = False
                    elif own and cmd == "start":
                        reset(model, data, motion)
                        panel_paused = False
                        motion.start()
                paused = panel_paused or controls.paused
                if not paused:
                    if own:
                        step(model, data, motion, qadr, vadr)
                    else:
                        mujoco.mj_step(model, data)
                state.publish(
                    phase=motion.label if own else "Autre modèle chargé : relancer demo.py",
                    running=motion.running if own else False,
                    holding=motion.holding if own else False,
                    paused=paused,
                    progress=motion.progress if own else 0,
                    soap_in_tray=soap_in_tray(data, qadr, motion.holding) if own else False,
                )
                period = model.opt.timestep * 100.0 / max(1.0, controls.speed)
                time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
