"""End-to-end headless WebViewer/HTTP test. No browser or GPU required."""

import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
# Always connect to the local server directly, never through a CI runner proxy.
HTTP = build_opener(ProxyHandler({}))


def available_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_status(url):
    with HTTP.open(url + "/api/status", timeout=2) as response:
        return json.load(response)


def main():
    viewer_port, controls_port = available_port(), available_port()
    if viewer_port == controls_port:
        controls_port = available_port()
    python = shutil.which("mjpython") if sys.platform == "darwin" else sys.executable
    if python is None:
        raise RuntimeError("mjpython is required for the macOS simulation smoke test")
    command = [
        python, "-u", str(ROOT / "demo.py"),
        "--no-browser", "--viewer-port", str(viewer_port),
        "--controls-port", str(controls_port),
    ]
    # WebViewer runs headlessly. -u keeps crash output available on failure.
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    url = f"http://127.0.0.1:{controls_port}"
    try:
        # macOS CI runners can spend considerable time initializing Studio.
        deadline = time.monotonic() + (120 if sys.platform == "darwin" else 45)
        while True:
            if process.poll() is not None:
                raise AssertionError("WebViewer exited before serving requests")
            try:
                get_status(url)
                break
            except (URLError, ConnectionError, TimeoutError, OSError):
                if time.monotonic() >= deadline:
                    raise AssertionError("Control panel did not start")
                time.sleep(0.2)

        req = Request(
            url + "/api/command",
            data=b'{"action":"start"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with HTTP.open(req, timeout=3) as response:
            assert response.status == 202, response.status

        deadline = time.monotonic() + 55
        max_progress = 0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("WebViewer crashed during scripted motion")
            try:
                state = get_status(url)
            except (URLError, TimeoutError):
                time.sleep(0.15)
                continue
            max_progress = max(max_progress, state["progress"])
            if state["progress"] == 100 and state["soap_in_tray"]:
                print("OK: WebViewer loop, controls, and soap placement completed.")
                return
            time.sleep(0.2)
        raise AssertionError(f"WebViewer motion timeout; max_progress={max_progress}, state={state!r}")
    except Exception as exc:
        raise AssertionError(f"{exc}\nWebViewer output:\n{stop_process(process)}") from exc
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()


def stop_process(process):
    if process.poll() is None:
        process.terminate()
    try:
        out, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        out, _ = process.communicate(timeout=5)
    return out


if __name__ == "__main__":
    main()
