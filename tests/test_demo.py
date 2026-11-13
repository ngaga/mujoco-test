"""Regression tests that exercise the actual MuJoCo model and runtime APIs."""

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import mujoco

from demo import ROOT, State, handler_factory, soap_joint_addresses, smoke_test


class MuJoCoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "bathroom.xml"))

    def test_model_loads_and_soap_joint_has_integer_offsets(self):
        qadr, vadr = soap_joint_addresses(self.model)
        self.assertIsInstance(qadr, int)
        self.assertIsInstance(vadr, int)
        self.assertGreaterEqual(qadr, 0)
        self.assertGreaterEqual(vadr, 0)
        self.assertEqual(self.model.jnt_type[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "soap_free")], mujoco.mjtJoint.mjJNT_FREE)

    def test_full_physics_cycle_deposits_soap(self):
        # This would have caught the NumPy-index crash in the viewer loop.
        smoke_test(self.model)

    def test_studio_webviewer_modules_import(self):
        from mujoco.studio import app, launch_web, messages, plugin, sim, viewer_protocol
        self.assertTrue(callable(launch_web.launch))


class ControlPanelTests(unittest.TestCase):
    def test_http_panel_status_and_commands(self):
        state = State(8080)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(origin + "/api/status", timeout=3) as resp:
                status = json.load(resp)
            self.assertEqual(status["phase"], "Prêt")
            req = Request(origin + "/api/command", data=b'{"action":"start"}', method="POST", headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 202)
            self.assertEqual(state.commands.get_nowait(), "start")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
