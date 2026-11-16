"""Offline, deterministic tests for passive contact and gripper-frame diagnostics."""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VLA = ROOT / "vla"
sys.path.insert(0, str(VLA))
spec = importlib.util.spec_from_file_location("diagnose_contacts", VLA / "diagnose_contacts.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class MockGeom:
    def __init__(self, geom_id):
        self.id = geom_id


class FakeModel:
    def __init__(self):
        self.names = list(diagnostics.PAD_GROUPS["fixed"]) + list(diagnostics.PAD_GROUPS["moving"]) + ["block_geom"]
        self.ids = {name: i for i, name in enumerate(self.names)}
        self.geom_size = np.zeros((len(self.names), 3))
        self.geom_size[self.ids["block_geom"]] = (0.012, 0.012, 0.012)
        self.site_name = "tcp"
        self.geom_bodyid = [1] * 8 + [2]
        self.body_parentid = [0, 0, 0]

    def geom(self, name):
        return MockGeom(self.ids[name])

    def body(self, name):
        return MockGeom(1)

    def site(self, name):
        return MockGeom(0)


class FakeData:
    def __init__(self, model):
        self.points = np.zeros((len(model.names), 3))
        for side, x, y in (("fixed", -0.03, 0.0), ("moving", 0.03, 0.0)):
            for i, name in enumerate(diagnostics.PAD_GROUPS[side]):
                self.points[model.ids[name]] = [x, 0.02 - 0.01*i, 0.012]
        self.points[model.ids["block_geom"]] = [0, 0, 0.012]
        self.geom_xpos = self.points
        self.geom_xmat = np.tile(np.eye(3).reshape(1, 9), (len(model.names), 1))

    def geom(self, geom_id):
        return Mock(xpos=self.geom_xpos[geom_id])

    def site(self, geom_id):
        return Mock(xpos=np.array([0.0, 0.0, 0.08]))


class GraspDiagnosticTests(unittest.TestCase):
    def test_fixed_to_moving_axis_and_signed_clearance(self):
        model = FakeModel()
        data = FakeData(model)
        m = diagnostics.inspect_alignment(model, data)
        self.assertTrue(np.allclose(m["world"]["closing_axis_xyz"], [1, 0, 0]))
        g = m["gripper_frame_mm"]
        self.assertAlmostEqual(g["fixed_pad_to_object_near_face_clearance"], 18)
        self.assertAlmostEqual(g["moving_pad_beyond_far_face_clearance"], 18)
        self.assertAlmostEqual(g["object_from_fixed_perpendicular"], -5)
        self.assertAlmostEqual(g["object_ahead_of_fixed_tip"], -20)
        self.assertEqual(m["object_tilt_deg"], 0)

    def test_lateral_sign_follows_gripper_not_world_x_y(self):
        model = FakeModel()
        data = FakeData(model)
        # Rotate all finger pads and box 90 degrees about world Z: the
        # object-to-jaw lateral measurement MUST remain the same.
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        data.points[:] = data.points @ rot.T
        data.geom_xmat[model.ids["block_geom"]] = rot.reshape(9)
        m = diagnostics.inspect_alignment(model, data)
        g = m["gripper_frame_mm"]
        self.assertAlmostEqual(g["fixed_pad_to_object_near_face_clearance"], 18)
        self.assertAlmostEqual(g["object_from_fixed_perpendicular"], -5)
        self.assertAlmostEqual(g["object_ahead_of_fixed_tip"], -20)

    def test_no_contact_is_not_reported_as_alignment_failure(self):
        m = FakeModel()
        d = FakeData(m)
        recorder = diagnostics.ContactRecorder(m, d)
        self.assertEqual(recorder.totals["fixed_pad"], 0)
        recorder.reset()
        self.assertIsNone(recorder.first_robot_contact)
        self.assertEqual(diagnostics.classify_contact(
            {"first_robot_contact": None}), "no_contact_with_robot")

    def test_markdown_has_explicit_signs_and_first_impact(self):
        report = {
            "object": "cube", "checkpoint": "mock", "episodes": 1,
            "successes": 0,
            "results": [{
                "episode": 1, "success": False, "gif": "episode_01.gif",
                "first_robot_contact": {
                    "contact_kind": "fixed_pad", "robot_geom": "fixed_jaw_pad_1",
                    "alignment": {
                        "gripper_frame_mm": {
                            "fixed_pad_to_object_near_face_clearance": -3.0,
                            "object_from_fixed_perpendicular": 5.0,
                            "object_ahead_of_fixed_tip": -8.0,
                        },
                    },
                },
            }],
        }
        text = diagnostics.report_markdown(report)
        self.assertIn("fixed_jaw_pad_1", text)
        self.assertIn("-3.0", text)
        self.assertIn("lateral", text.lower())
        self.assertIn("[GIF](episode_01.gif)", text)

    def test_cli_rejects_zero_episodes(self):
        args = diagnostics.parse_args(["--object", "rectangle", "--episodes", "3"])
        self.assertEqual(args.object, "rectangle")
        with self.assertRaises(SystemExit):
            diagnostics.parse_args(["--episodes", "0"])


if __name__ == "__main__":
    unittest.main()
