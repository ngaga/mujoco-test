"""Unit tests for the first supervised visual-conditioned MuJoCo policy.

No OpenGL is used here; the actual RGB/physical end-to-end training runs in
the Linux/Mesa CI job, where we measure held-out task success.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import mujoco

from imitation import (
    ImitationPolicy, REFERENCE_SOAP_XY, distinct_positions,
    fit_behavior_cloning, learned_run,
)
from robustness import MODEL_PATH


class BehavioralCloningTests(unittest.TestCase):
    def test_deterministic_disjoint_seeded_placements(self):
        train = distinct_positions(24, 2026)
        holdout = distinct_positions(12, 102026)
        self.assertEqual(train, distinct_positions(24, 2026))
        self.assertFalse(set(train) & set(holdout))
        self.assertEqual(len(train), 24)
        with self.assertRaises(ValueError):
            distinct_positions(0, 2026)

    def test_fit_observation_conditioned_teacher_actions(self):
        # Synthetic teacher actions linear in image-derived x and y.
        # This tests the *learner* independent of preprogrammed Motion phases.
        rng = np.random.default_rng(71)
        visual_xy = rng.uniform(
            [-0.32, -0.045], [-0.22, 0.055], size=(16, 2),
        )
        times = np.array([0.02, 0.04, 0.06, 0.08])
        weights = rng.normal(size=(len(times), 3, 4))
        offsets = visual_xy - REFERENCE_SOAP_XY
        features = np.column_stack([np.ones(len(offsets)), offsets])
        actions = np.stack([features @ weight for weight in weights], axis=1)
        policy = fit_behavior_cloning(actions, times, visual_xy)
        self.assertTrue(np.allclose(policy.weights, weights, atol=1e-10))
        held_out = (-0.27, 0.03)
        position, closed = policy.action(times[1], held_out)
        expected = np.array(
            [1.0, held_out[0] - REFERENCE_SOAP_XY[0],
             held_out[1] - REFERENCE_SOAP_XY[1]]
        ) @ weights[1]
        self.assertTrue(np.allclose(position, expected[:3], atol=1e-10))
        self.assertEqual(closed, bool(expected[3] >= 0.5))

    def test_time_interpolation_and_save_load(self):
        times = np.array([0.02, 0.04])
        weights = np.zeros((2, 3, 4))
        weights[0, 0] = [-0.42, -0.06, 1.16, 0]
        weights[1, 0] = [-0.25, 0.01, 0.94, 1]
        policy = ImitationPolicy(times, weights)
        middle, held = policy.action(0.03, REFERENCE_SOAP_XY)
        self.assertTrue(np.allclose(middle, [-0.335, -0.025, 1.05]))
        self.assertTrue(held)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.npz"
            policy.save(path)
            loaded = ImitationPolicy.load(path)
            self.assertEqual(loaded.action(0.03, REFERENCE_SOAP_XY),
                             policy.action(0.03, REFERENCE_SOAP_XY))
            self.assertTrue(np.array_equal(policy.weights, loaded.weights))

    def test_policy_requires_visual_observation(self):
        model = ImitationPolicy(np.array([0.02, 0.04]),
                                np.zeros((2, 3, 4)))
        with self.assertRaisesRegex(ValueError, "RGB"):
            model.action(0.02, None)

    def test_nonidentifiable_or_invalid_demonstrations_fail(self):
        times = np.array([0.02, 0.04])
        acts = np.zeros((4, 2, 4))
        all_same_xy = np.repeat([[-0.27, 0.005]], 4, axis=0)
        with self.assertRaisesRegex(ValueError, "independent"):
            fit_behavior_cloning(acts, times, all_same_xy)
        with self.assertRaises(ValueError):
            fit_behavior_cloning(np.zeros((3, 2, 4)), times,
                                 np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            ImitationPolicy(np.array([0.02, 0.02]),
                            np.zeros((2, 3, 4)))

    def test_no_camera_detection_prevents_learned_pickup(self):
        # No fake world-pose fallback, even though MjData knows soap XY.
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        policy = ImitationPolicy(
            np.array([0.02, 0.04]), np.zeros((2, 3, 4)),
        )
        fake_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        renderer = mock.Mock()
        renderer.render.return_value = fake_rgb
        result = learned_run(model, renderer, policy, (-0.27, 0.005))
        self.assertFalse(result["success"])
        self.assertIsNone(result["observed_xy"])
        self.assertIn("RGB", result["reason"])


if __name__ == "__main__":
    unittest.main()
