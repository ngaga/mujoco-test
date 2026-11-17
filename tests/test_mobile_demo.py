"""Minimal regression tests for the mobile platform and VLA/teacher boundary."""
import unittest

import numpy as np

from mobile_demo.core import ACTION_DIM, CAMERAS, Expert, HOME, MobileEnv, TASK, validate_action


class MobileDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MobileEnv(cubes=1)

    def setUp(self):
        self.env.reset()

    def test_scene_and_controls(self):
        self.assertEqual(self.env.model.nu, ACTION_DIM)
        self.assertEqual(self.env.state().shape, (ACTION_DIM,))
        np.testing.assert_allclose(self.env.state(), HOME, atol=0.0001)

    def test_action_contract(self):
        np.testing.assert_allclose(validate_action(HOME), HOME)
        with self.assertRaises(ValueError):
            validate_action(np.zeros(6))
        with self.assertRaises(ValueError):
            validate_action(np.full(7, np.nan))
        with self.assertRaises(ValueError):
            validate_action(np.full(7, 99.))

    def test_physical_cube_untouched_by_reset(self):
        self.assertLess(abs(self.env.cube_position(0)[0] + 0.73), 0.001)
        self.assertFalse(self.env.in_bin(0))
        self.env.step(HOME, frames=10)
        self.assertFalse(self.env.in_bin(0))

    def test_expert_labels_are_not_vla(self):
        teacher = Expert()
        self.assertEqual(teacher.phase, "navigate")
        self.assertEqual(teacher.action(self.env).shape, (ACTION_DIM,))
        self.assertIn("red cube", TASK)
        self.assertEqual(CAMERAS, ("front", "wrist"))


if __name__ == "__main__":
    unittest.main()