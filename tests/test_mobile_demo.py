"""Minimal regression tests for the mobile platform and VLA/teacher boundary."""
import unittest

import numpy as np

from mobile_demo.core import ACTION_DIM, CAMERAS, Expert, GRIP_CLOSE, HOME, MobileEnv, TASK, validate_action


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

    def test_gripper_tuning(self):
        """Both jaws must be symmetrical, within limits, and slightly stronger."""
        model = self.env.model
        self.assertAlmostEqual(GRIP_CLOSE, 0.051)
        self.assertLessEqual(GRIP_CLOSE, self.env.model.joint("left_jaw").range[1])
        for name in ("left_pad", "right_pad"):
            geom = model.geom(name)
            self.assertAlmostEqual(float(geom.friction[0]), 6.0, places=6)
        for name in ("jaw_left", "jaw_right"):
            actuator = model.actuator(name)
            self.assertAlmostEqual(float(actuator.forcerange[1]), 18.0, places=6)
            self.assertAlmostEqual(float(actuator.forcerange[0]), -18.0, places=6)
        teacher = Expert(phase="close", entered=-1.0)
        # The close phase must command the updated jaw position symmetrically.
        self.env.data.qpos[self.env.qpos_addr[4]] = -0.18
        action = teacher.action(self.env)
        np.testing.assert_allclose(action[5:], [GRIP_CLOSE, GRIP_CLOSE])

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

    def test_drop_recovery_without_reset(self):
        teacher = Expert(max_retries=2, phase="transport", entered=-1.)
        teacher.attempts_per_cube[0] = 1
        original_xyz = self.env.cube_position(0).copy()
        action = teacher.action(self.env)
        self.assertEqual(teacher.phase, "recover")
        self.assertEqual(teacher.retries, 1)
        self.assertEqual(teacher.attempts_per_cube[0], 1)
        self.assertIsNone(teacher.failure)
        self.assertEqual(action[5], 0.)
        self.assertEqual(teacher.retry_events[0]["reason"], "dropped during transport")
        np.testing.assert_allclose(self.env.cube_position(0), original_xyz)

    def test_reacquire_current_cube_position_after_drop(self):
        import mujoco
        teacher = Expert(max_retries=3, phase="navigate", entered=-1.)
        adr = int(self.env.model.jnt_qposadr[self.env.model.joint("cube_0").id])
        # Only the TEST scenario moves the cube to simulate its new drop pose.
        self.env.data.qpos[adr:adr + 2] = [-0.35, -0.40]
        mujoco.mj_forward(self.env.model, self.env.data)
        action = teacher.action(self.env)
        np.testing.assert_allclose(action[:2], [-0.63, -0.40], atol=0.002)

    def test_retries_exhausted_skip_to_next_cube(self):
        env = MobileEnv(cubes=2)
        teacher = Expert(max_retries=1, phase="transport", entered=-1.)
        teacher.attempts_per_cube[0] = 2
        teacher.action(env)
        self.assertEqual(teacher.skipped_cubes, [0])
        self.assertEqual(teacher.cube, 1)
        self.assertEqual(teacher.phase, "recover")
        self.assertIsNone(teacher.failure)

    def test_last_cube_exhaustion_terminal(self):
        teacher = Expert(max_retries=1, phase="transport", entered=-1.)
        teacher.attempts_per_cube[0] = 2
        teacher.action(self.env)
        self.assertIsNotNone(teacher.failure)
        self.assertEqual(teacher.skipped_cubes, [0])

    def test_expert_labels_are_not_vla(self):
        teacher = Expert()
        self.assertEqual(teacher.phase, "navigate")
        self.assertEqual(teacher.action(self.env).shape, (ACTION_DIM,))
        self.assertIn("red cube", TASK)
        self.assertEqual(CAMERAS, ("front", "wrist"))


if __name__ == "__main__":
    unittest.main()