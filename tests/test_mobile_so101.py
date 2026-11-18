"""Physical regression tests of the actual SO-101 on our old 3DOF mobile base."""
import unittest

import mujoco
import numpy as np

from mobile_demo.so101 import (
    ACTUATORS, ARM_JOINTS, CLOSED, ExpertSO101,
    HOME, MobileSO101Env,
)


class MobileSO101Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MobileSO101Env(cubes=1)

    def setUp(self):
        self.env.reset()

    def test_vendor_arm_meshes_joints_tcp_and_cameras(self):
        model = self.env.model
        self.assertEqual(model.nu, 9)
        self.assertGreaterEqual(model.nmesh, 16)
        self.assertEqual(tuple(model.actuator(i).name for i in range(9)), ACTUATORS)
        for name in ARM_JOINTS:
            model.joint(name)
        for name in ("fixed_jaw_pad_1", "moving_jaw_pad_1"):
            model.geom(name)
        model.site("tcp")
        model.camera("wrist")
        model.camera("front")
        with self.assertRaises(KeyError):
            model.joint("reach")  # The old toy cartesian arm is removed.
        np.testing.assert_allclose(self.env.state(), HOME, atol=1e-3)

    def test_original_planar_base_physically_moves(self):
        initial = self.env.state()
        target = initial.copy()
        target[0] += 0.04
        self.env.step(target, frames=160)
        state = self.env.state()
        self.assertGreater(state[0], initial[0] + 0.001)
        self.assertLess(state[0], initial[0] + 0.08)
        self.assertAlmostEqual(self.env.model.opt.timestep, 0.002, places=6)

    def test_real_moving_jaw_is_actuated(self):
        initial = self.env.state()
        command = initial.copy()
        command[8] = CLOSED
        self.env.step(command, frames=350)
        state = self.env.state()
        self.assertNotAlmostEqual(float(state[8]), float(initial[8]), places=3)
        self.assertTrue(np.isfinite(state).all())

    def test_scripted_navigator_has_nine_commands(self):
        initial = self.env.cube_position(0).copy()
        expert = ExpertSO101(max_retries=2)
        command = expert.action(self.env)
        self.assertEqual(command.shape, (9,))
        self.assertTrue(np.isfinite(command).all())
        self.assertGreater(command[0], self.env.state()[0])
        np.testing.assert_allclose(self.env.cube_position(0), initial, atol=1e-9)
        self.assertFalse(expert.done)

    def test_cube_held_above_bin_does_not_count_as_delivered(self):
        # This recreates the previously misleading 19.998s observation.
        joint = self.env.model.joint("cube_0")
        adr = int(self.env.model.jnt_qposadr[joint.id])
        self.env.data.qpos[adr:adr + 3] = [0.9521, 0.7525, 0.1458]
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertFalse(self.env.in_bin(0))
        self.assertEqual(self.env.placed(), 0)

    def test_real_cube_free_joint_and_correct_size(self):
        model = self.env.model
        joint = model.joint("cube_0")
        self.assertEqual(int(model.jnt_type[joint.id]), int(mujoco.mjtJoint.mjJNT_FREE))
        np.testing.assert_allclose(model.geom("cube_geom_0").size, [0.012] * 3)


if __name__ == "__main__":
    unittest.main()
