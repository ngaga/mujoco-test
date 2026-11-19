"""Physical regression tests of SO-101 on Stanford TidyBot CAD base."""
import unittest
from importlib.util import find_spec

import mujoco
import numpy as np

from mobile_demo.so101 import (
    ACTUATORS, ARM_JOINTS, CLOSED, ExpertSO101,
    HOME, MobileSO101Env, MAX_BIN_APPROACH_BASE_X, BIN_LEFT_EDGE, VISUAL_BASE_FRONT, VISUAL_BIN_MARGIN,
)


@unittest.skipUnless(
    find_spec("sim2cell") is not None,
    "Pinned SO-101 assets are optional in generic CI; mandatory in mobile-so101.yml",
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

    def test_tidybot_cad_is_present_and_old_wheels_are_gone(self):
        model = self.env.model
        for suffix in ("body", "bumper", "bottom_plate", "top_plate", "arm_plate"):
            model.geom(f"tidybot_{suffix}_visual")
            model.geom(f"tidybot_{suffix}_collision")
        with self.assertRaises(KeyError):
            model.geom("wheel_a")

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

    def test_already_placed_cube_is_not_regrasped_from_inside_bin(self):
        joint = self.env.model.joint("cube_0")
        adr = int(self.env.model.jnt_qposadr[joint.id])
        self.env.data.qpos[adr:adr + 3] = [0.95, 0.90, 0.042]
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertTrue(self.env.in_bin(0))
        for phase in ("navigate", "hover", "descend", "recover"):
            expert = ExpertSO101(cube=0, phase=phase)
            action = expert.action(self.env)
            # A cube already in the bin must not be regrasped. Completion
            # still waits for 1.25 seconds of continuous stability.
            self.assertFalse(expert.done, phase)
            self.assertIsNotNone(expert._all_settled_since, phase)
            self.assertEqual(expert.attempts[0], 0, phase)
            self.assertTrue(np.isfinite(action).all())

    def test_requested_three_cubes_are_only_three_visible_physical_bodies(self):
        for n in (1, 3, 5):
            env = MobileSO101Env(cubes=n)
            self.assertEqual(env.cubes, n)
            self.assertEqual(len(env.cube_bodies), n)
            names = [env.model.body(i).name for i in range(env.model.nbody)]
            self.assertEqual(sum(name.startswith("red_") for name in names), n)
            for k in range(n, 5):
                self.assertNotIn(f"red_{k}", names)
            self.assertEqual(env.placed(), 0)

    def test_visual_wheel_clears_bin_left_wall(self):
        self.assertLessEqual(MAX_BIN_APPROACH_BASE_X
                             + VISUAL_BASE_FRONT + VISUAL_BIN_MARGIN,
                             BIN_LEFT_EDGE + 1e-8)
        # Old bin goal was at X=0.56: visible wheel reached X=0.785
        # despite the colliding chassis only reaching X=0.73.
        self.assertLess(MAX_BIN_APPROACH_BASE_X, 0.56)

    def test_no_false_success_if_earlier_cube_left_the_bin(self):
        env = MobileSO101Env(cubes=3)
        for k in (1, 2):
            adr = int(env.model.jnt_qposadr[env.model.joint(f"cube_{k}").id])
            env.data.qpos[adr:adr + 3] = [0.94 + 0.025 * (k - 1), 0.9, 0.042]
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env.in_bin(2))
        self.assertFalse(env.in_bin(0))
        expert = ExpertSO101(cube=2, phase="settle", entered=-5.0)
        expert.action(env)
        self.assertFalse(expert.done)
        self.assertEqual(expert.cube, 0)
        self.assertEqual(expert.phase, "recover")

    def test_completion_requires_stable_full_bin(self):
        env = MobileSO101Env(cubes=3)
        for k in range(3):
            adr = int(env.model.jnt_qposadr[env.model.joint(f"cube_{k}").id])
            env.data.qpos[adr:adr + 3] = [0.92 + k * 0.025, 0.90, 0.042]
        mujoco.mj_forward(env.model, env.data)
        self.assertEqual(env.placed(), 3)
        expert = ExpertSO101(cube=2, phase="settle", entered=-4.0)
        expert.action(env)
        self.assertFalse(expert.done)
        self.assertIsNotNone(expert._all_settled_since)

    def test_stacked_cube_below_bin_rim_counts_as_delivered(self):
        # Bin wall top is 131mm and the cube half-height is 12mm.
        # Three stacked cubes can have a centre near 89mm.
        env = MobileSO101Env(cubes=1)
        adr = int(env.model.jnt_qposadr[env.model.joint("cube_0").id])
        env.data.qpos[adr:adr + 3] = [0.89, 0.90, 0.0891]
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env.in_bin(0))
        env.data.qpos[adr + 2] = 0.1458
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env.in_bin(0))

    def test_real_cube_free_joint_and_correct_size(self):
        model = self.env.model
        joint = model.joint("cube_0")
        self.assertEqual(int(model.jnt_type[joint.id]), int(mujoco.mjtJoint.mjJNT_FREE))
        np.testing.assert_allclose(model.geom("cube_geom_0").size, [0.012] * 3)


if __name__ == "__main__":
    unittest.main()
