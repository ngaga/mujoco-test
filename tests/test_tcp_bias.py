"""TCP-to-fixed-jaw calibration and robot-only Jacobian correction tests."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vla"))
import tcp_bias


class Ref:
    def __init__(self, index):
        self.id = index


class Model:
    def __init__(self):
        self.pad_names = [
            *[f"fixed_jaw_pad_{i}" for i in range(1, 5)],
            *[f"moving_jaw_pad_{i}" for i in range(1, 5)]
        ]
        self.ids = {name: idx for idx, name in enumerate(self.pad_names)}
        self.nv = 12
        self.jnt_range = np.tile([-3.3, 3.3], (6, 1))
        self.jnt_limited = np.ones(6, dtype=int)

    def geom(self, name):
        return Ref(self.ids[name])

    def site(self, name):
        self.check_name(name, "tcp")
        return Ref(0)

    def body(self, name):
        self.check_name(name, "Fixed_Jaw")
        return Ref(0)

    @staticmethod
    def check_name(actual, expected):
        if actual != expected:
            raise KeyError(actual)


class Data:
    def __init__(self):
        self.qpos = np.zeros(13)
        self.qvel = np.zeros(12)
        self.geom_xpos = np.zeros((8, 3))
        self.site_xpos = np.array([0, 0, 0.05])
        self.xmat = np.eye(3).reshape(9)
        self.forward()

    def forward(self):
        # Exact fake body kinematics: first arm joint translates the whole jaw
        # +100mm/radian along world X, without altering jaw gap.
        self.geom_xpos[:4] = [-0.02 + 0.1 * self.qpos[0], 0, 0.02]
        self.geom_xpos[4:] = [0.02 + 0.1 * self.qpos[0], 0, 0.02]

    def site(self, _id):
        return types.SimpleNamespace(xpos=self.site_xpos)

    def body(self, _id):
        return types.SimpleNamespace(xmat=self.xmat)


class TCPBiasTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.state = Data()
        self.scratch = Data()

    def test_measured_tcp_to_fixed_pad_is_rigid_local_offset(self):
        c = tcp_bias.tcp_calibration(self.model, self.state)
        self.assertEqual(c["fixed_pad_from_tcp_local_mm"], [-20.0, 0.0, -30.0])
        self.assertEqual(c["jaw_midpoint_from_tcp_local_mm"], [0.0, 0.0, -30.0])
        self.assertEqual(c["closing_axis_world"], [1.0, 0.0, 0.0])

    def test_zero_bias_has_exact_reference_policy_action(self):
        nominal = np.array([0, -1, 1, 0.4, -0.5, 1.2], dtype=float)
        result, stats = tcp_bias.corrected_targets(
            self.model, self.state, self.scratch, nominal, 0
        )
        self.assertTrue(np.array_equal(result, nominal))
        self.assertEqual(stats["achieved_closing_mm"], 0)
        self.assertTrue(np.array_equal(self.state.qpos, np.zeros(13)))

    def test_direction_and_jaw_unchanged_under_local_kinematics(self):
        def jac(_model, _scratch, jacp, jacr, _geom_id):
            jacp[0, 0] = 0.1

        fake_mujoco = types.SimpleNamespace(
            mj_forward=lambda model, data: data.forward(),
            mj_jacGeom=jac,
        )
        with patch.dict(sys.modules, {"mujoco": fake_mujoco}):
            nominal = np.array([0, -1, 1, 0.4, -0.5, 1.2])
            plus, plus_stats = tcp_bias.corrected_targets(
                self.model, self.state, self.scratch, nominal, 5
            )
            minus, minus_stats = tcp_bias.corrected_targets(
                self.model, self.state, self.scratch, nominal, -5
            )
        self.assertGreater(plus_stats["achieved_closing_mm"], 4)
        self.assertLess(minus_stats["achieved_closing_mm"], -4)
        self.assertAlmostEqual(plus[5], 1.2)
        self.assertAlmostEqual(minus[5], 1.2)
        self.assertEqual(self.state.qpos.tolist(), [0] * 13)
        self.assertLessEqual(abs(plus[0]-nominal[0]), 0.12)

    def test_excessive_or_invalid_corrections_rejected(self):
        nominal = np.zeros(6)
        with self.assertRaisesRegex(ValueError, "12mm"):
            tcp_bias.corrected_targets(self.model, self.state,
                                       self.scratch, nominal, 15)
        with self.assertRaisesRegex(ValueError, "six finite"):
            tcp_bias.corrected_targets(self.model, self.state,
                                       self.scratch, np.ones(5), 0)


if __name__ == "__main__":
    unittest.main()
