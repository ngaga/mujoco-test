"""Protocol and simulator-grounded tests for the RGB robustness benchmark.

No OpenGL context is required here: real RGB rendering is checked explicitly
by the Linux Mesa CI step; non-rendering cases also run on macOS.
"""
import math
import unittest

import mujoco
import numpy as np

from rgb_benchmark import (
    NOMINAL_MASS_KG, RGBScenario, model_for_scenario, nominal_scenarios,
    run_rgb_trial, stress_scenarios,
)
from robustness import make_scenarios


class BlankRenderer:
    """Models an unavailable/dark RGB image to rule out hidden GT fallback."""

    def update_scene(self, data, camera):
        assert camera == "overhead"

    def render(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)


class RGBRobustnessTests(unittest.TestCase):
    def test_nominal_pairs_are_identical_to_previous_ground_truth_benchmark(self):
        cases = nominal_scenarios(100, seed=2026)
        old = make_scenarios(100, seed=2026)
        self.assertEqual(len(cases), 100)
        self.assertEqual(sum(c.injected_drop for c in cases), 50)
        for new, previous in zip(cases, old):
            self.assertEqual((new.x, new.y, new.injected_drop),
                             (previous.x, previous.y, previous.injected_drop))
            self.assertEqual(new.group, "nominal")
            self.assertEqual(new.yaw_deg, 0)
            self.assertEqual(new.shape, "box")
            self.assertEqual(new.mass_kg, NOMINAL_MASS_KG)

    def test_stress_is_seeded_balanced_and_changes_distinct_factors(self):
        cases = stress_scenarios(24, seed=2026)
        self.assertEqual(cases, stress_scenarios(24, seed=2026))
        self.assertNotEqual(cases, stress_scenarios(24, seed=2027))
        self.assertEqual(len(cases), 24)
        self.assertEqual(sum(c.injected_drop for c in cases), 12)
        for group in ("yaw", "mass", "ellipsoid", "combined"):
            sample = [c for c in cases if c.group == group]
            self.assertEqual(len(sample), 6)
            self.assertEqual(sum(c.injected_drop for c in sample), 3)
        self.assertTrue(any(c.yaw_deg != 0 for c in cases))
        self.assertTrue(any(c.mass_kg != NOMINAL_MASS_KG for c in cases))
        self.assertTrue(any(c.shape == "ellipsoid" for c in cases))
        self.assertTrue(all(-0.32 <= c.x <= -0.22 and -0.045 <= c.y <= 0.055
                            for c in cases))
        with self.assertRaises(ValueError):
            stress_scenarios(-1)

    def test_variable_shape_and_mass_are_physical_not_fake_collision_tags(self):
        nominal = RGBScenario(0, -0.27, 0.005, False)
        rounded = RGBScenario(1, -0.27, 0.005, False, "ellipsoid", 8, 0.21, "ellipsoid")
        heavy = RGBScenario(2, -0.27, 0.005, False, "mass", 0, 0.42, "box")
        for case in (nominal, rounded, heavy):
            model = model_for_scenario(case)
            soap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soap")
            soap_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "soap_geom")
            self.assertAlmostEqual(float(model.body_mass[soap_body]),
                                   case.mass_kg, delta=1e-5)
            expected = (mujoco.mjtGeom.mjGEOM_ELLIPSOID
                        if case.shape == "ellipsoid" else mujoco.mjtGeom.mjGEOM_BOX)
            self.assertEqual(model.geom_type[soap_geom], expected)
            self.assertEqual(
                model.jnt_type[mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, "soap_free"
                )],
                mujoco.mjtJoint.mjJNT_FREE,
            )
        with self.assertRaises(ValueError):
            model_for_scenario(RGBScenario(3, -0.27, 0, False, shape="sphere"))
        with self.assertRaises(ValueError):
            model_for_scenario(RGBScenario(4, -0.27, 0, False, mass_kg=0.0))

    def test_no_rgb_target_cannot_cheat_with_mujoco_position(self):
        case = RGBScenario(5, -0.27, 0.005, True)
        model = model_for_scenario(case)
        result = run_rgb_trial(model, BlankRenderer(), case, recovery_enabled=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["rendered_frames"], 1)
        self.assertIsNone(result["initial_camera_xy"])
        self.assertIsNone(result["retry_camera_xy"])
        self.assertEqual(result["final_xyz"][:2], [-0.27, 0.005])

    def test_shape_mass_dont_perturb_seeded_position_reset(self):
        case = RGBScenario(6, -0.245, -0.019, False, "combined", -16, 0.19, "ellipsoid")
        model = model_for_scenario(case)
        data = mujoco.MjData(model)
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "soap_free")
        adr = int(model.jnt_qposadr[j])
        mujoco.mj_resetData(model, data)
        data.qpos[adr:adr+2] = [case.x, case.y]
        half = math.radians(case.yaw_deg) / 2
        data.qpos[adr+3:adr+7] = [math.cos(half), 0, 0, math.sin(half)]
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(float(data.qpos[adr]), case.x)
        self.assertAlmostEqual(float(data.qpos[adr+1]), case.y)
        self.assertAlmostEqual(float(data.qpos[adr+6]), math.sin(half))


if __name__ == "__main__":
    unittest.main()
