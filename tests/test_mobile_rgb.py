"""CPU-only RGB segmentation and camera projection tests (no GL needed)."""
import unittest
from types import SimpleNamespace

import numpy as np

from mobile_demo.vision_rgb import red_components, calibrated_ground_intersection
from mobile_demo.so101 import ARM_HOME, ExpertSO101


class RGBLocalizationUnitTests(unittest.TestCase):
    def test_separate_connected_red_cubes_instead_of_average(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[5:12, 4:11] = [225, 25, 15]
        rgb[40:49, 42:53] = [230, 20, 22]
        components = red_components(rgb)
        self.assertEqual(len(components), 2)
        self.assertEqual(components[0][2], 99)
        self.assertAlmostEqual(components[0][0], 47.0)
        self.assertAlmostEqual(components[0][1], 44.0)

    def test_blue_bin_is_not_misclassified_as_a_red_cube(self):
        rgb = np.full((64, 64, 3), [22, 66, 225], dtype=np.uint8)
        self.assertEqual(red_components(rgb), [])

    def test_simple_camera_ray_hits_known_plane(self):
        class Model:
            def camera(self, name):
                return SimpleNamespace(id=0, fovy=90.0)
        data = SimpleNamespace(
            cam_xpos=np.array([[0.0, 0.0, 1.0]]),
            cam_xmat=np.array([np.eye(3).flatten()]),
        )
        center = calibrated_ground_intersection(Model(), data, "camera",
                                                 49.5, 49.5, (100, 100),
                                                 target_height=0.0)
        np.testing.assert_allclose(center, [0.0, 0.0], atol=1e-10)
        offset = calibrated_ground_intersection(Model(), data, "camera",
                                                 74.5, 49.5, (100, 100),
                                                 target_height=0.0)
        np.testing.assert_allclose(offset, [0.5, 0.0], atol=1e-9)

    def test_no_valid_intersection_if_camera_sees_above_horizon(self):
        class Model:
            def camera(self, name):
                return SimpleNamespace(id=0, fovy=60.0)
        # Camera local -Z is world +Z (looking upwards).
        r = np.diag([1, -1, -1]).flatten()
        data = SimpleNamespace(cam_xpos=np.array([[0., 0., 1.]]),
                               cam_xmat=np.array([r]))
        self.assertIsNone(calibrated_ground_intersection(
            Model(), data, "camera", 49.5, 49.5, (100, 100), 0.0))


    def test_rgb_pregrasp_close_gate_uses_pixels_not_oracle_cube_xy(self):
        # The image-derived pick XY and hidden cube XY intentionally disagree.
        # A correct non-oracle gate trusts TCP vs the image command, even
        # when that can mean an actual missed grasp in imperfect perception.
        class FakeEnv:
            data = SimpleNamespace(time=2.0)

            def __init__(self, tcp_xy):
                self.tcp_xy = np.array(tcp_xy, dtype=float)

            def in_bin(self, index):
                return False

            def cube_position(self, index):
                return np.array([0.65, 0.0, 0.012])  # evaluation truth only

            def state(self):
                return np.r_[[0.0, 0.0, 0.0], ARM_HOME].copy()

            def tcp_position(self):
                return np.r_[self.tcp_xy, 0.018]

        def prepared_expert():
            return ExpertSO101(
                phase="descend", entered=0.0,
                _pick_base_goal=np.array([0.0, 0.0]),
                _phase_arm_q=ARM_HOME[:5].copy(),
                _vision_xy=np.array([0.39, 0.0]),
                _last_vision_time=2.0, _last_ik_time=2.0,
                _last_ik_error=0.0,
                vision=SimpleNamespace(error_m=[]),
            )

        aligned_to_pixels = prepared_expert()
        aligned_to_pixels.action(FakeEnv([0.39, 0.0]))
        self.assertEqual(aligned_to_pixels.phase, "close")
        np.testing.assert_allclose(aligned_to_pixels._pick_xy, [0.39, 0.0])

        aligned_to_oracle_only = prepared_expert()
        aligned_to_oracle_only.action(FakeEnv([0.65, 0.0]))
        self.assertEqual(aligned_to_oracle_only.phase, "descend")
        self.assertEqual(aligned_to_oracle_only.attempts[0], 0)


if __name__ == "__main__":
    unittest.main()
