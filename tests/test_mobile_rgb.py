"""CPU-only RGB segmentation and camera projection tests (no GL needed)."""
import unittest
from types import SimpleNamespace

import numpy as np

from mobile_demo.vision_rgb import red_components, calibrated_ground_intersection


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


if __name__ == "__main__":
    unittest.main()
