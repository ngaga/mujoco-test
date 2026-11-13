import unittest

import numpy as np

from motion import Motion
from vision import estimate_soap_xy, pixel_on_plane, soap_centroid


class VisionTests(unittest.TestCase):
    def test_yellow_soap_center(self):
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rgb[50:70, 100:120] = (245, 175, 38)
        self.assertEqual(soap_centroid(rgb), (109.5, 59.5))

    def test_missing_soap(self):
        rgb = np.full((120, 160, 3), (140, 140, 140), dtype=np.uint8)
        self.assertIsNone(soap_centroid(rgb))

    def test_camera_center_intersects_ground_directly_below(self):
        center = ((160 - 1) / 2, (120 - 1) / 2)
        xy = pixel_on_plane(center, (120, 160, 3), (0.2, -0.1, 1.75),
                            np.eye(3), 60, 0.866)
        self.assertAlmostEqual(xy[0], 0.2, places=6)
        self.assertAlmostEqual(xy[1], -0.1, places=6)

    def test_rgb_observation_projects_world_xy(self):
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rgb[55:65, 90:100] = (240, 172, 45)
        xy = estimate_soap_xy(rgb, (0, 0, 1.75), np.eye(3), 60, 0.866)
        self.assertIsNotNone(xy)
        self.assertGreater(xy[0], 0)
        self.assertAlmostEqual(xy[1], 0, places=6)

    def test_retarget_approach_only(self):
        motion = Motion()
        motion.start()
        motion.retarget_soap((-0.21, 0.09))
        self.assertEqual(motion.phases[0].target[:2], (-0.21, 0.09))
        self.assertEqual(motion.phases[2].target[:2], (-0.21, 0.09))
        self.assertEqual(motion.phases[4].target[:2], (0.255, -0.025))
        motion.advance(2.1)
        self.assertGreaterEqual(motion.index, 2)
        motion.retarget_soap((-0.4, -0.3))
        self.assertEqual(motion.phases[2].target[:2], (-0.21, 0.09))


if __name__ == "__main__":
    unittest.main()
