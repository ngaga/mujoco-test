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

    def test_retarget_is_continuous_and_bounded(self):
        motion = Motion()
        baseline = Motion()
        motion.start()
        baseline.start()
        for _ in range(30):
            motion.advance(0.01)
            baseline.advance(0.01)
        previous = motion.position
        motion.retarget_soap((-0.21, 0.09))
        self.assertEqual(motion.position, previous)  # no instant position jump
        self.assertEqual(motion.phases[0].target[:2], (-0.27, 0.005))
        self.assertAlmostEqual(motion.target_offset[0], 0.06)
        self.assertAlmostEqual(motion.target_offset[1], 0.085)
        motion.advance(0.01)
        baseline.advance(0.01)
        # Visual correction speed must never exceed 0.22 m/s per axis.
        for axis in (0, 1):
            deviation = motion.position[axis] - baseline.position[axis]
            self.assertLessEqual(abs(deviation), 0.00221)

    def test_visual_correction_converges_without_phase_jumps(self):
        motion = Motion()
        motion.start()
        motion.retarget_soap((-0.21, 0.09))
        for _ in range(200):
            motion.advance(0.01)
        self.assertEqual(motion.index, 2)
        self.assertAlmostEqual(motion.offset[0], 0.06, delta=0.002)
        self.assertAlmostEqual(motion.offset[1], 0.085, delta=0.002)
        # Later observations cannot move a target already being picked up.
        motion.retarget_soap((-0.4, -0.3))
        self.assertAlmostEqual(motion.target_offset[0], 0.06)
        self.assertAlmostEqual(motion.target_offset[1], 0.085)
        for _ in range(200):
            motion.advance(0.01)
        self.assertGreaterEqual(motion.index, 4)
        self.assertFalse(motion.finished)


if __name__ == "__main__":
    unittest.main()
