"""Camera calibration regression for the oblique bathroom RGB camera.

A tilted camera is desirable for visibility, but all image-to-soap-world
projections must use its real MuJoCo extrinsics rather than assuming nadir.
"""
import unittest
from pathlib import Path
from math import atan2, degrees, tan, radians

import mujoco
import numpy as np

from vision import pixel_on_plane
from physical_grasp import SOAP_TOP_Z

ROOT = Path(__file__).resolve().parents[1]


class CameraPoseTests(unittest.TestCase):
    def test_tilt_and_world_projection_both_scenes(self):
        for scene, plane in (("bathroom.xml", 0.866),
                             ("bathroom_physical.xml", SOAP_TOP_Z)):
            with self.subTest(scene=scene):
                model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / scene))
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
                self.assertGreaterEqual(camera_id, 0)
                origin = data.cam_xpos[camera_id]
                rotation = data.cam_xmat[camera_id].reshape(3, 3)
                forward = -rotation[:, 2]
                tilt = degrees(atan2(np.linalg.norm(forward[:2]), abs(forward[2])))
                self.assertGreater(tilt, 18)
                self.assertLess(tilt, 35)
                self.assertLess(forward[2], 0)
                self.assertLess(origin[1], -0.2)

                # Cast known world points through calibrated pinhole camera,
                # then invert the pixel to plane projection.
                h, w = 480, 640
                f = (0.5 * h) / tan(radians(float(model.cam_fovy[camera_id])) / 2)
                for x, y in [(-0.27, 0.005), (-0.20, 0.01), (0.25, -0.025)]:
                    world = np.array([x, y, plane])
                    local = rotation.T @ (world - origin)
                    self.assertLess(local[2], 0)
                    u = f * (local[0] / -local[2]) + (w - 1) / 2
                    v = (h - 1) / 2 - f * (local[1] / -local[2])
                    self.assertTrue(0 <= u < w and 0 <= v < h)
                    recovered = pixel_on_plane(
                        (u, v), (h, w, 3), origin, rotation,
                        float(model.cam_fovy[camera_id]), plane,
                    )
                    self.assertIsNotNone(recovered)
                    self.assertAlmostEqual(recovered[0], x, places=5)
                    self.assertAlmostEqual(recovered[1], y, places=5)


if __name__ == "__main__":
    unittest.main()
