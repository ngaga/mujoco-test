"""Minimal RGB feedback for the MuJoCo bathroom demo.

This locates the distinctive yellow soap in a rendered image. It is a
baseline for visual servoing, not a general-purpose object detector.
"""
from math import radians, tan

import numpy as np


def soap_centroid(rgb, min_pixels=25):
    """Return (u, v) in image pixels, or None when the soap isn't visible."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected an HxWx3 RGB image")
    image = rgb[:, :, :3].astype(np.float32)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    yellow = (
        (red > 105)
        & (green > 70)
        & (red > 1.07 * green)
        & (blue < 0.67 * green)
    )
    ys, xs = np.nonzero(yellow)
    if len(xs) < min_pixels:
        return None
    return float(xs.mean()), float(ys.mean())


def pixel_on_plane(pixel, shape, camera_pos, camera_rot, fovy_degrees, plane_z):
    """Project an RGB pixel onto a horizontal world-Z plane.

    MuJoCo camera coordinates have +X right, +Y up and -Z forward.
    camera_rot is the 3x3 camera-to-world rotation (data.cam_xmat).
    The known surface height is geometry calibration, not object pose.
    """
    height, width = shape[:2]
    if height < 2 or width < 2:
        raise ValueError("Image must be at least 2 pixels wide and high")
    focal = 0.5 * height / tan(radians(fovy_degrees) / 2)
    u, v = pixel
    direction_camera = np.array(
        [(u - (width - 1) / 2) / focal,
         ((height - 1) / 2 - v) / focal,
         -1.0],
        dtype=np.float64,
    )
    origin = np.asarray(camera_pos, dtype=np.float64)
    rotation = np.asarray(camera_rot, dtype=np.float64).reshape(3, 3)
    direction = rotation @ direction_camera
    if abs(direction[2]) < 1e-8:
        return None
    scale = (plane_z - origin[2]) / direction[2]
    if scale <= 0:
        return None
    hit = origin + scale * direction
    return float(hit[0]), float(hit[1])


def estimate_soap_xy(rgb, camera_pos, camera_rot, fovy_degrees, plane_z):
    """Locate soap using only a rendered RGB image and camera calibration."""
    pixel = soap_centroid(rgb)
    if pixel is None:
        return None
    return pixel_on_plane(pixel, rgb.shape, camera_pos, camera_rot,
                          fovy_degrees, plane_z)
