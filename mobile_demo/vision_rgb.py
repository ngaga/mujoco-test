"""Calibrated RGB-only local cube localisation from actual SO-101 cameras.

The mobile long-range navigator remains privileged in this FIRST hybrid
experiment. At hover and descent, however, the robot's commanded target XY
comes ONLY from pixels, the camera calibration and an assumed tabletop height.
Neither the ground-truth cube pose nor the MuJoCo object ID is used for local
RGB detection. The wrist camera is independently evaluated and used when
the fixed front camera misses the target.

No trained VLA is claimed. This is classical color segmentation / geometry.
"""
from __future__ import annotations

from collections import deque
import math

import mujoco
import numpy as np


def red_components(rgb: np.ndarray, min_pixels: int = 5):
    """Return independent connected red blobs; never average all red cubes."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected RGB HxWx3 image")
    rgb = np.asarray(rgb, dtype=np.uint8)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mask = (r > 95) & (r > g * 1.45) & (r > b * 1.45) & ((r - g) > 45)
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    components = []
    for y0, x0 in zip(*np.where(mask)):
        if seen[y0, x0]:
            continue
        q = deque([(int(y0), int(x0))])
        seen[y0, x0] = True
        sx = sy = count = 0
        while q:
            y, x = q.pop()
            sx += x
            sy += y
            count += 1
            for yy, xx in ((y - 1, x), (y + 1, x),
                           (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    q.append((yy, xx))
        if count >= min_pixels:
            components.append((sx / count, sy / count, count))
    return sorted(components, key=lambda x: -x[2])


def calibrated_ground_intersection(model, data, camera, px, py, hw,
                                   target_height=0.018):
    """Project an image pixel to the known workspace plane, using camera pose.

    MuJoCo camera matrices use local +x right / +y up / -z forward.
    This uses a known tabletop elevation, NOT a cube ground-truth position.
    """
    height, width = hw
    if height < 2 or width < 2:
        raise ValueError("Camera image too small")
    cam = model.camera(camera)
    camera_id = cam.id
    fovy = float(np.asarray(cam.fovy).reshape(-1)[0])
    focal = 0.5 * height / math.tan(math.radians(fovy) * 0.5)
    ray_camera = np.array([(px - (width - 1) / 2) / focal,
                           - (py - (height - 1) / 2) / focal, -1.0])
    # Camera local -> world transformation.
    rot = data.cam_xmat[camera_id].reshape((3, 3))
    ray_world = rot @ ray_camera
    pos = data.cam_xpos[camera_id]
    if ray_world[2] >= -1e-5:
        return None
    t = (float(target_height) - pos[2]) / ray_world[2]
    if not 0 < t < 8:
        return None
    return (pos + t * ray_world)[:2]


class SO101RGBLocalizer:
    """Two actual MuJoCo RGB streams; no reading cube pose for estimation."""

    def __init__(self, model, height=224, width=224, ground_z=0.018):
        self.model = model
        self.height, self.width = height, width
        self.ground_z = ground_z
        self.renderers = {
            name: mujoco.Renderer(model, height=height, width=width)
            for name in ("front", "wrist")
        }
        self.frames = 0
        self.frames_with_target = 0
        self.camera_detections = {"front": 0, "wrist": 0}
        self.estimated = []
        self.error_m = []
        self.last_details = {}
        self.images = {}

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()

    def discover(self, env, cameras=("front",)):
        """Return all world-plane red detections, with no expected cube XY.

        Used during long-range RGB navigation. Candidate selection is entirely
        handled by the navigator, which has no MuJoCo object position API.
        """
        self.frames += 1
        found = []
        for camera in cameras:
            renderer = self.renderers[camera]
            renderer.update_scene(env.data, camera=camera)
            rgb = renderer.render().copy()
            self.images[camera] = rgb
            for u, v, count in red_components(rgb):
                xy = calibrated_ground_intersection(
                    env.model, env.data, camera, u, v, rgb.shape[:2],
                    target_height=self.ground_z)
                if xy is not None:
                    found.append(np.asarray(xy, dtype=float))
                    self.camera_detections[camera] += 1
        if found:
            self.frames_with_target += 1
        return found

    def observe(self, env, expected_xy, max_distance=0.18):
        """Use base-relative expected grasp workspace ONLY to select the blob.

        Long-range oracle navigation has already positioned the mobile base.
        A base-relative region is NOT an oracle cube position; all local
        measurement comes from actual RGB pixels.
        """
        self.frames += 1
        expected = np.asarray(expected_xy, dtype=float)
        candidates = {}
        diagnostic = {}
        for camera, renderer in self.renderers.items():
            renderer.update_scene(env.data, camera=camera)
            rgb = renderer.render().copy()
            self.images[camera] = rgb
            blobs = red_components(rgb)
            valid = []
            for u, v, count in blobs:
                world_xy = calibrated_ground_intersection(
                    env.model, env.data, camera, u, v, rgb.shape[:2],
                    target_height=self.ground_z)
                if world_xy is None:
                    continue
                distance = float(np.linalg.norm(world_xy - expected))
                if distance <= max_distance:
                    valid.append((distance, world_xy, count, u, v))
            if not valid:
                diagnostic[camera] = {"blobs": len(blobs), "valid": False}
                continue
            distance, xy, count, u, v = min(valid, key=lambda item: item[0])
            candidates[camera] = (xy, count)
            self.camera_detections[camera] += 1
            diagnostic[camera] = {
                "valid": True, "blob_pixels": count,
                "pixel_uv": [round(u, 2), round(v, 2)],
                "xy": np.asarray(xy).round(4).tolist(),
                "distance_from_expected": round(distance, 4),
            }
        self.last_details = diagnostic
        if not candidates:
            return None
        self.frames_with_target += 1
        # Front camera supplies a stable global view. If the wrist agrees
        # within 25mm, it contributes half the estimate. If it disagrees
        # (occlusion/multiple objects), do not average inconsistent points.
        front = candidates.get("front")
        wrist = candidates.get("wrist")
        if front is not None and wrist is not None:
            if np.linalg.norm(front[0] - wrist[0]) <= 0.025:
                estimate = (front[0] + wrist[0]) / 2
            else:
                estimate = front[0]
        elif front is not None:
            estimate = front[0]
        else:
            estimate = wrist[0]
        self.estimated.append(np.asarray(estimate, dtype=float).round(5).tolist())
        return np.asarray(estimate, dtype=float)

    def report(self):
        return {
            "sensor": "actual MuJoCo-rendered RGB front + wrist",
            "algorithm": "red connected-component segmentation + calibrated ray-plane intersection",
            "workspace_plane_height_m": self.ground_z,
            "frames": self.frames,
            "frames_with_target": self.frames_with_target,
            "camera_detections": self.camera_detections,
            "last_details": self.last_details,
            "sample_estimates_xy": self.estimated[:15],
            "mean_error_m": (round(float(np.mean(self.error_m)), 5)
                             if self.error_m else None),
            "max_error_m": (round(float(np.max(self.error_m)), 5)
                            if self.error_m else None),
            "error_samples": len(self.error_m),
            "error_definition": "Post-hoc comparison only; true cube XY never used for RGB estimate",
        }
