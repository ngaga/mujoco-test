"""RGB-only cube discovery and approach for the simulated SO-101 mobile base.

This module is intentionally independent of MuJoCo cube IDs and object poses.
It uses camera detections and the base's measured (but still idealized) XY
odometry. The controller's separate task supervisor may still use simulation
truth to associate a selected visual target with an object for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import BIN_CENTER


# Known room patrol waypoints, NOT object locations. With the front camera
# pitched down ~45 degrees, scans while moving cover the arena from the -X side.
# The arm can only grasp from -X in the present scripted pick controller.
PATROL = (
    (-1.22, -0.88), (-0.75, -0.88), (-0.22, -0.88),
    (-1.22, -0.27), (-0.75, -0.27), (-0.22, -0.27),
    (-1.22,  0.34), (-0.75,  0.34), (-0.22,  0.34),
    (-1.22,  0.95), (-0.75,  0.95), (-0.22,  0.95),
)
PICK_CLEARANCE = 0.39
MAX_SWEEPS = 3


@dataclass
class RGBObjectNavigator:
    """Visual target selection + simple map-aware sweep, with no cube-pose API."""

    target_xy: np.ndarray | None = None
    last_observation_time: float = -1.0
    scans: int = 0
    frames_with_candidates: int = 0
    discoveries: int = 0
    approaches: int = 0
    patrol_index: int = 0
    patrol_visits: int = 0
    sweeps: int = 0
    ignored_xy: list[np.ndarray] = field(default_factory=list)
    failure: str | None = None
    selected_xy: list[list[float]] = field(default_factory=list)

    @staticmethod
    def selectable(xy) -> bool:
        """Filter known blue-bin footprint, walls and ungraspable margins."""
        x, y = np.asarray(xy, dtype=float)
        if not np.isfinite([x, y]).all():
            return False
        if not (-1.05 <= x <= 1.18 and -1.35 <= y <= 1.35):
            # A grasp from -X requires the base center >= -1.5.
            return False
        # Red objects already deposited in the known bin are not new targets.
        if abs(x - BIN_CENTER[0]) <= 0.21 and abs(y - BIN_CENTER[1]) <= 0.21:
            return False
        return True

    def forget_target(self) -> None:
        self.target_xy = None
        self.last_observation_time = -1.0

    def ignore_target(self) -> None:
        if self.target_xy is not None:
            self.ignored_xy.append(self.target_xy.copy())
        self.forget_target()

    def _valid(self, xy):
        return (self.selectable(xy) and
                not any(np.linalg.norm(xy - ignored) < 0.09
                        for ignored in self.ignored_xy))

    def step(self, env, vision, t: float):
        """Return (safe XY base goal, arrived visual target or None).

        Only the joint-state odometry, rendered pixels, and known static room
        geometry are used. There is deliberately NO cube_position/in_bin call.
        """
        base_xy = np.asarray(env.state()[:2], dtype=float)
        if self.last_observation_time < 0 or t - self.last_observation_time >= 0.35:
            self.last_observation_time = t
            self.scans += 1
            candidates = [np.asarray(xy, dtype=float)
                          for xy in vision.discover(env, cameras=("front",))
                          if self._valid(np.asarray(xy, dtype=float))]
            if candidates:
                self.frames_with_candidates += 1
                if self.target_xy is None:
                    # Selection comes from the camera image, not a named cube.
                    self.target_xy = min(candidates,
                                         key=lambda xy: np.linalg.norm(xy - base_xy)).copy()
                    self.discoveries += 1
                    self.selected_xy.append(self.target_xy.round(4).tolist())
                else:
                    nearby = [xy for xy in candidates
                              if np.linalg.norm(xy - self.target_xy) < 0.12]
                    if nearby:
                        best = min(nearby, key=lambda xy: np.linalg.norm(xy - self.target_xy))
                        self.target_xy = 0.75 * self.target_xy + 0.25 * best

        if self.target_xy is not None:
            goal = self.target_xy - np.array([PICK_CLEARANCE, 0.0])
            if np.linalg.norm(base_xy - goal) <= 0.018:
                self.approaches += 1
                return goal, self.target_xy.copy()
            return goal, None

        waypoint = np.asarray(PATROL[self.patrol_index], dtype=float)
        if np.linalg.norm(base_xy - waypoint) < 0.06:
            self.patrol_index = (self.patrol_index + 1) % len(PATROL)
            self.patrol_visits += 1
            if self.patrol_index == 0:
                self.sweeps += 1
                if self.sweeps >= MAX_SWEEPS:
                    self.failure = "RGB patrol exhausted without finding another undelivered cube"
                    return base_xy.copy(), None
            waypoint = np.asarray(PATROL[self.patrol_index], dtype=float)
        return waypoint, None

    def report(self):
        return {
            "navigation_source": "front RGB pixels + base joint odometry + fixed room patrol",
            "uses_cube_ground_truth_for_navigation": False,
            "scans": self.scans,
            "frames_with_candidates": self.frames_with_candidates,
            "discoveries": self.discoveries,
            "approaches": self.approaches,
            "patrol_visits": self.patrol_visits,
            "sweeps": self.sweeps,
            "ignored_targets": len(self.ignored_xy),
            "selected_xy": self.selected_xy[:20],
            "failure": self.failure,
        }
