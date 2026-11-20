"""Joint-space motion planning for the *official* TidyBot Kinova Gen3.

Python OMPL RRTConnect plans the seven native arm joints while MuJoCo tests
collision geometry against the hotel bin, countertop, sink, cabinet and cubes.
The base is held at a fixed pose while planning each arm segment.

The checker intentionally ignores the selected cube for grasping and models a
non-contact cargo proxy at the TCP for transport. Neither object is welded to
or moved by the simulation planner; physics still decides success.

This is simulation-only clearance checking, not a robot-safety certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import mujoco
import numpy as np


class PlanningError(RuntimeError):
    """No collision-free movement found: never fall back to an unsafe path."""


@dataclass
class PlanResult:
    waypoints: list[np.ndarray]
    method: str
    checks: int
    planning_seconds: float
    max_joint_step: float


class MuJoCoCollisionChecker:
    """Query native MuJoCo collision geometry on separate, non-live MjData."""

    OBSTACLE_PREFIXES = ("hotel_bin_", "counter_", "sink_", "vanity_",
                         "hotel_wall", "hotel_cube_geom_")

    def __init__(self, env, *, margin=0.012):
        if margin < 0:
            raise ValueError("Safety margin must be nonnegative")
        self.env = env
        self.model = env.model
        self.data = mujoco.MjData(self.model)
        self.margin = float(margin)
        self.checks = 0
        self.last_blocker = None
        self.contacts = 0
        self.contact_examples = []
        self.smallest_contact_distance = 0.0

        arm_root = self.model.body("gen3/base_link").id
        parent = self.model.body_parentid

        def in_arm(body):
            while body > 0:
                if body == arm_root:
                    return True
                body = int(parent[body])
            return False

        self.arm_geoms = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if (in_arm(int(self.model.geom_bodyid[gid])) and
                    (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid])):
                self.arm_geoms.append(gid)

        # A visual-only, zero-mass cube rigidly parented under the gripper for
        # *planning checks only*. The real cube remains a free physical body.
        self.proxy = self.model.geom("hotel_cargo_clearance_proxy").id
        self.obstacles = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(self.OBSTACLE_PREFIXES):
                self.obstacles.append((gid, name))
        self.bin_geoms = {gid for gid, name in self.obstacles if name.startswith("hotel_bin_")}
        self.robot_geoms = set(self.arm_geoms)
        self._fromto = np.empty(6, dtype=float)
        if not self.arm_geoms or not self.bin_geoms:
            raise RuntimeError("Actual Gen3 collision meshes / blue bin collision geoms absent")

    def _margin(self, name):
        # Nonzero space around the blue rim; smaller near the countertop so
        # pre-grasp configurations near the work surface can be evaluated.
        if name.startswith("hotel_bin_"):
            return self.margin
        if name.startswith("hotel_cube_geom_"):
            return 0.005
        return min(self.margin, 0.006)

    def valid(self, q, base, *, carrying=False, ignore_cube=None):
        q = np.asarray(q, dtype=float)
        base = np.asarray(base, dtype=float)
        self.checks += 1
        self.last_blocker = None
        if (q.shape != (7,) or base.shape != (3,) or not np.isfinite(q).all()
                or not np.isfinite(base).all()):
            self.last_blocker = "invalid joint/base vector"
            return False
        # The env's native 7-DOF limits, including the 2/4/6 hardware bounds.
        from .hotel_vla import ARM_LOW, ARM_HIGH
        if np.any(q < ARM_LOW - 1e-5) or np.any(q > ARM_HIGH + 1e-5):
            self.last_blocker = "arm joint limits"
            return False
        self.data.qpos[:] = self.env.data.qpos
        self.data.qpos[:3] = base
        self.data.qpos[self.env.arm_qadr] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        query = self.arm_geoms + ([self.proxy] if carrying else [])
        for robot_geom in query:
            for obstacle_geom, name in self.obstacles:
                if (ignore_cube is not None and
                        name == f"hotel_cube_geom_{ignore_cube}"):
                    continue
                separation = self._margin(name)
                # MuJoCo returns distmax when pair lies outside search radius.
                dist = mujoco.mj_geomDistance(self.model, self.data,
                                              robot_geom, obstacle_geom,
                                              separation + 0.001, self._fromto)
                if dist < separation:
                    self.last_blocker = (f"{self.model.geom(robot_geom).name} vs "
                                         f"{name}: {dist:.4f}m < {separation:.4f}m")
                    return False
        return True

    def segment_valid(self, start, goal, base, *, carrying=False,
                      ignore_cube=None, joint_increment=0.045):
        start, goal = np.asarray(start), np.asarray(goal)
        if start.shape != (7,) or goal.shape != (7,):
            return False
        steps = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / joint_increment)))
        for i in range(steps + 1):
            if not self.valid(start + (goal - start) * (i / steps), base,
                              carrying=carrying, ignore_cube=ignore_cube):
                return False
        return True

    def record_contacts(self, data, *, phase):
        """Called *every physics step* to catch transient actual arm/bin hits."""
        seen = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = int(contact.geom1), int(contact.geom2)
            if ((a in self.robot_geoms and b in self.bin_geoms) or
                    (b in self.robot_geoms and a in self.bin_geoms)):
                seen.add((min(a, b), max(a, b)))
                self.smallest_contact_distance = min(self.smallest_contact_distance,
                                                      float(contact.dist))
                if len(self.contact_examples) < 12:
                    self.contact_examples.append({
                        "time": round(float(data.time), 3),
                        "phase": phase,
                        "robot_geom": self.model.geom(a if a in self.robot_geoms else b).name,
                        "bin_geom": self.model.geom(b if b in self.bin_geoms else a).name,
                        "penetration_m": round(max(0.0, -float(contact.dist)), 5),
                    })
        if seen:
            self.contacts += 1  # steps with contact, not number of contact points


class JointMotionPlanner:
    """Seven-joint RRTConnect with deterministic swept-segment verification."""

    def __init__(self, checker, *, mode="ompl", timeout=2.0,
                 joint_increment=0.045, ompl_range=0.45):
        if mode not in ("ompl", "vertical"):
            raise ValueError("mode must be ompl or vertical")
        self.checker = checker
        self.mode = mode
        self.timeout = float(timeout)
        self.joint_increment = float(joint_increment)
        self.ompl_range = float(ompl_range)
        if self.timeout <= 0 or self.joint_increment <= 0:
            raise ValueError("Planner time and interpolation step must be positive")
        self.plans = []
        self.rrt_calls = 0
        self.rrt_solutions = 0
        self.rejections = 0
        self.last_error = None
        self._ob = self._og = None
        if mode == "ompl":
            try:
                from ompl import base as ob
                from ompl import geometric as og
            except ImportError as exc:
                raise RuntimeError(
                    "OMPL Python bindings missing. Install `python -m pip install ompl==2.0.1` "
                    "in the environment used by mjpython, or use --planner vertical."
                ) from exc
            self._ob, self._og = ob, og

    def plan(self, start, target, base, *, carrying=False, ignore_cube=None):
        start = np.asarray(start, dtype=float).copy()
        target = np.asarray(target, dtype=float).copy()
        base = np.asarray(base, dtype=float).copy()
        start_checks = self.checker.checks
        then = time.perf_counter()
        kwargs = dict(carrying=carrying, ignore_cube=ignore_cube)
        if not self.checker.valid(start, base, **kwargs):
            self.rejections += 1
            raise PlanningError("Start configuration violates obstacle clearance: "
                                + str(self.checker.last_blocker))
        if not self.checker.valid(target, base, **kwargs):
            self.rejections += 1
            raise PlanningError("IK target violates obstacle clearance: "
                                + str(self.checker.last_blocker))
        if self.checker.segment_valid(start, target, base,
                                      joint_increment=self.joint_increment, **kwargs):
            coarse = [start, target]
            method = "swept-direct" if self.mode == "vertical" else "direct-shortcut"
        elif self.mode == "vertical":
            self.rejections += 1
            raise PlanningError("Direct joint interpolation intersects the bin/countertop: "
                                + str(self.checker.last_blocker))
        else:
            coarse = self._rrt(start, target, base, **kwargs)
            method = "RRTConnect"
        # Always check every segment at fine joint increments, even when OMPL
        # says it solved a path (OMPL's default subdivision is not enough).
        traj = [coarse[0].copy()]
        for a, b in zip(coarse, coarse[1:]):
            if not self.checker.segment_valid(a, b, base,
                                              joint_increment=self.joint_increment, **kwargs):
                self.rejections += 1
                raise PlanningError("Returned path failed independent swept MuJoCo check: "
                                    + str(self.checker.last_blocker))
            steps = max(1, int(math.ceil(float(np.max(np.abs(b-a))) / 0.085)))
            for k in range(1, steps + 1):
                traj.append(a + (b - a) * (k / steps))
        record = PlanResult(traj, method, self.checker.checks - start_checks,
                            round(time.perf_counter() - then, 3), self.joint_increment)
        self.plans.append(record)
        return record

    def _rrt(self, start, target, base, *, carrying=False, ignore_cube=None):
        from .hotel_vla import ARM_LOW, ARM_HIGH
        ob, og = self._ob, self._og
        space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for i in range(7):
            bounds.setLow(i, float(ARM_LOW[i]))
            bounds.setHigh(i, float(ARM_HIGH[i]))
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)
        def valid(state):
            q = np.array([state[i] for i in range(7)], dtype=float)
            return self.checker.valid(q, base, carrying=carrying,
                                      ignore_cube=ignore_cube)
        setup.setStateValidityChecker(valid)
        # Small enough OMPL subdivisions to reject narrow bin-wall crossings.
        setup.getSpaceInformation().setStateValidityCheckingResolution(0.0015)
        # OMPL 2.0.1 nanobind: ob.State(space) has no Python constructor.
        # Allocate native states through the StateSpace instead.
        a, b = space.allocState(), space.allocState()
        for i in range(7):
            a[i] = float(start[i])
            b[i] = float(target[i])
        setup.setStartAndGoalStates(a, b)
        planner = og.RRTConnect(setup.getSpaceInformation())
        planner.setRange(self.ompl_range)
        setup.setPlanner(planner)
        self.rrt_calls += 1
        if not setup.solve(self.timeout) or not setup.haveExactSolutionPath():
            self.rejections += 1
            raise PlanningError("OMPL RRTConnect found no exact collision-free path")
        self.rrt_solutions += 1
        states = setup.getSolutionPath().getStates()
        path = [np.array([s[i] for i in range(7)], dtype=float) for s in states]
        return path

    def report(self):
        return {
            "mode": self.mode,
            "plans_succeeded": len(self.plans),
            "rrt_calls": self.rrt_calls,
            "rrt_solutions": self.rrt_solutions,
            "rejections": self.rejections,
            "planning_seconds_total": round(sum(p.planning_seconds for p in self.plans), 3),
            "checks_in_successful_plans": sum(p.checks for p in self.plans),
            "waypoints_generated": sum(len(p.waypoints) for p in self.plans),
            "plan_methods": [p.method for p in self.plans],
            "joint_edge_sample_max_rad": self.joint_increment,
            "clearance_margin_m": self.checker.margin,
            "bin_arm_contact_physics_steps": self.checker.contacts,
            "worst_bin_contact_penetration_m": round(-self.checker.smallest_contact_distance, 5),
            "contact_examples": self.checker.contact_examples,
        }
