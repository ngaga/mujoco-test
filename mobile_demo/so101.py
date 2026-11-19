"""Mobile SO-101: the real sim2cell/Menagerie arm and jaw on our planar base.

Uses the *same pinned SO-ARM100 MuJoCo mechanical model as the independent
pretrained SmolVLA evaluation*, including actual moving jaw, collision meshes,
finger pads, joint ranges, TCP site and wrist camera. No surrogate fingers,
welded objects, object teleportation or fake VLA inference.

The base uses Stanford TidyBot Menagerie CAD/collisions with 3DOF planar position-servos and the original
navigate -> transit waypoint logic. This is NOT a physical powered-caster drivetrain model.
An independent oracle controls the six arm joints via the upstream DLS IK.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .core import BIN_CENTER, CUBE_COUNT, SCENE
from .tidybot_base import mount_tidybot_base

BASE_JOINTS = ("base_x", "base_y", "base_yaw")
ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
JOINTS = BASE_JOINTS + ARM_JOINTS
BASE_ACTUATORS = ("drive_x", "drive_y", "drive_yaw")
ACTUATORS = BASE_ACTUATORS + ARM_JOINTS
ARM_HOME = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 1.4])
HOME = np.r_[[-1.35, -0.86, 0.0], ARM_HOME]
SLEW = np.array([0.15, 0.15, 0.9, 1.4, 1.4, 1.4, 1.4, 1.4, 2.4])
OPEN = 1.4
CLOSED = -0.05
# TidyBot's 2023 shell extends farther forward than the old LeKiwi-like base.
# Place the small SO-101 at the *front edge* rather than buried in the 60kg CAD
# shell. Pick at a reachable standoff, avoiding the SO-101 full-extension
# singularity that caused pregrasp target contact and pushed some cubes away.
# This is a simulation kinematic research mount, not validated real hardware.
PICK_CLEARANCE = 0.54
MOUNT_XYZ = (0.34, 0.0, 0.20)
# Approximate frontal reach of the TidyBot CAD shell (the exact collision
# geometry is simulated, not approximated by this scheduling guard).
# Retain the public constant name for existing probe/test compatibility.
VISUAL_BASE_FRONT = 0.30
VISUAL_BIN_MARGIN = 0.025
BIN_LEFT_EDGE = float(BIN_CENTER[0]) - 0.20
MAX_BIN_APPROACH_BASE_X = BIN_LEFT_EDGE - VISUAL_BASE_FRONT - VISUAL_BIN_MARGIN


def upstream_arm_xml() -> Path:
    """Find the exact upstream arm+mesh assets installed from the pinned git SHA."""
    try:
        import sim2cell
    except ImportError as exc:
        raise RuntimeError(
            "The physical SO-101 assets are not installed. Install the pinned "
            "sim2cell dependency in mobile_demo/so101_requirements.txt."
        ) from exc
    path = Path(sim2cell.__file__).resolve().parent / "assets/so_arm100/so_arm100.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing actual SO-101 MJCF: {path}")
    return path


def compose_xml(cubes: int = CUBE_COUNT) -> str:
    """Nest the unmodified mechanical arm subtree under our mobile base.

    The actual mesh files are read from the installed upstream package. We
    never turn the vendor arm into toy capsules or unconstrained XYZ sliders.
    """
    if not 1 <= cubes <= CUBE_COUNT:
        raise ValueError(f"cubes must be within 1..{CUBE_COUNT}")
    arm_path = upstream_arm_xml()
    src = ET.parse(arm_path).getroot()
    root = ET.parse(SCENE).getroot()
    root.set("model", "EarlyBots mobile platform with real SO-101 manipulator")
    option = root.find("option")
    option.set("cone", "elliptic")
    option.set("impratio", "10")

    base = root.find("./worldbody/body[@name='mobile_base']")
    assert base is not None, "Original navigator base missing"
    old_stage = base.find("./body[@name='reach_stage']")
    if old_stage is None:
        raise RuntimeError("Expected old Cartesian arm to replace")
    base.remove(old_stage)
    mount = ET.SubElement(
        base, "body", {"name": "so101_mount",
                       "pos": " ".join(map(str, MOUNT_XYZ)),
                       "euler": "0 0 1.5707963267948966"})
    vendor_base = src.find("./worldbody/body[@name='Base']")
    if vendor_base is None:
        raise RuntimeError("Upstream SO-101 Base body missing")
    mount.append(copy.deepcopy(vendor_base))

    original_default = root.find("default")
    vendor_defaults = src.find("default")
    if original_default is None or vendor_defaults is None:
        raise RuntimeError("Missing MJCF defaults")
    for node in vendor_defaults:
        original_default.append(copy.deepcopy(node))

    assets = ET.Element("asset")
    vendor_assets = src.find("asset")
    if vendor_assets is None:
        raise RuntimeError("Missing SO-101 assets")
    for node in vendor_assets:
        clone = copy.deepcopy(node)
        if clone.tag == "mesh":
            file = clone.get("file")
            if not file:
                raise RuntimeError("Upstream mesh lacks file")
            # The original XML uses compiler meshdir="assets/".
            mesh = (arm_path.parent / "assets" / file).resolve()
            if not mesh.is_file():
                raise FileNotFoundError(f"Missing collision/visual mesh: {mesh}")
            clone.set("file", str(mesh))
        assets.append(clone)
    root.insert(list(root).index(root.find("worldbody")), assets)

    actuators = root.find("actuator")
    assert actuators is not None
    for node in list(actuators):
        if node.get("name") not in BASE_ACTUATORS:
            actuators.remove(node)
    for node in src.find("actuator"):
        actuators.append(copy.deepcopy(node))

    exclude_src = src.find("contact")
    if exclude_src is not None:
        contact = ET.SubElement(root, "contact")
        for node in exclude_src:
            contact.append(copy.deepcopy(node))

    # This workcell uses 24mm blocks (the same as the successful SO-101 VLA
    # experiments), not the original 48mm blocks that could be wider than the
    # actual jaw aperture. This difference is explicit, not a fair A/B test.
    # Do not just mark the extras inactive: remove their bodies from the
    # compiled visual AND physical scene so --cubes 3 shows precisely 3.
    worldbody = root.find("worldbody")
    for i in range(CUBE_COUNT):
        cube = worldbody.find(f"body[@name='red_{i}']")
        if i >= cubes:
            worldbody.remove(cube)
            continue
        geom = cube.find(f"./geom[@name='cube_geom_{i}']")
        cube.set("pos", " ".join([*cube.get("pos").split()[:2], "0.0125"]))
        geom.set("size", "0.012 0.012 0.012")
        geom.set("mass", "0.02")
        geom.set("friction", "1.2 0.005 0.0001")
    mount_tidybot_base(root)
    return ET.tostring(root, encoding="unicode")


def make_model(cubes: int = CUBE_COUNT) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(compose_xml(cubes=cubes))
    if model.nu != len(JOINTS):
        raise RuntimeError(f"Expected 9 independent drives, got {model.nu}")
    for i, name in enumerate(ACTUATORS):
        if model.actuator(name).id != i:
            raise RuntimeError(f"Actuator order mismatch: {name} -> {model.actuator(name).id}")
    model.site("tcp")
    model.camera("front")
    model.camera("wrist")
    for name in ("fixed_jaw_pad_1", "moving_jaw_pad_1"):
        model.geom(name)
    return model


class MobileSO101Env:
    """The controller receives base and motor state; privileged poses are for oracle only."""

    def __init__(self, cubes=1):
        if not 1 <= cubes <= CUBE_COUNT:
            raise ValueError("cubes must be between 1 and 5")
        self.model = make_model(cubes=cubes)
        self.data = mujoco.MjData(self.model)
        self.cubes = cubes
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id]
                              for n in JOINTS], dtype=int)
        self.cube_bodies = [self.model.body(f"red_{i}").id for i in range(cubes)]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = HOME
        self.data.ctrl[:] = HOME
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        return self.data.qpos[self.qadr].copy()

    def cube_position(self, i):
        return self.data.xpos[self.cube_bodies[i]].copy()

    def tcp_position(self):
        return self.data.site("tcp").xpos.copy()

    def in_bin(self, i):
        # Real blue bin wall top is z=0.131m. A 24mm cube stacked on other
        # cubes can have its centre at 0.09-0.12m and still be physically
        # inside the bin. Our old 85mm cutoff falsely flagged stacked cubes
        # as missed deliveries and made the robot regrasp them.
        x, y, z = self.cube_position(i)
        return bool(abs(x - BIN_CENTER[0]) < 0.17
                    and abs(y - BIN_CENTER[1]) < 0.17 and 0.025 < z < 0.120)

    def placed(self):
        return sum(self.in_bin(i) for i in range(self.cubes))

    def step(self, command, frames=None):
        target = np.asarray(command, dtype=float)
        if target.shape != (9,) or not np.isfinite(target).all():
            raise ValueError("Expected 9 finite targets: [x,y,yaw,SO101_6]")
        if frames is None:
            frames = round(1 / (15 * self.model.opt.timestep))
        dt = self.model.opt.timestep
        for _ in range(frames):
            self.data.ctrl[:] += np.clip(target - self.data.ctrl, -SLEW * dt, SLEW * dt)
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def render(self, camera="front", height=224, width=224):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()
        finally:
            renderer.close()


@dataclass
class ExpertSO101:
    """Oracle pickup/transport state machine, never presented as trained VLA."""

    cube: int = 0
    phase: str = "navigate"
    entered: float = 0.0
    attempts: list[int] = field(default_factory=lambda: [0] * CUBE_COUNT)
    retry_events: list[dict] = field(default_factory=list)
    max_retries: int = 2
    done: bool = False
    failure: str | None = None
    lift_successes: int = 0
    _phase_arm_q: np.ndarray | None = None
    _carry_offset: np.ndarray | None = None
    _pick_xy: np.ndarray | None = None
    _pick_base_goal: np.ndarray | None = None
    _hold_arm_q: np.ndarray | None = None
    _last_ik_error: float | None = None
    _last_ik_time: float = -1.0
    max_ik_error: float = 0.0
    # Do not accept a fleeting instant inside the bin as final delivery:
    # previously an object left the bin in the last physics step after done.
    _all_settled_since: float | None = None
    # Optional real RGB localizer. Navigation, post-grasp checks and bin
    # success still use simulator truth in this staged/hybrid experiment.
    vision: object | None = None
    # Optional long-range RGB target finder; its navigation step never reads cube truth.
    navigator: object | None = None
    _vision_xy: np.ndarray | None = None
    _last_vision_time: float = -1.0

    def advance(self, phase, t):
        self.phase, self.entered = phase, t
        self._all_settled_since = None
        self._phase_arm_q = None
        self._last_ik_time = -1.0

    def recover(self, env, reason):
        event = {"cube": self.cube, "time": round(float(env.data.time), 3),
                 "reason": reason, "attempts": self.attempts[self.cube],
                 "cube_xyz": env.cube_position(self.cube).round(4).tolist()}
        self.retry_events.append(event)
        self._pick_xy = None
        self._pick_base_goal = None
        self._hold_arm_q = None
        self._carry_offset = None
        self._all_settled_since = None
        self._vision_xy = None
        self._last_vision_time = -1.0
        if self.navigator is not None:
            # Bad grasp: search for the displaced object again using pixels.
            # An exhausted target is ignored so the next patrol can progress.
            if self.attempts[self.cube] > self.max_retries:
                self.navigator.ignore_target()
            else:
                self.navigator.forget_target()
            self.advance("recover", float(env.data.time))
        elif self.attempts[self.cube] <= self.max_retries:
            self.advance("recover", float(env.data.time))
        elif self.cube + 1 < env.cubes:
            self.cube += 1
            self.advance("recover", float(env.data.time))
        else:
            self.failure = "Retry budget exhausted: " + reason
        return self.safe(env)

    def safe(self, env):
        cmd = env.state()
        cmd[3:] = ARM_HOME
        cmd[2] = 0
        return cmd

    def ik(self, env, target):
        if self._phase_arm_q is None:
            from sim2cell.ik import solve_ik
            # DLS IK is non-convex: search several mechanically valid seeds,
            # as confirmed by the reproducible arm reachability scan.
            best_err, best_q = float("inf"), None
            for seed in (env.state()[3:8], ARM_HOME[:5],
                         np.array([0.0, -2.2, 2.2, 1.57, 0.0]),
                         np.array([0.0, -1.0, 1.0, 1.57, 0.0])):
                q, err = solve_ik(env.model, env.data, np.asarray(target),
                                   q_init=seed, max_iters=130, tol=0.0005)
                if err < best_err:
                    best_err, best_q = float(err), q.copy()
            self._phase_arm_q = best_q
            self._last_ik_error = best_err
            self.max_ik_error = max(self.max_ik_error, best_err)
        return self._phase_arm_q

    def action(self, env):
        if self.done or self.failure:
            return self.safe(env)
        t = float(env.data.time)
        elapsed = t - self.entered
        # The long-range navigator uses only rendered camera detections,
        # base state and a fixed room patrol. Do not read any cube position
        # before it has selected a visual target and reached its approach pose.
        if self.navigator is not None and self.phase == "navigate":
            cmd = self.safe(env)
            goal, visual_target = self.navigator.step(env, self.vision, t)
            cmd[:2] = goal
            if self.navigator.failure:
                self.failure = self.navigator.failure
                return self.safe(env)
            if visual_target is not None:
                # Oracle object-ID association is retained ONLY for the
                # subsequent scripted grasp-success/recovery supervisor and
                # evaluation. It never sets a navigation target or base goal.
                remaining = [i for i in range(env.cubes) if not env.in_bin(i)]
                if not remaining:
                    self.failure = "No undelivered objects to associate with RGB target"
                    return self.safe(env)
                self.cube = min(remaining, key=lambda i:
                                np.linalg.norm(env.cube_position(i)[:2] - visual_target))
                self._pick_xy = visual_target.copy()
                self._pick_base_goal = np.asarray(goal, dtype=float).copy()
                self._vision_xy = visual_target.copy()
                self._last_vision_time = -1.0
                self.advance("hover", t)
            return cmd

        # Never claim success while the cube is still held in transport.
        # The bin occupancy check is meaningful only before pickup or after
        # the release+settle stage, with an actual low resting cube height.
        # Once a cube settles inside the blue bin, it must not be picked again,
        # including when this happens mid-approach/descend during a retry.
        # Regrasping an already delivered cube caused the real jaw geometry
        # to press into the bin FLOOR on the five-cube scenario.
        if (self.phase in ("navigate", "recover", "hover", "descend")
                and (self.navigator is None or self.phase in ("hover", "descend"))
                and env.in_bin(self.cube)):
            if self.navigator is not None:
                # This cube reached the bin without a new pickup (e.g. a
                # previous loose release). Move on to vision-driven search.
                if env.placed() == env.cubes:
                    if self._all_settled_since is None:
                        self._all_settled_since = t
                    if t - self._all_settled_since >= 1.25:
                        self.done = True
                else:
                    self.navigator.forget_target()
                    self.advance("recover", t)
                return self.safe(env)
            if self.cube == env.cubes - 1:
                missing = next((i for i in range(env.cubes)
                                if not env.in_bin(i)), None)
                if missing is None:
                    if self._all_settled_since is None:
                        self._all_settled_since = t
                    if t - self._all_settled_since >= 1.25:
                        self.done = True
                    return self.safe(env)
                # Previously deposited blocks can be knocked out by a later
                # release. Don't mark complete until ALL are still in bin.
                if self.attempts[missing] > self.max_retries:
                    self.failure = f"Previously delivered cube {missing} left bin and exhausted attempts"
                    return self.safe(env)
                self.cube = missing
                self.advance("recover", t)
                return self.safe(env)
            self.cube += 1
            self.advance("recover", t)
        if self.phase == "recover":
            cmd = self.safe(env)
            if elapsed >= 1.3:
                self.advance("navigate", t)
            return cmd
        cube = env.cube_position(self.cube)
        current = env.state()
        cmd = current.copy()
        cmd[2] = 0.0
        cmd[3:] = ARM_HOME

        if self.phase == "navigate":
            # Same approach side and physical X/Y+yaw navigation as core.Expert.
            cmd[:2] = cube[:2] - np.array([PICK_CLEARANCE, 0])
            if np.linalg.norm(current[:2] - cmd[:2]) < 0.018 and elapsed > 0.5:
                # A grasped cube moves with the robot. Never chase its
                # changing world position while closing or lifting.
                self._pick_xy = cube[:2].copy()
                self._pick_base_goal = cmd[:2].copy()
                self.advance("hover", t)
            return cmd

        if self.phase in ("hover", "descend", "close", "lift"):
            # Progressive autonomy: only HOVER/DESCEND grasp target switches
            # from privileged simulator cube XY to actual RGB measurements.
            # Navigation still uses oracle cube XY and bin delivery uses truth.
            if self.vision is not None and self.phase in ("hover", "descend"):
                if self._pick_base_goal is None:
                    raise RuntimeError("RGB grasp requires navigator's base-relative approach pose")
                expected_xy = self._pick_base_goal + np.array([PICK_CLEARANCE, 0.0])
                if self._last_vision_time < 0 or t - self._last_vision_time > 0.30:
                    measurement = self.vision.observe(env, expected_xy)
                    self._last_vision_time = t
                    if measurement is not None:
                        self._vision_xy = measurement
                        # Evaluation metric ONLY, never used as input by estimator
                        # or as a control or gating condition.
                        self.vision.error_m.append(
                            float(np.linalg.norm(measurement - cube[:2])))
                if self._vision_xy is None:
                    if elapsed > 4.0:
                        self.attempts[self.cube] += 1
                        return self.recover(env, "RGB cube target not visible")
                    cmd[:2] = self._pick_base_goal
                    return cmd
                pick_xy = self._vision_xy.copy()
            elif self.phase in ("hover", "descend"):
                pick_xy = cube[:2].copy()
            else:
                pick_xy = (self._pick_xy if self._pick_xy is not None
                           else (self._vision_xy if self._vision_xy is not None
                                 else cube[:2].copy()))
            if self.vision is not None and self.phase in ("hover", "descend"):
                # While the jaws are OPEN, correct the actual mobile base pose
                # using pixels so the SO-101 wrist stays in its reachable
                # workspace instead of pushing the cube away at arm reach limit.
                cmd[:2] = np.asarray(pick_xy) - np.array([PICK_CLEARANCE, 0.0])
            else:
                # Once fingers close, keep the last corrected base position:
                # moving the mobile chassis would tear the object from the grip.
                cmd[:2] = (self._pick_base_goal if self._pick_base_goal is not None
                           else pick_xy - np.array([PICK_CLEARANCE, 0]))
            # In RGB mode we know tabletop geometry, not the cube's actual Z.
            # On the oracle baseline, retain the adaptive Z for elevated cubes.
            grasp_z = (0.018 if self.vision is not None else
                       max(0.018, float(cube[2]) + 0.006))
            height = {"hover": max(0.095, grasp_z + 0.075),
                      "descend": grasp_z,
                      "close": grasp_z, "lift": max(0.20, grasp_z + 0.182)}[self.phase]
            if (self.phase in ("hover", "descend")
                    and (self._last_ik_time < 0 or t - self._last_ik_time >= 0.25)):
                # A position command cached at phase entry stops working if
                # the base yields under load or contact nudges the cube.
                self._phase_arm_q = None
                self._last_ik_time = t
            if self.phase == "close":
                # Keep the descent pose, not a new IK solution that might
                # disturb a delicately contacted cube before closure.
                cmd[3:8] = self._phase_arm_q if self._phase_arm_q is not None else self.ik(env, [*pick_xy, height])
                cmd[8] = CLOSED
                if elapsed >= 0.9:
                    self.advance("lift", t)
            else:
                cmd[3:8] = self.ik(env, [*pick_xy, height])
                cmd[8] = OPEN if self.phase in ("hover", "descend") else CLOSED
                if (self._last_ik_error is not None and self._last_ik_error > 0.045):
                    # IK misses still count against the per-cube retry budget.
                    self.attempts[self.cube] += 1
                    return self.recover(env, f"unreachable TCP ({self._last_ik_error:.3f} m)")
                if self.phase == "hover" and elapsed > 1.4:
                    self.advance("descend", t)
                elif self.phase == "descend" and elapsed > 1.4:
                    tcp = env.tcp_position()
                    # In RGB mode the close decision must use the visual target,
                    # never the cube's privileged MuJoCo pose. In oracle mode
                    # pick_xy is that same oracle pose, preserving the baseline.
                    xy_error = float(np.linalg.norm(tcp[:2] - pick_xy))
                    z_error = abs(float(tcp[2]) - height)
                    if xy_error < 0.009 and z_error < 0.014:
                        # Close on the last successful approach target: do
                        # not create a new IK solution during finger contact.
                        self._pick_xy = pick_xy.copy()
                        self._pick_base_goal = np.asarray(cmd[:2]).copy()
                        aligned_q = self._phase_arm_q.copy()
                        self.attempts[self.cube] += 1
                        self.advance("close", t)
                        self._phase_arm_q = aligned_q
                    elif elapsed > 3.5:
                        self.attempts[self.cube] += 1
                        return self.recover(env,
                            f"pregrasp TCP misaligned (xy={xy_error:.3f}m,z={z_error:.3f}m)")
                elif self.phase == "lift" and elapsed > 1.6:
                    if cube[2] < 0.065:
                        return self.recover(env, "cube not retained after lift")
                    self.lift_successes += 1
                    # Preserve target joint angles throughout base travel.
                    # Continuously setting target = measured angles caused
                    # motor commands to follow gravity instead of resisting it.
                    self._hold_arm_q = self._phase_arm_q.copy()
                    self._carry_offset = env.tcp_position()[:2] - current[:2]
                    self.advance("transport", t)
            return cmd

        # Reuse original two-stage navigator: travel below the bin in Y,
        # then approach from that side with chassis clearance.
        offset = (self._carry_offset if self._carry_offset is not None
                  else np.array([PICK_CLEARANCE, 0]))
        base_goal = BIN_CENTER - offset
        # Keep the whole TidyBot CAD shell clear of the bin during approach;
        # actual robot-bin mesh collisions remain enabled and instrumented.
        base_goal[0] = min(float(base_goal[0]), MAX_BIN_APPROACH_BASE_X)
        cmd[3:8] = self._hold_arm_q if self._hold_arm_q is not None else current[3:8]
        cmd[8] = CLOSED
        if self.phase in ("transport", "approach_bin"):
            cmd[:2] = base_goal + (
                np.array([0, -0.49]) if self.phase == "transport" else 0)
            if cube[2] < 0.065 and elapsed > 0.35:
                return self.recover(env, "dropped during moving-base transport")
            if np.linalg.norm(current[:2] - cmd[:2]) < 0.025 and elapsed > 0.4:
                self.advance("approach_bin" if self.phase == "transport" else "release", t)
        elif self.phase == "release":
            cmd[:2] = base_goal
            cmd[8] = OPEN
            if elapsed > 0.75:
                self.advance("settle", t)
        elif self.phase == "settle":
            cmd[:2] = base_goal
            cmd[8] = OPEN
            if elapsed > 1.0:
                if not env.in_bin(self.cube) or env.state()[8] < 0.8:
                    return self.recover(env, "missed blue bin or jaw not open after release")
                if self.navigator is not None:
                    if env.placed() == env.cubes:
                        if self._all_settled_since is None:
                            self._all_settled_since = t
                        if t - self._all_settled_since >= 1.25:
                            self.done = True
                    else:
                        self.navigator.forget_target()
                        self.advance("recover", t)
                elif self.cube == env.cubes - 1:
                    missing = next((i for i in range(env.cubes)
                                    if not env.in_bin(i)), None)
                    if missing is None:
                        if self._all_settled_since is None:
                            self._all_settled_since = t
                        if t - self._all_settled_since >= 1.25:
                            self.done = True
                    else:
                        self._all_settled_since = None
                        if self.attempts[missing] > self.max_retries:
                            self.failure = f"Cube {missing} still outside bin after retry budget"
                        else:
                            self.cube = missing
                            self.advance("recover", t)
                else:
                    self.cube += 1
                    self.advance("recover", t)
        else:
            raise RuntimeError(f"Unknown phase: {self.phase}")
        return cmd


def execute(cubes=1, max_steps=1000, max_retries=2,
            viewer=False, viewer_port=8080, perception="oracle"):
    if perception not in ("oracle", "rgb", "rgb-nav"):
        raise ValueError("perception must be 'oracle', 'rgb' or 'rgb-nav'")
    env = MobileSO101Env(cubes=cubes)
    expert = ExpertSO101(max_retries=max_retries)
    if perception in ("rgb", "rgb-nav"):
        from .vision_rgb import SO101RGBLocalizer
        expert.vision = SO101RGBLocalizer(env.model)
    if perception == "rgb-nav":
        from .rgb_navigation import RGBObjectNavigator
        expert.navigator = RGBObjectNavigator(pick_clearance=PICK_CLEARANCE)
    from contextlib import nullcontext
    if viewer:
        from mujoco.studio import launch_web, viewer_protocol
        ctx = launch_web.launch(viewer_protocol.ViewerConfig(
            title="EarlyBots | Stanford TidyBot base + SO-101",
            http_port=viewer_port))
    else:
        ctx = nullcontext(None)
    telemetry = []
    try:
      with ctx as display:
        last = time.perf_counter()
        for step in range(max_steps):
            if display and not display.is_running():
                break
            cmd = expert.action(env)
            env.step(cmd)
            if step % 15 == 0 or expert.done or expert.failure:
                telemetry.append({
                    "time": round(float(env.data.time), 3),
                    "phase": expert.phase,
                    "base_xy": env.state()[:2].round(4).tolist(),
                    "cube_xyz": env.cube_position(expert.cube).round(4).tolist(),
                    "tcp_xyz": env.tcp_position().round(4).tolist(),
                    "arm_q": env.state()[3:8].round(4).tolist(),
                    "arm_target": env.data.ctrl[3:8].round(4).tolist(),
                })
            if display:
                display.sync(env.model, env.data)
                remain = 1 / 15 - (time.perf_counter() - last)
                if remain > 0:
                    time.sleep(remain)
                last = time.perf_counter()
            if expert.done or expert.failure:
                break
    finally:
        # Release the two headless camera renderers; viewer is separate.
        if expert.vision is not None:
            expert.vision.close()
    result = {
        "robot": "sim2cell SO-101 arm mounted on Stanford TidyBot Menagerie CAD base",
        "base": "Stanford TidyBot 2023 visual + collision meshes, 3DOF X/Y/yaw position servo (not caster dynamics)",
        "policy": (
            "RGB front navigation + RGB front/wrist local grasp + oracle task supervisor + DLS IK"
            if perception == "rgb-nav" else
            "oracle navigation + RGB front/wrist local grasp + DLS IK"
            if perception == "rgb" else
            "oracle navigation + oracle grasp + DLS IK"),
        "perception": perception,
        "rgb_perception": expert.vision.report() if expert.vision is not None else None,
        "rgb_navigation": (expert.navigator.report()
                           if expert.navigator is not None else None),
        "cube_side_m": 0.024,
        "cubes_requested": cubes, "cubes_in_bin": env.placed(),
        "success": bool(expert.done and env.placed() == cubes),
        "lift_successes": expert.lift_successes,
        "attempts_per_cube": expert.attempts[:cubes],
        "retries": len(expert.retry_events),
        "retry_events": expert.retry_events,
        "max_ik_error_m": round(expert.max_ik_error, 5),
        "failure": expert.failure,
        "final_phase": expert.phase,
        "steps": step + 1,
        "simulation_seconds": round(float(env.data.time), 3),
        "tcp_xyz": env.tcp_position().round(4).tolist(),
        "cube_xyz": [env.cube_position(i).round(4).tolist() for i in range(cubes)],
        "base_xy": env.state()[:2].round(4).tolist(),
        "telemetry_1hz": telemetry,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--cubes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--perception", choices=("oracle", "rgb", "rgb-nav"), default="oracle",
                        help="rgb uses local camera targeting; rgb-nav also finds cubes and drives to them using front RGB + base odometry")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--require-success", action="store_true",
                        help="Explicit opt-in: fail if full physical delivery fails")
    parser.add_argument("--output", type=Path, default=Path("mobile_runs/mobile_so101.json"))
    args = parser.parse_args(argv)
    if args.check_model:
        model = make_model()
        print(json.dumps({"robot": "real SO-101 MuJoCo model",
                          "actuators": model.nu,
                          "meshes": model.nmesh,
                          "geoms": model.ngeom,
                          "tcp": "tcp",
                          "cameras": ["front", "wrist"]}, indent=2))
        return
    report = execute(args.cubes, args.max_steps, args.max_retries,
                     args.viewer, args.viewer_port, args.perception)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.require_success and not report["success"]:
        raise SystemExit("SO-101 physical delivery failed: see report")


if __name__ == "__main__":
    main()
