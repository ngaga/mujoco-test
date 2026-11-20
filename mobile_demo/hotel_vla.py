"""Hotel bathroom pick-place experiment with official Stanford TidyBot (Kinova/2F85).

Three *physical* red cubes, initially scattered on the countertop.
If dropped, they stay where physics leaves them and the teacher can retry. The blue bin is on the countertop. Nothing is welded/teleported.

Policies:
  scripted: privileged MuJoCo-pose + numerical IK engineering baseline, NOT VLA.
  smolvla: real LeRobot inference, requires a task-trained, 11D local checkpoint.

    mjpython -m mobile_demo.hotel_vla --check-model
    mjpython -m mobile_demo.hotel_vla --policy scripted --viewer
    mjpython -m mobile_demo.hotel_vla --policy smolvla --checkpoint PATH --viewer

The model's base X/Y/yaw drives are ideal position servos (not caster dynamics).
This is a research demo, not evidence of real-robot collision safety.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext

import mujoco
import numpy as np

from .tidybot_full import ensure_full_assets

TASK = "Tidy the hotel bathroom: put all three red cubes into the blue bin."
CAMERAS = {"hotel_front": "camera1", "wrist": "camera2"}
ACTUATORS = ("joint_x", "joint_y", "joint_th", *(f"joint_{i}" for i in range(1, 8)), "fingers_actuator")
STATE_JOINTS = (*ACTUATORS[:-1], "right_driver_joint")
CONTROL_HZ = 15
CUBES = 3
CUBE_HALF = 0.012
COUNTER_Z = 0.755
BIN_XY = np.array([0.785, 0.590], dtype=float)
BIN_BOTTOM_Z = COUNTER_Z + 0.028
BIN_INNER_HALF = 0.132
OPEN_GRIPPER = 0.0
CLOSE_GRIPPER = 235.0
ARM_NAMES = tuple(f"joint_{i}" for i in range(1, 8))
# joint 2/4/6 are limited in the native model, others use conservative software limits.
ARM_LOW = np.array([-math.pi, -2.23, -math.pi, -2.56, -math.pi, -2.08, -math.pi])
ARM_HIGH = -ARM_LOW
# Absolute actuator target convention. Gripper is in native 0..255 units, NOT radians.
ACTION_LOW = np.r_[[-1.1, -1.4, -math.pi], ARM_LOW, [0.0]]
ACTION_HIGH = np.r_[[0.30, 1.05, math.pi], ARM_HIGH, [255.0]]
MAX_TARGET_RATE = np.r_[[0.16, 0.16, 0.65], np.full(7, 0.85), [420.0]]
# All three spawns are explicit, orientations intentionally differ.
SPAWNS = (
    ((0.701, -0.335, COUNTER_Z + 0.018), (0.9950, 0.0, 0.0, 0.0998)),
    ((0.695, -0.050, COUNTER_Z + 0.018), (0.9886, 0.095, 0.025, 0.108)),
    ((0.687, 0.235, COUNTER_Z + 0.018), (0.963, -0.038, 0.070, -0.256)),
)


def _add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _static_box(world, name, pos, size, rgba, collision=True):
    return _add(world, "geom", name=name, type="box", pos=" ".join(map(str, pos)),
                size=" ".join(map(str, size)), rgba=rgba,
                contype=int(collision), conaffinity=int(collision), group=0 if collision else 2)


def compose_hotel_xml(*, xml_text=None, assets=None):
    """Build room around pinned official robot. Accept fixtures for offline XML tests."""
    if xml_text is None:
        source, assets = ensure_full_assets()
        xml_text = source.read_text(encoding="utf-8")
    if assets is None:
        raise ValueError("Official model requires the pinned, verified mesh map")
    root = ET.fromstring(xml_text)
    root.set("model", "EarlyBots | hotel bathroom | real TidyBot Kinova Robotiq")
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Expected official Stanford TidyBot compiler")
    compiler.attrib.pop("meshdir", None)
    for mesh in root.findall("./asset/mesh"):
        key = mesh.get("file")
        if key not in assets:
            raise ValueError(f"Missing pinned official TidyBot mesh: {key}")
        mesh.set("file", str(Path(assets[key]).resolve()))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Expected official Stanford TidyBot worldbody")
    # Elevated mast camera sees the countertop: the native base camera at 33 cm
    # is BELOW hotel countertop height and cannot perceive its clutter.
    robot_base = world.find("body[@name='base_link']")
    if robot_base is None:
        raise ValueError("Expected the native base_link body")
    _add(robot_base, "geom", name="hotel_camera_mast", type="capsule",
         fromto="-0.12 0 .34 -0.12 0 1.03", size="0.013",
         mass="0", contype="0", conaffinity="0", group="2",
         rgba="0.32 0.35 0.38 1")
    _add(robot_base, "camera", name="hotel_front", pos="-0.12 0 1.04",
         xyaxes="0 1 0 -0.42 0 -0.91", fovy="70")
    _add(world, "light", name="hotel_light", pos="0 -0.8 2.8", dir="0 0.25 -1",
         diffuse="0.95 0.93 0.88")
    _add(world, "camera", name="hotel_overview", pos="0.65 -2.1 2.05",
         xyaxes="1 0 0 0 0.65 0.76", fovy="65")
    _add(world, "geom", name="hotel_floor", type="plane", size="3 3 0.01",
         rgba="0.83 0.81 0.76 1", friction="1.0 0.01 0.0001")
    _static_box(world, "wall_mirror", (1.1, 1.19, 1.42), (0.82, 0.018, 0.58),
                "0.66 0.77 0.80 0.6", collision=False)
    _static_box(world, "hotel_wall", (0, 1.26, 1.0), (2.4, 0.06, 1.0),
                "0.88 0.86 0.81 1", collision=True)
    # Cabinet under a stone countertop, with a real recessed sink. Crucially,
    # there is no invisible tabletop slab underneath the sink hole.
    # Thin cabinet panels leave the sink physically hollow below the basin.
    _static_box(world, "vanity_back_panel", (1.51, .22, .35),
                (.025, .59, .35), "0.39 0.30 0.24 1")
    _static_box(world, "vanity_left_panel", (1.15, -.36, .35),
                (.37, .02, .35), "0.39 0.30 0.24 1")
    _static_box(world, "vanity_right_panel", (1.15, .80, .35),
                (.37, .02, .35), "0.39 0.30 0.24 1")
    stone = "0.82 0.81 0.78 1"
    for name, x, y, sx, sy in (
        ("front", .75, .24, .20, .64),
        ("back", 1.395, .24, .155, .64),
        ("left", 1.095, -.27, .145, .13),
        ("right", 1.095, .535, .145, .345),
    ):
        _static_box(world, "counter_" + name, (x, y, COUNTER_Z - .02),
                    (sx, sy, .02), stone)
    # Sink basin floor and four inner walls, physically below countertop height.
    _static_box(world, "sink_bottom", (1.095, .025, .617),
                (.145, .17, .012), "0.96 0.96 0.94 1")
    for name, x, y, sx, sy in (
        ("near", .944, .025, .007, .18), ("far", 1.246, .025, .007, .18),
        ("west", 1.095, -.152, .15, .007), ("east", 1.095, .202, .15, .007),
    ):
        _static_box(world, "sink_wall_" + name, (x, y, .68),
                    (sx, sy, .065), "0.95 0.95 0.94 1")
    # Small rim/base of faucet, decoration only (does not block motion).
    _static_box(world, "faucet", (1.38, .15, .79), (.016, .016, .055),
                "0.64 0.65 0.68 1", collision=False)
    # Blue open bin ON the real physical countertop; bottom is not at floor z=0.
    bx, by = BIN_XY
    _static_box(world, "hotel_bin_bottom", (bx, by, COUNTER_Z + .014),
                (.152, .152, .014), "0.04 0.19 0.88 1")
    for name, dx, dy, sx, sy in (
        ("north", 0, .145, .152, .007), ("south", 0, -.145, .152, .007),
        ("west", -.145, 0, .007, .138), ("east", .145, 0, .007, .138),
    ):
        _static_box(world, "hotel_bin_" + name,
                    (bx + dx, by + dy, COUNTER_Z + .077), (sx, sy, .063),
                    "0.04 0.19 0.78 1")

    # Zero-mass/zero-contact cargo proxy, used ONLY by the planner to check
    # swept clearance of a held 24mm cube. Real cube remains freejoint physics.
    bracelet = root.find(".//body[@name='bracelet_link']")
    if bracelet is None:
        raise ValueError("Official Kinova bracelet_link missing")
    _add(bracelet, "geom", name="hotel_cargo_clearance_proxy", type="box",
         pos="0 0 -0.195", size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
         mass="0", contype="0", conaffinity="0", group="4",
         rgba="0.95 0.08 0.07 0")

    # Cube free-joints must be added to *all* official keyframes, otherwise
    # mj_resetDataKeyframe sees a dimension mismatch or invalid cube quats.
    key_suffix = []
    for index, (xyz, quat) in enumerate(SPAWNS):
        quat = np.asarray(quat, dtype=float)
        quat /= np.linalg.norm(quat)
        body = _add(world, "body", name=f"hotel_red_{index}",
                    pos=" ".join(f"{v:.7f}" for v in xyz),
                    quat=" ".join(f"{v:.9f}" for v in quat))
        _add(body, "freejoint", name=f"hotel_cube_joint_{index}")
        _add(body, "geom", name=f"hotel_cube_geom_{index}", type="box",
             size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}", mass="0.020",
             friction="1.2 0.005 0.0001", rgba="0.95 0.08 0.07 1")
        key_suffix.extend([*xyz, *quat])
    for key in root.findall("./keyframe/key"):
        native = key.get("qpos")
        if not native:
            raise ValueError("Every native TidyBot keyframe must have qpos")
        key.set("qpos", native + " " + " ".join(f"{v:.9f}" for v in key_suffix))
    return ET.tostring(root, encoding="unicode")


def make_model():
    model = mujoco.MjModel.from_xml_string(compose_hotel_xml())
    for name in ACTUATORS:
        model.actuator(name)
    for name in (*CAMERAS, "hotel_overview"):
        model.camera(name)
    model.site("pinch_site")
    if model.nu != len(ACTUATORS):
        raise RuntimeError(f"Unexpected actuator count: {model.nu} (expected 11)")
    return model


def validate_action(value):
    action = np.asarray(value, dtype=float).reshape(-1)
    if action.shape != (len(ACTUATORS),) or not np.isfinite(action).all():
        raise ValueError("VLA must output exactly 11 finite native actuator targets")
    if np.any(action < ACTION_LOW - 1e-3) or np.any(action > ACTION_HIGH + 1e-3):
        raise ValueError("VLA action outside checked base/arm/gripper command limits")
    return np.clip(action, ACTION_LOW, ACTION_HIGH)


class HotelEnv:
    def __init__(self):
        self.model = make_model()
        self.data = mujoco.MjData(self.model)
        self.action_ids = [self.model.actuator(n).id for n in ACTUATORS]
        self.joint_adr = np.array([int(self.model.jnt_qposadr[self.model.joint(n).id])
                                   for n in STATE_JOINTS], dtype=int)
        self.cube_ids = [self.model.body(f"hotel_red_{i}").id for i in range(CUBES)]
        self.arm_qadr = np.array([int(self.model.jnt_qposadr[self.model.joint(n).id])
                                       for n in ARM_NAMES], dtype=int)
        self.arm_vadr = np.array([int(self.model.jnt_dofadr[self.model.joint(n).id])
                                       for n in ARM_NAMES], dtype=int)
        self.site_id = self.model.site("pinch_site").id
        self.contact_monitor = None
        self.monitor_phase = "inactive"
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.ctrl[:] = np.asarray(self.model.key("home").ctrl)
        # Start in the open aisle facing the vanity, not intersecting cabinets.
        self.data.qpos[:3] = [-0.48, -0.76, 0.0]
        self.data.ctrl[:3] = self.data.qpos[:3]
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return np.asarray(self.data.qpos[self.joint_adr], dtype=np.float32).copy()

    def cube_pos(self, index):
        """PRIVILEGED: evaluation and scripted IK only, never passed to VLA."""
        return np.asarray(self.data.xpos[self.cube_ids[index]], dtype=float).copy()

    def placed(self, index):
        x, y, z = self.cube_pos(index)
        # Require center over the bin bottom, not outside its walls.
        return bool(abs(x - BIN_XY[0]) < BIN_INNER_HALF - CUBE_HALF and
                    abs(y - BIN_XY[1]) < BIN_INNER_HALF - CUBE_HALF and
                    BIN_BOTTOM_Z + CUBE_HALF - .006 < z < BIN_BOTTOM_Z + .135)

    def count_placed(self):
        return sum(self.placed(i) for i in range(CUBES))

    def cubes_settled(self):
        """Physics-based final check; all objects have low linear speed in the bin."""
        return (self.count_placed() == CUBES and
                all(np.linalg.norm(self.data.cvel[body_id, 3:]) < 0.10
                    for body_id in self.cube_ids))

    def observation(self, renderers):
        images = {}
        for camera, dest in CAMERAS.items():
            renderers[camera].update_scene(self.data, camera=camera)
            image = np.asarray(renderers[camera].render(), dtype=np.uint8).copy()
            if image.shape != (224, 224, 3):
                raise RuntimeError(f"Invalid camera shape {camera}: {image.shape}")
            images[dest] = image
        return {"state": self.state(), "images": images, "task": TASK}

    def step(self, action):
        target = validate_action(action)
        dt = self.model.opt.timestep
        frames = max(1, round(1 / (CONTROL_HZ * dt)))
        for _ in range(frames):
            ids = self.action_ids
            previous = self.data.ctrl[ids]
            self.data.ctrl[ids] = previous + np.clip(
                target - previous, -MAX_TARGET_RATE * dt, MAX_TARGET_RATE * dt)
            mujoco.mj_step(self.model, self.data)
            if self.contact_monitor is not None:
                self.contact_monitor.record_contacts(self.data, phase=self.monitor_phase)
        return self.state()


class CompatibleSmolVLA:
    """No privileged poses, no silent retargeting from SO-101 or other robots."""
    def __init__(self, checkpoint: str, device: str):
        path = Path(checkpoint).expanduser().resolve()
        contract_file = path / "hotel_action_contract.json"
        if not contract_file.is_file():
            raise ValueError(
                "Expected a LOCAL trained TidyBot checkpoint with hotel_action_contract.json. "
                "The existing SO-101 six-action checkpoint is incompatible. "
                "See mobile_demo/HOTEL_VLA.md."
            )
        contract = json.loads(contract_file.read_text(encoding="utf-8"))
        expected = {"actuators": list(ACTUATORS), "state_joints": list(STATE_JOINTS),
                    "cameras": list(CAMERAS.values()), "control_hz": CONTROL_HZ,
                    "action_convention": "absolute_native_targets_si_gripper_0_255"}
        if any(contract.get(key) != val for key, val in expected.items()):
            raise ValueError("Checkpoint robot/camera/action convention differs from hotel TidyBot")
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        import torch
        self.torch = torch
        self.policy = SmolVLAPolicy.from_pretrained(str(path)).to(device).eval()
        def last(features, key):
            obj = features.get(key)
            return None if obj is None else list(obj.shape)[-1]
        if (last(self.policy.config.input_features, "observation.state") != len(STATE_JOINTS)
                or last(self.policy.config.output_features, "action") != len(ACTUATORS)):
            raise ValueError("Checkpoint must have exactly 11 observation and action dimensions")
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        self.policy.reset()

    def action(self, env: HotelEnv, renderers):
        obs = env.observation(renderers)
        t = self.torch
        batch = {"observation.state": t.as_tensor(obs["state"]).unsqueeze(0).to(self.device),
                 "task": obs["task"]}
        for key, image in obs["images"].items():
            batch[f"observation.images.{key}"] = (
                t.as_tensor(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0)
        with t.no_grad():
            predicted = self.post(self.policy.select_action(self.pre(batch)))
        return validate_action(predicted.detach().cpu().numpy())


class ScriptedIK:
    """Deterministic privileged-pose controller. Debug baseline only, not VLA."""
    phases = ("navigate_pick", "hover", "lower", "close", "lift",
              "transit_retract", "navigate_bin", "preplace", "place", "open", "retract",
              "depart_bin", "clear_bin", "recover_release", "recover_backoff",
              "recover_home", "settle_all")

    def __init__(self, env: HotelEnv, retries=-1, planner="vertical", planner_timeout=2.5):
        from .collision_planner import MuJoCoCollisionChecker, JointMotionPlanner
        from .hotel_recovery import RetryScheduler
        self.scheduler = RetryScheduler(CUBES, max_retries=retries)
        self.checker = MuJoCoCollisionChecker(env)
        self.planner = JointMotionPlanner(self.checker, mode=planner, timeout=planner_timeout)
        self.planner_mode = planner
        env.contact_monitor = self.checker
        self.arm_path = []
        self.path_cursor = 0
        self.vertical_index = 0
        self.planner_errors = []
        self.env = env
        self.model = env.model
        self.work = mujoco.MjData(self.model)
        self.cube = 0
        self.phase = "navigate_pick"
        self.entered = 0.0
        self.retries = 0
        self.max_retries = retries
        self.failure = None
        self.done = False
        self.attempts = self.scheduler.attempts
        self.recovery_base = None
        self.recovery_arm = None
        self.base_goal = None
        self.arm_goal = None
        self.prev_ik_error = None
        self.home_arm = env.model.key("home").ctrl[3:10].copy()
        self.pose_target = None
        self.settling_since = None
        self._last_clamp_phase = None

    def planner_report(self):
        return self.planner.report()

    def retry_report(self):
        return self.scheduler.report(self.env.placed)

    def recover(self, reason):
        """Abort this attempt, keep physical object where it fell, resume work.

        No simulator reset, no manual cube position change, no welded cargo.
        After repeated misses on one cube, rotate to another undelivered cube.
        This is privileged oracle supervision, not learned VLA recovery.
        """
        if self.done or self.failure or self.phase.startswith("recover_"):
            return
        old_cube = self.cube
        event = self.scheduler.failed(old_cube, reason,
                                      self.env.cube_pos(old_cube),
                                      self.env.data.time, self.phase)
        self.retries += 1
        # Retry the same cube on the first miss; alternate afterwards.
        rotate = event["attempt"] % 2 == 0
        next_cube = self.scheduler.choose(self.env.placed,
                                          current=old_cube, rotate=rotate)
        if next_cube is None:
            self.failure = ("No recoverable cubes left under retry budget: " + reason
                            if self.env.count_placed() != CUBES else None)
            self.done = self.env.count_placed() == CUBES
            return
        self.cube = next_cube
        self.recovery_arm = self.env.data.ctrl[self.env.action_ids][3:10].copy()
        current_base = self.env.data.qpos[:3].copy()
        self.recovery_base = current_base.copy()
        # Back away from the tabletop / bin before trying a new arm pose.
        self.recovery_base[0] = max(ACTION_LOW[0] + .05, current_base[0] - .24)
        self.advance("recover_release")
        print(f"Recovery {self.retries}: {event['reason']} at {event['cube_xyz']}; "
              f"retry cube {self.cube}, bin={self.env.count_placed()}/{CUBES}",
              flush=True)

    def after_step(self):
        """Detect real drops from world poses, never from intended gripper motion."""
        if self.failure or self.done or self.phase.startswith("recover_"):
            return
        if self.phase not in ("lift", "transit_retract", "navigate_bin", "preplace"):
            return
        if self.env.placed(self.cube):
            return
        if self.env.data.time - self.entered < .55:
            return
        tcp = self.env.data.site(self.env.site_id).xpos
        xyz = self.env.cube_pos(self.cube)
        distance = float(np.linalg.norm(xyz - tcp))
        # Conservative threshold: a held 24mm cube sits near pinch_site.
        # If it falls on the floor, the distance quickly grows as arm moves.
        if distance > .18:
            self.recover(f"Cube {self.cube} lost from gripper "
                         f"(distance={distance:.3f}m)")

    def recovery_command(self, cmd, elapsed):
        """Open, back off with static arm, then stow before starting next pick."""
        cmd[-1] = OPEN_GRIPPER
        if self.phase == "recover_release":
            cmd[3:10] = self.recovery_arm
            if elapsed > .75:
                self.advance("recover_backoff")
        elif self.phase == "recover_backoff":
            cmd[:3] = self.recovery_base
            cmd[3:10] = self.recovery_arm
            if (np.linalg.norm(self.env.data.qpos[:2] -
                               self.recovery_base[:2]) < .035 or elapsed > 6):
                self.advance("recover_home")
        elif self.phase == "recover_home":
            cmd[:3] = self.recovery_base
            cmd[3:10] = self.home_arm
            err = float(np.max(np.abs(self.env.data.qpos[self.env.arm_qadr] -
                                      self.home_arm)))
            if err < .16 and elapsed > .3:
                self.advance("navigate_pick")
            elif elapsed > 12:
                # Do not remain stuck forever; log a new retry of this cube.
                self.advance("navigate_pick")
        return validate_action(np.clip(cmd, ACTION_LOW, ACTION_HIGH))

    def advance(self, phase):
        # During Robotiq OPENING, freeze the Gen3 posture at its
        # *current commanded* angles. Re-running multi-seed IK during release
        # can switch to another 7D solution and sweep a finger into the wall.
        held = (self.env.data.ctrl[self.env.action_ids][3:10].copy()
                if phase == "open" else None)
        self.phase = phase
        self.entered = float(self.env.data.time)
        self.arm_goal = held
        self.arm_path = []
        self.path_cursor = 0
        self.vertical_index = 0
        self.settling_since = None

    def solve_ik(self, xyz, base_goal, *, avoid_bin=False, carrying=False):
        """Damped-least-squares 6D IK, then collision-filter candidate solutions.

        For delivery, DO NOT accept an IK answer solely because TCP is close.
        Another joint configuration may reach the same TCP without hitting bin.
        """
        model, data = self.model, self.work
        site_id = self.env.site_id
        desired = np.diag([1.0, -1.0, -1.0])
        best_error, best = math.inf, None
        best_safe_error, best_safe = math.inf, None
        current = self.env.data.qpos[self.env.arm_qadr]
        for seed in (current, self.home_arm,
                     np.array([0, .8, 2.3, -1.8, 0, .7, 0]),
                     np.array([0, -.8, 2.3, -1.8, 0, -.7, 0]),
                     np.array([.3, -.5, 1.3, -1.2, .25, .2, 0])):
            data.qpos[:] = self.env.data.qpos
            data.qvel[:] = 0
            data.qpos[:3] = base_goal
            data.qpos[self.env.arm_qadr] = np.clip(seed, ARM_LOW, ARM_HIGH)
            seed_error, seed_best = math.inf, None
            for _ in range(110):
                mujoco.mj_forward(model, data)
                site = data.site(site_id)
                pos_err = np.asarray(xyz) - site.xpos
                rotation = np.asarray(site.xmat).reshape(3, 3)
                rot_err = .5 * sum(
                    (np.cross(rotation[:, i], desired[:, i]) for i in range(3)),
                    np.zeros(3))
                err = np.r_[pos_err, rot_err * .18]
                score = float(np.linalg.norm(pos_err) + .045 * np.linalg.norm(rot_err))
                if score < seed_error:
                    seed_error = score
                    seed_best = data.qpos[self.env.arm_qadr].copy()
                if np.linalg.norm(pos_err) < .006 and np.linalg.norm(rot_err) < .27:
                    break
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                j = np.vstack([jacp[:, self.env.arm_vadr],
                               .18 * jacr[:, self.env.arm_vadr]])
                dq = j.T @ np.linalg.solve(j @ j.T + .035**2 * np.eye(6), err)
                data.qpos[self.env.arm_qadr] = np.clip(
                    data.qpos[self.env.arm_qadr] + np.clip(dq, -.13, .13),
                    ARM_LOW, ARM_HIGH)
            if seed_best is not None and seed_error < best_error:
                best_error, best = seed_error, seed_best
            if avoid_bin and seed_best is not None and seed_error < .09:
                if self.checker.valid(seed_best, base_goal, carrying=carrying,
                                      ignore_cube=self.cube):
                    if seed_error < best_safe_error:
                        best_safe_error, best_safe = seed_error, seed_best
            if not avoid_bin and best_error < .019:
                break
            if avoid_bin and best_safe_error < .019:
                break
        if avoid_bin:
            self.prev_ik_error = best_safe_error if best_safe is not None else best_error
            if best_safe is None or best_safe_error > .075:
                return None
            return best_safe
        self.prev_ik_error = best_error
        if best is None or best_error > .09:
            return None
        return best

    def command(self):
        env = self.env
        now = float(env.data.time)
        elapsed = now - self.entered
        cmd = env.data.ctrl[self.env.action_ids].copy()
        if self.done or self.failure:
            return cmd
        if self.phase.startswith("recover_"):
            return self.recovery_command(cmd, elapsed)
        if self.phase == "settle_all":
            if env.cubes_settled() and elapsed > .9:
                self.done = True
            elif env.count_placed() != CUBES and elapsed > .8:
                target = self.scheduler.choose(env.placed, current=self.cube, rotate=True)
                if target is not None:
                    self.cube = target
                    self.advance("depart_bin")
            return validate_action(np.clip(cmd, ACTION_LOW, ACTION_HIGH))
        # Scripted mode MAY inspect all cube world poses. The VLA NEVER does.
        if env.placed(self.cube) and self.phase in ("navigate_pick", "hover"):
            return self.next_cube(cmd)
        cube = env.cube_pos(self.cube)
        # Clearance must exceed the vanity but remain inside Gen3's reachable
        # volume from the safe base standoff. 0.92m caused an IK/servo stall.
        carry_height = max(COUNTER_Z + .10, cube[2] + .11)
        # Countertop items are higher and near the edge of the Gen3's
        # workspace. A 53cm standoff balances reach with countertop collision clearance
        # hover while keeping the TidyBot chassis short of the vanity.
        pick_standoff = .57 if cube[2] < .20 else .53
        # Try slightly different safe reach when a pose repeatedly fails.
        variants = (0.0, .04, -.035, .065, -.055)
        pick_standoff += variants[self.attempts[self.cube] % len(variants)]
        take_base = np.array([cube[0] - pick_standoff, cube[1], 0.0])
        # The previous 57cm standoff left the TCP too far from bin center:
        # last baseline stopped before deposit. 51cm is still outside cabinet.
        put_base = np.array([BIN_XY[0] - .495, BIN_XY[1], 0.0])
        cmd[-1] = CLOSE_GRIPPER if self.phase in (
            "close", "lift", "transit_retract", "navigate_bin", "preplace", "place") else OPEN_GRIPPER
        if self.phase == "navigate_pick":
            cmd[:3] = take_base
            cmd[3:10] = self.home_arm
            if np.linalg.norm(env.data.qpos[:2] - take_base[:2]) < .025 and elapsed > .5:
                self.advance("hover")
        elif self.phase == "clear_bin":
            # Keep the folded, empty gripper static while the BASE first
            # backs away from the bin. Switching to home arm at the rim
            # previously scraped the blue south wall between pickups.
            exit_base = np.array([put_base[0] - .22, BIN_XY[1] - .45, 0.0])
            cmd[:3] = exit_base
            if (np.linalg.norm(env.data.qpos[:2] - exit_base[:2]) < .03
                    and elapsed > .5):
                self.advance("navigate_pick")
        elif self.phase == "navigate_bin":
            cmd[:3] = put_base
            if np.linalg.norm(env.data.qpos[:2] - put_base[:2]) < .025 and elapsed > .5:
                self.advance("preplace")
        else:
            # Keep base where it is; solve IK relative to physical current base pose.
            base_goal = (take_base if self.phase in
                         ("hover", "lower", "close", "lift", "transit_retract")
                         else put_base)
            cmd[:3] = base_goal
            if self.phase == "depart_bin":
                # Leave the blue bin WITHOUT the carried cube. Move back
                # behind its front wall before changing base translation.
                goal = np.array([put_base[0] + .27, put_base[1],
                                 COUNTER_Z + .145])
            elif self.phase == "transit_retract":
                # Fold the carried cube inward before crossing the rim.
                goal = np.array([take_base[0] + .28, take_base[1],
                                 COUNTER_Z + .145])
            elif self.phase in ("hover", "lower", "close", "lift"):
                goal = np.array([cube[0], cube[1],
                                 cube[2] + (.14 if self.phase == "hover" else
                                            .026 if self.phase in ("lower", "close") else
                                            carry_height - cube[2])])
            else:
                # For the final place/retract, enforce multiple Cartesian Z
                # waypoints at fixed bin center; check/planning every joint edge.
                if self.phase == "preplace":
                    goal_z = BIN_BOTTOM_Z + .139
                elif self.phase == "place":
                    goal_z = BIN_BOTTOM_Z + .139 - (self.vertical_index + 1) * (.064 / 4)
                elif self.phase == "retract":
                    goal_z = BIN_BOTTOM_Z + .075 + (self.vertical_index + 1) * (.064 / 4)
                else:
                    goal_z = BIN_BOTTOM_Z + .075
                goal = np.array([BIN_XY[0], BIN_XY[1], goal_z])
            # The elevated cube pickup also needs collision-safe joint
            # routing: the direct hover target previously stalled joints 5/6.
            delivery = self.phase in (
                "hover", "transit_retract", "preplace", "place",
                "retract", "depart_bin")
            if self.arm_goal is None:
                self.arm_goal = self.solve_ik(
                    goal, base_goal, avoid_bin=delivery,
                    carrying=self.phase in ("transit_retract", "preplace", "place"))
                if self.arm_goal is None:
                    self.recover(f"IK unreachable in {self.phase}, "
                                 f"error={self.prev_ik_error:.3f}m")
                    return self.recovery_command(cmd, 0.0)
                if delivery:
                    from .collision_planner import PlanningError
                    try:
                        result = self.planner.plan(
                            env.data.qpos[self.env.arm_qadr], self.arm_goal,
                            base_goal, carrying=self.phase in ("transit_retract", "preplace", "place"),
                            ignore_cube=self.cube)
                        self.arm_path = result.waypoints[1:]
                        self.path_cursor = 0
                    except PlanningError as exc:
                        reason = f"Planner refused {self.phase} cube {self.cube}: {exc}"
                        self.planner_errors.append(reason)
                        self.recover(reason)
                        return self.recovery_command(cmd, 0.0)
            if delivery and self.arm_path:
                actual = env.data.qpos[self.env.arm_qadr]
                while (self.path_cursor < len(self.arm_path) and
                       np.max(np.abs(actual - self.arm_path[self.path_cursor])) < .065):
                    self.path_cursor += 1
                cmd[3:10] = self.arm_path[min(self.path_cursor, len(self.arm_path) - 1)]
            else:
                cmd[3:10] = self.arm_goal
            distance = np.linalg.norm(env.data.qpos[self.env.arm_qadr] - self.arm_goal)
            path_finished = (not delivery or self.path_cursor >= len(self.arm_path))
            if (self.phase in ("hover", "lift", "transit_retract",
                                   "preplace", "retract", "depart_bin")
                    and path_finished and distance < .15 and elapsed > .5):
                next_phase = {"hover": "lower", "lift": "transit_retract",
                              "transit_retract": "navigate_bin",
                              "preplace": "place", "retract": "done",
                              "depart_bin": "clear_bin"}[self.phase]
                if self.phase == "retract" and self.vertical_index < 3:
                    self.vertical_index += 1
                    self.arm_goal = None
                    self.arm_path = []
                    self.path_cursor = 0
                elif next_phase == "done":
                    # Assess physical result after release, not model intention.
                    if elapsed > 1.25:
                        stable = (env.placed(self.cube) and
                                  np.linalg.norm(env.data.cvel[env.cube_ids[self.cube], 3:]) < .10)
                        if stable:
                            if self.settling_since is None:
                                self.settling_since = now
                            elif now - self.settling_since > .65:
                                return self.next_cube(cmd)
                        else:
                            self.settling_since = None
                            if elapsed > 3.0:
                                self.recover(
                                    f"Cube {self.cube} did not settle inside bin after release")
                                return self.recovery_command(cmd, 0.0)
                else:
                    self.advance(next_phase)
            elif self.phase == "lower" and distance < .10 and elapsed > .7:
                self.advance("close")
            elif self.phase == "close" and elapsed > 1.15:
                self.advance("lift")
            elif (self.phase == "place" and path_finished and
                  distance < .10 and elapsed > .15):
                if self.vertical_index < 3:
                    self.vertical_index += 1
                    self.arm_goal = None
                    self.arm_path = []
                    self.path_cursor = 0
                else:
                    self.advance("open")
            elif self.phase == "open" and elapsed > .9:
                self.advance("retract")
        if elapsed > 22.0 and not self.phase.startswith("recover_"):
            self.recover(f"Timed out in phase {self.phase} on cube {self.cube}")
            return self.recovery_command(cmd, 0.0)
        # Native Gen3 actuators may start at exact +/-pi and occasionally
        # slightly exceed our conservative *software* range during movement.
        # Clamp ONLY the scripted IK teacher. Model/VLA outputs are still
        # strictly rejected by validate_action, never silently retargeted.
        bounded = np.clip(cmd, ACTION_LOW, ACTION_HIGH)
        if np.any(np.abs(bounded - cmd) > 1e-6) and self._last_clamp_phase != self.phase:
            limited = [(name, round(float(a), 4), round(float(b), 4))
                       for name, a, b in zip(ACTUATORS, cmd, bounded)
                       if abs(a - b) > 1e-6]
            print(f"Scripted IK command clamped in {self.phase}: {limited}", flush=True)
            self._last_clamp_phase = self.phase
        return validate_action(bounded)

    def next_cube(self, cmd):
        current = self.cube
        next_cube = self.scheduler.choose(self.env.placed,
                                          current=current, rotate=True)
        if next_cube is None:
            if self.env.count_placed() == CUBES:
                # Wait for gravity and contacts to settle: placement != settled.
                self.advance("settle_all")
            else:
                self.failure = "No eligible undelivered cubes under retry budget"
        else:
            self.cube = next_cube
            # Do not drive the base with the gripper still inside the bin.
            if self.phase in ("navigate_pick", "hover"):
                self.advance("navigate_pick")
            else:
                self.advance("depart_bin")
        return cmd


def run(args):
    env = HotelEnv()
    script = (ScriptedIK(env, args.retries, planner=args.planner,
                        planner_timeout=args.planner_timeout)
              if args.policy == "scripted" else None)
    model_policy = (CompatibleSmolVLA(args.checkpoint, args.device)
                    if args.policy == "smolvla" else None)
    renderers = ({camera: mujoco.Renderer(env.model, width=224, height=224)
                  for camera in CAMERAS} if model_policy else {})
    frames = 0
    error = None
    vla_stable_since = None
    vla_done = False
    injected_drop = False
    drop_open_until = -1.0
    ctx = nullcontext(None)
    if args.viewer:
        from mujoco.studio import launch_web, viewer_protocol
        ctx = launch_web.launch(viewer_protocol.ViewerConfig(
            title="EarlyBots | Hotel bathroom | Kinova Gen3 / Robotiq", http_port=args.viewer_port))
    try:
        with ctx as viewer:
            tick = time.perf_counter()
            for frames in range(1, args.max_steps + 1):
                if viewer and not viewer.is_running():
                    error = "Viewer closed"
                    break
                try:
                    action = (script.command() if script else model_policy.action(env, renderers))
                    if (args.inject_drop_once and script and not injected_drop and
                            script.phase == "transit_retract" and
                            env.data.time - script.entered > .6):
                        injected_drop = True
                        drop_open_until = float(env.data.time) + 1.0
                        print("Injected real physical gripper opening once during carry",
                              flush=True)
                    if injected_drop and env.data.time < drop_open_until:
                        action = action.copy()
                        action[-1] = OPEN_GRIPPER
                    env.monitor_phase = script.phase if script else "VLA"
                    contacts_before = script.checker.contacts if script else 0
                    env.step(action)
                    if script:
                        if (args.planner == "ompl" and
                                script.checker.contacts > contacts_before and
                                not script.phase.startswith("recover_")):
                            # Safety interrupt: cancel planned movement, retry.
                            script.recover("Physical arm/bin contact: "
                                           + str(script.checker.contact_examples[-1]))
                        else:
                            script.after_step()
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    error = f"Controller error: {type(exc).__name__}: {exc}"
                    print(error, flush=True)
                    break
                if frames % 15 == 0 or env.count_placed() == CUBES:
                    print(f"t={env.data.time:.1f}s cubes={env.count_placed()}/3 "
                          f"phase={script.phase if script else 'VLA'}", flush=True)
                if viewer:
                    viewer.sync(env.model, env.data)
                    delay = 1 / CONTROL_HZ - (time.perf_counter() - tick)
                    if delay > 0:
                        time.sleep(delay)
                    tick = time.perf_counter()
                if model_policy:
                    stable = (env.cubes_settled() and
                              env.data.ctrl[env.action_ids[-1]] < 30.0)
                    if stable:
                        if vla_stable_since is None:
                            vla_stable_since = float(env.data.time)
                        if env.data.time - vla_stable_since >= 1.0:
                            vla_done = True
                            break
                    else:
                        vla_stable_since = None
                if script and (script.done or script.failure):
                    break
            if viewer and args.hold_viewer:
                try:
                    while viewer.is_running():
                        viewer.sync(env.model, env.data)
                        time.sleep(.05)
                except KeyboardInterrupt:
                    pass
    finally:
        for renderer in renderers.values():
            renderer.close()
    if script and not script.done and not script.failure and frames >= args.max_steps:
        error = (f"Episode reached {args.max_steps} steps: "
                 f"{env.count_placed()}/{CUBES} cubes placed, "
                 f"{script.retries} retries performed")
    result = {"robot": "official Stanford TidyBot Gen3/Robotiq 2F85",
              "scene": "hotel bathroom: three countertop cubes, countertop blue bin",
              "policy": args.policy, "vla_inference_executed": model_policy is not None,
              "privileged_object_poses_used_by_policy": script is not None,
              "cubes": CUBES, "delivered": env.count_placed(),
              "success": bool(env.cubes_settled() and
                              (script.done if script else vla_done)),
              "failure": ((script.failure or error) if script else error),
              "simulation_seconds": round(float(env.data.time), 2),
              "steps": frames, "final_cube_xyz": [env.cube_pos(i).round(4).tolist()
                                                     for i in range(CUBES)],
              "script_retries": script.retries if script else None,
              "script_last_ik_error": script.prev_ik_error if script else None,
              "planner": script.planner_report() if script else None,
              "planner_errors": script.planner_errors if script else None,
              "recovery": script.retry_report() if script else None,
              "injected_gripper_drop": injected_drop,
              "last_arm_q": env.data.qpos[env.arm_qadr].round(4).tolist(),
              "last_arm_target": (script.arm_goal.round(4).tolist()
                                  if script and script.arm_goal is not None else None),
              "last_arm_joint_error_rad": (
                  round(float(np.linalg.norm(
                      env.data.qpos[env.arm_qadr] - script.arm_goal)), 4)
                  if script and script.arm_goal is not None else None)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if args.require_success and not result["success"]:
        raise SystemExit("Physical pickup+delivery failed; see JSON report")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--policy", choices=("scripted", "smolvla"), default="scripted")
    parser.add_argument("--checkpoint", help="Local trained 11D TidyBot SmolVLA checkpoint")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--retries", type=int, default=-1,
                        help="Additional retries per cube; -1 keeps revisiting "
                             "all undelivered cubes until --max-steps")
    parser.add_argument("--inject-drop-once", action="store_true",
                        help="Physics fault test: open Robotiq during first transport, "
                             "then verify that the controller attempts recovery")
    parser.add_argument("--planner", choices=("vertical", "ompl"), default="vertical",
                        help="Vertical Cartesian deposit with checked direct arm moves, or OMPL RRTConnect")
    parser.add_argument("--planner-timeout", type=float, default=2.5,
                        help="Wall-clock seconds allowed for each RRTConnect query")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--no-hold-viewer", dest="hold_viewer", action="store_false")
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("mobile_runs/hotel_vla.json"))
    args = parser.parse_args(argv)
    if args.planner_timeout <= 0:
        parser.error("--planner-timeout must be positive")
    if args.inject_drop_once and args.policy != "scripted":
        parser.error("--inject-drop-once requires the scripted recovery controller")
    if args.max_steps < 1 or args.retries < -1:
        parser.error("--max-steps must be positive; --retries must be >= -1")
    if args.policy == "smolvla" and not args.checkpoint:
        parser.error("Real VLA inference requires a trained --checkpoint for 11D TidyBot")
    if args.check_model:
        env = HotelEnv()
        assert env.state().shape == (11,), f"Joint state must be flat 11D, got {env.state().shape}"
        assert env.arm_qadr.shape == (7,) and env.arm_vadr.shape == (7,)
        print(json.dumps({"model": env.model.nbody, "nu": env.model.nu,
                          "cameras": list(CAMERAS) + ["hotel_overview"],
                          "cubes": CUBES, "state_dim": len(env.state()),
                          "policy": "no inference in --check-model"}, indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
