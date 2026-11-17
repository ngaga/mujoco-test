"""Physical MuJoCo environment, 7-actuator command contract, and oracle expert.

The base is holonomic-position-actuated and the arm Cartesian (2-DOF): this is
LeKiwi-*inspired*, NOT the original LeKiwi/ SO-101 kinematics or wheel physics.
The expert uses privileged cube positions. No VLA is involved in expert rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).with_name("scene.xml")
TASK = "Pick up every red cube and put it in the blue bin."
JOINTS = ("base_x", "base_y", "base_yaw", "reach", "lift", "left_jaw", "right_jaw")
CAMERAS = ("front", "wrist")
ACTION_DIM = len(JOINTS)
CUBE_COUNT = 5
BIN_CENTER = np.array([0.95, 0.90], dtype=np.float64)
# Positions as (x,y,yaw,reach,lift,left-jaw,right-jaw), in SI units.
HOME = np.array([-1.21, -0.86, 0, 0.28, -0.18, 0, 0], dtype=np.float64)
LOW = np.array([-1.50, -1.50, -np.pi, 0.06, -0.475, 0, 0], dtype=np.float64)
HIGH = np.array([1.50, 1.50, np.pi, 0.38, 0.08, 0.055, 0.055], dtype=np.float64)
# Max slew rate for commanded position setpoints, not motor velocity limits.
SLEW = np.array([0.7, 0.7, 1.3, 0.4, 0.33, 0.16, 0.16], dtype=np.float64)


def validate_action(action):
    arr = np.asarray(action, dtype=np.float64)
    if arr.shape != (ACTION_DIM,) or not np.isfinite(arr).all():
        raise ValueError(f"Expected {ACTION_DIM} finite SI joint targets, got {arr.shape}")
    if (arr < LOW - 1e-6).any() or (arr > HIGH + 1e-6).any():
        raise ValueError("Action outside documented mobile joint limits")
    return arr


class MobileEnv:
    """Low-level simulator. The policy receives *no* privileged object state."""

    def __init__(self, cubes=CUBE_COUNT):
        if not 1 <= cubes <= CUBE_COUNT:
            raise ValueError("cubes must be in [1,5]")
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.cubes = cubes
        self.qpos_addr = [int(self.model.joint(j).qposadr[0]) for j in JOINTS]
        self.cube_bodies = [self.model.body(f"red_{i}").id for i in range(CUBE_COUNT)]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for idx, value in zip(self.qpos_addr, HOME):
            self.data.qpos[idx] = value
        self.data.ctrl[:] = HOME
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        return np.array([self.data.qpos[i] for i in self.qpos_addr], dtype=np.float32)

    def cube_position(self, index):
        """Privileged simulator state for validation and the scripted teacher ONLY."""
        return np.asarray(self.data.xpos[self.cube_bodies[index]], dtype=np.float64).copy()

    def in_bin(self, index):
        x, y, z = self.cube_position(index)
        return bool(abs(x - BIN_CENTER[0]) < 0.17
                    and abs(y - BIN_CENTER[1]) < 0.17
                    and 0.025 < z < 0.15)

    def placed(self):
        return sum(self.in_bin(i) for i in range(self.cubes))

    def step(self, action, frames=10):
        """Execute 7 target positions at ~50Hz with continuous MuJoCo contacts.

        A non-oracle policy can call this without access to cube positions.
        Targets are rate-limited in simulator time so the mobile base cannot
        teleport between cubes. For VLA use, treat targets as absolute joints.
        """
        target = validate_action(action)
        for _ in range(frames):
            delta = target - self.data.ctrl
            self.data.ctrl[:] += np.clip(delta, -SLEW * self.model.opt.timestep,
                                         SLEW * self.model.opt.timestep)
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def render(self, camera="front", size=(224, 224), renderer=None):
        if camera not in (*CAMERAS, "overview"):
            raise ValueError("Unknown camera")
        owns = renderer is None
        if owns:
            renderer = mujoco.Renderer(self.model, height=size[0], width=size[1])
        try:
            renderer.update_scene(self.data, camera=camera)
            return np.asarray(renderer.render(), dtype=np.uint8).copy()
        finally:
            if owns:
                renderer.close()

    def observation(self, renderers):
        """Strict policy-facing RGB and proprioception; no cube pose keys."""
        images = {k: self.render(k, renderer=renderers[k]) for k in CAMERAS}
        return {"state": self.state(), "images": images, "task": TASK}


@dataclass
class Expert:
    """Oracle teacher used to debug mechanics and *collect* demonstration data.

    It deliberately sees ground-truth cube positions, so its success would
    NOT demonstrate autonomous perception or VLA generalization.
    """

    cube: int = 0
    phase: str = "navigate"
    entered: float = 0.0
    done: bool = False
    failure: str | None = None
    lifts: int = 0

    def action(self, env: MobileEnv):
        if self.done or self.failure:
            return env.data.ctrl.copy()
        t = float(env.data.time)
        elapsed = t - self.entered
        cube = env.cube_position(self.cube)
        current = env.state()
        target = current.astype(np.float64)
        # Fixed grip pose: cube is 48mm wide, gripper approaches along world X.
        target[2] = 0.0
        target[3] = 0.28
        target[4] = -0.18
        target[5:] = 0.0
        if self.phase in ("navigate", "lower", "close", "lift"):
            target[:2] = (cube[0] - 0.28, cube[1])
        else:
            target[:2] = (BIN_CENTER[0] - 0.28, BIN_CENTER[1])
        if self.phase == "navigate":
            if np.linalg.norm(current[:2] - target[:2]) < 0.022 and elapsed > 0.35:
                self.advance("lower", t)
        elif self.phase == "lower":
            target[4] = -0.465
            if abs(current[4] - target[4]) < 0.012 and elapsed > 0.5:
                self.advance("close", t)
        elif self.phase == "close":
            target[4] = -0.465
            target[5:] = 0.051
            if elapsed > 0.80:
                self.advance("lift", t)
        elif self.phase == "lift":
            target[5:] = 0.051
            if abs(current[4] - (-0.18)) < 0.02 and elapsed > 0.70:
                if env.cube_position(self.cube)[2] < 0.10:
                    self.failure = f"cube {self.cube}: gripper did not lift object"
                else:
                    self.lifts += 1
                    self.advance("transport", t)
        elif self.phase == "transport":
            target[5:] = 0.051
            if np.linalg.norm(current[:2] - target[:2]) < 0.025 and elapsed > 0.35:
                self.advance("release", t)
        elif self.phase == "release":
            if elapsed > 0.60:
                self.advance("settle", t)
        elif self.phase == "settle":
            if elapsed > 1.1:
                if not env.in_bin(self.cube):
                    self.failure = f"cube {self.cube}: release missed blue bin"
                elif self.cube == env.cubes - 1:
                    self.done = True
                else:
                    self.cube += 1
                    self.advance("navigate", t)
        return np.clip(target, LOW, HIGH)

    def advance(self, phase, t):
        self.phase = phase
        self.entered = t