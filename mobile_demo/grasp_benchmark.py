"""Paired MuJoCo comparison: original gripper, grip tuning, and a cube collar.

This is an oracle-controlled mechanical benchmark, NOT a VLA evaluation.
Variants are assembled from the unchanged mobile_demo/scene.xml at runtime;
default robot behaviour and the main scene are never overwritten.

Run:
    python -m mobile_demo.grasp_benchmark --check-models
    python -m mobile_demo.grasp_benchmark --trials 8 --output mobile_runs/grasp_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .core import CUBE_COUNT, Expert, MobileEnv, SCENE

VARIANTS = (
    "baseline",
    "friction_8",
    "force_24",
    "collar_only",
    "short_pads_only",
    "collar_short_pads",
)


def make_model(variant: str):
    """Apply exactly one named mechanical treatment to the stock MJCF."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    root = ET.parse(SCENE).getroot()
    if variant == "friction_8":
        for name in ("left_pad", "right_pad", "left_stop_front",
                     "right_stop_front", "left_stop_back", "right_stop_back"):
            geom = root.find(f".//geom[@name='{name}']")
            assert geom is not None, name
            geom.set("friction", "8.0 0.01 0.002")
    if variant == "force_24":
        for name in ("jaw_left", "jaw_right"):
            actuator = root.find(f".//position[@name='{name}']")
            assert actuator is not None, name
            actuator.set("forcerange", "-24 24")
    if variant in ("short_pads_only", "collar_short_pads"):
        # Stock pads reach z=58 mm at the close pose, ABOVE a cube-top collar.
        # Shorter pads end at z=42 mm, BELOW the 46 mm collar underside;
        # without this adjustment the rim cannot mechanically hook the jaws.
        for name in ("left_pad", "right_pad"):
            geom = root.find(f".//geom[@name='{name}']")
            assert geom is not None, name
            geom.set("size", "0.024 0.008 0.012")
    if variant in ("collar_only", "collar_short_pads"):
        for i in range(CUBE_COUNT):
            cube = root.find(f".//body[@name='red_{i}']")
            assert cube is not None, i
            main = cube.find(f"geom[@name='cube_geom_{i}']")
            assert main is not None
            # Preserve total mass at 12 grams. The lip overlaps 3 mm into
            # the cube top and sticks out 3 mm on each of four sides.
            main.set("mass", "0.011")
            ET.SubElement(cube, "geom", {
                "name": f"cube_collar_{i}",
                "type": "box",
                "pos": "0 0 0.023",
                "size": "0.027 0.027 0.002",
                "mass": "0.001",
                "friction": "2.4 0.01 0.002",
                "rgba": "0.75 0.06 0.08 1",
            })
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 7:
        raise RuntimeError(f"Unexpected action space: {model.nu} actuators")
    return model


def scenarios(seed: int, trials: int):
    """Identical independent starting configurations for every variant."""
    rng = np.random.default_rng(seed)
    return [(float(dx), float(dy), float(yaw)) for dx, dy, yaw in
            zip(rng.uniform(-0.020, 0.020, trials),
                rng.uniform(-0.020, 0.020, trials),
                rng.uniform(-0.12, 0.12, trials))]


def run_one(env, offsets, max_steps, max_retries):
    env.reset()
    dx, dy, yaw = offsets
    joint = env.model.joint("cube_0")
    adr = int(env.model.jnt_qposadr[joint.id])
    env.data.qpos[adr] += dx
    env.data.qpos[adr + 1] += dy
    env.data.qpos[adr + 3:adr + 7] = [
        math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    ]
    mujoco.mj_forward(env.model, env.data)
    teacher = Expert(max_retries=max_retries)
    frames = round(1 / (15 * env.model.opt.timestep))
    ever_lifted = False
    for step in range(max_steps):
        action = teacher.action(env)
        env.step(action, frames=frames)
        ever_lifted = ever_lifted or bool(env.cube_position(0)[2] > 0.10)
        if teacher.done or teacher.failure:
            break
    reasons = Counter(event["reason"] for event in teacher.retry_events)
    return {
        "success": bool(teacher.done and env.in_bin(0)),
        "ever_lifted_over_10cm": ever_lifted,
        "completed_steps": step + 1,
        "simulation_seconds": round(float(env.data.time), 2),
        "attempts": teacher.attempts_per_cube[0],
        "retries": teacher.retries,
        "failure": teacher.failure,
        "retry_reasons": dict(reasons),
        "final_cube_xyz": env.cube_position(0).round(4).tolist(),
        "start_offset_xy_yaw": [round(v, 4) for v in offsets],
    }


def benchmark(trials, seed, max_steps, max_retries, variants):
    setups = scenarios(seed, trials)
    report = {
        "description": "Oracle-controlled paired physics trial; not VLA",
        "robot": "7-actuator LeKiwi-inspired Cartesian proxy, not SO-101",
        "scene": str(SCENE),
        "seed": seed,
        "trials_per_variant": trials,
        "max_steps": max_steps,
        "max_retries": max_retries,
        "variants": {},
    }
    for variant in variants:
        env = MobileEnv(cubes=1)
        env.model = make_model(variant)
        env.data = mujoco.MjData(env.model)
        env.qpos_addr = [int(env.model.jnt_qposadr[env.model.joint(j).id])
                         for j in ("base_x", "base_y", "base_yaw", "reach",
                                   "lift", "left_jaw", "right_jaw")]
        env.cube_bodies = [env.model.body(f"red_{i}").id for i in range(CUBE_COUNT)]
        rows = [run_one(env, s, max_steps, max_retries) for s in setups]
        causes = Counter()
        for row in rows:
            causes.update(row["retry_reasons"])
        aggregate = {
            "successes": sum(row["success"] for row in rows),
            "lifted_at_least_once": sum(row["ever_lifted_over_10cm"] for row in rows),
            "total_retries": sum(row["retries"] for row in rows),
            "total_attempts": sum(row["attempts"] for row in rows),
            "retry_causes": dict(causes),
            "episodes": rows,
        }
        report["variants"][variant] = aggregate
        print(f"{variant}: successes={aggregate['successes']}/{trials}, "
              f"lifted={aggregate['lifted_at_least_once']}/{trials}, "
              f"retries={aggregate['total_retries']}, "
              f"retry_causes={dict(causes)}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--check-models", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("mobile_runs/grasp_benchmark.json"))
    args = parser.parse_args(argv)
    if args.trials < 1 or args.max_steps < 1 or args.max_retries < 0:
        parser.error("trials, max-steps must be positive; max-retries non-negative")
    if args.check_models:
        for name in VARIANTS:
            model = make_model(name)
            print(f"{name}: {model.ngeom} geoms, {model.nu} actuators", flush=True)
        return
    report = benchmark(args.trials, args.seed, args.max_steps,
                       args.max_retries, args.variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved paired MuJoCo results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
