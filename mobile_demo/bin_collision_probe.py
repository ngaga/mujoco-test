"""Measure robot-body penetration into the blue bin during real MuJoCo rollouts.

Not a proximity sensor. This is a privileged physics diagnostic that reads
MuJoCo's actual geom contact manifold at every 2ms integration step.
No collisions are disabled or visual meshes swapped to fake a clean movie.

Usage:
  python -m mobile_demo.bin_collision_probe --cubes 1
  python -m mobile_demo.bin_collision_probe --cubes 5 --max-steps 5000
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np

from .so101 import ExpertSO101, MobileSO101Env


def monitored_pairs(model):
    """Identify ALL descendant robot geoms and every physical blue-bin geom."""
    base = model.body("mobile_base").id
    bin_id = model.body("blue_bin").id
    robot = set()
    bins = set()
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        if body == bin_id:
            bins.add(gid)
        ancestor = body
        while ancestor > 0:
            if ancestor == base:
                robot.add(gid)
                break
            ancestor = int(model.body_parentid[ancestor])
    return robot, bins


def rollout(cubes=1, max_steps=1000, max_retries=2):
    env = MobileSO101Env(cubes=cubes)
    expert = ExpertSO101(max_retries=max_retries)
    robot, bins = monitored_pairs(env.model)
    collisions = Counter()
    phases = Counter()
    severe = Counter()
    episodes = []
    worst = 0.0
    count_physics_steps = 0
    first_collision = None
    for step in range(max_steps):
        command = expert.action(env)
        for _ in range(round(1 / (15 * env.model.opt.timestep))):
            env.step(command, frames=1)
            count_physics_steps += 1
            contacts = env.data.contact
            for contact in contacts:
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if not ((g1 in robot and g2 in bins) or (g2 in robot and g1 in bins)):
                    continue
                robot_id, bin_id = (g1, g2) if g1 in robot else (g2, g1)
                robot_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, robot_id)
                bin_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, bin_id)
                if not robot_name:
                    robot_name = env.model.body(int(env.model.geom_bodyid[robot_id])).name
                key = f"{robot_name} | {bin_name}"
                depth = max(0.0, -float(contact.dist))
                worst = max(worst, depth)
                collisions[key] += 1
                phases[expert.phase] += 1
                if depth > 0.002:
                    severe[key] += 1
                if first_collision is None:
                    first_collision = {"time": round(float(env.data.time), 3),
                                       "phase": expert.phase, "geom_pair": key,
                                       "depth_m": round(depth, 6),
                                       "base_xy": env.state()[:2].round(4).tolist(),
                                       "tcp_xyz": env.tcp_position().round(4).tolist()}
                if depth > 0.002 and len(episodes) < 35:
                    episodes.append({"time": round(float(env.data.time), 3),
                                     "phase": expert.phase, "pair": key,
                                     "depth_m": round(depth, 6),
                                     "tcp_z": round(float(env.tcp_position()[2]), 4)})
        if expert.done or expert.failure:
            break
    return {
        "robot": "SO101 physical contact diagnostic, no visual geom modifications",
        "cubes": cubes, "success": bool(expert.done and env.placed() == cubes),
        "placed": env.placed(), "attempts": expert.attempts[:cubes],
        "retries": len(expert.retry_events), "failure": expert.failure,
        "physics_steps": count_physics_steps,
        "simulation_seconds": round(float(env.data.time), 3),
        "robot_bin_contact_samples": sum(collisions.values()),
        "robot_bin_contact_pairs": dict(collisions),
        "phase_contact_samples": dict(phases),
        "penetration_more_than_2mm_samples": sum(severe.values()),
        "severe_geom_pairs": dict(severe),
        "max_robot_bin_penetration_mm": round(worst * 1000, 4),
        "first_collision": first_collision,
        "severe_examples": episodes,
        "collision_definition": "Robot and blue bin collision geoms, not red cubes vs bin",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cubes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("mobile_runs/bin_collision_probe.json"))
    args = p.parse_args()
    result = rollout(args.cubes, args.max_steps, args.max_retries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "severe_examples"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
