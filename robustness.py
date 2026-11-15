"""Reproducible paired MuJoCo benchmark for physical soap restocking.

Run two *identical* seeded scenarios with and without automatic recovery.
The first pose is varied at reset (valid domain randomization); the soap is
NEVER moved kinematically during grasp, carrying, dropping or replanning.
This headless benchmark uses MuJoCo pose truth to choose approach XY, NOT RGB.
RGB robustness requires separate, slower camera-rendered experiments.
"""
import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco

from motion import HOME
from physical_grasp import ContactMonitor, PhysicalMotion, step_physical
from recovery import RecoveryManager

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "bathroom_physical.xml"
# Narrow, physically accessible placement range. The soap stays entirely
# supported by the counter and away from the amenities tray.
SOAP_X = (-0.32, -0.22)
SOAP_Y = (-0.045, 0.055)
DEFAULT_SEED = 2026
DEFAULT_TRIALS = 100
STEPS_LIMIT = 18000


@dataclass(frozen=True)
class Scenario:
    index: int
    x: float
    y: float
    injected_drop: bool

    def as_dict(self):
        return {"index": self.index, "x": round(self.x, 5),
                "y": round(self.y, 5), "injected_drop": self.injected_drop}


def make_scenarios(count=DEFAULT_TRIALS, seed=DEFAULT_SEED):
    """Exactly half forced drops for even N, randomized independently of XY."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    scenarios = [
        Scenario(i, rng.uniform(*SOAP_X), rng.uniform(*SOAP_Y), False)
        for i in range(count)
    ]
    faults = set(rng.sample(range(count), count // 2))
    return [
        Scenario(s.index, s.x, s.y, s.index in faults)
        for s in scenarios
    ]


def in_tray(xyz):
    x, y, z = xyz
    return 0.075 < x < 0.445 and -0.12 < y < 0.22 and 0.82 < z < 0.92


def run_trial(model, scenario, recovery_enabled=True, max_steps=STEPS_LIMIT):
    """Return observed task outcome, not an assumed completion from the script."""
    data = mujoco.MjData(model)
    motion = PhysicalMotion()
    monitor = ContactMonitor(model)
    manager = RecoveryManager(max_attempts=2 if recovery_enabled else 1)

    soap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "soap_free")
    if soap_id < 0:
        raise ValueError("Missing physical soap joint")
    qadr = int(model.jnt_qposadr[soap_id])
    vadr = int(model.jnt_dofadr[soap_id])
    mujoco.mj_resetData(model, data)
    data.qpos[qadr:qadr + 2] = [scenario.x, scenario.y]
    data.mocap_pos[0] = HOME
    mujoco.mj_forward(model, data)

    # Use the same privileged XY for the first planned approach in BOTH
    # configurations: it isolates recovery rather than measuring perception.
    motion.retry_from(HOME, (scenario.x, scenario.y))
    manager.start()
    injected = False
    last_pose_before_replan = None
    last_pose_after_replan = None
    for tick in range(max_steps):
        if (scenario.injected_drop and not injected
                and manager.attempt == 1
                and motion.index == 4 and motion.elapsed >= 0.25):
            injected = True
        forced_open = (scenario.injected_drop and injected
                       and manager.attempt == 1
                       and manager.status == "running")
        step_physical(model, data, motion, force_open=forced_open)
        xyz = tuple(float(v) for v in data.qpos[qadr:qadr + 3])
        linear_velocity = tuple(float(v) for v in data.qvel[vadr:vadr + 3])
        left, right = monitor.contacts(data)
        wants_observation = manager.after_step(
            motion=motion,
            soap_xyz=xyz,
            linear_velocity=linear_velocity,
            both_contacts=left and right,
            in_tray=in_tray(xyz),
            dt=float(model.opt.timestep),
        )
        if wants_observation:
            # The benchmark intentionally emulates the nonvisual mode.
            last_pose_before_replan = tuple(float(v) for v in data.qpos[qadr:qadr + 7])
            manager.restart_at(
                motion=motion,
                palm_xyz=tuple(float(v) for v in data.mocap_pos[0]),
                soap_xy=xyz[:2],
            )
            last_pose_after_replan = tuple(float(v) for v in data.qpos[qadr:qadr + 7])
            if last_pose_before_replan != last_pose_after_replan:
                raise AssertionError("Recovery must never teleport the free soap")
        if manager.status in ("success", "failed"):
            if manager.attempt > manager.max_attempts:
                raise AssertionError("Exceeded maximum regrasp count")
            if manager.status == "success" and not (manager.grasp_verified and in_tray(xyz)):
                raise AssertionError("False success: soap lacks verified lift or tray placement")
            return {
                **scenario.as_dict(),
                "recovery_enabled": bool(recovery_enabled),
                "status": manager.status,
                "attempts": manager.attempt,
                "fault_triggered": injected,
                "grasp_verified": bool(manager.grasp_verified),
                "final_xyz": [round(v, 5) for v in xyz],
                "reason": manager.last_error,
                "sim_seconds": round((tick + 1) * float(model.opt.timestep), 3),
                "replanned": last_pose_before_replan is not None,
            }
    return {
        **scenario.as_dict(),
        "recovery_enabled": bool(recovery_enabled),
        "status": "timeout",
        "attempts": manager.attempt,
        "fault_triggered": injected,
        "grasp_verified": bool(manager.grasp_verified),
        "final_xyz": [round(float(v), 5) for v in data.qpos[qadr:qadr + 3]],
        "reason": "Exceeded step budget",
        "sim_seconds": round(max_steps * float(model.opt.timestep), 3),
        "replanned": last_pose_before_replan is not None,
    }


def summarize(rows):
    n = len(rows)
    successes = sum(r["status"] == "success" for r in rows)
    faults = [r for r in rows if r["injected_drop"]]
    clean = [r for r in rows if not r["injected_drop"]]
    errors = Counter((r["reason"] or r["status"]) for r in rows if r["status"] != "success")
    return {
        "runs": n,
        "successes": successes,
        "success_rate_pct": round(100 * successes / n, 1) if n else 0,
        "clean_successes": sum(r["status"] == "success" for r in clean),
        "clean_runs": len(clean),
        "fault_successes": sum(r["status"] == "success" for r in faults),
        "fault_runs": len(faults),
        "second_attempt_successes": sum(
            r["status"] == "success" and r["attempts"] == 2 for r in rows
        ),
        "failure_reasons": dict(sorted(errors.items())),
    }


def benchmark(count=DEFAULT_TRIALS, seed=DEFAULT_SEED, model=None):
    """Paired, deterministic 100-placement evaluation for both policies."""
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    scenarios = make_scenarios(count, seed)
    baseline, recovery = [], []
    for scenario in scenarios:
        baseline.append(run_trial(model, scenario, recovery_enabled=False))
        recovery.append(run_trial(model, scenario, recovery_enabled=True))
    by_index = {r["index"]: r for r in baseline}
    rescued = [r["index"] for r in recovery
               if r["status"] == "success" and by_index[r["index"]]["status"] != "success"]
    regressed = [r["index"] for r in recovery
                 if r["status"] != "success" and by_index[r["index"]]["status"] == "success"]
    return {
        "protocol": "paired seeded free-soap MuJoCo grasp, GT XY, half forced jaw release",
        "seed": seed,
        "positions": count,
        "x_range_m": list(SOAP_X),
        "y_range_m": list(SOAP_Y),
        "baseline": summarize(baseline),
        "with_recovery": summarize(recovery),
        "rescued_indices": rescued,
        "regressed_indices": regressed,
        "scenarios": [s.as_dict() for s in scenarios],
        "baseline_results": baseline,
        "recovery_results": recovery,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "robustness_report.json")
    args = parser.parse_args()
    report = benchmark(args.trials, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    a, b = report["baseline"], report["with_recovery"]
    print(f"100-position benchmark: N={args.trials}, seed={args.seed}")
    print(f"No recovery: {a['successes']}/{a['runs']} ({a['success_rate_pct']}%)")
    print(f"With recovery: {b['successes']}/{b['runs']} ({b['success_rate_pct']}%)")
    print(f"Forced-drop recovery: {b['fault_successes']}/{b['fault_runs']}")
    print(f"Rescued: {len(report['rescued_indices'])}; regressed: {len(report['regressed_indices'])}")
    print(f"Full paired results: {args.output}")


if __name__ == "__main__":
    main()
