"""Held-out 24-spawn SO-101 yellow rectangular soap experiment.

Paired conditions: baseline (0 mm), full trajectory (-5 mm) and *approach*
only (-5 mm, fades when the policy first closes the jaw after opening).
Pre-registered seed 2027 is new, distinct from the exploratory seed 123.
The same exact 24 spawn coordinates and per-episode torch seeds are used
in each condition; no pretrained weights are updated.

    mjpython vla/eval_phase_bias.py --check-env
    mjpython vla/eval_phase_bias.py --episodes 24 --seed 2027 --device mps
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from approach_gate import BiasSchedule
from diagnose_contacts import (ContactRecorder, annotated_frame,
                               inspect_alignment, physical_control_action)
from eval_soap import (MODEL_ID, SOAP_SIZE, SOAP_TASK, open_soap_environment,
                       observation_to_batch, validate_action, device_for,
                       load_model, CONTROL_HZ)
from tcp_bias import corrected_targets, tcp_calibration

MODES = BiasSchedule.MODES
DEFAULT_OUTPUT = Path("vla_runs/heldout_phase_bias")


def generate_spawns(seed, episodes):
    """Exact reference SO-101 reachable-XY distribution, independent of policy."""
    rng = np.random.default_rng(seed)
    spawns = []
    for i in range(episodes):
        r = rng.uniform(0.19, 0.27)
        theta = rng.uniform(np.deg2rad(-20), np.deg2rad(35))
        spawns.append([float(r * np.sin(theta)), float(-r * np.cos(theta))])
    if len({(round(x, 8), round(y, 8)) for x, y in spawns}) != episodes:
        raise RuntimeError("Spawn generator produced duplicate XY positions")
    return spawns


def _frame(obs, metrics, recorder, ep, step, mode, active_bias):
    from PIL import Image, ImageDraw
    result = Image.fromarray(annotated_frame(obs, metrics, recorder, ep, step))
    ImageDraw.Draw(result).text(
        (265, 2), f"{mode} {active_bias:+.2f}mm", fill="#ffe07d"
    )
    return np.asarray(result)


def _write_gif(path, frames):
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), duration=3.0 / CONTROL_HZ, loop=0)


def episode(env, scratch, policy, pre, post, torch, device,
            spawn_xy, ep_number, mode, max_steps, capture_frames):
    """Actual 2ms physics; the only difference is post-policy joint correction."""
    obs = env.reset(block_pos=np.asarray(spawn_xy, dtype=float))
    policy.reset()
    recorder = ContactRecorder(env.model, env.data)
    schedule = BiasSchedule(mode)
    original_position = np.asarray(env.block_pos).copy()
    initial_tcp = tcp_calibration(env.model, env.data)
    start = time.perf_counter()
    max_z = float(original_position[2])
    max_tilt = 0.0
    frames = []
    control_trace = []
    success = False
    first_contact_step = None
    max_contact_force_n = 0.0
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            batch = observation_to_batch(obs, device, torch)
            batch["task"] = SOAP_TASK
            nominal = validate_action(
                post(policy.select_action(pre(batch))).detach().cpu().numpy(),
                env.model
            )
        active_bias = schedule.next_bias(nominal[5])
        applied, bias_stats = corrected_targets(
            env.model, env.data, scratch, nominal, active_bias
        )
        obs = physical_control_action(env, applied, recorder, step)
        max_z = max(max_z, float(env.block_pos[2]))
        metrics = inspect_alignment(env.model, env.data)
        max_tilt = max(max_tilt, metrics["object_tilt_deg"])
        max_contact_force_n = max(max_contact_force_n,
                                  recorder.current_step_max_force_n)
        if first_contact_step is None and recorder.first_robot_contact is not None:
            first_contact_step = step
        if (step <= 5 or step % 15 == 0 or step == first_contact_step
                or step == schedule.transition_step):
            control_trace.append({
                "step": step,
                "jaw_target_rad": round(float(nominal[5]), 5),
                "active_bias_mm": round(active_bias, 3),
                "achieved_closing_mm": bias_stats["achieved_closing_mm"],
                "max_arm_joint_delta_rad": bias_stats["max_arm_joint_delta_rad"],
                "target_joints_rad": np.round(nominal, 5).tolist(),
                "applied_joints_rad": np.round(applied, 5).tolist(),
                "first_contact_observed": recorder.first_robot_contact is not None,
            })
        if capture_frames and (step == 1 or step % 3 == 0):
            frames.append(_frame(obs, metrics, recorder, ep_number,
                                 step, mode, active_bias))
        if env.is_success():
            success = True
            break
    row = {
        "episode": ep_number,
        "initial_xy_m": [round(float(v), 7) for v in spawn_xy],
        "success": bool(success),
        "steps": step,
        "max_object_height_m": round(max_z, 5),
        "lifted_5cm_above_initial": bool(max_z > original_position[2] + .05),
        "max_object_tilt_deg": round(max_tilt, 2),
        "end_xyz_m": np.round(env.block_pos, 6).tolist(),
        "max_contact_force_n": round(max_contact_force_n, 3),
        "first_robot_contact": recorder.first_robot_contact,
        "first_contact_kind": (recorder.first_robot_contact["contact_kind"]
                               if recorder.first_robot_contact else "none"),
        "contact_onset_counts": recorder.totals,
        "schedule": schedule.summary(),
        "initial_tcp_vs_fingers": initial_tcp,
        "control_trace": control_trace,
        "wall_seconds": round(time.perf_counter() - start, 2),
        "gif": None,
    }
    return row, frames


def _paired(report):
    reference = report["conditions"][0]["results"]
    base = [bool(r["success"]) for r in reference]
    comparisons = []
    for condition in report["conditions"][1:]:
        trials = condition["results"]
        if [r["initial_xy_m"] for r in trials] != [
                r["initial_xy_m"] for r in reference]:
            raise AssertionError("Conditions did not use exactly paired spawn XY")
        helped = [t["episode"] for b, t in zip(base, trials) if not b and t["success"]]
        hurt = [t["episode"] for b, t in zip(base, trials) if b and not t["success"]]
        comparisons.append({
            "condition": condition["mode"],
            "baseline_failed_corrected_succeeded": helped,
            "baseline_succeeded_corrected_failed": hurt,
            "net_improved_episodes": len(helped) - len(hurt),
        })
    return comparisons


def report_markdown(report):
    n = report["episodes_per_condition"]
    lines = [
        "# Held-out yellow rectangular soap: phase-gated grasp bias",
        "",
        f"Pre-registered **{n} new XY positions** from seed {report['seed']}, "
        "not the exploratory seed 123. All are paired across three policies.",
        "The checkpoint, cameras and object are unchanged; no model fine-tuning.",
        "The corrective offset is computed from *robot joints + jaw geometry only*.",
        "",
        "| Condition | Success | Lift >=5 cm | First fixed hit | First moving hit | Other robot hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in report["conditions"]:
        trials = c["results"]
        cnt = lambda side: sum(x["first_contact_kind"] == side for x in trials)
        lines.append(
            f"| {c['mode']} | {c['successes']}/{n} "
            f"({100*c['successes']/n:.1f}%) | "
            f"{sum(x['lifted_5cm_above_initial'] for x in trials)}/{n} | "
            f"{cnt('fixed_pad')} | {cnt('moving_pad')} | "
            f"{cnt('other_robot_geom')} |"
        )
    lines += ["", "## Paired changes versus baseline", ""]
    for comparison in report["paired_comparisons"]:
        lines += [
            f"**{comparison['condition']}**: improved episodes "
            f"{comparison['baseline_failed_corrected_succeeded']}; "
            f"worsened episodes "
            f"{comparison['baseline_succeeded_corrected_failed']}; "
            f"net {comparison['net_improved_episodes']:+d}.",
            "",
        ]
    lines += [
        "## Sample GIFs (real front and wrist cameras with jaw-offset overlay)",
        "",
    ]
    for c in report["conditions"]:
        for row in c["results"]:
            if row["gif"]:
                lines.append(
                    f"- {c['mode']}, episode {row['episode']}, "
                    f"{'success' if row['success'] else 'failure'}: "
                    f"[GIF]({row['gif']})"
                )
    lines += [
        "",
        "## What this study can and cannot establish",
        "",
        "- All 24 spawn XY coordinates are stored in JSON and shared across variants.",
        "- The approach gate watches only the **VLA jaw motor setpoint**: "
        "once opened >=0.85 rad, first closing <=0.60 rad fades the "
        "-5 mm bias over five motor steps. If this does not happen by step 120, "
        "a fixed horizon triggers the fade. No contact or object pose is "
        "used to decide the gate.",
        "- Full-trajectory bias remains active during final placement, "
        "which may create new end-to-end failures even if pickup improves.",
        "- Success means physically inside the blue tray with tool retreated. "
        "Lift alone is not proof of stable grasp or final placement.",
        "- Sample size is moderate, and adjusting the algorithm after seeing "
        "these results would require a new independent holdout for an honest claim.",
        "",
    ]
    return "\n".join(lines)


def check_env(seed):
    import mujoco
    env = open_soap_environment(seed, SOAP_SIZE)
    try:
        env.reset(block_pos=generate_spawns(seed, 1)[0])
        scratch = mujoco.MjData(env.model)
        home = env.data.ctrl[:6].copy()
        nominal, original = corrected_targets(
            env.model, env.data, scratch, home, 0
        )
        test, shifted = corrected_targets(
            env.model, env.data, scratch, home, -5
        )
        if not np.array_equal(nominal, home):
            raise AssertionError("Baseline must be bit-for-bit original")
        if shifted["achieved_closing_mm"] >= 0:
            raise AssertionError("Negative correction moved wrong way")
        if not np.isclose(test[5], home[5]):
            raise AssertionError("The jaw motor command changed")
        g = BiasSchedule("approach")
        values = [g.next_bias(v) for v in
                  (0.1, 1.3, 1.3, 0.5, 0.4, 0.3, 0.2, 0.1)]
        if values[0] != -5 or values[-1] != 0:
            raise AssertionError("Approach gate failed its expected decay")
        return {
            "ok": True,
            "robot": "SO-101",
            "object_size_m": list(SOAP_SIZE),
            "new_spawn_seed": seed,
            "nominal_unchanged": True,
            "achieved_negative_bias_mm": shifted["achieved_closing_mm"],
            "unchanged_jaw_motor": True,
            "approach_gate_check": values,
            "tcp_calibration": tcp_calibration(env.model, env.data),
        }
    finally:
        env._renderer.close()


def evaluate(args):
    import mujoco
    import torch
    spawns = generate_spawns(args.seed, args.episodes)
    device = device_for(args.device, torch)
    policy, pre, post = load_model(args.repo, device)
    conditions = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        env = open_soap_environment(args.seed, SOAP_SIZE)
        scratch = mujoco.MjData(env.model)
        rows = []
        captured = {True: False, False: False}
        try:
            for index, spawn in enumerate(spawns, 1):
                torch.manual_seed(args.seed * 100 + index)
                # Collect until one success AND one failure for each mode.
                row, frames = episode(
                    env, scratch, policy, pre, post, torch, device,
                    spawn, index, mode, args.max_steps,
                    not all(captured.values()),
                )
                if frames and not captured[row["success"]]:
                    filename = f"{mode}/episode_{index:02d}_{'success' if row['success'] else 'failure'}.gif"
                    _write_gif(args.output_dir / filename, frames)
                    row["gif"] = filename
                    captured[row["success"]] = True
                rows.append(row)
                print(
                    f"HOLDOUT {mode} {index}/{len(spawns)}: "
                    f"success={row['success']} first={row['first_contact_kind']} "
                    f"gate={row['schedule']['transition_step']}", flush=True,
                )
        finally:
            env._renderer.close()
        conditions.append({
            "mode": mode,
            "successes": sum(int(row["success"]) for row in rows),
            "lifted_count": sum(int(row["lifted_5cm_above_initial"]) for row in rows),
            "results": rows,
        })

    report = {
        "experiment": "pre-registered paired holdout after exploratory +/-5mm sweep",
        "object": "yellow 36x24x16mm rectangular mini-soap",
        "object_size_m": list(SOAP_SIZE),
        "seed": args.seed,
        "episodes_per_condition": args.episodes,
        "max_steps": args.max_steps,
        "checkpoint": args.repo,
        "task": SOAP_TASK,
        "device": device,
        "spawn_xy_m": spawns,
        "no_training": True,
        "control_uses_object_ground_truth": False,
        "conditions": conditions,
    }
    report["paired_comparisons"] = _paired(report)
    path = args.output_dir / "evaluation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(report_markdown(report) + "\n")
    print("HOLDOUT FINAL RESULTS", flush=True)
    for c in conditions:
        print(f"{c['mode']}: {c['successes']}/{len(spawns)} "
              f"(lifted {c['lifted_count']}/{len(spawns)})", flush=True)
    print(f"Report: {args.output_dir / 'report.md'}", flush=True)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--max-steps", type=int, default=240)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                   default="auto")
    p.add_argument("--repo", default=MODEL_ID)
    p.add_argument("--check-env", action="store_true")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1:
        p.error("--episodes and --max-steps must be positive")
    return args


def main():
    args = parse_args()
    if args.check_env:
        print(json.dumps(check_env(args.seed), indent=2))
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
