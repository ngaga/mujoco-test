"""Paired SmolVLA SO-101 TCP-bias ablation with real MuJoCo contacts.

Compare 0, -5, +5mm fixed-jaw closing-axis bias on exactly matched
initial object poses and torch RNG seeds. No adaptation from test results.
The correction NEVER observes object pose: it uses only predicted robot
joints, pre-measured robot geometry and the fixed experimental offset.

  mjpython vla/eval_tcp_bias.py --check-env
  mjpython vla/eval_tcp_bias.py --object cube --episodes 5 --device mps
  mjpython vla/eval_tcp_bias.py --object rectangle --episodes 5 --device mps
"""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_contacts import (
    ContactRecorder, SIZE, annotated_frame, inspect_alignment, physical_control_action
)
from eval_soap import (
    MODEL_ID, SOAP_TASK, open_soap_environment, observation_to_batch,
    validate_action, device_for, load_model, CONTROL_HZ
)
from tcp_bias import corrected_targets, tcp_calibration

OFFSETS_MM = (0.0, -5.0, 5.0)


def check_environment(obj):
    import mujoco
    env = open_soap_environment(123, SIZE[obj])
    try:
        env.reset()
        ghost = mujoco.MjData(env.model)
        before = tcp_calibration(env.model, env.data)
        original = env.data.ctrl[:6].copy()
        checks = {}
        for bias in OFFSETS_MM:
            corrected, stats = corrected_targets(
                env.model, env.data, ghost, original, bias
            )
            checks[str(int(bias))] = {
                "nominal_joint_targets": original.round(5).tolist(),
                "corrected_joint_targets": corrected.round(5).tolist(),
                "shift": stats,
            }
            if not np.isclose(corrected[5], original[5]):
                raise RuntimeError("The bias changed jaw command")
        if checks["-5"]["shift"]["achieved_closing_mm"] >= 0:
            raise RuntimeError("Negative jaw-frame bias moved the wrong way")
        if checks["5"]["shift"]["achieved_closing_mm"] <= 0:
            raise RuntimeError("Positive jaw-frame bias moved the wrong way")
        return {
            "ok": True,
            "object": obj,
            "declared_tcp_vs_jaws": before,
            "jacobian_correction_probes": checks,
            "no_object_pose_used_to_correct_commands": True,
        }
    finally:
        env._renderer.close()


def run_episode(env, ghost, policy, pre, post, torch, device,
                ep, bias_mm, max_steps, gif_path):
    obs = env.reset()
    policy.reset()
    rec = ContactRecorder(env.model, env.data)
    calibration_at_spawn = tcp_calibration(env.model, env.data)
    initial = np.asarray(env.block_pos).round(6).tolist()
    frames = []
    success = False
    control_metrics = []
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            batch = observation_to_batch(obs, device, torch)
            batch["task"] = SOAP_TASK
            nominal = validate_action(
                post(policy.select_action(pre(batch))).detach().cpu().numpy(),
                env.model,
            )
        applied, stats = corrected_targets(env.model, env.data, ghost, nominal, bias_mm)
        obs = physical_control_action(env, applied, rec, step)
        if step <= 10 or step % 10 == 0 or rec.current_step_robot_contacts:
            control_metrics.append({
                "step": step,
                "nominal_joint_targets": nominal.round(5).tolist(),
                "applied_joint_targets": applied.round(5).tolist(),
                "bias": stats,
                "gripper_alignment": inspect_alignment(env.model, env.data)["gripper_frame_mm"],
            })
        if gif_path is not None and (step == 1 or step % 3 == 0):
            frame = annotated_frame(
                obs, inspect_alignment(env.model, env.data), rec, ep, step
            )
            # Existing real-camera GIF plus experimental label; no synthetic motion.
            from PIL import Image, ImageDraw
            im = Image.fromarray(frame)
            ImageDraw.Draw(im).text(
                (260, 3), f"BIAS {bias_mm:+g}mm",
                fill="#ffde66"
            )
            frames.append(np.asarray(im))
        if env.is_success():
            success = True
            break
    if gif_path is not None and frames:
        import imageio.v3 as iio
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(gif_path, np.stack(frames),
                    duration=3 / CONTROL_HZ, loop=0)
    return {
        "episode": ep,
        "success": bool(success),
        "steps": step,
        "initial_object_xyz": initial,
        "final_object_xyz": np.asarray(env.block_pos).round(6).tolist(),
        "first_contact_kind": rec.first_robot_contact["contact_kind"] if rec.first_robot_contact else "none",
        "first_robot_contact": rec.first_robot_contact,
        "first_fixed_pad_contact": rec.first_fixed_contact,
        "first_moving_pad_contact": rec.first_moving_contact,
        "contact_onset_counts": rec.totals,
        "calibration_at_spawn": calibration_at_spawn,
        "sampled_control_trace": control_metrics,
        "gif": gif_path.name if gif_path is not None else None,
    }


def report_md(report):
    lines = [
        f"# TCP jaw-axis bias ablation: {report['object']}",
        "",
        "Exactly the same pretrained SmolVLA model and input cameras.",
        "The offset is a bounded Jacobian correction of the VLA's motor",
        "setpoints, computed from robot geometry and joint angles only.",
        "**No ground-truth object position drives the controller.**",
        "Bias applies throughout the motion, including placement; it is",
        "not limited to grasp phase, which can itself cause new failures.",
        "",
        "| Bias (mm) | Success | First fixed contact | First moving contact | First other robot contact |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_bias = report["conditions"]
    for condition in by_bias:
        rows = condition["episodes"]
        counts = {name: sum(r["first_contact_kind"] == name for r in rows)
                  for name in ("fixed_pad", "moving_pad", "other_robot_geom")}
        lines.append(
            f"| {condition['closing_bias_mm']:+g} | "
            f"{condition['successes']}/{len(rows)} | "
            f"{counts['fixed_pad']} | {counts['moving_pad']} | "
            f"{counts['other_robot_geom']} |"
        )
    lines.extend([
        "", "## Detailed first-impact results", "",
        "| Bias | Episode | Success | First contact | Near-face gap (mm) | Lateral (mm) | GIF |",
        "|---:|---:|---|---|---:|---:|---|",
    ])
    for condition in by_bias:
        for ep in condition["episodes"]:
            first = ep["first_robot_contact"]
            gap = first["alignment"]["gripper_frame_mm"] if first else {}
            lines.append(
                f"| {condition['closing_bias_mm']:+g} | {ep['episode']} | "
                f"{'yes' if ep['success'] else 'no'} | "
                f"{ep['first_contact_kind']} | "
                f"{gap.get('fixed_pad_to_object_near_face_clearance', 'n/a')} | "
                f"{gap.get('object_from_fixed_perpendicular', 'n/a')} | "
                f"{'[GIF](' + condition['name'] + '/' + ep['gif'] + ')' if ep['gif'] else 'none'} |"
            )
    lines += [
        "",
        "## Experimental caveats",
        "",
        "- Same seeded spawn distribution and per-episode PyTorch random seed",
        "  across corrections. A success-rate difference on five episodes",
        "  is not enough to calibrate a production gripper.",
        "- The calibration reports the declared TCP to rigid fixed pad offset",
        "  in the moving Fixed_Jaw body frame, plus dynamic open-jaw midpoint.",
        "- A 5mm command is a desired Cartesian shift. The report logs the",
        "  actual shift achieved by damped Jacobian IK and joint clipping.",
        "- A correction that fails or changes placement does not prove the",
        "  camera is miscalibrated. First-contact geometry must be inspected.",
        "",
    ]
    return "\n".join(lines)


def evaluate(args):
    import mujoco
    import torch
    device = device_for(args.device, torch)
    # Load weights only once. Each condition creates a fresh identical
    # robot environment with same seed, preserving the spawn RNG sequence.
    policy, pre, post = load_model(args.repo, device)
    conditions = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for bias in args.biases:
        label = ("zero" if bias == 0 else
                 ("minus" if bias < 0 else "plus") + str(abs(int(bias))) + "mm")
        env = open_soap_environment(args.seed, SIZE[args.object])
        scratch = mujoco.MjData(env.model)
        results = []
        try:
            for ep in range(1, args.episodes + 1):
                # Pair both simulation initializations and stochastic policy RNG.
                torch.manual_seed(args.seed + ep)
                gif = args.output_dir / label / f"episode_{ep:02d}.gif" if ep <= 3 else None
                row = run_episode(env, scratch, policy, pre, post, torch,
                                  device, ep, bias, args.max_steps, gif)
                results.append(row)
                print(f"BIAS {bias:+g}mm {args.object} episode {ep}/{args.episodes}: "
                      f"success={row['success']} first={row['first_contact_kind']}", flush=True)
        finally:
            env._renderer.close()
        conditions.append({
            "name": label,
            "closing_bias_mm": bias,
            "successes": sum(int(e["success"]) for e in results),
            "episodes": results,
        })
    report = {
        "object": args.object, "checkpoint": args.repo,
        "instruction": SOAP_TASK, "seed": args.seed,
        "episodes_per_condition": args.episodes,
        "max_steps": args.max_steps, "device": device,
        "bias_in_robot_frame": "fixed jaw towards moving jaw (horizontal)",
        "correction_source": "robot kinematics and joint targets only (not object pose)",
        "conditions": conditions,
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(report_md(report))
    print(report_md(report), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=tuple(SIZE), default="cube")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--repo", default=MODEL_ID)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--biases", type=float, nargs="+", default=list(OFFSETS_MM))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("vla_runs/tcp_bias_ablation"))
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")
    if any(abs(x) > 12 for x in args.biases):
        parser.error("Each bias must be between -12 and +12mm")
    if len(set(args.biases)) != len(args.biases):
        parser.error("Duplicate bias values are not permitted")
    return args


def main():
    args = parse_args()
    if args.check_env:
        print(json.dumps(check_environment(args.object), indent=2))
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
