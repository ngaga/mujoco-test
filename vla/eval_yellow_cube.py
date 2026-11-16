"""Benchmark a yellow 24 mm cube ("soap") without VLA fine-tuning.

Only its MATERIAL COLOR and English task instruction differ from the red
24 mm reference block. Geometry, mass, contact/friction, motor action
space and two MuJoCo cameras stay identical. This tests visual/language
transfer without a geometry confound.

    mjpython vla/eval_yellow_cube.py --check-env
    mjpython vla/eval_yellow_cube.py --episodes 20 --device mps
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_soap import (MODEL_ID, SOAP_TASK, device_for, load_model,
                       open_soap_environment, check_environment,
                       run_episode, save_gif)

CUBE_SIZE = (0.024, 0.024, 0.024)
DEFAULT_OUTPUT = Path("vla_runs/yellow_cube_evaluation.json")
DEFAULT_REPORT = Path("vla_runs/yellow_cube_report.md")
DEFAULT_GIFS = Path("vla_runs/yellow_cube_gifs")


def markdown_report(report):
    results = report["results"]
    successes = [x for x in results if x["success"]]
    failures = [x["episode"] for x in results if not x["success"]]
    gifs = [
        f"- Episode {row['episode']} ({'success' if row['success'] else 'failure'}, "
        f"{row['steps']} actions): [view actual two-camera GIF]({row['gif']})"
        for row in results if row.get("gif")
    ]
    mean_steps = (f"{sum(x['steps'] for x in successes) / len(successes):.1f}"
                  if successes else "n/a")
    return "\n".join([
        "# Zero-shot transfer: yellow 24 mm cube",
        "",
        f"**Result: {report['successes']}/{report['episodes']} "
        f"({100 * report['success_rate']:.0f}%)**",
        "",
        "- Genuine pretrained SmolVLA SO-101 joint control, no retraining",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Instruction: `{report['task']}`",
        "- Robot/physics: original SO-101 MuJoCo reference workcell",
        "- Yellow cube: 24 × 24 × 24 mm, 20 g, original reference cube friction",
        "- Unchanged: geometry, mass, orientation initialization, position "
        "distribution, two cameras, arm/jaws, and success criterion",
        "- Changed together: object color (red -> yellow) and language "
        "(red block -> yellow soap)",
        f"- Seed: {report['seed']}, cap: {report['max_steps']} actions",
        f"- Average successful steps: {mean_steps}",
        f"- Failed episodes: {', '.join(map(str, failures)) if failures else 'none'}",
        "",
        "## Real MuJoCo GIFs (front camera | wrist camera)",
        *gifs,
        "",
        "## Comparison (prior independent baseline)",
        "",
        "- Red 24 mm cube / red-block instruction: **15/20 (75%)**",
        "- Yellow 36 × 24 × 16 mm mini-soap / yellow-soap instruction: "
        "**7/20 (35%)**",
        f"- Yellow 24 mm cube / yellow-soap instruction: "
        f"**{report['successes']}/{report['episodes']} "
        f"({100 * report['success_rate']:.0f}%)**",
        "",
        "Interpret cautiously: the cube baseline comes from a prior execution;"
        " all 20 trials use the same seed, but color and instruction are"
        " changed together in the new condition. This is not an isolated"
        " color-only experiment, nor evidence for full-size hotel soap.",
        "",
    ])


def evaluate(args):
    import torch
    device = device_for(args.device, torch)
    env = open_soap_environment(args.seed, object_size=CUBE_SIZE)
    try:
        # Guard the experimental control: the object must be identical
        # to the original cube in physical size and mass.
        geom = env.model.geom("block_geom")
        if not np.allclose(geom.size, (0.012, 0.012, 0.012)):
            raise RuntimeError("Cube geometry deviates from original reference")
        if not np.isclose(env.model.body("block").mass, 0.02, rtol=0.01):
            raise RuntimeError("Cube mass deviates from original reference")

        policy, pre, post = load_model(args.repo, device)
        results = []
        saved_success = False
        saved_failure = False
        for i in range(args.episodes):
            row, frames = run_episode(
                env, policy, pre, post, torch, device, i, args.max_steps, True
            )
            first_outcome = ((row["success"] and not saved_success)
                             or (not row["success"] and not saved_failure))
            if frames and (i == 0 or first_outcome):
                filename = (f"episode_{i+1:02d}_"
                            f"{'success' if row['success'] else 'failure'}.gif")
                path = args.gif_dir / filename
                save_gif(frames, path)
                row["gif"] = str(path.relative_to(args.markdown.parent))
            saved_success |= row["success"]
            saved_failure |= not row["success"]
            results.append(row)
            print(f"YELLOW CUBE episode {i+1}/{args.episodes}: "
                  f"success={row['success']} steps={row['steps']}", flush=True)

        count = sum(row["success"] for row in results)
        report = {
            "checkpoint": args.repo,
            "policy": "pretrained SmolVLA (zero-shot, no fine-tuning)",
            "task": SOAP_TASK,
            "object": {
                "type": "yellow cube",
                "size_m": list(CUBE_SIZE),
                "mass_kg": 0.02,
            },
            "changed": ["object RGBA", "language task"],
            "held_fixed": [
                "box size", "mass", "friction", "spawn distribution",
                "arm and joint setpoints", "cameras", "success test",
            ],
            "seed": args.seed,
            "max_steps": args.max_steps,
            "device": device,
            "episodes": args.episodes,
            "successes": count,
            "success_rate": count / args.episodes,
            "results": results,
            "success_definition": "MuJoCo object physically in blue tray and TCP retreated",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report) + "\n")
        print(f"YELLOW CUBE: {count}/{args.episodes} "
              f"({count/args.episodes:.1%}); report={args.markdown}", flush=True)
        return report
    finally:
        env._renderer.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--repo", default=MODEL_ID)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--gif-dir", type=Path, default=DEFAULT_GIFS)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    # A pair of relative paths is used for Markdown-local GIF links.
    if args.gif_dir.parent != args.markdown.parent:
        parser.error("--gif-dir and --markdown must share their parent directory")
    return args


def main():
    args = parse_args()
    if args.check_env:
        print(json.dumps(check_environment(args.seed, object_size=CUBE_SIZE),
                         indent=2), flush=True)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
