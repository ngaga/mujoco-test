"""Zero-shot yellow rectangular soap transfer in the real SO-101 MuJoCo cell.

This deliberately keeps the pretrained SmolVLA checkpoint and articulated
SO-101 observation/action contract unchanged. The reference block is replaced
by a 36x24x16 mm yellow mini-soap (20 g). No training and no privileged block
state enters the policy.

Example:
    mjpython vla/eval_soap.py --check-env
    mjpython vla/eval_soap.py --episodes 20 --device mps
"""
import argparse
import json
import math
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from run_so101 import (CAMERA_MAP, CONTROL_HZ, MAX_STEPS, MODEL_ID,
                       device_for, load_model, observation_to_batch,
                       validate_action, validate_observation)

SOAP_SIZE = (0.036, 0.024, 0.016)  # metres, a *mini* soap fitting SO-101's jaws
SOAP_HALF = tuple(d / 2 for d in SOAP_SIZE)
SOAP_MASS = 0.020  # kg
SOAP_RGBA = "0.96 0.78 0.10 1"
SOAP_TASK = "Pick up the yellow soap and place it in the blue tray."
DEFAULT_OUTPUT = Path("vla_runs/soap_zero_shot.json")
DEFAULT_MARKDOWN = Path("vla_runs/soap_report.md")
DEFAULT_GIFS = Path("vla_runs/soap_gifs")


def make_soap_xml(source_xml: str, object_size=SOAP_SIZE) -> str:
    """Change only the object shape/color in the pinned upstream cell."""
    root = ET.fromstring(source_xml)
    geom = root.find(".//body[@name='block']/geom[@name='block_geom']")
    material = root.find(".//material[@name='block_red']")
    if geom is None or material is None:
        raise ValueError("Pinned sim2cell pickplace.xml no longer matches its block contract")
    if geom.get("type") != "box":
        raise ValueError("Expected a physical MuJoCo box geom, not a visual-only replacement")
    geom.set("size", " ".join(f"{x/2:.6f}" for x in object_size))
    geom.set("mass", str(SOAP_MASS))
    material.set("rgba", SOAP_RGBA)
    # Keep the authored original friction, solver settings, articulated arm,
    # cameras, actuators, tray and light unchanged.
    return ET.tostring(root, encoding="unicode")


def open_soap_environment(seed: int, object_size=SOAP_SIZE):
    """Compile physically different MJCF beside its includes/meshes, then clean up.

    Temporary XML lives next to the upstream scene so its relative <include>
    resolves. The installed third-party XML remains untouched after the call.
    """
    import mujoco
    from sim2cell.env import PickPlaceEnv, SCENE_XML

    scene_path = Path(SCENE_XML)
    xml = make_soap_xml(scene_path.read_text(), object_size=object_size)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", prefix=".earlybots_soap_",
                                         dir=scene_path.parent, encoding="utf-8",
                                         delete=False) as file:
            temp_path = Path(file.name)
            file.write(xml)
        model = mujoco.MjModel.from_xml_path(str(temp_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    class SoapEnv(PickPlaceEnv):
        def __init__(self):
            # Same setup as the *pinned* upstream PickPlaceEnv, except for
            # loading our modified geometry from a freshly compiled MJCF.
            self.model = model
            self.data = mujoco.MjData(model)
            self.rng = np.random.default_rng(seed)
            self.cameras = tuple(CAMERA_MAP)
            self.control_hz = CONTROL_HZ
            self.n_substeps = max(1, round(1 / (CONTROL_HZ * model.opt.timestep)))
            self._renderer = mujoco.Renderer(model, height=224, width=224)
            self._home_key = model.key("home")
            self._block_qadr = model.jnt_qposadr[model.joint("block_free").id]
            self.randomizer = None

        def reset(self, block_pos=None):
            # Upstream reset uses its fixed BLOCK_HALF=0.012 as z. The new
            # soap half-height is 0.008: correct it BEFORE policy observation.
            super().reset(block_pos=block_pos)
            self.data.qpos[self._block_qadr + 2] = object_size[2] / 2
            mujoco.mj_forward(self.model, self.data)
            return self.get_obs()

    return SoapEnv()


def check_environment(seed=123, object_size=SOAP_SIZE):
    env = open_soap_environment(seed, object_size=object_size)
    try:
        state, images = validate_observation(env.reset())
        import mujoco
        geom = env.model.geom("block_geom")
        assert env.model.nu == 6
        assert np.allclose(geom.size, np.asarray(object_size) / 2)
        assert np.allclose(env.data.qpos[env._block_qadr + 2], object_size[2] / 2)
        assert np.allclose(env.model.mat("block_red").rgba[:3],
                           [float(x) for x in SOAP_RGBA.split()[:3]])
        validate_observation(env.step(env.data.ctrl[:6].copy()))
        return {
            "ok": True, "object": ("yellow cube" if len(set(object_size)) == 1
                                     else "yellow rectangular mini-soap"),
            "dimensions_m": object_size, "mass_kg": SOAP_MASS,
            "arm_actuators": int(env.model.nu),
            "joint_shape": list(state.shape),
            "cameras": {name: list(image.shape) for name, image in images.items()},
            "simulation_timestep": float(env.model.opt.timestep),
            "model_loaded": False,
        }
    finally:
        env._renderer.close()


def run_episode(env, policy, pre, post, torch, device, index, max_steps, record_gif):
    obs = env.reset()
    policy.reset()
    start = time.perf_counter()
    initial = env.block_pos.copy()
    max_z = float(initial[2])
    frames = []
    success = False
    steps = 0
    for step in range(max_steps):
        with torch.no_grad():
            batch = observation_to_batch(obs, device, torch)
            batch["task"] = SOAP_TASK  # actual unseen language prompt
            action = post(policy.select_action(pre(batch)))
        targets = validate_action(action.detach().cpu().numpy(), env.model)
        obs = env.step(targets)
        steps = step + 1
        max_z = max(max_z, float(env.block_pos[2]))
        if record_gif and step % 3 == 0:
            # Two *real* rendered RGB camera feeds, not an illustrative GIF.
            frames.append(np.concatenate(
                (obs["images"]["front"], obs["images"]["wrist"]), axis=1
            ))
        if env.is_success():
            success = True
            break

    return {
        "episode": index + 1, "success": bool(success), "steps": steps,
        "initial_xyz": [round(float(x), 5) for x in initial],
        "final_xyz": [round(float(x), 5) for x in env.block_pos],
        "max_object_height_m": round(max_z, 5),
        "tcp_final_xyz": [round(float(x), 5) for x in env.tcp_pos],
        "wall_seconds": round(time.perf_counter() - start, 2),
    }, frames


def save_gif(frames, path):
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each GIF contains both live cameras, at 5 sampled frames per second.
    iio.imwrite(path, np.stack(frames), duration=3 / CONTROL_HZ, loop=0)


def report_markdown(report):
    rows = report["results"]
    success_steps = [x["steps"] for x in rows if x["success"]]
    failures = [x["episode"] for x in rows if not x["success"]]
    gif_lines = [
        f"- Episode {x['episode']}: [{'success' if x['success'] else 'failure'}]"
        f"({x['gif']})" for x in rows if x.get("gif")
    ]
    return "\n".join([
        "# SmolVLA zero-shot: yellow rectangular mini-soap",
        "",
        f"- Checkpoint: `{report['checkpoint']}` (unchanged, no fine-tuning)",
        f"- Actual prompt: `{SOAP_TASK}`",
        "- Robot: SO-101 six-joint MuJoCo arm with front and wrist RGB",
        "- Soap: 36 × 24 × 16 mm, yellow, 20 g; same gripping friction as reference cube",
        "- Reference task: red 24 × 24 × 24 mm cube into blue tray",
        "- Criterion: soap in tray **and** gripper retreated, from actual physics",
        f"- Seed: {report['seed']}; maximum {report['max_steps']} actions per episode",
        "",
        f"## Result: {report['successes']}/{report['episodes']} ({100 * report['success_rate']:.0f}%)",
        "",
        f"Mean successful episode length: "
        f"{sum(success_steps)/len(success_steps):.1f} steps."
        if success_steps else "No successful episodes.",
        f"Failed episodes: {', '.join(str(x) for x in failures) if failures else 'none'}.",
        "",
        "## Actual RGB camera GIFs (front | wrist)",
        *gif_lines,
        "",
        "The GIFs record genuine rendered SO-101 joint-control rollouts, not"
        " scripted or kinematic pick-and-place. These results do not establish"
        " transfer to a full-size hotel soap or to the EarlyBots floating-palm robot.",
        "",
    ])


def evaluate(args):
    import torch
    device = device_for(args.device, torch)
    env = open_soap_environment(args.seed)
    try:
        policy, pre, post = load_model(args.repo, device)
        results = []
        first_success_gif = False
        first_failure_gif = False
        for i in range(args.episodes):
            row, frames = run_episode(env, policy, pre, post, torch, device, i,
                                      args.max_steps, bool(args.gif_dir))
            # Always keep episode 1 plus first success and first failure;
            # representative GIFs rather than a potentially huge 20-GIF zip.
            first_of_outcome = (
                (row["success"] and not first_success_gif)
                or (not row["success"] and not first_failure_gif)
            )
            if args.gif_dir and frames and (i == 0 or first_of_outcome):
                gif_path = args.gif_dir / (
                    f"episode_{i+1:02d}_{'success' if row['success'] else 'failure'}.gif"
                )
                save_gif(frames, gif_path)
                row["gif"] = str(gif_path)
            if row["success"]:
                first_success_gif = True
            else:
                first_failure_gif = True
            results.append(row)
            print(f"SOAP zero-shot episode {i + 1}/{args.episodes}: "
                  f"success={row['success']} steps={row['steps']}", flush=True)

        successes = sum(x["success"] for x in results)
        report = {
            "checkpoint": args.repo, "zero_shot": True, "fine_tuned": False,
            "task": SOAP_TASK, "environment": "sim2cell SO-101 MuJoCo",
            "object": {"type": "yellow rectangular mini-soap",
                       "size_m": SOAP_SIZE, "mass_kg": SOAP_MASS},
            "training_reference": "red 24mm cube", "seed": args.seed,
            "max_steps": args.max_steps, "device": device,
            "episodes": args.episodes, "successes": successes,
            "success_rate": successes / args.episodes, "results": results,
            "success_definition": "MuJoCo soap in blue tray and TCP retreated",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report_markdown(report))
        print(f"SOAP zero-shot: {successes}/{args.episodes}; report: "
              f"{args.markdown}; JSON: {args.output}", flush=True)
        return report
    finally:
        env._renderer.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--repo", default=MODEL_ID)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--gif-dir", type=Path, default=DEFAULT_GIFS)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    return args


def main():
    args = parse_args()
    if args.check_env:
        print(json.dumps(check_environment(args.seed), indent=2), flush=True)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
