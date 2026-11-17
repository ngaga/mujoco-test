"""Mobile cube-to-blue-bin demo, and strict compatibility gate for real VLA.

Examples:
    python -m mobile_demo.run --check-model
    python -m mobile_demo.run --cubes 1 --smoke-test
    mjpython -m mobile_demo.run --cubes 5 --viewer
    MUJOCO_GL=osmesa python -m mobile_demo.run --cubes 1 --gif mobile.gif

The default `expert` controller is oracle/scripted, NOT VLA. `smolvla`
requires a checkpoint TRAINED for this 7D action/state/camera convention.
The existing 6D fixed SO101 model cannot directly drive the mobile prototype.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .core import ACTION_DIM, CAMERAS, Expert, MobileEnv, SCENE, TASK


def make_smolvla(checkpoint, device):
    """Load a compatible *fine-tuned* model. Never silently pad 6D outputs."""
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    def dim(features, name):
        feature = features.get(name)
        return None if feature is None else list(feature.shape)[-1]

    state_dim = dim(policy.config.input_features, "observation.state")
    action_dim = dim(policy.config.output_features, "action")
    if state_dim != ACTION_DIM or action_dim != ACTION_DIM:
        raise ValueError(
            f"Incompatible checkpoint: input state={state_dim}, action={action_dim}; "
            f"mobile scene needs BOTH {ACTION_DIM}. The fixed SO-101 checkpoint "
            "does not navigate and must not be padded/truncated."
        )
    pre, post = make_pre_post_processors(policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post, torch


def predict_smolvla(components, obs, device):
    policy, pre, post, torch = components
    batch = {"observation.state": torch.as_tensor(obs["state"]).unsqueeze(0).to(device),
             "task": obs["task"]}
    for src, dst in zip(CAMERAS, ("camera1", "camera2")):
        rgb = obs["images"][src]
        if rgb.ndim != 3 or rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Expected valid 224x224 RGB camera: {src}")
        batch[f"observation.images.{dst}"] = (
            torch.as_tensor(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.)
    with torch.no_grad():
        result = post(policy.select_action(pre(batch)))
    out = np.asarray(result.detach().cpu().numpy(), dtype=float).reshape(-1)
    if out.shape != (ACTION_DIM,):
        raise ValueError(f"Invalid VLA action dimension: {out.shape}")
    return out


def execute(args):
    env = MobileEnv(cubes=args.cubes)
    expert = Expert() if args.policy == "expert" else None
    components = None
    renderers = None
    frames_for_gif = []
    dataset = {"state": [], "action": [], "front": [], "wrist": []}
    if args.policy == "smolvla":
        if not args.checkpoint:
            raise ValueError("--policy smolvla requires --checkpoint with mobile 7D weights")
        components = make_smolvla(args.checkpoint, args.device)
        components[0].reset()
    if args.policy == "smolvla" or args.record:
        renderers = {c: mujoco.Renderer(env.model, height=224, width=224)
                     for c in CAMERAS}
    gif_renderer = (mujoco.Renderer(env.model, height=480, width=640)
                    if args.gif else None)
    dt = env.model.opt.timestep
    steps_per_policy = round(1. / (15 * dt))
    t_last = time.perf_counter()
    try:
        if args.viewer:
            from mujoco.studio import launch_web, viewer_protocol
            cfg = viewer_protocol.ViewerConfig(
                title="EarlyBots | Mobile cube collection (oracle teacher)",
                http_port=args.viewer_port)
            viewer_context = launch_web.launch(cfg)
        else:
            from contextlib import nullcontext
            viewer_context = nullcontext(None)
        with viewer_context as viewer:
            print(f"Starting: policy={args.policy}, cubes={args.cubes}, task={TASK}", flush=True)
            if args.viewer:
                print(f"MuJoCo WebViewer http://127.0.0.1:{args.viewer_port}/", flush=True)
            for step in range(args.max_steps):
                if viewer and not viewer.is_running():
                    break
                if args.policy == "expert":
                    action = expert.action(env)
                else:
                    obs = env.observation(renderers)
                    action = predict_smolvla(components, obs, args.device)
                if args.record and step % 3 == 0:
                    obs = env.observation(renderers)
                    dataset["state"].append(obs["state"])
                    dataset["action"].append(action.copy())
                    dataset["front"].append(obs["images"]["front"])
                    dataset["wrist"].append(obs["images"]["wrist"])
                env.step(action, frames=steps_per_policy)
                if viewer:
                    viewer.sync(env.model, env.data)
                    delay = dt * steps_per_policy - (time.perf_counter() - t_last)
                    if delay > 0:
                        time.sleep(delay)
                    t_last = time.perf_counter()
                if gif_renderer and step % 6 == 0:
                    frames_for_gif.append(env.render("overview", renderer=gif_renderer))
                if step % 30 == 0:
                    pos = env.cube_position(expert.cube if expert else 0)
                    print(f"t={env.data.time:.1f}s cubes={env.placed()}/{env.cubes} "
                          f"phase={expert.phase if expert else 'VLA'} "
                          f"base={env.state()[:2].round(3).tolist()} "
                          f"cube_xyz={pos.round(3).tolist()}", flush=True)
                if expert and (expert.done or expert.failure):
                    break
                if env.placed() == env.cubes and not expert:
                    break
    finally:
        if gif_renderer is not None:
            gif_renderer.close()
        if renderers:
            for renderer in renderers.values():
                renderer.close()
    report = {
        "policy": args.policy,
        "vla_inference_executed": args.policy == "smolvla",
        "expert_uses_privileged_cube_poses": args.policy == "expert",
        "robot": "LeKiwi-inspired simplified mobile Cartesian arm (not real LeKiwi)",
        "wheels": "visual only, holonomic X/Y/yaw servo approximation",
        "task": TASK,
        "cubes_requested": args.cubes,
        "cubes_in_bin": env.placed(),
        "simulation_seconds": float(env.data.time),
        "expert_physical_lifts": expert.lifts if expert else None,
        "failure": expert.failure if expert else None,
        "success": env.placed() == args.cubes and (expert is None or expert.done),
        "final_positions": [env.cube_position(i).round(4).tolist() for i in range(args.cubes)],
    }
    if args.gif and frames_for_gif:
        import imageio.v2 as imageio
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif, frames_for_gif, duration=6 * dt * steps_per_policy)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.record, **{k: np.asarray(v) for k, v in dataset.items()})
        report["recorded_frames"] = len(dataset["state"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cubes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--policy", choices=("expert", "smolvla"), default="expert")
    parser.add_argument("--checkpoint", help="Fine-tuned 7D mobile SmolVLA repository or path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--viewer", action="store_true", help="MuJoCo Studio WebViewer")
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--record", type=Path, help="Raw expert demonstrations, not a LeRobot dataset")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.check_model:
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        print(f"Scene OK: {model.nbody} bodies, {model.nu} actuators")
        return
    result = execute(args)
    if args.smoke_test and not result["success"]:
        raise SystemExit("Mobile smoke test FAILED, see measured report above")


if __name__ == "__main__":
    main()