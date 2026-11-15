"""Run pretrained SmolVLA SO-101 on the author's real MuJoCo PickPlaceEnv.

NOT our XYZ palm. No VLA training. Both front/wrist RGB camera observations,
six motor states and language are passed to the published checkpoint.

Examples (see vla/README.md):
    python vla/run_so101.py --check-env
    mjpython vla/run_so101.py --episodes 1 --device mps
    mjpython vla/run_so101.py --episodes 20 --seed 123 --device mps
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from fetch_smolvla import MODEL_ID

TASK = "Pick up the red block and place it in the blue tray."
CAMERA_MAP = {"front": "camera1", "wrist": "camera2"}
STATE_DIM = 6
ACTION_DIM = 6
CONTROL_HZ = 15.0
MAX_STEPS = 240
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "vla_runs" / "so101_evaluation.json"


def validate_observation(obs):
    """Reject missing/black images, malformed motor state and uncalibrated input."""
    if not isinstance(obs, dict) or "state" not in obs or "images" not in obs:
        raise ValueError("Expected SO-101 state and camera images")
    state = np.asarray(obs["state"], dtype=np.float32)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError(f"SO-101 state must be six finite values; got {state.shape}")
    images = {}
    for camera in CAMERA_MAP:
        if camera not in obs["images"]:
            raise ValueError(f"Missing {camera} SO-101 camera")
        rgb = np.asarray(obs["images"][camera])
        if (rgb.ndim != 3 or rgb.shape[2] != 3
                or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 32):
            raise ValueError(f"Invalid {camera} RGB image: {rgb.shape}/{rgb.dtype}")
        if not np.any(rgb):
            raise ValueError(f"{camera} RGB frame is black; renderer may be broken")
        images[camera] = np.ascontiguousarray(rgb)
    return state, images


def observation_to_batch(obs, device, torch):
    """Match author's front->camera1 / wrist->camera2 LeRobot input mapping."""
    state, images = validate_observation(obs)
    batch = {"observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
             "task": TASK}
    for name, target in CAMERA_MAP.items():
        img = torch.from_numpy(images[name]).permute(2, 0, 1).float() / 255.0
        batch[f"observation.images.{target}"] = img.unsqueeze(0).to(device)
    return batch


def validate_action(action, model=None):
    """Six finite SO-101 joint position targets, never EarlyBots floating palm XYZ."""
    arr = np.asarray(action, dtype=np.float64)
    if arr.shape == (1, ACTION_DIM):
        arr = arr[0]
    if arr.shape != (ACTION_DIM,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"Expected six finite joint targets, got {arr.shape}")
    if model is not None and model.nu != ACTION_DIM:
        raise ValueError(f"SO-101 must have six actuators, got {model.nu}")
    if np.any(np.abs(arr) > 10):
        raise ValueError("Implausible joint target: magnitude above 10")
    return arr.astype(np.float32)


def device_for(requested, torch):
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; use --device cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(repo, device):
    """Use the model's saved normalization and postprocessing, not raw weights."""
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    model = SmolVLAPolicy.from_pretrained(repo).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=model.config, pretrained_path=repo,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return model, pre, post


def open_environment(seed):
    """Import matching third-party SO-ARM100 simulation, not a fake substitute."""
    try:
        from sim2cell.env import PickPlaceEnv
    except ImportError as exc:
        raise RuntimeError(
            "Install the matching SO-ARM100 sim2cell workcell first. "
            "See vla/README.md."
        ) from exc
    return PickPlaceEnv(seed=seed, image_hw=(224, 224),
                        cameras=tuple(CAMERA_MAP), control_hz=CONTROL_HZ)


def check_environment(seed=123):
    """Real physics + both real RGB cameras, NO model download/inference."""
    env = open_environment(seed)
    try:
        if env.model.nu != ACTION_DIM:
            raise RuntimeError(f"Wrong SO-101 actuator count: {env.model.nu}")
        for joint in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch",
                      "Wrist_Roll", "Jaw"):
            env.model.joint(j)  # raises for an incompatible robot
        state, images = validate_observation(env.reset())
        obs2 = env.step(env.data.ctrl[:ACTION_DIM].copy())
        state2, images2 = validate_observation(obs2)
        if any(images[c].shape != images2[c].shape for c in CAMERA_MAP):
            raise RuntimeError("RGB frame dimensions changed after stepping")
        return {
            "ok": True,
            "robot": "SO-ARM100/101",
            "actuators": env.model.nu,
            "joint_state_shape": list(state.shape),
            "cameras": list(CAMERA_MAP),
            "frame_shape": list(images["front"].shape),
            "control_hz": CONTROL_HZ,
            "physics_timestep": float(env.model.opt.timestep),
            "checkpoint_loaded": False,
        }
    finally:
        if hasattr(env, "_renderer"):
            env._renderer.close()


def run_episode(env, policy, pre, post, torch, device,
                max_steps=MAX_STEPS, index=0, gif_frames=False):
    """True image->VLA->six joint targets->MuJoCo physics closed-loop control."""
    obs = env.reset()
    policy.reset()
    start = time.perf_counter()
    frames = []
    success = False
    executed = 0
    initial_pos = tuple(float(x) for x in env.block_pos)
    for step in range(max_steps):
        with torch.no_grad():
            batch = pre(observation_to_batch(obs, device, torch))
            action = post(policy.select_action(batch))
        target = validate_action(action.detach().cpu().numpy(), env.model)
        obs = env.step(target)
        executed += 1
        if gif_frames and index == 0 and step % 2 == 0:
            frames.append(obs["images"]["front"].copy())
        if env.is_success():
            success = True
            break
    return {
        "episode": index,
        "success": bool(success),
        "steps": executed,
        "start_block": [round(x, 5) for x in initial_pos],
        "end_block": [round(float(x), 5) for x in env.block_pos],
        "tcp_final": [round(float(x), 5) for x in env.tcp_pos],
        "wall_seconds": round(time.perf_counter() - start, 2),
        "frames": frames,
    }


def evaluate(args):
    import torch
    device = device_for(args.device, torch)
    env = open_environment(args.seed)
    try:
        policy, pre, post = load_model(args.repo, device)
        results = []
        for index in range(args.episodes):
            row = run_episode(env, policy, pre, post, torch, device,
                              args.max_steps, index, bool(args.gif))
            frames = row.pop("frames")
            if args.gif and index == 0 and frames:
                import imageio.v3 as iio
                args.gif.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(args.gif, np.stack(frames),
                            duration=2.0 / CONTROL_HZ, loop=0)
            results.append(row)
            print(f"VLA episode {index + 1}/{args.episodes}: "
                  f"success={row['success']} steps={row['steps']}", flush=True)
        report = {
            "checkpoint": args.repo, "instruction": TASK,
            "robot": "sim2cell MuJoCo SO-ARM100",
            "policy": "pretrained SmolVLA (no EarlyBots training)",
            "seed": args.seed, "max_steps": args.max_steps,
            "device": device, "episodes": args.episodes,
            "successes": sum(x["success"] for x in results),
            "results": results,
            "success_definition": "MuJoCo block in tray and gripper retreated",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"SmolVLA real physics: {report['successes']}/{args.episodes}. "
              f"Details: {args.output}", flush=True)
        return report
    finally:
        if hasattr(env, "_renderer"):
            env._renderer.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true",
                        help="check real arm joints, cameras and contact simulation without weights")
    parser.add_argument("--repo", default=MODEL_ID,
                        help="pretrained HF ID or downloaded local model directory")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
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
