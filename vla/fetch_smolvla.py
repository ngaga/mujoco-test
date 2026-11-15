"""Download and inspect a pre-trained SmolVLA Pick & Place checkpoint.

This is an OPTIONAL experimental model, not a replacement for the EarlyBots
palm controller. Its outputs are SO-ARM100/101 joint targets, whereas the
EarlyBots MuJoCo demonstration uses a floating XYZ palm and two jaw motors.

Use a SEPARATE Python 3.12 virtual environment:
    python3.12 -m venv .venv-vla
    source .venv-vla/bin/activate
    python -m pip install "huggingface_hub>=0.28,<2"
    python vla/fetch_smolvla.py --metadata
    python vla/fetch_smolvla.py --download
    python -m pip install "lerobot[smolvla]"
    python vla/fetch_smolvla.py --load

The --download command retrieves ~0.9GB of pretrained weights from the
author's Hugging Face repository, plus normalization and configuration.
No training is performed and no robot actuator is commanded.
"""
import argparse
import json
from pathlib import Path

MODEL_ID = "ahmedsohail2003/smolvla-so101-pickplace-v2"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_*_normalizer_processor.safetensors",
    "policy_postprocessor_step_*_unnormalizer_processor.safetensors",
    "train_config.json",
    "README.md",
)
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "checkpoints" / "smolvla-so101-pickplace-v2"


def describe_config(config):
    """Return explicit architectural compatibility facts, not a false adapter."""
    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    return {
        "pretrained_repository": MODEL_ID,
        "model_type": config.get("type", "unknown"),
        "input_features": input_features,
        "output_features": output_features,
        "max_action_dim": config.get("max_action_dim"),
        "earlybots_compatible_without_adapter": False,
        "reason": (
            "The checkpoint consumes SO-ARM100/101 motor state and camera "
            "images and predicts joint targets; the EarlyBots prototype "
            "currently commands floating palm XYZ and two sliding jaws."
        ),
    }


def inspect_local(path):
    path = Path(path)
    config = path / "config.json"
    if not config.exists():
        raise FileNotFoundError(f"Missing {config}; run --metadata first.")
    return describe_config(json.loads(config.read_text()))


def download_checkpoint(local_dir, metadata_only=False):
    from huggingface_hub import snapshot_download

    patterns = ("config.json", "README.md", "policy_preprocessor.json",
                "policy_postprocessor.json") if metadata_only else MODEL_FILES
    return Path(snapshot_download(
        repo_id=MODEL_ID,
        repo_type="model",
        allow_patterns=list(patterns),
        local_dir=str(local_dir),
    ))


def probe_model(local_dir, requested_device="auto"):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if requested_device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = requested_device
    policy = SmolVLAPolicy.from_pretrained(str(local_dir))
    policy = policy.to(device).eval()
    return {
        "loaded": True,
        "device": device,
        "model_class": type(policy).__name__,
        "note": (
            "Weights loaded, but NO pick-and-place was executed. "
            "Use the matching SO-ARM100 simulation and camera/state "
            "normalization to validate actual actions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    op = parser.add_mutually_exclusive_group(required=True)
    op.add_argument("--metadata", action="store_true",
                    help="download only config and model card (no large weights)")
    op.add_argument("--download", action="store_true",
                    help="download the real pretrained weights (~0.9GB)")
    op.add_argument("--load", action="store_true",
                    help="load a previously downloaded checkpoint with LeRobot")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    args = parser.parse_args()
    if args.metadata or args.download:
        directory = download_checkpoint(args.local_dir, metadata_only=args.metadata)
    else:
        directory = args.local_dir
    print(json.dumps(inspect_local(directory), indent=2, ensure_ascii=False))
    if args.download:
        weights = directory / "model.safetensors"
        if not weights.exists() or weights.stat().st_size < 10_000_000:
            raise RuntimeError("Missing or incomplete model.safetensors; checkpoint not usable")
        print(f"Checkpoint downloaded: {weights} ({weights.stat().st_size / 1e6:.1f} MB)")
    if args.load:
        print(json.dumps(probe_model(directory, args.device), indent=2))


if __name__ == "__main__":
    main()
