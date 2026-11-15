"""First EarlyBots behavior-cloning experiment in MuJoCo (not a VLA).

Collect REAL physics demonstrations from the old motion controller and an
actual RGB image of each initial soap placement. Fit a small, supervised,
time-indexed linear policy conditioned on the image-derived soap XY.

At evaluation the learned model issues the palm XYZ and the jaw motor
command directly. The old scripted Motion controller is NOT called.
Only MuJoCo contact and free-soap state are used to score success.

Example (Ubuntu with Mesa offscreen rendering):
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python imitation.py train
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python imitation.py evaluate
On a Mac desktop use mjpython instead of python.

Limitations: the trained model is a visual-conditioned motion primitive,
not a reactive neural VLA or a fully closed-loop dexterous policy. Time is
part of the input, and the soap detector is manually designed.
"""
import argparse
import json
import math
import random
from pathlib import Path

import mujoco
import numpy as np

from camera_preview import rgb_png
from motion import HOME
from physical_grasp import (
    ContactMonitor, JAW_CLOSED, PhysicalMotion, SOAP_TOP_Z, step_physical,
)
from robustness import MODEL_PATH, SOAP_X, SOAP_Y, in_tray
from vision import estimate_soap_xy

ROOT = Path(__file__).resolve().parent
DURATION = sum(p.duration for p in PhysicalMotion().phases)
DATA_STRIDE = 10  # 20 ms at the physical model's 2ms integration rate
DEFAULT_TRAIN = 24
DEFAULT_TEST = 10
DEFAULT_SEED = 2026
REFERENCE_SOAP_XY = (-0.27, 0.005)
MAX_STEP_PADDING = 400


def distinct_positions(count, seed):
    """Deterministically select placements; training and test seeds differ."""
    if count < 1:
        raise ValueError("Must request at least one position")
    rng = random.Random(seed)
    return [(rng.uniform(*SOAP_X), rng.uniform(*SOAP_Y))
            for _ in range(count)]


def soap_addresses(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "soap_free")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def scene_reset(model, data, xy):
    """Only reset the soap at the START of a run; never during motion."""
    adr, _ = soap_addresses(model)
    mujoco.mj_resetData(model, data)
    data.qpos[adr:adr + 2] = xy
    data.mocap_pos[0] = HOME
    mujoco.mj_forward(model, data)


def camera_xy(model, data, renderer):
    """The visual feature actually comes from rendered RGB pixels."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    renderer.update_scene(data, camera="overhead")
    frame = renderer.render()
    xy = estimate_soap_xy(
        frame, data.cam_xpos[cid], data.cam_xmat[cid],
        float(model.cam_fovy[cid]), SOAP_TOP_Z,
    )
    return xy, frame


def physically_verified(model, data, monitor, max_height, bilateral_seen):
    adr, _ = soap_addresses(model)
    xyz = tuple(float(v) for v in data.qpos[adr:adr + 3])
    return bool(bilateral_seen and max_height > 0.892 and in_tray(xyz))


def teacher_run(model, renderer, xy, images_dir=None, run_id=0,
                collect=False):
    """Collect the teacher's real MuJoCo commands, not teleported soap."""
    data = mujoco.MjData(model)
    motion = PhysicalMotion()
    monitor = ContactMonitor(model)
    adr, _ = soap_addresses(model)
    scene_reset(model, data, xy)
    observed, first_rgb = camera_xy(model, data, renderer)
    if observed is None:
        return {"success": False, "reason": "RGB soap not detected at start",
                "xy": xy, "observed_xy": None}
    motion.start()
    motion.retarget_soap(observed)
    first_path = None
    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)
        first_path = images_dir / f"teacher_{run_id:03d}_initial.png"
        first_path.write_bytes(rgb_png(first_rgb))
    samples = []
    max_height = -1.0
    bilateral_seen = False
    sampled_times = []
    for tick in range(math.ceil(DURATION / model.opt.timestep) + MAX_STEP_PADDING):
        step_physical(model, data, motion)
        z = float(data.qpos[adr + 2])
        max_height = max(max_height, z)
        if motion.index in (3, 4, 5):
            lft, rgt = monitor.contacts(data)
            bilateral_seen |= lft and rgt
        if collect and (tick + 1) % DATA_STRIDE == 0:
            target = np.asarray(data.mocap_pos[0], dtype=np.float64)
            jaw = float(data.ctrl[0] / JAW_CLOSED)
            samples.append([float(target[0]), float(target[1]),
                            float(target[2]), jaw])
            sampled_times.append(float(data.time))
        if motion.finished:
            break
    ok = physically_verified(model, data, monitor, max_height, bilateral_seen)
    out = {
        "success": ok, "xy": [float(v) for v in xy],
        "observed_xy": [float(v) for v in observed],
        "height_m": round(max_height, 5),
        "final_xyz": [round(float(v), 5) for v in data.qpos[adr:adr + 3]],
        "frames": len(samples),
        "initial_image": str(first_path) if first_path else None,
    }
    if collect:
        out["actions"] = np.asarray(samples, dtype=np.float64)
        out["times"] = np.asarray(sampled_times, dtype=np.float64)
    return out


class ImitationPolicy:
    """Supervised visual-conditioned action lookup, learned from demonstrations.

    Each 20ms time knot has a learned linear mapping from initial RGB soap
    centroid world XY to the four commands (palm XYZ, jaw open/closed).
    The weights are FIT from teacher actions. No scripted motion function is
    invoked to produce an action at evaluation.
    """

    def __init__(self, times, weights):
        self.times = np.asarray(times, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        if (self.times.ndim != 1 or len(self.times) < 2
                or self.weights.shape != (len(self.times), 3, 4)
                or not np.all(np.diff(self.times) > 0)):
            raise ValueError("Invalid learned policy shape or time knots")

    def action(self, time_s, observed_xy):
        if observed_xy is None:
            raise ValueError("RGB observation required; no ground-truth fallback")
        if len(observed_xy) != 2:
            raise ValueError("Expected detected soap (x,y)")
        offset = np.asarray(observed_xy) - REFERENCE_SOAP_XY
        feature = np.array([1.0, float(offset[0]), float(offset[1])])
        t = float(time_s)
        idx = int(np.searchsorted(self.times, t, side="right"))
        if idx <= 0:
            weights = self.weights[0]
        elif idx >= len(self.times):
            weights = self.weights[-1]
        else:
            lo, hi = idx - 1, idx
            alpha = (t - self.times[lo]) / (self.times[hi] - self.times[lo])
            weights = (1 - alpha) * self.weights[lo] + alpha * self.weights[hi]
        prediction = feature @ weights
        return tuple(float(v) for v in prediction[:3]), bool(prediction[3] >= 0.5)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=self.times, weights=self.weights)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arr:
            return cls(arr["times"], arr["weights"])


def fit_behavior_cloning(actions, times, observed_xy):
    """Fit a least-squares model independently for every action-time knot.

    Input shape: [successful_demos, knots, 4]. Held-out scenarios must not
    appear in this fit. Require enough XY variation for identifiability.
    """
    actions = np.asarray(actions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    observed_xy = np.asarray(observed_xy, dtype=np.float64)
    if (actions.ndim != 3 or actions.shape[2] != 4
            or observed_xy.shape != (actions.shape[0], 2)
            or times.shape != (actions.shape[1],)
            or actions.shape[0] < 4):
        raise ValueError("Invalid teacher demonstration tensors")
    dx = observed_xy - REFERENCE_SOAP_XY
    features = np.column_stack((np.ones(len(dx)), dx))
    if np.linalg.matrix_rank(features) != 3:
        raise ValueError("Training needs independent visual XY variations")
    # One learned 3x4 coefficient matrix per time knot. This is supervised
    # behavioral cloning, not a manually hard-coded trajectory table.
    weights = np.stack([
        np.linalg.lstsq(features, actions[:, frame, :], rcond=None)[0]
        for frame in range(actions.shape[1])
    ])
    return ImitationPolicy(times, weights)


def train(model, renderer, train_positions, output_dir, save_images=True):
    """Capture scripted teacher demonstrations and fit independent policy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded = []
    metadata = []
    for i, xy in enumerate(train_positions):
        run = teacher_run(
            model, renderer, xy,
            images_dir=output_dir / "teacher_images" if save_images else None,
            run_id=i, collect=True,
        )
        metadata.append({k: v for k, v in run.items()
                         if k not in ("actions", "times")})
        if run["success"]:
            recorded.append(run)
    if len(recorded) < 4:
        raise RuntimeError("Fewer than four successful physical teacher demonstrations")
    # Every teacher uses the same simulation timestep and phase schedule.
    # Fail rather than silently truncate data if a trajectory differs.
    first = recorded[0]["times"]
    if not all(len(x["times"]) == len(first)
               and np.allclose(x["times"], first) for x in recorded):
        raise RuntimeError("Teacher traces are not time-aligned")
    actions = np.stack([x["actions"] for x in recorded])
    observed = np.array([x["observed_xy"] for x in recorded])
    policy = fit_behavior_cloning(actions, first, observed)
    np.savez_compressed(
        output_dir / "teacher_dataset.npz",
        actions=actions, time_s=first, observed_xy=observed,
        initial_xy=np.array([x["xy"] for x in recorded]),
        teacher_success=np.ones(len(recorded), dtype=bool),
    )
    policy.save(output_dir / "learned_policy.npz")
    (output_dir / "teacher_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return policy, metadata


def learned_run(model, renderer, policy, xy):
    """Apply only the learned commands, not PhysicalMotion.advance or the teacher."""
    data = mujoco.MjData(model)
    monitor = ContactMonitor(model)
    adr, _ = soap_addresses(model)
    scene_reset(model, data, xy)
    observed, _ = camera_xy(model, data, renderer)
    if observed is None:
        return {"success": False, "reason": "RGB soap not detected at start",
                "xy": xy, "observed_xy": None}
    max_height, bilateral_seen = -1.0, False
    dt = float(model.opt.timestep)
    max_steps = math.ceil((policy.times[-1] + dt) / dt) + MAX_STEP_PADDING
    for tick in range(max_steps):
        t = (tick + 1) * dt
        xyz_target, grip_closed = policy.action(t, observed)
        data.mocap_pos[0] = xyz_target
        data.ctrl[:] = JAW_CLOSED if grip_closed else 0.0
        mujoco.mj_step(model, data)
        height = float(data.qpos[adr + 2])
        max_height = max(max_height, height)
        if grip_closed:
            lft, rgt = monitor.contacts(data)
            bilateral_seen |= lft and rgt and height > 0.892
    ok = physically_verified(model, data, monitor, max_height, bilateral_seen)
    return {
        "success": ok, "xy": [float(v) for v in xy],
        "observed_xy": [float(v) for v in observed],
        "height_m": round(max_height, 5),
        "final_xyz": [round(float(v), 5) for v in data.qpos[adr:adr + 3]],
    }


def evaluate(model, renderer, policy, positions):
    """Same seeded held-out placements, teacher vs independent learned policy."""
    runs = []
    for i, xy in enumerate(positions):
        teacher = teacher_run(model, renderer, xy)
        learner = learned_run(model, renderer, policy, xy)
        runs.append({"case": i, "xy": [round(v, 5) for v in xy],
                     "teacher": teacher, "learner": learner})
    return {
        "holdout_trials": len(runs),
        "teacher_successes": sum(x["teacher"]["success"] for x in runs),
        "learned_successes": sum(x["learner"]["success"] for x in runs),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "both"),
                        nargs="?", default="both")
    parser.add_argument("--train-demos", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--test-demos", type=int, default=DEFAULT_TEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "imitation_runs")
    parser.add_argument("--save-images", action="store_true",
                        help="persist initial RGB frame for each teacher rollout")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    renderer = mujoco.Renderer(model, width=640, height=480)
    try:
        train_positions = distinct_positions(args.train_demos, args.seed)
        test_positions = distinct_positions(args.test_demos, args.seed + 100000)
        if set(train_positions).intersection(test_positions):
            raise RuntimeError("Data leakage: held-out positions seen during training")
        if args.action in ("train", "both"):
            policy, metadata = train(
                model, renderer, train_positions,
                args.output_dir, save_images=args.save_images,
            )
            trained = sum(x["success"] for x in metadata)
            print(f"Physical teacher data: {trained}/{len(metadata)} demos succeeded",
                  flush=True)
            print(f"Learned action policy saved: {args.output_dir / 'learned_policy.npz'}",
                  flush=True)
        else:
            policy = ImitationPolicy.load(args.output_dir / "learned_policy.npz")
            metadata = None
        if args.action in ("evaluate", "both"):
            report = evaluate(model, renderer, policy, test_positions)
            report.update({
                "seed": args.seed, "train_seed": args.seed,
                "test_seed": args.seed + 100000,
                "train_demos_requested": args.train_demos,
                "train_demos_used": (sum(x["success"] for x in metadata)
                                     if metadata is not None else None),
                "policy_type": "learned RGB-conditioned time-knot linear behavioral cloning",
                "scope": "initial RGB yellow-soap location; open-loop learned palm and jaw commands",
                "model_uses_teacher_at_eval": False,
            })
            (args.output_dir / "evaluation.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            )
            print(
                f"HELD OUT: teacher={report['teacher_successes']}/"
                f"{report['holdout_trials']}; "
                f"learned={report['learned_successes']}/"
                f"{report['holdout_trials']}",
                flush=True,
            )
            print(f"Full paired evaluation: {args.output_dir / 'evaluation.json'}")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
