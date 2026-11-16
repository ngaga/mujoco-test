"""Instrument SO-101 gripper alignment and real MuJoCo contacts, no policy changes.

Both jaw finger pads are measured in WORLD XYZ and in a dynamically computed
closing-axis/tangent frame. Contact events are observed at every 2 ms physics
step, not merely at the 15 Hz policy interval. Nothing is fed back to SmolVLA.

  mjpython vla/diagnose_contacts.py --object cube --episodes 5 --device mps
  mjpython vla/diagnose_contacts.py --object rectangle --episodes 5 --device mps
  mjpython vla/diagnose_contacts.py --object cube --check-env
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from eval_soap import (SOAP_SIZE, SOAP_TASK, MODEL_ID, open_soap_environment,
                       device_for, load_model, observation_to_batch,
                       validate_action, CONTROL_HZ)
from eval_yellow_cube import CUBE_SIZE

PAD_GROUPS = {
    "fixed": tuple(f"fixed_jaw_pad_{i}" for i in range(1, 5)),
    "moving": tuple(f"moving_jaw_pad_{i}" for i in range(1, 5)),
}
SIZE = {"cube": CUBE_SIZE, "rectangle": SOAP_SIZE}


def _point(values):
    return [round(float(x), 6) for x in values]


def _unit_xy(diff):
    p = np.asarray(diff, dtype=float).copy()
    p[2] = 0.0
    norm = np.linalg.norm(p)
    if norm < 1e-6:
        raise RuntimeError("Finger pads have degenerate horizontal closing axis")
    return p / norm


def inspect_alignment(model, data):
    """Robot-derived closing and forward axes, never a guessed global X/Y.

    Horizontal closing axis points fixed-pad -> moving-pad. The forward
    axis points from the fixed finger's base pad #4 to its tip pad #1.
    Positive signed distances are explicitly defined, in millimetres.
    """
    positions = {
        side: np.stack([data.geom(model.geom(name).id).xpos
                        for name in names])
        for side, names in PAD_GROUPS.items()
    }
    fixed = positions["fixed"].mean(axis=0)
    moving = positions["moving"].mean(axis=0)
    closing = _unit_xy(moving - fixed)
    tangent = np.array([-closing[1], closing[0], 0.0])
    tip = positions["fixed"][0]
    base = positions["fixed"][-1]
    forward = _unit_xy(tip - base)
    object_geom = model.geom("block_geom").id
    center = data.geom_xpos[object_geom].copy()
    axes = data.geom_xmat[object_geom].reshape(3, 3).copy()
    half_size = model.geom_size[object_geom].copy()
    # Support of the rotated box projected onto the closing axis.
    half_width = sum(half_size[k] * abs(float(np.dot(axes[:, k], closing)))
                     for k in range(3))
    obj_relative_fixed = center - fixed
    obj_relative_middle = center - (fixed + moving) / 2
    obj_relative_tip = center - tip
    jaw_gap = float(np.dot(moving - fixed, closing))
    upright_angle = math.degrees(math.acos(
        np.clip(abs(float(axes[2, 2])), -1, 1)
    ))
    return {
        "world": {
            "object_center_xyz_m": _point(center),
            "fixed_pad_centroid_xyz_m": _point(fixed),
            "moving_pad_centroid_xyz_m": _point(moving),
            "fixed_tip_pad_xyz_m": _point(tip),
            "moving_tip_pad_xyz_m": _point(positions["moving"][0]),
            "closing_axis_xyz": _point(closing),
            "tangent_axis_xyz": _point(tangent),
            "forward_fixed_finger_xyz": _point(forward),
            "tcp_xyz_m": _point(data.site(model.site("tcp").id).xpos),
        },
        "gripper_frame_mm": {
            "object_from_fixed_along_closing": round(float(np.dot(obj_relative_fixed, closing))*1000, 2),
            "object_from_fixed_perpendicular": round(float(np.dot(obj_relative_fixed, tangent))*1000, 2),
            "object_from_jaw_midpoint_along_closing": round(float(np.dot(obj_relative_middle, closing))*1000, 2),
            "object_from_jaw_midpoint_perpendicular": round(float(np.dot(obj_relative_middle, tangent))*1000, 2),
            "fixed_pad_to_object_near_face_clearance": round(float(np.dot(obj_relative_fixed, closing)-half_width)*1000, 2),
            "moving_pad_beyond_far_face_clearance": round(float(jaw_gap - np.dot(obj_relative_fixed, closing)-half_width)*1000, 2),
            "object_ahead_of_fixed_tip": round(float(np.dot(obj_relative_tip, forward))*1000, 2),
            "object_from_fixed_tip_z": round(float(obj_relative_tip[2])*1000, 2),
            "jaw_pad_centroid_gap": round(jaw_gap*1000, 2),
            "object_projected_half_width": round(half_width*1000, 2),
        },
        "object_tilt_deg": round(upright_angle, 2),
    }


def _is_robot_geom(model, geom_id):
    """Follow the body's parent chain to the SO-101 'Base' body."""
    base = model.body("Base").id
    body = int(model.geom_bodyid[geom_id])
    while body > 0:
        if body == base:
            return True
        body = int(model.body_parentid[body])
    return False


def _geom_name(model, gid):
    import mujoco
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom:{gid}"


def _contact_side(name):
    if name.startswith("fixed_jaw_pad_"):
        return "fixed_pad"
    if name.startswith("moving_jaw_pad_"):
        return "moving_pad"
    return "other_robot_geom"


class ContactRecorder:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.object_id = model.geom("block_geom").id
        self.events = []
        self.first_robot_contact = None
        self.first_fixed_contact = None
        self.first_moving_contact = None
        self._active_pairs = set()
        self.totals = {"fixed_pad": 0, "moving_pad": 0, "other_robot_geom": 0}
        self.current_step_max_force_n = 0.0
        self.current_step_robot_contacts = set()

    def reset(self):
        self.events.clear()
        self.first_robot_contact = None
        self.first_fixed_contact = None
        self.first_moving_contact = None
        self._active_pairs.clear()
        self.totals = dict.fromkeys(self.totals, 0)
        self.current_step_max_force_n = 0.0
        self.current_step_robot_contacts.clear()

    def begin_action(self):
        self.current_step_max_force_n = 0.0
        self.current_step_robot_contacts.clear()

    def observe_substep(self, step_number, substep_number):
        import mujoco
        pairs_now = set()
        for idx in range(self.data.ncon):
            c = self.data.contact[idx]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.object_id not in (g1, g2):
                continue
            other = g2 if g1 == self.object_id else g1
            if not _is_robot_geom(self.model, other):
                continue  # table and tray contacts aren't grasp contacts
            name = _geom_name(self.model, other)
            side = _contact_side(name)
            pair = (self.object_id, other)
            pairs_now.add(pair)
            self.current_step_robot_contacts.add(side)
            forces = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, idx, forces)
            force = float(np.linalg.norm(forces[:3]))
            self.current_step_max_force_n = max(self.current_step_max_force_n, force)
            if pair in self._active_pairs:
                continue
            self.totals[side] += 1
            e = {
                "time_s": round(float(self.data.time), 5),
                "action_step": step_number,
                "physics_substep": substep_number,
                "contact_kind": side,
                "robot_geom": name,
                "soap_geom": "block_geom",
                "contact_world_xyz_m": _point(c.pos),
                "contact_force_N": round(force, 3),
                "penetration_mm": round(max(0.0, -float(c.dist))*1000, 3),
                "alignment": inspect_alignment(self.model, self.data),
            }
            self.events.append(e)
            if self.first_robot_contact is None:
                self.first_robot_contact = e
            if side == "fixed_pad" and self.first_fixed_contact is None:
                self.first_fixed_contact = e
            if side == "moving_pad" and self.first_moving_contact is None:
                self.first_moving_contact = e
        self._active_pairs = pairs_now


def physical_control_action(env, action, recorder, step_number):
    """Exactly the reference env integration schedule, with passive probes."""
    import mujoco
    recorder.begin_action()
    env.data.ctrl[:6] = np.asarray(action, dtype=float)
    for sub in range(env.n_substeps):
        mujoco.mj_step(env.model, env.data)
        recorder.observe_substep(step_number, sub)
    return env.get_obs()


def annotated_frame(obs, alignment, recorder, episode, step):
    """Actual two MuJoCo camera frames, with measured 3D-grasp telemetry."""
    from PIL import Image, ImageDraw
    img = np.concatenate((obs["images"]["front"], obs["images"]["wrist"]), axis=1)
    out = Image.new("RGB", (img.shape[1], img.shape[0] + 56), (22, 25, 30))
    out.paste(Image.fromarray(img), (0, 56))
    draw = ImageDraw.Draw(out)
    g = alignment["gripper_frame_mm"]
    draw.text((4, 3), f"Ep {episode} / step {step}: FRONT | WRIST", fill="white")
    draw.text((4, 19), f"fixed near-face {g['fixed_pad_to_object_near_face_clearance']:+.1f}mm | "
              f"moving far-face {g['moving_pad_beyond_far_face_clearance']:+.1f}mm", fill="#ffe07d")
    draw.text((4, 35), f"lateral {g['object_from_fixed_perpendicular']:+.1f}mm | "
              f"tip fore-aft {g['object_ahead_of_fixed_tip']:+.1f}mm | "
              f"touch: {','.join(sorted(recorder.current_step_robot_contacts)) or 'none'}",
              fill="#91d8f5")
    return np.asarray(out)


def classify_contact(row):
    """Describe observed contact, not presume a calibration defect."""
    first = row["first_robot_contact"]
    if first is None:
        return "no_contact_with_robot"
    return first["contact_kind"]


def run_episode(env, policy, pre, post, torch, device, episode, max_steps, gif_path):
    rec = ContactRecorder(env.model, env.data)
    obs = env.reset()
    policy.reset()
    rec.reset()
    start_xyz = _point(env.block_pos)
    initial_tilt = inspect_alignment(env.model, env.data)["object_tilt_deg"]
    trace = []
    frames = []
    success = False
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            batch = observation_to_batch(obs, device, torch)
            batch["task"] = SOAP_TASK
            targets = validate_action(post(policy.select_action(pre(batch))).detach().cpu().numpy(), env.model)
        obs = physical_control_action(env, targets, rec, step)
        metrics = inspect_alignment(env.model, env.data)
        trace.append({
            "step": step,
            "time_s": round(float(env.data.time), 4),
            "joint_target_rad": _point(targets),
            "joint_actual_rad": _point(env.data.qpos[:6]),
            "alignment": metrics,
            "active_robot_contact_kinds": sorted(rec.current_step_robot_contacts),
            "max_robot_contact_force_n": round(rec.current_step_max_force_n, 3),
        })
        if gif_path is not None and (step == 1 or step % 3 == 0):
            frames.append(annotated_frame(obs, metrics, rec, episode, step))
        if env.is_success():
            success = True
            break
    if gif_path is not None and frames:
        import imageio.v3 as iio
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(gif_path, np.stack(frames), duration=3/CONTROL_HZ, loop=0)
    return {
        "episode": episode,
        "success": bool(success),
        "steps": step,
        "start_object_xyz_m": start_xyz,
        "end_object_xyz_m": _point(env.block_pos),
        "start_tilt_deg": initial_tilt,
        "final_tilt_deg": trace[-1]["alignment"]["object_tilt_deg"],
        "first_robot_contact": rec.first_robot_contact,
        "first_fixed_pad_contact": rec.first_fixed_contact,
        "first_moving_pad_contact": rec.first_moving_contact,
        "first_contact_type": classify_contact({"first_robot_contact": rec.first_robot_contact}),
        "contact_onset_counts": rec.totals,
        "contact_onsets": rec.events,
        "step_trace": trace,
        "gif": str(gif_path) if gif_path is not None else None,
    }


def report_markdown(report):
    lines = [
        f"# SO-101 contact / jaw-alignment diagnostic: {report['object']}",
        "",
        f"- Pretrained checkpoint: {report['checkpoint']}",
        f"- Instruction: {SOAP_TASK}",
        "- All coordinates are measured from actual MuJoCo data, NEVER supplied to VLA.",
        "- Positive closing axis = fixed jaw pad centroid toward moving jaw pad centroid.",
        "- Near-face clearance >0: fixed pad centroid is behind the near projected object face; "
        "<0: centroid overlaps the projected object silhouette. This 2D projection "
        "is not by itself proof of physical collision.",
        "- Far-face clearance >0: moving pad centroid is beyond the far object face.",
        "- Lateral sign = 90-degree CCW rotation of horizontal closing axis.",
        "- Forward sign = fixed finger pad #4 (base) toward pad #1 (tip).",
        "- Each contact event contains exact colliding geom, normal force, world point, "
        "object tilt and gripper coordinates at the **2ms** physics step.",
        "",
        f"## Success: {report['successes']}/{report['episodes']}",
        "",
        "| Episode | Success | First robot impact | Fixed near-face (mm) | Lateral (mm) | Object tip-ahead (mm) | GIF |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for e in report["results"]:
        c = e["first_robot_contact"]
        g = c["alignment"]["gripper_frame_mm"] if c else None
        lines.append(
            f"| {e['episode']} | {'yes' if e['success'] else 'no'} "
            f"| {c['contact_kind'] + ' (' + c['robot_geom'] + ')' if c else 'none'} "
            f"| {g['fixed_pad_to_object_near_face_clearance'] if g else 'n/a'} "
            f"| {g['object_from_fixed_perpendicular'] if g else 'n/a'} "
            f"| {g['object_ahead_of_fixed_tip'] if g else 'n/a'} "
            f"| {'[GIF](' + e['gif'] + ')' if e['gif'] else 'not recorded'} |"
        )
    lines += [
        "",
        "## How to interpret",
        "",
        "Compare a successful and failed trial **at first robot contact**, not only "
        "the last video frame. A first impact from a non-pad mesh or a fixed tip "
        "with negative near-face clearance is consistent with an approach/pad "
        "placement problem, but does NOT prove a camera calibration error. "
        "Distinguish world XYZ from pad-derived closing, forward, and lateral axes.",
        "",
        "The raw JSON includes every 15 Hz target, actual joint state, pad geometry "
        "and every new robot/object contact observed at 2 ms resolution. "
        "Probes are passive. No correction, teleportation, or fine-tuning was used.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", choices=tuple(SIZE), default="cube")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-steps", type=int, default=240)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--repo", default=MODEL_ID)
    p.add_argument("--check-env", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("vla_runs/contact_diagnostics"))
    args = p.parse_args(argv)
    if args.episodes < 1 or args.max_steps < 1:
        p.error("--episodes and --max-steps must be >= 1")
    return args


def main():
    args = parse_args()
    env = open_soap_environment(args.seed, SIZE[args.object])
    try:
        if args.check_env:
            env.reset()
            metrics = inspect_alignment(env.model, env.data)
            print(json.dumps({"ok": True, "object": args.object,
                              "gripper": metrics}, indent=2))
            return
        import torch
        device = device_for(args.device, torch)
        policy, pre, post = load_model(args.repo, device)
        output = args.output_dir / args.object
        output.mkdir(parents=True, exist_ok=True)
        results = []
        for ep in range(1, args.episodes + 1):
            # Record selected cases: first attempt and first five episodes. All
            # detailed JSON traces are always recorded, regardless of GIF.
            gif = output / f"episode_{ep:02d}.gif" if ep <= 5 else None
            row = run_episode(env, policy, pre, post, torch, device, ep,
                              args.max_steps, gif)
            if gif is not None:
                row["gif"] = gif.name
            results.append(row)
            print(f"CONTACT {args.object} {ep}/{args.episodes} "
                  f"success={row['success']} first={row['first_contact_type']}", flush=True)
        report = {"object": args.object, "seed": args.seed,
                  "checkpoint": args.repo, "device": device,
                  "episodes": args.episodes,
                  "successes": sum(int(r["success"]) for r in results),
                  "results": results, "probes_only": True}
        (output / "contacts.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "report.md").write_text(report_markdown(report))
        print(f"Contact diagnosis: {report['successes']}/{args.episodes}. "
              f"See {output / 'report.md'}", flush=True)
    finally:
        env._renderer.close()


if __name__ == "__main__":
    main()
