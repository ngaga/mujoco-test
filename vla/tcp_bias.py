"""SO-101 TCP geometry and bounded, robot-frame joint target correction.

This is an experimental *post-policy* joint-space IK correction, NOT a
camera calibration, NOT a weight update, and NOT an object-aware controller.
The direction is computed from SO-101 finger geometry at nominal targets.
No privileged object position is used to choose a motor command.

All lengths in metres in computation, mm at the public interface.
"""
import numpy as np

from diagnose_contacts import PAD_GROUPS

ARM_DOF = 5
JAW_DOF = 6


def pad_centres(model, data):
    out = {}
    for side, names in PAD_GROUPS.items():
        out[side] = np.mean([data.geom_xpos[model.geom(n).id] for n in names], axis=0)
    return out


def closing_axis(model, data):
    positions = pad_centres(model, data)
    direction = positions["moving"] - positions["fixed"]
    direction[2] = 0
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError("Degenerate SO-101 horizontal jaw closing axis")
    return direction / length


def tcp_calibration(model, data):
    """Measure fixed pad and jaw midpoint relative to the *declared* tcp site.

    Dynamic moving jaw position is reported but must not be assumed fixed.
    Local coordinates are expressed in the rigid Fixed_Jaw body frame.
    """
    centres = pad_centres(model, data)
    tcp = np.asarray(data.site(model.site("tcp").id).xpos, dtype=float).copy()
    frame = np.asarray(data.body(model.body("Fixed_Jaw").id).xmat).reshape(3, 3)
    mid = (centres["fixed"] + centres["moving"]) / 2
    return {
        "tcp_world_mm": (tcp * 1000).round(3).tolist(),
        "fixed_pad_from_tcp_local_mm": (frame.T @ (centres["fixed"] - tcp) * 1000).round(3).tolist(),
        "moving_pad_from_tcp_local_mm": (frame.T @ (centres["moving"] - tcp) * 1000).round(3).tolist(),
        "jaw_midpoint_from_tcp_local_mm": (frame.T @ (mid - tcp) * 1000).round(3).tolist(),
        "closing_axis_world": closing_axis(model, data).round(6).tolist(),
        "jaw_opening_mm": round(float(np.linalg.norm(centres["moving"] - centres["fixed"])) * 1000, 3),
        "note": "Jaw midpoint varies with jaw angle; fixed pad offset should remain rigid in Fixed_Jaw frame.",
    }


def _fixed_jacobian(model, data):
    """Mean world-position and rotation Jacobian of four fixed pads."""
    import mujoco
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    for name in PAD_GROUPS["fixed"]:
        p = np.zeros((3, model.nv), dtype=float)
        r = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacGeom(model, data, p, r, model.geom(name).id)
        jacp += p / len(PAD_GROUPS["fixed"])
        jacr += r / len(PAD_GROUPS["fixed"])
    return jacp[:, :ARM_DOF], jacr[:, :ARM_DOF]


def corrected_targets(model, state, scratch, nominal, closing_mm,
                      damping=0.015, rotation_weight=0.015,
                      max_joint_shift=0.12):
    """Shift fixed pad centre along local jaw closing direction using 5 arm joints.

    Build Jacobian at NOMINAL VLA motor targets (not object-ground-truth pose).
    Damp least-squares to prefer small joint motion and orientation stability.
    No modification of the policy or real state. Jaw target is unchanged.
    """
    import mujoco
    nominal = np.asarray(nominal, dtype=float).reshape(-1)
    if nominal.shape != (JAW_DOF,) or not np.all(np.isfinite(nominal)):
        raise ValueError("Need six finite SO-101 joint setpoints")
    if abs(closing_mm) > 12:
        raise ValueError("Safety bound for experiment: |bias| <= 12mm")
    if closing_mm == 0:
        return nominal.copy(), {
            "requested_closing_mm": 0.0,
            "achieved_closing_mm": 0.0,
            "achieved_lateral_mm": 0.0,
            "max_arm_joint_delta_rad": 0.0,
        }
    scratch.qpos[:] = state.qpos
    scratch.qvel[:] = 0
    scratch.qpos[:JAW_DOF] = nominal
    mujoco.mj_forward(model, scratch)
    direction = closing_axis(model, scratch)
    before = pad_centres(model, scratch)["fixed"]
    jacp, jacr = _fixed_jacobian(model, scratch)
    # Orientation residual in radians weighted by metres/radian.
    a = np.vstack((jacp, rotation_weight * jacr))
    desired = np.concatenate((direction * (closing_mm / 1000), np.zeros(3)))
    lhs = a.T @ a + (damping * damping) * np.eye(ARM_DOF)
    delta = np.linalg.solve(lhs, a.T @ desired)
    delta = np.clip(delta, -max_joint_shift, max_joint_shift)
    corrected = nominal.copy()
    corrected[:ARM_DOF] += delta
    for j in range(ARM_DOF):
        bound = np.asarray(model.jnt_range[j], dtype=float)
        if model.jnt_limited[j]:
            corrected[j] = np.clip(corrected[j], bound[0], bound[1])
    scratch.qpos[:JAW_DOF] = corrected
    mujoco.mj_forward(model, scratch)
    actual_displacement = pad_centres(model, scratch)["fixed"] - before
    tangent = np.array([-direction[1], direction[0], 0.0])
    return corrected, {
        "requested_closing_mm": float(closing_mm),
        "achieved_closing_mm": round(1000 * float(actual_displacement @ direction), 3),
        "achieved_lateral_mm": round(1000 * float(actual_displacement @ tangent), 3),
        "max_arm_joint_delta_rad": round(float(np.max(np.abs(corrected[:ARM_DOF] - nominal[:ARM_DOF]))), 5),
    }
