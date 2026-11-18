"""Find IK-reachable pickup and lift poses for the real SO-101 mobile mount.

This is a diagnostic with actual model kinematics, not a physical grasp score.
It tests all candidates on one fixed cube, with the same orientation constraint
as the sim2cell grasp expert. Use small sweeps to locate an attainable mount.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .so101 import ARM_HOME, MOUNT_XYZ, MobileSO101Env


def reach_scan():
    from sim2cell.ik import solve_ik
    env = MobileSO101Env(cubes=1)
    mount_id = env.model.body("so101_mount").id
    base_x = env.model.jnt_qposadr[env.model.joint("base_x").id]
    base_y = env.model.jnt_qposadr[env.model.joint("base_y").id]
    cube = env.cube_position(0).copy()
    options = []
    for height in (0.20, 0.23, 0.25, 0.27):
        for clearance in (0.27, 0.30, 0.33, 0.36, 0.39):
            env.reset()
            env.model.body_pos[mount_id] = [MOUNT_XYZ[0], MOUNT_XYZ[1], height]
            env.data.qpos[base_x] = cube[0] - clearance
            env.data.qpos[base_y] = cube[1]
            mujoco.mj_forward(env.model, env.data)
            errors = {}
            qs = {}
            for target_z in (0.095, 0.018, 0.16):
                target = np.array([cube[0], cube[1], target_z])
                best = (float("inf"), None)
                for q0 in (
                    ARM_HOME[:5],
                    np.array([0.0, -2.2, 2.2, 1.57, 0.0]),
                    np.array([0.0, -1.0, 1.0, 1.57, 0.0]),
                ):
                    q, err = solve_ik(env.model, env.data, target, q_init=q0,
                                      max_iters=130, tol=0.0005)
                    if err < best[0]:
                        best = (float(err), q)
                errors[f"{target_z:.3f}"] = round(best[0], 5)
                qs[f"{target_z:.3f}"] = best[1].round(4).tolist()
            options.append({
                "mount_z": height, "cube_clearance_x": clearance,
                "tcp_errors_m": errors, "joint_targets": qs,
                "worst_error_m": round(max(errors.values()), 5),
                "pickup_error_m": errors["0.018"],
            })
    options.sort(key=lambda x: (x["worst_error_m"], x["pickup_error_m"]))
    return {"status": "kinematic reachability scan, not contact success",
            "original_mount": list(MOUNT_XYZ), "original_pick_clearance": 0.39,
            "candidates": options}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("mobile_runs/so101_reach_scan.json"))
    args = p.parse_args()
    report = reach_scan()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["candidates"][:10], indent=2), flush=True)
    print("Full scan:", args.output, flush=True)


if __name__ == "__main__":
    main()
