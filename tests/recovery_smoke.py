"""Headless physical test: force one drop, settle, regrasp, finish.

This uses MuJoCo's actual freely simulated soap pose as the retry observation,
matching the documented fallback when --visual-servo is disabled.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco

from motion import HOME
from physical_grasp import ContactMonitor, PhysicalMotion, step_physical
from recovery import RecoveryManager


def run_forced_drop(model, *, steps=15000):
    data = mujoco.MjData(model)
    motion = PhysicalMotion()
    monitor = ContactMonitor(model)
    recovery = RecoveryManager(max_attempts=2)
    soap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "soap_free")
    qadr, vadr = int(model.jnt_qposadr[soap_id]), int(model.jnt_dofadr[soap_id])
    mujoco.mj_resetData(model, data)
    data.mocap_pos[0] = HOME
    mujoco.mj_forward(model, data)
    initial_xy = tuple(float(v) for v in data.qpos[qadr:qadr + 2])
    motion.start()
    recovery.start()
    injected = False
    second_attempt_xy = None
    retry_soap_xyz = None
    unchanged_during_replan = None
    for _ in range(steps):
        if (not injected and recovery.attempt == 1 and motion.index == 4
                and motion.elapsed >= 0.25):
            injected = True
        force_open = injected and recovery.attempt == 1 and recovery.status == "running"
        step_physical(model, data, motion, force_open=force_open)
        xyz = tuple(float(v) for v in data.qpos[qadr:qadr + 3])
        velocity = tuple(float(v) for v in data.qvel[vadr:vadr + 3])
        left, right = monitor.contacts(data)
        placed = (0.075 < xyz[0] < 0.445
                  and -0.12 < xyz[1] < 0.22 and 0.82 < xyz[2] < 0.92)
        ready = recovery.after_step(
            motion=motion, soap_xyz=xyz, linear_velocity=velocity,
            both_contacts=left and right, in_tray=placed,
            dt=float(model.opt.timestep),
        )
        if ready:
            before = tuple(float(v) for v in data.qpos[qadr:qadr + 7])
            started = recovery.restart_at(
                motion=motion,
                palm_xyz=tuple(float(v) for v in data.mocap_pos[0]),
                soap_xy=xyz[:2],
            )
            after = tuple(float(v) for v in data.qpos[qadr:qadr + 7])
            if started:
                second_attempt_xy = xyz[:2]
                retry_soap_xyz = xyz
                unchanged_during_replan = before == after
        if recovery.status in ("success", "failed"):
            return {
                "status": recovery.status,
                "attempt": recovery.attempt,
                "initial_xy": initial_xy,
                "retry_xy": second_attempt_xy,
                "retry_soap_xyz": retry_soap_xyz,
                "soap_pose_unchanged_on_replan": unchanged_during_replan,
                "injected": injected,
                "final_xyz": xyz,
                "grasp_verified": recovery.grasp_verified,
                "last_error": recovery.last_error,
            }
    raise RuntimeError("Recovery rollout exceeded test step budget")


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "bathroom_physical.xml"))
    result = run_forced_drop(model)
    print(f"RECOVERY CHECK: {result}", flush=True)
    if not result["injected"]:
        raise RuntimeError("The drop was never injected")
    if result["status"] != "success" or result["attempt"] != 2:
        raise RuntimeError("Forced drop did not recover in exactly two attempts")
    if not result["soap_pose_unchanged_on_replan"]:
        raise RuntimeError("A retry must never teleport or reset the soap body")
    if result["retry_xy"] is None:
        raise RuntimeError("A retry never found the dropped soap")
    if (sum((a-b)**2 for a,b in zip(result["retry_xy"],result["initial_xy"]))
            **0.5 < 0.015):
        raise RuntimeError("Retry did not use a meaningfully displaced dropped soap")
    if not result["grasp_verified"]:
        raise RuntimeError("Final grasp was not physically verified")
    print("OK: injected drop, re-localization, regrasp, and tray placement", flush=True)


if __name__ == "__main__":
    main()
