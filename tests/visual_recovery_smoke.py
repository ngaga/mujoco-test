"""Full RGB-camera recovery regression after an injected physical soap drop.

No privileged MuJoCo XY is passed to the retry planner. It uses the exact
RGB observation and calibrated projection used by the interactive app.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco

from demo import State, observe_soap, soap_joint_addresses
from motion import HOME
from physical_grasp import ContactMonitor, PhysicalMotion, SOAP_TOP_Z, step_physical
from recovery import RecoveryManager


def run_camera_recovery(model, renderer, max_steps=16000):
    data = mujoco.MjData(model)
    motion = PhysicalMotion()
    monitor = ContactMonitor(model)
    recovery = RecoveryManager()
    state = State(8080)
    qadr, vadr = soap_joint_addresses(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    mujoco.mj_resetData(model, data)
    data.mocap_pos[0] = HOME
    mujoco.mj_forward(model, data)
    motion.start()
    recovery.start()

    initial_xy = observe_soap(model, data, motion, renderer, camera_id, state, SOAP_TOP_Z)
    if initial_xy is None:
        raise AssertionError("Angled RGB camera cannot find soap at initial pose")
    injected = False
    missed = 0
    retry_xy = None
    retry_actual_xy = None
    for tick in range(max_steps):
        if (not injected and recovery.attempt == 1
                and motion.index == 4 and motion.elapsed >= 0.25):
            injected = True
        force_open = injected and recovery.attempt == 1 and recovery.status == "running"
        if motion.running and motion.index < 2 and tick % 50 == 0:
            observe_soap(model, data, motion, renderer, camera_id, state, SOAP_TOP_Z)
        step_physical(model, data, motion, force_open=force_open)
        xyz = tuple(float(v) for v in data.qpos[qadr:qadr + 3])
        velocity = tuple(float(v) for v in data.qvel[vadr:vadr + 3])
        left, right = monitor.contacts(data)
        placed = 0.075 < xyz[0] < 0.445 and -0.12 < xyz[1] < 0.22 and 0.82 < xyz[2] < 0.92
        ready = recovery.after_step(
            motion=motion, soap_xyz=xyz, linear_velocity=velocity,
            both_contacts=left and right, in_tray=placed,
            dt=float(model.opt.timestep),
        )
        if ready:
            measured_xy = observe_soap(
                model, data, motion, renderer, camera_id, state, SOAP_TOP_Z,
            )
            recovery.note_visual_observation(measured_xy)
            if measured_xy is None:
                missed += 1
            else:
                retry_xy = measured_xy
                retry_actual_xy = xyz[:2]
                # An incorrect camera detection must not trigger a blind grasp.
                error = sum((a - b) ** 2 for a, b in zip(retry_xy, retry_actual_xy)) ** 0.5
                if error > 0.025:
                    raise AssertionError(f"Camera localization error {error:.3f}m: measured={retry_xy}, actual={retry_actual_xy}")
            recovery.restart_at(
                motion=motion, palm_xyz=tuple(float(v) for v in data.mocap_pos[0]),
                soap_xy=measured_xy,
            )
        if recovery.status in ("success", "failed"):
            result = dict(
                status=recovery.status, attempt=recovery.attempt,
                injected=injected, missed=missed,
                retry_xy=retry_xy, retry_actual_xy=retry_actual_xy,
                error=recovery.last_error, final_xyz=xyz, in_tray=placed,
            )
            print(f"CAMERA RECOVERY CHECK: {result}", flush=True)
            return result
    raise AssertionError("RGB camera recovery exceeded step budget")


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "bathroom_physical.xml"))
    renderer = mujoco.Renderer(model, width=640, height=480)
    try:
        result = run_camera_recovery(model, renderer)
    finally:
        renderer.close()
    if not (result["injected"] and result["status"] == "success"
            and result["attempt"] == 2 and result["in_tray"]
            and result["retry_xy"] is not None):
        raise AssertionError(f"Angled RGB camera failed to regrasp physically: {result}")
    print("OK: tilted RGB camera observed dropped soap and guided successful second pickup")


if __name__ == "__main__":
    main()
