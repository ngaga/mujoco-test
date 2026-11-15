"""Seeded physical manipulation robustness measured through RGB observations.

Unlike robustness.py (which uses privileged MuJoCo XY), both the initial
pickup and any recovery here use the actual rendered RGB camera image.
MuJoCo pose/contact truth is used only to score outcomes and log errors.

Run on Linux with Mesa:
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
      python rgb_benchmark.py --trials 100 --stress-trials 24
On macOS with a graphical desktop, use mjpython rgb_benchmark.py instead.
"""
import argparse
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco

from motion import HOME
from physical_grasp import ContactMonitor, PhysicalMotion, SOAP_TOP_Z, step_physical
from recovery import RecoveryManager
from robustness import (
    DEFAULT_SEED, DEFAULT_TRIALS, MODEL_PATH, STEPS_LIMIT,
    in_tray, make_scenarios, summarize,
)
from vision import estimate_soap_xy

ROOT = Path(__file__).resolve().parent
NOMINAL_MASS_KG = 0.297  # 0.15 * 0.10 * 0.044 * 450 kg/m^3
SOAP_HALF_SIZE = (0.075, 0.05, 0.022)
STRESS_SEED_OFFSET = 10007
OBSERVE_EVERY_STEPS = 150   # 0.3 simulated seconds, limited render calls


@dataclass(frozen=True)
class RGBScenario:
    index: int
    x: float
    y: float
    injected_drop: bool
    group: str = "nominal"
    yaw_deg: float = 0.0
    mass_kg: float = NOMINAL_MASS_KG
    shape: str = "box"

    def as_dict(self):
        return {
            "index": self.index,
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "injected_drop": self.injected_drop,
            "group": self.group,
            "yaw_deg": round(self.yaw_deg, 3),
            "mass_kg": round(self.mass_kg, 4),
            "shape": self.shape,
        }


def nominal_scenarios(count=DEFAULT_TRIALS, seed=DEFAULT_SEED):
    """Reuse identical XY and fault allocation as the earlier GT benchmark."""
    return [
        RGBScenario(s.index, s.x, s.y, s.injected_drop)
        for s in make_scenarios(count, seed)
    ]


def stress_scenarios(count=24, seed=DEFAULT_SEED):
    """Four explicit OOD conditions, each with comparable forced-drop cases.

    Alter shape/mass/yaw independently as well as in combination, preserving
    the conservative reachable XY domain for a fair first stress evaluation.
    """
    if count < 0:
        raise ValueError("stress scenario count must be nonnegative")
    rng = random.Random(seed + STRESS_SEED_OFFSET)
    groups = ("yaw", "mass", "ellipsoid", "combined")
    result = []
    for index in range(count):
        group = groups[index % len(groups)]
        yaw = rng.uniform(-20, 20) if group in ("yaw", "combined") else 0.0
        mass = rng.uniform(0.18, 0.42) if group in ("mass", "combined") else NOMINAL_MASS_KG
        shape = "ellipsoid" if group in ("ellipsoid", "combined") else "box"
        # Two out of every four cases inject a grip failure. Fault assignment
        # is independent of which variation is being measured.
        inject = ((index // 4 + index % 4) % 2 == 0)
        result.append(RGBScenario(
            1000 + index, rng.uniform(-0.32, -0.22),
            rng.uniform(-0.045, 0.055), inject,
            group, yaw, mass, shape,
        ))
    return result


def model_for_scenario(scenario, xml_source=None):
    """Compile physically consistent collision size, density and inertia.

    Only *before* resetting/running the trial. No runtime editing of soap
    geom, freejoint, body inertia or dynamic state.
    """
    if xml_source is None:
        xml_source = MODEL_PATH.read_text()
    if scenario.shape not in ("box", "ellipsoid"):
        raise ValueError("Unsupported soap geom shape: " + str(scenario.shape))
    if not 0.05 <= scenario.mass_kg <= 1.0:
        raise ValueError("Soap mass must be in 0.05..1.0 kg")
    root = ET.fromstring(xml_source)
    geom = root.find(".//body[@name='soap']/geom[@name='soap_geom']")
    if geom is None:
        raise ValueError("soap_geom was not found in physical scene")
    geom.set("type", scenario.shape)
    hx, hy, hz = SOAP_HALF_SIZE
    volume = (8.0 * hx * hy * hz if scenario.shape == "box"
              else (4.0 / 3.0) * math.pi * hx * hy * hz)
    geom.set("density", f"{scenario.mass_kg / volume:.10f}")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def observe_rgb(model, data, renderer, camera_id):
    """Image-only centroid -> calibrated world XY.

    No reading from the free-soap joint, or from geom/body position, is
    permitted here. The known soap-top height is fixed scene calibration.
    """
    renderer.update_scene(data, camera="overhead")
    rgb = renderer.render()
    return estimate_soap_xy(
        rgb, data.cam_xpos[camera_id], data.cam_xmat[camera_id],
        float(model.cam_fovy[camera_id]), SOAP_TOP_Z,
    )


def run_rgb_trial(model, renderer, scenario, recovery_enabled=True,
                  max_steps=STEPS_LIMIT):
    """Run a single image-guided physical trial, returning all observations."""
    data = mujoco.MjData(model)
    motion = PhysicalMotion()
    monitor = ContactMonitor(model)
    manager = RecoveryManager(max_attempts=2 if recovery_enabled else 1)
    soap_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "soap_free")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    if soap_joint < 0 or camera_id < 0:
        raise ValueError("Model missing soap_free or overhead camera")
    qadr, vadr = int(model.jnt_qposadr[soap_joint]), int(model.jnt_dofadr[soap_joint])
    mujoco.mj_resetData(model, data)
    data.qpos[qadr:qadr + 2] = [scenario.x, scenario.y]
    half_angle = math.radians(scenario.yaw_deg) / 2
    data.qpos[qadr + 3:qadr + 7] = [math.cos(half_angle), 0, 0, math.sin(half_angle)]
    data.mocap_pos[0] = HOME
    mujoco.mj_forward(model, data)

    observations = 0
    detections = 0
    measured_errors = []  # diagnostics only, never fed to the controller
    initial_xy = None
    retry_xy = None
    retry_actual_xy = None
    reason = None
    injected = False
    pose_unchanged = True

    def view():
        nonlocal observations, detections
        xy = observe_rgb(model, data, renderer, camera_id)
        observations += 1
        if xy is not None:
            detections += 1
            # Physics truth here is used solely for the benchmark report.
            actual = data.qpos[qadr:qadr + 2]
            measured_errors.append(math.hypot(xy[0] - float(actual[0]),
                                               xy[1] - float(actual[1])))
        return xy

    initial_xy = view()
    if initial_xy is None:
        return {
            **scenario.as_dict(),
            "recovery_enabled": recovery_enabled,
            "status": "failed",
            "attempts": 0,
            "fault_triggered": False,
            "reason": "Savon non détecté sur la première image RGB",
            "grasp_verified": False,
            "final_xyz": [round(float(v), 5) for v in data.qpos[qadr:qadr + 3]],
            "sim_seconds": 0.0,
            "rendered_frames": observations,
            "detections": detections,
            "initial_camera_xy": None,
            "retry_camera_xy": None,
            "retry_actual_xy": None,
            "mean_camera_error_m": None,
            "max_camera_error_m": None,
            "replanned": False,
            "pose_unchanged_on_replan": True,
        }

    motion.retry_from(HOME, initial_xy)
    manager.start()

    for tick in range(max_steps):
        if (scenario.injected_drop and not injected
                and manager.attempt == 1
                and motion.index == 4 and motion.elapsed >= 0.25):
            injected = True
        forced_open = (scenario.injected_drop and injected
                       and manager.attempt == 1
                       and manager.status == "running")
        step_physical(model, data, motion, force_open=forced_open)
        # As in the live app, update the damped approach servo while the
        # object remains partially visible. Missing detections keep last XY.
        if (manager.status == "running" and motion.running and motion.index < 2
                and tick % OBSERVE_EVERY_STEPS == 0):
            current_xy = view()
            if current_xy is not None and -0.6 < current_xy[0] < -0.08:
                motion.retarget_soap(current_xy)

        xyz = tuple(float(v) for v in data.qpos[qadr:qadr + 3])
        linear_velocity = tuple(float(v) for v in data.qvel[vadr:vadr + 3])
        left, right = monitor.contacts(data)
        wants_observation = manager.after_step(
            motion=motion, soap_xyz=xyz, linear_velocity=linear_velocity,
            both_contacts=left and right, in_tray=in_tray(xyz),
            dt=float(model.opt.timestep),
        )
        if wants_observation:
            soap_before = tuple(float(v) for v in data.qpos[qadr:qadr + 7])
            candidate = view()
            manager.note_visual_observation(candidate)
            if candidate is not None:
                retry_xy = candidate
                retry_actual_xy = xyz[:2]
            manager.restart_at(
                motion=motion, palm_xyz=tuple(float(v) for v in data.mocap_pos[0]),
                soap_xy=candidate,
            )
            soap_after = tuple(float(v) for v in data.qpos[qadr:qadr + 7])
            pose_unchanged = pose_unchanged and soap_before == soap_after
            if not pose_unchanged:
                raise AssertionError("Replanning modified the soap free-joint state")
        if manager.status in ("success", "failed"):
            reason = manager.last_error
            break
    else:
        reason = "Exceeded simulation step budget"

    status = manager.status if manager.status in ("success", "failed") else "timeout"
    if manager.attempt > manager.max_attempts:
        raise AssertionError("Controller exceeded bounded retry count")
    if status == "success" and not (manager.grasp_verified and in_tray(xyz)):
        raise AssertionError("Task reported success without verified contact/lift/placement")
    return {
        **scenario.as_dict(),
        "recovery_enabled": recovery_enabled,
        "status": status,
        "attempts": manager.attempt,
        "fault_triggered": injected,
        "reason": reason,
        "grasp_verified": bool(manager.grasp_verified),
        "final_xyz": [round(v, 5) for v in xyz],
        "sim_seconds": round((tick + 1) * float(model.opt.timestep), 3),
        "rendered_frames": observations,
        "detections": detections,
        "initial_camera_xy": [round(v, 5) for v in initial_xy],
        "retry_camera_xy": [round(v, 5) for v in retry_xy] if retry_xy else None,
        "retry_actual_xy": [round(v, 5) for v in retry_actual_xy] if retry_actual_xy else None,
        "mean_camera_error_m": (round(sum(measured_errors) / len(measured_errors), 5)
                                if measured_errors else None),
        "max_camera_error_m": round(max(measured_errors), 5) if measured_errors else None,
        "replanned": bool(manager.retry_positions),
        "pose_unchanged_on_replan": pose_unchanged,
    }


def paired_rgb_benchmark(scenarios, width=640, height=480, verbose=False):
    """Match initial world, camera, shape and fault for each policy pair."""
    results_no, results_yes = [], []
    initial_xml = MODEL_PATH.read_text()
    cached_nominal = None
    nominal_camera = None
    try:
        for index, scenario in enumerate(scenarios):
            is_nominal = (scenario.shape == "box"
                          and scenario.mass_kg == NOMINAL_MASS_KG)
            if is_nominal:
                if cached_nominal is None:
                    cached_nominal = model_for_scenario(scenario, initial_xml)
                    nominal_camera = mujoco.Renderer(
                        cached_nominal, width=width, height=height
                    )
                model, renderer = cached_nominal, nominal_camera
            else:
                model = model_for_scenario(scenario, initial_xml)
                renderer = mujoco.Renderer(model, width=width, height=height)
            try:
                before = run_rgb_trial(model, renderer, scenario, recovery_enabled=False)
                after = run_rgb_trial(model, renderer, scenario, recovery_enabled=True)
                results_no.append(before)
                results_yes.append(after)
                if verbose:
                    print(
                        f"case={index + 1}/{len(scenarios)} {scenario.group} "
                        f"shape={scenario.shape} yaw={scenario.yaw_deg:.1f} "
                        f"mass={scenario.mass_kg:.3f} fault={scenario.injected_drop} "
                        f"no={before['status']} yes={after['status']}",
                        flush=True,
                    )
            finally:
                if not is_nominal:
                    renderer.close()
    finally:
        if nominal_camera is not None:
            nominal_camera.close()
    base_summary = summarize(results_no)
    recovery_summary = summarize(results_yes)
    before_lookup = {r["index"]: r for r in results_no}
    return {
        "positions": len(scenarios),
        "no_recovery": base_summary,
        "with_recovery": recovery_summary,
        "rescued_indices": [r["index"] for r in results_yes
                            if r["status"] == "success"
                            and before_lookup[r["index"]]["status"] != "success"],
        "regressed_indices": [r["index"] for r in results_yes
                              if r["status"] != "success"
                              and before_lookup[r["index"]]["status"] == "success"],
        "scenarios": [s.as_dict() for s in scenarios],
        "no_recovery_results": results_no,
        "recovery_results": results_yes,
    }


def by_stress_group(section):
    """Aggregate stress outcomes separately; never average away failures."""
    groups = {}
    for group in ("yaw", "mass", "ellipsoid", "combined"):
        baseline = [x for x in section["no_recovery_results"] if x["group"] == group]
        recovery = [x for x in section["recovery_results"] if x["group"] == group]
        groups[group] = {
            "without_recovery": summarize(baseline),
            "with_recovery": summarize(recovery),
        }
    return groups


def benchmark(trials=DEFAULT_TRIALS, stress_trials=24, seed=DEFAULT_SEED,
              width=640, height=480, verbose=False):
    nominal = paired_rgb_benchmark(
        nominal_scenarios(trials, seed), width, height, verbose,
    )
    stress = paired_rgb_benchmark(
        stress_scenarios(stress_trials, seed), width, height, verbose,
    ) if stress_trials else None
    return {
        "protocol": (
            "Paired actual RGB observation on every first and retry grasp; "
            "real MuJoCo contacts/lift/placement score; half injected drops"
        ),
        "seed": seed,
        "camera": "overhead (tilted oblique, 640x480 RGB by default)",
        "pose_truth_used_for_control": False,
        "mujooco_truth_used_for_scoring": True,
        "scoring": "MuJoCo simulator: bilateral contacts, lift and tray placement",
        "nominal": nominal,
        "stress": stress,
        "stress_by_group": by_stress_group(stress) if stress else None,
        "limitations": [
            "Soap color segmentation specific to yellow soap",
            "Fixed soap top-plane calibration, not 6DoF pose estimation",
            "Conservative reachable XY rectangle for nominal and stress",
            "No random lighting, occluders, clutter, viewpoint or gripper geometry",
            "MuJoCo ground truth required only for outcome scoring",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--stress-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=ROOT / "rgb_robustness_report.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    report = benchmark(
        args.trials, args.stress_trials, args.seed,
        args.width, args.height, args.verbose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name in ("nominal", "stress"):
        section = report[name]
        if section:
            x, y = section["no_recovery"], section["with_recovery"]
            print(
                f"RGB {name}: N={section['positions']}, "
                f"without={x['successes']}/{x['runs']} ({x['success_rate_pct']}%), "
                f"with={y['successes']}/{y['runs']} ({y['success_rate_pct']}%); "
                f"rescued={len(section['rescued_indices'])}, "
                f"regressed={len(section['regressed_indices'])}",
                flush=True,
            )
    if report["stress_by_group"]:
        for group, rows in report["stress_by_group"].items():
            print(
                f"RGB stress {group}: "
                f"{rows['with_recovery']['successes']}/"
                f"{rows['with_recovery']['runs']} with recovery", flush=True
            )
    print(f"Full unfiltered RGB trial results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
