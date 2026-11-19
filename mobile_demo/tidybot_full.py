"""Official complete Stanford TidyBot 2023 (Kinova Gen3 + Robotiq 2F-85).

This independent MuJoCo mode uses the WHOLE pinned Menagerie tidybot.xml,
not the SO-101 controller. It demonstrates the model, cameras and safe small
joint movements. It does NOT yet contain a trained/planned full-TidyBot
pick-and-place policy. The other --robot mode provides the experimental
SO-101 cube collection task on the same TidyBot CAD chassis.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .core import CUBE_COUNT, SCENE
from .tidybot_base import CACHE, MENAGERIE_SHA, MESH_SHA as BASE_MESH_SHA, _git_hash, ensure_assets

ROOT_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    + MENAGERIE_SHA + "/stanford_tidybot/"
)
FULL_XML_SHA = "6b37af55d21552190548dbac8f9392a2388e90c9"
FULL_XML = CACHE / "full_tidybot.xml"

# Pin exact Git blobs, NOT merely file names or branch aliases. The extra
# bracelet_no_vision_link is not referenced by Menagerie's TidyBot XML.
FULL_MESH_SHA = {
    "gen3/base_link.stl": "0a35a8ae68fa0b8d24507f2cd2a142a59b8fc91e",
    "gen3/shoulder_link.stl": "69ecb7695c73980fa9e76563d5a66fca373db44a",
    "gen3/half_arm_1_link.stl": "1df304c0eb9e8630f3be09b39154c1955ad0771b",
    "gen3/half_arm_2_link.stl": "78bc722ab2d8ac217b57dc2b85860a9597c771ee",
    "gen3/forearm_link.stl": "ac791a3548a21d8ae1650b4b14d173605e6f85bb",
    "gen3/spherical_wrist_1_link.stl": "3bbf04a6de017405f402d8b95c3b22d441389a57",
    "gen3/spherical_wrist_2_link.stl": "14bbff58f2f200dd1e1b97ba5226da3f768b69e4",
    "gen3/bracelet_with_vision_link.stl": "146b69b50b2fd61546c3dc7a4860a037874f084e",
    "2f85/base.stl": "1b3bb6af165ee13361dec7b8346ba7f8b6c6b22d",
    "2f85/driver.stl": "57595939f486f75f55dc2ef49b8f805d56f78f58",
    "2f85/coupler.stl": "170b0dad58d839c2d2f7d1d8328df4413a76057d",
    "2f85/follower.stl": "3cda9be71fb61d292b3c98848f0d820a11610422",
    "2f85/pad.stl": "fa0ef62b3dea4dd2e0c55005beb744c8fafc0066",
    "2f85/silicone_pad.stl": "9a0de4352fd9a9adbd48ee8768d78553b9a407f8",
    "2f85/spring_link.stl": "d92de810c4039750286e788d065008090086b06e",
}


def _fetch_verified(relative: str, expected_sha: str, local: Path) -> Path:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_file() and _git_hash(local.read_bytes()) == expected_sha:
        return local
    try:
        with urllib.request.urlopen(ROOT_URL + relative, timeout=45) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Missing TidyBot official asset {relative}; fetch it once with "
            "'python -m mobile_demo.tidybot_full --fetch' before using offline."
        ) from exc
    if _git_hash(payload) != expected_sha:
        raise RuntimeError(f"Pinned Stanford TidyBot checksum mismatch: {relative}")
    local.write_bytes(payload)
    return local


def ensure_full_assets() -> tuple[Path, dict[str, Path]]:
    """Cache all 20 official STL meshes and the full robot XML, SHA verified."""
    base = ensure_assets()
    mapping = {"base/" + name: path for name, path in base.items()}
    for relative, sha in FULL_MESH_SHA.items():
        mapping[relative] = _fetch_verified(
            "assets/" + relative, sha, CACHE / relative
        )
    xml = _fetch_verified("tidybot.xml", FULL_XML_SHA, FULL_XML)
    return xml, mapping


def compose_full_xml(
    cubes: int = CUBE_COUNT,
    *,
    xml_text: str | None = None,
    assets: dict[str, Path] | None = None,
) -> str:
    """Attach EarlyBots' arena to the unmodified official articulated robot.

    Injected xml_text/assets permit network-independent structural tests.
    All robot arm joints, tendon couplings, equality constraints and actuators
    are preserved verbatim from upstream. Only file paths and world geometry
    are changed for the standalone full robot simulation.
    """
    if not 1 <= cubes <= CUBE_COUNT:
        raise ValueError(f"cubes must be in 1..{CUBE_COUNT}")
    if xml_text is None:
        full_xml, verified_assets = ensure_full_assets()
        xml_text = full_xml.read_text(encoding="utf-8")
        assets = verified_assets
    if assets is None:
        raise ValueError("When supplying XML, supply its verified mesh paths")
    root = ET.fromstring(xml_text)
    root.set("model", "EarlyBots | full Stanford TidyBot Gen3 and Robotiq")
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Full TidyBot missing compiler")
    compiler.attrib.pop("meshdir", None)
    for mesh in root.findall("./asset/mesh"):
        relative = mesh.get("file")
        if relative not in assets:
            raise ValueError(f"Missing full TidyBot mesh: {relative}")
        mesh.set("file", str(Path(assets[relative]).resolve()))

    official_world = root.find("worldbody")
    assert official_world is not None, "Missing official TidyBot robot body"
    scene = ET.parse(SCENE).getroot()
    source_world = scene.find("worldbody")
    assert source_world is not None
    # Keep the original TidyBot, add the familiar red cubes and blue bin.
    for geom in source_world.findall("geom"):
        if geom.get("name") == "floor":
            official_world.append(copy.deepcopy(geom))
    cube_key_qpos = []
    for body in source_world.findall("body"):
        name = body.get("name", "")
        if name == "blue_bin" or name.startswith("red_"):
            if name.startswith("red_") and int(name[4:]) >= cubes:
                continue
            copy_body = copy.deepcopy(body)
            if name.startswith("red_"):
                cube_geom = copy_body.find("geom")
                assert cube_geom is not None
                copy_body.set("pos", " ".join([*copy_body.get("pos").split()[:2], "0.0125"]))
                cube_geom.set("size", "0.012 0.012 0.012")
                cube_geom.set("mass", "0.02")
                cube_geom.set("friction", "1.2 0.005 0.0001")
                # An extra free joint contributes 7 qpos (XYZ + quaternion).
                # Extend native Stanford home/retract keyframes accordingly.
                cube_key_qpos.extend([*copy_body.get("pos").split(), "1", "0", "0", "0"])
            official_world.append(copy_body)
    for key in root.findall("./keyframe/key"):
        key.set("qpos", " ".join([*(key.get("qpos") or "").split(), *cube_key_qpos]))
    ET.SubElement(official_world, "light", {"pos": "-0.5 -1 3.2", "dir": "0.2 0.3 -1"})
    return ET.tostring(root, encoding="unicode")


def make_full_model(cubes: int = 1) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(compose_full_xml(cubes=cubes))
    for name in ("base", "wrist"):
        model.camera(name)
    for name in ("joint_x", "joint_y", "joint_1", "joint_7", "fingers_actuator"):
        model.actuator(name)
    model.site("pinch_site")
    return model


def check_model(cubes: int = 1) -> dict:
    model = make_full_model(cubes)
    return {
        "robot": "Full Stanford TidyBot (2023), official pinned Menagerie model",
        "base": "Stanford X/Y/yaw position servos",
        "arm": "Kinova Gen3, seven joints",
        "gripper": "Robotiq 2F-85 with tendon linkage",
        "cameras": ["base", "wrist"],
        "actuators": model.nu,
        "meshes": model.nmesh,
        "cubes": cubes,
        "mode": "tidybot",
        "task_policy": "not implemented: model + joints demonstration only",
    }


def execute(cubes=1, max_steps=300, viewer=False, viewer_port=8080) -> dict:
    """Real MuJoCo physics, small base/arm motions, NOT a cube-picking policy."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    model = make_full_model(cubes=cubes)
    data = mujoco.MjData(model)
    home = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home)
    home_ctrl = model.key_ctrl[home].copy()
    data.ctrl[:] = home_ctrl
    mujoco.mj_forward(model, data)
    from contextlib import nullcontext

    if viewer:
        from mujoco.studio import launch_web, viewer_protocol
        ctx = launch_web.launch(viewer_protocol.ViewerConfig(
            title="EarlyBots | Full Stanford TidyBot Kinova + Robotiq",
            http_port=viewer_port,
        ))
    else:
        ctx = nullcontext(None)
    with ctx as display:
        last = time.perf_counter()
        for step in range(max_steps):
            # Tiny commanded motions let us verify all actuated joints react.
            # The entire task policy (grasp planning) is still out of scope.
            t = float(data.time)
            data.ctrl[:] = home_ctrl
            data.ctrl[model.actuator("joint_x").id] = 0.08 * math.sin(t * 0.45)
            data.ctrl[model.actuator("joint_1").id] = home_ctrl[
                model.actuator("joint_1").id
            ] + 0.08 * math.sin(t * 0.75)
            for _ in range(round(1 / (15 * model.opt.timestep))):
                mujoco.mj_step(model, data)
            if display:
                display.sync(model, data)
                if not display.is_running():
                    break
                remaining = 1 / 15 - (time.perf_counter() - last)
                if remaining > 0:
                    time.sleep(remaining)
                last = time.perf_counter()
    return {
        "robot": "Full Stanford TidyBot, Kinova Gen3 + Robotiq 2F-85",
        "mode": "tidybot",
        "simulation_ok": bool(np.isfinite(data.qpos).all()),
        "task_success": None,
        "task_policy": "not implemented; this is an actuated model smoke demo",
        "cubes_requested": cubes,
        "cubes_delivered": None,
        "arm_actuators": 8,
        "base_actuators": 3,
        "cameras": ["base", "wrist"],
        "steps": step + 1,
        "simulation_seconds": round(float(data.time), 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--cubes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-port", type=int, default=8080)
    args = parser.parse_args(argv)
    if args.fetch:
        xml, assets = ensure_full_assets()
        print(json.dumps({"full_xml": str(xml), "verified_meshes": len(assets)}, indent=2))
        return
    report = (check_model(args.cubes) if args.check_model else
              execute(args.cubes, args.max_steps, args.viewer, args.viewer_port))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
