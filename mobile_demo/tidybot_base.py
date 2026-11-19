"""Use the official Stanford TidyBot (2023) Menagerie CAD hull in mobile demos.

The original Menagerie base has x/y/yaw servo joints, NOT simulated caster
drive dynamics. We preserve our navigation API while replacing ALL custom
LeKiwi-ish base geoms with the exact upstream visual and collision meshes.

Only the 5 binary STL files are downloaded (once) at runtime. The vendor's
original base.xml and MIT license are committed alongside this helper.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

# Upstream pin: google-deepmind/mujoco_menagerie, main @ 2026-11-19.
MENAGERIE_SHA = "370c43965323c0944b433b202aac5058b5f907a9"
MENAGERIE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    + MENAGERIE_SHA + "/stanford_tidybot/assets/base/"
)
# Official git blob SHA1 (computed with Git's "blob {n}\0" preimage).
MESH_SHA = {
    "arm_plate.stl": "d10cac797d9d7d2d54e9098fa6cb8d36d95dd2e2",
    "body.stl": "f0bab861a5ce1e69b205d180bb72e2a394c2e83b",
    "bottom_plate.stl": "055b9b2ed6d615df5d370768b0d61f85f0c37589",
    "bumper.stl": "d96eabbe7524f7789a224589090066ebf03cab18",
    "top_plate.stl": "3197ed61101a104b9fa97f5dd4426cf9e41da797",
}
HERE = Path(__file__).resolve().parent
CACHE = HERE / "assets" / "tidybot"
BASE_XML = HERE / "vendor" / "tidybot" / "base.xml"


def _git_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def ensure_assets() -> dict[str, Path]:
    """Fetch only missing official STLs, SHA-check, and cache for offline runs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, expected in MESH_SHA.items():
        dest = CACHE / name
        if dest.exists() and _git_hash(dest.read_bytes()) == expected:
            paths[name] = dest
            continue
        try:
            with urllib.request.urlopen(MENAGERIE_URL + name, timeout=30) as resp:
                payload = resp.read()
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"Missing verified TidyBot asset {name}. An initial internet "
                "connection is required; run python -m mobile_demo.tidybot_base "
                "--fetch, then retry offline."
            ) from exc
        if _git_hash(payload) != expected:
            raise RuntimeError(f"TidyBot asset checksum mismatch: {name}")
        dest.write_bytes(payload)
        paths[name] = dest
    return paths


def mount_tidybot_base(root: ET.Element, assets: dict[str, Path] | None = None) -> ET.Element:
    """Replace only the old base shell; preserve scene, joints, cameras and arm.

    Assets, meshes and geometry are derived from the official base.xml. Visual
    and physical collision geoms remain separate and coincident as upstream.
    """
    if assets is None:
        assets = ensure_assets()
    upstream = ET.parse(BASE_XML).getroot()
    base = root.find("./worldbody/body[@name='mobile_base']")
    if base is None:
        raise ValueError("Expected mobile_base in EarlyBots scene template")
    if root.find("./asset/mesh[@name='tidybot_body']") is not None:
        return root

    output_assets = root.find("asset")
    if output_assets is None:
        output_assets = ET.Element("asset")
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Scene missing worldbody")
        root.insert(list(root).index(worldbody), output_assets)

    src_assets = upstream.find("asset")
    if src_assets is None:
        raise ValueError("Upstream TidyBot base missing asset entries")
    for item in src_assets:
        clone = copy.deepcopy(item)
        if clone.tag == "mesh":
            filename = Path(clone.attrib["file"]).name
            if filename not in assets:
                raise ValueError(f"Unrecognized TidyBot mesh {filename}")
            clone.set("name", "tidybot_" + Path(filename).stem)
            clone.set("file", str(Path(assets[filename]).resolve()))
        elif clone.tag == "material":
            clone.set("name", "tidybot_" + clone.attrib["name"])
        else:
            continue
        output_assets.append(clone)

    src_body = upstream.find("./worldbody/body[@name='base_link']")
    if src_body is None:
        raise ValueError("Upstream TidyBot base missing body")
    original_geoms = list(base.findall("geom"))
    for geom in original_geoms:
        base.remove(geom)
    for geom in src_body.findall("geom"):
        clone = copy.deepcopy(geom)
        kind = clone.attrib.pop("class")
        original_mesh = clone.attrib["mesh"]
        clone.set("name", f"tidybot_{original_mesh}_{kind}")
        clone.set("mesh", f"tidybot_{original_mesh}")
        clone.set("type", "mesh")
        if "material" in clone.attrib:
            clone.set("material", "tidybot_" + clone.attrib["material"])
        # Explicit defaults from the official model, avoiding name clashes
        # with the separate imported SO-101's vendor defaults.
        clone.set("group", "2" if kind == "visual" else "3")
        clone.set("mass", clone.attrib.get("mass", "0"))
        clone.set("contype", "0" if kind == "visual" else "1")
        clone.set("conaffinity", "0" if kind == "visual" else "1")
        base.append(clone)

    # The old LeKiwi-like proxy was ~3 kg; its X/Y position servos were tuned
    # for that inertia (kp=550, kv=85 N*s/m). TidyBot's official CAD body is
    # 60 kg, so reusing those gains makes it underdamped. The base sways while
    # the SO-101 approaches a cube and can push the target away. Retune the
    # *still idealized* planar servos for the modeled mass; this is NOT a
    # measured powered-caster motor model or a physically validated drive.
    # For 60 kg, kp=6000 N/m and kv=1200 N*s/m are near critical damping.
    for name in ("drive_x", "drive_y"):
        servo = root.find(f"./actuator/position[@name='{name}']")
        if servo is None:
            raise ValueError(f"Expected EarlyBots planar actuator {name}")
        servo.set("kp", "6000")
        servo.set("kv", "1200")
        servo.set("forcelimited", "true")
        servo.set("forcerange", "-900 900")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Cache and SHA-verify official STL meshes")
    args = parser.parse_args()
    if not args.fetch:
        parser.error("Specify --fetch")
    for name, path in ensure_assets().items():
        print(f"{name}: verified ({path})")


if __name__ == "__main__":
    main()
