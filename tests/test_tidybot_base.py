"""Unit tests for Stanford TidyBot model composition (no network required)."""
from pathlib import Path
import xml.etree.ElementTree as ET
import unittest
from unittest.mock import patch
from mobile_demo.core import SCENE
from mobile_demo.tidybot_base import MESH_SHA, mount_tidybot_base
from mobile_demo.so101 import MOUNT_XYZ, add_visual_mount_bracket


class TidyBotCompositionTests(unittest.TestCase):
    def test_so101_visual_bracket_bridges_deck_without_changing_physics(self):
        root = ET.parse(SCENE).getroot()
        base = root.find("./worldbody/body[@name='mobile_base']")
        assert base is not None
        original_joints = [j.attrib.copy() for j in base.findall("joint")]
        add_visual_mount_bracket(base)
        names = ("so101_mount_brace_left", "so101_mount_brace_right",
                 "so101_mount_plate")
        for name in names:
            geom = base.find(f"./geom[@name='{name}']")
            self.assertIsNotNone(geom)
            self.assertEqual(geom.get("mass"), "0")
            self.assertEqual(geom.get("contype"), "0")
            self.assertEqual(geom.get("conaffinity"), "0")
            self.assertEqual(geom.get("group"), "2")
        plate = base.find("./geom[@name='so101_mount_plate']")
        self.assertAlmostEqual(float(plate.get("pos").split()[0]), MOUNT_XYZ[0])
        # The visible bracket may NOT add joints, motor controls or contacts.
        self.assertEqual([j.attrib for j in base.findall("joint")], original_joints)

    def test_replaces_entire_custom_base_with_official_hull(self):
        fake = {name: Path("/tmp") / name for name in MESH_SHA}
        root = ET.parse(SCENE).getroot()
        with patch("mobile_demo.tidybot_base.ensure_assets", return_value=fake):
            mount_tidybot_base(root)
        base = root.find("./worldbody/body[@name='mobile_base']")
        names = {node.get("name") for node in base.findall("geom")}
        self.assertEqual(len(names), 10)
        self.assertIn("tidybot_body_collision", names)
        self.assertIn("tidybot_body_visual", names)
        self.assertNotIn("wheel_a", names)
        self.assertEqual(
            {n.get("name") for n in base.findall("joint")},
            {"base_x", "base_y", "base_yaw"},
        )
        self.assertIsNotNone(base.find("./body[@name='reach_stage']"))
        self.assertIsNotNone(base.find("./camera[@name='front']"))
        self.assertEqual(
            len(root.findall("./asset/mesh")), 5)
        self.assertEqual(
            root.find("./worldbody/body[@name='mobile_base']/geom[@name='tidybot_body_visual']").get("mass"),
            "60",
        )
        self.assertEqual(
            root.find("./worldbody/body[@name='mobile_base']/geom[@name='tidybot_body_collision']").get("contype"),
            "1",
        )
        # Idempotent, so programmatic builders cannot add geoms twice.
        with patch("mobile_demo.tidybot_base.ensure_assets", return_value=fake):
            mount_tidybot_base(root)
        self.assertEqual(len(base.findall("geom")), 10)


if __name__ == "__main__":
    unittest.main()
