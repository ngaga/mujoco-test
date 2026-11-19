"""Offline structural regression of the two Stanford TidyBot robot choices."""
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

from mobile_demo.tidybot_full import (
    FULL_MESH_SHA, FULL_XML_SHA, compose_full_xml,
)

FAKE_OFFICIAL_MODEL = """
<mujoco model="tidybot">
  <compiler angle="radian" meshdir="assets"/>
  <asset>
    <mesh file="base/body.stl"/>
    <mesh file="gen3/shoulder_link.stl"/>
    <mesh class="2f85" file="2f85/pad.stl"/>
  </asset>
  <worldbody>
    <body name="base_link">
      <joint name="joint_x" type="slide"/>
      <joint name="joint_y" type="slide"/>
      <camera name="base"/>
      <body name="gen3/base_link">
        <joint name="joint_1"/>
        <body name="bracelet_link">
          <camera name="wrist"/>
          <site name="pinch_site"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="joint_x" joint="joint_x"/>
    <position name="joint_1" joint="joint_1"/>
  </actuator>
</mujoco>
"""


class FullTidyBotCompositionTests(unittest.TestCase):
    def setUp(self):
        self.assets = {
            "base/body.stl": Path("/tmp/tidybot/base/body.stl"),
            "gen3/shoulder_link.stl": Path("/tmp/tidybot/gen3/shoulder_link.stl"),
            "2f85/pad.stl": Path("/tmp/tidybot/2f85/pad.stl"),
        }

    def test_full_mode_keeps_gen3_and_gripper_and_earlybots_arena(self):
        xml = compose_full_xml(3, xml_text=FAKE_OFFICIAL_MODEL, assets=self.assets)
        root = ET.fromstring(xml)
        self.assertIsNotNone(root.find("./worldbody/body[@name='base_link']"))
        self.assertIsNotNone(root.find(".//body[@name='gen3/base_link']"))
        self.assertIsNotNone(root.find(".//camera[@name='base']"))
        self.assertIsNotNone(root.find(".//camera[@name='wrist']"))
        self.assertIsNotNone(root.find(".//site[@name='pinch_site']"))
        self.assertIsNotNone(root.find("./worldbody/body[@name='blue_bin']"))
        for index in range(3):
            body = root.find(f"./worldbody/body[@name='red_{index}']")
            self.assertIsNotNone(body)
            self.assertEqual(body.find("geom").get("size"), "0.012 0.012 0.012")
        self.assertIsNone(root.find("./worldbody/body[@name='red_3']"))
        self.assertIsNone(root.find("./worldbody/body[@name='red_4']"))
        self.assertEqual(len(root.findall("./asset/mesh")), 3)
        for element in root.findall("./asset/mesh"):
            self.assertTrue(Path(element.get("file")).is_absolute())
        self.assertIsNone(root.find("compiler").get("meshdir"))

    def test_full_asset_manifest_is_pinned_and_unambiguous(self):
        self.assertEqual(len(FULL_MESH_SHA), 15)
        self.assertEqual(len(FULL_XML_SHA), 40)
        for key, sha in FULL_MESH_SHA.items():
            self.assertIn(key.split("/")[0], ("gen3", "2f85"))
            self.assertEqual(len(sha), 40)

    def test_incomplete_official_assets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing full TidyBot mesh"):
            compose_full_xml(
                1,
                xml_text=FAKE_OFFICIAL_MODEL,
                assets={"base/body.stl": Path("/tmp/body.stl")},
            )

    def test_invalid_cube_counts_are_rejected(self):
        for n in (0, 6, -1):
            with self.assertRaises(ValueError):
                compose_full_xml(n, xml_text=FAKE_OFFICIAL_MODEL, assets=self.assets)

    def test_full_robot_cannot_claim_so101_task_success(self):
        from mobile_demo.so101 import main
        with self.assertRaises(SystemExit):
            main(["--robot", "tidybot", "--require-success"])
        with self.assertRaises(SystemExit):
            main(["--robot", "tidybot", "--perception", "rgb-nav"])


if __name__ == "__main__":
    unittest.main()
