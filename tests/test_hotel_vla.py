"""XML/contract tests do not fetch Menagerie meshes or SmolVLA weights."""
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mobile_demo.hotel_vla import (
    ACTION_LOW, ACTION_HIGH, ACTUATORS, BIN_XY, CAMERAS, COUNTER_Z,
    CUBES, SPAWNS, STATE_JOINTS, compose_hotel_xml, validate_action,
)


FAKE_FULL = """<mujoco model="tidybot">
 <compiler meshdir="assets"/>
 <asset><mesh file="base/body.stl"/></asset>
 <worldbody><body name="base_link"><joint name="joint_x"/>
 <joint name="joint_y"/><joint name="joint_th"/>
 <camera name="base"/><camera name="wrist"/>
 <site name="pinch_site"/><body name="bracelet_link"/></body></worldbody>
 <keyframe>
  <key name="home" qpos="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" ctrl="0"/>
  <key name="retract" qpos="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" ctrl="0"/>
 </keyframe>
</mujoco>"""


class HotelTests(unittest.TestCase):
    def setUp(self):
        self.root = ET.fromstring(compose_hotel_xml(
            xml_text=FAKE_FULL,
            assets={"base/body.stl": Path("/tmp/verified_tidybot_body.stl")}))

    def test_three_cubes_initially_on_counter(self):
        self.assertEqual(CUBES, 3)
        self.assertGreater(SPAWNS[0][0][2], COUNTER_Z)
        self.assertGreater(SPAWNS[1][0][2], COUNTER_Z)
        self.assertGreater(SPAWNS[2][0][2], COUNTER_Z)
        ys = [xyz[1] for xyz, _ in SPAWNS]
        self.assertGreater(min(abs(a-b) for i,a in enumerate(ys)
                               for b in ys[i+1:]), .20)
        for i in range(CUBES):
            body = self.root.find(f"./worldbody/body[@name='hotel_red_{i}']")
            self.assertIsNotNone(body)
            self.assertIsNotNone(body.find("freejoint"))
            self.assertEqual(body.find("geom").get("size"), "0.012 0.012 0.012")
        self.assertEqual(len(self.root.findall("./worldbody/body[freejoint]")), 3)

    def test_sink_recess_blue_bin_and_elevated_front_camera(self):
        names = {g.get("name"): g for g in self.root.findall("./worldbody/geom")}
        self.assertIn("sink_bottom", names)
        self.assertIn("counter_front", names)
        self.assertIn("hotel_bin_bottom", names)
        self.assertIn("hotel_floor", names)
        self.assertGreater(float(names["hotel_bin_bottom"].get("pos").split()[2]), COUNTER_Z)
        self.assertNotIn("vanity_cabinet", names)  # would fill the sink cavity
        self.assertIsNotNone(self.root.find(".//geom[@name='hotel_cargo_clearance_proxy']"))
        proxy = self.root.find(".//geom[@name='hotel_cargo_clearance_proxy']")
        self.assertEqual(proxy.get("contype"), "0")
        self.assertEqual(proxy.get("conaffinity"), "0")
        self.assertIsNotNone(self.root.find(".//camera[@name='hotel_front']"))
        self.assertEqual(set(CAMERAS), {"hotel_front", "wrist"})

    def test_native_keyframes_extended_without_losing_robot_positions(self):
        for key in self.root.findall("./keyframe/key"):
            coords = np.asarray([float(x) for x in key.get("qpos").split()])
            self.assertEqual(coords.shape, (18 + 7 * CUBES,))
            self.assertTrue(np.allclose(coords[:18], 0))
            for i in range(CUBES):
                quat = coords[18 + i * 7 + 3:18 + i * 7 + 7]
                self.assertAlmostEqual(float(np.linalg.norm(quat)), 1., places=6)

    def test_strict_vla_contract(self):
        self.assertEqual(len(ACTUATORS), 11)
        self.assertEqual(len(STATE_JOINTS), 11)
        good = (ACTION_LOW + ACTION_HIGH) / 2
        np.testing.assert_allclose(validate_action(good), good)
        with self.assertRaises(ValueError):
            validate_action(np.zeros(6))  # legacy SO-101 checkpoint
        with self.assertRaises(ValueError):
            validate_action(np.r_[np.zeros(10), 300.])
        with self.assertRaises(ValueError):
            validate_action(np.r_[np.zeros(10), np.nan])


if __name__ == "__main__":
    unittest.main()
