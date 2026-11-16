"""Yellow cube experimental control: identical cube physics, different color/task."""
import importlib.util
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLA = ROOT / "vla"
sys.path.insert(0, str(VLA))
spec = importlib.util.spec_from_file_location("eval_yellow_cube", VLA / "eval_yellow_cube.py")
cube = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cube)
from eval_soap import make_soap_xml, SOAP_RGBA, SOAP_TASK


class YellowCubeExperimentTests(unittest.TestCase):
    def test_exact_reference_object_geometry_mass_friction(self):
        source = (
            '<mujoco><asset>'
            '<material name="block_red" rgba="0.85 0.15 0.12 1"/>'
            '<material name="tray_blue" rgba="0.20 0.35 0.65 1"/>'
            '</asset><worldbody><body name="block"><freejoint name="block_free"/>'
            '<geom name="block_geom" type="box" size="0.012 0.012 0.012" '
            'material="block_red" mass="0.02" friction="1.2 0.005 0.0001" '
            'solimp="2 1 0.01" solref="0.01 1"/>'
            '</body></worldbody></mujoco>'
        )
        original = ET.fromstring(source)
        actual = ET.fromstring(make_soap_xml(source, object_size=cube.CUBE_SIZE))
        old_geom = original.find(".//geom[@name='block_geom']")
        new_geom = actual.find(".//geom[@name='block_geom']")
        # Compare every physical attribute, not just box shape.
        self.assertEqual(old_geom.attrib, new_geom.attrib)
        self.assertEqual(cube.CUBE_SIZE, (0.024, 0.024, 0.024))
        self.assertEqual(
            actual.find(".//material[@name='block_red']").get("rgba"),
            SOAP_RGBA,
        )
        self.assertEqual(
            actual.find(".//material[@name='tray_blue']").attrib,
            original.find(".//material[@name='tray_blue']").attrib,
        )

    def test_real_checkpoint_and_new_language_prompt_without_training(self):
        self.assertEqual(
            cube.MODEL_ID, "ahmedsohail2003/smolvla-so101-pickplace-v2"
        )
        self.assertEqual(
            SOAP_TASK, "Pick up the yellow soap and place it in the blue tray."
        )
        args = cube.parse_args(["--episodes", "20", "--seed", "123"])
        self.assertEqual(args.episodes, 20)
        self.assertEqual(args.seed, 123)
        with self.assertRaises(SystemExit):
            cube.parse_args(["--episodes", "0"])

    def test_report_uses_measured_results_not_expected_successes(self):
        report = {
            "checkpoint": cube.MODEL_ID,
            "task": SOAP_TASK,
            "successes": 1,
            "episodes": 2,
            "success_rate": 0.5,
            "seed": 123,
            "max_steps": 240,
            "results": [
                {"episode": 1, "success": False, "steps": 240,
                 "gif": "yellow_cube_gifs/episode_01_failure.gif"},
                {"episode": 2, "success": True, "steps": 100,
                 "gif": "yellow_cube_gifs/episode_02_success.gif"},
            ],
        }
        md = cube.markdown_report(report)
        self.assertIn("1/2 (50%)", md)
        self.assertIn("24 × 24 × 24 mm", md)
        self.assertIn("Failed episodes: 1", md)
        self.assertIn("yellow_cube_gifs/episode_02_success.gif", md)
        self.assertIn("Changed together", md)


if __name__ == "__main__":
    unittest.main()
