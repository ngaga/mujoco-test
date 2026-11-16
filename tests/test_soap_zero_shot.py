"""Offline tests for genuine zero-shot SOAP runner (no model download)."""
import contextlib
import importlib.util
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VLA = ROOT / "vla"
sys.path.insert(0, str(VLA))
spec = importlib.util.spec_from_file_location("eval_soap", VLA / "eval_soap.py")
soap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soap)


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.data, dim))

    def permute(self, *axes):
        return Tensor(self.data.transpose(axes))

    def float(self):
        return Tensor(self.data.astype(np.float32))

    def __truediv__(self, num):
        return Tensor(self.data / num)

    def to(self, device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class Torch:
    from_numpy = staticmethod(Tensor)
    no_grad = staticmethod(contextlib.nullcontext)


def observation():
    return {
        "state": np.arange(6, dtype=np.float32),
        "images": {
            "front": np.full((224, 224, 3), 100, dtype=np.uint8),
            "wrist": np.full((224, 224, 3), 150, dtype=np.uint8),
        },
    }


class FakeEnv:
    def __init__(self):
        self.model = Mock(nu=6)
        self.block_pos = np.array([0.01, -0.22, 0.008])
        self.tcp_pos = np.array([0, 0, 0.12])
        self.actions = []

    def reset(self):
        return observation()

    def step(self, command):
        self.actions.append(np.asarray(command))
        if len(self.actions) == 2:
            self.block_pos = np.array([-0.18, -0.13, 0.024])
        return observation()

    def is_success(self):
        return len(self.actions) >= 2


class SoapZeroShotTests(unittest.TestCase):
    def test_mjcf_replacement_changes_only_physical_object_shape_and_color(self):
        source = (
            '<mujoco><include file="so_arm100.xml"/><asset>'
            '<material name="block_red" rgba="0.85 0.15 0.12 1"/>'
            '<material name="tray_blue" rgba="0.20 0.35 0.65 1"/>'
            '</asset><worldbody><body name="block"><freejoint name="block_free"/>'
            '<geom name="block_geom" type="box" size="0.012 0.012 0.012"'
            ' material="block_red" mass="0.02" friction="1.2 0.005 0.0001"/>'
            '</body></worldbody></mujoco>'
        )
        original = ET.fromstring(source)
        result = ET.fromstring(soap.make_soap_xml(source))
        self.assertEqual(result.find("./include").get("file"), "so_arm100.xml")
        geom = result.find(".//geom[@name='block_geom']")
        self.assertEqual(tuple(float(x) for x in geom.get("size").split()), soap.SOAP_HALF)
        self.assertEqual(float(geom.get("mass")), 0.02)
        self.assertEqual(geom.get("friction"), original.find(".//geom").get("friction"))
        self.assertEqual(result.find(".//material[@name='tray_blue']").get("rgba"),
                         "0.20 0.35 0.65 1")
        self.assertEqual(result.find(".//material[@name='block_red']").get("rgba"),
                         soap.SOAP_RGBA)
        with self.assertRaisesRegex(ValueError, "block contract"):
            soap.make_soap_xml("<mujoco><worldbody/></mujoco>")

    def test_true_closed_loop_and_new_instruction(self):
        env = FakeEnv()
        policy = Mock()
        policy.select_action.return_value = Tensor([[0, -1.57, 1.57, 1.57, -1.57, 0.7]])
        batches = []
        def identity(batch):
            if isinstance(batch, dict):
                batches.append(batch)
            return batch
        row, frames = soap.run_episode(env, policy, identity, lambda x:x,
                                       Torch, "cpu", 0, 3, True)
        self.assertTrue(row["success"])
        self.assertEqual(row["steps"], 2)
        self.assertEqual(len(env.actions), 2)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (224, 448, 3))
        self.assertEqual(batches[0]["task"], soap.SOAP_TASK)
        self.assertNotIn("block_pos", batches[0])
        policy.reset.assert_called_once()

    def test_cli_and_markdown_report(self):
        args = soap.parse_args(["--episodes", "20", "--seed", "123"])
        self.assertEqual(args.episodes, 20)
        self.assertEqual(args.repo, soap.MODEL_ID)
        with self.assertRaises(SystemExit):
            soap.parse_args(["--episodes", "0"])
        report = {
            "checkpoint": soap.MODEL_ID, "seed": 123, "max_steps": 240,
            "successes": 1, "episodes": 2, "success_rate": 0.5,
            "results": [
                {"episode": 1, "success": True, "steps": 100,
                 "gif": "vla_runs/soap_gifs/episode_01_success.gif"},
                {"episode": 2, "success": False, "steps": 240},
            ],
        }
        doc = soap.report_markdown(report)
        self.assertIn("1/2 (50%)", doc)
        self.assertIn("Failed episodes: 2", doc)
        self.assertIn("episode_01_success.gif", doc)
        self.assertIn("36 × 24 × 16 mm", doc)


if __name__ == "__main__":
    unittest.main()
