"""Offline regression tests for SmolVLA checkpoint downloader metadata.

Network and 0.9GB safetensors downloads are deliberately not triggered by CI.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fetch_smolvla", ROOT / "vla" / "fetch_smolvla.py",
)
vla = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vla)


class SmolVLACheckpointTests(unittest.TestCase):
    def test_default_is_real_public_finetuned_pick_place_model(self):
        self.assertEqual(
            vla.MODEL_ID,
            "ahmedsohail2003/smolvla-so101-pickplace-v2",
        )
        self.assertIn("model.safetensors", vla.MODEL_FILES)
        self.assertTrue(any("normalizer" in x for x in vla.MODEL_FILES))
        self.assertIn("policy_preprocessor.json", vla.MODEL_FILES)
        self.assertIn("policy_postprocessor.json", vla.MODEL_FILES)

    def test_config_probe_is_explicit_about_incompatible_action_space(self):
        sample = {
            "type": "smolvla", "max_action_dim": 32,
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.camera1": {"type": "VISUAL", "shape": [3, 224, 224]},
                "observation.images.camera2": {"type": "VISUAL", "shape": [3, 224, 224]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        }
        report = vla.describe_config(sample)
        self.assertEqual(report["model_type"], "smolvla")
        self.assertEqual(report["output_features"]["action"]["shape"], [6])
        self.assertFalse(report["earlybots_compatible_without_adapter"])
        self.assertIn("floating palm", report["reason"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FileNotFoundError):
                vla.inspect_local(path)
            (path / "config.json").write_text(json.dumps(sample))
            self.assertEqual(vla.inspect_local(path), report)

    def test_no_network_on_local_inspect_and_no_fake_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}")
            self.assertFalse(vla.inspect_local(path)["earlybots_compatible_without_adapter"])
            self.assertFalse((path / "model.safetensors").exists())

    def test_download_pattern_does_not_omit_pre_post_processing(self):
        # Unit-test only the arguments. Do not connect to HF during CI.
        import sys
        with mock.patch.dict(sys.modules, {
            "huggingface_hub": mock.Mock(snapshot_download=mock.Mock(
                return_value="/tmp/stub-model-path"
            ))
        }):
            path = vla.download_checkpoint(Path("/tmp/stub-model-path"))
        self.assertEqual(str(path), "/tmp/stub-model-path")


if __name__ == "__main__":
    unittest.main()
