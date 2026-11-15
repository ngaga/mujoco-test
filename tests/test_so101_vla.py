"""SmolVLA/SO-101 observation-action integration contract (no weight download)."""
import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VLA = ROOT / "vla"
sys.path.insert(0, str(VLA))
spec = importlib.util.spec_from_file_location("run_so101", VLA / "run_so101.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def observation():
    front = np.full((224, 224, 3), 117, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 85, dtype=np.uint8)
    return {"state": np.arange(6, dtype=np.float32),
            "images": {"front": front, "wrist": wrist}}


class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, dim))

    def permute(self, *axes):
        return FakeTensor(self.data.transpose(axes))

    def float(self):
        return FakeTensor(self.data.astype(np.float32))

    def __truediv__(self, other):
        return FakeTensor(self.data / other)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeTorch:
    from_numpy = staticmethod(FakeTensor)
    no_grad = staticmethod(contextlib.nullcontext)


class SimulatedEnv:
    def __init__(self):
        self.model = Mock(nu=6)
        self.block_pos = np.array([0.0, -0.22, 0.012])
        self.tcp_pos = np.array([0.0, 0.0, 0.12])
        self.actions = []

    def reset(self):
        return observation()

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        if len(self.actions) == 2:
            self.block_pos = np.array([-0.18, -0.13, 0.024])
        return observation()

    def is_success(self):
        return len(self.actions) >= 2


class SO101IntegrationTests(unittest.TestCase):
    def test_reference_task_and_camera_mapping_are_exact(self):
        self.assertEqual(run.TASK, "Pick up the red block and place it in the blue tray.")
        self.assertEqual(run.CAMERA_MAP, {"front": "camera1", "wrist": "camera2"})
        self.assertEqual(run.ACTION_DIM, 6)
        self.assertEqual(run.CONTROL_HZ, 15.0)
        self.assertEqual(run.MAX_STEPS, 240)

    def test_camera_rgb_and_joint_state_are_required(self):
        state, images = run.validate_observation(observation())
        self.assertEqual(state.shape, (6,))
        self.assertEqual(images["front"].shape, (224, 224, 3))
        bad = observation()
        del bad["images"]["wrist"]
        with self.assertRaisesRegex(ValueError, "wrist"):
            run.validate_observation(bad)
        black = observation()
        black["images"]["front"][:] = 0
        with self.assertRaisesRegex(ValueError, "black"):
            run.validate_observation(black)
        wrong = observation()
        wrong["state"] = np.zeros(3)
        with self.assertRaisesRegex(ValueError, "six"):
            run.validate_observation(wrong)

    def test_tensors_match_published_preprocessor_example(self):
        batch = run.observation_to_batch(observation(), "cpu", FakeTorch)
        self.assertEqual(set(batch), {
            "observation.state",
            "observation.images.camera1",
            "observation.images.camera2",
            "task",
        })
        self.assertEqual(batch["observation.state"].data.shape, (1, 6))
        self.assertEqual(batch["observation.images.camera1"].data.shape,
                         (1, 3, 224, 224))
        self.assertAlmostEqual(
            float(batch["observation.images.camera1"].data[0, 1, 2, 3]),
            117 / 255, places=6,
        )
        self.assertEqual(batch["task"], run.TASK)

    def test_joint_commands_validated_before_driving_actuators(self):
        model = Mock(nu=6)
        self.assertEqual(run.validate_action(np.arange(6).reshape(1, 6),
                                              model).shape, (6,))
        for invalid in ([1, 2, 3], [1, 2, 3, 4, 5, float("nan")],
                        [20, 0, 0, 0, 0, 0]):
            with self.assertRaises(ValueError):
                run.validate_action(invalid, model)
        with self.assertRaises(ValueError):
            run.validate_action(np.zeros(6), Mock(nu=2))

    def test_closed_loop_uses_model_actions_without_scripted_teacher(self):
        env = SimulatedEnv()
        policy = Mock()
        policy.select_action.return_value = FakeTensor(
            np.array([[0.0, -1.57, 1.57, 1.57, -1.57, 0.7]])
        )
        identity = lambda x: x
        row = run.run_episode(env, policy, identity, identity,
                              FakeTorch, "cpu", max_steps=3)
        self.assertTrue(row["success"])
        self.assertEqual(row["steps"], 2)
        self.assertEqual(len(env.actions), 2)
        self.assertTrue(np.allclose(env.actions[0],
                                    [0.0, -1.57, 1.57, 1.57, -1.57, 0.7]))
        self.assertEqual(policy.select_action.call_count, 2)
        policy.reset.assert_called_once()

    def test_cli_has_check_without_loading_weights(self):
        args = run.parse_args(["--check-env", "--seed", "123"])
        self.assertTrue(args.check_env)
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.repo, run.MODEL_ID)
        with self.assertRaises(SystemExit):
            run.parse_args(["--episodes", "0"])


if __name__ == "__main__":
    unittest.main()
