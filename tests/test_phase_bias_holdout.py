"""Offline regression tests for non-oracle phase-gated SmolVLA holdout."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VLA = ROOT / "vla"
sys.path.insert(0, str(VLA))
from approach_gate import BiasSchedule

spec = importlib.util.spec_from_file_location("eval_phase_bias", VLA / "eval_phase_bias.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class GateTests(unittest.TestCase):
    def test_reference_is_exactly_zero_and_full_is_constant(self):
        baseline = BiasSchedule("baseline")
        full = BiasSchedule("full")
        for jaw in (0, 1.4, 1.4, 0.0, -0.05, 1.4):
            self.assertEqual(baseline.next_bias(jaw), 0)
            self.assertEqual(full.next_bias(jaw), -5)
        self.assertIsNone(baseline.transition_step)

    def test_approach_only_has_hysteresis_fades_then_never_reactivates(self):
        g = BiasSchedule("approach")
        jaws = (0.1, 1.3, 1.3, 0.5, 0.4, 0.3, 0.2, 0.1, 1.4, 0.4)
        biases = [g.next_bias(jaw) for jaw in jaws]
        self.assertEqual(biases[:3], [-5, -5, -5])
        for measured, expected in zip(biases[3:8], [-4, -3, -2, -1, 0]):
            self.assertAlmostEqual(measured, expected, places=10)
        self.assertEqual(biases[8:], [0, 0])
        self.assertEqual(g.summary()["transition_step"], 4)
        self.assertEqual(g.summary()["transition_reason"], "first_close_after_open")

    def test_horizon_deactivates_even_without_open_close_sequence(self):
        g = BiasSchedule("approach", max_approach_steps=3, fade_steps=2)
        self.assertEqual(g.next_bias(0.2), -5)
        self.assertEqual(g.next_bias(0.2), -5)
        self.assertEqual(g.next_bias(0.2), -2.5)
        self.assertEqual(g.next_bias(0.2), 0)
        self.assertEqual(g.summary()["transition_reason"], "max_approach_horizon")
        self.assertFalse(g.saw_open_jaw)

    def test_invalid_requests_and_non_finite_jaw_rejected(self):
        for mode in ("contact_oracle", "manual_teleport"):
            with self.assertRaises(ValueError):
                BiasSchedule(mode)
        with self.assertRaises(ValueError):
            BiasSchedule("approach", bias_mm=-50)
        with self.assertRaises(ValueError):
            BiasSchedule("approach").next_bias(float("nan"))


class HeldoutTests(unittest.TestCase):
    def test_seed_generates_new_distinct_reachable_and_reproducible_positions(self):
        a = study.generate_spawns(2027, 24)
        b = study.generate_spawns(2027, 24)
        old = study.generate_spawns(123, 24)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        self.assertEqual(len(set(map(tuple, a))), 24)
        self.assertTrue(all(np.linalg.norm(p) >= 0.19 for p in a))
        self.assertTrue(all(np.linalg.norm(p) <= 0.27 for p in a))
        self.assertNotEqual(a, old)
        self.assertEqual(study.MODES, ("baseline", "full", "approach"))

    def test_paired_successes_are_matched_by_exact_initial_pose(self):
        def row(index, success):
            return {"episode": index, "success": success,
                    "initial_xy_m": [0.01 * index, -0.20]}
        report = {"conditions": [
            {"mode": "baseline", "results": [row(1, False), row(2, True)]},
            {"mode": "full", "results": [row(1, True), row(2, False)]},
            {"mode": "approach", "results": [row(1, True), row(2, True)]},
        ]}
        differences = study._paired(report)
        self.assertEqual(differences[0]["baseline_failed_corrected_succeeded"], [1])
        self.assertEqual(differences[0]["baseline_succeeded_corrected_failed"], [2])
        self.assertEqual(differences[1]["net_improved_episodes"], 1)
        report["conditions"][2]["results"][0]["initial_xy_m"] = [0.2, -0.2]
        with self.assertRaisesRegex(AssertionError, "paired"):
            study._paired(report)

    def test_reporting_zero_hits_is_not_fabricated_success(self):
        def row(index, ok):
            return {"episode": index, "success": ok, "gif": None,
                    "lifted_5cm_above_initial": False,
                    "first_contact_kind": "none"}
        r = {"episodes_per_condition": 2, "seed": 2027,
             "conditions": [
                 {"mode": "baseline", "successes": 0, "results": [row(1, False), row(2, False)]},
                 {"mode": "full", "successes": 1, "results": [row(1, True), row(2, False)]},
                 {"mode": "approach", "successes": 2, "results": [row(1, True), row(2, True)]},
             ],
             "paired_comparisons": [
                 {"condition": "full", "baseline_failed_corrected_succeeded": [1],
                  "baseline_succeeded_corrected_failed": [], "net_improved_episodes": 1},
                 {"condition": "approach", "baseline_failed_corrected_succeeded": [1,2],
                  "baseline_succeeded_corrected_failed": [], "net_improved_episodes": 2},
             ]}
        text = study.report_markdown(r)
        self.assertIn("baseline | 0/2", text)
        self.assertIn("approach | 2/2", text)
        self.assertIn("seed 2027", text)
        self.assertIn("No contact or object pose", text)

    def test_default_is_pre_registered_new_24_spawn_experiment(self):
        args = study.parse_args([])
        self.assertEqual((args.seed, args.episodes, args.max_steps), (2027, 24, 240))
        self.assertEqual(args.repo, study.MODEL_ID)
        with self.assertRaises(SystemExit):
            study.parse_args(["--episodes", "0"])


if __name__ == "__main__":
    unittest.main()
