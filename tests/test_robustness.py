"""Regression tests for reproducible paired physical robustness reports."""
import unittest

import mujoco

from robustness import (MODEL_PATH, Scenario, benchmark, make_scenarios,
                        run_trial, summarize)


class RobustnessTests(unittest.TestCase):
    def test_random_positions_are_reproducible_and_bounded(self):
        first = make_scenarios(100, seed=2026)
        self.assertEqual(first, make_scenarios(100, seed=2026))
        self.assertNotEqual(first, make_scenarios(100, seed=2027))
        self.assertEqual(len(first), 100)
        self.assertEqual(sum(s.injected_drop for s in first), 50)
        self.assertEqual(len({(s.x, s.y) for s in first}), 100)
        for case in first:
            self.assertGreaterEqual(case.x, -0.32)
            self.assertLessEqual(case.x, -0.22)
            self.assertGreaterEqual(case.y, -0.045)
            self.assertLessEqual(case.y, 0.055)

    def test_invalid_trial_count(self):
        with self.assertRaises(ValueError):
            make_scenarios(0)

    def test_summary_does_not_count_false_success(self):
        rows = [
            {"status": "success", "injected_drop": False,
             "attempts": 1, "reason": None},
            {"status": "failed", "injected_drop": True,
             "attempts": 1, "reason": "Savon perdu"},
        ]
        result = summarize(rows)
        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["fault_successes"], 0)
        self.assertEqual(result["failure_reasons"]["Savon perdu"], 1)

    def test_real_physics_nominal_and_forced_drop(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        clean = Scenario(0, -0.27, 0.005, False)
        good = run_trial(model, clean, recovery_enabled=True)
        self.assertEqual(good["status"], "success")
        self.assertTrue(good["grasp_verified"])
        dropped = Scenario(1, -0.27, 0.005, True)
        baseline = run_trial(model, dropped, recovery_enabled=False)
        recovered = run_trial(model, dropped, recovery_enabled=True)
        self.assertTrue(baseline["fault_triggered"])
        self.assertNotEqual(baseline["status"], "success")
        self.assertEqual(baseline["attempts"], 1)
        self.assertEqual(recovered["status"], "success")
        self.assertEqual(recovered["attempts"], 2)
        self.assertTrue(recovered["replanned"])

    def test_short_paired_experiment_produces_complete_trace(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        report = benchmark(4, seed=12, model=model)
        self.assertEqual(report["positions"], 4)
        self.assertEqual(report["baseline"]["runs"], 4)
        self.assertEqual(report["with_recovery"]["runs"], 4)
        self.assertEqual(report["baseline"]["fault_runs"], 2)
        self.assertEqual(report["with_recovery"]["fault_runs"], 2)
        for before, after in zip(report["baseline_results"],
                                 report["recovery_results"]):
            self.assertEqual(before["index"], after["index"])
            self.assertEqual(before["x"], after["x"])
            self.assertEqual(before["y"], after["y"])
            self.assertEqual(before["injected_drop"], after["injected_drop"])
            self.assertLessEqual(before["attempts"], 1)
            self.assertLessEqual(after["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
