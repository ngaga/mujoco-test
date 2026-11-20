"""Deterministic recovery order tests: no model weights or MuJoCo meshes."""
import unittest

from mobile_demo.hotel_recovery import RetryScheduler


class RetrySchedulerTests(unittest.TestCase):
    def test_retry_dropped_cube_at_new_floor_pose_without_reset(self):
        s = RetryScheduler(3, max_retries=-1)
        placed = lambda i: False
        event = s.failed(1, "slipped on lift", [0.5, -0.2, 0.012], 7.2, "lift")
        self.assertTrue(event["fell_to_floor"])
        self.assertEqual(event["cube_xyz"], [0.5, -0.2, 0.012])
        self.assertEqual(s.choose(placed, current=1, rotate=False), 1)
        self.assertEqual(len(s.retry_events), 1)

    def test_rotate_after_two_misses_and_revisit_earlier_cube(self):
        s = RetryScheduler(3, max_retries=-1)
        placed = lambda i: i == 2
        s.failed(0, "miss", [0.2, 0.2, 0.8], 1, "lower")
        self.assertEqual(s.choose(placed, current=0, rotate=False), 0)
        s.failed(0, "miss", [0.2, 0.2, 0.8], 2, "lower")
        self.assertEqual(s.choose(placed, current=0, rotate=True), 1)
        self.assertEqual(s.choose(placed, current=1, rotate=True), 0)

    def test_exhausted_target_never_blocks_remaining_cubes(self):
        s = RetryScheduler(3, max_retries=1)
        s.failed(0, "miss", [0.3, 0.1, .8], 1, "hover")
        s.failed(0, "miss", [0.3, 0.1, .8], 2, "hover")
        placed = lambda i: i == 2
        self.assertEqual(s.choose(placed, current=0, rotate=False), 1)
        self.assertEqual(s.report(placed)["exhausted_cubes"], [0])
        self.assertEqual(s.report(placed)["undelivered_cubes"], [0, 1])

    def test_unbounded_mode_remains_bounded_by_external_episode(self):
        s = RetryScheduler(3, max_retries=-1)
        placed = lambda i: i == 1
        for i in range(30):
            s.failed(0, "retry", [0, 0, 0.01], i, "recover")
        self.assertIn(0, s.eligible(placed))
        self.assertEqual(s.report(placed)["retries"], 30)
        self.assertTrue(s.report(placed)["unbounded_until_max_steps"])

    def test_everything_in_bin_has_no_remaining_target(self):
        s = RetryScheduler(3)
        in_bin = lambda i: True
        self.assertEqual(s.pending(in_bin), [])
        self.assertIsNone(s.choose(in_bin, current=2, rotate=True))


if __name__ == "__main__":
    unittest.main()
