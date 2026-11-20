"""Fast planner contract tests, no native TidyBot asset download required."""
import unittest
import numpy as np

from mobile_demo.collision_planner import JointMotionPlanner, PlanningError


class FakeChecker:
    """Two-joint obstacle; remaining five dimensions are free."""
    def __init__(self):
        self.checks = 0
        self.last_blocker = None
        self.margin = .012
        self.contacts = 0
        self.smallest_contact_distance = 0
        self.contact_examples = []

    def valid(self, q, base, *, carrying=False, ignore_cube=None):
        self.checks += 1
        q = np.asarray(q)
        if abs(q[0]) < .27 and abs(q[1]) < .45:
            self.last_blocker = "synthetic wall"
            return False
        return True

    def segment_valid(self, start, goal, base, *, carrying=False,
                      ignore_cube=None, joint_increment=.045):
        start = np.asarray(start)
        goal = np.asarray(goal)
        steps = max(1, int(np.ceil(np.max(np.abs(start-goal)) / joint_increment)))
        return all(self.valid(start + (goal-start)*k/steps, base,
                              carrying=carrying, ignore_cube=ignore_cube)
                   for k in range(steps+1))


START = np.array([-.85, 0, 0, 0, 0, 0, 0])
GOAL = np.array([.85, 0, 0, 0, 0, 0, 0])
BASE = np.array([.15, .59, 0])


class RRTConnectContractTests(unittest.TestCase):
    def test_direct_mode_must_reject_hidden_wall(self):
        planner = JointMotionPlanner(FakeChecker(), mode="vertical")
        with self.assertRaisesRegex(PlanningError, "intersects"):
            planner.plan(START, GOAL, BASE)
        self.assertEqual(planner.report()["rejections"], 1)

    def test_start_in_collision_rejected_even_if_goal_safe(self):
        planner = JointMotionPlanner(FakeChecker(), mode="vertical")
        with self.assertRaisesRegex(PlanningError, "Start configuration"):
            planner.plan(np.zeros(7), GOAL, BASE)

    def test_direct_swept_segment_is_returned_for_safe_move(self):
        checker = FakeChecker()
        planner = JointMotionPlanner(checker, mode="vertical")
        end = START + np.array([0, .5, 0, 0, 0, 0, 0])
        plan = planner.plan(START, end, BASE)
        self.assertEqual(plan.method, "swept-direct")
        self.assertGreater(len(plan.waypoints), 2)
        np.testing.assert_allclose(plan.waypoints[-1], end)
        self.assertEqual(planner.report()["rrt_calls"], 0)

    def test_rrt_connect_finds_and_validates_detour(self):
        checker = FakeChecker()
        planner = JointMotionPlanner(checker, mode="ompl", timeout=4.0)
        plan = planner.plan(START, GOAL, BASE)
        self.assertEqual(plan.method, "RRTConnect")
        self.assertEqual(planner.report()["rrt_calls"], 1)
        self.assertEqual(planner.report()["rrt_solutions"], 1)
        self.assertGreater(len(plan.waypoints), 2)
        np.testing.assert_allclose(plan.waypoints[0], START)
        np.testing.assert_allclose(plan.waypoints[-1], GOAL)
        for a, b in zip(plan.waypoints, plan.waypoints[1:]):
            self.assertTrue(checker.segment_valid(a, b, BASE), "unsafe returned segment")


if __name__ == "__main__":
    unittest.main()
