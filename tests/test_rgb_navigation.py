"""Regression: RGB navigation never consults MuJoCo cube coordinates."""
import unittest
from types import SimpleNamespace

import numpy as np

from mobile_demo.rgb_navigation import RGBObjectNavigator, PICK_CLEARANCE
from mobile_demo.so101 import ExpertSO101, PICK_CLEARANCE as SO101_PICK_CLEARANCE


class OdometryOnlyEnv:
    def __init__(self, xy=(-1.21, -0.86)):
        self.xy = np.asarray(xy, dtype=float)
        self.data = SimpleNamespace(time=0.0)

    def state(self):
        return np.r_[self.xy, [0.0], np.zeros(6)]

    def cube_position(self, index):
        raise AssertionError("Navigation must never query cube ground-truth pose")

    def in_bin(self, index):
        raise AssertionError("Navigation must never query oracle bin occupancy")


class FakeRGB:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def discover(self, env, cameras=("front",)):
        assert cameras == ("front",)
        self.calls += 1
        return [np.asarray(xy, dtype=float) for xy in self.values]


class RGBNavigationTests(unittest.TestCase):
    def test_visual_object_sets_goal_without_object_pose(self):
        env = OdometryOnlyEnv()
        vision = FakeRGB([(-0.73, -0.86)])
        nav = RGBObjectNavigator()
        goal, arrived = nav.step(env, vision, 0.0)
        np.testing.assert_allclose(goal, [-0.73 - PICK_CLEARANCE, -0.86])
        self.assertIsNone(arrived)
        self.assertEqual(nav.discoveries, 1)
        env.xy = goal.copy()
        goal2, arrived2 = nav.step(env, vision, 0.1)
        np.testing.assert_allclose(goal2, goal)
        np.testing.assert_allclose(arrived2, [-0.73, -0.86])

    def test_tidybot_rgb_standoff_matches_so101_controller(self):
        # The mobile navigation and arm must agree on the same relative reach.
        # The old hard-coded 0.39 m with the TidyBot 0.54 m mount was wrong.
        nav = RGBObjectNavigator(pick_clearance=SO101_PICK_CLEARANCE)
        env = OdometryOnlyEnv()
        visual_target = np.array([-0.73, -0.86])
        goal, arrived = nav.step(env, FakeRGB([visual_target]), 0.0)
        np.testing.assert_allclose(goal + [SO101_PICK_CLEARANCE, 0.0],
                                   visual_target)
        self.assertIsNone(arrived)
        self.assertAlmostEqual(nav.report()['pick_clearance_m'],
                               SO101_PICK_CLEARANCE)
    def test_bin_objects_and_unreachable_coordinates_are_excluded(self):
        self.assertFalse(RGBObjectNavigator.selectable((0.95, 0.90)))
        self.assertFalse(RGBObjectNavigator.selectable((-1.45, -0.5)))
        self.assertTrue(RGBObjectNavigator.selectable((-0.73, -0.86)))
        env = OdometryOnlyEnv()
        vision = FakeRGB([(0.95, 0.90), (-1.45, -0.5)])
        goal, arrived = RGBObjectNavigator().step(env, vision, 0.0)
        self.assertIsNone(arrived)
        np.testing.assert_allclose(goal, [-0.75, -0.88], atol=1e-8)

    def test_patrol_works_with_no_cube_position_access(self):
        env = OdometryOnlyEnv()
        nav = RGBObjectNavigator()
        vision = FakeRGB([])
        visited = []
        for n in range(8):
            goal, target = nav.step(env, vision, n * 0.5)
            visited.append(tuple(goal))
            self.assertIsNone(target)
            env.xy = np.asarray(goal)
        self.assertGreater(nav.patrol_visits, 4)
        self.assertGreater(len(set(visited)), 4)
        self.assertGreater(vision.calls, 4)

    def test_rgb_expert_navigation_does_not_touch_oracle_before_handoff(self):
        env = OdometryOnlyEnv()
        expert = ExpertSO101()
        expert.vision = FakeRGB([(-0.73, -0.86)])
        expert.navigator = RGBObjectNavigator()
        cmd = expert.action(env)
        np.testing.assert_allclose(cmd[:2], [-0.73 - PICK_CLEARANCE, -0.86])
        self.assertEqual(expert.phase, "navigate")
        self.assertEqual(expert.navigator.discoveries, 1)

    def test_ignored_target_is_not_selected_again(self):
        env = OdometryOnlyEnv()
        nav = RGBObjectNavigator()
        vision = FakeRGB([(-0.73, -0.86)])
        nav.step(env, vision, 0.0)
        nav.ignore_target()
        _, arrived = nav.step(env, vision, 0.5)
        self.assertIsNone(arrived)
        self.assertIsNone(nav.target_xy)
        self.assertEqual(len(nav.ignored_xy), 1)


if __name__ == "__main__":
    unittest.main()
