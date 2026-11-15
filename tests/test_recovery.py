"""Unit tests for limited, non-resetting regrasp recovery."""
import unittest

from motion import HOME
from physical_grasp import PhysicalMotion
from recovery import RecoveryManager, RECOVERY_TIMEOUT


class RecoveryTests(unittest.TestCase):
    def new_running(self):
        motion = PhysicalMotion()
        motion.start()
        manager = RecoveryManager(max_attempts=2)
        manager.start()
        motion.index = 4
        return motion, manager

    def induce_failure(self, motion, manager):
        for _ in range(90):
            manager.after_step(
                motion=motion, soap_xyz=(-0.22, 0.07, 0.812),
                linear_velocity=(0.0, 0.0, 0.0),
                both_contacts=False, in_tray=False, dt=0.002,
            )

    def test_retry_replans_from_current_pose_without_scene_reset(self):
        motion, manager = self.new_running()
        self.induce_failure(motion, manager)
        self.assertEqual(manager.status, "settling")
        self.assertFalse(motion.running)
        ready = False
        for _ in range(350):
            ready = manager.after_step(
                motion=motion, soap_xyz=(-0.22, 0.07, 0.812),
                linear_velocity=(0.0, 0.0, 0.0),
                both_contacts=False, in_tray=False, dt=0.002,
            )
            if ready:
                break
        self.assertTrue(ready)
        old_palm = (0.06, -0.10, 1.12)
        self.assertTrue(manager.restart_at(
            motion=motion, palm_xyz=old_palm, soap_xy=(-0.22, 0.07),
        ))
        self.assertEqual(manager.attempt, 2)
        self.assertEqual(manager.status, "running")
        self.assertEqual(motion.position, old_palm)
        self.assertEqual(motion.phases[0].target[:2], (-0.22, 0.07))
        self.assertEqual(motion.phases[2].target[:2], (-0.22, 0.07))
        self.assertFalse(manager.grasp_verified)
        self.assertEqual(manager.retry_positions, [(-0.22, 0.07)])
        self.assertTrue(motion.running)

    def test_second_failure_is_terminal_without_third_attempt(self):
        motion, manager = self.new_running()
        self.induce_failure(motion, manager)
        for _ in range(350):
            ready = manager.after_step(
                motion=motion, soap_xyz=(-0.20, 0.0, 0.812),
                linear_velocity=(0.0, 0.0, 0.0),
                both_contacts=False, in_tray=False, dt=0.002,
            )
            if ready:
                break
        self.assertTrue(ready)
        self.assertTrue(manager.restart_at(
            motion=motion, palm_xyz=HOME, soap_xy=(-0.20, 0),
        ))
        motion.index = 4
        self.induce_failure(motion, manager)
        self.assertEqual(manager.status, "failed")
        self.assertEqual(manager.attempt, 2)
        self.assertFalse(motion.running)
        self.assertIn("2 tentatives", manager.last_error)

    def test_outside_countertop_never_retries(self):
        motion, manager = self.new_running()
        self.induce_failure(motion, manager)
        for _ in range(int(RECOVERY_TIMEOUT / 0.002) + 3):
            ready = manager.after_step(
                motion=motion, soap_xyz=(0.70, -0.02, 0.02),
                linear_velocity=(0.0, 0.0, 0.0),
                both_contacts=False, in_tray=False, dt=0.002,
            )
            self.assertFalse(ready)
            if manager.status == "failed":
                break
        self.assertEqual(manager.status, "failed")
        self.assertEqual(manager.attempt, 1)
        self.assertEqual(manager.retry_positions, [])

    def test_missing_camera_observation_does_not_fallback_to_truth(self):
        motion, manager = self.new_running()
        self.induce_failure(motion, manager)
        for _ in range(350):
            ready = manager.after_step(
                motion=motion, soap_xyz=(-0.2, 0.0, 0.812),
                linear_velocity=(0.0, 0.0, 0.0),
                both_contacts=False, in_tray=False, dt=0.002,
            )
            if ready:
                break
        self.assertTrue(ready)
        self.assertFalse(manager.restart_at(motion=motion, palm_xyz=HOME, soap_xy=None))
        self.assertEqual(manager.attempt, 1)
        self.assertEqual(manager.status, "settling")

    def test_retarget_after_retry_has_no_double_offset(self):
        motion = PhysicalMotion()
        motion.retry_from((0.0, 0.0, 1.16), (-0.21, 0.09))
        motion.retarget_soap((-0.21, 0.09))
        self.assertAlmostEqual(motion.target_offset[0], 0.0)
        self.assertAlmostEqual(motion.target_offset[1], 0.0)


if __name__ == "__main__":
    unittest.main()
