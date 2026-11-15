"""Outcome supervision must distinguish true contact pickup from scripted motion."""
import unittest
from pathlib import Path

import mujoco

import physical_grasp
from demo import soap_joint_addresses
from grasp_supervisor import GraspSupervisor, MIN_LIFT_Z
from motion import HOME
from physical_grasp import ContactMonitor, PhysicalMotion, step_physical

ROOT = Path(__file__).resolve().parents[1]


class GraspSupervisorUnitTests(unittest.TestCase):
    def test_completed_script_without_lift_is_a_failure(self):
        supervisor = GraspSupervisor()
        supervisor.start()
        result = supervisor.observe(
            phase_index=8, finished=True, soap_z=0.812,
            both_contacts=False, in_tray=False, dt=0.002,
        )
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.grasp_verified)
        self.assertEqual(result.failure_reason, "Prise non vérifiée")

    def test_verified_lift_and_real_placement_are_required(self):
        supervisor = GraspSupervisor()
        supervisor.start()
        supervisor.observe(
            phase_index=3, finished=False, soap_z=1.0,
            both_contacts=True, in_tray=False, dt=0.002,
        )
        self.assertTrue(supervisor.outcome.grasp_verified)
        result = supervisor.observe(
            phase_index=8, finished=True, soap_z=0.84,
            both_contacts=False, in_tray=True, dt=0.002,
        )
        self.assertEqual(result.status, "success")

    def test_loss_during_transport_stops_after_bounded_grace(self):
        supervisor = GraspSupervisor()
        supervisor.start()
        supervisor.observe(
            phase_index=3, finished=False, soap_z=MIN_LIFT_Z + 0.03,
            both_contacts=True, in_tray=False, dt=0.002,
        )
        for _ in range(80):
            result = supervisor.observe(
                phase_index=4, finished=False, soap_z=0.812,
                both_contacts=False, in_tray=False, dt=0.002,
            )
        self.assertEqual(result.status, "failed")
        self.assertIn("perdu", result.failure_reason)

    def test_planned_descent_is_not_mistaken_for_drop(self):
        supervisor = GraspSupervisor()
        supervisor.start()
        supervisor.observe(
            phase_index=3, finished=False, soap_z=1.01,
            both_contacts=True, in_tray=False, dt=0.002,
        )
        for _ in range(100):
            result = supervisor.observe(
                phase_index=5, finished=False, soap_z=0.845,
                both_contacts=True, in_tray=False, dt=0.002,
            )
        self.assertEqual(result.status, "running")

    def test_reset_and_external_failure(self):
        supervisor = GraspSupervisor()
        supervisor.start()
        supervisor.fail("Savon non détecté")
        self.assertEqual(supervisor.outcome.status, "failed")
        supervisor.reset()
        self.assertEqual(supervisor.outcome.status, "idle")
        self.assertIsNone(supervisor.outcome.failure_reason)
        self.assertFalse(supervisor.outcome.grasp_verified)


class GraspSupervisorPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(
            str(ROOT / "models" / "bathroom_physical.xml")
        )
        cls.qadr, _ = soap_joint_addresses(cls.model)

    def simulate(self, open_jaws=False):
        model = self.model
        data = mujoco.MjData(model)
        motion = PhysicalMotion()
        monitor = ContactMonitor(model)
        supervisor = GraspSupervisor()
        mujoco.mj_resetData(model, data)
        data.mocap_pos[0] = HOME
        mujoco.mj_forward(model, data)
        motion.start()
        supervisor.start()
        saved = physical_grasp.JAW_CLOSED
        if open_jaws:
            physical_grasp.JAW_CLOSED = physical_grasp.JAW_OPEN
        try:
            for _ in range(6500):
                step_physical(model, data, motion)
                left, right = monitor.contacts(data)
                x, y, z = map(float, data.qpos[self.qadr:self.qadr + 3])
                in_tray = 0.075 < x < 0.445 and -0.12 < y < 0.22 and 0.82 < z < 0.92
                result = supervisor.observe(
                    phase_index=motion.index, finished=motion.finished,
                    soap_z=z, both_contacts=left and right,
                    in_tray=in_tray, dt=float(model.opt.timestep),
                )
                if result.status == "failed":
                    motion.running = False
                    motion.finished = False
                    return result, in_tray
                if result.status == "success":
                    return result, in_tray
            self.fail("Physical cycle did not reach a terminal outcome")
        finally:
            physical_grasp.JAW_CLOSED = saved

    def test_real_physics_cycle_reports_success(self):
        outcome, in_tray = self.simulate()
        self.assertEqual(outcome.status, "success")
        self.assertTrue(outcome.grasp_verified)
        self.assertTrue(in_tray)

    def test_open_jaws_fail_without_fake_success(self):
        outcome, in_tray = self.simulate(open_jaws=True)
        self.assertEqual(outcome.status, "failed")
        self.assertFalse(outcome.grasp_verified)
        self.assertFalse(in_tray)


if __name__ == "__main__":
    unittest.main()
