"""The physical scene must carry the soap through actual contact only."""
import unittest
from pathlib import Path

import mujoco

from physical_grasp import (ContactMonitor, PhysicalMotion, JAW_CLOSED,
                            JAW_OPEN, run_contact_smoke, step_physical)

ROOT = Path(__file__).resolve().parents[1]


class PhysicalGraspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(
            str(ROOT / "models" / "bathroom_physical.xml")
        )

    def test_two_motorized_jaws_and_free_soap(self):
        model = self.model
        self.assertEqual(model.nu, 2)
        self.assertEqual(model.nmocap, 1)
        palm_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_free")
        self.assertNotEqual(palm_joint, -1)
        self.assertEqual(model.jnt_type[palm_joint], mujoco.mjtJoint.mjJNT_FREE)
        self.assertGreater(model.neq, 0)
        monitor = ContactMonitor(model)
        self.assertNotEqual(monitor.left, monitor.right)
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "soap_free")
        self.assertEqual(model.jnt_type[j], mujoco.mjtJoint.mjJNT_FREE)

    def test_jaws_actuate_without_moving_soap_by_assignment(self):
        model = self.model
        data = mujoco.MjData(model)
        motion = PhysicalMotion()
        mujoco.mj_resetData(model, data)
        motion.start()
        self.assertEqual(JAW_OPEN, 0)
        self.assertGreater(JAW_CLOSED, 0)
        # Full acceptance test checks physics after the close-and-lift.
        result = run_contact_smoke(model)
        self.assertTrue(result["bilateral_contacts"])
        self.assertGreater(result["highest_z"], 0.892)
        self.assertTrue(result["in_tray"])


if __name__ == "__main__":
    unittest.main()
