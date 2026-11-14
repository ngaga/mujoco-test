"""Contact-only parallel-jaw grasp experiment.

The palm pose is still commanded by the existing scripted MuJoCo mocap path.
The two jaw slides are genuine mass-bearing MuJoCo bodies driven by position
actuators, and the soap is never assigned an artificial transport pose.
"""
from dataclasses import replace

import mujoco

from motion import Motion, PHASES

# In the physical scene the soap rests at z=0.812 on the 0.79-high counter.
SOAP_REST_Z = 0.812
SOAP_TOP_Z = SOAP_REST_Z + 0.022
GRIPPER_SOAP_DELTA_Z = 0.119
# Actuator target exceeds the nominal first-contact position: force-limited
# position servos compress the contact slightly to generate normal force.
JAW_CLOSED = 0.042
JAW_OPEN = 0.0

PHYSICAL_PHASES = tuple(
    replace(p, target=(p.target[0], p.target[1], 0.963))
    if i in (5, 6) else p
    for i, p in enumerate(PHASES)
)


class PhysicalMotion(Motion):
    """Same guided trajectory, with the drop height corrected for the tray."""

    def reset(self):
        super().reset()
        self.phases = PHYSICAL_PHASES


class ContactMonitor:
    """Observe genuine soap-pad contacts, not state-machine hold flags."""

    def __init__(self, model):
        self.soap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "soap_geom")
        self.left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad")
        self.right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad")
        if min(self.soap, self.left, self.right) < 0:
            raise ValueError("Physical grasp scene must contain soap and both jaw pads")

    def contacts(self, data):
        left = right = False
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.soap in pair:
                left = left or self.left in pair
                right = right or self.right in pair
        return left, right


def step_physical(model, data, motion):
    """Never touch soap.qpos/qvel. Let MuJoCo resolve all contact forces."""
    motion.advance(model.opt.timestep)
    data.mocap_pos[0] = motion.position
    closed = 2 <= motion.index <= 5 and motion.running
    data.ctrl[:] = JAW_CLOSED if closed else JAW_OPEN
    mujoco.mj_step(model, data)


def run_contact_smoke(model, iterations=1500):
    """Verify pickup and free release: real lift, bilateral contact, final drop.

    This can fail on a physics regression, even when the old scripted smoke
    test succeeds. No visual rendering is required in CI.
    """
    from demo import reset, soap_joint_addresses, soap_in_tray

    data = mujoco.MjData(model)
    motion = PhysicalMotion()
    monitor = ContactMonitor(model)
    qadr, _ = soap_joint_addresses(model)
    reset(model, data, motion)
    motion.start()
    highest = SOAP_REST_Z
    bilateral_during_lift = False
    reached_transport = False
    released = False
    for _ in range(iterations):
        step_physical(model, data, motion)
        z = float(data.qpos[qadr + 2])
        if not 0.0 <= z <= 2.0:
            raise RuntimeError(f"Soap diverged in physics: z={z}")
        highest = max(highest, z)
        if motion.index in (3, 4):
            left, right = monitor.contacts(data)
            bilateral_during_lift |= left and right
        if motion.index == 4:
            reached_transport = True
        if motion.index >= 7:
            released = True
    soap_xyz = tuple(float(v) for v in data.qpos[qadr:qadr + 3])
    info = {
        "highest_z": round(highest, 4),
        "bilateral_contacts": bilateral_during_lift,
        "transport_phase": reached_transport,
        "released": released,
        "soap_xyz": soap_xyz,
        "in_tray": soap_in_tray(data, qadr, False),
    }
    if not bilateral_during_lift:
        raise RuntimeError(f"Physical grasp did not establish bilateral contact: {info}")
    if highest < SOAP_REST_Z + 0.08:
        raise RuntimeError(f"Physical grasp did not lift soap by at least 8 cm: {info}")
    if not (reached_transport and released and motion.finished):
        raise RuntimeError(f"Physical grasp cycle did not finish: {info}")
    if not info["in_tray"]:
        raise RuntimeError(f"Soap was not released into tray: {info}")
    print(f"OK: contact-only physical pick, carry and release: {info}")
    return info
