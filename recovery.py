"""Two-attempt non-resetting recovery for a dropped soap in MuJoCo.

All recovery decisions are based on simulation evidence (free-soap pose,
velocity and bilateral contacts). A camera observation can supply retry XY
when visual servoing is enabled. No soap qpos or qvel is ever changed.
"""
from grasp_supervisor import GraspSupervisor
from physical_grasp import SOAP_REST_Z

MAX_ATTEMPTS = 2
RECOVERY_TIMEOUT = 3.0
MIN_SETTLE_TIME = 0.4
STABLE_HOLD_TIME = 0.2
OBSERVATION_INTERVAL = 0.2

# Conservative envelope for the fixed parallel gripper and countertop.
MIN_X, MAX_X = -0.56, -0.09
MIN_Y, MAX_Y = -0.30, 0.30


class RecoveryManager:
    """Coordinate pickup verification and one optional regrasp from live state."""

    def __init__(self, max_attempts=MAX_ATTEMPTS):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts
        self.supervisor = GraspSupervisor()
        self.reset()

    def reset(self):
        self.supervisor.reset()
        self.attempt = 0
        self.status = "idle"
        self.last_error = None
        self._waiting = 0.0
        self._stable = 0.0
        self._last_observation = -OBSERVATION_INTERVAL
        self.ready_for_observation = False
        self.retry_positions = []

    def start(self):
        self.reset()
        self.attempt = 1
        self.status = "running"
        self.supervisor.start()

    def fail_terminal(self, reason, motion):
        self.status = "failed"
        self.last_error = reason
        self.supervisor.fail(reason)
        self.ready_for_observation = False
        motion.running = False
        motion.finished = False
        motion.label = "Échec : " + reason

    def _recoverable(self, xyz, linear_velocity):
        x, y, z = xyz
        return (MIN_X <= x <= MAX_X and MIN_Y <= y <= MAX_Y
                and abs(z - SOAP_REST_Z) <= 0.03
                and max(abs(v) for v in linear_velocity) < 0.04)

    def after_step(self, *, motion, soap_xyz, linear_velocity,
                   both_contacts, in_tray, dt):
        """Update from one physics step. Returns True if fresh XY is needed."""
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        if self.status == "running":
            outcome = self.supervisor.observe(
                phase_index=motion.index, finished=motion.finished,
                soap_z=soap_xyz[2], both_contacts=both_contacts,
                in_tray=in_tray, dt=dt,
            )
            if outcome.status == "success":
                self.status = "success"
            elif outcome.status == "failed":
                self.last_error = outcome.failure_reason
                motion.running = False
                motion.finished = False
                if self.attempt >= self.max_attempts:
                    self.fail_terminal(
                        f"{outcome.failure_reason} (après {self.attempt} tentatives)",
                        motion,
                    )
                else:
                    self.status = "settling"
                    motion.label = "Prise ratée : récupération du savon"
                    self._waiting = 0.0
                    self._stable = 0.0
                    self._last_observation = -OBSERVATION_INTERVAL
            return False

        if self.status != "settling":
            return False
        self._waiting += dt
        if self._recoverable(soap_xyz, linear_velocity):
            self._stable += dt
        else:
            self._stable = 0.0
        if self._waiting >= RECOVERY_TIMEOUT:
            self.fail_terminal(
                "Savon non récupérable : hors de portée, instable ou non détecté",
                motion,
            )
            return False
        self.ready_for_observation = (
            self._waiting >= MIN_SETTLE_TIME
            and self._stable >= STABLE_HOLD_TIME
            and self._waiting - self._last_observation >= OBSERVATION_INTERVAL
        )
        if self.ready_for_observation:
            self._last_observation = self._waiting
        return self.ready_for_observation

    def restart_at(self, *, motion, palm_xyz, soap_xy):
        """Start attempt #2 without resetting MuJoCo or teleporting soap."""
        if self.status != "settling" or not self.ready_for_observation:
            return False
        self.ready_for_observation = False
        if soap_xy is None or len(soap_xy) != 2:
            return False
        x, y = map(float, soap_xy)
        if not (MIN_X <= x <= MAX_X and MIN_Y <= y <= MAX_Y):
            return False
        motion.retry_from(palm_xyz, (x, y))
        self.retry_positions.append((x, y))
        self.attempt += 1
        self.supervisor.start()
        self.status = "running"
        self.last_error = None
        motion.label = f"Tentative {self.attempt}/{self.max_attempts} : approche"
        return True

    @property
    def grasp_verified(self):
        return self.supervisor.outcome.grasp_verified
