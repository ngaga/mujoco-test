"""Outcome supervision for the EarlyBots physical soap-handling experiment.

This is a deliberately conservative *simulation* oracle, using MuJoCo's
measured contact pairs and object position. It is NOT a camera perception
policy and makes no real-world safety claims.
"""
from dataclasses import dataclass

SOAP_REST_Z = 0.812
MIN_LIFT_Z = SOAP_REST_Z + 0.08
DROP_GRACE_SECONDS = 0.15


@dataclass
class GraspOutcome:
    status: str = "idle"          # idle, running, success, failed
    failure_reason: str | None = None
    grasp_verified: bool = False


class GraspSupervisor:
    """Track real task success rather than trusting scripted phase completion."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.outcome = GraspOutcome()
        self._drop_duration = 0.0

    def start(self):
        self.reset()
        self.outcome.status = "running"

    def fail(self, reason):
        if self.outcome.status == "running":
            self.outcome.status = "failed"
            self.outcome.failure_reason = reason
        return self.outcome

    def observe(self, *, phase_index, finished, soap_z, both_contacts,
                in_tray, dt):
        """Consume one MuJoCo physics step and return the current outcome.

        Pickup is confirmed only with simultaneously observed bilateral
        contacts AND >8 cm of soap lift. Once transport begins, a sustained
        loss of height fails the operation. Completing the script without
        a correctly placed soap is *not* success.
        """
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        result = self.outcome
        if result.status != "running":
            return result
        if both_contacts and soap_z >= MIN_LIFT_Z and phase_index in (3, 4, 5):
            result.grasp_verified = True

        # During transport and the still-closed part of the descent, verify
        # that the soap is actually off the counter. Allow brief slips/noise.
        if phase_index in (4, 5) and not finished:
            lost = (not result.grasp_verified or soap_z < MIN_LIFT_Z)
            self._drop_duration = self._drop_duration + dt if lost else 0.0
            if self._drop_duration >= DROP_GRACE_SECONDS:
                result.status = "failed"
                result.failure_reason = (
                    "Savon non soulevé ou perdu pendant le transport"
                )
        if finished and result.status == "running":
            if result.grasp_verified and in_tray:
                result.status = "success"
            else:
                result.status = "failed"
                result.failure_reason = (
                    "Prise non vérifiée" if not result.grasp_verified
                    else "Savon absent du bac après dépose"
                )
        return result
