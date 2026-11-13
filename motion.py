"""Deterministic scripted pick-and-place with smooth camera-guided XY approach.

This is still a kinematic demonstration, not physical grasping.
"""
from dataclasses import dataclass
from math import sqrt

HOME = (-0.42, -0.06, 1.16)
SOAP_START = (-0.27, 0.005, 0.844)
SOAP_TARGET = (0.255, -0.025, 0.847)
SOAP_OFFSET = (0.0, 0.0, -0.087)


@dataclass(frozen=True)
class Phase:
    name: str
    target: tuple[float, float, float]
    duration: float
    holding: bool


PHASES = (
    Phase("Approche du savon", (-0.27, 0.005, 1.11), 1.15, False),
    Phase("Descente", (-0.27, 0.005, 0.931), 0.85, False),
    Phase("Saisie simulée", (-0.27, 0.005, 0.931), 0.35, True),
    Phase("Levée", (-0.27, 0.005, 1.16), 1.05, True),
    Phase("Transport vers le bac", (0.255, -0.025, 1.16), 1.75, True),
    Phase("Dépose", (0.255, -0.025, 0.934), 0.95, True),
    Phase("Libération", (0.255, -0.025, 0.934), 0.45, False),
    Phase("Retour au repos", HOME, 1.30, False),
)


def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    t = t * t * (3 - 2 * t)
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class Motion:
    """Advance in simulated seconds, independently of wall-clock time."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.position = HOME
        self.phases = PHASES
        self.index = 0
        self.elapsed = 0.0
        # The first four script phases remain in their uncorrected coordinates;
        # a continuous visual correction is applied to their XY positions.
        self.from_position = HOME
        self.target_offset = [0.0, 0.0]
        self.offset = [0.0, 0.0]
        self.offset_velocity = [0.0, 0.0]
        self.running = False
        self.finished = False
        self.holding = False
        self.label = "Prêt"

    def start(self) -> None:
        self.reset()
        self.running = True
        self.label = PHASES[0].name

    def retarget_soap(self, xy: tuple[float, float]) -> None:
        """Accept RGB-based target updates during approach/descent only.

        Unlike replacing phase endpoints mid-flight, a spring-damper correction
        preserves position continuity between camera observations.
        """
        if not self.running or self.index > 1:
            return
        self.target_offset = [xy[0] - SOAP_START[0], xy[1] - SOAP_START[1]]

    def _advance_offset(self, dt: float) -> None:
        # Critical damping, with explicit acceleration and speed limits.
        # This keeps camera pixel noise from creating visible position jumps.
        omega = 9.0
        max_accel = 1.5   # m/s²
        max_speed = 0.22  # m/s, correction only
        for axis in range(2):
            error = self.target_offset[axis] - self.offset[axis]
            accel = omega * omega * error - 2 * omega * self.offset_velocity[axis]
            accel = max(-max_accel, min(max_accel, accel))
            velocity = self.offset_velocity[axis] + accel * dt
            velocity = max(-max_speed, min(max_speed, velocity))
            self.offset_velocity[axis] = velocity
            self.offset[axis] += velocity * dt

    def advance(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        if not self.running:
            return
        while dt > 0 and self.running:
            phase = self.phases[self.index]
            remaining = max(0.0, phase.duration - self.elapsed)
            delta = min(dt, remaining)
            self.elapsed += delta
            dt -= delta

            baseline = _lerp(
                self.from_position, phase.target,
                min(1.0, self.elapsed / phase.duration),
            )
            if self.index < 4:
                self._advance_offset(delta)
                self.position = (
                    baseline[0] + self.offset[0],
                    baseline[1] + self.offset[1],
                    baseline[2],
                )
            else:
                self.position = baseline
            self.holding = phase.holding
            self.label = phase.name

            if self.elapsed >= phase.duration - 1e-9:
                if self.index >= 4:
                    # Make scripted endpoints exact (and preserve HOME equality).
                    self.position = phase.target
                self.index += 1
                self.elapsed = 0.0
                # The transport phase starts from the actual corrected grip
                # position; before it, the baseline must not double-count XY.
                self.from_position = (
                    phase.target if self.index < 4 else self.position
                )
                if self.index == len(self.phases):
                    self.running = False
                    self.finished = True
                    self.holding = False
                    self.label = "Savon rangé"
                else:
                    self.label = self.phases[self.index].name
                    self.holding = self.phases[self.index].holding

    @property
    def progress(self) -> int:
        if self.finished:
            return 100
        if not self.running:
            return 0
        duration = sum(p.duration for p in self.phases)
        completed = sum(p.duration for p in self.phases[: self.index])
        return min(99, round(100 * (completed + self.elapsed) / duration))

    @property
    def distance_to_target(self) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(self.position, PHASES[-1].target)))
