"""Deterministic scripted pick-and-place motion; not autonomous grasp planning."""

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
        self.index = 0
        self.elapsed = 0.0
        self.from_position = HOME
        self.running = False
        self.finished = False
        self.holding = False
        self.label = "Prêt"

    def start(self) -> None:
        self.reset()
        self.running = True
        self.label = PHASES[0].name

    def advance(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        if not self.running:
            return
        while dt > 0 and self.running:
            phase = PHASES[self.index]
            remaining = max(0.0, phase.duration - self.elapsed)
            delta = min(dt, remaining)
            self.elapsed += delta
            dt -= delta
            self.position = _lerp(self.from_position, phase.target, min(1.0, self.elapsed / phase.duration))
            self.holding = phase.holding
            self.label = phase.name
            if self.elapsed >= phase.duration - 1e-9:
                self.position = phase.target
                self.index += 1
                self.elapsed = 0.0
                self.from_position = self.position
                if self.index == len(PHASES):
                    self.running = False
                    self.finished = True
                    self.holding = False
                    self.label = "Savon rangé"
                else:
                    self.label = PHASES[self.index].name
                    self.holding = PHASES[self.index].holding

    @property
    def progress(self) -> int:
        if self.finished:
            return 100
        if not self.running:
            return 0
        duration = sum(p.duration for p in PHASES)
        completed = sum(p.duration for p in PHASES[: self.index])
        return min(99, round(100 * (completed + self.elapsed) / duration))

    @property
    def distance_to_target(self) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(self.position, PHASES[-1].target)))
