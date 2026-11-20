"""Finite-budget or persistent retry ordering for the MuJoCo hotel task.

This module does not observe simulation state itself: caller supplies current
bin membership. It never resets or teleports objects and is not a VLA policy.
"""
from __future__ import annotations


class RetryScheduler:
    """Revisit dropped/undelivered items without getting stuck on one cube."""

    def __init__(self, cubes=3, max_retries=-1):
        if cubes < 1 or max_retries < -1:
            raise ValueError("cubes must be positive; max_retries >= -1")
        self.cubes = cubes
        self.max_retries = max_retries
        self.attempts = [0] * cubes  # failures, not the initial attempt
        self.retry_events = []

    def failed(self, index, reason, xyz, t, phase):
        self.attempts[index] += 1
        event = {
            "cube": index,
            "attempt": self.attempts[index],
            "time": round(float(t), 3),
            "phase": str(phase),
            "reason": str(reason),
            "cube_xyz": [round(float(v), 4) for v in xyz],
            "fell_to_floor": bool(float(xyz[2]) < .20),
        }
        self.retry_events.append(event)
        return event

    def pending(self, in_bin):
        """List of all not-yet-delivered cubes, including exhausted ones."""
        return [i for i in range(self.cubes) if not in_bin(i)]

    def eligible(self, in_bin):
        return [i for i in self.pending(in_bin)
                if self.max_retries == -1 or self.attempts[i] <= self.max_retries]

    def choose(self, in_bin, *, current, rotate=False):
        """First failure retries same cube; later failures round-robin.

        This prevents endlessly retrying an inaccessible cube while other
        reachable cubes are still waiting. With -1 it will revisit forever
        subject to the episode's max_steps, never claiming false success.
        """
        candidates = set(self.eligible(in_bin))
        if not candidates:
            return None
        if not rotate and current in candidates:
            return current
        return next(((current + offset) % self.cubes
                     for offset in range(1, self.cubes + 1)
                     if (current + offset) % self.cubes in candidates), None)

    def report(self, in_bin):
        return {
            "retries": len(self.retry_events),
            "attempts_per_cube": list(self.attempts),
            "retry_events": list(self.retry_events),
            "undelivered_cubes": self.pending(in_bin),
            "exhausted_cubes": [i for i in self.pending(in_bin)
                                if self.max_retries >= 0
                                and self.attempts[i] > self.max_retries],
            "max_retries_per_cube": self.max_retries,
            "unbounded_until_max_steps": self.max_retries == -1,
        }
