"""Phase-gated, robot-only bias schedule for pretrained SO-101 SmolVLA.

No object state, contact oracle or camera pixels are used to select the bias.
Thresholds come from the pinned reference jaw open (~1.4 rad) and closed
(~-0.05 rad) settings; the phase change is irreversible within an episode.
"""
import math


class BiasSchedule:
    MODES = ("baseline", "full", "approach")

    def __init__(self, mode, bias_mm=-5.0, open_threshold=0.85,
                 close_threshold=0.60, fade_steps=5,
                 max_approach_steps=120):
        if mode not in self.MODES:
            raise ValueError(f"Unknown bias mode: {mode}")
        if not math.isfinite(bias_mm) or abs(bias_mm) > 12:
            raise ValueError("Bias must be finite and within +/-12 mm")
        if not (open_threshold > close_threshold >= 0):
            raise ValueError("Jaw thresholds must have hysteresis")
        if fade_steps < 1 or max_approach_steps < 1:
            raise ValueError("Fade steps and horizon must be positive")
        self.mode = mode
        self.bias_mm = float(bias_mm)
        self.open_threshold = float(open_threshold)
        self.close_threshold = float(close_threshold)
        self.fade_steps = int(fade_steps)
        self.max_approach_steps = int(max_approach_steps)
        self.reset()

    def reset(self):
        self.step = 0
        self.saw_open_jaw = False
        self.transition_step = None
        self.transition_reason = None
        self._fade_counter = 0

    def next_bias(self, jaw_setpoint_rad):
        """Return bias for the upcoming 15 Hz motor command, in millimetres."""
        jaw = float(jaw_setpoint_rad)
        if not math.isfinite(jaw):
            raise ValueError("Jaw motor setpoint must be finite")
        self.step += 1
        if self.mode == "baseline":
            return 0.0
        if self.mode == "full":
            return self.bias_mm
        if jaw >= self.open_threshold:
            self.saw_open_jaw = True
        if self.transition_step is None:
            if self.saw_open_jaw and jaw <= self.close_threshold:
                self.transition_step = self.step
                self.transition_reason = "first_close_after_open"
            elif self.step >= self.max_approach_steps:
                self.transition_step = self.step
                self.transition_reason = "max_approach_horizon"
        if self.transition_step is None:
            return self.bias_mm
        # First close gets 4/5 of the bias, and the fifth gets zero.
        # Once faded the bias cannot reappear during retries/placement.
        self._fade_counter = min(self.fade_steps, self._fade_counter + 1)
        return self.bias_mm * (1 - self._fade_counter / self.fade_steps)

    def summary(self):
        return {
            "mode": self.mode,
            "requested_bias_mm": self.bias_mm if self.mode != "baseline" else 0,
            "jaw_open_seen": self.saw_open_jaw,
            "transition_step": self.transition_step,
            "transition_reason": self.transition_reason,
            "transition_fade_steps": self.fade_steps,
            "open_threshold_rad": self.open_threshold,
            "close_threshold_rad": self.close_threshold,
            "max_approach_steps": self.max_approach_steps,
        }
