"""Shared action → PWM → wheel model for simulation and hardware."""

import numpy as np


def action_to_pwm(action, deadband=0.12, trim=(1.0, 1.0)):
    u = np.clip(np.asarray(action, dtype=float) * np.asarray(trim), -1.0, 1.0)
    if not np.isfinite(u).all():
        raise ValueError("Non-finite motor command")
    pwm = np.where(np.abs(u) < 0.015, 0.0, np.sign(u) * (deadband + (1 - deadband) * np.abs(u)))
    return np.rint(pwm * 255).astype(np.int16)


def pwm_to_normalized(pwm, deadband=0.12):
    u = np.clip(np.asarray(pwm, dtype=float) / 255.0, -1.0, 1.0)
    return np.sign(u) * np.maximum(0.0, np.abs(u) - deadband) / (1 - deadband)
