"""Task rewards, separate from policy observations and robot hardware."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RewardConfig:
    goal: float = 20.0
    progress: float = 3.0
    approach: float = 0.4
    time_penalty: float = 0.006
    energy_penalty: float = 0.001
    reach_bonus: float = 10.0
    reach_distance: float = 10.0
    alignment: float = 0.0
    first_touch: float = 0.0
    carry: float = 0.0
    idle_penalty: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in vars(self).values()):
            raise ValueError("Reward weights must be finite and nonnegative")
        if not 4 <= self.reach_distance <= 30:
            raise ValueError("Reach distance must be between 4 and 30 cm")
