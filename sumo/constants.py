"""Physical constants, dimensions and sensor specifications for Mini-Sumo 500g (Haryama)."""

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True)
class DohyoConfig:
    """Official Mini-Sumo Dohyo specifications (All Japan / Robocore / International)."""

    diameter: float = 77.0  # cm
    radius: float = 38.5  # cm
    border_width: float = 2.5  # cm (Tawaramono white line)
    inner_radius: float = 36.0  # cm (black fighting zone)
    shikiri_length: float = 10.0  # cm (starting lines)
    shikiri_width: float = 1.0  # cm
    shikiri_separation: float = 10.0  # cm (distance between starting lines)
    friction: float = 1.35  # Steel/polyurethane dohyo with StickyMAX silicone tires


@dataclass(frozen=True)
class SumoRobotConfig:
    """Haryama Mini-Sumo 500g physical parameters."""

    name: str = "Haryama V3"
    category: str = "Mini-Sumo 500g"
    length: float = 10.0  # cm (max 10.0 cm)
    width: float = 10.0  # cm (max 10.0 cm)
    height: float = 4.68  # cm
    mass: float = 490.0  # g (safe margin below 500g limit)

    # Drivetrain: 2WD with StickyMAX 32mm wheels
    wheel_diameter: float = 3.2  # cm
    wheel_radius: float = 1.6  # cm
    wheel_width: float = 2.1  # cm
    track: float = 7.8  # cm (distance between wheels)
    motor_free_rpm: float = 750.0  # RPM (2S Li-Po / Li-Ion on micro metal gearmotor)
    motor_stall_torque_kgf_cm: float = 1.8  # kgf.cm
    motor_tau: float = 0.05  # Faster response than hockey bot
    deadband: float = 0.08

    # Wedge / Ramp specifications (Haryama V3 CAD)
    wedge_angle_deg: float = 35.0  # degrees
    wedge_depth: float = 3.5  # cm (front projection)
    wedge_friction: float = 0.15  # Low friction polished steel/titanium blade

    # Normal force transfer when under opponent
    wedge_lift_traction_bonus: float = 0.40  # +40% traction when under
    wedge_lift_victim_penalty: float = 0.60  # -60% traction for lifted opponent

    # Sensors: 4 line sensors (corners) + 5 distance sensors (front arc)
    line_sensor_offsets: Tuple[Tuple[float, float], ...] = (
        (4.2, 4.2),  # Front-Left
        (4.2, -4.2),  # Front-Right
        (-4.2, 4.2),  # Rear-Left
        (-4.2, -4.2),  # Rear-Right
    )
    # Sensor ray angles: -60°, -25°, 0°, +25°, +60°
    distance_sensor_angles_rad: Tuple[float, ...] = (
        math.radians(-60.0),
        math.radians(-25.0),
        0.0,
        math.radians(25.0),
        math.radians(60.0),
    )
    distance_sensor_range: float = 80.0  # cm
    distance_sensor_cone_rad: float = math.radians(12.0)  # ~12 deg FoV for ToF/Sharp

    @property
    def max_wheel_speed(self) -> float:
        """Linear speed at free RPM in cm/s."""
        return math.pi * self.wheel_diameter * (self.motor_free_rpm / 60.0)


@dataclass(frozen=True)
class SumoRewardConfig:
    """Reward function weights for Sumo PPO."""

    ring_out_win: float = 20.0  # Terminal win (opponent pushed out)
    ring_out_loss: float = -20.0  # Terminal loss (fell out of dohyo)
    push_progress: float = 4.0  # Shaping pushing opponent towards edge
    center_control: float = 0.5  # Bonus for staying near center
    wedge_advantage: float = 1.0  # Bonus for getting blade under opponent
    line_danger_penalty: float = 0.08  # Penalty when line sensors hit white border
    time_penalty: float = 0.005  # Slight pacing penalty per step
    energy_penalty: float = 0.001  # Energy smoothing penalty


SUMO_OBS_DIM = 18  # 4 line sensors + 5 distance sensors + 3 self vels + 2 opp rel pos + 2 last action + 2 wedge contact
SUMO_ACT_DIM = 2  # Continuous Left PWM, Right PWM [-1, 1]
