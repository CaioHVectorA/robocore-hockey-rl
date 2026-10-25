"""Physical constants, competitive chassis archetypes, and sensor specs for Mini-Sumo 500g."""

from dataclasses import dataclass
import math
from typing import Dict, Tuple


@dataclass(frozen=True)
class DohyoConfig:
    """Official Mini-Sumo Dohyo specifications (All Japan / RoboCore / JSumo)."""

    diameter: float = 77.0  # cm
    radius: float = 38.5  # cm
    border_width: float = 2.5  # cm (Tawaramono white border line)
    inner_radius: float = 36.0  # cm (black fighting surface radius)
    shikiri_length: float = 10.0  # cm (center starting lines)
    shikiri_width: float = 1.0  # cm
    shikiri_separation: float = 10.0  # cm (distance between the two Shikiri lines)
    friction: float = 1.25  # Steel/Polyurethane dohyo surface friction coefficient
    physics_hz: int = 240  # 240 Hz rigid-body substepping (8 substeps per 30 Hz control frame)
    control_hz: int = 30  # 30 Hz sensor & RL control loop


@dataclass(frozen=True)
class SumoRobotConfig:
    """Physical and geometric specification for a Mini-Sumo 500g robot archetype."""

    name: str = "Haryama V3"
    archetype_id: str = "haryama"
    subtitle: str = "2WD StickyMAX · Rampa Titânio 24°"
    length: float = 9.8  # cm (limit 10.0 cm)
    width: float = 9.8  # cm (limit 10.0 cm)
    mass: float = 490.0  # g (limit 500.0 g)
    downforce_mass_eq: float = 420.0  # g equivalent magnetic/low-CG downforce on steel dohyo

    # Drivetrain
    num_wheels: int = 2  # 2WD or 4WD
    wheel_diameter: float = 3.2  # cm (StickyMAX 32mm)
    wheel_width: float = 2.1  # cm
    track: float = 7.8  # cm (lateral distance between left and right wheel centers)
    axle_x: float = -1.2  # cm (rearward offset of main drive axle for long front wedge)
    motor_free_rpm: float = 520.0  # RPM (~87 cm/s realistic tournament speed)
    motor_stall_torque_kgf_cm: float = 1.8  # Total stall torque per side (kgf.cm)
    motor_tau: float = 0.045  # s (electromechanical response time)
    deadband: float = 0.06
    tire_friction: float = 1.28  # Silicone StickyMAX tire compound
    turn_scrub_factor: float = 1.0  # >1.0 for 4WD chassis due to skid-steer scrubbing

    # Wedge / Blade mechanics
    wedge_type: str = "tapered"  # "tapered", "wide_razor", "twin_fork", "v_prow", "armored"
    wedge_angle_deg: float = 24.0  # Lower angle helps in angled clashes
    wedge_reach_deg: float = 38.0  # Half-angle of frontal wedge zone
    wedge_lift_traction_bonus: float = 0.48  # +48% normal grip when opponent sits on our blade
    wedge_lift_victim_penalty: float = 0.65  # Opponent loses 65% grip when lifted onto blade
    side_skirt_armor: float = 0.0  # Reduction in lift penalty when attacked from the side

    # Visual styling (RGB tuples)
    primary_color: Tuple[int, int, int] = (34, 118, 226)
    accent_color: Tuple[int, int, int] = (255, 210, 40)

    # 4 Corner Optical Border Sensors (QRE1113 / TCRT5000): (local_x, local_y) in cm
    # Convention: +x is Forward, +y is Left, -y is Right
    line_sensor_offsets: Tuple[Tuple[float, float], ...] = (
        (4.4, 4.3),   # 0: Front-Left (FL)
        (4.4, -4.3),  # 1: Front-Right (FR)
        (-4.4, 4.3),  # 2: Rear-Left (RL)
        (-4.4, -4.3), # 3: Rear-Right (RR)
    )

    # 7 Competitive Distance Sensors (VL53L0X ToF / JSumo IR):
    # Ordered Left-to-Right in robot frame (+y is Left -> positive angles, -y is Right -> negative angles):
    # Index 0: L90 (+75° Left Flank)
    # Index 1: L45 (+36° Wide Left)
    # Index 2: L15 (+15° Front-Left Lock)
    # Index 3: C0  (  0° Dead-Center Lock)
    # Index 4: R15 (-15° Front-Right Lock)
    # Index 5: R45 (-36° Wide Right)
    # Index 6: R90 (-75° Right Flank)
    distance_sensor_angles_deg: Tuple[float, ...] = (75.0, 36.0, 15.0, 0.0, -15.0, -36.0, -75.0)
    distance_sensor_range: float = 75.0  # cm

    @property
    def distance_sensor_angles_rad(self) -> Tuple[float, ...]:
        return tuple(math.radians(a) for a in self.distance_sensor_angles_deg)

    @property
    def max_wheel_speed(self) -> float:
        """Linear wheel speed at free RPM in cm/s."""
        return math.pi * self.wheel_diameter * (self.motor_free_rpm / 60.0)


# ==============================================================================
# REAL-WORLD INSPIRED MINI-SUMO 500G CHASSIS ARCHETYPES
# ==============================================================================

CHASSIS_ARCHETYPES: Dict[str, SumoRobotConfig] = {
    "haryama": SumoRobotConfig(
        name="Haryama V3",
        archetype_id="haryama",
        subtitle="2WD StickyMAX · Rampa Titânio 22° (Equilibrado)",
        length=9.8,
        width=9.8,
        mass=495.0,
        downforce_mass_eq=460.0,
        num_wheels=2,
        track=7.8,
        axle_x=-1.2,
        motor_free_rpm=540.0,
        motor_stall_torque_kgf_cm=2.0,
        wedge_type="tapered",
        wedge_angle_deg=22.0,
        primary_color=(36, 124, 235),
        accent_color=(255, 215, 0),
    ),
    "ikazuchi_4wd": SumoRobotConfig(
        name="Ikazuchi 4WD",
        archetype_id="ikazuchi_4wd",
        subtitle="4WD Monobloco Japonês · Ímãs Neodímio (Pusher)",
        length=10.0,
        width=10.0,
        mass=500.0,
        downforce_mass_eq=560.0,
        num_wheels=4,
        wheel_diameter=2.8,
        track=8.2,
        axle_x=0.0,
        motor_free_rpm=490.0,
        motor_stall_torque_kgf_cm=2.2,
        turn_scrub_factor=1.35,  # 4WD resists fast turning, Vulnerable to flanking!
        wedge_type="wide_razor",
        wedge_angle_deg=21.0,
        wedge_reach_deg=40.0,
        primary_color=(215, 48, 62),
        accent_color=(240, 245, 255),
    ),
    "muramasa_fork": SumoRobotConfig(
        name="Muramasa Garfo",
        archetype_id="muramasa_fork",
        subtitle="2WD Cunha Bipartida (Garfo) · Anti-Rampa 18°",
        length=10.0,
        width=9.4,
        mass=485.0,
        downforce_mass_eq=450.0,
        num_wheels=2,
        wheel_diameter=3.2,
        track=7.4,
        axle_x=-1.5,
        motor_free_rpm=560.0,
        motor_stall_torque_kgf_cm=1.95,
        wedge_type="twin_fork",
        wedge_angle_deg=18.0,  # Razor-thin fork tips slip under flat blades on angled contact
        wedge_reach_deg=36.0,
        wedge_lift_traction_bonus=0.58,
        primary_color=(16, 168, 118),
        accent_color=(255, 196, 0),
    ),
    "kamikaze_prow": SumoRobotConfig(
        name="Kamikaze Velox",
        archetype_id="kamikaze_prow",
        subtitle="2WD Interceptor Europeu · Alta Agilidade 640 RPM",
        length=9.5,
        width=9.2,
        mass=470.0,
        downforce_mass_eq=390.0,
        num_wheels=2,
        wheel_diameter=3.2,
        track=6.8,
        axle_x=-0.8,
        motor_free_rpm=640.0,
        motor_stall_torque_kgf_cm=1.75,
        motor_tau=0.030,
        wedge_type="v_prow",
        wedge_angle_deg=25.0,
        primary_color=(148, 76, 235),
        accent_color=(0, 240, 255),
    ),
    "onimaru_tank": SumoRobotConfig(
        name="Onimaru Blindado",
        archetype_id="onimaru_tank",
        subtitle="2WD Núcleo de Latão + Saias Laterais Anti-Flanco",
        length=9.6,
        width=10.0,
        mass=500.0,
        downforce_mass_eq=510.0,
        num_wheels=2,
        wheel_diameter=3.2,
        track=8.0,
        axle_x=-1.0,
        motor_free_rpm=500.0,
        motor_stall_torque_kgf_cm=2.1,
        wedge_type="armored",
        wedge_angle_deg=23.0,
        side_skirt_armor=0.40,  # Harder to lift when hit on the flank
        primary_color=(225, 130, 25),
        accent_color=(255, 240, 150),
    ),
}


@dataclass(frozen=True)
class SumoRewardConfig:
    """Reward function weights for competitive Mini-Sumo PPO."""

    ring_out_win: float = 25.0  # Terminal win (opponent pushed out of Dohyo)
    ring_out_loss: float = -25.0  # Terminal loss (self fell out of Dohyo)
    push_progress: float = 6.0  # Potential-based shaping for pushing opponent radially outward
    approach_progress: float = 1.5  # Closing distance to opponent when not in contact
    alignment_bonus: float = 0.08  # Keeping front blade pointed at opponent
    center_control: float = 0.04  # Bonus for staying near center vs opponent
    wedge_advantage: float = 0.25  # Active contact with our blade under opponent
    shove_impulse_bonus: float = 0.15  # Actively driving opponent backward during collision
    line_danger_penalty: float = 0.15  # Penalty when line sensors trigger near edge without contact
    time_penalty: float = 0.008  # Pacing penalty per frame


SUMO_OBS_DIM = 24
SUMO_ACT_DIM = 2
