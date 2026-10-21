"""Versioned profiles. Units: centimetres, grams, seconds.

Rules are limits. Robot/actuator parameters are uncalibrated design assumptions;
see docs/COMPETITION_AND_CALIBRATION.md.
"""

from dataclasses import asdict, dataclass
import math

RULES_SOURCE = "https://robocore-eventos.s3.sa-east-1.amazonaws.com/public/RoboCore_Hockey_Rules.pdf"


@dataclass(frozen=True)
class ArenaConfig:
    name: str = "micro-2024"
    length: float = 300.0
    width: float = 150.0
    corner_radius: float = 12.5
    goal_width: float = 52.0
    goal_height: float = 18.0
    goal_depth: float = 12.0  # Assumed depth, NOT the opening height.
    puck_diameter: float = 3.17
    puck_thickness: float = 1.27
    puck_mass: float = 15.0  # Weigh the actual puck.
    match_seconds: float = 300.0
    rules_revision: str = "2024-01-15"
    source: str = RULES_SOURCE

    def __post_init__(self):
        if not all(
            math.isfinite(v)
            for v in (
                self.length,
                self.width,
                self.corner_radius,
                self.goal_width,
                self.goal_depth,
                self.goal_height,
                self.puck_diameter,
                self.puck_thickness,
                self.puck_mass,
                self.match_seconds,
            )
        ):
            raise ValueError("Physical parameters must be finite")
        if (
            min(
                self.length,
                self.width,
                self.corner_radius,
                self.goal_width,
                self.goal_depth,
                self.puck_diameter,
                self.puck_mass,
                self.match_seconds,
            )
            <= 0
        ):
            raise ValueError("Dimensions, duration and mass must be positive")
        if 2 * self.corner_radius >= min(self.length, self.width):
            raise ValueError("Corner radius exceeds arena")
        if self.goal_width >= self.width - 2 * self.corner_radius:
            raise ValueError("Goal must fit the straight end wall")

    @property
    def half_length(self):
        return self.length / 2

    @property
    def half_width(self):
        return self.width / 2

    @property
    def puck_radius(self):
        return self.puck_diameter / 2

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RobotConfig:
    length: float = 16.0
    width: float = 17.0
    mass: float = 454.0
    track: float = 14.5
    axle_x: float = -2.5
    skid_x: float = 6.8
    max_wheel_speed: float = 85.0
    motor_tau: float = 0.12
    tire_friction: float = 0.85
    skid_friction: float = 0.08
    deadband: float = 0.12
    inertia: float | None = None
    provenance: str = "unmeasured design model, not certified HockeyBot specifications"

    def __post_init__(self):
        values = (
            self.length,
            self.width,
            self.mass,
            self.track,
            self.axle_x,
            self.skid_x,
            self.max_wheel_speed,
            self.motor_tau,
            self.tire_friction,
            self.skid_friction,
            self.deadband,
        )
        if not all(math.isfinite(v) for v in values) or min(self.tire_friction, self.skid_friction) < 0:
            raise ValueError("Invalid physical parameter")
        if self.inertia is not None and (not math.isfinite(self.inertia) or self.inertia <= 0):
            raise ValueError("Inertia must be positive and finite")
        if not 0 < self.length <= 20 or not 0 < self.width <= 20 or not 0 < self.mass <= 454:
            raise ValueError("Micro robot: maximum 20 x 20 cm and 454 g")
        if not 0 < self.track <= self.width or self.motor_tau <= 0 or self.max_wheel_speed <= 0:
            raise ValueError("Invalid drivetrain")
        if not 0 <= self.deadband < 1 or self.axle_x >= self.skid_x:
            raise ValueError("Invalid deadband/support points")


@dataclass(frozen=True)
class SimulationConfig:
    control_hz: int = 30
    physics_hz: int = 240
    puck_friction: float = 0.08
    restitution: float = 0.55
    radio_latency: float = 0.03
    radio_jitter: float = 0.01
    packet_loss: float = 0.02
    watchdog_seconds: float = 0.20
    vision_latency: float = 1 / 30
    vision_noise: float = 0.18
    vision_dropout: float = 0.025

    def __post_init__(self):
        if self.physics_hz <= 0 or not 0 <= self.restitution <= 1 or self.watchdog_seconds <= 0:
            raise ValueError("Invalid solver or watchdog configuration")
        if self.control_hz <= 0 or self.physics_hz % self.control_hz:
            raise ValueError("Physics frequency must be a multiple of control frequency")
        if not 0 <= self.packet_loss < 1 or not 0 <= self.vision_dropout < 1:
            raise ValueError("Invalid loss probability")
        if (
            min(
                self.radio_latency,
                self.radio_jitter,
                self.vision_latency,
                self.vision_noise,
                self.puck_friction,
            )
            < 0
        ):
            raise ValueError("Noise, friction and delays must be nonnegative")


PROFILES = {
    "micro-2024": ArenaConfig(),
    "micro-large": ArenaConfig(name="micro-large", length=366.0, width=183.0, corner_radius=15.0),
    "tabletop": ArenaConfig(
        name="tabletop",
        length=180.0,
        width=120.0,
        corner_radius=20.0,
        goal_width=35.0,
        puck_diameter=5.0,
        puck_mass=38.0,
        puck_thickness=1.4,
        match_seconds=180.0,
        rules_revision="custom",
        source="practice table; not competition rules",
    ),
}


class HockeyConstants:
    """Compatibility aliases for the default profile. New code uses configs."""

    ARENA_LENGTH, ARENA_WIDTH = 300.0, 150.0
    HALF_LENGTH, HALF_WIDTH = 150.0, 75.0
    CORNER_RADIUS, GOAL_WIDTH, HALF_GOAL_WIDTH, GOAL_DEPTH = 12.5, 52.0, 26.0, 12.0
    ROBOTS_PER_TEAM = 3
    ROLE_STRIKER, ROLE_DEFENDER, ROLE_GOALIE = 0, 1, 2
    ROBOT_LENGTH, ROBOT_WIDTH, ROBOT_MASS = 16.0, 17.0, 454.0
    HALF_ROBOT_L, HALF_ROBOT_W = 8.0, 8.5
    WHEELBASE, MAX_WHEEL_SPEED, GRAVITY = 14.5, 85.0, 981.0
    PUCK_DIAMETER, PUCK_RADIUS, PUCK_SLIDING_FRICTION = 3.17, 1.585, 0.08
    DT, MATCH_TIME_SECONDS, BLE_CONTROL_RATE_HZ = 1 / 30, 300.0, 30.0
