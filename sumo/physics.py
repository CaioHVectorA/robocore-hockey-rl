"""Pymunk 2D physics engine for Mini-Sumo with StickyMAX tires and wedge dynamics."""

import math
from typing import Dict, Optional, Tuple
import numpy as np
import pymunk

from sumo.constants import DohyoConfig, SumoRobotConfig
from sumo.dohyo import Dohyo


class SumoRobotBody:
    """Rigid body, wheel drive and sensor anchors for a single Mini-Sumo robot."""

    def __init__(self, space: pymunk.Space, config: SumoRobotConfig, robot_id: int):
        self.config = config
        self.id = robot_id
        self.length = config.length
        self.width = config.width
        self.mass = config.mass

        # Pymunk Moment of Inertia for rectangle: m * (w^2 + l^2) / 12
        moment = self.mass * (self.length**2 + self.width**2) / 12.0
        self.body = pymunk.Body(self.mass, moment)

        # Main chassis shape: 10x10 cm box
        hl, hw = self.length / 2.0, self.width / 2.0
        # Wedge profile: slightly tapered front to simulate CAD blade
        vertices = [
            (-hl, -hw),
            (hl * 0.7, -hw),
            (hl, -hw * 0.9),  # Front-right blade tip
            (hl, hw * 0.9),   # Front-left blade tip
            (hl * 0.7, hw),
            (-hl, hw),
        ]
        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.friction = 0.5
        self.shape.elasticity = 0.15
        self.shape.collision_type = 1  # Sumo robots

        space.add(self.body, self.shape)

        # Dynamic state
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.target_left_speed = 0.0
        self.target_right_speed = 0.0
        self.traction_multiplier = 1.0  # Modified by wedge lifting
        self.wedge_advantage = False
        self.is_lifted = False
        self.ring_out = False

    @property
    def x(self) -> float:
        return self.body.position.x

    @property
    def y(self) -> float:
        return self.body.position.y

    @property
    def theta(self) -> float:
        return self.body.angle

    @property
    def vx(self) -> float:
        return self.body.velocity.x

    @property
    def vy(self) -> float:
        return self.body.velocity.y

    @property
    def omega(self) -> float:
        return self.body.angular_velocity

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def set_pose(self, x: float, y: float, theta: float):
        self.body.position = (x, y)
        self.body.angle = theta
        self.body.velocity = (0.0, 0.0)
        self.body.angular_velocity = 0.0
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.target_left_speed = 0.0
        self.target_right_speed = 0.0
        self.traction_multiplier = 1.0
        self.wedge_advantage = False
        self.is_lifted = False
        self.ring_out = False

    def get_world_point(self, local_x: float, local_y: float) -> Tuple[float, float]:
        """Convert robot-frame local coordinates to world frame."""
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        wx = self.x + cos_t * local_x - sin_t * local_y
        wy = self.y + sin_t * local_x + cos_t * local_y
        return wx, wy

    def step_motors(self, left_cmd: float, right_cmd: float, dt: float, dohyo_friction: float):
        """Simulate motor lag, wheel ground traction and apply forces to body."""
        # Clamp commands to [-1, 1]
        clamped_left = float(np.clip(left_cmd, -1.0, 1.0))
        clamped_right = float(np.clip(right_cmd, -1.0, 1.0))

        # Deadband
        if abs(clamped_left) < self.config.deadband:
            clamped_left = 0.0
        if abs(clamped_right) < self.config.deadband:
            clamped_right = 0.0

        max_spd = self.config.max_wheel_speed
        self.target_left_speed = clamped_left * max_spd
        self.target_right_speed = clamped_right * max_spd

        # Motor electrical/mechanical first-order response: v += (v_target - v) * (dt / tau)
        alpha = min(1.0, dt / max(0.001, self.config.motor_tau))
        self.left_wheel_speed += (self.target_left_speed - self.left_wheel_speed) * alpha
        self.right_wheel_speed += (self.target_right_speed - self.right_wheel_speed) * alpha

        # Forward velocity and yaw rate from differential drive kinematics
        v_forward_target = (self.left_wheel_speed + self.right_wheel_speed) / 2.0
        omega_target = (self.right_wheel_speed - self.left_wheel_speed) / self.config.track

        # Forward unit vector and lateral unit vector
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        fwd = pymunk.Vec2d(cos_t, sin_t)
        lat = pymunk.Vec2d(-sin_t, cos_t)

        # Decompose current body velocity
        v_fwd_curr = self.body.velocity.dot(fwd)
        v_lat_curr = self.body.velocity.dot(lat)

        # StickyMAX tire grip: normal force * friction * traction_multiplier
        effective_mu = dohyo_friction * self.traction_multiplier
        max_friction_accel = effective_mu * 981.0  # cm/s^2 (g = 981 cm/s^2)

        # Longitudinal drive force
        dv_fwd = v_forward_target - v_fwd_curr
        a_fwd = np.clip(dv_fwd / dt, -max_friction_accel, max_friction_accel)

        # Lateral grip (high resistance to skidding)
        a_lat = np.clip(-v_lat_curr / dt, -max_friction_accel * 1.2, max_friction_accel * 1.2)

        # Angular torque from differential drive
        d_omega = omega_target - self.body.angular_velocity
        max_alpha = (max_friction_accel / (self.config.track / 2.0))
        ang_accel = np.clip(d_omega / dt, -max_alpha, max_alpha)

        # Apply forces to body
        total_accel = fwd * a_fwd + lat * a_lat
        self.body.velocity += total_accel * dt
        self.body.angular_velocity += ang_accel * dt


class SumoPhysicsEngine:
    """Manages the 2D physics world, collision handling and wedge dynamics."""

    def __init__(self, dohyo_config: Optional[DohyoConfig] = None, robot_config: Optional[SumoRobotConfig] = None):
        self.dohyo_cfg = dohyo_config or DohyoConfig()
        self.robot_cfg = robot_config or SumoRobotConfig()
        self.dohyo = Dohyo(self.dohyo_cfg)

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)  # Top-down plane view
        self.space.damping = 0.92

        self.robot_blue = SumoRobotBody(self.space, self.robot_cfg, 0)
        self.robot_red = SumoRobotBody(self.space, self.robot_cfg, 1)
        self.robots = [self.robot_blue, self.robot_red]

        self.dt = 1.0 / 30.0  # 30 Hz step
        self.time = 0.0
        self.winner: Optional[int] = None

    def reset(self, layout: str = "standard"):
        """Reset robots to layout positions and clear physics state."""
        self.time = 0.0
        self.winner = None
        (b_x, b_y, b_th), (r_x, r_y, r_th) = self.dohyo.starting_positions(layout)
        self.robot_blue.set_pose(b_x, b_y, b_th)
        self.robot_red.set_pose(r_x, r_y, r_th)

    def _resolve_wedge_dynamics(self):
        """Calculate ramp/blade penetration and transfer normal force between robots."""
        b, r = self.robot_blue, self.robot_red
        dist = math.hypot(b.x - r.x, b.y - r.y)

        # Reset multipliers
        b.traction_multiplier = 1.0
        r.traction_multiplier = 1.0
        b.wedge_advantage = False
        r.wedge_advantage = False
        b.is_lifted = False
        r.is_lifted = False

        contact_distance = (b.length + r.length) * 0.55  # ~11 cm
        if dist > contact_distance:
            return

        # Vector from Blue to Red
        dx, dy = (r.x - b.x) / dist, (r.y - b.y) / dist

        # Blue alignment toward Red: dot(b.fwd, dir_to_red)
        b_dot = math.cos(b.theta) * dx + math.sin(b.theta) * dy
        # Red alignment toward Blue: dot(r.fwd, -dir_to_red)
        r_dot = math.cos(r.theta) * (-dx) + math.sin(r.theta) * (-dy)

        # A robot has wedge advantage if its front faces the opponent
        # and has better engagement angle than the opponent's front
        wedge_threshold = math.cos(math.radians(self.robot_cfg.wedge_angle_deg))

        blue_under = b_dot > wedge_threshold and (b_dot > r_dot + 0.15)
        red_under = r_dot > wedge_threshold and (r_dot > b_dot + 0.15)

        if blue_under:
            b.wedge_advantage = True
            r.is_lifted = True
            b.traction_multiplier = 1.0 + self.robot_cfg.wedge_lift_traction_bonus
            r.traction_multiplier = max(0.1, 1.0 - self.robot_cfg.wedge_lift_victim_penalty)
        elif red_under:
            r.wedge_advantage = True
            b.is_lifted = True
            r.traction_multiplier = 1.0 + self.robot_cfg.wedge_lift_traction_bonus
            b.traction_multiplier = max(0.1, 1.0 - self.robot_cfg.wedge_lift_victim_penalty)

    def step(self, blue_cmd: Tuple[float, float], red_cmd: Tuple[float, float]) -> Dict:
        """Advance simulation by 1 tick (1/30 s)."""
        self._resolve_wedge_dynamics()

        # Step motors
        self.robot_blue.step_motors(blue_cmd[0], blue_cmd[1], self.dt, self.dohyo_cfg.friction)
        self.robot_red.step_motors(red_cmd[0], red_cmd[1], self.dt, self.dohyo_cfg.friction)

        # Advance Pymunk space
        self.space.step(self.dt)
        self.time += self.dt

        # Check ring out (outer edge of dohyo)
        b_dist = self.dohyo.radial_distance(self.robot_blue.x, self.robot_blue.y)
        r_dist = self.dohyo.radial_distance(self.robot_red.x, self.robot_red.y)

        # Robot is ring-out when center is beyond outer radius
        self.robot_blue.ring_out = b_dist > self.dohyo.radius
        self.robot_red.ring_out = r_dist > self.dohyo.radius

        terminated = False
        if self.robot_blue.ring_out and self.robot_red.ring_out:
            self.winner = -1  # Draw
            terminated = True
        elif self.robot_red.ring_out:
            self.winner = 0  # Blue wins!
            terminated = True
        elif self.robot_blue.ring_out:
            self.winner = 1  # Red wins!
            terminated = True

        return {
            "time": self.time,
            "terminated": terminated,
            "winner": self.winner,
            "blue_ring_out": self.robot_blue.ring_out,
            "red_ring_out": self.robot_red.ring_out,
            "blue_dist": b_dist,
            "red_dist": r_dist,
            "blue_wedge": self.robot_blue.wedge_advantage,
            "red_wedge": self.robot_red.wedge_advantage,
        }
