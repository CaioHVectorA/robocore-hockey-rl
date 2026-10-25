"""Rigid-body Pymunk 2D physics for Mini-Sumo 500g with 240Hz substepping and wedge mechanics."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pymunk

from sumo.constants import CHASSIS_ARCHETYPES, DohyoConfig, SumoRobotConfig
from sumo.dohyo import Dohyo


def build_chassis_polygons(config: SumoRobotConfig) -> List[List[Tuple[float, float]]]:
    """Return convex polygon vertex lists (in local cm, +x forward, +y left) for Pymunk."""
    hl = config.length / 2.0
    hw = config.width / 2.0
    wtype = config.wedge_type

    if wtype == "wide_razor":
        # Ikazuchi 4WD: wide full-width rectangular bulldog with slight rear bevel
        return [[
            (-hl, -hw * 0.92),
            (hl, -hw),
            (hl, hw),
            (-hl, hw * 0.92),
        ]]
    if wtype == "twin_fork":
        # Muramasa Garfo: main body + two protruding left/right fork prongs (2 convex polys)
        base = [
            (-hl, -hw),
            (hl * 0.55, -hw),
            (hl * 0.55, hw),
            (-hl, hw),
        ]
        left_prong = [
            (hl * 0.45, hw * 0.32),
            (hl, hw * 0.42),
            (hl, hw * 0.96),
            (hl * 0.45, hw),
        ]
        right_prong = [
            (hl * 0.45, -hw),
            (hl, -hw * 0.96),
            (hl, -hw * 0.42),
            (hl * 0.45, -hw * 0.32),
        ]
        return [base, left_prong, right_prong]
    if wtype == "v_prow":
        # Kamikaze Velox: angled diamond prow for deflecting head-on charges
        return [[
            (-hl, -hw * 0.85),
            (hl * 0.55, -hw),
            (hl, 0.0),
            (hl * 0.55, hw),
            (-hl, hw * 0.85),
        ]]
    if wtype == "armored":
        # Onimaru Blindado: hexagonal heavy tank with sloped side skirts
        return [[
            (-hl, -hw * 0.78),
            (-hl * 0.2, -hw),
            (hl * 0.65, -hw),
            (hl, -hw * 0.75),
            (hl, hw * 0.75),
            (hl * 0.65, hw),
            (-hl * 0.2, hw),
            (-hl, hw * 0.78),
        ]]
    # Default "tapered" (Haryama V3 CAD profile)
    return [[
        (-hl, -hw * 0.90),
        (hl * 0.62, -hw),
        (hl, -hw * 0.86),
        (hl, hw * 0.86),
        (hl * 0.62, hw),
        (-hl, hw * 0.90),
    ]]


def visual_hull_outline(config: SumoRobotConfig) -> List[Tuple[float, float]]:
    """Return single continuous perimeter outline (including concave fork notch) for rendering."""
    hl = config.length / 2.0
    hw = config.width / 2.0
    wtype = config.wedge_type

    if wtype == "twin_fork":
        return [
            (-hl, -hw),
            (hl * 0.55, -hw),
            (hl, -hw * 0.96),
            (hl, -hw * 0.42),
            (hl * 0.52, -hw * 0.32),
            (hl * 0.52, hw * 0.32),
            (hl, hw * 0.42),
            (hl, hw * 0.96),
            (hl * 0.55, hw),
            (-hl, hw),
        ]
    return build_chassis_polygons(config)[0]


class SumoRobotBody:
    """Rigid body, drivetrain, and contact state for a single Mini-Sumo 500g robot."""

    def __init__(self, space: pymunk.Space, config: SumoRobotConfig, robot_id: int):
        self.config = config
        self.id = robot_id
        self.length = config.length
        self.width = config.width
        self.mass = config.mass

        # Rotational inertia of chassis (g * cm^2)
        moment = self.mass * (self.length**2 + self.width**2) / 11.5
        self.body = pymunk.Body(self.mass, moment)
        self.space = space
        space.add(self.body)

        self.shapes: List[pymunk.Poly] = []
        for poly_verts in build_chassis_polygons(config):
            s = pymunk.Poly(self.body, poly_verts, radius=0.08)
            s.friction = 0.25  # Low metal-on-metal chassis friction so wedges slide & push
            s.elasticity = 0.08  # Inelastic heavy steel/aluminum impact
            s.collision_type = 1
            # Shape filter group so parts of the SAME robot don't collide with each other
            s.filter = pymunk.ShapeFilter(group=robot_id + 10, categories=0b1, mask=0b1)
            self.shapes.append(s)
            space.add(s)

        # Drivetrain & contact states
        self.command = np.zeros(2, dtype=np.float64)
        self.wheel_target = np.zeros(2, dtype=np.float64)  # [left_cm_s, right_cm_s]
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.is_slipping = False

        # Wedge & collision telemetry
        self.traction_multiplier = 1.0
        self.wedge_advantage = False
        self.is_lifted = False
        self.in_contact = False
        self.contact_impulse_mag = 0.0
        self.motor_load = 0.0
        self.ring_out = False
        self.skid_trail: List[Tuple[float, float]] = []

    @property
    def x(self) -> float:
        return float(self.body.position.x)

    @property
    def y(self) -> float:
        return float(self.body.position.y)

    @property
    def theta(self) -> float:
        return float(self.body.angle)

    @property
    def vx(self) -> float:
        return float(self.body.velocity.x)

    @property
    def vy(self) -> float:
        return float(self.body.velocity.y)

    @property
    def omega(self) -> float:
        return float(self.body.angular_velocity)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def forward_speed(self) -> float:
        """Velocity projected along robot's forward +x axis (cm/s)."""
        return self.vx * math.cos(self.theta) + self.vy * math.sin(self.theta)

    @property
    def lateral_speed(self) -> float:
        """Velocity projected along robot's lateral +y (left) axis (cm/s)."""
        return -self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)

    def set_pose(self, x: float, y: float, theta: float):
        self.body.position = (float(x), float(y))
        self.body.angle = float(theta)
        self.body.velocity = (0.0, 0.0)
        self.body.angular_velocity = 0.0
        self.space.reindex_shapes_for_body(self.body)

        self.command[:] = 0.0
        self.wheel_target[:] = 0.0
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.is_slipping = False
        self.traction_multiplier = 1.0
        self.wedge_advantage = False
        self.is_lifted = False
        self.in_contact = False
        self.contact_impulse_mag = 0.0
        self.motor_load = 0.0
        self.ring_out = False
        self.skid_trail.clear()

    def get_world_point(self, local_x: float, local_y: float) -> Tuple[float, float]:
        """Convert robot-frame local coordinates (+x fwd, +y left) to world coordinates."""
        pt = self.body.local_to_world((local_x, local_y))
        return float(pt.x), float(pt.y)

    def apply_drivetrain_forces(self, dt: float, dohyo_friction: float):
        """Apply physical DC motor and StickyMAX tire forces at wheel contact patches."""
        c = self.config
        b = self.body

        # Apply deadband and normalize command [-1, 1] -> [left, right]
        cmd = np.clip(self.command, -1.0, 1.0)
        for i in range(2):
            if abs(cmd[i]) < c.deadband:
                cmd[i] = 0.0

        target_speeds = cmd * c.max_wheel_speed
        alpha = -math.expm1(-dt / max(0.005, c.motor_tau))
        self.wheel_target += (target_speeds - self.wheel_target) * alpha
        self.left_wheel_speed = float(self.wheel_target[0])
        self.right_wheel_speed = float(self.wheel_target[1])

        # Effective normal force includes gravity on mass + magnetic downforce, scaled by wedge lift
        effective_normal_mass = (c.mass + c.downforce_mass_eq) * self.traction_multiplier
        # Maximum tire grip per side (left side vs right side) in g*cm/s^2 (dynes)
        grip_per_side = 0.5 * dohyo_friction * c.tire_friction * effective_normal_mass * 981.0

        # DC Motor stall force per side: torque (kgf.cm) -> g*cm^2/s^2 divided by wheel radius
        wheel_radius = c.wheel_diameter / 2.0
        stall_force_per_side = (c.motor_stall_torque_kgf_cm * 1000.0 * 981.0) / wheel_radius

        self.is_slipping = False
        total_load = 0.0

        # i=0 is Left wheel (+y = +track/2), i=1 is Right wheel (-y = -track/2)
        for i, side_sign in enumerate((1.0, -1.0)):
            wheel_pt = (c.axle_x, side_sign * (c.track / 2.0))
            v_local = b.velocity_at_local_point(wheel_pt).rotated(-b.angle)

            # Longitudinal motor drive force from torque-speed curve
            speed_error = self.wheel_target[i] - v_local.x
            effort = np.clip(speed_error / max(10.0, c.max_wheel_speed * 0.65), -1.0, 1.0)
            fx = float(stall_force_per_side * effort)

            # Lateral tire resistance to sideways skidding
            lat_stiffness = effective_normal_mass * 981.0 * 0.045 / c.turn_scrub_factor
            fy = float(-v_local.y * lat_stiffness)

            # Friction circle limit (StickyMAX tire slip boundary)
            force_mag = math.hypot(fx, fy)
            if force_mag > grip_per_side:
                scale = grip_per_side / max(1e-6, force_mag)
                fx *= scale
                fy *= scale
                self.is_slipping = True

            total_load += abs(effort)
            # Apply local force rotated into world frame at the wheel patch
            f_world = pymunk.Vec2d(fx, fy).rotated(b.angle)
            b.apply_force_at_world_point(f_world, b.local_to_world(wheel_pt))

        # Front wedge / rear skid support drag (prevents endless free spinning)
        omega_drag = -b.angular_velocity * self.mass * 180.0 * c.turn_scrub_factor
        b.torque += omega_drag
        self.motor_load = float(total_load * 0.5)


class SumoPhysicsEngine:
    """Manages 240Hz rigid-body Pymunk simulation, collisions, and wedge lifting."""

    def __init__(
        self,
        dohyo_config: Optional[DohyoConfig] = None,
        blue_config: Optional[SumoRobotConfig] = None,
        red_config: Optional[SumoRobotConfig] = None,
    ):
        self.dohyo_cfg = dohyo_config or DohyoConfig()
        self.blue_cfg = blue_config or CHASSIS_ARCHETYPES["haryama"]
        self.red_cfg = red_config or CHASSIS_ARCHETYPES["ikazuchi_4wd"]
        self.dohyo = Dohyo(self.dohyo_cfg)

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.iterations = 30
        self.space.collision_slop = 0.01

        self.robot_blue = SumoRobotBody(self.space, self.blue_cfg, 0)
        self.robot_red = SumoRobotBody(self.space, self.red_cfg, 1)
        self.robots = [self.robot_blue, self.robot_red]

        # Register Pymunk collision handler between robots
        self.space.on_collision(1, 1, post_solve=self._on_robot_collision)

        self.contact_points_world: List[Tuple[float, float]] = []
        self._substep_contacts: List[Tuple[float, float]] = []
        self._substep_impulse: float = 0.0

        self.time = 0.0
        self.winner: Optional[int] = None

    def set_chassis(
        self,
        blue_config: Optional[SumoRobotConfig] = None,
        red_config: Optional[SumoRobotConfig] = None,
    ):
        """Rebuild physics bodies if chassis archetypes change between matches."""
        need_rebuild = False
        if blue_config is not None and blue_config.archetype_id != self.blue_cfg.archetype_id:
            self.blue_cfg = blue_config
            need_rebuild = True
        if red_config is not None and red_config.archetype_id != self.red_cfg.archetype_id:
            self.red_cfg = red_config
            need_rebuild = True

        if need_rebuild:
            for r in self.robots:
                for s in r.shapes:
                    self.space.remove(s)
                self.space.remove(r.body)
            self.robot_blue = SumoRobotBody(self.space, self.blue_cfg, 0)
            self.robot_red = SumoRobotBody(self.space, self.red_cfg, 1)
            self.robots = [self.robot_blue, self.robot_red]

    def reset(self, layout: str = "standard"):
        """Reset robots to non-overlapping tournament starting positions."""
        self.time = 0.0
        self.winner = None
        self.contact_points_world.clear()
        (b_x, b_y, b_th), (r_x, r_y, r_th) = self.dohyo.starting_positions(layout)
        self.robot_blue.set_pose(b_x, b_y, b_th)
        self.robot_red.set_pose(r_x, r_y, r_th)

    def _on_robot_collision(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """Callback triggered by Pymunk solver whenever the two robots physically collide."""
        impulse = arbiter.total_impulse.length
        self._substep_impulse = max(self._substep_impulse, float(impulse))
        if arbiter.contact_point_set.points:
            for pt in arbiter.contact_point_set.points:
                p_a = pt.point_a
                self._substep_contacts.append((float(p_a.x), float(p_a.y)))

    def _update_wedge_mechanics(self, in_contact: bool):
        """Determine which robot's wedge slips underneath during physical contact."""
        b, r = self.robot_blue, self.robot_red
        b.in_contact = in_contact
        r.in_contact = in_contact

        # Smoothly decay traction modifiers back to 1.0 when separated
        if not in_contact:
            b.traction_multiplier = 1.0
            r.traction_multiplier = 1.0
            b.wedge_advantage = False
            r.wedge_advantage = False
            b.is_lifted = False
            r.is_lifted = False
            return

        dx = r.x - b.x
        dy = r.y - b.y
        dist = max(1e-4, math.hypot(dx, dy))
        ux, uy = dx / dist, dy / dist

        # Alignment of each robot's front (+x axis) toward the opponent's center
        b_align = math.cos(b.theta) * ux + math.sin(b.theta) * uy
        r_align = math.cos(r.theta) * (-ux) + math.sin(r.theta) * (-uy)

        b_reach = math.cos(math.radians(b.config.wedge_reach_deg))
        r_reach = math.cos(math.radians(r.config.wedge_reach_deg))

        b_front_engaged = b_align >= b_reach
        r_front_engaged = r_align >= r_reach

        # Wedge score combines alignment and lower blade angle advantage
        # Lower wedge_angle_deg -> higher score (slips under steeper blade)
        b_wedge_score = b_align + (30.0 - b.config.wedge_angle_deg) * 0.024 if b_front_engaged else -1.0
        r_wedge_score = r_align + (30.0 - r.config.wedge_angle_deg) * 0.024 if r_front_engaged else -1.0

        if b_front_engaged and not r_front_engaged:
            # Blue's front wedge hit Red's side or rear!
            b.wedge_advantage = True
            r.is_lifted = True
            b.traction_multiplier = 1.0 + b.config.wedge_lift_traction_bonus
            penalty = r.config.wedge_lift_victim_penalty * (1.0 - r.config.side_skirt_armor)
            r.traction_multiplier = max(0.18, 1.0 - penalty)
        elif r_front_engaged and not b_front_engaged:
            # Red's front wedge hit Blue's side or rear!
            r.wedge_advantage = True
            b.is_lifted = True
            r.traction_multiplier = 1.0 + r.config.wedge_lift_traction_bonus
            penalty = b.config.wedge_lift_victim_penalty * (1.0 - b.config.side_skirt_armor)
            b.traction_multiplier = max(0.18, 1.0 - penalty)
        elif b_front_engaged and r_front_engaged:
            # Head-on blade-vs-blade clash!
            diff = b_wedge_score - r_wedge_score
            if diff > 0.06:
                # Blue's lower/better-aligned wedge slips under Red's blade
                b.wedge_advantage = True
                r.is_lifted = True
                b.traction_multiplier = 1.0 + b.config.wedge_lift_traction_bonus * 0.75
                r.traction_multiplier = max(0.32, 1.0 - r.config.wedge_lift_victim_penalty * 0.60)
            elif diff < -0.06:
                # Red's lower/better-aligned wedge slips under Blue's blade
                r.wedge_advantage = True
                b.is_lifted = True
                r.traction_multiplier = 1.0 + r.config.wedge_lift_traction_bonus * 0.75
                b.traction_multiplier = max(0.32, 1.0 - b.config.wedge_lift_victim_penalty * 0.60)
            else:
                # Pure head-on shoving lock (Oshi-zumo): slight lateral wedge deflection torque
                b.wedge_advantage = False
                r.wedge_advantage = False
                b.is_lifted = False
                r.is_lifted = False
                b.traction_multiplier = 1.0
                r.traction_multiplier = 1.0
                cross_b = math.cos(b.theta) * uy - math.sin(b.theta) * ux
                shear_sign = 1.0 if cross_b >= 0.0 else -1.0
                b.body.torque += shear_sign * 42000.0
                r.body.torque -= shear_sign * 42000.0

    def step(self, blue_cmd: Tuple[float, float], red_cmd: Tuple[float, float]) -> Dict:
        """Advance simulation by one 30Hz control step using 8 substeps at 240Hz."""
        self.robot_blue.command = np.array(blue_cmd, dtype=np.float64)
        self.robot_red.command = np.array(red_cmd, dtype=np.float64)

        substeps = max(1, self.dohyo_cfg.physics_hz // self.dohyo_cfg.control_hz)
        sub_dt = 1.0 / float(self.dohyo_cfg.physics_hz)

        self._substep_contacts.clear()
        self._substep_impulse = 0.0

        for _ in range(substeps):
            self.robot_blue.apply_drivetrain_forces(sub_dt, self.dohyo_cfg.friction)
            self.robot_red.apply_drivetrain_forces(sub_dt, self.dohyo_cfg.friction)
            self.space.step(sub_dt)
            self.time += sub_dt

            # Update wedge mechanics dynamically during substep if colliding
            in_contact_now = len(self._substep_contacts) > 0 or (
                math.hypot(self.robot_blue.x - self.robot_red.x, self.robot_blue.y - self.robot_red.y)
                <= (self.robot_blue.length + self.robot_red.length) * 0.52
            )
            self._update_wedge_mechanics(in_contact_now)

        self.contact_points_world = self._substep_contacts[-6:]
        self.robot_blue.contact_impulse_mag = self._substep_impulse
        self.robot_red.contact_impulse_mag = self._substep_impulse

        # Record tire skid marks when slipping or under heavy collision load
        for r in self.robots:
            if r.is_slipping or r.in_contact:
                r.skid_trail.append((r.x, r.y))
                if len(r.skid_trail) > 45:
                    r.skid_trail.pop(0)

        # Check ring-out (robot center exits the 38.5 cm outer edge of the Dohyo)
        b_dist = self.dohyo.radial_distance(self.robot_blue.x, self.robot_blue.y)
        r_dist = self.dohyo.radial_distance(self.robot_red.x, self.robot_red.y)

        self.robot_blue.ring_out = b_dist > self.dohyo.radius
        self.robot_red.ring_out = r_dist > self.dohyo.radius

        terminated = False
        if self.robot_blue.ring_out and self.robot_red.ring_out:
            # Whoever went further out first lost
            self.winner = 0 if r_dist >= b_dist else 1
            terminated = True
        elif self.robot_red.ring_out:
            self.winner = 0  # Blue wins!
            terminated = True
        elif self.robot_blue.ring_out:
            self.winner = 1  # Red wins!
            terminated = True

        return {
            "terminated": terminated,
            "winner": self.winner,
            "time": self.time,
            "blue_radial_dist": b_dist,
            "red_radial_dist": r_dist,
            "in_contact": self.robot_blue.in_contact,
            "contact_impulse": self._substep_impulse,
        }
