"""Rigid-body hockey: motor forces at the axle and geometric Pymunk contacts.

Units: cm, g, s. World +y points up. Possession is diagnostic, never a constraint.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import pymunk
from core.constants import ArenaConfig, RobotConfig, SimulationConfig
from core.geometry import (
    chassis_outline,
    wheel_footprints,
    collision_parts,
    estimated_inertia,
    footprint_vertices,
)
from core.actuators import action_to_pwm, pwm_to_normalized
from core.arena_geometry import arena_wall_segments


@dataclass
class HockeyRobot:
    team: int
    role_id: int
    body: pymunk.Body
    config: RobotConfig
    touched_puck: bool = False
    has_puck_in_scoop: bool = False
    is_slipping: bool = False
    motor_gain_l: float = 1.0
    motor_gain_r: float = 1.0
    deadband: float = 0.12
    command: np.ndarray = field(default_factory=lambda: np.zeros(2))
    wheel_target: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_packet: float = -math.inf

    @property
    def x(self):
        return self.body.position.x

    @x.setter
    def x(self, v):
        self.body.position = (v, self.y)

    @property
    def y(self):
        return self.body.position.y

    @y.setter
    def y(self, v):
        self.body.position = (self.x, v)

    @property
    def theta(self):
        return self.body.angle

    @theta.setter
    def theta(self, v):
        self.body.angle = v

    @property
    def vx(self):
        return self.body.velocity.x

    @property
    def vy(self):
        return self.body.velocity.y

    @property
    def omega(self):
        return self.body.angular_velocity

    @property
    def inertia_z(self):
        return self.body.moment

    @inertia_z.setter
    def inertia_z(self, v):
        self.body.moment = v

    @property
    def mass(self):
        return self.body.mass

    @property
    def v_left(self):
        return (
            self.body.velocity_at_local_point((self.config.axle_x, self.config.track / 2))
            .rotated(-self.theta)
            .x
        )

    @property
    def v_right(self):
        return (
            self.body.velocity_at_local_point((self.config.axle_x, -self.config.track / 2))
            .rotated(-self.theta)
            .x
        )


class HockeyPuck:
    def __init__(self, body):
        self.body = body
        self.last_touch_team = -1
        self.trail = []

    @property
    def x(self):
        return self.body.position.x

    @x.setter
    def x(self, v):
        self.body.position = (v, self.y)

    @property
    def y(self):
        return self.body.position.y

    @y.setter
    def y(self, v):
        self.body.position = (self.x, v)

    @property
    def vx(self):
        return self.body.velocity.x

    @vx.setter
    def vx(self, v):
        self.body.velocity = (v, self.vy)

    @property
    def vy(self):
        return self.body.velocity.y

    @vy.setter
    def vy(self, v):
        self.body.velocity = (self.vx, v)


class HockeyPhysicsEngine:
    def __init__(
        self,
        robots_per_team=3,
        domain_randomization=True,
        arena=None,
        robot_config=None,
        simulation=None,
        seed=None,
        opponent_robots=None,
    ):
        if not 1 <= robots_per_team <= 3:
            raise ValueError("Use one to three robots per team")
        self.arena = arena or ArenaConfig()
        self.robot_config = robot_config or RobotConfig()
        self.simulation = simulation or SimulationConfig()
        self.robots_per_team = robots_per_team
        self.opponent_robots = robots_per_team if opponent_robots is None else opponent_robots
        if not isinstance(self.opponent_robots, int) or not 0 <= self.opponent_robots <= 3:
            raise ValueError("Use zero to three opposing robots")
        self.domain_randomization = domain_randomization
        self.rng = np.random.default_rng(seed)
        self.reset_match(seed)

    @property
    def all_robots(self):
        return self.blue_robots + self.red_robots

    @property
    def dt(self):
        return 1 / self.simulation.control_hz

    def reset_match(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.score_blue = self.score_red = 0
        self.match_time = 0.0
        self.reset_positions(reset_clock=True)

    def reset_positions(self, faceoff=True, reset_clock=False):
        if reset_clock:
            self.match_time = 0.0
        self.steps = 0
        self.episode_time = 0.0
        self.goal_latched = False
        self.ble_queue = []
        self.sequence = 0
        self.delivered_sequence = -1
        self.friction_mu = self.simulation.puck_friction * (
            self.rng.uniform(0.8, 1.2) if self.domain_randomization else 1.0
        )
        self.space = pymunk.Space()
        self.space.iterations = 20
        self.space.collision_slop = 0.02
        self._build_walls()
        self.blue_robots = []
        self.red_robots = []
        cfg = self.robot_config
        for team, collection in [(0, self.blue_robots), (1, self.red_robots)]:
            sign = 1 if team == 0 else -1
            for role in range(self.robots_per_team if team == 0 else self.opponent_robots):
                parts = collision_parts(cfg)
                body = pymunk.Body(cfg.mass, cfg.inertia or estimated_inertia(cfg))
                x = (-self.arena.length * 0.22, -self.arena.length * 0.32, -self.arena.half_length + 18)[
                    role
                ] * sign
                y = (0.0, self.arena.width * 0.23, 0.0)[role] * sign
                body.position = (x, y)
                body.angle = 0 if team == 0 else math.pi
                self.space.add(body)
                for p in parts:
                    s = pymunk.Poly(body, p)
                    s.friction = 0.4
                    s.elasticity = math.sqrt(self.simulation.restitution)
                    s.collision_type = 2
                    self.space.add(s)
                r = HockeyRobot(team, role, body, cfg, deadband=cfg.deadband)
                if self.domain_randomization:
                    r.motor_gain_l, r.motor_gain_r = self.rng.uniform(0.94, 1.06, 2)
                    r.deadband = float(self.rng.uniform(0.10, 0.14))
                collection.append(r)
        a = self.arena
        body = pymunk.Body(a.puck_mass, pymunk.moment_for_circle(a.puck_mass, 0, a.puck_radius))
        body.position = tuple(self.rng.uniform(-1.0, 1.0, 2)) if faceoff else (0.0, 0.0)
        s = pymunk.Circle(body, a.puck_radius)
        s.elasticity = math.sqrt(self.simulation.restitution)
        s.friction = 0.2
        s.collision_type = 1
        self.space.add(body, s)
        self.puck = HockeyPuck(body)
        self.space.on_collision(1, 2, post_solve=self._puck_contact)
        self.ble_latency_ms = 0.0

    def _build_walls(self):
        for p, q in arena_wall_segments(self.arena):
            s = pymunk.Segment(self.space.static_body, p, q, 0.15)
            s.elasticity = math.sqrt(self.simulation.restitution)
            s.friction = 0.2
            self.space.add(s)

    def _puck_contact(self, arbiter, space, data):
        for r in self.all_robots:
            if r.body is arbiter.shapes[1].body:
                r.touched_puck = True
                self.puck.last_touch_team = r.team

    def _motor_forces(self, r, dt):
        c = r.config
        b = r.body
        actual = pwm_to_normalized(action_to_pwm(r.command, c.deadband), r.deadband)
        target = actual * c.max_wheel_speed * np.array([r.motor_gain_l, r.motor_gain_r])
        r.wheel_target += (target - r.wheel_target) * (-math.expm1(-dt / c.motor_tau))
        grip = c.tire_friction * c.mass * 981 * c.wheel_load_fraction / 2
        # DC torque-speed approximation at rated voltage. kgf.cm -> g.cm²/s².
        stall_force = c.motor_stall_torque_kgf_cm * 1000 * 981 / (c.wheel_diameter / 2)
        r.is_slipping = False
        for i, side in enumerate([1, -1]):
            point = (c.axle_x, side * c.track / 2)
            v = b.velocity_at_local_point(point).rotated(-b.angle)
            fx = stall_force * np.clip((r.wheel_target[i] - v.x) / c.max_wheel_speed, -1, 1)
            fy = -v.y * c.mass / (2 * max(0.04, c.motor_tau))
            mag = math.hypot(fx, fy)
            if mag > grip:
                fx *= grip / mag
                fy *= grip / mag
                r.is_slipping = True
            b.apply_force_at_local_point((fx, fy), point)
        for side in (-1, 1):
            support = (c.skid_x, side * c.skid_track / 2)
            v = b.velocity_at_local_point(support).rotated(-b.angle)
            speed = v.length
            if speed > 1e-5:
                drag = min(
                    c.skid_friction * c.mass * 981 * (1 - c.wheel_load_fraction) / 2,
                    c.mass * speed / dt * 0.05,
                )
                b.apply_force_at_local_point(tuple(-v * drag / speed), support)

    def _resolve_arena_boundaries(self):
        a = self.arena
        p = self.puck
        if self.goal_latched or abs(p.y) + a.puck_radius > a.goal_width / 2:
            return 0
        goal = (
            1 if p.x - a.puck_radius > a.half_length else (-1 if p.x + a.puck_radius < -a.half_length else 0)
        )
        if goal:
            self.goal_latched = True
            self.score_blue += int(goal == 1)
            self.score_red += int(goal == -1)
        return goal

    def step(self, blue_actions, red_actions):
        blue = np.asarray(blue_actions, dtype=float)
        red = np.asarray(red_actions, dtype=float)
        if red.size == 0:
            red = red.reshape(0, 2)
        if blue.shape != (self.robots_per_team, 2) or red.shape != (self.opponent_robots, 2):
            raise ValueError("Wheel action count differs from the robots actually present")
        acts = np.concatenate([blue, red])
        if not np.isfinite(acts).all():
            raise ValueError("Expected finite wheel actions for both teams")
        acts = np.clip(acts, -1, 1)
        sim = self.simulation
        jitter = self.rng.uniform(-sim.radio_jitter, sim.radio_jitter) if self.domain_randomization else 0.0
        delay = max(0.0, sim.radio_latency + jitter)
        self.ble_latency_ms = delay * 1000
        if not self.domain_randomization or self.rng.random() >= sim.packet_loss:
            self.ble_queue.append((self.episode_time + delay, self.sequence, acts.copy()))
        self.sequence += 1
        for r in self.all_robots:
            r.touched_puck = False
        goal = 0
        dt = 1 / sim.physics_hz
        for _ in range(sim.physics_hz // sim.control_hz):
            due = [p for p in self.ble_queue if p[0] <= self.episode_time + 1e-9]
            self.ble_queue = [p for p in self.ble_queue if p[0] > self.episode_time + 1e-9]
            if due:
                _, seq, commands = max(due, key=lambda p: p[1])
                if seq > self.delivered_sequence:
                    self.delivered_sequence = seq
                    for r, u in zip(self.all_robots, commands):
                        r.command = u
                        r.last_packet = self.episode_time
            for r in self.all_robots:
                if self.episode_time - r.last_packet > sim.watchdog_seconds:
                    r.command = np.zeros(2)
                self._motor_forces(r, dt)
            p = self.puck.body
            speed = p.velocity.length
            if speed > 0:
                p.velocity = p.velocity * max(0.0, 1 - self.friction_mu * 981 * dt / speed)
            self.space.step(dt)
            self.episode_time += dt
            self.match_time += dt
            if not goal:
                goal = self._resolve_arena_boundaries()
        for r in self.all_robots:
            local = r.body.world_to_local(self.puck.body.position)
            c = r.config
            r.has_puck_in_scoop = (
                c.length * 0.28 + self.arena.puck_radius < local.x < c.length / 2 + self.arena.puck_radius
                and abs(local.y) < c.width * 0.22
            )
        self.steps += 1
        self.puck.trail.append((self.puck.x, self.puck.y))
        self.puck.trail = self.puck.trail[-40:]
        return {
            "goal_scored": goal,
            "puck_speed": self.puck.body.velocity.length,
            "ble_latency_ms": self.ble_latency_ms,
            "blue_has_scoop": any(r.has_puck_in_scoop for r in self.blue_robots),
            "red_has_scoop": any(r.has_puck_in_scoop for r in self.red_robots),
        }

    def apply_puck_impulse(self, vx, vy):
        self.puck.body.apply_impulse_at_world_point(
            (vx * self.arena.puck_mass, vy * self.arena.puck_mass), self.puck.body.position
        )

    def place_robot(self, robot, x, y, theta=None):
        # Keep bench placement on the legal side of wall segments. Deep initial
        # segment penetration has ambiguous normals in a contact solver.
        angle = robot.theta if theta is None else theta
        vertices = [pymunk.Vec2d(*p).rotated(angle) for p in footprint_vertices(robot.config)]
        a = self.arena
        x = float(
            np.clip(
                x,
                -a.half_length - min(p.x for p in vertices) + 0.3,
                a.half_length - max(p.x for p in vertices) - 0.3,
            )
        )
        y = float(
            np.clip(
                y,
                -a.half_width - min(p.y for p in vertices) + 0.3,
                a.half_width - max(p.y for p in vertices) - 0.3,
            )
        )
        for _ in range(8):
            for vertex in vertices:
                vx, vy = vertex.x + x, vertex.y + y
                if abs(vx) > a.half_length - a.corner_radius and abs(vy) > a.half_width - a.corner_radius:
                    center = pymunk.Vec2d(
                        math.copysign(a.half_length - a.corner_radius, vx),
                        math.copysign(a.half_width - a.corner_radius, vy),
                    )
                    delta = pymunk.Vec2d(vx, vy) - center
                    if delta.length > a.corner_radius - 0.3:
                        correction = delta.normalized() * (delta.length - a.corner_radius + 0.3)
                        x -= correction.x
                        y -= correction.y
        robot.body.position = (x, y)
        robot.body.velocity = (0, 0)
        robot.body.angular_velocity = 0
        if theta is not None:
            robot.body.angle = theta
        robot.wheel_target[:] = 0
        robot.command[:] = 0
        self.space.reindex_shapes_for_body(robot.body)
        self.place_puck(self.puck.x, self.puck.y, self.puck.vx, self.puck.vy)

    def place_puck(self, x, y, vx=0.0, vy=0.0):
        requested = pymunk.Vec2d(x, y)
        radius = self.arena.puck_radius
        solids = [s for s in self.space.shapes if s.collision_type == 2]

        def clear(point):
            return all(s.point_query(point).distance >= radius - 1e-7 for s in solids)

        position = requested
        if not clear(requested):
            # Bench/reset operation only: deep initial penetration across a
            # concave body's internal seams is ambiguous for a contact solver.
            # Find free placement using EXTERNAL edges, never apply an in-game
            # ejection force or attach the puck to a robot.
            candidates = []
            for r in self.all_robots:
                polygons = [chassis_outline(robot=r.config), *wheel_footprints(r.config)]
                for polygon in polygons:
                    points = [r.body.local_to_world(p) for p in polygon]
                    for a, b in zip(points, points[1:] + points[:1]):
                        edge = b - a
                        t = float(np.clip((requested - a).dot(edge) / edge.length_squared, 0, 1))
                        normal = pymunk.Vec2d(edge.y, -edge.x).normalized()
                        candidates.append(a + edge * t + normal * (radius + 0.03))
                        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                            candidates.append(
                                a + pymunk.Vec2d(math.cos(angle), math.sin(angle)) * (radius + 0.03)
                            )
            candidates.sort(key=lambda p: (p - requested).length_squared)
            position = next((p for p in candidates if clear(p)), None)
            if position is None:
                raise ValueError("No free puck placement around the selected robot assembly")
        self.puck.body.position = position
        self.puck.body.velocity = (vx, vy)
        self.space.reindex_shapes_for_body(self.puck.body)
