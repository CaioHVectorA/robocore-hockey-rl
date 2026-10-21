"""Rigid-body hockey: motor forces at the axle and geometric Pymunk contacts.

Units: cm, g, s. World +y points up. Possession is diagnostic, never a constraint.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import pymunk
from core.constants import ArenaConfig, RobotConfig, SimulationConfig
from core.geometry import convex_chassis_parts
from core.actuators import action_to_pwm, pwm_to_normalized


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
    ):
        if not 1 <= robots_per_team <= 3:
            raise ValueError("Use one to three robots per team")
        self.arena = arena or ArenaConfig()
        self.robot_config = robot_config or RobotConfig()
        self.simulation = simulation or SimulationConfig()
        self.robots_per_team = robots_per_team
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
            for role in range(self.robots_per_team):
                parts = convex_chassis_parts(role, cfg)
                areas = np.array([pymunk.area_for_poly(p) for p in parts])
                moment = sum(
                    pymunk.moment_for_poly(cfg.mass * a / areas.sum(), p) for a, p in zip(areas, parts)
                )
                body = pymunk.Body(cfg.mass, cfg.inertia or moment)
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
        a = self.arena
        hl, hw, rc = a.half_length, a.half_width, a.corner_radius

        def wall(p, q):
            s = pymunk.Segment(self.space.static_body, p, q, 0.15)
            s.elasticity = math.sqrt(self.simulation.restitution)
            s.friction = 0.2
            self.space.add(s)

        wall((-hl + rc, hw), (hl - rc, hw))
        wall((-hl + rc, -hw), (hl - rc, -hw))
        for sx in [-1, 1]:
            wall((sx * hl, -hw + rc), (sx * hl, -a.goal_width / 2))
            wall((sx * hl, a.goal_width / 2), (sx * hl, hw - rc))
            for sy in [-1, 1]:
                wall((sx * hl, sy * a.goal_width / 2), (sx * (hl + a.goal_depth), sy * a.goal_width / 2))
            wall((sx * (hl + a.goal_depth), -a.goal_width / 2), (sx * (hl + a.goal_depth), a.goal_width / 2))
        for cx, cy, start in [
            (hl - rc, hw - rc, 0),
            (-hl + rc, hw - rc, 90),
            (-hl + rc, -hw + rc, 180),
            (hl - rc, -hw + rc, 270),
        ]:
            angles = np.linspace(math.radians(start), math.radians(start + 90), 13)
            pts = [(cx + rc * math.cos(t), cy + rc * math.sin(t)) for t in angles]
            for p, q in zip(pts, pts[1:]):
                wall(p, q)

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
        grip = c.tire_friction * c.mass * 981 * 0.75 / 2
        r.is_slipping = False
        for i, side in enumerate([1, -1]):
            point = (c.axle_x, side * c.track / 2)
            v = b.velocity_at_local_point(point).rotated(-b.angle)
            fx = (r.wheel_target[i] - v.x) * c.mass / (2 * c.motor_tau)
            fy = -v.y * c.mass / (2 * max(0.04, c.motor_tau))
            mag = math.hypot(fx, fy)
            if mag > grip:
                fx *= grip / mag
                fy *= grip / mag
                r.is_slipping = True
            b.apply_force_at_local_point((fx, fy), point)
        v = b.velocity_at_local_point((c.skid_x, 0)).rotated(-b.angle)
        speed = v.length
        if speed > 1e-5:
            drag = min(c.skid_friction * c.mass * 981 * 0.25, c.mass * speed / dt * 0.1)
            b.apply_force_at_local_point(tuple(-v * drag / speed), (c.skid_x, 0))

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
        acts = np.asarray(list(blue_actions) + list(red_actions), dtype=float)
        if acts.shape != (2 * self.robots_per_team, 2) or not np.isfinite(acts).all():
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
                c.length / 2 - 2 < local.x < c.length / 2 + self.arena.puck_radius + 1 and abs(local.y) < 3
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
        from core.geometry import chassis_outline

        angle = robot.theta if theta is None else theta
        vertices = [pymunk.Vec2d(*p).rotated(angle) for p in chassis_outline(robot.role_id, robot.config)]
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

    def place_puck(self, x, y, vx=0.0, vy=0.0):
        self.puck.body.position = (x, y)
        self.puck.body.velocity = (vx, vy)
        self.space.reindex_shapes_for_body(self.puck.body)
