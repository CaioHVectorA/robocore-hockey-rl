"""Timestamped state and the single actor observation contract (simulation/real)."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
import math
import numpy as np
from core.constants import ArenaConfig, RobotConfig

OBSERVATION_VERSION = "overhead-v2-63"
OBS_DIM = 63
MAX_TRACK_AGE = 0.25


def wrap_angle(x):
    return math.atan2(math.sin(x), math.cos(x))


@dataclass
class EntityState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    confidence: float = 0.0
    last_seen: float = -math.inf
    timestamp: float = 0.0

    def age(self, now):
        return max(0.0, now - self.last_seen)

    def valid(self, now):
        return self.confidence > 0 and self.age(now) <= MAX_TRACK_AGE

    def predict(self, now):
        dt = max(0.0, min(now - self.timestamp, MAX_TRACK_AGE))
        return replace(
            self,
            x=self.x + self.vx * dt,
            y=self.y + self.vy * dt,
            theta=wrap_angle(self.theta + self.omega * dt),
            timestamp=now,
            confidence=self.confidence if self.age(now) <= MAX_TRACK_AGE else 0.0,
        )


@dataclass
class WorldState:
    timestamp: float
    robots: dict[int, EntityState] = field(default_factory=dict)
    puck: EntityState = field(default_factory=EntityState)


def tracking_ready(state, team=0, max_age=0.15, robots_per_team=3):
    """Common simulation/hardware movement gate; never substitutes missing poses."""
    required = [state.puck] + [state.robots.get(team * 3 + i) for i in range(robots_per_team)]
    return all(
        e is not None and e.valid(state.timestamp) and e.age(state.timestamp) <= max_age for e in required
    )


def build_team_observation(
    state: WorldState, arena=None, robot=None, team=0, previous_actions=None, robots_per_team=3
):
    """63 features: 23 self/puck/goals/role/action/reliability + 5 x 8 entities.

    No wheel encoders or true possession. Team IDs: blue 0..2, red 3..5.
    Relative vectors use the robot heading; missing entities have confidence 0.
    """
    arena = arena or ArenaConfig()
    robot = robot or RobotConfig()
    now = state.timestamp
    result = np.zeros((robots_per_team, OBS_DIM), np.float32)
    previous_actions = (
        np.zeros((robots_per_team, 2)) if previous_actions is None else np.asarray(previous_actions)
    )
    if previous_actions.shape != (robots_per_team, 2):
        raise ValueError("Invalid previous action shape")
    for role in range(robots_per_team):
        idx = team * 3 + role
        s = state.robots.get(idx, EntityState()).predict(now)
        if not s.valid(now):
            result[role, 20] = result[role, 22] = 1.0
            continue
        c, sn = math.cos(s.theta), math.sin(s.theta)

        def rotate(dx, dy):
            return (dx * c + dy * sn, -dx * sn + dy * c)

        def rel(x, y):
            dx, dy = rotate(x - s.x, y - s.y)
            return [dx / arena.length, dy / arena.length]

        p = state.puck.predict(now)
        puck = rel(p.x, p.y) if p.valid(now) else [0.0, 0.0]
        pv = rotate(p.vx - s.vx, p.vy - s.vy) if p.valid(now) else (0.0, 0.0)
        sign = 1 if team == 0 else -1
        th = s.theta + (math.pi if team else 0.0)
        role_hot = [float(role == j) for j in range(3)]
        vals = [
            *puck,
            pv[0] / 120,
            pv[1] / 120,
            *rel(sign * arena.half_length, 0),
            *rel(-sign * arena.half_length, 0),
            sign * s.x / arena.half_length,
            sign * s.y / arena.half_width,
            math.cos(th),
            math.sin(th),
            (s.vx * c + s.vy * sn) / robot.max_wheel_speed,
            s.omega / 12.0,
            *role_hot,
            *previous_actions[role],
            s.confidence,
            min(1.0, s.age(now) / MAX_TRACK_AGE),
            p.confidence,
            min(1.0, p.age(now) / MAX_TRACK_AGE),
        ]
        others = [team * 3 + j for j in range(3) if j != role] + [(1 - team) * 3 + j for j in range(3)]
        for other in others:
            e = state.robots.get(other, EntityState()).predict(now)
            if not e.valid(now):
                vals.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                continue
            vx, vy = rotate(e.vx - s.vx, e.vy - s.vy)
            vals.extend(
                [
                    *rel(e.x, e.y),
                    vx / 120,
                    vy / 120,
                    math.cos(e.theta - s.theta),
                    math.sin(e.theta - s.theta),
                    e.confidence,
                    min(1.0, e.age(now) / MAX_TRACK_AGE),
                ]
            )
        result[role] = np.clip(vals, -4.0, 4.0)
    return result
