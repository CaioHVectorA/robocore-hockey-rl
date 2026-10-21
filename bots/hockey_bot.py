"""Deterministic tactical baseline. Can act on the SAME tracked state as PPO."""

import math
import numpy as np
from vision.state import EntityState, WorldState, wrap_angle


class HeuristicHockeyBot:
    def __init__(self, team=0, difficulty=0.85):
        self.team = team
        self.difficulty = difficulty

    def get_actions(self, engine, state=None):
        a = engine.arena
        now = engine.episode_time
        if state is None:
            state = WorldState(
                now,
                {
                    r.team * 3 + r.role_id: EntityState(r.x, r.y, r.theta, r.vx, r.vy, r.omega, 1.0, now, now)
                    for r in engine.all_robots
                },
                EntityState(
                    engine.puck.x, engine.puck.y, 0, engine.puck.vx, engine.puck.vy, 0, 1.0, now, now
                ),
            )
        p = state.puck
        sign = 1 if self.team == 0 else -1
        actions = []
        for role in range(engine.robots_per_team):
            r = state.robots.get(self.team * 3 + role, EntityState())
            if not r.valid(state.timestamp) or not p.valid(state.timestamp):
                actions.append((0.0, 0.0))
                continue
            gx = sign * a.half_length
            dist = math.hypot(p.x - r.x, p.y - r.y)
            if role == 2 and sign * p.x > -a.length * 0.30:
                tx = -gx + sign * 16
                ty = float(np.clip(p.y * 0.5, -a.goal_width * 0.35, a.goal_width * 0.35))
            elif role == 1 and (sign * p.x > 0 or dist > 55):
                tx = float(np.clip(p.x - sign * 40, -a.length * 0.25, a.length * 0.25))
                ty = float(np.clip(p.y + (22 if p.y < 0 else -22), -a.half_width + 18, a.half_width - 18))
            else:
                # Approach behind puck toward goal, then push through it.
                goalie = state.robots.get((1 - self.team) * 3 + 2)
                target_y = (-1 if goalie and goalie.y > 0 else 1) * a.goal_width * 0.30
                goal_angle = math.atan2(target_y - p.y, gx - p.x)
                behind = (p.x - math.cos(goal_angle) * 13, p.y - math.sin(goal_angle) * 13)
                alignment = abs(wrap_angle(math.atan2(p.y - r.y, p.x - r.x) - goal_angle))
                tx, ty = (
                    (p.x + math.cos(goal_angle) * 8, p.y + math.sin(goal_angle) * 8)
                    if dist < 24 and alignment < 0.7
                    else behind
                )
            tx = float(np.clip(tx, -a.half_length + 10, a.half_length - 10))
            ty = float(np.clip(ty, -a.half_width + 10, a.half_width - 10))
            dx, dy = tx - r.x, ty - r.y
            distance = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            err = wrap_angle(angle - r.theta)
            direction = 1
            if abs(err) > 2.2:
                direction = -1
                err = wrap_angle(err + math.pi)
            forward = direction * self.difficulty * min(1.0, distance / 14) * max(0.0, math.cos(err))
            turn = float(np.clip(err * 0.7, -0.8, 0.8))
            if distance < 2.0:
                forward = 0.0
                turn = float(np.clip(wrap_angle((0 if sign == 1 else math.pi) - r.theta) * 0.4, -0.4, 0.4))
            # Yield rather than blindly driving into a teammate.
            for idx, mate in state.robots.items():
                if idx // 3 == self.team and idx % 3 != role and math.hypot(mate.x - r.x, mate.y - r.y) < 21:
                    bearing = wrap_angle(math.atan2(mate.y - r.y, mate.x - r.x) - r.theta)
                    if abs(bearing) < 0.6:
                        forward *= 0.2
                        turn += -0.35 if bearing >= 0 else 0.35
            actions.append(tuple(np.clip([forward - turn, forward + turn], -1, 1)))
        return actions
