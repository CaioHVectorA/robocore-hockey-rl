"""Gymnasium team-control environment. Training episodes and full matches differ."""

from __future__ import annotations
from dataclasses import asdict
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.constants import PROFILES
from core.hockey_physics import HockeyPhysicsEngine
from vision.camera_tracker import SimulatedPartialVisionSensor
from vision.state import EntityState, WorldState, OBS_DIM, build_team_observation, tracking_ready

SCENARIOS = ("faceoff", "shooting", "defending", "random")


class RoboCoreHockeyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        robots_per_team=3,
        max_steps=900,
        opponent_difficulty=0.85,
        domain_randomization=True,
        use_partial_vision=True,
        profile="micro-2024",
        arena=None,
        robot_config=None,
        simulation=None,
        scenario="faceoff",
        shaping_gamma=0.995,
    ):
        super().__init__()
        from bots.hockey_bot import HeuristicHockeyBot

        if scenario not in SCENARIOS:
            raise ValueError("Unknown training scenario")
        self.robots_per_team = robots_per_team
        self.max_steps = max_steps
        self.scenario = scenario
        self.shaping_gamma = shaping_gamma
        self.engine = HockeyPhysicsEngine(
            robots_per_team, domain_randomization, arena or PROFILES[profile], robot_config, simulation
        )
        self.arena = self.engine.arena
        self.simulation = self.engine.simulation
        self.opponent_bot = HeuristicHockeyBot(1, opponent_difficulty)
        self.use_partial_vision = use_partial_vision
        self.vision_sensor = SimulatedPartialVisionSensor(domain_randomization, self.simulation)
        self.obs_dim = OBS_DIM
        self.act_dim = 2
        self.observation_space = spaces.Box(-4.0, 4.0, (robots_per_team, OBS_DIM), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (robots_per_team, 2), np.float32)
        self.previous_actions = np.zeros((robots_per_team, 2), np.float32)
        self.previous_red_actions = np.zeros((robots_per_team, 2), np.float32)
        self.metrics = {}
        self._ended = False

    def _truth_state(self):
        e = self.engine
        t = e.episode_time
        robots = {
            r.team * 3 + r.role_id: EntityState(r.x, r.y, r.theta, r.vx, r.vy, r.omega, 1.0, t, t)
            for r in e.all_robots
        }
        p = e.puck
        return WorldState(t, robots, EntityState(p.x, p.y, 0.0, p.vx, p.vy, 0.0, 1.0, t, t))

    @property
    def state(self):
        return self.vision_sensor.state if self.use_partial_vision else self._truth_state()

    def get_team_obs(self, team=0):
        prev = self.previous_actions if team == 0 else self.previous_red_actions
        return build_team_observation(
            self.state, self.arena, self.engine.robot_config, team, prev, self.robots_per_team
        )

    def get_robot_obs(self, robot):
        return self.get_team_obs(robot.team)[robot.role_id]

    def _potential(self):
        p = self.engine.puck
        a = self.arena
        # Progress toward opponent goal, independent of contact or possession flags.
        return float(np.clip(1 - math.hypot(a.half_length - p.x, p.y) / a.length, 0.0, 1.0))

    def _approach(self):
        p = self.engine.puck
        return float(
            np.clip(
                1
                - min(math.hypot(r.x - p.x, r.y - p.y) for r in self.engine.blue_robots) / self.arena.length,
                0.0,
                1.0,
            )
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.engine.rng = np.random.default_rng(seed)
        options = options or {}
        preserve = options.get("preserve_match", False)
        if preserve:
            self.engine.reset_positions(reset_clock=False)
        else:
            self.engine.reset_match()
        scenario = options.get("scenario", self.scenario)
        if scenario not in SCENARIOS:
            raise ValueError("Unknown scenario")
        a = self.arena
        e = self.engine
        r = e.blue_robots[0]
        if scenario == "shooting":
            x = e.rng.uniform(a.length * 0.12, a.length * 0.30)
            y = e.rng.uniform(-a.goal_width * 0.4, a.goal_width * 0.4)
            e.place_robot(r, x - 12, y, 0.0)
            e.place_puck(x, y)
        elif scenario == "defending":
            x = -a.length * 0.2
            y = e.rng.uniform(-a.goal_width * 0.35, a.goal_width * 0.35)
            e.place_puck(x, y, -e.rng.uniform(60, 120), 0.0)
        elif scenario == "random":
            # Disjoint spawn bands prevent initial overlaps between chassis.
            for r in e.all_robots:
                sign = -1 if r.team == 0 else 1
                x = sign * (30 + 30 * r.role_id + e.rng.uniform(0, 12))
                y = e.rng.uniform(-a.half_width + 15, a.half_width - 15)
                e.place_robot(r, x, y, e.rng.uniform(-math.pi, math.pi))
            e.place_puck(
                e.rng.uniform(-a.length * 0.15, a.length * 0.15), e.rng.uniform(-a.width * 0.3, a.width * 0.3)
            )
        self.previous_actions[:] = 0
        self.previous_red_actions[:] = 0
        self.vision_sensor.reset()
        self.vision_sensor.update(e, prime=True)
        self.prev_potential = self._potential()
        self.prev_approach = self._approach()
        self._ended = False
        self.metrics = dict(shots_blue=0, shots_red=0, touches_blue=0, idle_steps=0, tracking_lost_steps=0)
        self._shot_side = 0
        return self.get_team_obs(), dict(scenario=scenario)

    def step(self, action, custom_red_actions=None):
        if self._ended:
            raise RuntimeError("Episode ended; reset before stepping")
        act = np.asarray(action, np.float32)
        if act.shape != (self.robots_per_team, 2) or not np.isfinite(act).all():
            raise ValueError("Invalid action shape/values")
        act = np.clip(act, -1, 1)
        if not tracking_ready(self.state, robots_per_team=self.robots_per_team):
            act = np.zeros_like(act)
        red = (
            custom_red_actions
            if custom_red_actions is not None
            else self.opponent_bot.get_actions(self.engine, self.state)
        )
        if not tracking_ready(self.state, team=1, robots_per_team=self.robots_per_team):
            red = np.zeros_like(act)
        info = self.engine.step(act, red)
        self.previous_actions = act.copy()
        self.previous_red_actions = np.asarray(red, np.float32).copy()
        self.vision_sensor.update(self.engine)
        phi = self._potential()
        approach = self._approach()
        # Potential shaping telescopes and offers no stationary touch bonus.
        terminated = info["goal_scored"] != 0
        shaping = 3.0 * (self.shaping_gamma * (0.0 if terminated else phi) - self.prev_potential)
        pursuit = 0.4 * (self.shaping_gamma * (0.0 if terminated else approach) - self.prev_approach)
        self.prev_potential = phi
        self.prev_approach = approach
        reward = shaping + pursuit - 0.006
        reward += 20.0 * info["goal_scored"]
        energy = 0.001 * np.mean(np.square(act), axis=1)
        indiv = reward - energy
        self.metrics["touches_blue"] += int(any(r.touched_puck for r in self.engine.blue_robots))
        self.metrics["idle_steps"] += int(max(math.hypot(r.vx, r.vy) for r in self.engine.blue_robots) < 2.0)
        self.metrics["tracking_lost_steps"] += int(
            not all(
                self.state.robots.get(i, EntityState()).valid(self.state.timestamp)
                for i in range(self.robots_per_team)
            )
        )
        p = self.engine.puck
        side = 1 if p.vx > 35 else (-1 if p.vx < -35 else 0)
        if side and side != self._shot_side and abs(p.y) < self.arena.goal_width / 2:
            self.metrics["shots_blue" if side == 1 else "shots_red"] += 1
        self._shot_side = side
        truncated = self.engine.steps >= self.max_steps
        self._ended = terminated or truncated
        info.update(self.metrics)
        info["individual_rewards"] = indiv.astype(np.float32)
        info["episode_seconds"] = self.engine.episode_time
        return self.get_team_obs(), float(indiv.mean()), terminated, truncated, info

    def configuration(self):
        return dict(
            arena=asdict(self.arena),
            robot=asdict(self.engine.robot_config),
            simulation=asdict(self.simulation),
            robots_per_team=self.robots_per_team,
            scenario=self.scenario,
            max_steps=self.max_steps,
        )
