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
from rl.tasks import RewardConfig

SCENARIOS = (
    "faceoff",
    "shooting",
    "defending",
    "random",
    "solo",
    "solo_finish",
    "solo_carry",
    "solo_search",
    "duel",
)


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
        opponent_robots=None,
        objective="score",
        reward_config=None,
        spawn_distance=60.0,
        random_heading=True,
    ):
        super().__init__()
        from bots.hockey_bot import HeuristicHockeyBot

        if scenario not in SCENARIOS:
            raise ValueError("Unknown training scenario")
        self.robots_per_team = robots_per_team
        self.max_steps = max_steps
        self.scenario = scenario
        self.shaping_gamma = shaping_gamma
        if objective not in ("score", "reach"):
            raise ValueError("Unknown task objective")
        if max_steps <= 0 or not math.isfinite(spawn_distance) or not 15 <= spawn_distance <= 150:
            raise ValueError("Invalid episode length/spawn distance")
        self.objective = objective
        self.reward_config = reward_config or RewardConfig()
        self.spawn_distance = spawn_distance
        self.random_heading = random_heading
        self.engine = HockeyPhysicsEngine(
            robots_per_team,
            domain_randomization,
            arena or PROFILES[profile],
            robot_config,
            simulation,
            opponent_robots=opponent_robots,
        )
        self.opponent_robots = self.engine.opponent_robots
        if scenario == "duel" and (robots_per_team != 1 or self.opponent_robots != 1):
            raise ValueError("Duel requires one robot on each team")
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
        self.previous_red_actions = np.zeros((self.opponent_robots, 2), np.float32)
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
            self.state,
            self.arena,
            self.engine.robot_config,
            team,
            prev,
            self.robots_per_team if team == 0 else self.opponent_robots,
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
        self.initialization = scenario
        if scenario == "duel":
            red = e.red_robots[0]
            layout = str(e.rng.choice(["faceoff", "random", "attack", "defend"], p=[0.35, 0.35, 0.15, 0.15]))
            self.initialization = layout
            if layout == "random":
                for bot, side in ((r, -1), (red, 1)):
                    e.place_robot(
                        bot,
                        side * e.rng.uniform(25, a.length * 0.35),
                        e.rng.uniform(-a.width * 0.3, a.width * 0.3),
                        e.rng.uniform(-math.pi, math.pi),
                    )
                e.place_puck(
                    e.rng.uniform(-a.length * 0.15, a.length * 0.15),
                    e.rng.uniform(-a.width * 0.25, a.width * 0.25),
                )
            elif layout in ("attack", "defend"):
                side = 1 if layout == "attack" else -1
                attacker, defender = (r, red) if side == 1 else (red, r)
                px, py = (
                    side * (a.half_length - e.rng.uniform(20, 40)),
                    e.rng.uniform(-a.goal_width * 0.25, a.goal_width * 0.25),
                )
                bearing = math.atan2(-py, side * (a.half_length + a.goal_depth / 2) - px)
                distance = e.rng.uniform(13, 20)
                e.place_robot(
                    attacker,
                    px - math.cos(bearing) * distance,
                    py - math.sin(bearing) * distance,
                    bearing + e.rng.uniform(-0.25, 0.25),
                )
                e.place_robot(
                    defender,
                    side * (a.half_length - 24),
                    (1 if py < 0 else -1) * (a.goal_width / 2 + 14),
                    math.pi if side == 1 else 0,
                )
                e.place_puck(px, py)
        elif scenario in ("solo_finish", "solo_carry"):
            if scenario == "solo_finish":
                px, py = (
                    a.half_length - e.rng.uniform(8, 30),
                    e.rng.uniform(-a.goal_width * 0.22, a.goal_width * 0.22),
                )
                distance, heading_noise = e.rng.uniform(12, 18), 0.18
            else:
                px, py = (
                    e.rng.uniform(-a.length * 0.05, a.half_length - 35),
                    e.rng.uniform(-a.width * 0.25, a.width * 0.25),
                )
                distance, heading_noise = e.rng.uniform(12, 25), 0.4
            bearing = math.atan2(-py, a.half_length + a.goal_depth / 2 - px)
            e.place_robot(
                r,
                px - math.cos(bearing) * distance,
                py - math.sin(bearing) * distance,
                bearing + (e.rng.uniform(-heading_noise, heading_noise) if self.random_heading else 0),
            )
            e.place_puck(px, py)
        elif scenario in ("solo", "solo_search"):
            broad = scenario == "solo_search"
            px = e.rng.uniform(-a.length * (0.25 if broad else 0.12), a.length * (0.3 if broad else 0.22))
            py = e.rng.uniform(-a.width * (0.35 if broad else 0.25), a.width * (0.35 if broad else 0.25))
            bearing = e.rng.uniform(-math.pi, math.pi) if broad else e.rng.uniform(-1.0, 1.0)
            distance = e.rng.uniform(self.spawn_distance * 0.5, self.spawn_distance)
            heading = e.rng.uniform(-math.pi, math.pi) if self.random_heading else bearing
            e.place_robot(r, px - math.cos(bearing) * distance, py - math.sin(bearing) * distance, heading)
            e.place_puck(px, py)
        elif scenario == "shooting":
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
        self.last_reward_components = dict.fromkeys(
            ("goal", "progress", "approach", "time_penalty", "energy_penalty", "reach_bonus"), 0.0
        )
        self.episode_reward_components = dict(self.last_reward_components)
        self._shot_side = 0
        return self.get_team_obs(), dict(scenario=scenario, initialization=self.initialization)

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
        if not tracking_ready(self.state, team=1, robots_per_team=self.opponent_robots):
            red = np.zeros((self.opponent_robots, 2), np.float32)
        info = self.engine.step(act, red)
        self.previous_actions = act.copy()
        self.previous_red_actions = np.asarray(red, np.float32).reshape(self.opponent_robots, 2).copy()
        self.vision_sensor.update(self.engine)
        phi = self._potential()
        approach = self._approach()
        # Potential shaping telescopes and offers no stationary touch bonus.
        rc = self.reward_config
        distance = min(
            math.hypot(r.x - self.engine.puck.x, r.y - self.engine.puck.y) for r in self.engine.blue_robots
        )
        reached = self.objective == "reach" and distance <= rc.reach_distance and info["goal_scored"] >= 0
        terminated = info["goal_scored"] != 0 or reached
        shaping = (rc.progress if self.objective == "score" else 0.0) * (
            self.shaping_gamma * (0.0 if terminated else phi) - self.prev_potential
        )
        pursuit = rc.approach * (self.shaping_gamma * (0.0 if terminated else approach) - self.prev_approach)
        self.prev_potential = phi
        self.prev_approach = approach
        energy = rc.energy_penalty * np.mean(np.square(act), axis=1)
        # Expose signed, weighted terms. Their sum is exactly the team mean reward.
        components = dict(
            goal=rc.goal * info["goal_scored"],
            progress=shaping,
            approach=pursuit,
            time_penalty=-rc.time_penalty,
            energy_penalty=-float(energy.mean()),
            reach_bonus=rc.reach_bonus * reached,
        )
        reward = sum(value for key, value in components.items() if key != "energy_penalty")
        indiv = reward - energy
        self.last_reward_components = components
        for key, value in components.items():
            self.episode_reward_components[key] += value
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
        info["reward_components"] = components
        info["episode_reward_components"] = dict(self.episode_reward_components)
        info["episode_seconds"] = self.engine.episode_time
        info["distance_to_puck"] = distance
        info["is_success"] = bool(reached or info["goal_scored"] == 1)
        info["result"] = (
            "success"
            if info["is_success"]
            else "opponent_goal"
            if info["goal_scored"] == -1
            else "timeout"
            if truncated
            else "running"
        )
        return self.get_team_obs(), float(indiv.mean()), terminated, truncated, info

    def configuration(self):
        return dict(
            arena=asdict(self.arena),
            robot=asdict(self.engine.robot_config),
            simulation=asdict(self.simulation),
            robots_per_team=self.robots_per_team,
            opponent_robots=self.opponent_robots,
            objective=self.objective,
            reward=asdict(self.reward_config),
            spawn_distance=self.spawn_distance,
            random_heading=self.random_heading,
            scenario=self.scenario,
            max_steps=self.max_steps,
        )
