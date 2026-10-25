"""Gymnasium environment for Mini-Sumo 500g RL with multi-chassis and hybrid league support."""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sumo.bots import BaseSumoBot, COMPETITIVE_BOT_REGISTRY, SearchAndDestroyBot
from sumo.constants import (
    CHASSIS_ARCHETYPES,
    DohyoConfig,
    SUMO_ACT_DIM,
    SUMO_OBS_DIM,
    SumoRewardConfig,
    SumoRobotConfig,
)
from sumo.dohyo import Dohyo
from sumo.physics import SumoPhysicsEngine
from sumo.sensors import SumoSensorPack


LAYOUT_CHOICES = ("standard", "back_to_back", "side_by_side", "angled", "parallel_north")


class SumoEnv(gym.Env):
    """Mini-Sumo 500g 1v1 Reinforcement Learning Environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        dohyo_config: Optional[DohyoConfig] = None,
        blue_config: Optional[SumoRobotConfig] = None,
        red_config: Optional[SumoRobotConfig] = None,
        reward_config: Optional[SumoRewardConfig] = None,
        opponent: Optional[BaseSumoBot] = None,
        randomize_matchups: bool = False,
        max_episode_seconds: float = 10.0,
    ):
        super().__init__()
        self.dohyo_cfg = dohyo_config or DohyoConfig()
        self.blue_cfg = blue_config or CHASSIS_ARCHETYPES["haryama"]
        self.red_cfg = red_config or CHASSIS_ARCHETYPES["ikazuchi_4wd"]
        self.reward_cfg = reward_config or SumoRewardConfig()
        self.randomize_matchups = randomize_matchups

        self.dohyo = Dohyo(self.dohyo_cfg)
        self.engine = SumoPhysicsEngine(self.dohyo_cfg, self.blue_cfg, self.red_cfg)

        self.sensors_blue = SumoSensorPack(self.blue_cfg, self.dohyo)
        self.sensors_red = SumoSensorPack(self.red_cfg, self.dohyo)

        self.opponent: BaseSumoBot = opponent or SearchAndDestroyBot(1)
        self.opponent_pool: List[BaseSumoBot] = [
            cls(1) for cls in COMPETITIVE_BOT_REGISTRY.values()
        ]
        self.chassis_list: List[SumoRobotConfig] = list(CHASSIS_ARCHETYPES.values())

        self.max_steps = max(1, round(max_episode_seconds * self.dohyo_cfg.control_hz))
        self.step_count = 0
        self.current_layout = "standard"

        self.observation_space = spaces.Box(
            low=-4.0, high=4.0, shape=(SUMO_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(SUMO_ACT_DIM,), dtype=np.float32
        )

        self.prev_action_blue: Tuple[float, float] = (0.0, 0.0)
        self.prev_action_red: Tuple[float, float] = (0.0, 0.0)
        self.prev_opp_radial_frac: float = 0.0
        self.prev_self_radial_frac: float = 0.0
        self.prev_inter_robot_dist: float = 0.0
        self.rng = np.random.default_rng()

    def set_opponent(self, opponent: BaseSumoBot, red_config: Optional[SumoRobotConfig] = None):
        """Set explicit opponent controller and optional chassis archetype."""
        self.opponent = opponent
        if red_config is not None:
            self.red_cfg = red_config
            self.engine.set_chassis(self.blue_cfg, self.red_cfg)
            self.sensors_red = SumoSensorPack(self.red_cfg, self.dohyo)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        # Select chassis and opponent if randomize_matchups is enabled
        if self.randomize_matchups and "layout" not in options:
            layout = str(self.rng.choice(LAYOUT_CHOICES))
            red_chassis = self.chassis_list[int(self.rng.integers(0, len(self.chassis_list)))]
            if "keep_opponent" not in options and self.opponent_pool:
                self.opponent = self.opponent_pool[int(self.rng.integers(0, len(self.opponent_pool)))]
            self.red_cfg = red_chassis
            self.engine.set_chassis(self.blue_cfg, self.red_cfg)
            self.sensors_red = SumoSensorPack(self.red_cfg, self.dohyo)
        else:
            layout = options.get("layout", "standard")
            if "blue_chassis" in options or "red_chassis" in options:
                b_cfg = options.get("blue_chassis", self.blue_cfg)
                r_cfg = options.get("red_chassis", self.red_cfg)
                if isinstance(b_cfg, str):
                    b_cfg = CHASSIS_ARCHETYPES[b_cfg]
                if isinstance(r_cfg, str):
                    r_cfg = CHASSIS_ARCHETYPES[r_cfg]
                self.blue_cfg = b_cfg
                self.red_cfg = r_cfg
                self.engine.set_chassis(self.blue_cfg, self.red_cfg)
                self.sensors_blue = SumoSensorPack(self.blue_cfg, self.dohyo)
                self.sensors_red = SumoSensorPack(self.red_cfg, self.dohyo)

        self.current_layout = layout
        self.step_count = 0
        self.engine.reset(layout=layout)

        if options.get("random_heading", self.randomize_matchups):
            self.engine.robot_blue.body.angle += float(self.rng.uniform(-0.22, 0.22))
            self.engine.robot_red.body.angle += float(self.rng.uniform(-0.22, 0.22))
            self.engine.space.reindex_shapes_for_body(self.engine.robot_blue.body)
            self.engine.space.reindex_shapes_for_body(self.engine.robot_red.body)

        self.opponent.reset()
        self.sensors_blue.reset(initial_search_dir=1.0)
        self.sensors_red.reset(initial_search_dir=-1.0)

        self.prev_action_blue = (0.0, 0.0)
        self.prev_action_red = (0.0, 0.0)

        b, r = self.engine.robot_blue, self.engine.robot_red
        self.prev_opp_radial_frac = self.dohyo.normalized_radial_pos(r.x, r.y)
        self.prev_self_radial_frac = self.dohyo.normalized_radial_pos(b.x, b.y)
        self.prev_inter_robot_dist = math.hypot(r.x - b.x, r.y - b.y)

        # Prime sensor rays for both robots on frame 0
        self.sensors_red.read_line_sensors(r)
        self.sensors_red.read_distance_sensors(r, b)
        obs = self.sensors_blue.get_observation(b, r, self.prev_action_blue)

        return obs, {
            "layout": self.current_layout,
            "blue_chassis": self.blue_cfg.name,
            "red_chassis": self.red_cfg.name,
            "opponent": self.opponent.name,
        }

    def step(
        self,
        action: np.ndarray,
        red_action_override: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        blue_act = (
            float(np.clip(action[0], -1.0, 1.0)),
            float(np.clip(action[1], -1.0, 1.0)),
        )

        b, r = self.engine.robot_blue, self.engine.robot_red
        if red_action_override is not None:
            red_act = (
                float(np.clip(red_action_override[0], -1.0, 1.0)),
                float(np.clip(red_action_override[1], -1.0, 1.0)),
            )
            # Still update Red's sensor rays for accurate visualization
            self.sensors_red.read_line_sensors(r)
            self.sensors_red.read_distance_sensors(r, b)
        else:
            red_act = self.opponent.get_action(r, b, self.sensors_red)

        sim_res = self.engine.step(blue_act, red_act)
        self.prev_action_blue = blue_act
        self.prev_action_red = red_act

        obs = self.sensors_blue.get_observation(b, r, self.prev_action_blue)

        # Compute competitive Sumo reward
        rc = self.reward_cfg
        reward = -rc.time_penalty

        curr_opp_radial = self.dohyo.normalized_radial_pos(r.x, r.y)
        curr_self_radial = self.dohyo.normalized_radial_pos(b.x, b.y)
        curr_dist = math.hypot(r.x - b.x, r.y - b.y)

        # 1. Push progress (rewarding moving the opponent toward the outer edge)
        delta_opp_radial = curr_opp_radial - self.prev_opp_radial_frac
        reward += rc.push_progress * delta_opp_radial

        # 2. Closing distance when not yet in contact
        if not b.in_contact:
            delta_close = (self.prev_inter_robot_dist - curr_dist) / self.dohyo.radius
            reward += rc.approach_progress * delta_close

        self.prev_opp_radial_frac = curr_opp_radial
        self.prev_self_radial_frac = curr_self_radial
        self.prev_inter_robot_dist = curr_dist

        # 3. Target lock alignment bonus (front 3 sensors L15, C0, R15 -> obs[6, 7, 8])
        if obs[7] < 0.85:  # C0 locked on
            reward += rc.alignment_bonus * (1.0 - obs[7])
        elif obs[6] < 0.85 or obs[8] < 0.85:
            reward += rc.alignment_bonus * 0.5

        # 4. Center dominance bonus
        if curr_self_radial < 0.55:
            reward += rc.center_control * (1.0 - curr_self_radial)

        # 5. Wedge advantage & active shoving during physical contact
        if b.in_contact:
            if b.wedge_advantage:
                reward += rc.wedge_advantage
            elif b.is_lifted:
                reward -= rc.wedge_advantage * 0.6
            if delta_opp_radial > 0.002:
                reward += rc.shove_impulse_bonus

        # 6. White line danger penalty when touching border without pushing opponent
        max_line = float(np.max(obs[0:4]))
        if max_line > 0.35 and not b.in_contact:
            reward -= rc.line_danger_penalty * max_line

        # 7. Terminal Ring-Out outcome
        terminated = bool(sim_res["terminated"])
        truncated = self.step_count >= self.max_steps and not terminated

        result_str = "in_progress"
        if terminated:
            if sim_res["winner"] == 0:
                reward += rc.ring_out_win
                result_str = "win"
            elif sim_res["winner"] == 1:
                reward += rc.ring_out_loss
                result_str = "loss"
            else:
                result_str = "draw"
        elif truncated:
            # Slight tie-break reward if opponent is closer to ring-out at timeout
            reward += 2.0 * (curr_opp_radial - curr_self_radial)
            result_str = "draw"

        info = {
            "winner": sim_res["winner"],
            "result": result_str,
            "time": sim_res["time"],
            "opp_radial_frac": curr_opp_radial,
            "self_radial_frac": curr_self_radial,
            "wedge_advantage": b.wedge_advantage,
            "in_contact": b.in_contact,
            "blue_chassis": self.blue_cfg.name,
            "red_chassis": self.red_cfg.name,
            "opponent": self.opponent.name,
        }
        return obs, float(reward), terminated, truncated, info
