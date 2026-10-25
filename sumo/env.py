"""Gymnasium environment for Mini-Sumo 500g RL training."""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sumo.bots import BaseSumoBot, SearchAndDestroyBot
from sumo.constants import (
    DohyoConfig,
    SumoRewardConfig,
    SumoRobotConfig,
    SUMO_ACT_DIM,
    SUMO_OBS_DIM,
)
from sumo.dohyo import Dohyo
from sumo.physics import SumoPhysicsEngine
from sumo.sensors import SumoSensorPack


class SumoEnv(gym.Env):
    """Mini-Sumo 1v1 Reinforcement Learning Environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        dohyo_config: Optional[DohyoConfig] = None,
        robot_config: Optional[SumoRobotConfig] = None,
        reward_config: Optional[SumoRewardConfig] = None,
        opponent: Optional[BaseSumoBot] = None,
        max_episode_seconds: float = 12.0,
    ):
        super().__init__()
        self.dohyo_cfg = dohyo_config or DohyoConfig()
        self.robot_cfg = robot_config or SumoRobotConfig()
        self.reward_cfg = reward_config or SumoRewardConfig()
        self.dohyo = Dohyo(self.dohyo_cfg)
        self.engine = SumoPhysicsEngine(self.dohyo_cfg, self.robot_cfg)

        self.sensors_blue = SumoSensorPack(self.robot_cfg, self.dohyo)
        self.sensors_red = SumoSensorPack(self.robot_cfg, self.dohyo)

        self.opponent = opponent or SearchAndDestroyBot(1)
        self.max_steps = max(1, round(max_episode_seconds * 30.0))
        self.step_count = 0

        self.observation_space = spaces.Box(
            low=-4.0, high=4.0, shape=(SUMO_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(SUMO_ACT_DIM,), dtype=np.float32
        )

        self.prev_action_blue = (0.0, 0.0)
        self.prev_action_red = (0.0, 0.0)
        self.prev_opp_radial_frac = 0.0
        self.prev_self_radial_frac = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        layout = options.get("layout", "standard")
        random_heading = options.get("random_heading", False)

        self.step_count = 0
        self.engine.reset(layout=layout)

        if random_heading and seed is not None:
            rng = np.random.default_rng(seed)
            # Add small random heading jitter
            self.engine.robot_blue.body.angle += rng.uniform(-0.35, 0.35)
            self.engine.robot_red.body.angle += rng.uniform(-0.35, 0.35)

        self.prev_action_blue = (0.0, 0.0)
        self.prev_action_red = (0.0, 0.0)
        self.prev_opp_radial_frac = self.dohyo.normalized_radial_pos(
            self.engine.robot_red.x, self.engine.robot_red.y
        )
        self.prev_self_radial_frac = self.dohyo.normalized_radial_pos(
            self.engine.robot_blue.x, self.engine.robot_blue.y
        )

        obs = self.sensors_blue.get_observation(
            self.engine.robot_blue, self.engine.robot_red, self.prev_action_blue
        )
        return obs, {}

    def step(
        self,
        action: np.ndarray,
        red_action_override: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        blue_act = (float(action[0]), float(action[1]))

        # Determine opponent action
        if red_action_override is not None:
            red_act = red_action_override
        else:
            red_act = self.opponent.get_action(
                self.engine.robot_red, self.engine.robot_blue, self.sensors_red
            )

        sim_res = self.engine.step(blue_act, red_act)
        self.prev_action_blue = blue_act
        self.prev_action_red = red_act

        # Compute rewards
        rc = self.reward_cfg
        terminated = sim_res["terminated"]
        truncated = self.step_count >= self.max_steps and not terminated

        # 1. Terminal reward
        terminal_rew = 0.0
        if terminated:
            if sim_res["winner"] == 0:
                terminal_rew = rc.ring_out_win
            elif sim_res["winner"] == 1:
                terminal_rew = rc.ring_out_loss
            else:
                terminal_rew = -5.0  # Simultaneous ring-out penalty

        # 2. Push progress shaping (opp radial distance increases -> good)
        opp_radial = self.dohyo.normalized_radial_pos(
            self.engine.robot_red.x, self.engine.robot_red.y
        )
        push_progress = rc.push_progress * (opp_radial - self.prev_opp_radial_frac)
        self.prev_opp_radial_frac = opp_radial

        # 3. Center control shaping (keep self near center)
        self_radial = self.dohyo.normalized_radial_pos(
            self.engine.robot_blue.x, self.engine.robot_blue.y
        )
        center_control = rc.center_control * (self.prev_self_radial_frac - self_radial)
        self.prev_self_radial_frac = self_radial

        # 4. Line danger penalty
        line_readings = self.sensors_blue.read_line_sensors(self.engine.robot_blue)
        line_penalty = -rc.line_danger_penalty * float(np.mean(line_readings))

        # 5. Wedge advantage bonus
        wedge_bonus = rc.wedge_advantage if sim_res["blue_wedge"] else 0.0

        # 6. Energy & time penalty
        energy_penalty = -rc.energy_penalty * float(np.mean(np.square(action)))
        time_penalty = -rc.time_penalty

        reward = float(
            terminal_rew
            + push_progress
            + center_control
            + line_penalty
            + wedge_bonus
            + energy_penalty
            + time_penalty
        )

        obs = self.sensors_blue.get_observation(
            self.engine.robot_blue, self.engine.robot_red, self.prev_action_blue
        )

        info = {
            "winner": sim_res["winner"],
            "time": sim_res["time"],
            "blue_wedge": sim_res["blue_wedge"],
            "red_wedge": sim_res["red_wedge"],
            "blue_dist": sim_res["blue_dist"],
            "red_dist": sim_res["red_dist"],
            "terminal_reward": terminal_rew,
        }

        return obs, reward, terminated, truncated, info
