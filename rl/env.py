"""
Ambiente Gymnasium para o Hockey de Robôs RoboCore (Classe 1lb / HockeyBot / Falcon).

Espaço de Observação Egocêntrico de 18 dimensões contínuas em [-1.0, 1.0]:
 0..1:   Posição do disco (x, y) normalizada pela arena
 2..3:   Velocidade do disco (vx, vy) normalizada
 4..5:   Vetor relativo robô -> disco (dx, dy)
 6..7:   Vetor relativo disco -> gol adversário (+90, 0)
 8..9:   Posição do próprio robô (x, y)
10..11:  Orientação do chassi: cos(θ), sin(θ)
12..13:  Velocidade das rodas: v_left / v_max, v_right / v_max
14:      Indicador de retenção na pá frontal: 1.0 se disco está no scoop, 0.0 caso contrário
15:      Distância relativa até o próprio gol a defender (-90, 0)
16..17:  Vetor relativo até o oponente mais próximo (dx, dy)
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.constants import HockeyConstants as C
from core.hockey_physics import HockeyPhysicsEngine, HockeyRobot
from bots.hockey_bot import HeuristicHockeyBot


class RoboCoreHockeyEnv(gym.Env):
    """Ambiente Gymnasium para Hockey de Robôs com recompensa de retenção e condução na pá."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        robots_per_team: int = 1,
        max_steps: int = 900,  # 15 segundos por episódio acelerado de treino
        opponent_difficulty: float = 0.88,
        domain_randomization: bool = True,
    ):
        super().__init__()
        self.engine = HockeyPhysicsEngine(
            robots_per_team=robots_per_team,
            domain_randomization=domain_randomization,
        )
        self.max_steps = max_steps
        self.opponent_bot = HeuristicHockeyBot(team=1, difficulty=opponent_difficulty)

        self.obs_dim = 18
        self.act_dim = 2  # (u_left, u_right) em [-1.0, 1.0]^2

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32
        )

        self.prev_puck_goal_dist: float = 0.0
        self.prev_robot_puck_dist: float = 0.0

    def get_robot_obs(self, robot: HockeyRobot) -> np.ndarray:
        """Gera observação de 18 dimensões com orientação egocêntrica invariante de time."""
        sign = 1.0 if robot.team == 0 else -1.0
        p = self.engine.puck
        v_scale = 120.0

        # Disco
        px = sign * p.x / C.HALF_LENGTH
        py = sign * p.y / C.HALF_WIDTH
        pvx = sign * p.vx / v_scale
        pvy = sign * p.vy / v_scale

        # Robô
        rx = sign * robot.x / C.HALF_LENGTH
        ry = sign * robot.y / C.HALF_WIDTH
        theta = robot.theta if robot.team == 0 else (robot.theta + math.pi)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Vetores relativos
        rel_px = (sign * p.x - sign * robot.x) / C.ARENA_LENGTH
        rel_py = (sign * p.y - sign * robot.y) / C.ARENA_WIDTH

        goal_dx = (C.HALF_LENGTH - sign * p.x) / C.ARENA_LENGTH
        goal_dy = (0.0 - sign * p.y) / C.ARENA_WIDTH

        own_goal_d = math.hypot(sign * robot.x - (-C.HALF_LENGTH), sign * robot.y - 0.0) / C.ARENA_LENGTH
        scoop_flag = 1.0 if robot.has_puck_in_scoop else 0.0

        opps = self.engine.red_robots if robot.team == 0 else self.engine.blue_robots
        opp = opps[0]
        opp_dx = (sign * opp.x - sign * robot.x) / C.ARENA_LENGTH
        opp_dy = (sign * opp.y - sign * robot.y) / C.ARENA_WIDTH

        obs = np.array(
            [
                px,
                py,
                pvx,
                pvy,
                rel_px,
                rel_py,
                goal_dx,
                goal_dy,
                rx,
                ry,
                cos_t,
                sin_t,
                robot.v_left / C.MAX_WHEEL_SPEED,
                robot.v_right / C.MAX_WHEEL_SPEED,
                scoop_flag,
                own_goal_d,
                opp_dx,
                opp_dy,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.reset_positions(faceoff=True)
        r0 = self.engine.blue_robots[0]
        p = self.engine.puck
        self.prev_puck_goal_dist = math.hypot(C.HALF_LENGTH - p.x, 0.0 - p.y)
        self.prev_robot_puck_dist = math.hypot(p.x - r0.x, p.y - r0.y)
        return self.get_robot_obs(r0), {}

    def step(
        self,
        action: np.ndarray,
        custom_red_actions: List[Tuple[float, float]] | None = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = np.asarray(action, dtype=np.float32)
        blue_actions = [(float(act[0]), float(act[1]))]

        if custom_red_actions is not None:
            red_actions = custom_red_actions
        else:
            red_actions = self.opponent_bot.get_actions(self.engine)

        info = self.engine.step(blue_actions, red_actions)
        r0 = self.engine.blue_robots[0]
        p = self.engine.puck

        # --- MODELAGEM DE RECOMPENSA FÍSICA PARA HOCKEY DE ROBÔS ---
        reward = -0.005  # Leve incentivo para decidir a jogada rápido

        # 1. Aproximação do robô até o disco
        curr_rp_dist = math.hypot(p.x - r0.x, p.y - r0.y)
        delta_rp = (self.prev_robot_puck_dist - curr_rp_dist) / C.ARENA_LENGTH
        reward += 3.8 * delta_rp
        self.prev_robot_puck_dist = curr_rp_dist

        # 2. Bônus por reter o disco na pá frontal (scoop) e conduzi-lo em direção ao gol
        if r0.has_puck_in_scoop:
            reward += 0.06
            v_lin = 0.5 * (r0.v_left + r0.v_right) / C.MAX_WHEEL_SPEED
            goal_angle = math.atan2(0.0 - r0.y, C.HALF_LENGTH - r0.x)
            align = math.cos(r0.theta - goal_angle)
            if v_lin > 0 and align > 0:
                reward += 0.08 * v_lin * align

        if r0.touched_puck:
            reward += 0.25

        # 3. Progresso do disco em direção ao gol adversário (+90, 0)
        curr_pg_dist = math.hypot(C.HALF_LENGTH - p.x, 0.0 - p.y)
        delta_pg = (self.prev_puck_goal_dist - curr_pg_dist) / C.ARENA_LENGTH
        reward += 8.5 * delta_pg
        self.prev_puck_goal_dist = curr_pg_dist

        # 4. Velocidade positiva do disco rumo ao gol
        if p.vx > 15.0:
            reward += 0.03 * (p.vx / 100.0)

        # 5. Gol Marcado (+16.0) / Gol Sofrido (-14.0)
        terminated = False
        if info["goal_scored"] == 1:
            reward += 16.0
            terminated = True
        elif info["goal_scored"] == -1:
            reward -= 14.0
            terminated = True

        truncated = self.engine.steps >= self.max_steps
        return self.get_robot_obs(r0), float(reward), terminated, truncated, info
