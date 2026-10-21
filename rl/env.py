"""
Ambiente Gymnasium Multi-Agente Oficial para Hockey de Robôs RoboCore (Hockey 3v3).

Cada partida oficial de Hockey de Robôs é disputada em 3v3 (6 robôs na arena de 180x120 cm):
 - Robô 0: Atacante / Centro (busca a posse, conduz e arremata)
 - Robô 1: Ala / Defensor (contenção intermediária, apoio e cobertura de rebotes)
 - Robô 2: Goleiro (proteção da pequena área e rebatida em frente à baliza)

Espaço de Observação Egocêntrico de 27 dimensões por robô:
  0..1:   Posição do disco (x, y)
  2..3:   Velocidade do disco (vx, vy)
  4..5:   Vetor relativo robô -> disco (dx, dy)
  6..7:   Vetor relativo disco -> gol adversário (+90, 0)
  8..9:   Posição do próprio robô (x, y)
  10..11: Orientação do chassi: cos(θ), sin(θ)
  12..13: Velocidade das rodas: v_left / v_max, v_right / v_max
  14:     Indicador de retenção na pá frontal: 1.0 se disco está no scoop, 0.0 caso contrário
  15:     Distância relativa até o próprio gol a defender (-90, 0)
  16:     Papel tático normalizado: role_id / 2.0 (0=Atacante, 0.5=Ala, 1.0=Goleiro)
  17..18: Vetor relativo até Companheiro 1 (dx, dy)
  19..20: Vetor relativo até Companheiro 2 (dx, dy)
  21..22: Vetor relativo até Oponente 0 (dx, dy)
  23..24: Vetor relativo até Oponente 1 (dx, dy)
  25..26: Vetor relativo até Oponente 2 (dx, dy)
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.constants import HockeyConstants as C
from core.hockey_physics import HockeyPhysicsEngine, HockeyRobot
from bots.hockey_bot import HeuristicHockeyBot
from vision.camera_tracker import SimulatedPartialVisionSensor


class RoboCoreHockeyEnv(gym.Env):
    """
    Ambiente Gymnasium Multi-Agente 3v3 para Hockey de Robôs RoboCore.
    Suporta controle simultâneo dos 3 robôs de cada equipe e observações egocêntricas de 27D.
    Integra percepção de visão computacional parcial com rastreamento das marcas físicas no chassi
    e filtro de Kalman cinemático em tempo real.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        robots_per_team: int = C.ROBOTS_PER_TEAM,
        max_steps: int = 900,  # 15 segundos acelerados por episódio
        opponent_difficulty: float = 0.88,
        domain_randomization: bool = True,
        use_partial_vision: bool = True,
    ):
        super().__init__()
        self.robots_per_team = robots_per_team
        self.engine = HockeyPhysicsEngine(
            robots_per_team=robots_per_team,
            domain_randomization=domain_randomization,
        )
        self.max_steps = max_steps
        self.opponent_bot = HeuristicHockeyBot(team=1, difficulty=opponent_difficulty)
        self.heuristic_blue_bot = HeuristicHockeyBot(team=0, difficulty=0.88)
        self.use_partial_vision = use_partial_vision
        self.vision_sensor = SimulatedPartialVisionSensor(enable_dropouts=domain_randomization)

        self.obs_dim = 27
        self.act_dim = 2  # (u_left, u_right) por robô em [-1.0, 1.0]^2

        # Espaço de observação [3, 27] para o time ou [27] para um único robô
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.robots_per_team, self.obs_dim), dtype=np.float32
        )
        # Espaço de ação [3, 2] para o trio de robôs
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.robots_per_team, self.act_dim), dtype=np.float32
        )

        self.prev_puck_goal_dist: float = 0.0
        self.prev_striker_puck_dist: float = 0.0

    def get_robot_obs(self, robot: HockeyRobot) -> np.ndarray:
        """
        Gera observação egocêntrica de 27 dimensões para um robô específico.
        Utiliza o sensor de visão computacional simulada (rastreamento de marcas no chassi,
        ruído de quantização, oclusões em choques mecânicos e filtro de Kalman) quando ativado.
        """
        sign = 1.0 if robot.team == 0 else -1.0
        v_scale = 120.0

        if self.use_partial_vision:
            idx = robot.role_id if robot.team == 0 else (self.robots_per_team + robot.role_id)
            r_state = self.vision_sensor.tracked_robots[idx]
            p_state = self.vision_sensor.tracked_puck

            p_x = p_state.x
            p_y = p_state.y
            p_vx = p_state.vx
            p_vy = p_state.vy

            r_x = r_state.x
            r_y = r_state.y
            r_th = r_state.theta
        else:
            p = self.engine.puck
            if self.engine.domain_randomization:
                p_noise_x = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
                p_noise_y = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
                r_noise_x = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
                r_noise_y = float(np.random.normal(0.0, C.VISION_POS_NOISE_STD))
                t_noise = float(np.random.normal(0.0, C.VISION_ANGLE_NOISE_STD))
            else:
                p_noise_x = p_noise_y = r_noise_x = r_noise_y = t_noise = 0.0

            p_x = p.x + p_noise_x
            p_y = p.y + p_noise_y
            p_vx = p.vx
            p_vy = p.vy
            r_x = robot.x + r_noise_x
            r_y = robot.y + r_noise_y
            r_th = robot.theta + t_noise

        # Disco normalizado
        px = sign * p_x / C.HALF_LENGTH
        py = sign * p_y / C.HALF_WIDTH
        pvx = sign * p_vx / v_scale
        pvy = sign * p_vy / v_scale

        # Robô normalizado
        rx = sign * r_x / C.HALF_LENGTH
        ry = sign * r_y / C.HALF_WIDTH
        theta = r_th if robot.team == 0 else (r_th + math.pi)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Vetores relativos disco e balizas
        rel_px = (sign * p_x - sign * r_x) / C.ARENA_LENGTH
        rel_py = (sign * p_y - sign * r_y) / C.ARENA_WIDTH

        goal_dx = (C.HALF_LENGTH - sign * p_x) / C.ARENA_LENGTH
        goal_dy = (0.0 - sign * p_y) / C.ARENA_WIDTH

        own_goal_d = math.hypot(sign * r_x - (-C.HALF_LENGTH), sign * r_y - 0.0) / C.ARENA_LENGTH
        scoop_flag = 1.0 if robot.has_puck_in_scoop else 0.0
        role_norm = float(robot.role_id) / 2.0

        # Vetores relativos aos companheiros de equipe e oponentes
        if self.use_partial_vision:
            same_team_indices = (
                range(0, self.robots_per_team)
                if robot.team == 0
                else range(self.robots_per_team, 2 * self.robots_per_team)
            )
            mate_vecs = []
            for m_idx in same_team_indices:
                if m_idx != idx:
                    m_state = self.vision_sensor.tracked_robots[m_idx]
                    mx = (sign * m_state.x - sign * r_x) / C.ARENA_LENGTH
                    my = (sign * m_state.y - sign * r_y) / C.ARENA_WIDTH
                    mate_vecs.extend([mx, my])
            while len(mate_vecs) < 4:
                mate_vecs.extend([0.0, 0.0])

            opp_indices = (
                range(self.robots_per_team, 2 * self.robots_per_team)
                if robot.team == 0
                else range(0, self.robots_per_team)
            )
            opp_vecs = []
            for o_idx in opp_indices:
                o_state = self.vision_sensor.tracked_robots[o_idx]
                ox = (sign * o_state.x - sign * r_x) / C.ARENA_LENGTH
                oy = (sign * o_state.y - sign * r_y) / C.ARENA_WIDTH
                opp_vecs.extend([ox, oy])
            while len(opp_vecs) < 6:
                opp_vecs.extend([0.0, 0.0])
        else:
            same_team = self.engine.blue_robots if robot.team == 0 else self.engine.red_robots
            mates = [m for m in same_team if m is not robot]
            mate_vecs = []
            for m in mates:
                mx = (sign * m.x - sign * r_x) / C.ARENA_LENGTH
                my = (sign * m.y - sign * r_y) / C.ARENA_WIDTH
                mate_vecs.extend([mx, my])
            while len(mate_vecs) < 4:
                mate_vecs.extend([0.0, 0.0])

            opp_team = self.engine.red_robots if robot.team == 0 else self.engine.blue_robots
            opp_vecs = []
            for o in opp_team:
                ox = (sign * o.x - sign * r_x) / C.ARENA_LENGTH
                oy = (sign * o.y - sign * r_y) / C.ARENA_WIDTH
                opp_vecs.extend([ox, oy])
            while len(opp_vecs) < 6:
                opp_vecs.extend([0.0, 0.0])

        obs = np.array(
            [
                px, py, pvx, pvy,
                rel_px, rel_py,
                goal_dx, goal_dy,
                rx, ry,
                cos_t, sin_t,
                robot.v_left / C.MAX_WHEEL_SPEED,
                robot.v_right / C.MAX_WHEEL_SPEED,
                scoop_flag,
                own_goal_d,
                role_norm,
                *mate_vecs[:4],
                *opp_vecs[:6],
            ],
            dtype=np.float32,
        )
        return obs

    def get_team_obs(self, team: int = 0) -> np.ndarray:
        """Retorna as observações de todos os robôs do time [robots_per_team, 27]."""
        robots = self.engine.blue_robots if team == 0 else self.engine.red_robots
        return np.array([self.get_robot_obs(r) for r in robots], dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.reset_positions(faceoff=True)
        if self.use_partial_vision:
            self.vision_sensor.reset()
            self.vision_sensor.update(self.engine)
        p = self.engine.puck
        self.prev_puck_goal_dist = math.hypot(C.HALF_LENGTH - p.x, 0.0 - p.y)
        r0 = self.engine.blue_robots[0]
        self.prev_striker_puck_dist = math.hypot(p.x - r0.x, p.y - r0.y)
        return self.get_team_obs(team=0), {}

    def step(
        self,
        action: Union[np.ndarray, List[Tuple[float, float]]],
        custom_red_actions: List[Tuple[float, float]] | None = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executa 1 tick na arena 3v3.
        Suporta ação [3, 2] para os 3 robôs, [6] achatado ou [2] (atacante com bot nos alas).
        """
        act = np.asarray(action, dtype=np.float32)
        n_rob = self.robots_per_team

        # Formatação das ações do Time Azul
        if act.ndim == 2 and act.shape[0] >= n_rob:
            blue_actions = [(float(act[i][0]), float(act[i][1])) for i in range(n_rob)]
        elif act.ndim == 1 and act.shape[0] == (n_rob * 2):
            blue_actions = [(float(act[i * 2]), float(act[i * 2 + 1])) for i in range(n_rob)]
        elif act.ndim == 1 and act.shape[0] == 2:
            # Controle manual/único no Atacante, outros robôs seguem heurística azul
            auto_acts = self.heuristic_blue_bot.get_actions(self.engine)
            blue_actions = [(float(act[0]), float(act[1]))] + auto_acts[1:n_rob]
        else:
            blue_actions = [(0.0, 0.0) for _ in range(n_rob)]

        # Ações do Time Vermelho (Oponente)
        if custom_red_actions is not None:
            red_actions = custom_red_actions
        else:
            red_actions = self.opponent_bot.get_actions(self.engine)

        info = self.engine.step(blue_actions, red_actions)

        # Atualização do sensor de visão computacional parcial
        if self.use_partial_vision:
            self.vision_sensor.update(self.engine)
            info["vision_confidence"] = [
                self.vision_sensor.tracked_robots[i].confidence
                for i in range(self.robots_per_team * 2)
            ]
            info["vision_tracked"] = [
                self.vision_sensor.tracked_robots[i].tracked
                for i in range(self.robots_per_team * 2)
            ]
            info["puck_vision_confidence"] = self.vision_sensor.tracked_puck.confidence
            info["puck_vision_tracked"] = self.vision_sensor.tracked_puck.tracked
        p = self.engine.puck
        blue_robots = self.engine.blue_robots
        r_striker = blue_robots[0] if len(blue_robots) > 0 else None
        r_defender = blue_robots[1] if len(blue_robots) > 1 else None
        r_goalie = blue_robots[2] if len(blue_robots) > 2 else None

        # --- RECOMPENSAS COLETIVAS E ESPECÍFICAS DE CADA PAPEL 3v3 ---
        # 1. Recompensa Compartilhada de Equipe: Gol e Avanço do Disco
        shared_team_reward = -0.004

        curr_pg_dist = math.hypot(C.HALF_LENGTH - p.x, 0.0 - p.y)
        delta_pg = (self.prev_puck_goal_dist - curr_pg_dist) / C.ARENA_LENGTH
        shared_team_reward += 7.5 * delta_pg
        self.prev_puck_goal_dist = curr_pg_dist

        if p.vx > 15.0:
            shared_team_reward += 0.025 * (p.vx / 100.0)

        # 2. Recompensas Individuais de cada papel
        indiv_rewards = [0.0] * n_rob

        # Atacante (Role 0): Condução, perseguição e finalização
        if r_striker:
            curr_rp = math.hypot(p.x - r_striker.x, p.y - r_striker.y)
            delta_rp = (self.prev_striker_puck_dist - curr_rp) / C.ARENA_LENGTH
            indiv_rewards[0] += 3.5 * delta_rp
            self.prev_striker_puck_dist = curr_rp

            if r_striker.has_puck_in_scoop:
                indiv_rewards[0] += 0.08
                v_lin = 0.5 * (r_striker.v_left + r_striker.v_right) / C.MAX_WHEEL_SPEED
                goal_ang = math.atan2(0.0 - r_striker.y, C.HALF_LENGTH - r_striker.x)
                align = math.cos(r_striker.theta - goal_ang)
                if v_lin > 0 and align > 0:
                    indiv_rewards[0] += 0.10 * v_lin * align

            if info.get("blue_scoop_ejected", False) and p.vx > 25.0:
                indiv_rewards[0] += 1.2
            if r_striker.touched_puck:
                indiv_rewards[0] += 0.20
            if r_striker.is_slipping:
                indiv_rewards[0] -= 0.01

        # Ala / Defensor (Role 1): Contenção intermediária e apoio de rebote
        if r_defender:
            is_midfield = -55.0 < r_defender.x < -10.0
            if is_midfield:
                indiv_rewards[1] += 0.015
            if r_defender.touched_puck:
                indiv_rewards[1] += 0.25
            if r_defender.has_puck_in_scoop:
                indiv_rewards[1] += 0.06
            if r_defender.is_slipping:
                indiv_rewards[1] -= 0.01

        # Goleiro (Role 2): Proteção da pequena área e interceptação de chutes
        if r_goalie:
            in_crease = (r_goalie.x < -68.0) and (abs(r_goalie.y) < C.HALF_GOAL_WIDTH + 4.0)
            if in_crease:
                indiv_rewards[2] += 0.02
                # Bônus por alinhar altura Y com o puck quando ele vem no nosso gol
                if p.vx < -10.0 and abs(r_goalie.y - p.y) < 7.0:
                    indiv_rewards[2] += 0.04
            if r_goalie.touched_puck:
                indiv_rewards[2] += 0.40  # Defesa de goleiro importante!
            if r_goalie.is_slipping:
                indiv_rewards[2] -= 0.01

        # Gol Marcado (+16.0) ou Sofrido (-14.0)
        terminated = False
        if info["goal_scored"] == 1:
            shared_team_reward += 16.0
            terminated = True
        elif info["goal_scored"] == -1:
            shared_team_reward -= 14.0
            terminated = True

        total_indiv = [shared_team_reward + r_ind for r_ind in indiv_rewards]
        mean_reward = float(np.mean(total_indiv))

        info["individual_rewards"] = total_indiv
        info["blue_has_scoop"] = any(r.has_puck_in_scoop for r in blue_robots)
        info["red_has_scoop"] = any(r.has_puck_in_scoop for r in self.engine.red_robots)

        truncated = self.engine.steps >= self.max_steps
        return self.get_team_obs(team=0), mean_reward, terminated, truncated, info
