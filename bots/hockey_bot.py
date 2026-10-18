"""
Bot Heurístico Especialista para Hockey de Robôs (RoboCore HockeyBot).

Comportamento tático:
1. Predição balística de deslizamento com desaceleração de Coulomb (a = -mu_k * g)
2. Interceptação com a pá frontal côncava (scoop) para travar a posse
3. Condução direcionada à baliza adversária e disparo com giro rápido
4. Posicionamento de cobertura e bloqueio defensivo quando o oponente tem a posse
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np

from core.constants import HockeyConstants as C
from core.hockey_physics import HockeyPhysicsEngine, HockeyRobot, HockeyPuck


def wrap_angle(ang: float) -> float:
    return math.atan2(math.sin(ang), math.cos(ang))


class HeuristicHockeyBot:
    """Controlador analítico para equipe de Hockey (time Azul ou Vermelho)."""

    def __init__(self, team: int = 1, difficulty: float = 0.92):
        self.team = team  # 0 = Azul (ataca +90), 1 = Vermelho (ataca -90)
        self.difficulty = difficulty

    def get_actions(self, engine: HockeyPhysicsEngine) -> List[Tuple[float, float]]:
        robots = engine.blue_robots if self.team == 0 else engine.red_robots
        puck = engine.puck
        attack_goal_x = C.HALF_LENGTH if self.team == 0 else -C.HALF_LENGTH
        defend_goal_x = -C.HALF_LENGTH if self.team == 0 else C.HALF_LENGTH

        actions = []
        for r in robots:
            act = self._compute_robot_action(r, puck, attack_goal_x, defend_goal_x)
            actions.append(act)
        return actions

    def _compute_robot_action(
        self,
        r: HockeyRobot,
        puck: HockeyPuck,
        attack_x: float,
        defend_x: float,
    ) -> Tuple[float, float]:
        # 1. Predição de onde o disco vai parar ou desacelerar devido ao atrito de Coulomb
        speed = math.hypot(puck.vx, puck.vy)
        pred_time = min(0.35, speed / (C.PUCK_SLIDING_FRICTION * C.GRAVITY + 1e-5))
        pred_px = puck.x + puck.vx * pred_time * 0.7
        pred_py = puck.y + puck.vy * pred_time * 0.7

        dist_puck = math.hypot(pred_px - r.x, pred_py - r.y)

        # 2. Se já estiver com o disco engatado na pá frontal (scoop) -> ARREMATE A GOL!
        if r.has_puck_in_scoop:
            target_goal_y = max(-C.HALF_GOAL_WIDTH + 4.0, min(C.HALF_GOAL_WIDTH - 4.0, puck.y * 0.5))
            goal_ang = math.atan2(target_goal_y - r.y, attack_x - r.x)
            err_ang = wrap_angle(goal_ang - r.theta)

            # Se bem alinhado ao gol, aceleração total! Se desalinhado, curva agressiva
            fwd = 1.0 * self.difficulty
            turn = float(np.clip(err_ang * 2.2, -1.0, 1.0))
            ul = float(np.clip(fwd - turn, -1.0, 1.0))
            ur = float(np.clip(fwd + turn, -1.0, 1.0))
            return ul, ur

        # 3. Se a bola está perigosa na zona de defesa, postura de cobertura
        is_puck_behind = (r.x > pred_px) if self.team == 0 else (r.x < pred_px)
        if is_puck_behind and abs(pred_px - defend_x) < 45.0:
            # Precisa contornar para ficar entre a bola e o próprio gol
            guard_y = float(np.clip(puck.y * 0.8, -C.HALF_GOAL_WIDTH, C.HALF_GOAL_WIDTH))
            target_x = defend_x + (25.0 if self.team == 0 else -25.0)
            ang = math.atan2(guard_y - r.y, target_x - r.x)
            return self._drive_toward_angle(r, ang, speed=0.85 * self.difficulty)

        # 4. Chegar por trás do disco apontando para o gol (ponto de aproximação)
        approach_dist = 14.0
        angle_to_goal = math.atan2(0.0 - pred_py, attack_x - pred_px)
        approach_x = pred_px - math.cos(angle_to_goal) * approach_dist
        approach_y = pred_py - math.sin(angle_to_goal) * approach_dist

        # Se já está atrás da bola, vai direto nela com a pá frontal
        if dist_puck < 22.0 and not is_puck_behind:
            desired_ang = math.atan2(puck.y - r.y, puck.x - r.x)
            return self._drive_toward_angle(r, desired_ang, speed=1.0 * self.difficulty)

        desired_ang = math.atan2(approach_y - r.y, approach_x - r.x)
        return self._drive_toward_angle(r, desired_ang, speed=0.90 * self.difficulty)

    def _drive_toward_angle(self, r: HockeyRobot, target_ang: float, speed: float) -> Tuple[float, float]:
        err = wrap_angle(target_ang - r.theta)

        # Reversão inteligente de marcha à ré se o alvo estiver a mais de 105 graus atrás
        direction = 1.0
        if abs(err) > (math.pi * 0.60):
            err = wrap_angle(err + math.pi)
            direction = -1.0

        v_lin = direction * speed * max(0.2, math.cos(err))
        omega = float(np.clip(err * 1.85, -1.0, 1.0))

        ul = float(np.clip(v_lin - omega, -1.0, 1.0))
        ur = float(np.clip(v_lin + omega, -1.0, 1.0))
        return ul, ur
