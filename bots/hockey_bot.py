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
        speed = math.hypot(puck.vx, puck.vy)
        pred_time = min(0.35, speed / (C.PUCK_SLIDING_FRICTION * C.GRAVITY + 1e-5))
        pred_px = puck.x + puck.vx * pred_time * 0.7
        pred_py = puck.y + puck.vy * pred_time * 0.7

        # 1. Se o robô já está com o disco no scoop -> ARREMATE A GOL IMEDIATO!
        if r.has_puck_in_scoop:
            target_goal_y = max(-C.HALF_GOAL_WIDTH + 4.0, min(C.HALF_GOAL_WIDTH - 4.0, puck.y * 0.5))
            goal_ang = math.atan2(target_goal_y - r.y, attack_x - r.x)
            err_ang = wrap_angle(goal_ang - r.theta)
            fwd = 1.0 * self.difficulty
            turn = float(np.clip(err_ang * 2.2, -1.0, 1.0))
            return float(np.clip(fwd - turn, -1.0, 1.0)), float(np.clip(fwd + turn, -1.0, 1.0))

        # --- COMPORTAMENTO POR PAPEL TÁTICO NO HOCKEY 3v3 ---
        # PAPEL 2: GOLEIRO (Guarda-redes da pequena área)
        if r.role_id == C.ROLE_GOALIE:
            goal_center_x = defend_x + (14.0 if self.team == 0 else -14.0)
            # Se o disco está dentro da grande área e ameaçando entrar, sai para limpar
            dist_to_puck = math.hypot(puck.x - r.x, puck.y - r.y)
            is_danger = abs(puck.x - defend_x) < 32.0 and abs(puck.y) < (C.HALF_GOAL_WIDTH + 6.0)

            if is_danger and dist_to_puck < 25.0:
                clearing_ang = math.atan2(puck.y - r.y, puck.x - r.x)
                return self._drive_toward_angle(r, clearing_ang, speed=1.0 * self.difficulty)

            # Patrulha a linha de gol acompanhando o Y do puck
            target_y = float(np.clip(pred_py, -C.HALF_GOAL_WIDTH + 3.5, C.HALF_GOAL_WIDTH - 3.5))
            face_dir = 0.0 if self.team == 0 else math.pi
            # Move-se para o ponto de guarda
            dy = target_y - r.y
            dx = goal_center_x - r.x
            if math.hypot(dx, dy) > 3.0:
                ang_to_pos = math.atan2(dy, dx)
                return self._drive_toward_angle(r, ang_to_pos, speed=0.85 * self.difficulty)
            else:
                # Alinha chassi de frente para a quadra
                err = wrap_angle(face_dir - r.theta)
                turn = float(np.clip(err * 1.5, -0.6, 0.6))
                return -turn, turn

        # PAPEL 1: ALA / DEFENSOR (Contenção intermediária e apoio de rebote)
        elif r.role_id == C.ROLE_DEFENDER:
            dist_puck = math.hypot(pred_px - r.x, pred_py - r.y)
            is_puck_in_our_half = (pred_px < 0) if self.team == 0 else (pred_px > 0)

            # Se o disco está solto na nossa metade e perto dele, ataca o disco
            if is_puck_in_our_half and dist_puck < 35.0:
                desired_ang = math.atan2(puck.y - r.y, puck.x - r.x)
                return self._drive_toward_angle(r, desired_ang, speed=0.92 * self.difficulty)

            # Posicionamento tático de contenção (ala oposta ao atacante)
            mid_x = (defend_x * 0.40) + (pred_px * 0.25)
            # Fica no lado oposto ou acompanhando a linha do disco
            wing_y = 28.0 if pred_py < 0 else -28.0
            ang_pos = math.atan2(wing_y - r.y, mid_x - r.x)
            return self._drive_toward_angle(r, ang_pos, speed=0.80 * self.difficulty)

        # PAPEL 0: ATACANTE (Pivô ofensivo que vai na caça do disco)
        dist_puck = math.hypot(pred_px - r.x, pred_py - r.y)
        is_puck_behind = (r.x > pred_px) if self.team == 0 else (r.x < pred_px)

        # Se está atrás da bola, contorna
        if is_puck_behind and abs(pred_px - defend_x) < 45.0:
            guard_y = float(np.clip(puck.y * 0.8, -C.HALF_GOAL_WIDTH, C.HALF_GOAL_WIDTH))
            target_x = defend_x + (25.0 if self.team == 0 else -25.0)
            ang = math.atan2(guard_y - r.y, target_x - r.x)
            return self._drive_toward_angle(r, ang, speed=0.88 * self.difficulty)

        approach_dist = 14.0
        angle_to_goal = math.atan2(0.0 - pred_py, attack_x - pred_px)
        approach_x = pred_px - math.cos(angle_to_goal) * approach_dist
        approach_y = pred_py - math.sin(angle_to_goal) * approach_dist

        if dist_puck < 22.0 and not is_puck_behind:
            desired_ang = math.atan2(puck.y - r.y, puck.x - r.x)
            return self._drive_toward_angle(r, desired_ang, speed=1.0 * self.difficulty)

        desired_ang = math.atan2(approach_y - r.y, approach_x - r.x)
        return self._drive_toward_angle(r, desired_ang, speed=0.92 * self.difficulty)

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
