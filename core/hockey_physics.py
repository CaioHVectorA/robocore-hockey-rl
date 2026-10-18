"""
Motor físico 2D de alta fidelidade para Hockey de Robôs (Classe RoboCore HockeyBot / 1lb).

Modelagem física:
1. Dinâmica de deslizamento do disco (Puck): Atrito seco de Coulomb (desaceleração linear a = -mu_k * g)
   com travamento por atrito estático quando v < v_stop (o puck para completamente, não rola!).
2. Chassi fiel do RoboCore HockeyBot (17 cm x 16 cm, 454 g):
   - Cinemática diferencial não-holonômica (v_L, v_R)
   - Inércia e atraso de resposta da placa Vespa / motores DC (tau = 35 ms)
   - Pá frontal côncava (scoop) que retém o disco durante condução e o libera em curvas bruscas
3. Ringue oficial com 4 cantos arredondados (R = 20 cm) para escorregamento contínuo do disco
4. Gols de 35 cm de largura embutidos nas paredes de fundo
5. Domain Randomization para transferência Sim-to-Real via Bluetooth BLE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np

from .constants import HockeyConstants as C


@dataclass
class HockeyRobot:
    team: int            # 0 = Azul (Esquerda, defende -90, ataca +90), 1 = Vermelho (Direita, ataca -90)
    role_id: int         # 0 = Principal (Atacante/Livre), 1 = Defensor (se time de múltiplos)
    x: float = 0.0       # Coordenada X (cm)
    y: float = 0.0       # Coordenada Y (cm)
    theta: float = 0.0   # Orientação do chassi (radianos)
    vx: float = 0.0      # Velocidade linear X (cm/s)
    vy: float = 0.0      # Velocidade linear Y (cm/s)
    omega: float = 0.0   # Velocidade angular (rad/s)
    v_left: float = 0.0  # Velocidade real roda esquerda (cm/s)
    v_right: float = 0.0 # Velocidade real roda direita (cm/s)
    cmd_left: float = 0.0
    cmd_right: float = 0.0
    touched_puck: bool = False
    has_puck_in_scoop: bool = False


@dataclass
class HockeyPuck:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    last_touch_team: int = -1
    trail: List[Tuple[float, float]] = field(default_factory=list)


class HockeyPhysicsEngine:
    """Motor físico oficial de Hockey de Robôs RoboCore a 60 Hz."""

    def __init__(self, robots_per_team: int = 1, domain_randomization: bool = True):
        self.robots_per_team = robots_per_team
        self.domain_randomization = domain_randomization

        self.blue_robots: List[HockeyRobot] = []
        self.red_robots: List[HockeyRobot] = []
        self.puck = HockeyPuck()

        self.score_blue: int = 0
        self.score_red: int = 0
        self.steps: int = 0
        self.match_time: float = 0.0

        # Parâmetros de atrito e piso sorteados por episódio (Domain Randomization)
        self.friction_mu: float = C.PUCK_SLIDING_FRICTION
        self.motor_gain_scale: float = 1.0

        # Buffer de atraso de comando BLE (simula ~30ms de latência de rádio Bluetooth)
        self.ble_latency_steps: int = 2
        self.blue_cmd_buffer: List[List[Tuple[float, float]]] = []
        self.red_cmd_buffer: List[List[Tuple[float, float]]] = []

        self.reset_match()

    @property
    def all_robots(self) -> List[HockeyRobot]:
        return self.blue_robots + self.red_robots

    def reset_match(self) -> None:
        self.blue_robots = [HockeyRobot(team=0, role_id=i) for i in range(self.robots_per_team)]
        self.red_robots = [HockeyRobot(team=1, role_id=i) for i in range(self.robots_per_team)]
        self.score_blue = 0
        self.score_red = 0
        self.reset_positions(faceoff=True)

    def reset_positions(self, faceoff: bool = True) -> None:
        """Reinicia posições para o Face-off (disco no centro)."""
        self.steps = 0
        self.match_time = 0.0

        if self.domain_randomization:
            # Varia atrito do disco no piso em +-15% e motores em +-8%
            self.friction_mu = C.PUCK_SLIDING_FRICTION * float(np.random.uniform(0.85, 1.18))
            self.motor_gain_scale = float(np.random.uniform(0.92, 1.08))
        else:
            self.friction_mu = C.PUCK_SLIDING_FRICTION
            self.motor_gain_scale = 1.0

        # Disco no centro da arena (Face-off)
        px = float(np.random.uniform(-1.5, 1.5)) if faceoff else 0.0
        py = float(np.random.uniform(-2.0, 2.0)) if faceoff else 0.0
        self.puck = HockeyPuck(x=px, y=py, vx=0.0, vy=0.0)

        # Spawns dos robôs (Azul na esquerda em X=-50 cm, Vermelho na direita em X=+50 cm)
        blue_spawns = [(-50.0, 0.0, 0.0), (-72.0, -22.0, 0.0), (-72.0, 22.0, 0.0)]
        red_spawns = [(50.0, 0.0, math.pi), (72.0, 22.0, math.pi), (72.0, -22.0, math.pi)]

        for i, r in enumerate(self.blue_robots):
            sx, sy, st = blue_spawns[i % len(blue_spawns)]
            r.x = sx + float(np.random.uniform(-1.0, 1.0))
            r.y = sy + float(np.random.uniform(-1.5, 1.5))
            r.theta = st
            r.vx = r.vy = r.omega = r.v_left = r.v_right = 0.0
            r.cmd_left = r.cmd_right = 0.0
            r.touched_puck = False
            r.has_puck_in_scoop = False

        for i, r in enumerate(self.red_robots):
            sx, sy, st = red_spawns[i % len(red_spawns)]
            r.x = sx + float(np.random.uniform(-1.0, 1.0))
            r.y = sy + float(np.random.uniform(-1.5, 1.5))
            r.theta = st
            r.vx = r.vy = r.omega = r.v_left = r.v_right = 0.0
            r.cmd_left = r.cmd_right = 0.0
            r.touched_puck = False
            r.has_puck_in_scoop = False

        self.blue_cmd_buffer.clear()
        self.red_cmd_buffer.clear()

    def apply_puck_impulse(self, impulse_vx: float, impulse_vy: float) -> None:
        """Aplica um impulso externo no disco (ex.: peteleco via mouse na interface)."""
        self.puck.vx = float(np.clip(self.puck.vx + impulse_vx, -250.0, 250.0))
        self.puck.vy = float(np.clip(self.puck.vy + impulse_vy, -250.0, 250.0))

    def step(
        self,
        blue_actions: List[Tuple[float, float]],
        red_actions: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        Executa 1 tick de física (dt = 1/60 s).
        blue_actions, red_actions: pares (u_left, u_right) em [-1.0, +1.0]^2.
        """
        self.steps += 1
        self.match_time += C.DT

        # 1. Enfileira comandos no buffer para simular atraso do Bluetooth BLE (~32 ms)
        scaled_blue = [
            (
                float(np.clip(ul, -1.0, 1.0)) * C.MAX_WHEEL_SPEED * self.motor_gain_scale,
                float(np.clip(ur, -1.0, 1.0)) * C.MAX_WHEEL_SPEED * self.motor_gain_scale,
            )
            for (ul, ur) in blue_actions
        ]
        scaled_red = [
            (
                float(np.clip(ul, -1.0, 1.0)) * C.MAX_WHEEL_SPEED * self.motor_gain_scale,
                float(np.clip(ur, -1.0, 1.0)) * C.MAX_WHEEL_SPEED * self.motor_gain_scale,
            )
            for (ul, ur) in red_actions
        ]

        self.blue_cmd_buffer.append(scaled_blue)
        self.red_cmd_buffer.append(scaled_red)

        effective_blue = self.blue_cmd_buffer.pop(0) if len(self.blue_cmd_buffer) > self.ble_latency_steps else scaled_blue
        effective_red = self.red_cmd_buffer.pop(0) if len(self.red_cmd_buffer) > self.ble_latency_steps else scaled_red

        for r, (cmd_l, cmd_r) in zip(self.blue_robots, effective_blue):
            r.cmd_left = cmd_l
            r.cmd_right = cmd_r
            r.touched_puck = False
            r.has_puck_in_scoop = False

        for r, (cmd_l, cmd_r) in zip(self.red_robots, effective_red):
            r.cmd_left = cmd_l
            r.cmd_right = cmd_r
            r.touched_puck = False
            r.has_puck_in_scoop = False

        # 2. Sub-stepping 2x para precisão de contato da pá e curvas arredondadas
        sub_dt = C.DT * 0.5
        goal_scored = 0

        for _ in range(2):
            self._integrate_robots(sub_dt)
            self._integrate_puck_sliding(sub_dt)
            self._resolve_robot_robot_collisions()
            self._resolve_robot_puck_collisions()
            goal_scored = self._resolve_arena_boundaries()
            if goal_scored != 0:
                break

        # Rastro de deslizamento do disco
        self.puck.trail.append((self.puck.x, self.puck.y))
        if len(self.puck.trail) > 50:
            self.puck.trail.pop(0)

        if goal_scored == 1:
            self.score_blue += 1
        elif goal_scored == -1:
            self.score_red += 1

        return {
            "goal_scored": goal_scored,
            "blue_has_scoop": self.blue_robots[0].has_puck_in_scoop if self.blue_robots else False,
            "red_has_scoop": self.red_robots[0].has_puck_in_scoop if self.red_robots else False,
            "puck_speed": math.hypot(self.puck.vx, self.puck.vy),
        }

    def _integrate_robots(self, dt: float) -> None:
        """Cinemática diferencial e resposta de 1ª ordem dos motores DC da Vespa."""
        alpha = min(1.0, dt / C.MOTOR_TAU)
        for r in self.all_robots:
            r.v_left += (r.cmd_left - r.v_left) * alpha
            r.v_right += (r.cmd_right - r.v_right) * alpha

            v_lin = 0.5 * (r.v_right + r.v_left)
            r.omega = (r.v_right - r.v_left) / C.WHEELBASE
            r.theta = math.atan2(math.sin(r.theta + r.omega * dt), math.cos(r.theta + r.omega * dt))

            r.vx = v_lin * math.cos(r.theta)
            r.vy = v_lin * math.sin(r.theta)
            r.x += r.vx * dt
            r.y += r.vy * dt

            self._clamp_robot_in_arena(r)

    def _integrate_puck_sliding(self, dt: float) -> None:
        """
        Física de atrito seco de Coulomb para disco deslizando no piso.
        Desaceleração a = -mu_k * g na direção oposta ao movimento.
        Se a velocidade cair abaixo de PUCK_STOP_SPEED, o atrito estático trava o disco.
        """
        p = self.puck
        speed = math.hypot(p.vx, p.vy)

        if speed > 1e-4:
            decel = self.friction_mu * C.GRAVITY  # ~196 cm/s^2
            speed_drop = decel * dt
            if speed <= speed_drop or speed < C.PUCK_STOP_SPEED:
                p.vx = 0.0
                p.vy = 0.0
            else:
                ratio = (speed - speed_drop) / speed
                p.vx *= ratio
                p.vy *= ratio

        p.x += p.vx * dt
        p.y += p.vy * dt

    def _clamp_robot_in_arena(self, r: HockeyRobot) -> None:
        """Mantém os robôs dentro dos limites da arena e cantos arredondados."""
        margin_x = C.HALF_ROBOT_L
        margin_y = C.HALF_ROBOT_W

        r.x = float(np.clip(r.x, -C.HALF_LENGTH + margin_x, C.HALF_LENGTH - margin_x))
        r.y = float(np.clip(r.y, -C.HALF_WIDTH + margin_y, C.HALF_WIDTH - margin_y))

        # Restrição nos 4 cantos arredondados (R = 20 cm)
        rc = C.CORNER_RADIUS
        cx_bound = C.HALF_LENGTH - rc
        cy_bound = C.HALF_WIDTH - rc

        if abs(r.x) > cx_bound and abs(r.y) > cy_bound:
            corner_cx = math.copysign(cx_bound, r.x)
            corner_cy = math.copysign(cy_bound, r.y)
            dx = r.x - corner_cx
            dy = r.y - corner_cy
            dist = math.hypot(dx, dy)
            max_d = rc - (margin_x + margin_y) * 0.45
            if dist > max_d and dist > 1e-5:
                r.x = corner_cx + (dx / dist) * max_d
                r.y = corner_cy + (dy / dist) * max_d

    def _resolve_robot_robot_collisions(self) -> None:
        """Colisão entre dois robôs HockeyBot."""
        robots = self.all_robots
        min_d = (C.ROBOT_WIDTH + C.ROBOT_LENGTH) * 0.5 * 0.94
        n = len(robots)
        for i in range(n):
            for j in range(i + 1, n):
                r1, r2 = robots[i], robots[j]
                dx = r2.x - r1.x
                dy = r2.y - r1.y
                d = math.hypot(dx, dy)
                if d < min_d and d > 1e-6:
                    overlap = 0.5 * (min_d - d)
                    nx, ny = dx / d, dy / d
                    r1.x -= nx * overlap
                    r1.y -= ny * overlap
                    r2.x += nx * overlap
                    r2.y += ny * overlap

    def _resolve_robot_puck_collisions(self) -> None:
        """
        Colisão realista entre o chassi 17x16 cm e o disco de hockey.
        Modela a pá frontal côncava (scoop) que retém e conduz o puck!
        """
        p = self.puck
        pr = C.PUCK_RADIUS
        hl = C.HALF_ROBOT_L  # 8.0 cm
        hw = C.HALF_ROBOT_W  # 8.5 cm

        for r in self.all_robots:
            dx = p.x - r.x
            dy = p.y - r.y

            cos_t = math.cos(r.theta)
            sin_t = math.sin(r.theta)

            # Transforma a posição do puck para o referencial local do robô
            local_x = dx * cos_t + dy * sin_t
            local_y = -dx * sin_t + dy * cos_t

            # Detecção de retenção na pá frontal côncava (scoop)
            # A pá fica na frente (+X local) centrada em Y
            is_in_scoop_zone = (
                local_x > (hl - C.SCOOP_DEPTH - pr * 0.5)
                and local_x < (hl + pr * 1.2)
                and abs(local_y) < (C.SCOOP_WIDTH * 0.5)
            )

            if is_in_scoop_zone:
                r.has_puck_in_scoop = True
                # A pá empurra o puck suavemente sem expulsá-lo lateralmente
                rebound_x = hl - C.SCOOP_DEPTH + pr + 0.15
                local_x = max(local_x, rebound_x)
                # Efeito de retenção lateral (afunila o disco em direção ao centro da pá)
                local_y *= 0.92

                # Reconverte para coordenadas globais
                p.x = r.x + local_x * cos_t - local_y * sin_t
                p.y = r.y + local_x * sin_t + local_y * cos_t

                # Transfere a velocidade de condução do robô para o puck
                p.vx = r.vx * 1.08 + float(np.random.uniform(-3.0, 3.0))
                p.vy = r.vy * 1.08 + float(np.random.uniform(-3.0, 3.0))

                r.touched_puck = True
                p.last_touch_team = r.team
                continue

            # Colisão corpo rígido padrão nas demais partes do chassi (traseira e laterais)
            closest_x = float(np.clip(local_x, -hl, hl))
            closest_y = float(np.clip(local_y, -hw, hw))

            diff_x = local_x - closest_x
            diff_y = local_y - closest_y
            dist_sq = diff_x * diff_x + diff_y * diff_y

            if dist_sq < pr * pr:
                dist = math.sqrt(max(dist_sq, 1e-8))
                lnx = diff_x / dist if dist > 1e-5 else 1.0
                lny = diff_y / dist if dist > 1e-5 else 0.0

                nx = lnx * cos_t - lny * sin_t
                ny = lnx * sin_t + lny * cos_t

                overlap = pr - dist + 0.08
                p.x += nx * overlap
                p.y += ny * overlap

                # Velocidade do ponto de impacto do robô
                rx = closest_x * cos_t - closest_y * sin_t
                ry = closest_x * sin_t + closest_y * cos_t
                v_point_x = r.vx - r.omega * ry
                v_point_y = r.vy + r.omega * rx

                rel_vx = p.vx - v_point_x
                rel_vy = p.vy - v_point_y
                vn = rel_vx * nx + rel_vy * ny

                if vn < 0:
                    impulse = -(1.0 + C.ROBOT_PUCK_RESTITUTION) * vn
                    p.vx += impulse * nx + v_point_x * 0.4
                    p.vy += impulse * ny + v_point_y * 0.4

                r.touched_puck = True
                p.last_touch_team = r.team

    def _resolve_arena_boundaries(self) -> int:
        """
        Colisão do disco com o ringue de hockey:
        - Paredes laterais superior/inferior (Y = +-60 cm)
        - Cantos arredondados (R = 20 cm) que mantêm o disco deslizando continuamente
        - Gols embutidos nas paredes de fundo (X = +-90 cm para |Y| < 17.5 cm)
        Retorna: +1 se Azul marcou (no gol direito), -1 se Vermelho marcou (no gol esquerdo), 0 sem gol.
        """
        p = self.puck
        pr = C.PUCK_RADIUS
        e = C.PUCK_WALL_RESTITUTION

        # 1. Checa Gols nas extremidades (|Y| < 17.5 cm)
        if abs(p.y) < C.HALF_GOAL_WIDTH - pr * 0.5:
            if p.x > C.HALF_LENGTH + pr * 0.8:
                return 1   # Gol do time Azul!
            if p.x < -C.HALF_LENGTH - pr * 0.8:
                return -1  # Gol do time Vermelho!

        # 2. Paredes laterais retas superior e inferior (Y = +-60 cm) para |X| <= 70 cm
        rc = C.CORNER_RADIUS
        straight_x = C.HALF_LENGTH - rc  # 90 - 20 = 70 cm
        straight_y = C.HALF_WIDTH - rc   # 60 - 20 = 40 cm

        if abs(p.x) <= straight_x:
            if p.y > C.HALF_WIDTH - pr:
                p.y = C.HALF_WIDTH - pr
                p.vy = -abs(p.vy) * e
            elif p.y < -C.HALF_WIDTH + pr:
                p.y = -C.HALF_WIDTH + pr
                p.vy = abs(p.vy) * e

        # 3. Paredes de fundo retas fora do gol (X = +-90 cm) para |Y| entre 17.5 cm e 40 cm
        if abs(p.y) >= C.HALF_GOAL_WIDTH - pr * 0.5 and abs(p.y) <= straight_y:
            if p.x > C.HALF_LENGTH - pr:
                p.x = C.HALF_LENGTH - pr
                p.vx = -abs(p.vx) * e
            elif p.x < -C.HALF_LENGTH + pr:
                p.x = -C.HALF_LENGTH + pr
                p.vx = abs(p.vx) * e

        # 4. Colisão nos 4 Cantos Arredondados (R = 20 cm)
        if abs(p.x) > straight_x and abs(p.y) > straight_y:
            corner_cx = math.copysign(straight_x, p.x)
            corner_cy = math.copysign(straight_y, p.y)
            dx = p.x - corner_cx
            dy = p.y - corner_cy
            dist = math.hypot(dx, dy)
            max_d = rc - pr

            if dist > max_d and dist > 1e-5:
                # Normal apontando para dentro do ringue
                nx = -(dx / dist)
                ny = -(dy / dist)
                penetration = dist - max_d
                p.x += nx * penetration
                p.y += ny * penetration

                vn = p.vx * nx + p.vy * ny
                if vn < 0:
                    p.vx -= (1.0 + e) * vn * nx
                    p.vy -= (1.0 + e) * vn * ny
                    # Leve atrito tangencial para o disco deslizar contornando a curva
                    p.vx *= 0.96
                    p.vy *= 0.96

        return 0
