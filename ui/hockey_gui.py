"""
Central Gráfica Interativa de Simulação de Hockey de Robôs RoboCore (Classe 1lb / HockeyBot).

Recursos:
- Ringue oficial com piso sintético/madeira, cantos arredondados (R = 20 cm) e balizas
- Robôs fiéis ao chassi RoboCore 17x16 cm (454 g) com pá frontal côncava (scoop) e rodas
- Mini Puck de hockey oficial com atrito seco de Coulomb e rastro dinâmico de deslizamento
- Aceleração de 1x a 100x + Botão 'Ficar Burro (Reset IA)'
- Interação ao vivo com o Mouse: arraste o puck ou os robôs, e dê 'petelecos' com botão direito
- Controle manual WASD / Setas simulando o aplicativo oficial RoboCore Joystick via Bluetooth!
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np
import pygame

from core.constants import HockeyConstants as C
from core.hockey_physics import HockeyRobot, HockeyPuck
from bots.hockey_bot import HeuristicHockeyBot
from rl.env import RoboCoreHockeyEnv
from rl.ppo_trainer import HockeyPPOTrainer


class UIButton:
    def __init__(self, rect: Tuple[int, int, int, int], label: str, color=(40, 52, 70)):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, active: bool = False) -> None:
        bg = (235, 94, 40) if active else self.color  # Laranja RoboCore quando ativo
        border = (255, 160, 100) if active else (75, 90, 115)
        pygame.draw.rect(screen, bg, self.rect, border_radius=6)
        pygame.draw.rect(screen, border, self.rect, width=2, border_radius=6)
        txt = font.render(self.label, True, (245, 248, 255))
        screen.blit(
            txt,
            (
                self.rect.centerx - txt.get_width() // 2,
                self.rect.centery - txt.get_height() // 2,
            ),
        )

    def clicked(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


def run_hockey_gui() -> None:
    pygame.init()
    W, H = 1300, 760
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        "RoboCore Hockey Simulator — Robôs 1lb (17×16 cm) & Ringue com Atrito de Coulomb"
    )
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Segoe UI", 19, bold=True)
    font_score = pygame.font.SysFont("Segoe UI", 26, bold=True)
    font_sub = pygame.font.SysFont("Segoe UI", 14, bold=True)
    font_body = pygame.font.SysFont("Segoe UI", 13)
    font_small = pygame.font.SysFont("Consolas", 12)

    env = RoboCoreHockeyEnv(robots_per_team=C.ROBOTS_PER_TEAM, max_steps=900)
    obs, _ = env.reset()
    trainer = HockeyPPOTrainer(obs_dim=env.obs_dim, act_dim=env.act_dim)
    heuristic_blue_bot = HeuristicHockeyBot(team=0, difficulty=0.92)

    # Carrega checkpoint pré-treinado se disponível
    import os
    ckpt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
        "hockey_ppo.pt",
    )
    if os.path.exists(ckpt_path):
        try:
            trainer.load(ckpt_path)
        except Exception:
            pass

    # Conversão de escala cm -> pixels para a arena de 180x120 cm
    scale = 3.65  # 1 cm = 3.65 px -> 180 cm = 657 px, 120 cm = 438 px
    field_cx, field_cy = 380, 400

    def cm_to_px(x_cm: float, y_cm: float) -> Tuple[int, int]:
        return int(field_cx + x_cm * scale), int(field_cy + y_cm * scale)

    def px_to_cm(px: int, py: int) -> Tuple[float, float]:
        return (px - field_cx) / scale, (py - field_cy) / scale

    speed_mult = 1
    speed_options = [1, 5, 20, 50, 100]
    blue_mode = "PPO_TRAIN"  # "PPO_TRAIN", "PPO_EVAL", "HEURISTIC", "HUMAN_WASD"

    episode_reward = 0.0
    reward_history: List[float] = []
    dragged_entity = None
    flick_start: Tuple[int, int] | None = None
    flick_curr: Tuple[int, int] | None = None

    # Botões do painel direito
    speed_btns = [
        UIButton((780 + i * 62, 110, 56, 32), f"{spd}x")
        for i, spd in enumerate(speed_options)
    ]
    btn_reset_ai = UIButton((1100, 110, 175, 32), "Ficar Burro (Reset)", color=(150, 42, 50))

    mode_btns = [
        ("PPO_TRAIN", UIButton((780, 178, 120, 32), "PPO (Treinar)")),
        ("PPO_EVAL", UIButton((906, 178, 120, 32), "PPO (Jogar)")),
        ("HEURISTIC", UIButton((1032, 178, 120, 32), "Bot Analítico")),
        ("HUMAN_WASD", UIButton((1158, 178, 117, 32), "App Joystick")),
    ]

    show_vision_overlay = True
    btn_faceoff = UIButton((780, 246, 155, 32), "Novo Face-off")
    btn_vision = UIButton((945, 246, 175, 32), "Visão Zenital (ON)", color=(35, 75, 55))
    btn_save_ckpt = UIButton((1130, 246, 145, 32), "Salvar Pesos")

    last_act = np.zeros(2, dtype=np.float32)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for spd, b in zip(speed_options, speed_btns):
                        if b.clicked(event.pos):
                            speed_mult = spd
                    if btn_reset_ai.clicked(event.pos):
                        trainer.reset_weights()
                        reward_history.clear()
                        env.engine.score_blue = 0
                        env.engine.score_red = 0
                    for m_id, b in mode_btns:
                        if b.clicked(event.pos):
                            blue_mode = m_id
                    if btn_faceoff.clicked(event.pos):
                        env.engine.reset_positions(faceoff=True)
                        obs = env.get_team_obs(team=0)
                    if btn_vision.clicked(event.pos):
                        show_vision_overlay = not show_vision_overlay
                        btn_vision.label = f"Visão Zenital ({'ON' if show_vision_overlay else 'OFF'})"
                        btn_vision.color = (35, 75, 55) if show_vision_overlay else (55, 60, 70)
                    if btn_save_ckpt.clicked(event.pos):
                        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                        trainer.save(ckpt_path)

                    # Arrastar robô ou puck
                    mx_cm, my_cm = px_to_cm(*event.pos)
                    if math.hypot(mx_cm - env.engine.puck.x, my_cm - env.engine.puck.y) < 5.0:
                        dragged_entity = "puck"
                    else:
                        for idx_r, r in enumerate(env.engine.all_robots):
                            if math.hypot(mx_cm - r.x, my_cm - r.y) < 8.0:
                                dragged_entity = ("robot", idx_r)
                                break

                elif event.button == 3:
                    # Inicia "Peteleco" no puck com o botão direito
                    px_px, py_px = cm_to_px(env.engine.puck.x, env.engine.puck.y)
                    flick_start = (px_px, py_px)
                    flick_curr = event.pos

            elif event.type == pygame.MOUSEMOTION:
                if flick_start is not None:
                    flick_curr = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragged_entity = None
                elif event.button == 3 and flick_start is not None and flick_curr is not None:
                    # Dispara o puck proporcionalmente à força puxada
                    dx_px = flick_curr[0] - flick_start[0]
                    dy_px = flick_curr[1] - flick_start[1]
                    ivx = dx_px * 0.95
                    ivy = dy_px * 0.95
                    env.engine.apply_puck_impulse(ivx, ivy)
                    flick_start = None
                    flick_curr = None

        # Arrastar entidade com botão esquerdo
        if dragged_entity is not None and pygame.mouse.get_pressed()[0]:
            mx_cm, my_cm = px_to_cm(*mouse_pos)
            mx_cm = float(np.clip(mx_cm, -C.HALF_LENGTH + 4.0, C.HALF_LENGTH - 4.0))
            my_cm = float(np.clip(my_cm, -C.HALF_WIDTH + 4.0, C.HALF_WIDTH - 4.0))
            if dragged_entity == "puck":
                env.engine.puck.x = mx_cm
                env.engine.puck.y = my_cm
                env.engine.puck.vx = 0.0
                env.engine.puck.vy = 0.0
            elif isinstance(dragged_entity, tuple) and dragged_entity[0] == "robot":
                r_target = env.engine.all_robots[dragged_entity[1]]
                r_target.x = mx_cm
                r_target.y = my_cm
                r_target.vx = r_target.vy = 0.0

        # Loop de passos por frame (1x até 100x)
        steps_now = speed_mult if blue_mode == "PPO_TRAIN" else min(speed_mult, 10)
        keys = pygame.key.get_pressed()

        for _ in range(steps_now):
            if blue_mode == "PPO_TRAIN":
                act, logp, val = trainer.select_action(obs, deterministic=False)
            elif blue_mode == "PPO_EVAL":
                act, logp, val = trainer.select_action(obs, deterministic=True)
            elif blue_mode == "HEURISTIC":
                acts = heuristic_blue_bot.get_actions(env.engine)
                act = np.array(acts, dtype=np.float32)
                logp, val = np.zeros(3), np.zeros(3)
            else:
                # App Joystick: Pilota o Atacante (robô 0) no teclado, enquanto Ala e Goleiro seguem o bot azul
                fwd = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN])
                turn = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
                if keys[pygame.K_SPACE]:
                    act_0 = np.array([-1.0, 1.0], dtype=np.float32)
                else:
                    act_0 = np.clip(np.array([fwd - 0.75 * turn, fwd + 0.75 * turn], dtype=np.float32), -1.0, 1.0)
                auto_acts = heuristic_blue_bot.get_actions(env.engine)
                act = np.array([act_0, auto_acts[1], auto_acts[2]], dtype=np.float32)
                logp, val = np.zeros(3), np.zeros(3)

            last_act = act
            next_obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            episode_reward += rew

            if blue_mode == "PPO_TRAIN":
                trainer.store_transition(obs, act, logp, info.get("individual_rewards", rew), val, done)
                if len(trainer.rew_buf) >= 1024:
                    trainer.update()

            obs = next_obs
            if done:
                reward_history.append(episode_reward)
                if len(reward_history) > 120:
                    reward_history.pop(0)
                episode_reward = 0.0
                obs, _ = env.reset()

        # ================= RENDERIZAÇÃO =================
        screen.fill((16, 20, 28))

        # Cabeçalho
        pygame.draw.rect(screen, (24, 32, 46), (0, 0, W, 56))
        screen.blit(
            font_title.render(
                "ROBOCORE HOCKEY RL — SIMULADOR FÍSICO COM PUCK DESLIZANTE (ATRITO DE COULOMB)",
                True,
                (240, 245, 255),
            ),
            (20, 16),
        )

        # --- 1. RINGUE DE HOCKEY OFICIAL (ESQUERDA) ---
        left_panel = pygame.Rect(16, 68, 745, 675)
        pygame.draw.rect(screen, (22, 28, 38), left_panel, border_radius=10)
        pygame.draw.rect(screen, (50, 66, 92), left_panel, width=2, border_radius=10)

        # Placar de Hockey
        score_txt = font_score.render(
            f"AZUL (RL/PPO)  {env.engine.score_blue}   ×   {env.engine.score_red}  VERMELHO (BOT)",
            True,
            (240, 248, 255),
        )
        screen.blit(score_txt, (field_cx - score_txt.get_width() // 2, 78))

        # Superfície da arena (piso sintético claro de ringue)
        r_pts = []
        rc = C.CORNER_RADIUS
        hl, hw = C.HALF_LENGTH, C.HALF_WIDTH

        # Constrói o polígono do ringue com os 4 cantos arredondados
        # Topo
        r_pts.append(cm_to_px(-hl + rc, -hw))
        r_pts.append(cm_to_px(hl - rc, -hw))
        # Canto superior direito
        for a_deg in range(-90, 1, 10):
            ang = math.radians(a_deg)
            r_pts.append(cm_to_px(hl - rc + rc * math.cos(ang), -hw + rc + rc * math.sin(ang)))
        # Lateral direita
        r_pts.append(cm_to_px(hl, hw - rc))
        # Canto inferior direito
        for a_deg in range(0, 91, 10):
            ang = math.radians(a_deg)
            r_pts.append(cm_to_px(hl - rc + rc * math.cos(ang), hw - rc + rc * math.sin(ang)))
        # Base
        r_pts.append(cm_to_px(-hl + rc, hw))
        # Canto inferior esquerdo
        for a_deg in range(90, 181, 10):
            ang = math.radians(a_deg)
            r_pts.append(cm_to_px(-hl + rc + rc * math.cos(ang), hw - rc + rc * math.sin(ang)))
        # Lateral esquerda
        r_pts.append(cm_to_px(-hl, -hw + rc))
        # Canto superior esquerdo
        for a_deg in range(180, 271, 10):
            ang = math.radians(a_deg)
            r_pts.append(cm_to_px(-hl + rc + rc * math.cos(ang), -hw + rc + rc * math.sin(ang)))

        pygame.draw.polygon(screen, (34, 44, 58), r_pts)  # Fundo da quadra de hockey
        pygame.draw.polygon(screen, (220, 235, 255), r_pts, 3)  # Borda de acrílico/MDF

        # Gols (balizas esquerda e direita)
        gl_x1, gl_y1 = cm_to_px(-hl - C.GOAL_DEPTH, -C.HALF_GOAL_WIDTH)
        gl_x2, gl_y2 = cm_to_px(-hl, C.HALF_GOAL_WIDTH)
        pygame.draw.rect(screen, (25, 45, 80), (gl_x1, gl_y1, gl_x2 - gl_x1, gl_y2 - gl_y1))
        pygame.draw.rect(screen, (60, 150, 255), (gl_x1, gl_y1, gl_x2 - gl_x1, gl_y2 - gl_y1), 2)

        gr_x1, gr_y1 = cm_to_px(hl, -C.HALF_GOAL_WIDTH)
        gr_x2, gr_y2 = cm_to_px(hl + C.GOAL_DEPTH, C.HALF_GOAL_WIDTH)
        pygame.draw.rect(screen, (80, 35, 35), (gl_x1 := gr_x1, gl_y1 := gr_y1, gr_x2 - gr_x1, gr_y2 - gr_y1))
        pygame.draw.rect(screen, (255, 80, 80), (gl_x1, gl_y1, gr_x2 - gr_x1, gr_y2 - gr_y1), 2)

        # Marcações oficiais: Linha central vermelha, círculo de Face-off central e semicírculos das áreas
        line_red = (225, 75, 75)
        line_blue = (75, 140, 230)
        pygame.draw.line(screen, line_red, cm_to_px(0, -hw), cm_to_px(0, hw), 2)
        pygame.draw.circle(screen, line_blue, cm_to_px(0, 0), int(20.0 * scale), 2)
        pygame.draw.circle(screen, line_red, cm_to_px(0, 0), 4)

        # Semicírculos de proteção das balizas (Goal Crease)
        pygame.draw.arc(
            screen,
            line_blue,
            pygame.Rect(
                cm_to_px(-hl - C.GOAL_AREA_RADIUS, -C.GOAL_AREA_RADIUS)[0],
                cm_to_px(-hl - C.GOAL_AREA_RADIUS, -C.GOAL_AREA_RADIUS)[1],
                int(C.GOAL_AREA_RADIUS * 2 * scale),
                int(C.GOAL_AREA_RADIUS * 2 * scale),
            ),
            -math.pi / 2,
            math.pi / 2,
            2,
        )
        pygame.draw.arc(
            screen,
            line_blue,
            pygame.Rect(
                cm_to_px(hl - C.GOAL_AREA_RADIUS, -C.GOAL_AREA_RADIUS)[0],
                cm_to_px(hl - C.GOAL_AREA_RADIUS, -C.GOAL_AREA_RADIUS)[1],
                int(C.GOAL_AREA_RADIUS * 2 * scale),
                int(C.GOAL_AREA_RADIUS * 2 * scale),
            ),
            math.pi / 2,
            3 * math.pi / 2,
            2,
        )

        # Rastro de atrito de deslizamento do Puck
        puck = env.engine.puck
        if len(puck.trail) >= 2:
            pts_trail = [cm_to_px(tx, ty) for tx, ty in puck.trail]
            pygame.draw.lines(screen, (255, 175, 75), False, pts_trail, 2)

        # Desenha os Robôs RoboCore (Chassi 17x16 cm com Pá Frontal Côncava e Rodas)
        for r in env.engine.all_robots:
            cos_t = math.cos(r.theta)
            sin_t = math.sin(r.theta)
            hl_r, hw_r = C.HALF_ROBOT_L, C.HALF_ROBOT_W

            # Perfil do chassi com rebaixo côncavo frontal (scoop) derivado do CAD
            local_chassis = [
                (-hl_r, -hw_r),                  # Traseira esquerda
                (hl_r, -hw_r),                   # Frente esquerda
                (hl_r, -r.scoop_width * 0.5),    # Borda externa da pá esquerda
                (hl_r - r.scoop_depth, -r.scoop_width * 0.35), # Fundo côncavo da pá
                (hl_r - r.scoop_depth, r.scoop_width * 0.35),  # Fundo côncavo da pá
                (hl_r, r.scoop_width * 0.5),     # Borda externa da pá direita
                (hl_r, hw_r),                    # Frente direita
                (-hl_r, hw_r),                   # Traseira direita
            ]
            poly_px = []
            for lx, ly in local_chassis:
                wx = r.x + lx * cos_t - ly * sin_t
                wy = r.y + lx * sin_t + ly * cos_t
                poly_px.append(cm_to_px(wx, wy))

            body_col = (28, 48, 76) if r.team == 0 else (76, 35, 35)
            border_col = (65, 160, 255) if r.team == 0 else (255, 95, 95)
            pygame.draw.polygon(screen, body_col, poly_px)
            pygame.draw.polygon(screen, border_col, poly_px, 2)

            # Rodas esquerda e direita
            wheel_l_cm = C.WHEEL_DIAMETER
            wheel_w_cm = 1.6
            for side_sign in (-1.0, 1.0):
                wy_loc = side_sign * (C.WHEELBASE * 0.5)
                # 4 cantos da roda
                w_corners = [
                    (-wheel_l_cm * 0.5, wy_loc - wheel_w_cm * 0.5),
                    (wheel_l_cm * 0.5, wy_loc - wheel_w_cm * 0.5),
                    (wheel_l_cm * 0.5, wy_loc + wheel_w_cm * 0.5),
                    (-wheel_l_cm * 0.5, wy_loc + wheel_w_cm * 0.5),
                ]
                wheel_px = []
                for lx, ly in w_corners:
                    wx = r.x + lx * cos_t - ly * sin_t
                    wy = r.y + lx * sin_t + ly * cos_t
                    wheel_px.append(cm_to_px(wx, wy))
                pygame.draw.polygon(screen, (15, 18, 22), wheel_px)
                pygame.draw.polygon(screen, (130, 140, 155), wheel_px, 1)

            # Efeito visual de derrapagem (Wheel Slip / fumaça de tração)
            if r.is_slipping:
                rcx_slip, rcy_slip = cm_to_px(r.x, r.y)
                pygame.draw.circle(screen, (255, 175, 45), (rcx_slip, rcy_slip), 14, 2)

            # Placa Vespa (ESP32) no centro do chassi
            rcx, rcy = cm_to_px(r.x, r.y)
            pygame.draw.circle(screen, (235, 94, 40), (rcx, rcy), 4)  # LED da Vespa

            # --- MARCAS FÍSICAS DE TETO DO CHASSI (RASTREAMENTO DE VISÃO COMPUTACIONAL) ---
            # 1. Marca Frontal de Proa: LED Branco/Amarelo (+4.5 cm) para definir vetor diretor
            proa_x = r.x + C.MARKER_FRONT_OFFSET * cos_t
            proa_y = r.y + C.MARKER_FRONT_OFFSET * sin_t
            proa_px = cm_to_px(proa_x, proa_y)
            proa_r = max(4, int(C.MARKER_RADIUS * scale))
            pygame.draw.circle(screen, (255, 255, 230), proa_px, proa_r)
            pygame.draw.circle(screen, (30, 35, 45), proa_px, proa_r, 1)

            # 2. Marca Traseira de Popa: Cor Tática do Papel (-4.5 cm)
            popa_x = r.x + C.MARKER_REAR_OFFSET * cos_t
            popa_y = r.y + C.MARKER_REAR_OFFSET * sin_t
            popa_px = cm_to_px(popa_x, popa_y)
            popa_r = max(4, int(C.MARKER_RADIUS * scale))

            # Cores das marcas por time e papel (0=Atacante, 1=Ala, 2=Goleiro)
            if r.team == 0:
                role_colors = {0: (50, 205, 250), 1: (255, 215, 40), 2: (70, 245, 120)}
            else:
                role_colors = {0: (255, 110, 40), 1: (240, 70, 200), 2: (170, 60, 245)}
            mark_col = role_colors.get(r.role_id, (200, 200, 200))
            pygame.draw.circle(screen, mark_col, popa_px, popa_r)
            pygame.draw.circle(screen, (20, 25, 30), popa_px, popa_r, 1)

            # Eixo longitudinal medido pela visão (conecta popa à proa)
            pygame.draw.line(screen, (190, 225, 255), popa_px, proa_px, 1)

            # Tag de Papel Tático 3v3 gravada na popa: A = Atacante, D = Defensor/Ala, G = Goleiro
            role_tag = ["A", "D", "G"][r.role_id % 3]
            role_surf = font_small.render(role_tag, True, (240, 250, 255))
            screen.blit(role_surf, (rcx - role_surf.get_width() // 2, rcy - 17))

            # Indicador de disco retido na pá frontal ou ejetado em giro
            if r.has_puck_in_scoop:
                scoop_front_x = r.x + (hl_r - r.scoop_depth * 0.5) * cos_t
                scoop_front_y = r.y + (hl_r - r.scoop_depth * 0.5) * sin_t
                pygame.draw.circle(screen, (46, 213, 115), cm_to_px(scoop_front_x, scoop_front_y), 5)
            elif r.scoop_ejected:
                scoop_front_x = r.x + (hl_r + 2.0) * cos_t
                scoop_front_y = r.y + (hl_r + 2.0) * sin_t
                pygame.draw.circle(screen, (255, 75, 75), cm_to_px(scoop_front_x, scoop_front_y), 6, 2)

        # Desenha o Mini Puck Oficial de Hockey (preto/cinza escuro com borda refletiva)
        px_px, py_px = cm_to_px(puck.x, puck.y)
        pr_px = max(6, int(C.PUCK_RADIUS * scale))
        pygame.draw.circle(screen, (12, 14, 18), (px_px, py_px), pr_px)
        pygame.draw.circle(screen, (220, 230, 245), (px_px, py_px), pr_px, 2)
        pygame.draw.circle(screen, (45, 55, 70), (px_px, py_px), max(2, pr_px // 2))

        # --- RETÍCULAS DE RASTREAMENTO DA VISÃO COMPUTACIONAL & KALMAN OVERLAY ---
        if show_vision_overlay and hasattr(env, "vision_sensor"):
            vs = env.vision_sensor
            # Rastreamento dos 6 Robôs na arena
            for idx_r, r in enumerate(env.engine.all_robots):
                tracked_state = vs.tracked_robots.get(idx_r)
                if tracked_state is None:
                    continue

                kf_px, kf_py = cm_to_px(tracked_state.x, tracked_state.y)
                conf = tracked_state.confidence
                role_char = ["A", "D", "G"][r.role_id % 3]
                team_prefix = "Azul" if r.team == 0 else "Verm"

                # Status de acordo com o nível de confiança do rastreador
                if conf >= 0.85:
                    box_col = (46, 213, 115)  # Verde: Rastreamento firme das marcas
                    status_lbl = f"{role_char} {int(conf * 100)}%"
                elif conf >= 0.40:
                    box_col = (255, 175, 45)  # Âmbar: Oclusão parcial / Predição Kalman
                    status_lbl = f"{role_char} KF {int(conf * 100)}%"
                else:
                    box_col = (255, 75, 75)   # Vermelho: Oclusão crítica por colisão
                    status_lbl = f"{role_char} OCL"

                box_sz = int(C.HALF_ROBOT_L * 2 * scale * 1.15)
                b_rect = pygame.Rect(kf_px - box_sz // 2, kf_py - box_sz // 2, box_sz, box_sz)
                c_len = max(6, box_sz // 4)

                # Cantoneiras de tracking OpenCV / ArUco
                pygame.draw.line(screen, box_col, (b_rect.left, b_rect.top), (b_rect.left + c_len, b_rect.top), 2)
                pygame.draw.line(screen, box_col, (b_rect.left, b_rect.top), (b_rect.left, b_rect.top + c_len), 2)
                pygame.draw.line(screen, box_col, (b_rect.right, b_rect.top), (b_rect.right - c_len, b_rect.top), 2)
                pygame.draw.line(screen, box_col, (b_rect.right, b_rect.top), (b_rect.right, b_rect.top + c_len), 2)
                pygame.draw.line(screen, box_col, (b_rect.left, b_rect.bottom), (b_rect.left + c_len, b_rect.bottom), 2)
                pygame.draw.line(screen, box_col, (b_rect.left, b_rect.bottom), (b_rect.left, b_rect.bottom - c_len), 2)
                pygame.draw.line(screen, box_col, (b_rect.right, b_rect.bottom), (b_rect.right - c_len, b_rect.bottom), 2)
                pygame.draw.line(screen, box_col, (b_rect.right, b_rect.bottom), (b_rect.right, b_rect.bottom - c_len), 2)

                # Vetor de velocidade estimado pelo Kalman
                if math.hypot(tracked_state.vx, tracked_state.vy) > 6.0:
                    vel_end = cm_to_px(
                        tracked_state.x + tracked_state.vx * 0.14,
                        tracked_state.y + tracked_state.vy * 0.14,
                    )
                    pygame.draw.line(screen, box_col, (kf_px, kf_py), vel_end, 1)

                lbl_surf = font_small.render(status_lbl, True, box_col)
                screen.blit(lbl_surf, (b_rect.left, b_rect.top - 14))

            # Rastreamento do Disco (Puck)
            p_state = vs.tracked_puck
            pk_px, pk_py = cm_to_px(p_state.x, p_state.y)
            p_conf = p_state.confidence
            p_col = (46, 213, 115) if p_conf >= 0.85 else ((255, 175, 45) if p_conf >= 0.40 else (255, 75, 75))
            p_rad = max(9, int(C.PUCK_RADIUS * scale * 1.5))
            pygame.draw.circle(screen, p_col, (pk_px, pk_py), p_rad, 1)
            pygame.draw.line(screen, p_col, (pk_px - p_rad - 3, pk_py), (pk_px + p_rad + 3, pk_py), 1)
            pygame.draw.line(screen, p_col, (pk_px, pk_py - p_rad - 3), (pk_px, pk_py + p_rad + 3), 1)
            p_lbl_txt = f"PUCK {int(p_conf * 100)}%" if p_state.tracked else "PUCK (NA PÁ/OCL)"
            p_lbl = font_small.render(p_lbl_txt, True, p_col)
            screen.blit(p_lbl, (pk_px + p_rad + 4, pk_py - 6))

        # Seta do Peteleco com botão direito
        if flick_start is not None and flick_curr is not None:
            pygame.draw.line(screen, (255, 75, 85), flick_start, flick_curr, 3)
            pygame.draw.circle(screen, (255, 75, 85), flick_curr, 6)

        # Telemetria da arena no rodapé para Hockey 3v3
        p_speed = math.hypot(puck.vx, puck.vy)
        b_robs = env.engine.blue_robots
        r0 = b_robs[0]
        r1 = b_robs[1]
        r2 = b_robs[2]
        has_scoop = any(r.has_puck_in_scoop for r in b_robs)
        foot1 = (
            f"Hockey 3v3 | Disco: ({puck.x:+5.1f}, {puck.y:+5.1f}) cm | Vel: {p_speed:5.1f} cm/s | "
            f"Atrito μ_k={env.engine.friction_mu:.2f} | Posse Azul: {'SIM' if has_scoop else 'NÃO'}"
        )
        foot2 = (
            f"Time Azul: Atacante A({r0.x:+4.0f}, {r0.y:+4.0f}) | "
            f"Ala D({r1.x:+4.0f}, {r1.y:+4.0f}) | Goleiro G({r2.x:+4.0f}, {r2.y:+4.0f}) | "
            f"BLE: {env.engine.ble_latency_ms:4.1f}ms"
        )
        screen.blit(font_small.render(foot1, True, (190, 210, 235)), (28, 680))
        screen.blit(font_small.render(foot2, True, (125, 230, 170)), (28, 704))

        # --- 2. PAINEL DIREITO: CONTROLES, CURVAS RL E FASES COGNITIVAS ---
        right_panel = pygame.Rect(770, 68, 515, 675)
        pygame.draw.rect(screen, (22, 28, 38), right_panel, border_radius=10)
        pygame.draw.rect(screen, (50, 66, 92), right_panel, width=2, border_radius=10)

        screen.blit(font_sub.render("ACELERAÇÃO DE TREINO E RESET", True, (200, 220, 245)), (782, 82))
        for spd, btn in zip(speed_options, speed_btns):
            btn.draw(screen, font_body, active=(speed_mult == spd))
        btn_reset_ai.draw(screen, font_body)

        screen.blit(font_sub.render("CONTROLADOR DO TIME AZUL (ROBÔ RL)", True, (200, 220, 245)), (782, 152))
        for m_id, btn in mode_btns:
            btn.draw(screen, font_body, active=(blue_mode == m_id))

        screen.blit(font_sub.render("AÇÕES RÁPIDAS DE RINGUE", True, (200, 220, 245)), (782, 220))
        btn_faceoff.draw(screen, font_body)
        btn_vision.draw(screen, font_body, active=show_vision_overlay)
        btn_save_ckpt.draw(screen, font_body)

        # Gráfico de evolução de recompensa PPO
        chart_rect = pygame.Rect(782, 298, 492, 195)
        pygame.draw.rect(screen, (15, 20, 28), chart_rect, border_radius=8)
        pygame.draw.rect(screen, (55, 72, 98), chart_rect, width=1, border_radius=8)
        screen.blit(
            font_small.render("EVOLUÇÃO DA RECOMPENSA (PPO NO HOCKEY DE ROBÔS)", True, (160, 195, 235)),
            (794, 306),
        )

        if len(reward_history) >= 2:
            r_min = min(reward_history)
            r_max = max(reward_history)
            span = max(5.0, r_max - r_min)
            pts = []
            for idx_r, rv in enumerate(reward_history):
                px = chart_rect.left + 14 + int((idx_r / max(1, len(reward_history) - 1)) * (chart_rect.width - 28))
                py = chart_rect.bottom - 20 - int(((rv - r_min) / span) * (chart_rect.height - 48))
                pts.append((px, py))
            pygame.draw.lines(screen, (235, 94, 40), False, pts, 2)
            avg_r = float(np.mean(reward_history[-15:]))
            screen.blit(
                font_small.render(f"Média Recente (15 ep): {avg_r:+6.2f}", True, (255, 160, 100)),
                (794, chart_rect.bottom - 20),
            )
        else:
            screen.blit(
                font_small.render("Aguardando acumular episódios (coloque em 50x ou 100x)...", True, (130, 150, 175)),
                (830, 388),
            )

        # Painel de Fase Cognitiva do Hockey & Percepção de Visão
        phase_rect = pygame.Rect(782, 508, 492, 220)
        pygame.draw.rect(screen, (15, 20, 28), phase_rect, border_radius=8)
        pygame.draw.rect(screen, (55, 72, 98), phase_rect, width=1, border_radius=8)

        st = trainer.total_steps
        if st < 5000:
            p_title = "FASE 1: Exploração Caótica (Giro sem rumo e perda rápida de disco)"
            p_col = (245, 105, 105)
        elif st < 20000:
            p_title = "FASE 2: Choque Frontal (Empurra o puck reto sem reter na pá)"
            p_col = (245, 195, 85)
        elif st < 55000:
            p_title = "FASE 3: Retenção no Scoop (Aprende a encaixar o puck e conduzir)"
            p_col = (105, 195, 255)
        else:
            p_title = "FASE 4: Finalização Cirúrgica (Curva rápida, tabela e gol!)"
            p_col = (75, 235, 145)

        screen.blit(font_sub.render(p_title, True, p_col), (792, 516))
        screen.blit(
            font_small.render(
                f"Passos PPO: {trainer.total_steps:,}  |  Updates: {trainer.updates_count}  |  Loss: {trainer.last_loss:.4f}",
                True,
                (210, 225, 245),
            ),
            (792, 538),
        )

        # Telemetria ao vivo da visão computacional e tracking de marcas
        vs = env.vision_sensor
        r0_t, r0_c = vs.tracked_robots[0].tracked, int(vs.tracked_robots[0].confidence * 100)
        r1_t, r1_c = vs.tracked_robots[1].tracked, int(vs.tracked_robots[1].confidence * 100)
        r2_t, r2_c = vs.tracked_robots[2].tracked, int(vs.tracked_robots[2].confidence * 100)
        pk_t, pk_c = vs.tracked_puck.tracked, int(vs.tracked_puck.confidence * 100)

        s0 = f"A[{'OK' if r0_t else 'KF'} {r0_c}%]"
        s1 = f"D[{'OK' if r1_t else 'KF'} {r1_c}%]"
        s2 = f"G[{'OK' if r2_t else 'KF'} {r2_c}%]"
        spk = f"Puck[{'OK' if pk_t else 'SCOOP'} {pk_c}%]"

        screen.blit(
            font_small.render(
                f"Visão Zenital (Câmera + Kalman): {s0}  {s1}  {s2}  {spk}",
                True,
                (70, 235, 160) if show_vision_overlay else (140, 150, 165),
            ),
            (792, 560),
        )

        screen.blit(
            font_small.render("Pipeline Sim-to-Real (Visão Parcial -> Kalman -> RL -> BLE):", True, (255, 160, 90)),
            (792, 586),
        )
        screen.blit(
            font_small.render("• Marcas no Chassi: Proa (LED Branco +4.5cm) + Popa (Cor Papel -4.5cm)", True, (185, 205, 230)),
            (792, 608),
        )
        screen.blit(
            font_small.render("• Oclusões em colisões tratadas com predição cinemática por Filtro Kalman", True, (185, 205, 230)),
            (792, 628),
        )
        screen.blit(
            font_small.render("• Disco na pá côncava: retenção mecânica e ejeção por giro rápido (>6.2 rad/s)", True, (185, 205, 230)),
            (792, 648),
        )
        screen.blit(
            font_small.render("• Comunicação Vespa BLE (30Hz) com compensação de deadband dos motores (12%)", True, (120, 235, 165)),
            (792, 668),
        )
        screen.blit(
            font_small.render("• Pressione 'App Joystick' para pilotar no teclado (WASD / Espaço para girar)", True, (160, 200, 240)),
            (792, 690),
        )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
