"""Responsive match viewer, manual bench and separate training process."""

from __future__ import annotations
import argparse
from collections import deque
import math
from pathlib import Path
import subprocess
import sys
import time
import pygame
import numpy as np
import torch
from core.constants import PROFILES
from ui.robot_renderer import draw_robot
from bots.hockey_bot import HeuristicHockeyBot
from rl.env import RoboCoreHockeyEnv
from rl.ppo_trainer import HockeyPPOTrainer

ROOT = Path(__file__).resolve().parents[1]
BG = (11, 18, 30)
PANEL = (21, 32, 48)
BORDER = (43, 59, 77)
TEXT = (232, 239, 245)
MUTED = (142, 164, 183)
ORANGE = (255, 155, 72)
BLUE = (34, 113, 217)
RED = (213, 66, 80)


def run_hockey_gui(profile="micro-2024", policy=None, screenshot=None, max_frames=None):
    pygame.init()
    torch.set_num_threads(1)
    screen = pygame.display.set_mode((1380, 860), pygame.RESIZABLE)
    pygame.display.set_caption("Hockey Lab · Simulação e robótica")
    clock = pygame.time.Clock()
    fonts = {n: pygame.font.SysFont("DejaVu Sans", n, bold=(n >= 22)) for n in [12, 14, 16, 18, 22, 28, 40]}
    env = RoboCoreHockeyEnv(profile=profile, max_steps=9000)
    obs, _ = env.reset(seed=42)
    bots = [HeuristicHockeyBot(0, 0.92), HeuristicHockeyBot(1, 0.88)]
    trainer = HockeyPPOTrainer()
    loaded = False
    mode = "Demonstração"
    paused = False
    speed = 1
    selected = 0
    status = "Bot tático × bot tático"
    show_tracking = False
    scenario = "faceoff"
    events = deque(["Partida iniciada · modelos físicos ainda aguardam calibração"], maxlen=4)
    buttons = []
    accumulator = 0.0
    frames = 0
    replay = deque(maxlen=900)
    replay_index = None
    training = None
    training_log = None
    training_started = None
    ckpt = Path(policy) if policy else ROOT / "checkpoints/hockey_v2.pt"
    if ckpt.exists():
        try:
            meta = trainer.load(ckpt)
            loaded = (
                trainer.updates_count > 0
                and meta.get("configuration", {}).get("arena") == env.arena.to_dict()
                and meta.get("configuration", {}).get("robot") == env.configuration()["robot"]
            )
            status = (
                f"Política disponível · {trainer.updates_count} atualizações"
                if loaded
                else "Checkpoint sem treino ou de outro modelo físico"
            )
        except (ValueError, RuntimeError) as exc:
            status = str(exc)
    flick = None
    drag = None
    match_over = False
    sudden_death = False

    def text(label, x, y, size=16, color=TEXT, width=None):
        label = str(label)
        if width:
            while fonts[size].size(label)[0] > width and len(label) > 3:
                label = label[:-4] + "…"
        screen.blit(fonts[size].render(label, True, color), (int(x), int(y)))

    def button(label, rect, callback, active=False):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(
            screen, (60, 71, 83) if hover else (ORANGE if active else (31, 47, 66)), rect, border_radius=8
        )
        surf = fonts[14].render(label, True, BG if active else TEXT)
        screen.blit(surf, surf.get_rect(center=rect.center))
        buttons.append((rect, callback))

    def load_policy():
        nonlocal loaded, status, mode
        try:
            meta = trainer.load(ckpt)
            if not trainer.updates_count:
                raise ValueError("Este checkpoint ainda não teve atualizações")
            if meta.get("configuration", {}).get("arena") != env.arena.to_dict():
                raise ValueError("O checkpoint usa outra arena")
            if meta.get("configuration", {}).get("robot") != env.configuration()["robot"]:
                raise ValueError("O checkpoint usa outro modelo de robô; treine com o HockeyBot atual")
            loaded = True
            mode = "Política PPO"
            status = f"PPO · {trainer.updates_count} atualizações"
        except (OSError, ValueError, RuntimeError) as exc:
            status = str(exc)

    def reset(new_match=True, new_scenario=None):
        nonlocal obs, accumulator, match_over, sudden_death, replay_index
        obs, _ = env.reset(options={"preserve_match": not new_match, "scenario": new_scenario or scenario})
        accumulator = 0.0
        replay_index = None
        if new_match:
            replay.clear()
        if new_match:
            match_over = False
            sudden_death = False

    def start_training():
        nonlocal training, training_log, status, training_started
        if training and training.poll() is None:
            status = "Treinamento já está em execução"
            return
        (ROOT / "runs").mkdir(exist_ok=True)
        if training_log:
            training_log.close()
        training_log = (ROOT / "runs/gui-training.log").open("w")
        training = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "train.py"),
                "--profile",
                profile,
                "--steps",
                "100000",
                "--save",
                str(ckpt),
            ],
            cwd=ROOT,
            stdout=training_log,
            stderr=subprocess.STDOUT,
        )
        training_started = time.monotonic()
        status = "Treino separado iniciado · 100 mil passos"

    def set_mode(value):
        nonlocal mode, status
        if value == "Política PPO":
            load_policy()
        else:
            mode = value
            status = (
                "WASD / setas · espaço gira · clique seleciona"
                if value == "Bancada manual"
                else "Bot tático × bot tático"
            )

    running = True
    try:
        while running:
            elapsed = min(clock.tick(60) / 1000, 0.1)
            w, h = screen.get_size()
            if w < 1100 or h < 760:
                screen = pygame.display.set_mode((max(w, 1100), max(h, 760)), pygame.RESIZABLE)
                w, h = screen.get_size()
            side_w = 330
            left_w = w - side_w - 56
            field_panel = pygame.Rect(20, 136, left_w, h - 258)
            a = env.arena
            scale = min(
                (field_panel.width - 80) / (a.length + 2 * a.goal_depth), (field_panel.height - 72) / a.width
            )
            cx, cy = field_panel.center

            def world(x, y):
                return (round(cx + x * scale), round(cy - y * scale))

            def metric(pos):
                return ((pos[0] - cx) / scale, (cy - pos[1]) / scale)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                        replay_index = None
                    elif event.key == pygame.K_r:
                        reset()
                    elif event.key == pygame.K_v:
                        show_tracking = not show_tracking
                    elif event.key == pygame.K_1:
                        set_mode("Demonstração")
                    elif event.key == pygame.K_2:
                        set_mode("Bancada manual")
                    elif event.key == pygame.K_3:
                        set_mode("Política PPO")
                    elif event.key == pygame.K_TAB:
                        selected = (selected + 1) % 3
                    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET) and replay:
                        paused = True
                        replay_index = len(replay) - 1 if replay_index is None else replay_index
                        replay_index = int(
                            np.clip(
                                replay_index + (-15 if event.key == pygame.K_LEFTBRACKET else 15),
                                0,
                                len(replay) - 1,
                            )
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    handled = False
                    if event.button == 1:
                        for rect, callback in buttons:
                            if rect.collidepoint(event.pos):
                                callback()
                                handled = True
                                break
                    if not handled and field_panel.collidepoint(event.pos):
                        mx, my = metric(event.pos)
                        if event.button == 1:
                            for r in env.engine.blue_robots:
                                if math.hypot(r.x - mx, r.y - my) < 12:
                                    selected = r.role_id
                            if paused:
                                drag = (
                                    "puck"
                                    if math.hypot(env.engine.puck.x - mx, env.engine.puck.y - my) < 6
                                    else min(
                                        env.engine.all_robots, key=lambda r: math.hypot(r.x - mx, r.y - my)
                                    )
                                )
                        elif event.button == 3:
                            flick = metric(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and drag is not None:
                        mx, my = metric(event.pos)
                        mx = float(np.clip(mx, -a.half_length + 12, a.half_length - 12))
                        my = float(np.clip(my, -a.half_width + 12, a.half_width - 12))
                        if drag == "puck":
                            env.engine.place_puck(mx, my)
                        else:
                            env.engine.place_robot(drag, mx, my)
                        env.vision_sensor.reset()
                        env.vision_sensor.update(env.engine, prime=True)
                        env.prev_potential = env._potential()
                        env.prev_approach = env._approach()
                        obs = env.get_team_obs()
                        drag = None
                    if event.button == 3 and flick:
                        mx, my = metric(event.pos)
                        env.engine.apply_puck_impulse((flick[0] - mx) * 4, (flick[1] - my) * 4)
                        flick = None
            if not running:
                break
            if training and training.poll() is not None and training_started is not None:
                status = (
                    "Treino concluído · Carregar pesos para avaliar"
                    if training.returncode == 0
                    else "Falha no treino · consulte runs/gui-training.log"
                )
                training_started = None
            if not paused and not match_over:
                accumulator += elapsed * speed
                for _ in range(min(12, int(accumulator / env.engine.dt))):
                    accumulator -= env.engine.dt
                    if mode == "Política PPO" and loaded:
                        actions, _, _ = trainer.select_action(obs, deterministic=True)
                    else:
                        actions = np.array(bots[0].get_actions(env.engine, env.state), np.float32)
                        if mode == "Bancada manual":
                            keys = pygame.key.get_pressed()
                            fwd = int(keys[pygame.K_w] or keys[pygame.K_UP]) - int(
                                keys[pygame.K_s] or keys[pygame.K_DOWN]
                            )
                            turn = int(keys[pygame.K_a] or keys[pygame.K_LEFT]) - int(
                                keys[pygame.K_d] or keys[pygame.K_RIGHT]
                            )
                            actions[selected] = (
                                [-1, 1]
                                if keys[pygame.K_SPACE]
                                else np.clip([fwd - turn * 0.7, fwd + turn * 0.7], -1, 1)
                            )
                    obs, _, term, trunc, info = env.step(actions)
                    replay.append(
                        dict(
                            robots=[(r.x, r.y, r.theta, r.team, r.role_id) for r in env.engine.all_robots],
                            puck=(env.engine.puck.x, env.engine.puck.y),
                            time=env.engine.match_time,
                        )
                    )
                    if term:
                        events.appendleft("GOL DO AZUL" if info["goal_scored"] == 1 else "GOL DO VERMELHO")
                        if sudden_death:
                            match_over = True
                        else:
                            reset(new_match=False)
                    if env.engine.match_time >= a.match_seconds and not sudden_death:
                        if env.engine.score_blue == env.engine.score_red:
                            sudden_death = True
                            events.appendleft("Morte súbita · próximo gol decide")
                        else:
                            match_over = True
                    if trunc and not term and not match_over:
                        reset(new_match=False)
                    if match_over:
                        break

            screen.fill(BG)
            buttons = []
            text("HOCKEY", 24, 18, 28)
            text("LAB", 155, 18, 28, ORANGE)
            text("SIMULAÇÃO  /  APRENDIZADO  /  VISÃO ZENITAL", 25, 58, 12, MUTED)
            text(f"{a.name.upper()}  ·  {a.length:g} × {a.width:g} cm  ·  3v3", w - 430, 28, 14, MUTED)
            pygame.draw.line(screen, BORDER, (20, 86), (w - 20, 86))
            text("AZUL", 45, 100, 18, (101, 174, 255))
            text(str(env.engine.score_blue), 140, 91, 28)
            text("×", 190, 97, 22, MUTED)
            text(str(env.engine.score_red), 230, 91, 28)
            text("VERMELHO", 285, 100, 18, (255, 126, 137))
            remaining = max(0.0, a.match_seconds - env.engine.match_time)
            time_label = (
                "GOLDEN GOAL" if sudden_death else f"{int(remaining) // 60:02d}:{int(remaining) % 60:02d}"
            )
            text(time_label, left_w - 170, 95, 28, ORANGE if sudden_death else TEXT)
            pygame.draw.rect(screen, PANEL, field_panel, border_radius=16)
            rink = pygame.Rect(0, 0, round(a.length * scale), round(a.width * scale))
            rink.center = (cx, cy)
            pygame.draw.rect(screen, (222, 233, 235), rink, border_radius=round(a.corner_radius * scale))
            pygame.draw.rect(
                screen, (187, 207, 214), rink, width=3, border_radius=round(a.corner_radius * scale)
            )
            for sign in [-1, 1]:
                goal = pygame.Rect(
                    min(
                        world(sign * a.half_length, 0)[0], world(sign * (a.half_length + a.goal_depth), 0)[0]
                    ),
                    world(0, a.goal_width / 2)[1],
                    round(a.goal_depth * scale),
                    round(a.goal_width * scale),
                )
                pygame.draw.rect(screen, BLUE if sign == -1 else RED, goal, border_radius=3)
                for yy in range(goal.top, goal.bottom, 7):
                    pygame.draw.line(screen, (151, 176, 190), (goal.left, yy), (goal.right, yy), 1)
                pygame.draw.circle(
                    screen, (123, 177, 200), world(sign * a.half_length, 0), round(22 * scale), 2
                )
            pygame.draw.line(screen, (195, 88, 103), world(0, a.half_width), world(0, -a.half_width), 2)
            pygame.draw.circle(screen, (115, 158, 182), world(0, 0), round(20 * scale), 2)
            pygame.draw.circle(screen, (195, 88, 103), world(0, 0), 3)
            for sign in [-1, 1]:
                pygame.draw.line(
                    screen,
                    (149, 183, 201),
                    world(sign * a.length * 0.25, -a.half_width),
                    world(sign * a.length * 0.25, a.half_width),
                    1,
                )
            data = replay[replay_index] if replay_index is not None and replay else None
            robots = (
                data["robots"]
                if data
                else [(r.x, r.y, r.theta, r.team, r.role_id) for r in env.engine.all_robots]
            )
            for x, y, theta, team, role in robots:
                draw_robot(screen, world, x, y, theta, BLUE if team == 0 else RED, env.engine.robot_config)
                center = world(x, y)
                label = fonts[12].render(str(team * 3 + role), True, (255, 255, 255))
                screen.blit(label, label.get_rect(center=center))
                if team == 0 and role == selected:
                    pygame.draw.circle(screen, ORANGE, center, round(13 * scale), 2)
                if show_tracking and not data:
                    track = env.state.robots.get(team * 3 + role)
                    if track and track.valid(env.state.timestamp):
                        pos = world(track.x, track.y)
                        pygame.draw.circle(screen, (0, 151, 117), pos, round(11 * scale), 1)
                        pygame.draw.line(
                            screen,
                            (0, 151, 117),
                            pos,
                            world(track.x + track.vx * 0.15, track.y + track.vy * 0.15),
                            2,
                        )
            p = data["puck"] if data else (env.engine.puck.x, env.engine.puck.y)
            if len(env.engine.puck.trail) > 1 and not data:
                pygame.draw.lines(
                    screen, (180, 158, 104), False, [world(*xy) for xy in env.engine.puck.trail], 2
                )
            pygame.draw.circle(screen, (18, 27, 37), world(*p), max(4, round(a.puck_radius * scale)))
            pygame.draw.circle(screen, (246, 205, 111), world(*p), max(4, round(a.puck_radius * scale)), 1)
            if flick:
                pygame.draw.line(screen, ORANGE, world(*flick), pygame.mouse.get_pos(), 2)
            if paused or match_over:
                label = "FIM DA PARTIDA" if match_over else ("REPLAY" if data else "PAUSADO")
                surf = fonts[28].render(label, True, BG)
                box = surf.get_rect(center=(cx, field_panel.top + 35))
                pygame.draw.rect(screen, ORANGE, box.inflate(24, 12), border_radius=8)
                screen.blit(surf, box)
            text(
                "MODELO 2D · +Y para cima · medidas em centímetros",
                field_panel.left + 16,
                field_panel.bottom - 24,
                12,
                (133, 157, 179),
            )

            sx = w - side_w - 20
            sy = 136
            panel = pygame.Rect(sx, sy, side_w, h - 156)
            pygame.draw.rect(screen, PANEL, panel, border_radius=16)
            x = sx + 18
            inner = side_w - 36
            text("CENTRAL DE CONTROLE", x, sy + 17, 16)
            text("Controlador azul", x, sy + 52, 12, MUTED)
            for j, name in enumerate(["Demonstração", "Bancada manual", "Política PPO"]):
                button(
                    name, (x, sy + 77 + j * 38, inner, 32), lambda value=name: set_mode(value), mode == name
                )
            y = sy + 203

            def pause():
                nonlocal paused, replay_index
                paused = not paused
                replay_index = None

            button("Continuar [P]" if paused else "Pausar [P]", (x, y, inner / 2 - 4, 33), pause)
            button("Nova partida [R]", (x + inner / 2 + 4, y, inner / 2 - 4, 33), lambda: reset())
            text("Velocidade da simulação", x, y + 49, 12, MUTED)

            def set_speed(value):
                nonlocal speed
                speed = value

            for i, value in enumerate([1, 2, 5, 10]):
                button(
                    f"{value}×",
                    (x + i * (inner / 4 + 1), y + 72, inner / 4 - 4, 31),
                    lambda v=value: set_speed(v),
                    speed == value,
                )
            text("Cenários de bancada", x, y + 121, 12, MUTED)
            for i, (name, label) in enumerate(
                [("shooting", "Finalização"), ("defending", "Defesa"), ("faceoff", "Face-off")]
            ):
                button(
                    label,
                    (x + i * (inner / 3 + 1), y + 145, inner / 3 - 3, 31),
                    lambda v=name: reset(new_scenario=v),
                )

            def tracking():
                nonlocal show_tracking
                show_tracking = not show_tracking

            button("Rastreamento [V]", (x, y + 190, inner, 32), tracking, show_tracking)
            text("TREINAMENTO", x, y + 244, 14, ORANGE)
            button("Iniciar treino em processo separado", (x, y + 271, inner, 34), start_training)
            button("Carregar pesos para avaliação", (x, y + 313, inner, 34), load_policy)
            text(f"Updates PPO: {trainer.updates_count:,}", x, y + 359, 14, MUTED)
            if training and training.poll() is None:
                text(
                    f"Treino ativo · {int(time.monotonic() - (training_started or time.monotonic()))} s",
                    x,
                    y + 384,
                    12,
                    (102, 213, 171),
                )
            if h >= 850:
                text("CONTROLES", x, y + 400, 12, MUTED)
                text("WASD / setas: mover · espaço: girar", x, y + 423, 12, width=inner)
                text("Tab / clique: selecionar · [ ]: replay", x, y + 443, 12, width=inner)
                text("Pausado: arraste para reposicionar", x, y + 463, 12, width=inner)
                text("Botão direito + puxar: impulso no disco", x, y + 483, 12, width=inner)

            footer_y = h - 105
            text(status, 26, footer_y, 14, ORANGE, width=left_w - 10)
            text(
                f"Disco {env.engine.puck.body.velocity.length:5.1f} cm/s  ·  Rádio {env.engine.ble_latency_ms:.0f} ms  ·  FPS {clock.get_fps():.0f}",
                26,
                footer_y + 28,
                12,
                MUTED,
            )
            for i, line in enumerate(list(events)[:2]):
                text(line, 26, footer_y + 52 + i * 19, 12, MUTED, width=left_w)
            pygame.display.flip()
            frames += 1
            if screenshot and frames == 1:
                pygame.image.save(screen, str(screenshot))
            if max_frames and frames >= max_frames:
                break
    finally:
        if training and training.poll() is None:
            training.terminate()
            try:
                training.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                training.kill()
        if training_log:
            training_log.close()
        pygame.quit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="micro-2024")
    p.add_argument("--policy")
    p.add_argument("--screenshot")
    p.add_argument("--frames", type=int)
    a = p.parse_args()
    run_hockey_gui(a.profile, a.policy, a.screenshot, a.frames)


if __name__ == "__main__":
    main()
