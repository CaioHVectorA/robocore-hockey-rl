"""Hockey Lab: interactive experiment design, live PPO and independent evaluation."""

from __future__ import annotations
import argparse
from collections import deque
import math
from pathlib import Path
import pygame
import numpy as np
import torch
from core.constants import PROFILES
from bots.hockey_bot import HeuristicHockeyBot
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import frame_from_env, curriculum_stage
from rl.training_config import TrainingConfig, preset_config, PRESET_LABELS
from ui.arena_view import draw_arena, projection
from ui.training_panel import TrainingPanel, BG, TEXT, MUTED, ACCENT, EDGE, GREEN, TABS
from ui.text_rendering import fit_text, render_text
from ui.timing import FrameClock

ROOT = Path(__file__).resolve().parents[1]


def run_hockey_gui(
    profile="micro-2024",
    policy=None,
    screenshot=None,
    max_frames=None,
    training_config=None,
    preset=None,
    auto_train=False,
    initial_tab="Treino",
    human=False,
):
    pygame.init()
    torch.set_num_threads(1)
    screen = pygame.display.set_mode((1600, 960), pygame.RESIZABLE)
    pygame.display.set_caption("Hockey Lab · Bancada de treino PPO")
    clock = FrameClock()
    fonts = {
        n: pygame.font.SysFont("Segoe UI,DejaVu Sans", n, bold=n >= 22) for n in (12, 14, 16, 18, 22, 28, 40)
    }
    config = training_config or TrainingConfig(profile=profile)
    if preset:
        config = preset_config(preset, config)
    env = config.make_env()
    obs, _ = env.reset(seed=config.seed)
    trainer = HockeyPPOTrainer()
    bot = HeuristicHockeyBot(0, 0.92)
    mode, status = "Demonstração", "Prévia do exercício · bot tático"
    loaded, paused, tracking = False, False, False
    selected, speed, accumulator, frames, episode = 0, 1, 0.0, 0, 1
    replay, events = deque(maxlen=900), deque(maxlen=3)
    replay_index, drag, flick = None, None, None
    match_over, sudden_death = False, False
    buttons = []
    last_live = None

    def prepare(cfg):
        nonlocal env, obs, config, selected, accumulator, mode, loaded, status
        nonlocal paused, match_over, sudden_death, drag, flick, replay_index, episode
        config = cfg
        env = cfg.make_env()
        obs, _ = env.reset(seed=cfg.seed)
        selected, accumulator = 0, 0.0
        mode, loaded, paused = "Demonstração", False, False
        status = f"Prévia · {cfg.name} · bot tático"
        match_over = sudden_death = False
        drag = flick = replay_index = None
        replay.clear()
        episode = 1

    def evaluate(path, cfg):
        nonlocal trainer, mode, loaded, status, obs
        candidate = HockeyPPOTrainer()
        meta = candidate.load(path)
        expected = cfg.make_env().configuration()
        if any(meta.get("configuration", {}).get(key) != expected[key] for key in ("arena", "robot")):
            raise ValueError("Pesos de outra arena ou outro modelo físico")
        if not candidate.updates_count:
            raise ValueError("Esses pesos ainda não tiveram atualizações de PPO")
        prepare(cfg)
        env.scenario = curriculum_stage(cfg, 1)
        obs, _ = env.reset(seed=cfg.seed)
        trainer, mode, loaded = candidate, "Política PPO", True
        status = f"Avaliação visual · pesos congelados na atualização {trainer.updates_count}"

    panel = TrainingPanel(ROOT, fonts, prepare, evaluate, config)
    panel.tab = initial_tab
    if policy:
        try:
            evaluate(Path(policy), config)
        except (ValueError, RuntimeError, OSError) as exc:
            panel.notify(exc, True)
    if auto_train:
        panel.start()

    def text(label, x, y, size=16, color=TEXT, width=None):
        value = fit_text(fonts[size], str(label), width)
        screen.blit(render_text(fonts[size], value, color), (int(x), int(y)))

    def button(label, rect, callback, active=False):
        rect = pygame.Rect(rect)
        pygame.draw.rect(screen, ACCENT if active else (37, 54, 73), rect, border_radius=7)
        surf = fonts[14].render(label, True, BG if active else TEXT)
        screen.blit(surf, surf.get_rect(center=rect.center))
        visible = screen.get_clip().clip(rect)
        if visible.height and visible.width:
            buttons.append((visible, callback))

    def reset(new_match=True, scenario=None):
        nonlocal obs, accumulator, replay_index, match_over, sudden_death
        reset_scenario = "faceoff" if mode == "Humano × PPO" else scenario or env.scenario
        obs, _ = env.reset(options={"preserve_match": not new_match, "scenario": reset_scenario})
        accumulator, replay_index = 0.0, None
        if new_match:
            replay.clear()
            match_over = sudden_death = False

    def set_mode(value):
        nonlocal mode, status
        panel.watch_live = False
        if value == "Humano × PPO" and (config.robots_per_team != 1 or config.opponent_robots != 1):
            panel.notify("Selecione o exercício 1v1 e carregue os pesos para jogar contra a IA", True)
            return
        if value in ("Política PPO", "Humano × PPO") and not loaded:
            panel.evaluate()
            if not loaded:
                return
        mode = value
        status = (
            "Você controla o vermelho · WASD / setas · IA controla o azul"
            if value == "Humano × PPO"
            else "Bancada manual · WASD / setas"
            if value == "Bancada manual"
            else f"Avaliação · atualização {trainer.updates_count}"
            if value == "Política PPO"
            else "Prévia do exercício · bot tático"
        )
        if value == "Humano × PPO":
            reset()

    def manual_action(keys):
        fwd = int(keys[pygame.K_w] or keys[pygame.K_UP]) - int(keys[pygame.K_s] or keys[pygame.K_DOWN])
        turn = int(keys[pygame.K_a] or keys[pygame.K_LEFT]) - int(keys[pygame.K_d] or keys[pygame.K_RIGHT])
        return [-1, 1] if keys[pygame.K_SPACE] else np.clip([fwd - turn * 0.7, fwd + turn * 0.7], -1, 1)

    def pause_view():
        nonlocal paused
        if panel.watch_live and panel.worker.active:
            panel.toggle_pause()
        else:
            paused = not paused

    def set_speed(value):
        nonlocal speed
        speed = value

    def toggle_tracking():
        nonlocal tracking
        tracking = not tracking

    if human:
        panel.tab = "Simular"
        set_mode("Humano × PPO")
    running = True
    try:
        while running:
            elapsed = min(clock.tick(60) / 1000, 0.1)
            w, h = screen.get_size()
            if w < 1180 or h < 800:
                screen = pygame.display.set_mode((max(1180, w), max(800, h)), pygame.RESIZABLE)
                w, h = screen.get_size()
            side_w = min(540, max(480, round(w * 0.34)))
            left_w = w - side_w - 56
            field = pygame.Rect(20, 144, left_w, h - 511)
            health_rect = pygame.Rect(20, h - 355, left_w, 98)
            metrics_rect = pygame.Rect(20, h - 252, left_w, 133)
            world, metric, scale = projection(field, env.arena)
            panel.worker.poll()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if panel.handle(event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        pause_view()
                    elif event.key == pygame.K_r and not panel.watch_live:
                        reset()
                    elif event.key == pygame.K_v:
                        tracking = not tracking
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        set_mode(
                            {
                                pygame.K_1: "Demonstração",
                                pygame.K_2: "Bancada manual",
                                pygame.K_3: "Política PPO",
                                pygame.K_4: "Humano × PPO",
                            }[event.key]
                        )
                    elif event.key == pygame.K_TAB:
                        selected = (selected + 1) % config.robots_per_team
                    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET) and replay:
                        paused, panel.watch_live = True, False
                        replay_index = len(replay) - 1 if replay_index is None else replay_index
                        replay_index = max(
                            0,
                            min(
                                len(replay) - 1,
                                replay_index + (-15 if event.key == pygame.K_LEFTBRACKET else 15),
                            ),
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    handled = False
                    if event.button == 1:
                        for rect, callback in buttons:
                            if rect.collidepoint(event.pos):
                                callback()
                                handled = True
                                break
                    if handled or panel.watch_live or not field.collidepoint(event.pos):
                        continue
                    mx, my = metric(event.pos)
                    if event.button == 1:
                        for r in env.engine.blue_robots:
                            if math.hypot(r.x - mx, r.y - my) < 12:
                                selected = r.role_id
                        if paused:
                            drag = (
                                "puck"
                                if math.hypot(env.engine.puck.x - mx, env.engine.puck.y - my) < 6
                                else min(env.engine.all_robots, key=lambda r: math.hypot(r.x - mx, r.y - my))
                            )
                    elif event.button == 3:
                        flick = (mx, my)
                elif event.type == pygame.MOUSEBUTTONUP and not panel.watch_live:
                    if event.button == 1 and drag is not None:
                        mx, my = metric(event.pos)
                        mx = float(np.clip(mx, -env.arena.half_length + 12, env.arena.half_length - 12))
                        my = float(np.clip(my, -env.arena.half_width + 12, env.arena.half_width - 12))
                        if drag == "puck":
                            env.engine.place_puck(mx, my)
                        else:
                            env.engine.place_robot(drag, mx, my)
                        env.vision_sensor.reset()
                        env.vision_sensor.update(env.engine, prime=True)
                        env.prev_potential, env.prev_approach = env._potential(), env._approach()
                        obs, drag = env.get_team_obs(), None
                    if event.button == 3 and flick:
                        mx, my = metric(event.pos)
                        env.engine.apply_puck_impulse((flick[0] - mx) * 4, (flick[1] - my) * 4)
                        flick = None
            if not running:
                break
            if panel.live:
                data = panel.worker.frame
                if data.get("timestamp") != last_live:
                    replay.append(data)
                    last_live = data.get("timestamp")
            elif not panel.watch_live and not paused and not match_over:
                accumulator += elapsed * speed
                for _ in range(min(16, int(accumulator / env.engine.dt))):
                    accumulator -= env.engine.dt
                    if mode in ("Política PPO", "Humano × PPO") and loaded:
                        actions, _, _ = trainer.select_action(obs, deterministic=True)
                    else:
                        actions = np.array(bot.get_actions(env.engine, env.state), np.float32)
                        if mode == "Bancada manual":
                            keys = pygame.key.get_pressed()
                            actions[selected] = manual_action(keys)
                    red = (
                        np.zeros((config.opponent_robots, 2)) if config.opponent in ("none", "zero") else None
                    )
                    if mode == "Humano × PPO":
                        red = np.array([manual_action(pygame.key.get_pressed())], np.float32)
                    obs, _, term, trunc, info = env.step(actions, red)
                    replay.append(frame_from_env(env))
                    task = mode != "Humano × PPO" and (
                        config.objective == "reach"
                        or config.opponent_robots == 0
                        or config.scenario != "faceoff"
                    )
                    if term:
                        events.appendleft("Objetivo atingido" if info["is_success"] else "Gol adversário")
                    if task and (term or trunc):
                        episode += 1
                        reset()
                        break
                    if env.engine.match_time >= env.arena.match_seconds:
                        if env.engine.score_blue == env.engine.score_red:
                            sudden_death = True
                        elif not sudden_death or term:
                            match_over = True
                    if term and sudden_death:
                        match_over = True
                    if (term or trunc) and not match_over:
                        reset(new_match=False)
                    if match_over:
                        break
            data = (
                panel.worker.display_frame()
                if panel.live
                else replay[replay_index]
                if replay_index is not None and replay
                else frame_from_env(env)
            )
            screen.fill(BG)
            buttons = []
            text("HOCKEY", 24, 17, 28)
            text("LAB", 155, 17, 28, ACCENT)
            live_label = (
                "TREINO AO VIVO · ambiente 1 · ações exploratórias"
                if panel.watch_live
                else "VOCÊ: VERMELHO (WASD / SETAS) · IA PPO: AZUL · 1v1"
                if mode == "Humano × PPO"
                else "REPLAY · quadro gravado"
                if replay_index is not None
                else "AVALIAÇÃO VISUAL · pesos congelados"
                if mode == "Política PPO"
                else "PRÉVIA DO EXERCÍCIO · bot tático"
                if mode == "Demonstração"
                else "BANCADA MANUAL"
            )
            text(live_label, 25, 58, 12, GREEN if panel.watch_live else MUTED, left_w - 10)
            text(
                f"{env.arena.name}  ·  {env.arena.length:g} × {env.arena.width:g} cm", w - 455, 24, 14, MUTED
            )
            pygame.draw.line(screen, EDGE, (20, 86), (w - 20, 86))
            text(f"AZUL {config.robots_per_team}", 32, 101, 16, (101, 174, 255))
            text(f"{data.get('score_blue', 0)}  ×  {data.get('score_red', 0)}", 145, 94, 28)
            text(
                "SOLO" if config.opponent_robots == 0 else f"VERMELHO {config.opponent_robots}",
                274,
                101,
                16,
                (255, 126, 137),
            )
            clock_seconds = (
                config.episode_seconds
                if panel.watch_live or config.opponent_robots == 0 or config.objective == "reach"
                else env.arena.match_seconds
            )
            elapsed_time = data.get("time", 0) if panel.watch_live else env.engine.match_time
            remaining = max(0, clock_seconds - elapsed_time)
            text(
                f"EP {data.get('episode', episode)} · {int(remaining) // 60:02d}:{int(remaining) % 60:02d}"
                if panel.watch_live or config.opponent_robots == 0
                else "GOLDEN GOAL"
                if sudden_death
                else f"{int(remaining) // 60:02d}:{int(remaining) % 60:02d}",
                left_w - 195,
                98,
                22,
            )
            draw_arena(screen, field, env.arena, env.engine.robot_config, data, fonts[12], selected, tracking)
            text(
                "Câmera zenital · centímetros · chassi comercial estimado",
                field.x + 16,
                field.bottom - 23,
                12,
                MUTED,
            )
            if flick:
                pygame.draw.line(screen, ACCENT, world(*flick), pygame.mouse.get_pos(), 2)
            panel.draw_metrics(screen, metrics_rect)
            panel.draw_health(screen, health_rect, data, clock.get_fps())
            content = panel.draw(screen, pygame.Rect(w - side_w - 20, 101, side_w, h - 121))
            if panel.tab == "Simular":
                screen.set_clip(content)
                x, y, width = content.x, content.y, content.width
                for j, value in enumerate(("Demonstração", "Bancada manual", "Política PPO", "Humano × PPO")):
                    button(
                        value,
                        (x, y + j * 39, width, 32),
                        lambda v=value: set_mode(v),
                        mode == value and not panel.watch_live,
                    )
                y += 171
                button("Pausar / continuar [P]", (x, y, width, 32), pause_view)
                button(
                    "Nova partida [R]",
                    (x, y + 41, width, 32),
                    lambda: reset() if not panel.watch_live else None,
                )
                text("Velocidade da prévia / avaliação", x, y + 87, 12, MUTED)
                for i, value in enumerate((1, 2, 5, 10)):
                    button(
                        f"{value}×",
                        (x + i * width / 4, y + 110, width / 4 - 5, 30),
                        lambda v=value: set_speed(v),
                        value == speed,
                    )
                button("Rastreamento [V]", (x, y + 155, width, 32), toggle_tracking, tracking)
                text("Pausado: arraste robôs/disco · direito: impulso", x, y + 204, 12, MUTED, width)
                text("Tab seleciona · [ ] replay · WASD controla", x, y + 226, 12, MUTED, width)
            screen.set_clip(None)
            footer = (
                "Preparando o processo de treino…"
                if panel.watch_live and not panel.live
                else f"Coleta real · atualização {data.get('updates', 0)} · retorno do episódio {data.get('episode_return', 0):.2f}"
                if panel.watch_live
                else status
            )
            text(footer, 25, h - 99, 14, ACCENT, left_w)
            text(
                f"{'Chegar até o disco' if config.objective == 'reach' else 'Buscar o disco e marcar gol'} · {config.robots_per_team}v{config.opponent_robots}",
                25,
                h - 71,
                12,
                MUTED,
                left_w,
            )
            text(
                "Para observar: ritmo 1×. Para treinar mais rápido: Máximo. Avaliar usa pesos congelados.",
                25,
                h - 47,
                12,
                MUTED,
                left_w,
            )
            panel.draw_details(screen)
            pygame.display.flip()
            frames += 1
            if screenshot and (frames >= max_frames if max_frames else frames == 1):
                Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, str(screenshot))
            if max_frames and frames >= max_frames:
                break
    finally:
        panel.worker.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Treino sem janela: python main.py --headless --preset solo_goal. Ajuda: python train.py --help",
    )
    parser.add_argument("--profile", choices=PROFILES, default="micro-2024")
    parser.add_argument("--policy")
    parser.add_argument("--screenshot")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--training-config")
    parser.add_argument("--preset", choices=PRESET_LABELS)
    parser.add_argument("--train", action="store_true", help="Start the configured session immediately")
    parser.add_argument(
        "--human", action="store_true", help="Você controla o vermelho contra PPO azul num exercício 1v1"
    )
    parser.add_argument("--tab", choices=TABS, default="Treino")
    args = parser.parse_args()
    config = TrainingConfig.load(args.training_config) if args.training_config else None
    run_hockey_gui(
        args.profile,
        args.policy,
        args.screenshot,
        args.frames,
        config,
        args.preset,
        args.train,
        args.tab,
        args.human,
    )


if __name__ == "__main__":
    main()
