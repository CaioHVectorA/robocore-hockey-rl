"""Generate a high-definition (720p HD) showcase video of 10 difficult hockey situations."""

import math
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import pygame

from rl.env import RoboCoreHockeyEnv
from rl.ppo_trainer import HockeyPPOTrainer
from rl.training import frame_from_env
from ui.arena_view import draw_arena

ARTIFACT_DIR = Path(r"C:\Users\caihe\.gemini\antigravity\brain\4d9de323-c3e6-49f7-b3ae-6f2299acb331")
OUTPUT_DIR = Path("videos")
CHECKPOINT = Path("checkpoints/solo-1v0-350rpm-1600iter.pt")

SITUATIONS = [
    {
        "num": 1,
        "title": "Canto Superior Defensivo",
        "desc": "Puck colado na quina de defesa; saida veloz para contra-ataque",
        "robot": (-45.0, 5.0, 0.0),
        "puck": (-66.0, 26.0),
    },
    {
        "num": 2,
        "title": "Canto Inferior Defensivo (180° Invertido)",
        "desc": "Robo de costas para a quina sul; laço de retorno e saida de zona",
        "robot": (-35.0, -10.0, math.pi),
        "puck": (-66.0, -26.0),
    },
    {
        "num": 3,
        "title": "Canto Superior Ofensivo (Quina Fechada)",
        "desc": "Puck espremido no fundo direito; puxada e chute em angulo agudo",
        "robot": (10.0, 10.0, 0.0),
        "puck": (62.0, 26.0),
    },
    {
        "num": 4,
        "title": "Canto Inferior Ofensivo (Bico da Trave)",
        "desc": "Resgate no bico da parede sul e finalizacao cruzada no gol",
        "robot": (5.0, -15.0, 0.2),
        "puck": (62.0, -26.0),
    },
    {
        "num": 5,
        "title": "Centro da Quadra com Inversao Total (180°)",
        "desc": "Robo entre o disco e o gol; contorno perfeito sem tocar contra",
        "robot": (15.0, 5.0, 0.0),
        "puck": (0.0, 0.0),
    },
    {
        "num": 6,
        "title": "Parede Lateral Norte (Rente ao Alumínio)",
        "desc": "Disco prensado na mureta superior; descolagem e arrancada",
        "robot": (-20.0, 15.0, 0.3),
        "puck": (0.0, 35.0),
    },
    {
        "num": 7,
        "title": "Parede Lateral Sul (Rente ao Alumínio)",
        "desc": "Aproximacao em alta velocidade pela borda inferior",
        "robot": (-25.0, -15.0, -0.3),
        "puck": (15.0, -35.0),
    },
    {
        "num": 8,
        "title": "Defesa de Emergencia na Boca do Gol",
        "desc": "Disco a centimetros da linha do gol azul; corte cirurgico para ataque",
        "robot": (-55.0, 16.0, -math.pi / 2),
        "puck": (-70.0, 4.0),
    },
    {
        "num": 9,
        "title": "Ataque Cruzado em 90° (Perpendicular)",
        "desc": "Abordagem transversal em alta velocidade com curva em L a 350 RPM",
        "robot": (-10.0, 22.0, -math.pi / 2),
        "puck": (-10.0, -12.0),
    },
    {
        "num": 10,
        "title": "Arrancada Ponta a Ponta (Full Pitch Sprint)",
        "desc": "Sprint maximo de 135 cm a 350 RPM com retencao e gol limpo",
        "robot": (-68.0, 0.0, 0.0),
        "puck": (-50.0, 0.0),
    },
]


def create_video():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    pygame.init()
    pygame.font.init()

    WIDTH, HEIGHT = 1280, 720
    FPS = 30
    surface = pygame.Surface((WIDTH, HEIGHT))

    # Fonts
    font_title = pygame.font.SysFont("Segoe UI", 26, bold=True)
    font_sub = pygame.font.SysFont("Segoe UI", 18)
    font_telemetry = pygame.font.SysFont("Consolas", 16)
    font_badge = pygame.font.SysFont("Segoe UI", 16, bold=True)
    font_goal = pygame.font.SysFont("Segoe UI", 48, bold=True)
    font_goal_sub = pygame.font.SysFont("Segoe UI", 22)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mp4_path = OUTPUT_DIR / "10_situacoes_hockeybot_350rpm.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, float(FPS), (WIDTH, HEIGHT))

    # Load policy
    trainer = HockeyPPOTrainer()
    trainer.load(str(CHECKPOINT))
    trainer.net.eval()

    # Environment
    env = RoboCoreHockeyEnv(profile="micro-practice", robots_per_team=1, opponent_robots=0, scenario="solo_carry")
    arena = env.arena

    # Arena projection rect
    arena_rect = pygame.Rect(40, 100, WIDTH - 80, HEIGHT - 180)
    scale = min(
        (arena_rect.width - 60) / (arena.length + 2 * arena.goal_depth),
        (arena_rect.height - 40) / arena.width,
    )
    cx, cy = arena_rect.center

    def world(x, y):
        return round(cx + x * scale), round(cy - y * scale)

    all_frames = []

    print(f"=== Generating Video for 10 Difficult Situations ({WIDTH}x{HEIGHT} @ {FPS}fps) ===")

    for sit in SITUATIONS:
        num = sit["num"]
        title = sit["title"]
        desc = sit["desc"]
        rx, ry, th = sit["robot"]
        px, py = sit["puck"]

        print(f"Recording Situation {num}/10: {title} ...")

        # Setup initial positions
        env.reset(seed=100 + num)
        r = env.engine.blue_robots[0]
        env.engine.place_robot(r, rx, ry, th)
        env.engine.place_puck(px, py)
        env.vision_sensor.reset()
        env.vision_sensor.update(env.engine, prime=True)
        obs = env.get_team_obs(0)

        step = 0
        max_steps = 450
        goal_scored = False
        goal_time = 0.0
        final_hold_frames = 0
        max_hold = 36  # ~1.2s celebration hold

        while step < max_steps:
            if not goal_scored:
                act = trainer.select_action(obs, deterministic=True)[0]
                obs, rew, term, trunc, info = env.step(act, None)
                step += 1
                if info["goal_scored"] == 1:
                    goal_scored = True
                    goal_time = env.engine.episode_time
            else:
                final_hold_frames += 1
                if final_hold_frames >= max_hold:
                    break

            # Render Frame
            surface.fill((10, 16, 27))  # Dark professional tech background

            # --- HEADER HUD ---
            # Badge
            badge_text = f"SITUACAO {num:02d}/10"
            badge_surf = font_badge.render(badge_text, True, (255, 255, 255))
            badge_bg = pygame.Rect(40, 18, badge_surf.get_width() + 18, 30)
            pygame.draw.rect(surface, (34, 113, 217), badge_bg, border_radius=6)
            surface.blit(badge_surf, (49, 23))

            # Title
            title_surf = font_title.render(title, True, (240, 246, 252))
            surface.blit(title_surf, (badge_bg.right + 15, 17))

            # Subtitle / Description
            desc_surf = font_sub.render(desc, True, (139, 155, 175))
            surface.blit(desc_surf, (40, 56))

            # Telemetry top right
            speed_cm_s = math.hypot(r.body.velocity.x, r.body.velocity.y)
            cur_time = goal_time if goal_scored else env.engine.episode_time
            telem_text = f"T: {cur_time:4.2f}s | V: {speed_cm_s:5.1f} cm/s | 350 RPM (2S)"
            telem_surf = font_telemetry.render(telem_text, True, (88, 166, 255))
            surface.blit(telem_surf, (WIDTH - telem_surf.get_width() - 40, 25))

            # --- ARENA VIEWPORT (VISÃO COMPUTACIONAL ATIVA) ---
            frame_data = frame_from_env(env)
            draw_arena(
                surface,
                arena_rect,
                arena,
                env.engine.robot_config,
                frame_data,
                font_sub,
                selected=0,
                tracking=True,
            )

            # --- FOOTER BAR (TELEMETRIA DE VISÃO COMPUTACIONAL) ---
            footer_text = (
                "Visão Computacional: 30 FPS | Rastreamento ArUco ID 0 (Círculo Verde) | "
                "Filtro de Latência & Posição Ativo | Ações Determinísticas PPO 350 RPM"
            )
            footer_surf = font_sub.render(footer_text, True, (0, 200, 150))
            surface.blit(footer_surf, (40, HEIGHT - 45))

            # --- GOAL OVERLAY BANNER ---
            if goal_scored:
                overlay_box = pygame.Rect(WIDTH // 2 - 220, HEIGHT // 2 - 60, 440, 110)
                pygame.draw.rect(surface, (13, 17, 23), overlay_box, border_radius=16)
                pygame.draw.rect(surface, (46, 160, 67), overlay_box, width=3, border_radius=16)

                goal_txt = font_goal.render("GOL!", True, (46, 160, 67))
                surface.blit(goal_txt, goal_txt.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))

                sub_txt = font_goal_sub.render(f"Finalizado em {goal_time:.2f}s", True, (240, 246, 252))
                surface.blit(sub_txt, sub_txt.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 25)))

            # Convert to OpenCV frame (BGR)
            frame_rgb = pygame.image.tobytes(surface, "RGB")
            img_bgr = cv2.cvtColor(np.frombuffer(frame_rgb, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)), cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)

            # Downsample for preview GIF (every 2nd frame)
            if step % 2 == 0 or goal_scored:
                all_frames.append(cv2.resize(img_bgr, (640, 360)))

        print(f"Situation {num}/10 completed in {goal_time:.2f}s!")

    writer.release()
    print(f"\nMP4 Video successfully written to: {mp4_path} ({mp4_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Copy MP4 to Artifact Directory
    if ARTIFACT_DIR.exists():
        shutil.copy2(mp4_path, ARTIFACT_DIR / "10_situacoes_hockeybot_350rpm.mp4")
        print(f"Copied MP4 to {ARTIFACT_DIR}")

    # Generate an optimized preview GIF (first 3 situations combined as sample for UI)
    print("Generating highlight GIF for chat UI...")
    gif_path = OUTPUT_DIR / "10_situacoes_hockeybot_350rpm.gif"
    from PIL import Image

    # Sample frames across the situations
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in all_frames[::3]]
    pil_frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=70,
        loop=0,
        optimize=True,
    )
    print(f"Highlight GIF written to: {gif_path} ({gif_path.stat().st_size / 1024 / 1024:.2f} MB)")

    if ARTIFACT_DIR.exists():
        shutil.copy2(gif_path, ARTIFACT_DIR / "10_situacoes_hockeybot_350rpm.gif")
        print(f"Copied GIF to {ARTIFACT_DIR}")

    print("\n=== ALL VIDEO TASKS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    create_video()
