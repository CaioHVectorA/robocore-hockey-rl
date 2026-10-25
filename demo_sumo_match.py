"""Script to simulate and record Mini-Sumo 500g autonomous matches."""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from sumo.bots import BullRushBot, MatadorBot, SearchAndDestroyBot
from sumo.env import SumoEnv
from sumo.renderer import SumoRenderer


def record_sumo_showcase(video_path: str = "videos/sumo_match_demo.mp4", gif_path: str = "videos/sumo_match_demo.gif"):
    os.makedirs("videos", exist_ok=True)

    rounds = [
        {
            "name": "Round 1: Frente a Frente (Standard)",
            "layout": "standard",
            "blue_bot": SearchAndDestroyBot(0, search_direction=1.0),
            "red_bot": BullRushBot(1),
            "blue_title": "Blue: Search & Destroy",
            "red_title": "Red: BullRush",
        },
        {
            "name": "Round 2: De Costas (Back to Back)",
            "layout": "back_to_back",
            "blue_bot": SearchAndDestroyBot(0, search_direction=1.0),
            "red_bot": SearchAndDestroyBot(1, search_direction=-1.0),
            "blue_title": "Blue: Search & Destroy",
            "red_title": "Red: Search & Destroy",
        },
        {
            "name": "Round 3: Esquiva Tática (Matador vs BullRush)",
            "layout": "angled",
            "blue_bot": MatadorBot(0),
            "red_bot": BullRushBot(1),
            "blue_title": "Blue: Matador Bot",
            "red_title": "Red: BullRush",
        },
    ]

    renderer = SumoRenderer(width=1280, height=720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    writer = cv2.VideoWriter(video_path, fourcc, fps, (1280, 720))

    gif_frames = []
    print(f"Iniciando gravação do Mini-Sumo em {video_path}...")

    total_frames = 0
    for r_idx, r_info in enumerate(rounds):
        env = SumoEnv(opponent=r_info["red_bot"], max_episode_seconds=8.0)
        obs, _ = env.reset(options={"layout": r_info["layout"]})
        blue_bot = r_info["blue_bot"]

        winner_name = None
        done = False
        frame_in_round = 0

        title = f"Mini-Sumo 500g · {r_info['name']}"
        subtitle = f"{r_info['blue_title']}  VS  {r_info['red_title']}"

        print(f"--- {title} ---")

        while not done and frame_in_round < 240:
            frame_in_round += 1
            # Blue action from its bot controller
            blue_act = blue_bot.get_action(
                env.engine.robot_blue, env.engine.robot_red, env.sensors_blue
            )
            act_arr = np.array(blue_act, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(act_arr)
            done = terminated or truncated

            if terminated:
                res = info.get("result", "draw")
                if res == "win":
                    winner_name = "VITÓRIA AZUL (BLUE WINS!)"
                elif res == "loss":
                    winner_name = "VITÓRIA VERMELHO (RED WINS!)"
                else:
                    winner_name = "EMPATE TÉCNICO (DRAW)"

            frame = renderer.draw_frame(env, title=title, subtitle=subtitle, winner_text=winner_name)
            writer.write(frame)
            total_frames += 1

            # Sample for GIF (every 2nd frame at half resolution)
            if total_frames % 2 == 0:
                small = cv2.resize(frame, (640, 360))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                gif_frames.append(Image.fromarray(rgb))

        # Hold winner frame for 1.2 seconds (36 frames)
        if winner_name:
            for _ in range(36):
                frame = renderer.draw_frame(env, title=title, subtitle=subtitle, winner_text=winner_name)
                writer.write(frame)
                total_frames += 1
                if total_frames % 2 == 0:
                    small = cv2.resize(frame, (640, 360))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    gif_frames.append(Image.fromarray(rgb))

    writer.release()
    print(f"Vídeo gravado com sucesso! Total frames: {total_frames} -> {video_path}")

    # Save GIF
    if gif_frames and gif_path:
        print(f"Salvando animação GIF ({len(gif_frames)} frames)...")
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=66,  # ~15 fps
            loop=0,
            optimize=True,
        )
        print(f"GIF salvo em {gif_path} ({os.path.getsize(gif_path) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    record_sumo_showcase()
