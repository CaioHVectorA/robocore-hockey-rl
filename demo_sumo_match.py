"""Tournament showcase recorder for Mini-Sumo 500g: 5 Chassis Archetypes, Sensor Bots & PPO Self-Play."""

import os
import cv2
import numpy as np
from PIL import Image
import torch

from sumo.bots import (
    BullRushBot,
    MatadorBot,
    NeuralPolicyBot,
    SearchAndDestroyBot,
    TawaraGaeshiBot,
    WavePendulumBot,
)
from sumo.constants import CHASSIS_ARCHETYPES, SUMO_ACT_DIM, SUMO_OBS_DIM
from sumo.env import SumoEnv
from sumo.policy import SumoPolicyNet
from sumo.renderer import SumoRenderer


def load_trained_policy(ckpt_path: str = "checkpoints/sumo/sumo_policy_latest.pt") -> SumoPolicyNet:
    policy = SumoPolicyNet(SUMO_OBS_DIM, SUMO_ACT_DIM)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            policy.load_state_dict(state["model_state_dict"])
        else:
            policy.load_state_dict(state)
    policy.eval()
    return policy


def record_sumo_showcase(
    video_path: str = "videos/sumo_match_demo.mp4",
    gif_path: str = "videos/sumo_match_demo.gif",
):
    os.makedirs("videos", exist_ok=True)
    policy = load_trained_policy()

    rounds = [
        {
            "name": "Round 1 · Frente a Frente (Garfo 18° vs Monobloco 4WD)",
            "layout": "standard",
            "blue_chassis": CHASSIS_ARCHETYPES["muramasa_fork"],
            "red_chassis": CHASSIS_ARCHETYPES["ikazuchi_4wd"],
            "blue_bot": NeuralPolicyBot(policy, bot_id=0, deterministic=True),
            "red_bot": BullRushBot(1),
            "blue_role": "AZUL · PPO Neural (Muramasa Garfo)",
            "red_role": "VERM · BullRush (Ikazuchi 4WD)",
        },
        {
            "name": "Round 2 · Esquiva Matador (Kamikaze Velox vs BullRush 4WD)",
            "layout": "standard",
            "blue_chassis": CHASSIS_ARCHETYPES["kamikaze_prow"],
            "red_chassis": CHASSIS_ARCHETYPES["ikazuchi_4wd"],
            "blue_bot": MatadorBot(0, dodge_side=-1.0),
            "red_bot": BullRushBot(1),
            "blue_role": "AZUL · Matador Flanker (Kamikaze)",
            "red_role": "VERM · BullRush (Ikazuchi 4WD)",
        },
        {
            "name": "Round 3 · De Costas (Haryama V3 vs Onimaru Blindado)",
            "layout": "back_to_back",
            "blue_chassis": CHASSIS_ARCHETYPES["haryama"],
            "red_chassis": CHASSIS_ARCHETYPES["onimaru_tank"],
            "blue_bot": SearchAndDestroyBot(0, search_direction=1.0),
            "red_bot": TawaraGaeshiBot(1),
            "blue_role": "AZUL · Karioka S&D (Haryama V3)",
            "red_role": "VERM · Tawara-Gaeshi (Onimaru)",
        },
        {
            "name": "Round 4 · Lado a Lado (PPO Neural vs Cobra Pendulum)",
            "layout": "side_by_side",
            "blue_chassis": CHASSIS_ARCHETYPES["haryama"],
            "red_chassis": CHASSIS_ARCHETYPES["kamikaze_prow"],
            "blue_bot": NeuralPolicyBot(policy, bot_id=0, deterministic=True),
            "red_bot": WavePendulumBot(1),
            "blue_role": "AZUL · PPO Neural (Haryama V3)",
            "red_role": "VERM · Cobra Pendulum (Kamikaze)",
        },
        {
            "name": "Round 5 · Diagonal Self-Play (Onimaru vs Muramasa Garfo)",
            "layout": "angled",
            "blue_chassis": CHASSIS_ARCHETYPES["onimaru_tank"],
            "red_chassis": CHASSIS_ARCHETYPES["muramasa_fork"],
            "blue_bot": NeuralPolicyBot(policy, bot_id=0, deterministic=True),
            "red_bot": SearchAndDestroyBot(1, search_direction=-1.0),
            "blue_role": "AZUL · PPO Neural (Onimaru Blindado)",
            "red_role": "VERM · Karioka S&D (Muramasa Garfo)",
        },
    ]

    renderer = SumoRenderer(width=1280, height=720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    writer = cv2.VideoWriter(video_path, fourcc, fps, (1280, 720))

    gif_frames = []
    total_frames = 0

    for r_info in rounds:
        env = SumoEnv(
            blue_config=r_info["blue_chassis"],
            red_config=r_info["red_chassis"],
            opponent=r_info["red_bot"],
            max_episode_seconds=6.0,
        )
        obs, _ = env.reset(
            options={
                "layout": r_info["layout"],
                "blue_chassis": r_info["blue_chassis"],
                "red_chassis": r_info["red_chassis"],
            }
        )
        blue_bot = r_info["blue_bot"]
        blue_bot.reset()

        title = f"Mini-Sumo 500g · {r_info['name']}"
        subtitle = (
            f"{r_info['blue_chassis'].name} ({r_info['blue_chassis'].subtitle})  VS  "
            f"{r_info['red_chassis'].name} ({r_info['red_chassis'].subtitle})"
        )

        winner_name = None
        done = False
        step_in_round = 0

        while not done and step_in_round < 180:
            step_in_round += 1
            blue_act = blue_bot.get_action(
                env.engine.robot_blue, env.engine.robot_red, env.sensors_blue
            )
            act_arr = np.array(blue_act, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(act_arr)
            done = terminated or truncated

            if terminated:
                res = info.get("result", "draw")
                if res == "win":
                    winner_name = f"VITÓRIA AZUL · {r_info['blue_chassis'].name.upper()}!"
                elif res == "loss":
                    winner_name = f"VITÓRIA VERMELHO · {r_info['red_chassis'].name.upper()}!"
                else:
                    winner_name = "EMPATE TÉCNICO!"

            frame = renderer.draw_frame(
                env,
                title=title,
                subtitle=subtitle,
                blue_role=r_info["blue_role"],
                red_role=r_info["red_role"],
                winner_text=winner_name,
            )
            writer.write(frame)
            total_frames += 1

            if total_frames % 2 == 0:
                small = cv2.resize(frame, (640, 360))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                gif_frames.append(Image.fromarray(rgb))

        print(
            f"  [{r_info['name']}] Finalizado em {env.engine.time:.2f}s "
            f"({step_in_round} frames) -> {winner_name or 'Timeout'}"
        )

        # Hold victory banner for 24 frames (0.8s)
        if winner_name:
            for _ in range(24):
                frame = renderer.draw_frame(
                    env,
                    title=title,
                    subtitle=subtitle,
                    blue_role=r_info["blue_role"],
                    red_role=r_info["red_role"],
                    winner_text=winner_name,
                )
                writer.write(frame)
                total_frames += 1
                if total_frames % 2 == 0:
                    small = cv2.resize(frame, (640, 360))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    gif_frames.append(Image.fromarray(rgb))

    writer.release()
    print(f"Vídeo HD gravado: {video_path} ({total_frames} frames)")

    if gif_frames and gif_path:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=66,
            loop=0,
            optimize=True,
        )
        print(f"GIF gravado: {gif_path} ({os.path.getsize(gif_path) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    record_sumo_showcase()
