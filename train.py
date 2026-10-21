"""
Script de Treinamento e Benchmark CLI para o Hockey de Robôs RoboCore (Classe 1lb).
Treina a política PPO contra o bot analítico especialista e salva os pesos em checkpoints/.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.constants import HockeyConstants as C
from rl.env import RoboCoreHockeyEnv
from rl.ppo_trainer import HockeyPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino PPO Multi-Agente para Hockey de Robôs RoboCore (3v3)")
    parser.add_argument("--steps", type=int, default=30000, help="Passos totais de treino")
    parser.add_argument("--save", type=str, default="hockey_ppo.pt", help="Nome do arquivo de checkpoint")
    args = parser.parse_args()

    env = RoboCoreHockeyEnv(robots_per_team=C.ROBOTS_PER_TEAM, max_steps=900)
    trainer = HockeyPPOTrainer(obs_dim=env.obs_dim, act_dim=env.act_dim)

    print(f"=== Iniciando Treinamento Multi-Agente PPO — Hockey 3v3 Oficial (6 Robôs 1lb) ===")
    print(f"Arena: 180x120 cm com cantos arredondados (R=20 cm) | Puck: atrito Coulomb mu_k = 0.20")
    print(f"Equipes: 3 Robôs por time (Atacante, Ala/Defensor, Goleiro) com placa Vespa ESP32")

    obs, _ = env.reset()
    ep_rew = 0.0
    ep_rewards = []

    for step in range(1, args.steps + 1):
        act, logp, val = trainer.select_action(obs, deterministic=False)
        next_obs, rew, term, trunc, info = env.step(act)
        done = term or trunc
        trainer.store_transition(obs, act, logp, info.get("individual_rewards", rew), val, done)
        ep_rew += rew
        obs = next_obs

        if len(trainer.rew_buf) >= 1024:
            trainer.update()

        if done:
            ep_rewards.append(ep_rew)
            ep_rew = 0.0
            obs, _ = env.reset()

        if step % 5000 == 0:
            mean_r = float(np.mean(ep_rewards[-20:])) if ep_rewards else 0.0
            print(
                f"[Step {step:6d}/{args.steps}] Recompensa Média: {mean_r:+7.2f} | "
                f"Placar (Azul {env.engine.score_blue} x {env.engine.score_red} Vermelho) | "
                f"Loss: {trainer.last_loss:.4f}"
            )

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, args.save)
    trainer.save(save_path)
    print(f"\nTreino concluído com sucesso! Pesos salvos em: {save_path}")


if __name__ == "__main__":
    main()
