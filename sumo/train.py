"""PPO Training pipeline for Mini-Sumo 500g autonomous combat."""

from dataclasses import dataclass
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False

from sumo.bots import BullRushBot, MatadorBot, SearchAndDestroyBot
from sumo.constants import SUMO_ACT_DIM, SUMO_OBS_DIM
from sumo.env import SumoEnv
from sumo.policy import SumoPolicyNet


@dataclass
class SumoPPOConfig:
    total_updates: int = 400
    rollout_steps: int = 1024
    num_envs: int = 4
    batch_size: int = 256
    ppo_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints/sumo"
    log_dir: str = "runs/sumo_ppo"


def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation across time."""
    t_steps = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(t_steps)):
        next_val = next_values[t]
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train_sumo(cfg: SumoPPOConfig | None = None):
    cfg = cfg or SumoPPOConfig()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print("==================================================")
    print(" Mini-Sumo 500g RL Training (PPO)")
    print(f" Device: {cfg.device} | Num Envs: {cfg.num_envs} | Steps per Env: {cfg.rollout_steps}")
    print(f" Batch Size: {cfg.batch_size} | Total Updates: {cfg.total_updates}")
    print(f" Checkpoints: {cfg.checkpoint_dir}")
    print("==================================================")

    # Opponent pool for curriculum
    sparring_opponents = [
        ("BullRush", BullRushBot(1)),
        ("SearchAndDestroy", SearchAndDestroyBot(1, search_direction=1.0)),
        ("Matador", MatadorBot(1)),
    ]

    # Create vector environments
    envs = [
        SumoEnv(opponent=sparring_opponents[i % len(sparring_opponents)][1])
        for i in range(cfg.num_envs)
    ]

    policy = SumoPolicyNet(SUMO_OBS_DIM, SUMO_ACT_DIM).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    writer = SummaryWriter(log_dir=cfg.log_dir) if HAS_TENSORBOARD else None

    # Storage buffers
    obs_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, SUMO_OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, SUMO_ACT_DIM), dtype=np.float32)
    logp_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    rew_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    val_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    done_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)

    current_obs = np.array([env.reset()[0] for env in envs], dtype=np.float32)

    win_count = 0
    loss_count = 0
    draw_count = 0
    global_step = 0
    start_time = time.time()

    for update in range(1, cfg.total_updates + 1):
        policy.eval()

        # Collect rollouts
        for step in range(cfg.rollout_steps):
            t_obs = torch.as_tensor(current_obs, dtype=torch.float32, device=cfg.device)
            with torch.no_grad():
                dist = policy.distribution(t_obs)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                value = policy.critic(t_obs).squeeze(-1)

            act_clamped = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            obs_buf[step] = current_obs
            act_buf[step] = act_clamped
            logp_buf[step] = log_prob.cpu().numpy()
            val_buf[step] = value.cpu().numpy()

            # Step all envs
            next_obs_list = []
            for e_idx, env in enumerate(envs):
                o, r, terminated, truncated, info = env.step(act_clamped[e_idx])
                rew_buf[step, e_idx] = r
                done = terminated or truncated
                done_buf[step, e_idx] = float(done)

                if done:
                    res = info.get("result", "draw")
                    if res == "win":
                        win_count += 1
                    elif res == "loss":
                        loss_count += 1
                    else:
                        draw_count += 1
                    o, _ = env.reset()

                next_obs_list.append(o)

            current_obs = np.array(next_obs_list, dtype=np.float32)
            global_step += cfg.num_envs

        # Bootstrap next values for GAE
        with torch.no_grad():
            t_obs = torch.as_tensor(current_obs, dtype=torch.float32, device=cfg.device)
            next_values = policy.critic(t_obs).squeeze(-1).cpu().numpy()

        next_val_buf = np.zeros_like(val_buf)
        next_val_buf[:-1] = val_buf[1:]
        next_val_buf[-1] = next_values

        adv, returns = compute_gae(
            rew_buf, val_buf, next_val_buf, done_buf, cfg.gamma, cfg.gae_lambda
        )

        # Flatten buffers for PPO updates
        b_obs = torch.as_tensor(obs_buf.reshape(-1, SUMO_OBS_DIM), dtype=torch.float32, device=cfg.device)
        b_act = torch.as_tensor(act_buf.reshape(-1, SUMO_ACT_DIM), dtype=torch.float32, device=cfg.device)
        b_logp = torch.as_tensor(logp_buf.reshape(-1), dtype=torch.float32, device=cfg.device)
        b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=cfg.device)
        b_ret = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=cfg.device)

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Optimization epochs
        policy.train()
        dataset_size = b_obs.shape[0]
        indices = np.arange(dataset_size)

        pi_losses, v_losses, ent_losses, kls = [], [], [], []

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, cfg.batch_size):
                end = start + cfg.batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logp = b_logp[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]

                dist = policy.distribution(mb_obs)
                new_logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                new_val = policy.critic(mb_obs).squeeze(-1)

                ratio = torch.exp(new_logp - mb_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_val, mb_ret)
                total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()

                pi_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                ent_losses.append(entropy.item())
                kls.append(approx_kl)

        total_matches = win_count + loss_count + draw_count
        win_rate = (win_count / total_matches * 100.0) if total_matches > 0 else 0.0

        if update % 5 == 0 or update == 1:
            fps = global_step / max(1e-3, time.time() - start_time)
            print(
                f"Update {update:3d}/{cfg.total_updates} | "
                f"Steps: {global_step:7d} | "
                f"FPS: {fps:5.0f} | "
                f"Win Rate: {win_rate:5.1f}% ({win_count}W/{loss_count}L/{draw_count}D) | "
                f"Pi Loss: {np.mean(pi_losses):.4f} | "
                f"V Loss: {np.mean(v_losses):.4f} | "
                f"KL: {np.mean(kls):.4f}"
            )

            # Tensorboard
            if writer is not None:
                writer.add_scalar("sumo/win_rate", win_rate, global_step)
                writer.add_scalar("sumo/pi_loss", np.mean(pi_losses), global_step)
                writer.add_scalar("sumo/v_loss", np.mean(v_losses), global_step)
                writer.add_scalar("sumo/entropy", np.mean(ent_losses), global_step)
                writer.add_scalar("sumo/approx_kl", np.mean(kls), global_step)

        # Save checkpoint periodically
        if update % 50 == 0 or update == cfg.total_updates:
            ckpt_path = Path(cfg.checkpoint_dir) / f"sumo_ppo_update_{update:04d}.pt"
            torch.save(
                {
                    "update": update,
                    "global_step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "win_rate": win_rate,
                },
                ckpt_path,
            )
            # Latest symlink/copy
            torch.save(policy.state_dict(), Path(cfg.checkpoint_dir) / "sumo_policy_latest.pt")
            print(f"  -> Checkpoint salvo em {ckpt_path}")

    if writer is not None:
        writer.close()
    print("Treinamento Mini-Sumo finalizado!")


if __name__ == "__main__":
    train_sumo()
