"""Hybrid Self-Play + Competitive Scripted League PPO Trainer for Mini-Sumo 500g."""

from __future__ import annotations
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False

from sumo.bots import (
    BaseSumoBot,
    BullRushBot,
    MatadorBot,
    NeuralPolicyBot,
    SearchAndDestroyBot,
    TawaraGaeshiBot,
    WavePendulumBot,
)
from sumo.constants import CHASSIS_ARCHETYPES, SUMO_ACT_DIM, SUMO_OBS_DIM
from sumo.env import LAYOUT_CHOICES, SumoEnv
from sumo.policy import SumoPolicyNet


@dataclass
class SumoPPOConfig:
    total_updates: int = 120
    rollout_steps: int = 512
    num_envs: int = 6
    batch_size: int = 256
    ppo_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.008
    max_grad_norm: float = 0.5
    selfplay_prob: float = 0.35  # 35% Self-Play snapshots, 65% Competitive Sensor Bots
    snapshot_interval: int = 20
    max_pool_size: int = 8
    bc_imitation_coef: float = 0.15  # Sensor-bot tactical guidance annealed over early training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints/sumo"
    log_dir: str = "runs/sumo_ppo"


def compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation across rollout time steps."""
    t_steps = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(t_steps)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def expert_sensor_teacher_action(obs_batch: np.ndarray) -> np.ndarray:
    """Compute reference sensor-based competitive actions from 24D observation batch."""
    actions = np.zeros((len(obs_batch), 2), dtype=np.float32)
    for i, obs in enumerate(obs_batch):
        line = obs[0:4]
        dist = obs[4:11]
        last_dir = float(obs[11])
        in_contact = obs[17] > 0.5

        if (line[0] > 0.35 or line[1] > 0.35) and not (in_contact and dist[3] < 0.14):
            actions[i] = (-1.0, -0.6) if line[0] >= line[1] else (-0.6, -1.0)
        elif line[2] > 0.40 or line[3] > 0.40:
            actions[i] = (1.0, 1.0)
        else:
            actions[i] = BaseSumoBot._track_with_7_sensors(dist, last_dir, spin_speed=0.75)
    return actions


def train_sumo(cfg: SumoPPOConfig | None = None) -> SumoPolicyNet:
    cfg = cfg or SumoPPOConfig()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print("==================================================================")
    print(" Mini-Sumo 500g RL · Liga Híbrida (Bots de Sensores + Self-Play)")
    print(f" Device: {cfg.device} | Envs: {cfg.num_envs} | Steps/Env: {cfg.rollout_steps}")
    print(f" Updates: {cfg.total_updates} | Self-Play Ratio: {cfg.selfplay_prob*100:.0f}%")
    print("==================================================================")

    scripted_league: List[BaseSumoBot] = [
        SearchAndDestroyBot(1, search_direction=1.0),
        BullRushBot(1),
        MatadorBot(1, dodge_side=1.0),
        WavePendulumBot(1),
        TawaraGaeshiBot(1),
        SearchAndDestroyBot(1, search_direction=-1.0),
    ]
    chassis_list = list(CHASSIS_ARCHETYPES.values())
    selfplay_pool: List[NeuralPolicyBot] = []

    envs = [
        SumoEnv(
            blue_config=CHASSIS_ARCHETYPES["haryama"],
            red_config=chassis_list[i % len(chassis_list)],
            opponent=copy.deepcopy(scripted_league[i % len(scripted_league)]),
            randomize_matchups=False,
        )
        for i in range(cfg.num_envs)
    ]

    policy = SumoPolicyNet(SUMO_OBS_DIM, SUMO_ACT_DIM).to(cfg.device)
    writer = SummaryWriter(log_dir=cfg.log_dir) if HAS_TENSORBOARD else None
    rng = np.random.default_rng(42)

    # Warm-start actor with competitive 7-sensor & border reflexes before PPO fine-tuning
    if cfg.bc_imitation_coef > 0.0:
        demo_obs = []
        for i, env in enumerate(envs):
            o, _ = env.reset(options={"layout": LAYOUT_CHOICES[i % len(LAYOUT_CHOICES)], "random_heading": True})
            for _ in range(400):
                demo_obs.append(o)
                t_act = expert_sensor_teacher_action(o[None, :])[0]
                o, _, term, trunc, _ = env.step(t_act)
                if term or trunc:
                    o, _ = env.reset(options={"layout": str(rng.choice(LAYOUT_CHOICES)), "random_heading": True})
        demo_obs_np = np.array(demo_obs, dtype=np.float32)
        demo_act_np = expert_sensor_teacher_action(demo_obs_np) * 1.15
        t_x = torch.as_tensor(demo_obs_np, dtype=torch.float32, device=cfg.device)
        t_y = torch.as_tensor(demo_act_np, dtype=torch.float32, device=cfg.device)
        bc_opt = torch.optim.Adam(policy.actor.parameters(), lr=4e-3)
        policy.train()
        for _ in range(450):
            pred = policy.actor(t_x)
            loss = F.mse_loss(pred, t_y)
            bc_opt.zero_grad()
            loss.backward()
            bc_opt.step()
        with torch.no_grad():
            policy.log_std.fill_(-1.8)

    optimizer = torch.optim.Adam(
        [
            {"params": policy.actor.parameters(), "lr": cfg.lr * 0.2},
            {"params": [policy.log_std], "lr": cfg.lr * 0.1},
            {"params": policy.critic.parameters(), "lr": cfg.lr},
        ],
        eps=1e-5,
    )

    obs_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, SUMO_OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, SUMO_ACT_DIM), dtype=np.float32)
    logp_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    rew_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    val_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    done_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)

    current_obs = np.array(
        [env.reset(options={"layout": LAYOUT_CHOICES[i % len(LAYOUT_CHOICES)]})[0] for i, env in enumerate(envs)],
        dtype=np.float32,
    )

    win_count = 0
    loss_count = 0
    draw_count = 0
    global_step = 0
    start_time = time.time()

    for update in range(1, cfg.total_updates + 1):
        policy.eval()

        # Add snapshot to self-play pool periodically
        if update == 1 or update % cfg.snapshot_interval == 0:
            snap_net = SumoPolicyNet(SUMO_OBS_DIM, SUMO_ACT_DIM).to("cpu")
            snap_net.load_state_dict({k: v.cpu() for k, v in policy.state_dict().items()})
            snap_net.eval()
            selfplay_pool.append(NeuralPolicyBot(snap_net, bot_id=1, deterministic=False))
            if len(selfplay_pool) > cfg.max_pool_size:
                selfplay_pool.pop(0)

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

                    # Sample next matchup from Hybrid League (Scripted Bots + Self-Play) & 5 Chassis
                    if selfplay_pool and rng.random() < cfg.selfplay_prob:
                        opp = copy.deepcopy(selfplay_pool[int(rng.integers(0, len(selfplay_pool)))])
                    else:
                        opp = copy.deepcopy(scripted_league[int(rng.integers(0, len(scripted_league)))])
                    red_chassis = chassis_list[int(rng.integers(0, len(chassis_list)))]
                    blue_chassis = chassis_list[int(rng.integers(0, len(chassis_list)))]
                    layout = str(rng.choice(LAYOUT_CHOICES))
                    env.set_opponent(opp, red_config=red_chassis)
                    o, _ = env.reset(
                        options={
                            "layout": layout,
                            "blue_chassis": blue_chassis,
                            "red_chassis": red_chassis,
                            "random_heading": True,
                        }
                    )

                next_obs_list.append(o)

            current_obs = np.array(next_obs_list, dtype=np.float32)
            global_step += cfg.num_envs

        with torch.no_grad():
            t_obs = torch.as_tensor(current_obs, dtype=torch.float32, device=cfg.device)
            next_values = policy.critic(t_obs).squeeze(-1).cpu().numpy()

        next_val_buf = np.zeros_like(val_buf)
        next_val_buf[:-1] = val_buf[1:]
        next_val_buf[-1] = next_values

        adv, returns = compute_gae(
            rew_buf, val_buf, next_val_buf, done_buf, cfg.gamma, cfg.gae_lambda
        )

        b_obs_np = obs_buf.reshape(-1, SUMO_OBS_DIM)
        b_obs = torch.as_tensor(b_obs_np, dtype=torch.float32, device=cfg.device)
        b_act = torch.as_tensor(act_buf.reshape(-1, SUMO_ACT_DIM), dtype=torch.float32, device=cfg.device)
        b_logp = torch.as_tensor(logp_buf.reshape(-1), dtype=torch.float32, device=cfg.device)
        b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=cfg.device)
        b_ret = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=cfg.device)

        # Annealed tactical teacher guidance so policy quickly masters sensor lock & border reflexes
        bc_weight = cfg.bc_imitation_coef * max(0.0, 1.0 - (update - 1) / max(1.0, cfg.total_updates * 0.65))
        if bc_weight > 0.0:
            teacher_np = expert_sensor_teacher_action(b_obs_np)
            b_teacher = torch.as_tensor(teacher_np, dtype=torch.float32, device=cfg.device)
        else:
            b_teacher = None

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        policy.train()
        dataset_size = b_obs.shape[0]
        indices = np.arange(dataset_size)

        pi_losses, v_losses, ent_losses, kls = [], [], [], []

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, cfg.batch_size):
                mb_idx = indices[start : start + cfg.batch_size]
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

                if b_teacher is not None and bc_weight > 0.0:
                    pred_mean = policy.actor(mb_obs)
                    bc_loss = F.mse_loss(pred_mean, b_teacher[mb_idx])
                    total_loss = total_loss + bc_weight * bc_loss

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

        if update % 10 == 0 or update == 1 or update == cfg.total_updates:
            fps = global_step / max(1e-3, time.time() - start_time)
            print(
                f"Update {update:3d}/{cfg.total_updates} | "
                f"Steps: {global_step:7d} | "
                f"FPS: {fps:4.0f} | "
                f"Win Rate: {win_rate:5.1f}% ({win_count}W/{loss_count}L/{draw_count}D) | "
                f"KL: {np.mean(kls):.4f}"
            )
            # Reset rolling window counters every 20 updates for fresh recent win rate
            if update % 20 == 0 and update < cfg.total_updates:
                win_count = loss_count = draw_count = 0

            if writer is not None:
                writer.add_scalar("sumo/win_rate", win_rate, global_step)
                writer.add_scalar("sumo/pi_loss", np.mean(pi_losses), global_step)
                writer.add_scalar("sumo/v_loss", np.mean(v_losses), global_step)
                writer.add_scalar("sumo/approx_kl", np.mean(kls), global_step)

        if update % 40 == 0 or update == cfg.total_updates:
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
            torch.save(policy.state_dict(), Path(cfg.checkpoint_dir) / "sumo_policy_latest.pt")

    if writer is not None:
        writer.close()
    print("Treinamento Mini-Sumo Liga Híbrida finalizado!")
    return policy


if __name__ == "__main__":
    train_sumo()
