"""
Treinador PPO (Proximal Policy Optimization) para o Hockey de Robôs RoboCore.
Controla continuamente a velocidade das duas rodas da placa Vespa (u_left, u_right) em [-1, 1]^2.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class HockeyPolicyNet(nn.Module):
    """Rede Ator-Crítico contínua leve para inferência em tempo real (< 0.2 ms)."""

    def __init__(self, obs_dim: int = 27, act_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.30))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs)


class HockeyPPOTrainer:
    """Treinador PPO Multi-Agente com Parameter Sharing para Hockey 3v3."""

    def __init__(
        self,
        obs_dim: int = 27,
        act_dim: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.device = torch.device(device)

        self.net = HockeyPolicyNet(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.obs_buf: List[np.ndarray] = []
        self.act_buf: List[np.ndarray] = []
        self.logp_buf: List[float] = []
        self.rew_buf: List[float] = []
        self.val_buf: List[float] = []
        self.done_buf: List[bool] = []

        self.total_steps: int = 0
        self.updates_count: int = 0
        self.last_loss: float = 0.0

    def reset_weights(self) -> None:
        """Reinicializa todos os pesos para zero aprendizado ('Ficar Burro')."""
        self.net = HockeyPolicyNet(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.clear_buffer()
        self.total_steps = 0
        self.updates_count = 0
        self.last_loss = 0.0

    def clear_buffer(self) -> None:
        self.obs_buf.clear()
        self.act_buf.clear()
        self.logp_buf.clear()
        self.rew_buf.clear()
        self.val_buf.clear()
        self.done_buf.clear()

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, Any, Any]:
        """
        Calcula ações para 1 robô ou para todo o trio de robôs [3, 27] em paralelo.
        """
        obs_arr = np.asarray(obs, dtype=np.float32)
        is_batch = (obs_arr.ndim == 2)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        if not is_batch:
            obs_t = obs_t.unsqueeze(0)

        mean, val = self.net(obs_t)
        std = torch.exp(torch.clamp(self.net.log_std, -1.8, 0.2))
        dist = Normal(mean, std)

        if deterministic:
            action = mean
            logp = torch.zeros(obs_t.shape[0], device=self.device)
        else:
            action = dist.sample()
            logp = dist.log_prob(action).sum(dim=-1)

        act_np = np.clip(action.cpu().numpy(), -1.0, 1.0)
        val_np = val.squeeze(-1).cpu().numpy()
        logp_np = logp.cpu().numpy()

        if not is_batch:
            return act_np[0], float(logp_np[0]), float(val_np[0])
        return act_np, logp_np, val_np

    def store_transition(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        logp: Any,
        rew: Any,
        val: Any,
        done: bool,
    ) -> None:
        """Armazena transição de um robô ou de toda a equipe 3v3."""
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 2 and obs_arr.shape[0] == 3:
            indiv_rews = rew if isinstance(rew, (list, tuple, np.ndarray)) else [rew, rew, rew]
            for i in range(3):
                self.obs_buf.append(obs_arr[i])
                self.act_buf.append(act[i])
                self.logp_buf.append(float(logp[i]))
                self.rew_buf.append(float(indiv_rews[i]))
                self.val_buf.append(float(val[i]))
                self.done_buf.append(done)
                self.total_steps += 1
        else:
            self.obs_buf.append(obs_arr)
            self.act_buf.append(act)
            self.logp_buf.append(float(logp))
            self.rew_buf.append(float(rew))
            self.val_buf.append(float(val))
            self.done_buf.append(done)
            self.total_steps += 1

    def update(self, epochs: int = 6, batch_size: int = 128) -> float:
        n = len(self.rew_buf)
        if n < 64:
            return 0.0

        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_val = 0.0

        for t in reversed(range(n)):
            mask = 0.0 if self.done_buf[t] else 1.0
            delta = self.rew_buf[t] + self.gamma * next_val * mask - self.val_buf[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.val_buf[t]
            next_val = self.val_buf[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.array(self.act_buf), dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(self.logp_buf, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        num_batches = 0
        indices = np.arange(n)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                mean, val = self.net(obs_t[idx])
                std = torch.exp(torch.clamp(self.net.log_std, -1.8, 0.2))
                dist = Normal(mean, std)
                logp = dist.log_prob(act_t[idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(logp - old_logp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.smooth_l1_loss(val.squeeze(-1), ret_t[idx])

                loss = policy_loss + 0.5 * value_loss - 0.008 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.net.actor.parameters()) + [self.net.log_std], 0.5)
                nn.utils.clip_grad_norm_(self.net.critic.parameters(), 1.0)
                self.optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

        with torch.no_grad():
            self.net.log_std.clamp_(-1.5, -0.20)

        self.clear_buffer()
        self.updates_count += 1
        self.last_loss = total_loss / max(1, num_batches)
        return self.last_loss

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.net.state_dict(),
                "total_steps": self.total_steps,
                "updates_count": self.updates_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.updates_count = ckpt.get("updates_count", 0)
