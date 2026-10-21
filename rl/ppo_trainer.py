"""Parameter-sharing PPO with independent temporal trajectories per robot."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from vision.state import OBS_DIM, OBSERVATION_VERSION


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.995, gae_lambda=0.95):
    """All arrays [time, agent]. Bootstrap truncations, stop traces at resets."""
    rewards = np.asarray(rewards, np.float32)
    values = np.asarray(values, np.float32)
    next_values = np.asarray(next_values, np.float32)
    terminated = np.asarray(terminated, bool)
    truncated = np.asarray(truncated, bool)
    if not all(x.shape == rewards.shape for x in (values, next_values, terminated, truncated)):
        raise ValueError("GAE tensors must have matching time/agent axes")
    adv = np.zeros_like(rewards)
    tail = np.zeros(rewards.shape[1:], np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        tail = delta + gamma * gae_lambda * (~(terminated[t] | truncated[t])) * tail
        adv[t] = tail
    return adv, adv + values


class HockeyPolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=2, hidden_dim=128):
        super().__init__()

        def mlp(out):
            return nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out),
            )

        self.actor = mlp(act_dim)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.8))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def distribution(self, obs):
        mean, value = self(obs)
        return Normal(mean, self.log_std.clamp(-3, 0.5).exp()), value

    def evaluate(self, obs, action):
        dist, value = self.distribution(obs)
        action = action.clamp(-1 + 1e-6, 1 - 1e-6)
        latent = torch.atanh(action)
        logp = (dist.log_prob(latent) - torch.log1p(-action.square())).sum(-1)
        return logp, value


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    logp: np.ndarray
    reward: np.ndarray
    value: np.ndarray
    next_value: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class HockeyPPOTrainer:
    def __init__(
        self,
        obs_dim=OBS_DIM,
        act_dim=2,
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_eps=0.2,
        device="cpu",
        seed=0,
        target_kl=0.02,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.target_kl = target_kl
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.net = HockeyPolicyNet(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer = []
        self.total_steps = 0
        self.environment_steps = 0
        self.updates_count = 0
        self.last_loss = 0.0
        self.metrics = {}

    def reset_weights(self):
        self.net = HockeyPolicyNet(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)
        self.buffer.clear()
        self.total_steps = self.environment_steps = self.updates_count = 0

    def clear_buffer(self):
        self.buffer.clear()

    @property
    def buffer_steps(self):
        return len(self.buffer)

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        arr = np.asarray(obs, np.float32)
        single = arr.ndim == 1
        x = torch.as_tensor(arr, device=self.device)
        if single:
            x = x.unsqueeze(0)
        dist, value = self.net.distribution(x)
        action = torch.tanh(dist.mean if deterministic else dist.sample()).clamp(-1 + 1e-6, 1 - 1e-6)
        logp, _ = self.net.evaluate(x, action)
        actions, logps, values = action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
        return (actions[0], float(logps[0]), float(values[0])) if single else (actions, logps, values)

    @torch.no_grad()
    def values(self, obs):
        return self.net(torch.as_tensor(np.asarray(obs, np.float32), device=self.device))[1].cpu().numpy()

    def store_transition(
        self, obs, act, logp, rew, val, terminated, truncated=False, *, next_obs, env_count=1
    ):
        obs = np.atleast_2d(np.asarray(obs, np.float32))
        n = len(obs)

        def vector(x):
            return np.broadcast_to(np.asarray(x), (n,)).copy()

        next_value = np.asarray(self.values(np.atleast_2d(next_obs)), np.float32)
        item = Transition(
            obs.copy(),
            np.asarray(act, np.float32).reshape(n, self.act_dim).copy(),
            vector(logp),
            vector(rew),
            vector(val),
            next_value,
            vector(terminated).astype(bool),
            vector(truncated).astype(bool),
        )
        if self.buffer and item.obs.shape != self.buffer[-1].obs.shape:
            raise ValueError("Agent count changed within rollout")
        self.buffer.append(item)
        self.total_steps += n
        if env_count < 1 or n % env_count:
            raise ValueError("Invalid environment count")
        self.environment_steps += env_count

    def update(self, epochs=6, batch_size=256):
        if not self.buffer:
            return 0.0

        def stack(key):
            return np.stack([getattr(t, key) for t in self.buffer])

        advantages, returns = compute_gae(
            stack("reward"),
            stack("value"),
            stack("next_value"),
            stack("terminated"),
            stack("truncated"),
            self.gamma,
            self.gae_lambda,
        )
        adv = advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        obs = tensor(stack("obs").reshape(-1, self.obs_dim))
        act = tensor(stack("action").reshape(-1, self.act_dim))
        old = tensor(stack("logp").reshape(-1))
        advt = tensor(adv)
        ret = tensor(returns.reshape(-1))
        losses = []
        kls = []
        clips = []
        entropies = []
        stop = False
        for _ in range(epochs):
            for idx in np.array_split(
                self.rng.permutation(len(obs)), max(1, math.ceil(len(obs) / batch_size))
            ):
                lp, value = self.net.evaluate(obs[idx], act[idx])
                logratio = lp - old[idx]
                ratio = logratio.exp()
                policy = -torch.minimum(
                    ratio * advt[idx], ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advt[idx]
                ).mean()
                vloss = 0.5 * (value - ret[idx]).square().mean()
                # Monte Carlo entropy estimate on fresh actions, including tanh Jacobian.
                dist, _ = self.net.distribution(obs[idx])
                sample = dist.rsample().tanh().clamp(-1 + 1e-6, 1 - 1e-6)
                entropy = -self.net.evaluate(obs[idx], sample)[0].mean()
                loss = policy + 0.5 * vloss - 0.002 * entropy
                with torch.no_grad():
                    kl = ((ratio - 1) - logratio).mean().item()
                if kl > 1.5 * self.target_kl:
                    stop = True
                    break
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                losses.append(loss.item())
                kls.append(kl)
                clips.append(((ratio - 1).abs() > self.clip_eps).float().mean().item())
                entropies.append(entropy.item())
            if stop:
                break
        variance = np.var(returns)
        explained = 1 - np.var(returns - stack("value")) / variance if variance > 1e-8 else 0.0
        self.last_loss = float(np.mean(losses)) if losses else 0.0
        self.updates_count += 1
        self.metrics = dict(
            loss=self.last_loss,
            approx_kl=float(np.mean(kls)) if kls else 0.0,
            clip_fraction=float(np.mean(clips)) if clips else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            explained_variance=float(explained),
            early_stop=stop,
            transitions=len(obs),
        )
        self.buffer.clear()
        return self.last_loss

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(
            format_version=2,
            observation_version=OBSERVATION_VERSION,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            model_state=self.net.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            total_steps=self.total_steps,
            environment_steps=self.environment_steps,
            updates_count=self.updates_count,
            hyperparameters=dict(
                lr=self.lr,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_eps=self.clip_eps,
                target_kl=self.target_kl,
            ),
            metadata=metadata or {},
        )
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(data, temp)
        temp.replace(path)

    def load(self, path, resume=False):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if ckpt.get("format_version") != 2 or ckpt.get("observation_version") != OBSERVATION_VERSION:
            raise ValueError(
                "Legacy/incompatible checkpoint. Train a v2 policy; the old 27D weights cannot be reused."
            )
        if ckpt["obs_dim"] != self.obs_dim or ckpt["act_dim"] != self.act_dim:
            raise ValueError("Checkpoint dimensions differ")
        self.net.load_state_dict(ckpt["model_state"])
        self.total_steps = ckpt["total_steps"]
        self.environment_steps = ckpt["environment_steps"]
        self.updates_count = ckpt["updates_count"]
        if resume:
            for key, val in ckpt["hyperparameters"].items():
                setattr(self, key, val)
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.buffer.clear()
        return ckpt.get("metadata", {})
