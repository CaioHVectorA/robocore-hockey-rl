"""Neural Actor-Critic Policy Network for Mini-Sumo 500g."""

import math
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from sumo.constants import SUMO_ACT_DIM, SUMO_OBS_DIM


class SumoPolicyNet(nn.Module):
    """Actor-Critic MLP for continuous wheel differential drive control in Sumo."""

    def __init__(
        self,
        obs_dim: int = SUMO_OBS_DIM,
        act_dim: int = SUMO_ACT_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        def make_backbone(out_dim: int):
            return nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.actor = make_backbone(act_dim)
        self.critic = make_backbone(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.8))

        # Orthogonal weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self.log_std.clamp(-3.0, 0.5).exp()
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from raw observation vector."""
        self.eval()
        t_obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            action = self.actor(t_obs)
        else:
            action = self.distribution(t_obs).sample()
        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).cpu().numpy()
