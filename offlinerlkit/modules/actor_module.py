import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
from typing import Union, Optional

from torch.distributions import Normal, TanhTransform, TransformedDistribution
EPS = 1e-7

# for SAC
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist
    
    def get_log_density(self, observations, action):
        logits = self.backbone(observations)
        mu = self.dist_net.mu(logits)
        if not self.dist_net._unbounded:
            mu = self.dist_net._max * torch.tanh(mu)
        if self.dist_net._c_sigma:
            sigma = torch.clamp(self.dist_net.sigma(logits), min=self.dist_net._sigma_min, max=self.dist_net._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.dist_net.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        action_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_prob = torch.sum(action_distribution.log_prob(action_clip), dim=-1)

        return logp_prob



# for TD3
class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions