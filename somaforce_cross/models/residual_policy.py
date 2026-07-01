"""Bounded residual policy placeholder."""

import torch
from torch import nn


class ResidualPolicy(nn.Module):
    """Maps nominal/task features and ``z_cross`` to a bounded residual action."""

    def __init__(self, input_dim: int, action_dim: int, residual_limit: float = 0.1):
        super().__init__()
        self.residual_limit = residual_limit
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, policy_input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(policy_input)) * self.residual_limit
