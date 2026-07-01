"""Probabilistic outer-product cross interaction."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CrossInteractionOutput:
    """Outputs of the SomaForce-Cross interaction module."""

    p_dir: torch.Tensor
    p_mag: torch.Tensor
    p_cross: torch.Tensor
    z_joint: torch.Tensor
    z_cross: torch.Tensor


class ProbabilisticCrossInteraction(nn.Module):
    """Build ``z_cross`` from direction and magnitude semantic distributions.

    The selected design is:

    ```text
    p_dir = softmax(E_dir(...))
    p_mag = softmax(E_mag(...))
    P_cross = p_dir outer p_mag
    z_joint = flatten(P_cross)
    z_cross = CrossEncoder(z_joint)
    ```
    """

    def __init__(self, dir_dim: int, mag_dim: int, n_dir: int, n_mag: int, z_dim: int):
        super().__init__()
        self.dir_head = nn.Linear(dir_dim, n_dir)
        self.mag_head = nn.Linear(mag_dim, n_mag)
        self.cross_encoder = nn.Sequential(
            nn.Linear(n_dir * n_mag, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, h_dir: torch.Tensor, h_mag: torch.Tensor) -> CrossInteractionOutput:
        p_dir = F.softmax(self.dir_head(h_dir), dim=-1)
        p_mag = F.softmax(self.mag_head(h_mag), dim=-1)
        p_cross = p_dir[..., :, None] * p_mag[..., None, :]
        z_joint = p_cross.flatten(start_dim=-2)
        z_cross = self.cross_encoder(z_joint)
        return CrossInteractionOutput(
            p_dir=p_dir,
            p_mag=p_mag,
            p_cross=p_cross,
            z_joint=z_joint,
            z_cross=z_cross,
        )
