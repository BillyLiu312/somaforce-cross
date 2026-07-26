"""Outer-product cross-semantic encoder."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class CrossEncoderOutput(NamedTuple):
    P_cross: torch.Tensor
    z_joint: torch.Tensor
    z_cross: torch.Tensor


class CrossEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 64) -> None:
        super().__init__()
        if hidden_dim != 128 or latent_dim != 64:
            raise ValueError("V1 CrossEncoder requires hidden_dim=128 and latent_dim=64")
        self.mlp = nn.Sequential(
            nn.Linear(13 * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, p_dir: torch.Tensor, p_mag: torch.Tensor) -> CrossEncoderOutput:
        for value, name, width in ((p_dir, "p_dir", 13), (p_mag, "p_mag", 5)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if value.ndim != 2 or value.shape[1] != width:
                raise ValueError(f"{name} must have shape [B, {width}], got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if p_dir.shape[0] != p_mag.shape[0]:
            raise ValueError("p_dir and p_mag must have the same batch size")
        P_cross = p_dir.unsqueeze(-1) * p_mag.unsqueeze(-2)
        z_joint = P_cross.reshape(P_cross.shape[0], 13 * 5)
        z_cross = self.mlp(z_joint)
        return CrossEncoderOutput(P_cross=P_cross, z_joint=z_joint, z_cross=z_cross)
