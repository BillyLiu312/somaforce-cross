"""Direction/magnitude semantic heads and clean-wrench magnitude targets."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class SemanticHeadOutput(NamedTuple):
    dir_logits: torch.Tensor
    p_dir: torch.Tensor
    mag_logits: torch.Tensor
    p_mag: torch.Tensor


class ForceSemanticHeads(nn.Module):
    def __init__(self, input_dim: int = 272, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim != 272 or hidden_dim != 128:
            raise ValueError("V1 ForceSemanticHeads requires input_dim=272 and hidden_dim=128")
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.dir_head = nn.Linear(hidden_dim, 13)
        self.mag_head = nn.Linear(hidden_dim, 5)

    def forward(self, fused_features: torch.Tensor) -> SemanticHeadOutput:
        if not isinstance(fused_features, torch.Tensor):
            raise TypeError("fused_features must be a torch.Tensor")
        if not fused_features.is_floating_point():
            raise TypeError("fused_features must have a floating-point dtype")
        if fused_features.ndim != 2 or fused_features.shape[1] != 272:
            raise ValueError(
                f"fused_features must have shape [B, 272], got {tuple(fused_features.shape)}"
            )
        if not torch.isfinite(fused_features).all():
            raise ValueError("fused_features must contain only finite values")
        hidden = self.backbone(fused_features)
        dir_logits = self.dir_head(hidden)
        mag_logits = self.mag_head(hidden)
        return SemanticHeadOutput(
            dir_logits=dir_logits,
            p_dir=torch.softmax(dir_logits, dim=-1),
            mag_logits=mag_logits,
            p_mag=torch.softmax(mag_logits, dim=-1),
        )


def magnitude_soft_targets(
    clean_wrench: torch.Tensor,
    F_scale: float | torch.Tensor,
    M_scale: float | torch.Tensor,
    *,
    sigma_mag: float = 0.15,
) -> torch.Tensor:
    """Return V1's five-bin soft magnitude target for any leading dimensions."""
    if not isinstance(clean_wrench, torch.Tensor):
        raise TypeError("clean_wrench must be a torch.Tensor")
    if not clean_wrench.is_floating_point():
        raise TypeError("clean_wrench must have a floating-point dtype")
    if clean_wrench.ndim < 1 or clean_wrench.shape[-1] != 6:
        raise ValueError(f"clean_wrench must have shape [..., 6], got {tuple(clean_wrench.shape)}")
    if not torch.isfinite(clean_wrench).all():
        raise ValueError("clean_wrench must contain only finite values")
    if not isinstance(sigma_mag, (float, int)) or sigma_mag <= 0:
        raise ValueError("sigma_mag must be positive")

    def _scale(value: float | torch.Tensor, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor) and value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        result = torch.as_tensor(value, device=clean_wrench.device, dtype=clean_wrench.dtype)
        if not torch.isfinite(result).all() or not bool((result > 0).item()):
            raise ValueError(f"{name} must be finite and positive")
        return result

    force_scale = _scale(F_scale, "F_scale")
    torque_scale = _scale(M_scale, "M_scale")
    force = torch.linalg.vector_norm(clean_wrench[..., :3], dim=-1) / force_scale
    torque = torch.linalg.vector_norm(clean_wrench[..., 3:], dim=-1) / torque_scale
    r_mag = torch.sqrt(0.5 * (force.square() + torque.square()))
    centers = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        device=clean_wrench.device,
        dtype=clean_wrench.dtype,
    )
    logits = -(r_mag.unsqueeze(-1) - centers).square() / (2.0 * float(sigma_mag) ** 2)
    return torch.softmax(logits, dim=-1)
