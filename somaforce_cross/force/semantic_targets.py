"""Clean-wrench direction, magnitude, and two-wrist semantic targets."""

from __future__ import annotations

from typing import NamedTuple

import torch

from somaforce_cross.force.semantic_heads import magnitude_soft_targets

_DIRECTION_TEMPERATURE = 0.15
_MAGNITUDE_SIGMA = 0.15


class TwoWristSemanticTargets(NamedTuple):
    per_wrist_direction: torch.Tensor
    per_wrist_magnitude: torch.Tensor
    global_direction: torch.Tensor
    global_magnitude: torch.Tensor
    g_contact: torch.Tensor


def _scale(
    value: float | torch.Tensor, name: str, reference: torch.Tensor
) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(result).all() or torch.any(result <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _wrench(clean_wrench: torch.Tensor) -> None:
    if not isinstance(clean_wrench, torch.Tensor):
        raise TypeError("clean_wrench must be a torch.Tensor")
    if not clean_wrench.is_floating_point():
        raise TypeError("clean_wrench must have a floating-point dtype")
    if clean_wrench.ndim < 1 or clean_wrench.shape[-1] != 6:
        raise ValueError(
            f"clean_wrench must have shape [..., 6], got {tuple(clean_wrench.shape)}"
        )
    if not torch.isfinite(clean_wrench).all():
        raise ValueError("clean_wrench must contain only finite values")


def direction_soft_targets(
    clean_wrench: torch.Tensor,
    F_scale: float | torch.Tensor,
    M_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Return fixed-temperature 13-class direction targets."""
    _wrench(clean_wrench)
    force_scale = _scale(F_scale, "F_scale", clean_wrench)
    torque_scale = _scale(M_scale, "M_scale", clean_wrench)
    x = torch.cat(
        (clean_wrench[..., :3] / force_scale, clean_wrench[..., 3:] / torque_scale),
        dim=-1,
    )
    absolute = x.abs()
    max_abs = absolute.amax(dim=-1)
    sum_abs = absolute.sum(dim=-1)
    eps = torch.finfo(clean_wrench.dtype).eps
    dominance = torch.where(sum_abs > 0, max_abs / sum_abs.clamp_min(eps), 0.0)
    r_force = torch.linalg.vector_norm(x[..., :3], dim=-1)
    r_torque = torch.linalg.vector_norm(x[..., 3:], dim=-1)
    strength = torch.sqrt(0.5 * (r_force.square() + r_torque.square())).clamp(0, 1)
    signed_profile = torch.stack((torch.relu(x), torch.relu(-x)), dim=-1).flatten(-2)
    signed_profile = signed_profile / max_abs.clamp_min(eps).unsqueeze(-1)
    axis_evidence = strength.unsqueeze(-1) * dominance.unsqueeze(-1) * signed_profile
    neutral_evidence = torch.maximum(1.0 - strength, 1.0 - dominance).unsqueeze(-1)
    evidence = torch.cat((axis_evidence, neutral_evidence), dim=-1)
    return torch.softmax(evidence / _DIRECTION_TEMPERATURE, dim=-1)


def two_wrist_semantic_targets(
    clean_wrench: torch.Tensor,
    contact_probability: torch.Tensor,
    F_scale: float | torch.Tensor,
    M_scale: float | torch.Tensor,
) -> TwoWristSemanticTargets:
    """Aggregate clean per-wrist targets using contact probability only."""
    _wrench(clean_wrench)
    if clean_wrench.ndim != 3 or clean_wrench.shape[1:] != (2, 6):
        raise ValueError("clean_wrench must have shape [B, 2, 6]")
    if not isinstance(contact_probability, torch.Tensor):
        raise TypeError("contact_probability must be a torch.Tensor")
    if not contact_probability.is_floating_point():
        raise TypeError("contact_probability must have a floating-point dtype")
    if contact_probability.shape != clean_wrench.shape[:2]:
        raise ValueError("contact_probability must have shape [B, 2]")
    if (
        contact_probability.device != clean_wrench.device
        or contact_probability.dtype != clean_wrench.dtype
    ):
        raise ValueError("contact_probability must match clean_wrench device and dtype")
    if not torch.isfinite(contact_probability).all() or torch.any(
        (contact_probability < 0) | (contact_probability > 1)
    ):
        raise ValueError("contact_probability must be finite and within [0, 1]")

    per_direction = direction_soft_targets(clean_wrench, F_scale, M_scale)
    per_magnitude = magnitude_soft_targets(
        clean_wrench, F_scale, M_scale, sigma_mag=_MAGNITUDE_SIGMA
    )
    weights = contact_probability.unsqueeze(-1)
    weight_sum = weights.sum(dim=1)
    # Preserve every positive contact weight; only the exact zero case uses a
    # fallback denominator before being replaced by the zero-wrench targets.
    safe_denominator = torch.where(
        weight_sum > 0,
        weight_sum,
        torch.ones_like(weight_sum),
    )
    global_direction = (weights * per_direction).sum(dim=1) / safe_denominator
    global_magnitude = (weights * per_magnitude).sum(dim=1) / safe_denominator

    zero_wrench = torch.zeros(
        clean_wrench.shape[0], 6, device=clean_wrench.device, dtype=clean_wrench.dtype
    )
    zero_direction = direction_soft_targets(zero_wrench, F_scale, M_scale)
    zero_magnitude = magnitude_soft_targets(
        zero_wrench, F_scale, M_scale, sigma_mag=_MAGNITUDE_SIGMA
    )
    no_contact = weight_sum == 0
    global_direction = torch.where(no_contact, zero_direction, global_direction)
    global_magnitude = torch.where(no_contact, zero_magnitude, global_magnitude)
    # log1p/expm1 retains tiny non-zero aggregate contact probabilities.
    g_contact = -torch.expm1(
        torch.log1p(-contact_probability).sum(dim=-1, keepdim=True)
    )
    return TwoWristSemanticTargets(
        per_wrist_direction=per_direction,
        per_wrist_magnitude=per_magnitude,
        global_direction=global_direction,
        global_magnitude=global_magnitude,
        g_contact=g_contact,
    )
