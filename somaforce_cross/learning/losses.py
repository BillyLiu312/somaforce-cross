"""Semantic auxiliary objectives for the Phase 5 deployable actor."""

from __future__ import annotations

from dataclasses import dataclass

import torch


SEMANTIC_TARGET_DIM = 31
DIR_SLICE = slice(12, 25)
MAG_SLICE = slice(25, 30)
WEIGHT_SLICE = slice(30, 31)
SEMANTIC_EPS = 1.0e-8
LAMBDA_DIR = 0.05
LAMBDA_MAG = 0.05


@dataclass(frozen=True)
class SemanticTargetFields:
    """Named Phase 4 target fields; clean wrench stays target-side only."""

    clean_wrench: torch.Tensor
    p_dir_target: torch.Tensor
    p_mag_target: torch.Tensor
    semantic_loss_weight: torch.Tensor


@dataclass(frozen=True)
class SemanticLossOutput:
    """Unscaled and combined semantic terms retained for PPO logging."""

    dir_loss: torch.Tensor
    mag_loss: torch.Tensor
    total: torch.Tensor
    entropy: torch.Tensor


def _require_distribution(value: object, *, name: str, width: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [B, {width}]")
    if value.shape[0] <= 0:
        raise ValueError(f"{name} requires a nonempty batch")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    expected = torch.ones(value.shape[0], device=value.device, dtype=value.dtype)
    if not torch.allclose(value.sum(dim=-1), expected, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError(f"{name} rows must sum to one")
    if torch.any(value < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return value


def split_semantic_target(semantic_target: object) -> SemanticTargetFields:
    """Validate and split the immutable `[B,31]` Phase 4 target layout."""
    if not isinstance(semantic_target, torch.Tensor):
        raise TypeError("semantic_target must be a torch.Tensor")
    if not semantic_target.is_floating_point():
        raise TypeError("semantic_target must have a floating-point dtype")
    if semantic_target.ndim != 2 or semantic_target.shape[1] != SEMANTIC_TARGET_DIM:
        raise ValueError("semantic_target must have shape [B, 31]")
    if semantic_target.shape[0] <= 0:
        raise ValueError("semantic_target requires a nonempty batch")
    if not torch.isfinite(semantic_target).all():
        raise ValueError("semantic_target must be finite")
    p_dir_target = _require_distribution(
        semantic_target[:, DIR_SLICE], name="semantic_target.p_dir_target", width=13
    )
    p_mag_target = _require_distribution(
        semantic_target[:, MAG_SLICE], name="semantic_target.p_mag_target", width=5
    )
    weight = semantic_target[:, WEIGHT_SLICE]
    if torch.any(weight < 0.0):
        raise ValueError("semantic_loss_weight must be nonnegative")
    return SemanticTargetFields(
        clean_wrench=semantic_target[:, :12],
        p_dir_target=p_dir_target,
        p_mag_target=p_mag_target,
        semantic_loss_weight=weight,
    )


def weighted_soft_target_kl(
    target: object,
    prediction: object,
    weight: object,
    *,
    eps: float = SEMANTIC_EPS,
) -> torch.Tensor:
    """Compute weighted `D_KL(target || prediction)` with a connected zero case."""
    if not isinstance(eps, float) or eps <= 0.0:
        raise ValueError("eps must be a positive float")
    if not isinstance(target, torch.Tensor) or not isinstance(prediction, torch.Tensor):
        raise TypeError("target and prediction must be torch.Tensor values")
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must have equal [B, C] shapes")
    target = _require_distribution(target, name="target", width=target.shape[1])
    prediction = _require_distribution(
        prediction, name="prediction", width=prediction.shape[1]
    )
    if target.device != prediction.device or target.dtype != prediction.dtype:
        raise ValueError("target and prediction must share device and dtype")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if (
        weight.shape != (target.shape[0], 1)
        or weight.device != target.device
        or weight.dtype != target.dtype
    ):
        raise ValueError("weight must have shape [B, 1] on target device and dtype")
    if not torch.isfinite(weight).all() or torch.any(weight < 0.0):
        raise ValueError("weight must be finite and nonnegative")
    target_safe = target.clamp_min(eps)
    prediction_safe = prediction.clamp_min(eps)
    per_sample = (target_safe * (target_safe.log() - prediction_safe.log())).sum(dim=-1)
    return (weight.squeeze(-1) * per_sample).sum() / weight.sum().clamp_min(1.0)


def semantic_losses(
    p_dir_pred: object,
    p_mag_pred: object,
    semantic_target: object,
    *,
    eps: float = SEMANTIC_EPS,
) -> SemanticLossOutput:
    """Return Phase 5 KL terms without ever routing clean wrench to the actor."""
    p_dir_pred = _require_distribution(p_dir_pred, name="p_dir_pred", width=13)
    p_mag_pred = _require_distribution(p_mag_pred, name="p_mag_pred", width=5)
    if p_dir_pred.shape[0] != p_mag_pred.shape[0]:
        raise ValueError("p_dir_pred and p_mag_pred batch sizes must match")
    target = split_semantic_target(semantic_target)
    if target.p_dir_target.shape[0] != p_dir_pred.shape[0]:
        raise ValueError("semantic_target batch size must match predictions")
    if target.p_dir_target.device != p_dir_pred.device:
        raise ValueError("semantic_target and predictions must share a device")
    if target.p_dir_target.dtype != p_dir_pred.dtype:
        raise ValueError("semantic_target and predictions must share a dtype")
    dir_loss = weighted_soft_target_kl(
        target.p_dir_target,
        p_dir_pred,
        target.semantic_loss_weight,
        eps=eps,
    )
    mag_loss = weighted_soft_target_kl(
        target.p_mag_target,
        p_mag_pred,
        target.semantic_loss_weight,
        eps=eps,
    )
    entropy = (
        -0.5
        * (
            (p_dir_pred * p_dir_pred.clamp_min(eps).log()).sum(dim=-1)
            + (p_mag_pred * p_mag_pred.clamp_min(eps).log()).sum(dim=-1)
        ).mean()
    )
    total = LAMBDA_DIR * dir_loss + LAMBDA_MAG * mag_loss
    return SemanticLossOutput(
        dir_loss=dir_loss,
        mag_loss=mag_loss,
        total=total,
        entropy=entropy,
    )
