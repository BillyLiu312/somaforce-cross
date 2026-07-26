"""Residual-only joint-position and velocity safety limiters."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


_JOINTS = 23


class ResidualInterval(NamedTuple):
    lower: torch.Tensor
    upper: torch.Tensor


def _base_vector(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.shape != (_JOINTS,):
        raise ValueError(f"{name} must have shape [23], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _parameter_vector(
    value: float | torch.Tensor,
    name: str,
    reference: torch.Tensor,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {value.device}")
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(_JOINTS).clone()
    elif tensor.shape == (_JOINTS,):
        tensor = tensor.clone()
    else:
        raise ValueError(f"{name} must be scalar or have shape [23], got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and torch.any(tensor <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and torch.any(tensor < 0):
        raise ValueError(f"{name} must be nonnegative")
    return tensor


def _action(value: object, name: str, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != _JOINTS:
        raise ValueError(f"{name} must have shape [B, 23], got {tuple(value.shape)}")
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {value.device}")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} must have dtype {reference.dtype}, got {value.dtype}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _residual(value: object, a_nom: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = _action(value, "residual", reference)
    if value.shape != a_nom.shape:
        raise ValueError("residual and a_nom must have the same shape")
    return value


def _include_zero(lower: torch.Tensor, upper: torch.Tensor) -> ResidualInterval:
    zero = torch.zeros((), device=lower.device, dtype=lower.dtype)
    return ResidualInterval(
        lower=torch.minimum(lower, zero),
        upper=torch.maximum(upper, zero),
    )


class JointMarginLimiter(nn.Module):
    """Limit normalized residuals by the remaining radian joint margin."""

    def __init__(
        self,
        *,
        default_joint_pos: torch.Tensor,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        action_scale: float | torch.Tensor,
        margin: float | torch.Tensor,
    ) -> None:
        super().__init__()
        default_joint_pos = _base_vector(default_joint_pos, "default_joint_pos")
        lower = _parameter_vector(joint_lower, "joint_lower", default_joint_pos)
        upper = _parameter_vector(joint_upper, "joint_upper", default_joint_pos)
        scale = _parameter_vector(action_scale, "action_scale", default_joint_pos, positive=True)
        margin_vector = _parameter_vector(
            margin, "margin", default_joint_pos, nonnegative=True
        )
        if torch.any(lower >= upper):
            raise ValueError("joint_lower must be strictly below joint_upper")
        if torch.any(lower + margin_vector > upper - margin_vector):
            raise ValueError("margin leaves an empty joint-safe interval")
        self.register_buffer("default_joint_pos", default_joint_pos.clone())
        self.register_buffer("joint_lower", lower)
        self.register_buffer("joint_upper", upper)
        self.register_buffer("action_scale", scale)
        self.register_buffer("margin", margin_vector)

    def residual_interval(self, a_nom: torch.Tensor) -> ResidualInterval:
        a_nom = _action(a_nom, "a_nom", self.default_joint_pos)
        q_nom = self.default_joint_pos + a_nom * self.action_scale
        lower = (self.joint_lower + self.margin - q_nom) / self.action_scale
        upper = (self.joint_upper - self.margin - q_nom) / self.action_scale
        return _include_zero(lower, upper)

    def forward(self, a_nom: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        a_nom = _action(a_nom, "a_nom", self.default_joint_pos)
        residual = _residual(residual, a_nom, self.default_joint_pos)
        interval = self.residual_interval(a_nom)
        return torch.clamp(residual, min=interval.lower, max=interval.upper)


class VelocityMarginLimiter(nn.Module):
    """Limit normalized residuals by the remaining one-step velocity margin."""

    def __init__(
        self,
        *,
        default_joint_pos: torch.Tensor,
        action_scale: float | torch.Tensor,
        velocity_limit: float | torch.Tensor,
        dt: float,
    ) -> None:
        super().__init__()
        default_joint_pos = _base_vector(default_joint_pos, "default_joint_pos")
        scale = _parameter_vector(action_scale, "action_scale", default_joint_pos, positive=True)
        velocity = _parameter_vector(
            velocity_limit, "velocity_limit", default_joint_pos, positive=True
        )
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise TypeError("dt must be a real number")
        dt_tensor = torch.tensor(float(dt), device=default_joint_pos.device, dtype=default_joint_pos.dtype)
        if not torch.isfinite(dt_tensor) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.register_buffer("default_joint_pos", default_joint_pos.clone())
        self.register_buffer("action_scale", scale)
        self.register_buffer("velocity_limit", velocity)
        self.register_buffer("dt", dt_tensor)

    def residual_interval(
        self,
        a_nom: torch.Tensor,
        current_joint_pos: torch.Tensor,
    ) -> ResidualInterval:
        a_nom = _action(a_nom, "a_nom", self.default_joint_pos)
        current_joint_pos = _action(
            current_joint_pos, "current_joint_pos", self.default_joint_pos
        )
        if current_joint_pos.shape != a_nom.shape:
            raise ValueError("current_joint_pos and a_nom must have the same shape")
        q_nom = self.default_joint_pos + a_nom * self.action_scale
        nominal_step = q_nom - current_joint_pos
        physical_limit = self.velocity_limit * self.dt
        lower = (-physical_limit - nominal_step) / self.action_scale
        upper = (physical_limit - nominal_step) / self.action_scale
        return _include_zero(lower, upper)

    def forward(
        self,
        a_nom: torch.Tensor,
        residual: torch.Tensor,
        current_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        a_nom = _action(a_nom, "a_nom", self.default_joint_pos)
        residual = _residual(residual, a_nom, self.default_joint_pos)
        interval = self.residual_interval(a_nom, current_joint_pos)
        return torch.clamp(residual, min=interval.lower, max=interval.upper)
