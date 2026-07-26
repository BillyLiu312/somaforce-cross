"""Contact-gated normalized residual composition with ordered safety limits."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from somaforce_cross.residual.safety import JointMarginLimiter, VelocityMarginLimiter


_JOINTS = 23


class ContactGatedResidualOutput(NamedTuple):
    delta_bounded: torch.Tensor
    delta_gated: torch.Tensor


class ResidualActionOutput(NamedTuple):
    delta_bounded: torch.Tensor
    delta_gated: torch.Tensor
    delta_safe: torch.Tensor
    a_total: torch.Tensor


def _batch_action(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != _JOINTS:
        raise ValueError(f"{name} must have shape [B, 23], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _compatible(value: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {value.device}")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} must have dtype {reference.dtype}, got {value.dtype}")


def _gain(
    value: object,
    name: str,
    reference: torch.Tensor,
    allowed_widths: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim != 2 or value.shape[0] != reference.shape[0] or value.shape[1] not in allowed_widths:
        widths = " or ".join(f"[B, {width}]" for width in allowed_widths)
        raise ValueError(f"{name} must have shape {widths}, got {tuple(value.shape)}")
    _compatible(value, reference, name)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if torch.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must be within [0, 1]")
    return value


class ContactGatedResidual(nn.Module):
    """Apply V1 authority, contact gain, and external safety gain."""

    def forward(
        self,
        raw_delta: torch.Tensor,
        authority: torch.Tensor,
        contact_gain: torch.Tensor,
        safety_gain: torch.Tensor,
    ) -> ContactGatedResidualOutput:
        raw_delta = _batch_action(raw_delta, "raw_delta")
        if not isinstance(authority, torch.Tensor):
            raise TypeError("authority must be a torch.Tensor")
        if not authority.is_floating_point():
            raise TypeError("authority must have a floating-point dtype")
        if authority.shape not in ((_JOINTS,), raw_delta.shape):
            raise ValueError(
                f"authority must have shape [23] or {tuple(raw_delta.shape)}, got {tuple(authority.shape)}"
            )
        _compatible(authority, raw_delta, "authority")
        if not torch.isfinite(authority).all():
            raise ValueError("authority must contain only finite values")
        if torch.any((authority < 0) | (authority > 1)):
            raise ValueError("authority must be within [0, 1]")
        contact_gain = _gain(contact_gain, "contact_gain", raw_delta, (1,))
        safety_gain = _gain(safety_gain, "safety_gain", raw_delta, (1, _JOINTS))
        delta_bounded = authority * torch.tanh(raw_delta)
        delta_gated = contact_gain * safety_gain * delta_bounded
        return ContactGatedResidualOutput(
            delta_bounded=delta_bounded,
            delta_gated=delta_gated,
        )


class ResidualActionComposer(nn.Module):
    """Compose residual actions in the fixed V1 safety order."""

    def __init__(
        self,
        joint_margin_limiter: JointMarginLimiter,
        velocity_margin_limiter: VelocityMarginLimiter,
        *,
        action_low: float | torch.Tensor | None = None,
        action_high: float | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(joint_margin_limiter, JointMarginLimiter):
            raise TypeError("joint_margin_limiter must be a JointMarginLimiter")
        if not isinstance(velocity_margin_limiter, VelocityMarginLimiter):
            raise TypeError("velocity_margin_limiter must be a VelocityMarginLimiter")
        for name in ("default_joint_pos", "action_scale"):
            joint_value = getattr(joint_margin_limiter, name)
            velocity_value = getattr(velocity_margin_limiter, name)
            if (
                joint_value.device != velocity_value.device
                or joint_value.dtype != velocity_value.dtype
                or not torch.equal(joint_value, velocity_value)
            ):
                raise ValueError(
                    f"joint and velocity limiters must use the same {name}"
                )
        self.joint_margin_limiter = joint_margin_limiter
        self.velocity_margin_limiter = velocity_margin_limiter
        self.gate = ContactGatedResidual()
        if (action_low is None) != (action_high is None):
            raise ValueError("action_low and action_high must be supplied together")
        if action_low is None:
            self.register_buffer("action_low", None)
            self.register_buffer("action_high", None)
        else:
            reference = joint_margin_limiter.default_joint_pos
            low = self._action_bound(action_low, "action_low", reference)
            high = self._action_bound(action_high, "action_high", reference)
            if torch.any(low >= high):
                raise ValueError("action_low must be strictly below action_high")
            self.register_buffer("action_low", low)
            self.register_buffer("action_high", high)

    @staticmethod
    def _action_bound(
        value: float | torch.Tensor,
        name: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor) and value.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {value.device}")
        result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if result.ndim == 0:
            result = result.expand(_JOINTS).clone()
        elif result.shape == (_JOINTS,):
            result = result.clone()
        else:
            raise ValueError(f"{name} must be scalar or have shape [23]")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must contain only finite values")
        return result

    def forward(
        self,
        a_nom: torch.Tensor,
        raw_delta: torch.Tensor,
        authority: torch.Tensor,
        contact_gain: torch.Tensor,
        safety_gain: torch.Tensor,
        current_joint_pos: torch.Tensor,
    ) -> ResidualActionOutput:
        a_nom = _batch_action(a_nom, "a_nom")
        raw_delta = _batch_action(raw_delta, "raw_delta")
        if raw_delta.shape != a_nom.shape:
            raise ValueError("raw_delta and a_nom must have the same shape")
        _compatible(raw_delta, a_nom, "raw_delta")
        gated = self.gate(raw_delta, authority, contact_gain, safety_gain)
        delta_safe = self.joint_margin_limiter(a_nom, gated.delta_gated)
        delta_safe = self.velocity_margin_limiter(
            a_nom, delta_safe, current_joint_pos
        )
        if self.action_low is not None:
            _compatible(self.action_low, a_nom, "action_low")
            lower = torch.minimum(self.action_low - a_nom, torch.zeros_like(a_nom))
            upper = torch.maximum(self.action_high - a_nom, torch.zeros_like(a_nom))
            delta_safe = torch.clamp(delta_safe, min=lower, max=upper)
        a_total = a_nom + delta_safe
        return ResidualActionOutput(
            delta_bounded=gated.delta_bounded,
            delta_gated=gated.delta_gated,
            delta_safe=delta_safe,
            a_total=a_total,
        )
