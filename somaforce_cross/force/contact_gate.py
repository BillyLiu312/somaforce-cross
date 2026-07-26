"""Stateless differentiable contact aggregation."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class ContactGateOutput(NamedTuple):
    per_wrist: torch.Tensor
    aggregate: torch.Tensor


class ContactGate(nn.Module):
    def forward(
        self,
        contact_probability: torch.Tensor,
        sensor_quality: torch.Tensor,
    ) -> ContactGateOutput:
        for value, name in (
            (contact_probability, "contact_probability"),
            (sensor_quality, "sensor_quality"),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if value.ndim != 2 or value.shape[1] != 2:
                raise ValueError(f"{name} must have shape [B, 2], got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            if torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must be within [0, 1]")
        if contact_probability.shape != sensor_quality.shape:
            raise ValueError("contact_probability and sensor_quality must have the same shape")
        per_wrist = contact_probability * sensor_quality
        aggregate = 1.0 - torch.prod(1.0 - per_wrist, dim=-1, keepdim=True)
        return ContactGateOutput(per_wrist=per_wrist, aggregate=aggregate)
