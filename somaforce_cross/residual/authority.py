"""Curriculum authority and stateful contact-gain ramping."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


_JOINTS = 23
_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


def _env_ids(
    env_ids: int | Iterable[int] | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(env_ids, int) and not isinstance(env_ids, bool):
        ids = torch.tensor([env_ids], device=device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        if env_ids.dtype == torch.bool:
            if env_ids.ndim != 1 or env_ids.numel() != batch_size:
                raise ValueError("boolean env_ids must have shape [batch_size]")
            ids = torch.nonzero(env_ids, as_tuple=False).flatten().to(device)
        else:
            if env_ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if env_ids.dtype not in _INTEGER_DTYPES:
                raise TypeError("env_ids must contain integer indices")
            ids = env_ids.to(device=device, dtype=torch.long)
    else:
        try:
            ids = torch.as_tensor(list(env_ids), device=device, dtype=torch.long)
        except (TypeError, ValueError) as exc:
            raise TypeError("env_ids must be an integer or iterable of integer indices") from exc
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
    if ids.numel() and ((ids < 0).any() or (ids >= batch_size).any()):
        raise IndexError("env_ids contains an index outside the state")
    if ids.numel() and torch.unique(ids).numel() != ids.numel():
        raise ValueError("env_ids must not contain duplicates")
    return ids


class PerJointAuthority(nn.Module):
    """Map V1 curriculum stages to the fixed normalized 23-D authority."""

    ARMS = (11, 12, 15, 16, 19, 20, 21, 22)
    WAIST = (2, 5, 8)
    LEGS = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point dtype")
        values = torch.zeros(4, _JOINTS, device=device, dtype=dtype)
        curriculum = (
            (0.00, 0.00, 0.00),
            (0.10, 0.05, 0.04),
            (0.18, 0.10, 0.08),
            (0.25, 0.15, 0.12),
        )
        for stage, (arms, waist, legs) in enumerate(curriculum):
            values[stage, self.ARMS] = arms
            values[stage, self.WAIST] = waist
            values[stage, self.LEGS] = legs
        self.register_buffer("authority_by_stage", values)

    def forward(self, stage: int | torch.Tensor) -> torch.Tensor:
        if isinstance(stage, bool):
            raise TypeError("stage must be an integer curriculum stage")
        if isinstance(stage, int):
            if stage < 0 or stage > 3:
                raise ValueError("stage must be one of C0-C3 (0-3)")
            return self.authority_by_stage[stage]
        if not isinstance(stage, torch.Tensor):
            raise TypeError("stage must be an integer or integer tensor")
        if stage.ndim not in (0, 1):
            raise ValueError("stage must be scalar or have shape [B]")
        if stage.dtype not in _INTEGER_DTYPES:
            raise TypeError("stage tensor must have an integer dtype")
        if stage.device != self.authority_by_stage.device:
            raise ValueError(
                f"stage must be on {self.authority_by_stage.device}, got {stage.device}"
            )
        indices = stage.to(dtype=torch.long)
        if indices.numel() and ((indices < 0).any() or (indices > 3).any()):
            raise ValueError("stage must contain only C0-C3 values (0-3)")
        return self.authority_by_stage[indices]


class ContactGainRamp(nn.Module):
    """Per-environment soft attack/release ramp for an external contact target."""

    def __init__(
        self,
        batch_size: int,
        *,
        attack_step: float,
        release_step: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _validate_batch_size(batch_size)
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point dtype")
        for value, name in ((attack_step, "attack_step"), (release_step, "release_step")):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not torch.isfinite(torch.tensor(float(value))) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.attack_step = float(attack_step)
        self.release_step = float(release_step)
        self.register_buffer("state", torch.zeros(batch_size, 1, device=device, dtype=dtype))

    @property
    def batch_size(self) -> int:
        return self.state.shape[0]

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a torch.Tensor")
        if not target.is_floating_point():
            raise TypeError("target must have a floating-point dtype")
        if target.shape != self.state.shape:
            raise ValueError(f"target must have shape {tuple(self.state.shape)}, got {tuple(target.shape)}")
        if target.device != self.state.device:
            raise ValueError(f"target must be on {self.state.device}, got {target.device}")
        if target.dtype != self.state.dtype:
            raise ValueError(f"target must have dtype {self.state.dtype}, got {target.dtype}")
        if not torch.isfinite(target).all():
            raise ValueError("target must contain only finite values")
        if torch.any((target < 0) | (target > 1)):
            raise ValueError("target must be within [0, 1]")
        with torch.no_grad():
            attacked = torch.minimum(target, self.state + self.attack_step)
            released = torch.maximum(target, self.state - self.release_step)
            self.state.copy_(torch.where(target > self.state, attacked, released))
        return self.state

    def reset(self, env_ids: int | Iterable[int] | torch.Tensor) -> torch.Tensor:
        ids = _env_ids(env_ids, batch_size=self.batch_size, device=self.state.device)
        with torch.no_grad():
            self.state[ids] = 0
        return self.state
