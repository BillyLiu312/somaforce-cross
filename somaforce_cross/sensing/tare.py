"""Explicitly gated episode tare calibration for two wrist wrenches."""

from __future__ import annotations

from typing import Iterable, NamedTuple

import torch


class TareOutput(NamedTuple):
    calibrated_wrench: torch.Tensor
    calibration_ready: torch.Tensor


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
            if env_ids.shape != (batch_size,):
                raise ValueError("boolean env_ids must have shape [batch_size]")
            ids = torch.nonzero(env_ids, as_tuple=False).flatten().to(device)
        else:
            if env_ids.ndim != 1 or env_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("env_ids must be a one-dimensional integer tensor")
            ids = env_ids.to(device=device, dtype=torch.long)
    else:
        try:
            ids = torch.as_tensor(list(env_ids), device=device, dtype=torch.long)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "env_ids must be an integer or iterable of integers"
            ) from exc
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
    if ids.numel() and ((ids < 0).any() or (ids >= batch_size).any()):
        raise IndexError("env_ids contains an index outside the calibrator")
    if ids.numel() and torch.unique(ids).numel() != ids.numel():
        raise ValueError("env_ids must not contain duplicates")
    return ids


class WristTareCalibrator:
    """Accumulate a caller-gated finite mean and freeze it per env and wrist.

    The caller alone decides when ``calibration_mask`` is true. This class has no
    contact or reset-wait heuristic and cannot update a wrist after it becomes
    ready until that environment is explicitly reset.
    """

    def __init__(
        self,
        batch_size: int,
        *,
        num_samples: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError("num_samples must be a positive integer")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point dtype")
        self.num_samples = num_samples
        self.bias = torch.zeros(batch_size, 2, 6, device=device, dtype=dtype)
        self.sample_sum = torch.zeros_like(self.bias)
        self.count = torch.zeros(batch_size, 2, device=device, dtype=torch.long)
        self.ready = torch.zeros(batch_size, 2, device=device, dtype=torch.bool)

    @property
    def batch_size(self) -> int:
        return self.bias.shape[0]

    @property
    def device(self) -> torch.device:
        return self.bias.device

    @property
    def dtype(self) -> torch.dtype:
        return self.bias.dtype

    def reset(self, env_ids: int | Iterable[int] | torch.Tensor) -> None:
        ids = _env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        with torch.no_grad():
            self.bias[ids] = 0
            self.sample_sum[ids] = 0
            self.count[ids] = 0
            self.ready[ids] = False

    def update(
        self, wrench: torch.Tensor, calibration_mask: torch.Tensor
    ) -> TareOutput:
        if not isinstance(wrench, torch.Tensor):
            raise TypeError("wrench must be a torch.Tensor")
        if not wrench.is_floating_point():
            raise TypeError("wrench must have a floating-point dtype")
        if wrench.shape != self.bias.shape:
            raise ValueError(
                f"wrench must have shape {tuple(self.bias.shape)}, got {tuple(wrench.shape)}"
            )
        if wrench.device != self.device or wrench.dtype != self.dtype:
            raise ValueError("wrench must match the calibrator device and dtype")
        if not torch.isfinite(wrench).all():
            raise ValueError("wrench must contain only finite values")
        if (
            not isinstance(calibration_mask, torch.Tensor)
            or calibration_mask.dtype != torch.bool
        ):
            raise TypeError("calibration_mask must be a boolean torch.Tensor")
        if calibration_mask.shape != self.ready.shape:
            raise ValueError(
                f"calibration_mask must have shape {tuple(self.ready.shape)}, got {tuple(calibration_mask.shape)}"
            )
        if calibration_mask.device != self.device:
            raise ValueError("calibration_mask must match the calibrator device")

        active = calibration_mask & ~self.ready
        with torch.no_grad():
            self.sample_sum.add_(torch.where(active[..., None], wrench.detach(), 0.0))
            self.count.add_(active.to(dtype=self.count.dtype))
            completed = active & (self.count >= self.num_samples)
            if completed.any():
                means = self.sample_sum / self.count.clamp_min(1)[..., None]
                self.bias[completed] = means[completed]
                self.ready[completed] = True
        return TareOutput(
            calibrated_wrench=wrench - self.bias,
            calibration_ready=self.ready.clone(),
        )
