"""Stateful, vectorized contact detection from physical wrist wrench."""

from __future__ import annotations

from typing import Iterable, NamedTuple

import torch

from somaforce_cross.sensing.tare import _env_ids


class ContactDetectorOutput(NamedTuple):
    r_force: torch.Tensor
    r_torque: torch.Tensor
    strength: torch.Tensor
    contact_state: torch.Tensor
    contact_probability: torch.Tensor


def _finite_positive_scalar(
    value: float | torch.Tensor,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if not torch.isfinite(result).all() or torch.any(result <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return result


class ContactDetector:
    """Detect two-wrist contact with hysteresis and probability smoothing."""

    def __init__(
        self,
        batch_size: int,
        *,
        F_scale: float | torch.Tensor,
        M_scale: float | torch.Tensor,
        on_threshold: float,
        off_threshold: float,
        probability_temperature: float,
        smoothing_alpha: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point dtype")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.F_scale = _finite_positive_scalar(
            F_scale, "F_scale", device=self.device, dtype=dtype
        )
        self.M_scale = _finite_positive_scalar(
            M_scale, "M_scale", device=self.device, dtype=dtype
        )

        for value, name in (
            (on_threshold, "on_threshold"),
            (off_threshold, "off_threshold"),
            (probability_temperature, "probability_temperature"),
            (smoothing_alpha, "smoothing_alpha"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real scalar")
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= off_threshold < on_threshold:
            raise ValueError(
                "thresholds must satisfy 0 <= off_threshold < on_threshold"
            )
        if probability_temperature <= 0:
            raise ValueError("probability_temperature must be positive")
        if not 0 <= smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be within [0, 1]")
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.probability_temperature = float(probability_temperature)
        self.smoothing_alpha = float(smoothing_alpha)
        self.contact_state = torch.zeros(
            batch_size, 2, device=self.device, dtype=torch.bool
        )
        self.contact_probability = torch.zeros(
            batch_size, 2, device=self.device, dtype=dtype
        )

    def reset(self, env_ids: int | Iterable[int] | torch.Tensor) -> None:
        ids = _env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        with torch.no_grad():
            self.contact_state[ids] = False
            self.contact_probability[ids] = 0

    def forward(
        self, physical_wrench: torch.Tensor, sample_valid: torch.Tensor
    ) -> ContactDetectorOutput:
        if not isinstance(physical_wrench, torch.Tensor):
            raise TypeError("physical_wrench must be a torch.Tensor")
        if not physical_wrench.is_floating_point():
            raise TypeError("physical_wrench must have a floating-point dtype")
        if physical_wrench.shape != (self.batch_size, 2, 6):
            raise ValueError(
                f"physical_wrench must have shape [{self.batch_size}, 2, 6]"
            )
        if physical_wrench.device != self.device or physical_wrench.dtype != self.dtype:
            raise ValueError("physical_wrench must match detector device and dtype")
        if (
            not isinstance(sample_valid, torch.Tensor)
            or sample_valid.dtype != torch.bool
        ):
            raise TypeError("sample_valid must be a boolean torch.Tensor")
        if sample_valid.shape != (self.batch_size, 2):
            raise ValueError(f"sample_valid must have shape [{self.batch_size}, 2]")
        if sample_valid.device != self.device:
            raise ValueError("sample_valid must match detector device")
        finite = torch.isfinite(physical_wrench).all(dim=-1)
        if torch.any(sample_valid & ~finite):
            raise ValueError("physical_wrench must be finite at valid samples")

        wrench = torch.where(sample_valid[..., None], physical_wrench, 0.0)
        r_force = torch.linalg.vector_norm(wrench[..., :3], dim=-1) / self.F_scale
        r_torque = torch.linalg.vector_norm(wrench[..., 3:], dim=-1) / self.M_scale
        strength = torch.sqrt(0.5 * (r_force.square() + r_torque.square()))
        threshold = torch.where(
            self.contact_state, self.off_threshold, self.on_threshold
        )
        next_state = strength >= threshold
        raw_probability = torch.sigmoid(
            (strength - threshold) / self.probability_temperature
        )
        next_probability = (
            self.smoothing_alpha * self.contact_probability
            + (1.0 - self.smoothing_alpha) * raw_probability
        )
        with torch.no_grad():
            self.contact_state.copy_(
                torch.where(sample_valid, next_state, self.contact_state)
            )
            self.contact_probability.copy_(
                torch.where(
                    sample_valid, next_probability, self.contact_probability
                ).detach()
            )
        return ContactDetectorOutput(
            r_force=r_force,
            r_torque=r_torque,
            strength=strength,
            contact_state=self.contact_state.clone(),
            contact_probability=self.contact_probability.clone(),
        )

    __call__ = forward
