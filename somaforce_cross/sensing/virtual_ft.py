"""Stateful, vectorized virtual wrist F/T corruption pipeline."""

from __future__ import annotations

from typing import Iterable, NamedTuple

import torch

from somaforce_cross.sensing.tare import _env_ids


class VirtualFTOutput(NamedTuple):
    clean_total_wrench: torch.Tensor
    calibrated_wrench: torch.Tensor
    observed_wrench_physical: torch.Tensor
    normalized_wrench: torch.Tensor
    sensor_quality: torch.Tensor
    saturation_mask: torch.Tensor
    dropout_mask: torch.Tensor


def _expand_parameter(
    value: float | torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    positive: bool = False,
    nonnegative: bool = False,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    result = torch.as_tensor(value, device=device, dtype=dtype)
    try:
        result = torch.broadcast_to(result, shape).clone()
    except RuntimeError as exc:
        raise ValueError(f"{name} must be broadcastable to {list(shape)}") from exc
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and torch.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and torch.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _quat_apply(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.linalg.cross(q_xyz, vector, dim=-1)
    return vector + q[..., :1] * t + torch.linalg.cross(q_xyz, t, dim=-1)


class VirtualFTSensor:
    """Apply the explicitly configured V1 physical corruption sequence.

    ``axis_misalignment_quat`` actively rotates nominal base-yaw sensor vectors
    into the misaligned sensor axes. ``scale_error`` is fractional and produces
    the multiplier ``1 + scale_error``. Bias, drift, noise, and saturation use
    physical ``[N, Nm]`` channel units; drift rate and random-walk standard
    deviation are per call. ``delay_steps`` is per env/wrist, and filtering is
    ``y = alpha * y_prev + (1 - alpha) * x`` from zero state after reset.

    The caller supplies both the diagnostic clean total wrench and the tare-
    calibrated wrench. Optional external compensation is subtracted from the
    latter before corruption and must use the same algorithm in simulation and
    hardware. This class does not estimate gravity or inertia.
    """

    max_delay_steps = 3

    def __init__(
        self,
        batch_size: int,
        *,
        axis_misalignment_quat: torch.Tensor,
        scale_error: float | torch.Tensor,
        additive_bias: float | torch.Tensor,
        drift_rate: float | torch.Tensor,
        drift_noise_std: float | torch.Tensor,
        white_noise_std: float | torch.Tensor,
        delay_steps: torch.Tensor,
        filter_alpha: float | torch.Tensor,
        force_saturation: float | torch.Tensor,
        torque_saturation: float | torch.Tensor,
        dropout_probability: float | torch.Tensor,
        F_scale: float | torch.Tensor,
        M_scale: float | torch.Tensor,
        seed: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point dtype")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.batch_size = batch_size
        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        self.seed = seed
        wrench_shape = (batch_size, 2, 6)
        wrist_shape = (batch_size, 2)

        if not isinstance(axis_misalignment_quat, torch.Tensor):
            raise TypeError("axis_misalignment_quat must be a torch.Tensor")
        if axis_misalignment_quat.device != self.device:
            raise ValueError("axis_misalignment_quat must match the sensor device")
        if axis_misalignment_quat.dtype != dtype:
            raise ValueError("axis_misalignment_quat must match the sensor dtype")
        try:
            quaternion = torch.broadcast_to(
                axis_misalignment_quat, (batch_size, 2, 4)
            ).clone()
        except RuntimeError as exc:
            raise ValueError(
                "axis_misalignment_quat must be broadcastable to [B, 2, 4]"
            ) from exc
        if not torch.isfinite(quaternion).all():
            raise ValueError("axis_misalignment_quat must contain only finite values")
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        if torch.any(norm == 0):
            raise ValueError("axis_misalignment_quat must contain non-zero quaternions")
        self.axis_misalignment_quat = quaternion / norm

        self.scale_error = _expand_parameter(
            scale_error, "scale_error", wrench_shape, device=self.device, dtype=dtype
        )
        if torch.any(1.0 + self.scale_error <= 0):
            raise ValueError("1 + scale_error must be positive")
        self.additive_bias = _expand_parameter(
            additive_bias,
            "additive_bias",
            wrench_shape,
            device=self.device,
            dtype=dtype,
        )
        self.drift_rate = _expand_parameter(
            drift_rate, "drift_rate", wrench_shape, device=self.device, dtype=dtype
        )
        self.drift_noise_std = _expand_parameter(
            drift_noise_std,
            "drift_noise_std",
            wrench_shape,
            device=self.device,
            dtype=dtype,
            nonnegative=True,
        )
        self.white_noise_std = _expand_parameter(
            white_noise_std,
            "white_noise_std",
            wrench_shape,
            device=self.device,
            dtype=dtype,
            nonnegative=True,
        )

        if not isinstance(delay_steps, torch.Tensor) or delay_steps.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("delay_steps must be an integer torch.Tensor")
        if delay_steps.device != self.device or delay_steps.shape != wrist_shape:
            raise ValueError(
                f"delay_steps must have shape {list(wrist_shape)} on {self.device}"
            )
        if torch.any((delay_steps < 0) | (delay_steps > self.max_delay_steps)):
            raise ValueError("delay_steps must be within [0, 3]")
        self.delay_steps = delay_steps.to(dtype=torch.long).clone()

        self.filter_alpha = _expand_parameter(
            filter_alpha, "filter_alpha", wrist_shape, device=self.device, dtype=dtype
        )
        if torch.any((self.filter_alpha < 0) | (self.filter_alpha > 1)):
            raise ValueError("filter_alpha must be within [0, 1]")
        force_limit = _expand_parameter(
            force_saturation,
            "force_saturation",
            (batch_size, 2, 3),
            device=self.device,
            dtype=dtype,
            positive=True,
        )
        torque_limit = _expand_parameter(
            torque_saturation,
            "torque_saturation",
            (batch_size, 2, 3),
            device=self.device,
            dtype=dtype,
            positive=True,
        )
        self.saturation_limit = torch.cat((force_limit, torque_limit), dim=-1)
        self.dropout_probability = _expand_parameter(
            dropout_probability,
            "dropout_probability",
            wrist_shape,
            device=self.device,
            dtype=dtype,
        )
        if torch.any((self.dropout_probability < 0) | (self.dropout_probability > 1)):
            raise ValueError("dropout_probability must be within [0, 1]")
        force_scale = _expand_parameter(
            F_scale,
            "F_scale",
            wrist_shape,
            device=self.device,
            dtype=dtype,
            positive=True,
        )
        torque_scale = _expand_parameter(
            M_scale,
            "M_scale",
            wrist_shape,
            device=self.device,
            dtype=dtype,
            positive=True,
        )
        self.normalization_scale = torch.cat(
            (
                force_scale[..., None].expand(-1, -1, 3),
                torque_scale[..., None].expand(-1, -1, 3),
            ),
            dim=-1,
        )

        self.drift_state = torch.zeros(wrench_shape, device=self.device, dtype=dtype)
        self.delay_buffer = torch.zeros(
            batch_size, 2, 4, 6, device=self.device, dtype=dtype
        )
        self.filter_state = torch.zeros(wrench_shape, device=self.device, dtype=dtype)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

    def reset(self, env_ids: int | Iterable[int] | torch.Tensor) -> None:
        ids = _env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        with torch.no_grad():
            self.drift_state[ids] = 0
            self.delay_buffer[ids] = 0
            self.filter_state[ids] = 0

    def _wrench(self, value: object, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if value.shape != (self.batch_size, 2, 6):
            raise ValueError(f"{name} must have shape [{self.batch_size}, 2, 6]")
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError(f"{name} must match the sensor device and dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return value

    def _factor(
        self, value: object, name: str, *, boolean: bool = False
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != (self.batch_size, 2) or value.device != self.device:
            raise ValueError(
                f"{name} must have shape [{self.batch_size}, 2] on {self.device}"
            )
        if boolean:
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must be boolean")
            return value
        if not value.is_floating_point() or value.dtype != self.dtype:
            raise TypeError(f"{name} must have sensor floating-point dtype")
        if not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)):
            raise ValueError(f"{name} must be finite and within [0, 1]")
        return value

    def forward(
        self,
        clean_total_wrench: torch.Tensor,
        calibrated_wrench: torch.Tensor,
        calibration_ready: torch.Tensor,
        calibration_health: torch.Tensor,
        valid: torch.Tensor,
        *,
        external_compensation: torch.Tensor | None = None,
        drift_draw: torch.Tensor | None = None,
        noise_draw: torch.Tensor | None = None,
        dropout_draw: torch.Tensor | None = None,
    ) -> VirtualFTOutput:
        clean_total_wrench = self._wrench(clean_total_wrench, "clean_total_wrench")
        calibrated_wrench = self._wrench(calibrated_wrench, "calibrated_wrench")
        calibration_ready = self._factor(
            calibration_ready, "calibration_ready", boolean=True
        )
        calibration_health = self._factor(calibration_health, "calibration_health")
        valid = self._factor(valid, "valid")
        if external_compensation is None:
            external_compensation = torch.zeros_like(calibrated_wrench)
        else:
            external_compensation = self._wrench(
                external_compensation, "external_compensation"
            )
        calibrated_wrench = calibrated_wrench - external_compensation

        if drift_draw is None:
            drift_draw = torch.randn(
                calibrated_wrench.shape,
                device=self.device,
                dtype=self.dtype,
                generator=self._generator,
            )
        else:
            drift_draw = self._wrench(drift_draw, "drift_draw")
        if noise_draw is None:
            noise_draw = torch.randn(
                calibrated_wrench.shape,
                device=self.device,
                dtype=self.dtype,
                generator=self._generator,
            )
        else:
            noise_draw = self._wrench(noise_draw, "noise_draw")

        force = _quat_apply(self.axis_misalignment_quat, calibrated_wrench[..., :3])
        moment = _quat_apply(self.axis_misalignment_quat, calibrated_wrench[..., 3:])
        corrupted = torch.cat((force, moment), dim=-1)
        corrupted = corrupted * (1.0 + self.scale_error)
        corrupted = corrupted + self.additive_bias
        with torch.no_grad():
            self.drift_state.add_(self.drift_rate + self.drift_noise_std * drift_draw)
        corrupted = corrupted + self.drift_state
        corrupted = corrupted + self.white_noise_std * noise_draw

        with torch.no_grad():
            self.delay_buffer[:, :, 1:].copy_(self.delay_buffer[:, :, :-1].clone())
            self.delay_buffer[:, :, 0].copy_(corrupted.detach())
        gather_index = self.delay_steps[..., None, None].expand(-1, -1, 1, 6)
        delayed = torch.gather(self.delay_buffer, 2, gather_index).squeeze(2)
        filtered = (
            self.filter_alpha[..., None] * self.filter_state
            + (1.0 - self.filter_alpha[..., None]) * delayed
        )
        with torch.no_grad():
            self.filter_state.copy_(filtered.detach())

        saturation_mask = filtered.abs() > self.saturation_limit
        saturated = torch.clamp(
            filtered, min=-self.saturation_limit, max=self.saturation_limit
        )
        if dropout_draw is None:
            dropout_draw = torch.rand(
                (self.batch_size, 2),
                device=self.device,
                dtype=self.dtype,
                generator=self._generator,
            )
        else:
            dropout_draw = self._factor(dropout_draw, "dropout_draw")
        dropout_mask = dropout_draw < self.dropout_probability
        observed = torch.where(dropout_mask[..., None], 0.0, saturated)
        normalized = observed / self.normalization_scale

        saturated_fraction = saturation_mask.to(dtype=self.dtype).mean(dim=-1)
        quality = (
            valid
            * calibration_ready.to(dtype=self.dtype)
            * calibration_health
            * (1.0 - saturated_fraction)
            * (~dropout_mask).to(dtype=self.dtype)
        ).clamp(0.0, 1.0)
        return VirtualFTOutput(
            clean_total_wrench=clean_total_wrench,
            calibrated_wrench=calibrated_wrench,
            observed_wrench_physical=observed,
            normalized_wrench=normalized,
            sensor_quality=quality,
            saturation_mask=saturation_mask,
            dropout_mask=dropout_mask,
        )

    __call__ = forward
