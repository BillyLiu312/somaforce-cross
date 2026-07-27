"""Stateful, vectorized virtual wrist F/T corruption pipeline."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class VirtualFTSensorParameters:
    """Explicit selected rows in the approved 90-D sensor mismatch schema."""

    axis_misalignment_quat: torch.Tensor
    scale_error: torch.Tensor
    additive_bias: torch.Tensor
    drift_rate: torch.Tensor
    drift_noise_std: torch.Tensor
    white_noise_std: torch.Tensor
    delay_steps: torch.Tensor
    filter_alpha: torch.Tensor
    force_saturation: torch.Tensor
    torque_saturation: torch.Tensor
    dropout_probability: torch.Tensor
    F_scale: torch.Tensor
    M_scale: torch.Tensor

    @classmethod
    def from_flat(cls, rows: torch.Tensor) -> VirtualFTSensorParameters:
        """Split explicit ``[K,90]`` rows in the frozen Phase 4A field order."""
        if not isinstance(rows, torch.Tensor):
            raise TypeError("sensor_mismatch must be a torch.Tensor")
        if not rows.is_floating_point():
            raise TypeError("sensor_mismatch must have a floating-point dtype")
        if rows.ndim != 2 or rows.shape[1] != 90:
            raise ValueError("sensor_mismatch must have shape [K, 90]")
        if not torch.isfinite(rows).all():
            raise ValueError("sensor_mismatch must contain only finite values")
        count = rows.shape[0]
        delay_values = rows[:, 68:70]
        if torch.any(delay_values != torch.round(delay_values)):
            raise ValueError("sensor mismatch delay rows must contain integers")
        return cls(
            axis_misalignment_quat=rows[:, 0:8].reshape(count, 2, 4),
            scale_error=rows[:, 8:20].reshape(count, 2, 6),
            additive_bias=rows[:, 20:32].reshape(count, 2, 6),
            drift_rate=rows[:, 32:44].reshape(count, 2, 6),
            drift_noise_std=rows[:, 44:56].reshape(count, 2, 6),
            white_noise_std=rows[:, 56:68].reshape(count, 2, 6),
            delay_steps=delay_values.to(dtype=torch.long),
            filter_alpha=rows[:, 70:72].reshape(count, 2),
            force_saturation=rows[:, 72:78].reshape(count, 2, 3),
            torque_saturation=rows[:, 78:84].reshape(count, 2, 3),
            dropout_probability=rows[:, 84:86].reshape(count, 2),
            F_scale=rows[:, 86:88].reshape(count, 2),
            M_scale=rows[:, 88:90].reshape(count, 2),
        )

    def flatten(self) -> torch.Tensor:
        """Return rows in axis/scale/bias/drift/noise/delay/filter/limit order."""
        count = self.axis_misalignment_quat.shape[0]
        return torch.cat(
            (
                self.axis_misalignment_quat.reshape(count, 8),
                self.scale_error.reshape(count, 12),
                self.additive_bias.reshape(count, 12),
                self.drift_rate.reshape(count, 12),
                self.drift_noise_std.reshape(count, 12),
                self.white_noise_std.reshape(count, 12),
                self.delay_steps.to(dtype=self.axis_misalignment_quat.dtype),
                self.filter_alpha.reshape(count, 2),
                self.force_saturation.reshape(count, 6),
                self.torque_saturation.reshape(count, 6),
                self.dropout_probability.reshape(count, 2),
                self.F_scale.reshape(count, 2),
                self.M_scale.reshape(count, 2),
            ),
            dim=-1,
        )

    @classmethod
    def canonicalize(
        cls,
        parameters: VirtualFTSensorParameters,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_delay_steps: int = 3,
    ) -> VirtualFTSensorParameters:
        """Validate all fields and return the exact values used by the sensor."""
        if not isinstance(parameters, cls):
            raise TypeError("parameters must be VirtualFTSensorParameters")
        if not isinstance(parameters.axis_misalignment_quat, torch.Tensor):
            raise TypeError("axis_misalignment_quat must be a torch.Tensor")
        if parameters.axis_misalignment_quat.ndim == 0:
            raise ValueError("axis_misalignment_quat must have shape [K, 2, 4]")
        count = parameters.axis_misalignment_quat.shape[0]
        axis = _selected_parameter(
            parameters.axis_misalignment_quat,
            "axis_misalignment_quat",
            (count, 2, 4),
            device=device,
            dtype=dtype,
        )
        scale = _selected_parameter(
            parameters.scale_error,
            "scale_error",
            (count, 2, 6),
            device=device,
            dtype=dtype,
        )
        bias = _selected_parameter(
            parameters.additive_bias,
            "additive_bias",
            (count, 2, 6),
            device=device,
            dtype=dtype,
        )
        drift_rate = _selected_parameter(
            parameters.drift_rate,
            "drift_rate",
            (count, 2, 6),
            device=device,
            dtype=dtype,
        )
        drift_noise = _selected_parameter(
            parameters.drift_noise_std,
            "drift_noise_std",
            (count, 2, 6),
            device=device,
            dtype=dtype,
        )
        white_noise = _selected_parameter(
            parameters.white_noise_std,
            "white_noise_std",
            (count, 2, 6),
            device=device,
            dtype=dtype,
        )
        delay = _selected_parameter(
            parameters.delay_steps,
            "delay_steps",
            (count, 2),
            device=device,
            dtype=dtype,
            integer=True,
        )
        alpha = _selected_parameter(
            parameters.filter_alpha,
            "filter_alpha",
            (count, 2),
            device=device,
            dtype=dtype,
        )
        force_limit = _selected_parameter(
            parameters.force_saturation,
            "force_saturation",
            (count, 2, 3),
            device=device,
            dtype=dtype,
        )
        torque_limit = _selected_parameter(
            parameters.torque_saturation,
            "torque_saturation",
            (count, 2, 3),
            device=device,
            dtype=dtype,
        )
        dropout = _selected_parameter(
            parameters.dropout_probability,
            "dropout_probability",
            (count, 2),
            device=device,
            dtype=dtype,
        )
        force_scale = _selected_parameter(
            parameters.F_scale,
            "F_scale",
            (count, 2),
            device=device,
            dtype=dtype,
        )
        torque_scale = _selected_parameter(
            parameters.M_scale,
            "M_scale",
            (count, 2),
            device=device,
            dtype=dtype,
        )

        norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        if torch.any(norm == 0):
            raise ValueError("axis_misalignment_quat must contain non-zero quaternions")
        if torch.any(1.0 + scale <= 0):
            raise ValueError("1 + scale_error must be positive")
        if torch.any(drift_noise < 0) or torch.any(white_noise < 0):
            raise ValueError("noise standard deviations must be nonnegative")
        if torch.any((delay < 0) | (delay > max_delay_steps)):
            raise ValueError(f"delay_steps must be within [0, {max_delay_steps}]")
        if torch.any((alpha < 0) | (alpha > 1)):
            raise ValueError("filter_alpha must be within [0, 1]")
        if torch.any(force_limit <= 0) or torch.any(torque_limit <= 0):
            raise ValueError("saturation limits must be positive")
        if torch.any((dropout < 0) | (dropout > 1)):
            raise ValueError("dropout_probability must be within [0, 1]")
        if torch.any(force_scale <= 0) or torch.any(torque_scale <= 0):
            raise ValueError("F_scale and M_scale must be positive")
        return cls(
            axis_misalignment_quat=axis / norm,
            scale_error=scale,
            additive_bias=bias,
            drift_rate=drift_rate,
            drift_noise_std=drift_noise,
            white_noise_std=white_noise,
            delay_steps=delay,
            filter_alpha=alpha,
            force_saturation=force_limit,
            torque_saturation=torque_limit,
            dropout_probability=dropout,
            F_scale=force_scale,
            M_scale=torque_scale,
        )


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


def _selected_parameter(
    value: object,
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    integer: bool = False,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if integer:
        if value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"{name} must have an integer dtype")
        return value.to(dtype=torch.long).clone()
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value.clone()


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

    def validate_parameters(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        parameters: VirtualFTSensorParameters,
    ) -> tuple[torch.Tensor, VirtualFTSensorParameters]:
        """Validate and normalize selected rows without changing sensor state."""
        ids = _env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        prepared = VirtualFTSensorParameters.canonicalize(
            parameters,
            device=self.device,
            dtype=self.dtype,
            max_delay_steps=self.max_delay_steps,
        )
        if prepared.axis_misalignment_quat.shape[0] != ids.numel():
            raise ValueError(
                f"parameters must contain exactly {ids.numel()} selected rows"
            )
        return ids, prepared

    def _apply_validated_parameters(
        self, ids: torch.Tensor, parameters: VirtualFTSensorParameters
    ) -> None:
        force_scale = parameters.F_scale[..., None].expand(-1, -1, 3)
        torque_scale = parameters.M_scale[..., None].expand(-1, -1, 3)
        with torch.no_grad():
            self.axis_misalignment_quat[ids] = parameters.axis_misalignment_quat
            self.scale_error[ids] = parameters.scale_error
            self.additive_bias[ids] = parameters.additive_bias
            self.drift_rate[ids] = parameters.drift_rate
            self.drift_noise_std[ids] = parameters.drift_noise_std
            self.white_noise_std[ids] = parameters.white_noise_std
            self.delay_steps[ids] = parameters.delay_steps
            self.filter_alpha[ids] = parameters.filter_alpha
            self.saturation_limit[ids] = torch.cat(
                (parameters.force_saturation, parameters.torque_saturation), dim=-1
            )
            self.dropout_probability[ids] = parameters.dropout_probability
            self.normalization_scale[ids] = torch.cat(
                (force_scale, torque_scale), dim=-1
            )

    def apply_parameters(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        parameters: VirtualFTSensorParameters,
    ) -> None:
        """Atomically apply parameters without resetting any dynamic state."""
        ids, prepared = self.validate_parameters(env_ids, parameters)
        self._apply_validated_parameters(ids, prepared)

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
