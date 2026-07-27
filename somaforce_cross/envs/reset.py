"""Pure-PyTorch Phase 4B2 selected-environment reset coordination."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch

from somaforce_cross.envs.action_history import NominalActionHistory
from somaforce_cross.envs.mismatch import (
    EpisodeParameterStore,
    PerEnvRandomStream,
    validate_env_ids,
)
from somaforce_cross.force import WristHistoryBuffer
from somaforce_cross.residual import (
    ContactGainRamp,
    ExecutedActionHistory,
)
from somaforce_cross.sensing import (
    ContactDetector,
    VirtualFTSensor,
    WristTareCalibrator,
)
from somaforce_cross.sensing.virtual_ft import VirtualFTSensorParameters


class SelectedResetHook(Protocol):
    """Caller-owned scaffold, sink, or reference/adapter selected reset."""

    def validate_reset(self, env_ids: torch.Tensor) -> None:
        """Validate reset readiness without changing any caller-owned state."""

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset exactly the already-validated selected rows."""


def _selected_float32(
    value: object,
    name: str,
    *,
    rows: int,
    trailing_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    expected = (rows, *trailing_shape)
    if value.shape != expected:
        raise ValueError(
            f"{name} must have shape {list(expected)}, got {list(value.shape)}"
        )
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


class EpisodeResetCoordinator:
    """Own Phase 4 counters and coordinate the ratified reset sequence.

    Physics synchronization and construction of the first Isaac observation are
    deliberately outside this pure-PyTorch state coordinator.
    """

    def __init__(
        self,
        *,
        parameter_store: EpisodeParameterStore,
        random_stream: PerEnvRandomStream,
        scaffold: SelectedResetHook,
        action_sink: SelectedResetHook,
        reference_adapter: SelectedResetHook,
        tare_calibrator: WristTareCalibrator,
        virtual_sensor: VirtualFTSensor,
        contact_detector: ContactDetector,
        contact_ramp: ContactGainRamp,
        wrist_history: WristHistoryBuffer,
        nominal_action_history: NominalActionHistory,
        executed_action_history: ExecutedActionHistory,
    ) -> None:
        self.parameter_store = parameter_store
        self.random_stream = random_stream
        self.scaffold = scaffold
        self.action_sink = action_sink
        self.reference_adapter = reference_adapter
        self.tare_calibrator = tare_calibrator
        self.virtual_sensor = virtual_sensor
        self.contact_detector = contact_detector
        self.contact_ramp = contact_ramp
        self.wrist_history = wrist_history
        self.nominal_action_history = nominal_action_history
        self.executed_action_history = executed_action_history

        self.batch_size = parameter_store.batch_size
        self.device = parameter_store.device
        self._validate_ownership_contract()
        self.reference_step = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.long
        )
        self.episode_length = torch.zeros_like(self.reference_step)

    def _validate_ownership_contract(self) -> None:
        for name, hook in (
            ("scaffold", self.scaffold),
            ("action_sink", self.action_sink),
            ("reference_adapter", self.reference_adapter),
        ):
            for method_name in ("validate_reset", "reset"):
                if not callable(getattr(hook, method_name, None)):
                    raise TypeError(f"{name} must provide {method_name}(env_ids)")

        batch_sizes = (
            ("random_stream", self.random_stream.batch_size),
            ("tare_calibrator", self.tare_calibrator.batch_size),
            ("virtual_sensor", self.virtual_sensor.batch_size),
            ("contact_detector", self.contact_detector.batch_size),
            ("contact_ramp", self.contact_ramp.batch_size),
            ("wrist_history", self.wrist_history.batch_size),
            ("nominal_action_history", self.nominal_action_history.batch_size),
            ("executed_action_history", self.executed_action_history.batch_size),
        )
        for name, batch_size in batch_sizes:
            if batch_size != self.batch_size:
                raise ValueError(f"{name} must have batch size {self.batch_size}")

        tensors = (
            ("random_stream", self.random_stream.seeds),
            ("tare_calibrator", self.tare_calibrator.bias),
            ("virtual_sensor", self.virtual_sensor.drift_state),
            ("contact_detector", self.contact_detector.contact_probability),
            ("contact_ramp", self.contact_ramp.state),
            ("wrist_history", self.wrist_history.storage),
            ("nominal_action_history", self.nominal_action_history.storage),
            ("executed_action_history", self.executed_action_history.storage),
        )
        for name, tensor in tensors:
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on {self.device}")
        floating_tensors = tuple(
            (name, tensor) for name, tensor in tensors if tensor.is_floating_point()
        )
        for name, tensor in floating_tensors:
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} must use torch.float32")
        if (
            self.nominal_action_history.storage.data_ptr()
            == self.executed_action_history.storage.data_ptr()
        ):
            raise ValueError("nominal and executed action histories must not alias")

    @property
    def previous_a_total(self) -> torch.Tensor:
        """Expose the newest executed normalized action ``[B,23]``."""
        return self.executed_action_history.storage[:, :, 0]

    def reset(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        *,
        physics_mismatch: torch.Tensor,
        scaffold_mismatch: torch.Tensor,
        sensor_mismatch: torch.Tensor,
        seeds: torch.Tensor,
        initial_wrist_frame: torch.Tensor,
        current_a_nom: torch.Tensor,
    ) -> torch.Tensor:
        """Validate all inputs, then execute the fixed Phase 4B2 reset order."""
        ids = validate_env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        prepared_parameters = self.parameter_store.validate_rows(
            ids, physics_mismatch, scaffold_mismatch, sensor_mismatch
        )
        sensor_ids = prepared_parameters[0]
        prepared_sensor = VirtualFTSensorParameters.from_flat(prepared_parameters[3])
        random_ids, prepared_seeds = self.random_stream.validate_seeds(ids, seeds)
        initial_frame = _selected_float32(
            initial_wrist_frame,
            "initial_wrist_frame",
            rows=ids.numel(),
            trailing_shape=(2, 14),
            device=self.device,
        )
        nominal_action = _selected_float32(
            current_a_nom,
            "current_a_nom",
            rows=ids.numel(),
            trailing_shape=(23,),
            device=self.device,
        )

        self.scaffold.validate_reset(ids)
        self.action_sink.validate_reset(ids)
        self.reference_adapter.validate_reset(ids)

        if ids.numel() == 0:
            return ids

        # 2. Apply the episode's explicit parameter rows and seed streams.
        self.parameter_store._apply_validated(*prepared_parameters)
        self.virtual_sensor._apply_validated_parameters(sensor_ids, prepared_sensor)
        self.random_stream._reseed_validated(random_ids, prepared_seeds)

        # 3. Reset caller-owned runtime state in the ratified order.
        self.scaffold.reset(ids)
        self.action_sink.reset(ids)
        self.reference_adapter.reset(ids)

        # 4. Reset calibration, sensing, detection, and contact authority.
        self.tare_calibrator.reset(ids)
        self.virtual_sensor.reset(ids)
        self.contact_detector.reset(ids)
        self.contact_ramp.reset(ids)

        # 5. Clear all selected environment-owned histories.
        self.wrist_history.reset(ids)
        self.nominal_action_history.reset(ids)
        self.executed_action_history.reset(ids)

        # 6-7. Install the explicit initial wrist frame and current a_nom.
        self.wrist_history.reset(ids, initial_frame)
        self.nominal_action_history.set_current(ids, nominal_action)

        # 8-9. Executed history remains zero; then counters return to zero.
        with torch.no_grad():
            self.reference_step[ids] = 0
            self.episode_length[ids] = 0
        return ids
