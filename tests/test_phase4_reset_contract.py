from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import torch

from somaforce_cross.envs import (
    EpisodeParameterStore,
    EpisodeResetCoordinator,
    NominalActionHistory,
    PerEnvRandomStream,
    VirtualFTSensorParameters,
    validate_env_ids,
)
from somaforce_cross.force import WristHistoryBuffer
from somaforce_cross.residual import ContactGainRamp, ExecutedActionHistory
from somaforce_cross.sensing import (
    ContactDetector,
    VirtualFTSensor,
    WristTareCalibrator,
)


def _sensor_rows(count: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    rows = torch.zeros(count, 90, device=device, dtype=torch.float32)
    quaternion = torch.tensor([2.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    rows[:, 0:8] = quaternion.repeat(2).expand(count, -1)
    rows[:, 8:20] = 0.1
    rows[:, 20:32] = 0.2
    rows[:, 32:44] = 0.01
    rows[:, 44:56] = 0.02
    rows[:, 56:68] = 0.03
    if count:
        rows[:, 68:70] = torch.tensor([0.0, 3.0], device=device)
    rows[:, 70:72] = 0.4
    rows[:, 72:78] = 100.0
    rows[:, 78:84] = 10.0
    rows[:, 84:86] = 0.25
    rows[:, 86:88] = 20.0
    rows[:, 88:90] = 2.0
    return rows


def _sensor(batch_size: int, *, device: torch.device | str = "cpu") -> VirtualFTSensor:
    device = torch.device(device)
    return VirtualFTSensor(
        batch_size,
        axis_misalignment_quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        .expand(batch_size, 2, 4)
        .clone(),
        scale_error=0.0,
        additive_bias=0.0,
        drift_rate=0.0,
        drift_noise_std=0.0,
        white_noise_std=0.0,
        delay_steps=torch.zeros(batch_size, 2, device=device, dtype=torch.long),
        filter_alpha=0.0,
        force_saturation=1.0,
        torque_saturation=1.0,
        dropout_probability=0.0,
        F_scale=1.0,
        M_scale=1.0,
        seed=0,
        device=device,
        dtype=torch.float32,
    )


class _RecordingHook:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_validation: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_validation = fail_validation
        self.mutation_count = 0

    def validate_reset(self, env_ids: torch.Tensor) -> None:
        assert env_ids.dtype == torch.int64
        self.events.append(f"validate_{self.name}")
        if self.fail_validation:
            raise ValueError(f"{self.name} validation failed")

    def reset(self, env_ids: torch.Tensor) -> None:
        assert env_ids.dtype == torch.int64
        self.events.append(f"reset_{self.name}")
        self.mutation_count += 1


def _coordinator_kwargs(
    batch_size: int,
    *,
    device: torch.device,
    events: list[str],
) -> dict[str, object]:
    return {
        "parameter_store": EpisodeParameterStore(batch_size, device=device),
        "random_stream": PerEnvRandomStream(batch_size, device=device),
        "scaffold": _RecordingHook("scaffold", events),
        "action_sink": _RecordingHook("action_sink", events),
        "reference_adapter": _RecordingHook("reference_adapter", events),
        "tare_calibrator": WristTareCalibrator(
            batch_size, num_samples=2, device=device
        ),
        "virtual_sensor": _sensor(batch_size, device=device),
        "contact_detector": ContactDetector(
            batch_size,
            F_scale=1.0,
            M_scale=1.0,
            on_threshold=0.5,
            off_threshold=0.25,
            probability_temperature=0.1,
            smoothing_alpha=0.5,
            device=device,
            dtype=torch.float32,
        ),
        "contact_ramp": ContactGainRamp(
            batch_size,
            attack_step=0.1,
            release_step=0.1,
            device=device,
        ),
        "wrist_history": WristHistoryBuffer(batch_size, device=device),
        "nominal_action_history": NominalActionHistory(batch_size, device=device),
        "executed_action_history": ExecutedActionHistory(batch_size, device=device),
    }


def _coordinator(
    batch_size: int = 3,
    *,
    device: torch.device | str = "cpu",
) -> tuple[EpisodeResetCoordinator, list[str]]:
    device = torch.device(device)
    events: list[str] = []
    coordinator = EpisodeResetCoordinator(
        **_coordinator_kwargs(batch_size, device=device, events=events)
    )
    return coordinator, events


def _reset_inputs(
    count: int, *, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    return {
        "physics_mismatch": torch.arange(
            count * 39, device=device, dtype=torch.float32
        ).reshape(count, 39),
        "scaffold_mismatch": torch.arange(
            count * 2, device=device, dtype=torch.float32
        ).reshape(count, 2),
        "sensor_mismatch": _sensor_rows(count, device=device),
        "seeds": torch.arange(101, 101 + count, device=device, dtype=torch.long),
        "initial_wrist_frame": torch.arange(
            count * 2 * 14, device=device, dtype=torch.float32
        ).reshape(count, 2, 14),
        "current_a_nom": torch.arange(
            count * 23, device=device, dtype=torch.float32
        ).reshape(count, 23),
    }


def _applied_sensor_parameters(
    sensor: VirtualFTSensor, env_ids: torch.Tensor
) -> VirtualFTSensorParameters:
    limits = sensor.saturation_limit[env_ids]
    scales = sensor.normalization_scale[env_ids]
    return VirtualFTSensorParameters(
        axis_misalignment_quat=sensor.axis_misalignment_quat[env_ids],
        scale_error=sensor.scale_error[env_ids],
        additive_bias=sensor.additive_bias[env_ids],
        drift_rate=sensor.drift_rate[env_ids],
        drift_noise_std=sensor.drift_noise_std[env_ids],
        white_noise_std=sensor.white_noise_std[env_ids],
        delay_steps=sensor.delay_steps[env_ids],
        filter_alpha=sensor.filter_alpha[env_ids],
        force_saturation=limits[..., :3],
        torque_saturation=limits[..., 3:],
        dropout_probability=sensor.dropout_probability[env_ids],
        F_scale=scales[..., 0],
        M_scale=scales[..., 3],
    )


def _owned_tensors(coordinator: EpisodeResetCoordinator) -> dict[str, torch.Tensor]:
    return {
        "physics": coordinator.parameter_store.physics_mismatch,
        "scaffold": coordinator.parameter_store.scaffold_mismatch,
        "sensor": coordinator.parameter_store.sensor_mismatch,
        "sensor_axis": coordinator.virtual_sensor.axis_misalignment_quat,
        "sensor_scale": coordinator.virtual_sensor.scale_error,
        "sensor_bias": coordinator.virtual_sensor.additive_bias,
        "sensor_drift_rate": coordinator.virtual_sensor.drift_rate,
        "sensor_drift_noise": coordinator.virtual_sensor.drift_noise_std,
        "sensor_white_noise": coordinator.virtual_sensor.white_noise_std,
        "sensor_delay_steps": coordinator.virtual_sensor.delay_steps,
        "sensor_filter_alpha": coordinator.virtual_sensor.filter_alpha,
        "sensor_saturation": coordinator.virtual_sensor.saturation_limit,
        "sensor_dropout": coordinator.virtual_sensor.dropout_probability,
        "sensor_normalization": coordinator.virtual_sensor.normalization_scale,
        "drift": coordinator.virtual_sensor.drift_state,
        "delay": coordinator.virtual_sensor.delay_buffer,
        "filter": coordinator.virtual_sensor.filter_state,
        "tare_bias": coordinator.tare_calibrator.bias,
        "tare_sum": coordinator.tare_calibrator.sample_sum,
        "tare_count": coordinator.tare_calibrator.count,
        "tare_ready": coordinator.tare_calibrator.ready,
        "detector_state": coordinator.contact_detector.contact_state,
        "detector_probability": coordinator.contact_detector.contact_probability,
        "ramp": coordinator.contact_ramp.state,
        "wrist": coordinator.wrist_history.storage,
        "nominal": coordinator.nominal_action_history.storage,
        "executed": coordinator.executed_action_history.storage,
        "reference": coordinator.reference_step,
        "episode": coordinator.episode_length,
        "seeds": coordinator.random_stream.seeds,
        "seeded": coordinator.random_stream._seeded,
    }


def test_validate_env_ids_accepts_all_approved_forms_and_empty() -> None:
    device = torch.device("cpu")
    assert torch.equal(
        validate_env_ids(2, batch_size=4, device=device), torch.tensor([2])
    )
    assert torch.equal(
        validate_env_ids((3, 1), batch_size=4, device=device), torch.tensor([3, 1])
    )
    assert torch.equal(
        validate_env_ids(
            torch.tensor([2, 0], dtype=torch.int16), batch_size=4, device=device
        ),
        torch.tensor([2, 0]),
    )
    assert torch.equal(
        validate_env_ids(
            torch.tensor([False, True, False, True]),
            batch_size=4,
            device=device,
        ),
        torch.tensor([1, 3]),
    )
    empty = validate_env_ids([], batch_size=4, device=device)
    assert empty.shape == (0,) and empty.dtype == torch.int64


@pytest.mark.parametrize(
    ("env_ids", "error"),
    (
        (True, TypeError),
        ([1.0], TypeError),
        ([True], TypeError),
        (torch.tensor([1.0]), TypeError),
        (torch.tensor(1), ValueError),
        ([[1]], ValueError),
        ([1, 1], ValueError),
        ([-1], IndexError),
        ([4], IndexError),
        (torch.tensor(True), ValueError),
    ),
)
def test_validate_env_ids_rejects_invalid_inputs(
    env_ids: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        validate_env_ids(env_ids, batch_size=4, device="cpu")  # type: ignore[arg-type]


def test_nominal_action_history_is_distinct_newest_first_and_selected() -> None:
    nominal = NominalActionHistory(3)
    executed = ExecutedActionHistory(3)
    assert nominal.storage.data_ptr() != executed.storage.data_ptr()

    first = torch.arange(69, dtype=torch.float32).reshape(3, 23)
    second = first + 100.0
    nominal.push(first)
    nominal.push(second)
    assert torch.equal(nominal.storage[:, :, 0], second)
    assert torch.equal(nominal.storage[:, :, 1], first)

    before_unselected = nominal.storage[1].clone()
    selected_current = torch.full((2, 23), 7.0)
    nominal.reset(torch.tensor([2, 0]), selected_current)
    assert torch.equal(nominal.storage[[2, 0], :, 0], selected_current)
    assert torch.count_nonzero(nominal.storage[[2, 0], :, 1:]) == 0
    assert torch.equal(nominal.storage[1], before_unselected)

    before = nominal.storage.clone()
    with pytest.raises(ValueError, match="duplicates"):
        nominal.reset([0, 0], torch.zeros(2, 23))
    assert torch.equal(nominal.storage, before)


def test_coordinator_rejects_float64_wrist_history_immediately() -> None:
    events: list[str] = []
    kwargs = _coordinator_kwargs(3, device=torch.device("cpu"), events=events)
    kwargs["wrist_history"] = WristHistoryBuffer(3, dtype=torch.float64)
    with pytest.raises(TypeError, match="wrist_history.*float32"):
        EpisodeResetCoordinator(**kwargs)  # type: ignore[arg-type]


def test_coordinator_rejects_batch_mismatch_immediately() -> None:
    events: list[str] = []
    kwargs = _coordinator_kwargs(3, device=torch.device("cpu"), events=events)
    kwargs["executed_action_history"] = ExecutedActionHistory(2)
    with pytest.raises(ValueError, match="executed_action_history.*batch size 3"):
        EpisodeResetCoordinator(**kwargs)  # type: ignore[arg-type]


def test_coordinator_rejects_nominal_executed_storage_alias() -> None:
    events: list[str] = []
    kwargs = _coordinator_kwargs(3, device=torch.device("cpu"), events=events)
    executed = ExecutedActionHistory(3)
    nominal = NominalActionHistory(3)
    nominal._storage = executed.storage
    kwargs["nominal_action_history"] = nominal
    kwargs["executed_action_history"] = executed
    with pytest.raises(ValueError, match="must not alias"):
        EpisodeResetCoordinator(**kwargs)  # type: ignore[arg-type]


def test_coordinator_rejects_hooks_missing_either_contract_method() -> None:
    class ResetOnly:
        def reset(self, env_ids: torch.Tensor) -> None:
            del env_ids

    class ValidateOnly:
        def validate_reset(self, env_ids: torch.Tensor) -> None:
            del env_ids

    events: list[str] = []
    kwargs = _coordinator_kwargs(3, device=torch.device("cpu"), events=events)
    kwargs["scaffold"] = ResetOnly()
    with pytest.raises(TypeError, match="scaffold.*validate_reset"):
        EpisodeResetCoordinator(**kwargs)  # type: ignore[arg-type]

    kwargs = _coordinator_kwargs(3, device=torch.device("cpu"), events=events)
    kwargs["action_sink"] = ValidateOnly()
    with pytest.raises(TypeError, match="action_sink.*reset"):
        EpisodeResetCoordinator(**kwargs)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_coordinator_rejects_device_mismatch_immediately() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    events: list[str] = []
    kwargs = _coordinator_kwargs(3, device=torch.device("cpu"), events=events)
    kwargs["parameter_store"] = EpisodeParameterStore(3, device=device)
    with pytest.raises(ValueError, match="random_stream.*must be on"):
        EpisodeResetCoordinator(**kwargs)  # type: ignore[arg-type]


def test_episode_parameter_store_exact_width_order_and_atomicity() -> None:
    store = EpisodeParameterStore(4)
    store.physics_mismatch.fill_(-1)
    store.scaffold_mismatch.fill_(-2)
    store.sensor_mismatch.fill_(-3)
    inputs = _reset_inputs(2)
    ids = torch.tensor([3, 1])
    before_unselected = (
        store.physics_mismatch[[0, 2]].clone(),
        store.scaffold_mismatch[[0, 2]].clone(),
        store.sensor_mismatch[[0, 2]].clone(),
    )
    store.apply(
        ids,
        inputs["physics_mismatch"],
        inputs["scaffold_mismatch"],
        inputs["sensor_mismatch"],
    )
    assert store.physics_mismatch.shape == (4, 39)
    assert store.scaffold_mismatch.shape == (4, 2)
    assert store.sensor_mismatch.shape == (4, 90)
    assert torch.equal(store.physics_mismatch[ids], inputs["physics_mismatch"])
    assert torch.equal(store.scaffold_mismatch[ids], inputs["scaffold_mismatch"])
    canonical_sensor = VirtualFTSensorParameters.canonicalize(
        VirtualFTSensorParameters.from_flat(inputs["sensor_mismatch"]),
        device=store.device,
        dtype=torch.float32,
    ).flatten()
    assert torch.equal(store.sensor_mismatch[ids], canonical_sensor)
    assert torch.equal(
        store.sensor_mismatch[ids, 0:8],
        torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(2, 2),
    )
    assert torch.equal(store.physics_mismatch[[0, 2]], before_unselected[0])
    assert torch.equal(store.scaffold_mismatch[[0, 2]], before_unselected[1])
    assert torch.equal(store.sensor_mismatch[[0, 2]], before_unselected[2])

    before = (
        store.physics_mismatch.clone(),
        store.scaffold_mismatch.clone(),
        store.sensor_mismatch.clone(),
    )
    bad_sensor = inputs["sensor_mismatch"].clone()
    bad_sensor[1, 10] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        store.apply(
            ids,
            inputs["physics_mismatch"],
            inputs["scaffold_mismatch"],
            bad_sensor,
        )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (
                store.physics_mismatch,
                store.scaffold_mismatch,
                store.sensor_mismatch,
            ),
            before,
            strict=True,
        )
    )

    invalid_sensor_rows: list[torch.Tensor] = []
    for column, value in (
        (8, -1.0),
        (44, -0.1),
        (56, -0.1),
        (68, 4.0),
        (72, 0.0),
        (78, 0.0),
        (84, 1.1),
        (86, 0.0),
        (88, 0.0),
    ):
        invalid = inputs["sensor_mismatch"].clone()
        invalid[0, column] = value
        invalid_sensor_rows.append(invalid)
    for invalid in invalid_sensor_rows:
        before_invalid = (
            store.physics_mismatch.clone(),
            store.scaffold_mismatch.clone(),
            store.sensor_mismatch.clone(),
        )
        with pytest.raises(ValueError):
            store.apply(
                ids,
                inputs["physics_mismatch"],
                inputs["scaffold_mismatch"],
                invalid,
            )
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                (
                    store.physics_mismatch,
                    store.scaffold_mismatch,
                    store.sensor_mismatch,
                ),
                before_invalid,
                strict=True,
            )
        )


def test_sensor_parameter_flat_order_is_exact() -> None:
    rows = _sensor_rows(2)
    parameters = VirtualFTSensorParameters.from_flat(rows)
    assert parameters.axis_misalignment_quat.shape == (2, 2, 4)
    assert parameters.scale_error.shape == (2, 2, 6)
    assert parameters.delay_steps.shape == (2, 2)
    assert parameters.delay_steps.dtype == torch.int64
    assert torch.equal(parameters.flatten(), rows)


def test_per_env_random_stream_is_deterministic_independent_and_order_free() -> None:
    first = PerEnvRandomStream(2)
    second = PerEnvRandomStream(2)
    first.reseed([0, 1], torch.tensor([11, 22]))
    second.reseed([1, 0], torch.tensor([22, 11]))
    first_draw = first.draw_virtual_ft()
    second_draw = second.draw_virtual_ft([1, 0])
    assert torch.equal(first_draw.drift[0], second_draw.drift[1])
    assert torch.equal(first_draw.drift[1], second_draw.drift[0])
    assert torch.equal(first_draw.noise[0], second_draw.noise[1])
    assert torch.equal(first_draw.dropout[1], second_draw.dropout[0])

    control = PerEnvRandomStream(2)
    control.reseed([0, 1], torch.tensor([11, 22]))
    control.draw_virtual_ft()
    first.reseed([0], torch.tensor([999]))
    env_one_after_other_reset = first.draw_virtual_ft([1])
    env_one_control = control.draw_virtual_ft([1])
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            env_one_after_other_reset, env_one_control, strict=True
        )
    )

    before_seed = first.seeds.clone()
    with pytest.raises(ValueError, match="nonnegative"):
        first.reseed([1], torch.tensor([-1]))
    assert torch.equal(first.seeds, before_seed)
    assert first.draw_virtual_ft([]).drift.shape == (0, 2, 6)


def test_reset_coordinator_follows_order_and_changes_only_selected_rows() -> None:
    coordinator, events = _coordinator()
    selected = torch.tensor([2, 0])
    inputs = _reset_inputs(2)

    coordinator.parameter_store.physics_mismatch.fill_(-1)
    coordinator.parameter_store.scaffold_mismatch.fill_(-1)
    coordinator.parameter_store.sensor_mismatch.fill_(-1)
    coordinator.tare_calibrator.bias.fill_(1)
    coordinator.tare_calibrator.sample_sum.fill_(2)
    coordinator.tare_calibrator.count.fill_(3)
    coordinator.tare_calibrator.ready.fill_(True)
    coordinator.virtual_sensor.drift_state.fill_(4)
    coordinator.virtual_sensor.delay_buffer.fill_(5)
    coordinator.virtual_sensor.filter_state.fill_(6)
    coordinator.contact_detector.contact_state.fill_(True)
    coordinator.contact_detector.contact_probability.fill_(0.8)
    coordinator.contact_ramp.state.fill_(0.7)
    coordinator.wrist_history.storage.fill_(8)
    coordinator.nominal_action_history.storage.fill_(9)
    coordinator.executed_action_history.storage.fill_(10)
    coordinator.reference_step.fill_(11)
    coordinator.episode_length.fill_(12)

    unselected_before = {
        "physics": coordinator.parameter_store.physics_mismatch[1].clone(),
        "scaffold": coordinator.parameter_store.scaffold_mismatch[1].clone(),
        "sensor": coordinator.parameter_store.sensor_mismatch[1].clone(),
        "tare_bias": coordinator.tare_calibrator.bias[1].clone(),
        "tare_sum": coordinator.tare_calibrator.sample_sum[1].clone(),
        "tare_count": coordinator.tare_calibrator.count[1].clone(),
        "tare_ready": coordinator.tare_calibrator.ready[1].clone(),
        "drift": coordinator.virtual_sensor.drift_state[1].clone(),
        "delay": coordinator.virtual_sensor.delay_buffer[1].clone(),
        "filter": coordinator.virtual_sensor.filter_state[1].clone(),
        "detector_state": coordinator.contact_detector.contact_state[1].clone(),
        "detector_probability": coordinator.contact_detector.contact_probability[
            1
        ].clone(),
        "ramp": coordinator.contact_ramp.state[1].clone(),
        "wrist": coordinator.wrist_history.storage[1].clone(),
        "nominal": coordinator.nominal_action_history.storage[1].clone(),
        "executed": coordinator.executed_action_history.storage[1].clone(),
        "reference": coordinator.reference_step[1].clone(),
        "episode": coordinator.episode_length[1].clone(),
    }

    returned = coordinator.reset(selected, **inputs)
    assert torch.equal(returned, selected)
    assert events == [
        "validate_scaffold",
        "validate_action_sink",
        "validate_reference_adapter",
        "reset_scaffold",
        "reset_action_sink",
        "reset_reference_adapter",
    ]
    assert torch.equal(
        coordinator.parameter_store.physics_mismatch[selected],
        inputs["physics_mismatch"],
    )
    assert torch.equal(
        coordinator.parameter_store.scaffold_mismatch[selected],
        inputs["scaffold_mismatch"],
    )
    applied_sensor = _applied_sensor_parameters(
        coordinator.virtual_sensor, selected
    ).flatten()
    assert torch.equal(
        coordinator.parameter_store.sensor_mismatch[selected], applied_sensor
    )
    assert torch.equal(
        applied_sensor[:, 0:8],
        torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(2, 2),
    )
    assert torch.equal(
        coordinator.virtual_sensor.delay_steps[selected],
        torch.tensor([[0, 3], [0, 3]]),
    )
    assert torch.count_nonzero(coordinator.tare_calibrator.bias[selected]) == 0
    assert torch.count_nonzero(coordinator.virtual_sensor.drift_state[selected]) == 0
    assert not coordinator.contact_detector.contact_state[selected].any()
    assert torch.count_nonzero(coordinator.contact_ramp.state[selected]) == 0
    expected_wrist = inputs["initial_wrist_frame"][:, :, None, :].expand(-1, -1, 16, -1)
    assert torch.equal(coordinator.wrist_history.storage[selected], expected_wrist)
    assert torch.equal(
        coordinator.nominal_action_history.storage[selected, :, 0],
        inputs["current_a_nom"],
    )
    assert (
        torch.count_nonzero(coordinator.nominal_action_history.storage[selected, :, 1:])
        == 0
    )
    assert (
        torch.count_nonzero(coordinator.executed_action_history.storage[selected]) == 0
    )
    assert torch.count_nonzero(coordinator.previous_a_total[selected]) == 0
    assert torch.count_nonzero(coordinator.reference_step[selected]) == 0
    assert torch.count_nonzero(coordinator.episode_length[selected]) == 0

    actual_unselected = {
        "physics": coordinator.parameter_store.physics_mismatch[1],
        "scaffold": coordinator.parameter_store.scaffold_mismatch[1],
        "sensor": coordinator.parameter_store.sensor_mismatch[1],
        "tare_bias": coordinator.tare_calibrator.bias[1],
        "tare_sum": coordinator.tare_calibrator.sample_sum[1],
        "tare_count": coordinator.tare_calibrator.count[1],
        "tare_ready": coordinator.tare_calibrator.ready[1],
        "drift": coordinator.virtual_sensor.drift_state[1],
        "delay": coordinator.virtual_sensor.delay_buffer[1],
        "filter": coordinator.virtual_sensor.filter_state[1],
        "detector_state": coordinator.contact_detector.contact_state[1],
        "detector_probability": coordinator.contact_detector.contact_probability[1],
        "ramp": coordinator.contact_ramp.state[1],
        "wrist": coordinator.wrist_history.storage[1],
        "nominal": coordinator.nominal_action_history.storage[1],
        "executed": coordinator.executed_action_history.storage[1],
        "reference": coordinator.reference_step[1],
        "episode": coordinator.episode_length[1],
    }
    assert all(
        torch.equal(actual_unselected[name], expected)
        for name, expected in unselected_before.items()
    )


def test_coordinator_store_and_sensor_match_for_general_quaternion() -> None:
    coordinator, _ = _coordinator()
    inputs = _reset_inputs(1)
    inputs["sensor_mismatch"][:, 0:8] = torch.tensor(
        [2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0]
    )
    selected = torch.tensor([1])
    coordinator.reset(selected, **inputs)
    applied = _applied_sensor_parameters(coordinator.virtual_sensor, selected)
    assert torch.equal(
        coordinator.parameter_store.sensor_mismatch[selected], applied.flatten()
    )
    assert torch.equal(
        applied.axis_misalignment_quat.square().sum(dim=-1),
        torch.ones(1, 2),
    )


def test_reset_coordinator_records_the_complete_fixed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, events = _coordinator()

    def record_method(obj: object, method_name: str, event: str) -> None:
        original = getattr(obj, method_name)

        def recorded(*args: object, **kwargs: object) -> object:
            events.append(event)
            return original(*args, **kwargs)

        monkeypatch.setattr(obj, method_name, recorded)

    record_method(coordinator.parameter_store, "_apply_validated", "episode_parameters")
    record_method(
        coordinator.virtual_sensor,
        "_apply_validated_parameters",
        "sensor_parameters",
    )
    record_method(coordinator.random_stream, "_reseed_validated", "random_reseed")
    record_method(coordinator.tare_calibrator, "reset", "tare")
    record_method(coordinator.virtual_sensor, "reset", "virtual_sensor")
    record_method(coordinator.contact_detector, "reset", "contact_detector")
    record_method(coordinator.contact_ramp, "reset", "contact_ramp")
    record_method(coordinator.nominal_action_history, "reset", "nominal_reset")
    record_method(coordinator.executed_action_history, "reset", "executed_reset")
    record_method(coordinator.nominal_action_history, "set_current", "nominal_current")

    original_wrist_reset = coordinator.wrist_history.reset

    def recorded_wrist_reset(
        env_ids: torch.Tensor, initial_frame: torch.Tensor | None = None
    ) -> torch.Tensor:
        events.append("wrist_initial" if initial_frame is not None else "wrist_reset")
        return original_wrist_reset(env_ids, initial_frame)

    monkeypatch.setattr(coordinator.wrist_history, "reset", recorded_wrist_reset)
    coordinator.reset([1], **_reset_inputs(1))
    assert events == [
        "validate_scaffold",
        "validate_action_sink",
        "validate_reference_adapter",
        "episode_parameters",
        "sensor_parameters",
        "random_reseed",
        "reset_scaffold",
        "reset_action_sink",
        "reset_reference_adapter",
        "tare",
        "virtual_sensor",
        "contact_detector",
        "contact_ramp",
        "wrist_reset",
        "nominal_reset",
        "executed_reset",
        "wrist_initial",
        "nominal_current",
    ]


def test_second_hook_validation_failure_is_globally_atomic() -> None:
    coordinator, events = _coordinator()
    action_sink = coordinator.action_sink
    assert isinstance(action_sink, _RecordingHook)
    action_sink.fail_validation = True

    initial_seeds = torch.tensor([31, 32, 33])
    coordinator.random_stream.reseed([0, 1, 2], initial_seeds)
    coordinator.random_stream.draw_virtual_ft()
    control = PerEnvRandomStream(3)
    control.reseed([0, 1, 2], initial_seeds)
    control.draw_virtual_ft()

    tensors = _owned_tensors(coordinator)
    before = {name: tensor.clone() for name, tensor in tensors.items()}
    with pytest.raises(ValueError, match="action_sink validation failed"):
        coordinator.reset([0, 2], **_reset_inputs(2))

    assert events == ["validate_scaffold", "validate_action_sink"]
    assert all(torch.equal(tensor, before[name]) for name, tensor in tensors.items())
    hooks = (
        coordinator.scaffold,
        coordinator.action_sink,
        coordinator.reference_adapter,
    )
    assert all(
        isinstance(hook, _RecordingHook) and hook.mutation_count == 0 for hook in hooks
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            coordinator.random_stream.draw_virtual_ft([1]),
            control.draw_virtual_ft([1]),
            strict=True,
        )
    )


def test_invalid_reset_input_changes_no_state_or_random_sequence() -> None:
    coordinator, events = _coordinator()
    initial_seeds = torch.tensor([5, 6, 7])
    coordinator.random_stream.reseed([0, 1, 2], initial_seeds)
    coordinator.random_stream.draw_virtual_ft()
    control = PerEnvRandomStream(3)
    control.reseed([0, 1, 2], initial_seeds)
    control.draw_virtual_ft()

    tensors = _owned_tensors(coordinator)
    before = {name: tensor.clone() for name, tensor in tensors.items()}
    inputs = _reset_inputs(2)
    inputs["current_a_nom"][1, 0] = torch.nan
    with pytest.raises(ValueError, match="current_a_nom.*finite"):
        coordinator.reset([0, 2], **inputs)
    assert events == []
    assert all(torch.equal(tensor, before[name]) for name, tensor in tensors.items())
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            coordinator.random_stream.draw_virtual_ft([1]),
            control.draw_virtual_ft([1]),
            strict=True,
        )
    )


def test_invalid_sensor_schema_changes_no_reset_state_or_random_sequence() -> None:
    coordinator, events = _coordinator()
    initial_seeds = torch.tensor([15, 16, 17])
    coordinator.random_stream.reseed([0, 1, 2], initial_seeds)
    coordinator.random_stream.draw_virtual_ft()
    control = PerEnvRandomStream(3)
    control.reseed([0, 1, 2], initial_seeds)
    control.draw_virtual_ft()
    tensors = _owned_tensors(coordinator)
    before = {name: tensor.clone() for name, tensor in tensors.items()}
    inputs = _reset_inputs(2)
    inputs["sensor_mismatch"][0, 84] = 1.1
    with pytest.raises(ValueError, match="dropout_probability"):
        coordinator.reset([0, 2], **inputs)
    assert events == []
    assert all(torch.equal(tensor, before[name]) for name, tensor in tensors.items())
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            coordinator.random_stream.draw_virtual_ft([1]),
            control.draw_virtual_ft([1]),
            strict=True,
        )
    )


def test_empty_reset_is_validated_no_op() -> None:
    coordinator, events = _coordinator()
    inputs = _reset_inputs(0)
    before = (
        coordinator.parameter_store.physics_mismatch.clone(),
        coordinator.wrist_history.storage.clone(),
        coordinator.nominal_action_history.storage.clone(),
        coordinator.executed_action_history.storage.clone(),
    )
    ids = coordinator.reset([], **inputs)
    assert ids.shape == (0,)
    assert events == [
        "validate_scaffold",
        "validate_action_sink",
        "validate_reference_adapter",
    ]
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (
                coordinator.parameter_store.physics_mismatch,
                coordinator.wrist_history.storage,
                coordinator.nominal_action_history.storage,
                coordinator.executed_action_history.storage,
            ),
            before,
            strict=True,
        )
    )


def test_c0_reset_uses_zero_wrist_history_and_independent_action_histories() -> None:
    coordinator, _ = _coordinator(batch_size=4)
    inputs = _reset_inputs(2)
    inputs["initial_wrist_frame"].zero_()
    inputs["current_a_nom"].fill_(2.0)
    ids = torch.tensor([3, 1])
    coordinator.reset(ids, **inputs)
    assert torch.count_nonzero(coordinator.wrist_history.storage[ids]) == 0
    assert torch.equal(
        coordinator.nominal_action_history.storage[ids, :, 0],
        torch.full((2, 23), 2.0),
    )
    assert (
        torch.count_nonzero(coordinator.nominal_action_history.storage[ids, :, 1:]) == 0
    )
    assert torch.count_nonzero(coordinator.executed_action_history.storage[ids]) == 0
    assert (
        coordinator.nominal_action_history.storage.data_ptr()
        != coordinator.executed_action_history.storage.data_ptr()
    )


def test_isaac_selected_reset_prevalidates_before_scene_mutation_and_never_steps() -> (
    None
):
    path = Path(__file__).resolve().parents[1] / "somaforce_cross/envs/residual_env.py"
    source = path.read_text(encoding="utf-8")
    reset_source = source[
        source.index("def _reset_idx") : source.index("def _owned_sensor_state")
    ]
    assert reset_source.index("self._validate_reset_plan") < reset_source.index(
        "self.adapter.write_scene_reset"
    )
    assert "self.sim.step" not in reset_source
    assert "initial_wrist_frame=torch.zeros" in reset_source
    assert "stable_env_seeds(self.cfg.seed, ids" in reset_source


def test_constructor_warmup_has_one_isolated_call_site() -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = root / "somaforce_cross/envs/residual_env.py"
    source = env_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(env_path))
    call_sites: list[tuple[str, str]] = []
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for method in (
            node for node in class_node.body if isinstance(node, ast.FunctionDef)
        ):
            for node in ast.walk(method):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_warmup_golden_reset_state"
                ):
                    call_sites.append((class_node.name, method.name))
    assert call_sites == [("SomaForceResidualEnv", "__init__")]

    env_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SomaForceResidualEnv"
    )
    warmup = next(
        node
        for node in env_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_warmup_golden_reset_state"
    )
    sim_steps = [
        node
        for node in ast.walk(warmup)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "sim"
    ]
    assert len(sim_steps) == 1

    isolated_paths = [root / "somaforce_cross/envs/reset.py"]
    isolated_paths.extend(
        sorted((root / "somaforce_cross/envs/task_adapters").glob("*.py"))
    )
    assert all(
        "_warmup_golden_reset_state" not in path.read_text(encoding="utf-8")
        for path in isolated_paths
    )


def test_golden_runtime_hash_is_unchanged() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "somaforce_cross"
        / "scaffold"
        / "pretrained_hdmi_isaac.py"
    )
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "e924dadeb945278ee85fad82e31be56f78bce4e62f5bb9f575f5938f6c4868a0"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_phase4_reset_cuda_float32() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    coordinator, _ = _coordinator(device=device)
    coordinator.reset(
        torch.tensor([2, 0], device=device), **_reset_inputs(2, device=device)
    )
    assert coordinator.wrist_history.storage.device == device
    assert coordinator.wrist_history.storage.dtype == torch.float32
    assert coordinator.parameter_store.sensor_mismatch.device == device
    draws = coordinator.random_stream.draw_virtual_ft([0, 2])
    assert draws.drift.device == device and draws.drift.dtype == torch.float32
