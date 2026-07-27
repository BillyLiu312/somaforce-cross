from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from somaforce_cross.sensing import (
    VirtualFTSensor,
    WristTareCalibrator,
    WristWrenchTransform,
)
from somaforce_cross.sensing.virtual_ft import VirtualFTSensorParameters


def _quat_z(angle: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    half = torch.tensor(angle / 2.0, dtype=dtype)
    return torch.stack(
        (
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
        )
    )


def _quat_y(angle: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    half = torch.tensor(angle / 2.0, dtype=dtype)
    return torch.stack(
        (
            torch.cos(half),
            torch.zeros_like(half),
            torch.sin(half),
            torch.zeros_like(half),
        )
    )


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _rotate(q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    while q.ndim < value.ndim:
        q = q.unsqueeze(0)
    q = q.expand(*value.shape[:-1], 4)
    xyz = q[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, value, dim=-1)
    return value + q[..., :1] * t + torch.linalg.cross(xyz, t, dim=-1)


def _transform_inputs(
    batch_size: int = 1, *, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(batch_size, 2, 6, dtype=dtype),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
        .expand(batch_size, 2, 4)
        .clone(),
        torch.zeros(batch_size, 2, 3, dtype=dtype),
        torch.zeros(batch_size, 2, 3, dtype=dtype),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype).expand(batch_size, 4).clone(),
    )


def _sensor(
    batch_size: int = 1,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    **overrides: object,
) -> VirtualFTSensor:
    device = torch.device(device)
    parameters: dict[str, object] = {
        "axis_misalignment_quat": torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype
        )
        .expand(batch_size, 2, 4)
        .clone(),
        "scale_error": 0.0,
        "additive_bias": 0.0,
        "drift_rate": 0.0,
        "drift_noise_std": 0.0,
        "white_noise_std": 0.0,
        "delay_steps": torch.zeros(batch_size, 2, device=device, dtype=torch.long),
        "filter_alpha": 0.0,
        "force_saturation": 1.0e6,
        "torque_saturation": 1.0e6,
        "dropout_probability": 0.0,
        "F_scale": 1.0,
        "M_scale": 1.0,
        "seed": 123,
    }
    parameters.update(overrides)
    return VirtualFTSensor(
        batch_size,
        device=device,
        dtype=dtype,
        **parameters,  # type: ignore[arg-type]
    )


def _run(
    sensor: VirtualFTSensor,
    calibrated: torch.Tensor,
    **kwargs: object,
):
    ready = torch.ones(sensor.batch_size, 2, device=sensor.device, dtype=torch.bool)
    health = torch.ones(sensor.batch_size, 2, device=sensor.device, dtype=sensor.dtype)
    valid = torch.ones_like(health)
    return sensor(
        calibrated.clone(),
        calibrated,
        ready,
        health,
        valid,
        **kwargs,  # type: ignore[arg-type]
    )


def _selected_sensor_parameters(
    count: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> VirtualFTSensorParameters:
    rows = torch.zeros(count, 90, device=device, dtype=dtype)
    rows[:, 0:8] = torch.tensor(
        [2.0, 0.0, 0.0, 0.0], device=device, dtype=dtype
    ).repeat(2)
    rows[:, 8:20] = 0.1
    rows[:, 20:32] = 0.2
    rows[:, 32:44] = 0.01
    rows[:, 44:56] = 0.02
    rows[:, 56:68] = 0.03
    rows[:, 68:70] = torch.tensor([1.0, 3.0], device=device, dtype=dtype)
    rows[:, 70:72] = 0.4
    rows[:, 72:78] = 100.0
    rows[:, 78:84] = 10.0
    rows[:, 84:86] = 0.2
    rows[:, 86:88] = 20.0
    rows[:, 88:90] = 2.0
    return VirtualFTSensorParameters.from_flat(rows)


def test_wrench_transform_sign_rotation_moment_shift_and_wrist_order() -> None:
    raw, joint_quat, joint_pos, sensor_pos, pelvis_quat = _transform_inputs()
    raw[0, 0, :3] = torch.tensor([1.0, 0.0, 0.0])
    raw[0, 1, :3] = torch.tensor([0.0, 1.0, 0.0])
    joint_quat[0, 0] = _quat_z(torch.pi / 2)
    joint_pos[0, 1, 0] = 1.0
    output = WristWrenchTransform()(
        raw, joint_quat * 3.0, joint_pos, sensor_pos, pelvis_quat * 2.0
    )
    assert torch.allclose(
        output[0, 0],
        torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=output.dtype),
        atol=1e-7,
    )
    assert torch.allclose(
        output[0, 1],
        torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, -1.0], dtype=output.dtype),
        atol=1e-7,
    )


def test_wrench_transform_arbitrary_pose_yaw_invariance() -> None:
    raw, joint_quat, joint_pos, sensor_pos, pelvis_quat = _transform_inputs()
    raw[0, 0] = torch.tensor([1.0, -2.0, 0.5, 0.3, -0.2, 0.8])
    raw[0, 1] = torch.tensor([-0.4, 1.2, 2.0, -0.7, 0.1, 0.4])
    joint_quat[0, 0] = _quat_y(0.4)
    joint_quat[0, 1] = _quat_mul(_quat_z(-0.3), _quat_y(0.2))
    joint_pos[0] = torch.tensor([[0.7, -0.2, 1.1], [-0.3, 0.5, 0.9]])
    sensor_pos[0] = torch.tensor([[0.6, -0.1, 1.0], [-0.4, 0.3, 1.0]])
    pelvis_quat[0] = _quat_mul(_quat_z(0.5), _quat_y(-0.25))
    transform = WristWrenchTransform()
    expected = transform(raw, joint_quat, joint_pos, sensor_pos, pelvis_quat)

    world_yaw = _quat_z(1.1)
    rotated = transform(
        raw,
        _quat_mul(world_yaw.expand(1, 2, 4), joint_quat),
        _rotate(world_yaw, joint_pos),
        _rotate(world_yaw, sensor_pos),
        _quat_mul(world_yaw.expand(1, 4), pelvis_quat),
    )
    assert torch.allclose(rotated, expected, atol=1e-7, rtol=1e-7)
    assert torch.isfinite(rotated).all()


def test_wrench_transform_rejects_shape_nonfinite_zero_quaternion_and_mismatch() -> (
    None
):
    values = list(_transform_inputs())
    with pytest.raises(ValueError):
        WristWrenchTransform()(*([torch.zeros(1, 2, 5)] + values[1:]))
    values[1][0, 0] = 0
    with pytest.raises(ValueError, match="non-zero"):
        WristWrenchTransform()(*values)
    values = list(_transform_inputs())
    values[0][0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        WristWrenchTransform()(*values)
    values = list(_transform_inputs())
    with pytest.raises(ValueError, match="dtype"):
        WristWrenchTransform()(values[0].float(), *values[1:])


def test_tare_mask_freeze_and_partial_reset() -> None:
    tare = WristTareCalibrator(2, num_samples=2, dtype=torch.float64)
    first = torch.zeros(2, 2, 6, dtype=torch.float64)
    first[0, 0] = 2.0
    first[1, 1] = 10.0
    mask = torch.tensor([[True, False], [False, True]])
    output = tare.update(first, mask)
    assert not output.calibration_ready.any()
    second = first.clone()
    second[0, 0] = 4.0
    second[1, 1] = 14.0
    output = tare.update(second, mask)
    assert torch.equal(output.calibration_ready, mask)
    assert torch.equal(tare.bias[0, 0], torch.full((6,), 3.0, dtype=torch.float64))
    assert torch.equal(tare.bias[1, 1], torch.full((6,), 12.0, dtype=torch.float64))
    tare.update(torch.full_like(second, 100.0), torch.ones_like(mask))
    assert torch.equal(tare.bias[0, 0], torch.full((6,), 3.0, dtype=torch.float64))
    assert torch.equal(tare.bias[1, 1], torch.full((6,), 12.0, dtype=torch.float64))
    tare.reset([0])
    assert torch.count_nonzero(tare.bias[0]) == 0
    assert torch.count_nonzero(tare.count[0]) == 0
    assert not tare.ready[0].any()
    assert tare.ready[1, 1]


def test_tare_requires_explicit_samples_and_valid_inputs() -> None:
    with pytest.raises(TypeError):
        WristTareCalibrator(1)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        WristTareCalibrator(1, num_samples=0)
    tare = WristTareCalibrator(1, num_samples=1)
    with pytest.raises(TypeError):
        tare.update(torch.zeros(1, 2, 6), torch.ones(1, 2))
    with pytest.raises(ValueError):
        tare.update(torch.zeros(1, 2, 5), torch.ones(1, 2, dtype=torch.bool))


def test_virtual_ft_axis_scale_bias_drift_noise_and_compensation_in_order() -> None:
    quaternion = _quat_z(torch.pi / 2).expand(1, 2, 4).clone()
    sensor = _sensor(
        axis_misalignment_quat=quaternion,
        scale_error=1.0,
        additive_bias=1.0,
        drift_rate=0.5,
        drift_noise_std=0.25,
        white_noise_std=0.5,
    )
    value = torch.zeros(1, 2, 6, dtype=torch.float64)
    value[..., 0] = 3.0
    compensation = torch.zeros_like(value)
    compensation[..., 0] = 1.0
    output = _run(
        sensor,
        value,
        external_compensation=compensation,
        drift_draw=torch.ones_like(value) * 2.0,
        noise_draw=torch.ones_like(value) * 4.0,
        dropout_draw=torch.ones(1, 2, dtype=torch.float64),
    )
    expected = torch.full_like(value, 4.0)
    expected[..., 1] = 8.0
    assert torch.allclose(
        output.calibrated_wrench[..., 0], torch.full((1, 2), 2.0, dtype=torch.float64)
    )
    assert torch.allclose(output.observed_wrench_physical, expected, atol=1e-7)


@pytest.mark.parametrize(
    ("overrides", "input_value", "expected", "draws"),
    [
        (
            {"axis_misalignment_quat": _quat_z(torch.pi / 2).expand(1, 2, 4).clone()},
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            {},
        ),
        ({"scale_error": 0.5}, [2.0] * 6, [3.0] * 6, {}),
        ({"additive_bias": 1.5}, [2.0] * 6, [3.5] * 6, {}),
        ({"drift_rate": 0.5}, [2.0] * 6, [2.5] * 6, {}),
        (
            {"drift_noise_std": 0.25},
            [2.0] * 6,
            [2.5] * 6,
            {"drift_draw": 2.0},
        ),
        (
            {"white_noise_std": 0.5},
            [2.0] * 6,
            [3.0] * 6,
            {"noise_draw": 2.0},
        ),
    ],
)
def test_virtual_ft_each_analog_corruption_isolated(
    overrides: dict[str, object],
    input_value: list[float],
    expected: list[float],
    draws: dict[str, float],
) -> None:
    sensor = _sensor(**overrides)
    value = torch.tensor(input_value, dtype=torch.float64).expand(1, 2, 6).clone()
    call_kwargs: dict[str, torch.Tensor] = {
        "drift_draw": torch.zeros_like(value),
        "noise_draw": torch.zeros_like(value),
        "dropout_draw": torch.ones(1, 2, dtype=torch.float64),
    }
    for name, draw in draws.items():
        call_kwargs[name] = torch.full_like(value, draw)
    output = _run(sensor, value, **call_kwargs)
    expected_tensor = torch.tensor(expected, dtype=torch.float64).expand_as(value)
    assert torch.allclose(output.observed_wrench_physical, expected_tensor, atol=1e-7)


def test_virtual_ft_per_wrist_delay_zero_to_three_steps() -> None:
    delays = torch.tensor([[0, 1], [2, 3]])
    sensor = _sensor(2, delay_steps=delays)
    outputs = []
    for sample in (1.0, 2.0, 3.0, 4.0):
        value = torch.full((2, 2, 6), sample, dtype=torch.float64)
        outputs.append(_run(sensor, value).observed_wrench_physical[..., 0])
    assert torch.equal(
        outputs[0], torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    )
    assert torch.equal(
        outputs[1], torch.tensor([[2.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    )
    assert torch.equal(
        outputs[2], torch.tensor([[3.0, 2.0], [1.0, 0.0]], dtype=torch.float64)
    )
    assert torch.equal(
        outputs[3], torch.tensor([[4.0, 3.0], [2.0, 1.0]], dtype=torch.float64)
    )


def test_virtual_ft_filter_formula_and_partial_reset_clear_state() -> None:
    sensor = _sensor(2, filter_alpha=0.5, drift_rate=1.0)
    value = torch.full((2, 2, 6), 2.0, dtype=torch.float64)
    first = _run(sensor, value).observed_wrench_physical
    second = _run(sensor, value).observed_wrench_physical
    assert torch.allclose(first, torch.full_like(first, 1.5))
    assert torch.allclose(second, torch.full_like(second, 2.75))
    before = sensor.filter_state[1].clone()
    sensor.reset([0])
    assert torch.count_nonzero(sensor.drift_state[0]) == 0
    assert torch.count_nonzero(sensor.delay_buffer[0]) == 0
    assert torch.count_nonzero(sensor.filter_state[0]) == 0
    assert torch.equal(sensor.filter_state[1], before)


def test_virtual_ft_saturation_dropout_normalization_and_quality() -> None:
    sensor = _sensor(
        force_saturation=2.0,
        torque_saturation=4.0,
        dropout_probability=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        F_scale=2.0,
        M_scale=4.0,
    )
    value = torch.tensor(
        [[[3.0, 1.0, -1.0, 8.0, 2.0, -2.0], [1.0] * 6]], dtype=torch.float64
    )
    ready = torch.ones(1, 2, dtype=torch.bool)
    health = torch.tensor([[0.8, 1.0]], dtype=torch.float64)
    valid = torch.ones(1, 2, dtype=torch.float64)
    output = sensor(
        value,
        value,
        ready,
        health,
        valid,
        drift_draw=torch.zeros_like(value),
        noise_draw=torch.zeros_like(value),
        dropout_draw=torch.tensor([[0.5, 0.5]], dtype=torch.float64),
    )
    assert torch.equal(
        output.saturation_mask[0, 0],
        torch.tensor([True, False, False, True, False, False]),
    )
    assert torch.allclose(
        output.observed_wrench_physical[0, 0],
        torch.tensor([2.0, 1.0, -1.0, 4.0, 2.0, -2.0], dtype=torch.float64),
    )
    assert torch.allclose(
        output.normalized_wrench[0, 0],
        torch.tensor([1.0, 0.5, -0.5, 1.0, 0.5, -0.5], dtype=torch.float64),
    )
    assert output.dropout_mask[0, 1]
    assert torch.count_nonzero(output.observed_wrench_physical[0, 1]) == 0
    assert torch.allclose(
        output.sensor_quality,
        torch.tensor([[0.8 * (4.0 / 6.0), 0.0]], dtype=torch.float64),
    )


def test_virtual_ft_quality_uses_only_declared_factors() -> None:
    value = torch.ones(1, 2, 6, dtype=torch.float64)
    sensors = [
        _sensor(),
        _sensor(
            scale_error=0.5, additive_bias=2.0, drift_rate=1.0, white_noise_std=3.0
        ),
    ]
    qualities = []
    for sensor in sensors:
        output = _run(
            sensor,
            value,
            drift_draw=torch.ones_like(value),
            noise_draw=torch.ones_like(value),
            dropout_draw=torch.ones(1, 2, dtype=torch.float64),
        )
        qualities.append(output.sensor_quality)
    assert torch.equal(qualities[0], qualities[1])
    assert torch.equal(qualities[0], torch.ones_like(qualities[0]))

    sensor = _sensor()
    ready = torch.tensor([[True, False]])
    health = torch.tensor([[0.25, 1.0]], dtype=torch.float64)
    valid = torch.tensor([[0.5, 1.0]], dtype=torch.float64)
    output = sensor(value, value, ready, health, valid)
    assert torch.equal(
        output.sensor_quality, torch.tensor([[0.125, 0.0]], dtype=torch.float64)
    )


def test_virtual_ft_seed_is_reproducible() -> None:
    kwargs = {
        "drift_noise_std": 0.2,
        "white_noise_std": 0.4,
        "dropout_probability": 0.5,
        "seed": 77,
    }
    first = _sensor(**kwargs)
    second = _sensor(**kwargs)
    value = torch.ones(1, 2, 6, dtype=torch.float64)
    for _ in range(3):
        output_a = _run(first, value)
        output_b = _run(second, value)
        assert torch.equal(
            output_a.observed_wrench_physical, output_b.observed_wrench_physical
        )
        assert torch.equal(output_a.dropout_mask, output_b.dropout_mask)


def test_virtual_ft_scientific_parameters_are_explicit_and_errors_are_rejected() -> (
    None
):
    signature = inspect.signature(VirtualFTSensor.__init__)
    scientific = (
        "axis_misalignment_quat",
        "scale_error",
        "additive_bias",
        "drift_rate",
        "drift_noise_std",
        "white_noise_std",
        "delay_steps",
        "filter_alpha",
        "force_saturation",
        "torque_saturation",
        "dropout_probability",
        "F_scale",
        "M_scale",
        "seed",
    )
    assert all(
        signature.parameters[name].default is inspect.Parameter.empty
        for name in scientific
    )
    with pytest.raises(ValueError):
        _sensor(delay_steps=torch.tensor([[0, 4]]))
    with pytest.raises(ValueError):
        _sensor(F_scale=0.0)
    sensor = _sensor()
    value = torch.zeros(1, 2, 6, dtype=torch.float64)
    value[0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        _run(sensor, value)
    with pytest.raises(ValueError):
        _run(sensor, torch.zeros(1, 1, 6, dtype=torch.float64))


def test_virtual_ft_selected_parameter_application_is_atomic_and_state_free() -> None:
    sensor = _sensor(3)
    sensor.drift_state.fill_(3.0)
    sensor.delay_buffer.fill_(4.0)
    sensor.filter_state.fill_(5.0)
    dynamic_before = (
        sensor.drift_state.clone(),
        sensor.delay_buffer.clone(),
        sensor.filter_state.clone(),
    )
    unselected_before = (
        sensor.axis_misalignment_quat[1].clone(),
        sensor.scale_error[1].clone(),
        sensor.additive_bias[1].clone(),
        sensor.drift_rate[1].clone(),
        sensor.drift_noise_std[1].clone(),
        sensor.white_noise_std[1].clone(),
        sensor.delay_steps[1].clone(),
        sensor.filter_alpha[1].clone(),
        sensor.saturation_limit[1].clone(),
        sensor.dropout_probability[1].clone(),
        sensor.normalization_scale[1].clone(),
    )
    parameters = _selected_sensor_parameters(2)
    sensor.apply_parameters([2, 0], parameters)

    assert torch.equal(
        sensor.axis_misalignment_quat[[2, 0], :, 0],
        torch.ones(2, 2, dtype=torch.float64),
    )
    assert torch.equal(sensor.scale_error[[2, 0]], parameters.scale_error)
    assert torch.equal(sensor.delay_steps[[2, 0]], parameters.delay_steps)
    assert torch.equal(
        sensor.saturation_limit[[2, 0]],
        torch.cat((parameters.force_saturation, parameters.torque_saturation), dim=-1),
    )
    expected_scale = torch.cat(
        (
            parameters.F_scale[..., None].expand(-1, -1, 3),
            parameters.M_scale[..., None].expand(-1, -1, 3),
        ),
        dim=-1,
    )
    assert torch.equal(sensor.normalization_scale[[2, 0]], expected_scale)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (
                sensor.axis_misalignment_quat[1],
                sensor.scale_error[1],
                sensor.additive_bias[1],
                sensor.drift_rate[1],
                sensor.drift_noise_std[1],
                sensor.white_noise_std[1],
                sensor.delay_steps[1],
                sensor.filter_alpha[1],
                sensor.saturation_limit[1],
                sensor.dropout_probability[1],
                sensor.normalization_scale[1],
            ),
            unselected_before,
            strict=True,
        )
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (sensor.drift_state, sensor.delay_buffer, sensor.filter_state),
            dynamic_before,
            strict=True,
        )
    )

    parameter_tensors = (
        sensor.axis_misalignment_quat,
        sensor.scale_error,
        sensor.additive_bias,
        sensor.drift_rate,
        sensor.drift_noise_std,
        sensor.white_noise_std,
        sensor.delay_steps,
        sensor.filter_alpha,
        sensor.saturation_limit,
        sensor.dropout_probability,
        sensor.normalization_scale,
    )
    invalid_parameters = (
        replace(
            parameters,
            scale_error=torch.full((2, 2, 6), -1.0, dtype=torch.float64),
        ),
        replace(
            parameters,
            drift_noise_std=torch.full((2, 2, 6), -0.1, dtype=torch.float64),
        ),
        replace(parameters, delay_steps=torch.full((2, 2), 4, dtype=torch.long)),
        replace(
            parameters,
            filter_alpha=torch.full((2, 2), 1.1, dtype=torch.float64),
        ),
        replace(
            parameters,
            force_saturation=torch.zeros(2, 2, 3, dtype=torch.float64),
        ),
        replace(
            parameters,
            dropout_probability=torch.full((2, 2), -0.1, dtype=torch.float64),
        ),
        replace(parameters, F_scale=torch.zeros(2, 2, dtype=torch.float64)),
        replace(
            parameters,
            axis_misalignment_quat=torch.zeros(2, 2, 4, dtype=torch.float64),
        ),
        replace(
            parameters,
            additive_bias=torch.full((2, 2, 6), torch.nan, dtype=torch.float64),
        ),
        replace(
            parameters,
            white_noise_std=torch.zeros(2, 2, 5, dtype=torch.float64),
        ),
        replace(
            parameters,
            torque_saturation=parameters.torque_saturation.float(),
        ),
        replace(
            parameters,
            delay_steps=parameters.delay_steps.to(dtype=torch.float64),
        ),
    )
    for invalid in invalid_parameters:
        before = tuple(tensor.clone() for tensor in parameter_tensors)
        with pytest.raises((TypeError, ValueError)):
            sensor.apply_parameters([2, 0], invalid)
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(parameter_tensors, before, strict=True)
        )


def test_virtual_ft_flat_parameters_reject_fractional_delay() -> None:
    rows = _selected_sensor_parameters(1).flatten()
    rows[:, 68] = 1.5
    with pytest.raises(ValueError, match="delay.*integers"):
        VirtualFTSensorParameters.from_flat(rows)


def test_sensing_float64_shapes_and_named_outputs() -> None:
    sensor = _sensor(3, dtype=torch.float64)
    value = torch.randn(3, 2, 6, dtype=torch.float64)
    output = _run(sensor, value)
    assert output.clean_total_wrench.shape == (3, 2, 6)
    assert output.calibrated_wrench.shape == (3, 2, 6)
    assert output.observed_wrench_physical.shape == (3, 2, 6)
    assert output.normalized_wrench.shape == (3, 2, 6)
    assert output.sensor_quality.shape == (3, 2)
    assert output.saturation_mask.shape == (3, 2, 6)
    assert output.dropout_mask.shape == (3, 2)
    assert output.normalized_wrench.dtype == torch.float64
    assert torch.isfinite(output.normalized_wrench).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sensing_cuda_smoke() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    sensor = _sensor(2, device=device, dtype=torch.float32)
    parameters = _selected_sensor_parameters(1, device=device, dtype=torch.float32)
    sensor.apply_parameters([1], parameters)
    assert sensor.delay_steps[1].tolist() == [1, 3]
    with pytest.raises(ValueError, match="must be on"):
        sensor.apply_parameters(
            [1], _selected_sensor_parameters(1, dtype=torch.float32)
        )
    value = torch.randn(2, 2, 6, device=device)
    output = _run(sensor, value)
    assert output.normalized_wrench.device == device
    tare = WristTareCalibrator(2, num_samples=1, device=device)
    tare_output = tare.update(value, torch.ones(2, 2, device=device, dtype=torch.bool))
    assert tare_output.calibrated_wrench.device == device
    raw, quat, joint, sensor_pos, pelvis = _transform_inputs(2, dtype=torch.float32)
    transformed = WristWrenchTransform()(
        raw.to(device),
        quat.to(device),
        joint.to(device),
        sensor_pos.to(device),
        pelvis.to(device),
    )
    assert transformed.device == device


def test_sensing_package_has_no_forbidden_imports_via_ast() -> None:
    sensing_dir = Path(__file__).resolve().parents[1] / "somaforce_cross" / "sensing"
    forbidden = ("isaac", "hdmi", "active_adaptation")
    for path in sorted(sensing_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.append(node.module or "")
                imported_names.extend(alias.name for alias in node.names)
        assert not any(
            token in name.lower() for name in imported_names for token in forbidden
        ), f"forbidden import in {path}: {imported_names}"
