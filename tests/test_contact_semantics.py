from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

import somaforce_cross.force as force_api
import somaforce_cross.sensing as sensing_api
from somaforce_cross.force.semantic_heads import magnitude_soft_targets
from somaforce_cross.force.semantic_targets import (
    TwoWristSemanticTargets,
    direction_soft_targets,
    two_wrist_semantic_targets,
)
from somaforce_cross.sensing.contact_detector import (
    ContactDetector,
    ContactDetectorOutput,
)


def _detector(
    batch_size: int = 2,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    **overrides: object,
) -> ContactDetector:
    parameters: dict[str, object] = {
        "F_scale": 10.0,
        "M_scale": 2.0,
        "on_threshold": 0.8,
        "off_threshold": 0.4,
        "probability_temperature": 0.2,
        "smoothing_alpha": 0.5,
    }
    parameters.update(overrides)
    return ContactDetector(
        batch_size,
        device=device,
        dtype=dtype,
        **parameters,  # type: ignore[arg-type]
    )


def _wrench_for_strength(
    strength: float, shape: tuple[int, int, int] = (2, 2, 6)
) -> torch.Tensor:
    wrench = torch.zeros(shape, dtype=torch.float64)
    wrench[..., 0] = strength * 10.0
    wrench[..., 3] = strength * 2.0
    return wrench


def test_detector_requires_explicit_valid_parameters() -> None:
    with pytest.raises(TypeError):
        ContactDetector(1)  # type: ignore[call-arg]
    for overrides in (
        {"F_scale": 0.0},
        {"M_scale": float("inf")},
        {"off_threshold": 0.8},
        {"on_threshold": 0.0},
        {"probability_temperature": 0.0},
        {"smoothing_alpha": 1.1},
    ):
        with pytest.raises((TypeError, ValueError)):
            _detector(**overrides)
    with pytest.raises(ValueError):
        _detector(batch_size=0)
    with pytest.raises(TypeError):
        _detector(dtype=torch.int64)


def test_detector_validates_shape_dtype_device_and_finite() -> None:
    detector = _detector()
    valid = torch.ones(2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        detector(torch.zeros(2, 2, 5, dtype=torch.float64), valid)
    with pytest.raises(ValueError, match="dtype"):
        detector(torch.zeros(2, 2, 6), valid)
    with pytest.raises(TypeError, match="boolean"):
        detector(torch.zeros(2, 2, 6, dtype=torch.float64), valid.float())
    wrench = torch.zeros(2, 2, 6, dtype=torch.float64)
    wrench[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        detector(wrench, valid)


def test_detector_on_off_hysteresis_inclusive_boundaries() -> None:
    detector = _detector(smoothing_alpha=0.0, on_threshold=0.5, off_threshold=0.25)
    valid = torch.ones(2, 2, dtype=torch.bool)
    output = detector(_wrench_for_strength(0.5), valid)
    assert output.contact_state.all()
    output = detector(_wrench_for_strength(0.25), valid)
    assert output.contact_state.all()
    output = detector(_wrench_for_strength(0.25 - 1e-6), valid)
    assert not output.contact_state.any()


def test_detector_probability_smoothing_exact_steps() -> None:
    detector = _detector(batch_size=1, smoothing_alpha=0.25)
    valid = torch.ones(1, 2, dtype=torch.bool)
    wrench = _wrench_for_strength(1.0, (1, 2, 6))
    first = detector(wrench, valid)
    raw_first = torch.sigmoid(torch.tensor((1.0 - 0.8) / 0.2, dtype=torch.float64))
    assert torch.allclose(first.contact_probability, 0.75 * raw_first.expand(1, 2))
    second = detector(wrench, valid)
    raw_second = torch.sigmoid(torch.tensor((1.0 - 0.4) / 0.2, dtype=torch.float64))
    expected = 0.25 * first.contact_probability + 0.75 * raw_second
    assert torch.allclose(second.contact_probability, expected)


def test_detector_partial_reset_isolates_selected_environments() -> None:
    detector = _detector(batch_size=3, smoothing_alpha=0.0)
    valid = torch.ones(3, 2, dtype=torch.bool)
    detector(_wrench_for_strength(1.0, (3, 2, 6)), valid)
    state_before = detector.contact_state.clone()
    probability_before = detector.contact_probability.clone()

    detector.reset([1])

    assert torch.equal(detector.contact_state[[0, 2]], state_before[[0, 2]])
    assert torch.equal(detector.contact_probability[[0, 2]], probability_before[[0, 2]])
    assert not detector.contact_state[1].any()
    assert torch.count_nonzero(detector.contact_probability[1]) == 0


def test_invalid_sample_holds_only_corresponding_wrist_and_environment() -> None:
    detector = _detector(smoothing_alpha=0.0)
    valid = torch.ones(2, 2, dtype=torch.bool)
    detector(_wrench_for_strength(1.0), valid)
    previous_state = detector.contact_state.clone()
    previous_probability = detector.contact_probability.clone()
    next_valid = torch.ones_like(valid)
    next_valid[0, 1] = False
    wrench = _wrench_for_strength(0.0)
    wrench[0, 1, 0] = torch.nan

    output = detector(wrench, next_valid)

    assert output.contact_state[0, 1] == previous_state[0, 1]
    assert output.contact_probability[0, 1] == previous_probability[0, 1]
    assert not output.contact_state[next_valid].any()
    assert torch.isfinite(output.r_force).all()
    assert torch.isfinite(output.r_torque).all()
    assert torch.isfinite(output.strength).all()


def test_invalid_nonfinite_positions_are_isolated() -> None:
    detector = _detector(smoothing_alpha=0.0)
    wrench = _wrench_for_strength(1.0)
    wrench[0, 0, 0] = torch.nan
    wrench[1, 1, 1] = torch.inf
    valid = torch.ones(2, 2, dtype=torch.bool)
    valid[0, 0] = False
    valid[1, 1] = False
    output = detector(wrench, valid)
    assert torch.isfinite(output.r_force).all()
    assert torch.isfinite(output.r_torque).all()
    assert torch.isfinite(output.strength).all()
    assert torch.equal(output.contact_state, valid)


def test_direction_force_and_torque_signed_category_order() -> None:
    wrench = torch.zeros(12, 6, dtype=torch.float64)
    for axis in range(6):
        scale = 10.0 if axis < 3 else 2.0
        wrench[2 * axis, axis] = scale * 2.0**0.5
        wrench[2 * axis + 1, axis] = -scale * 2.0**0.5
    target = direction_soft_targets(wrench, 10.0, 2.0)
    assert torch.equal(target.argmax(dim=-1), torch.arange(12))
    assert torch.isfinite(target).all()
    assert torch.allclose(target.sum(dim=-1), torch.ones(12, dtype=torch.float64))


def test_zero_low_strength_and_mixed_direction_favor_neutral() -> None:
    zero = direction_soft_targets(torch.zeros(1, 6), 1.0, 1.0)
    assert zero.argmax(dim=-1).item() == 12
    low = torch.tensor([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert direction_soft_targets(low, 1.0, 1.0).argmax(dim=-1).item() == 12
    axis = torch.tensor([[2.0**0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    mixed = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    axis_target = direction_soft_targets(axis, 1.0, 1.0)
    mixed_target = direction_soft_targets(mixed, 1.0, 1.0)
    assert mixed_target[0, 12] > axis_target[0, 12]
    assert mixed_target[0, 12] > mixed_target[0, :12].amax()


def test_direction_arbitrary_leading_dimensions_and_validation() -> None:
    target = direction_soft_targets(torch.randn(2, 3, 4, 6), 2.0, 3.0)
    assert target.shape == (2, 3, 4, 13)
    with pytest.raises(ValueError, match="shape"):
        direction_soft_targets(torch.zeros(2, 5), 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        direction_soft_targets(torch.full((1, 6), torch.nan), 1.0, 1.0)
    with pytest.raises(ValueError):
        direction_soft_targets(torch.zeros(1, 6), -1.0, 1.0)


def test_magnitude_target_reuses_existing_helper_exactly() -> None:
    wrench = torch.randn(3, 2, 6, dtype=torch.float64)
    output = two_wrist_semantic_targets(
        wrench, torch.ones(3, 2, dtype=torch.float64), 10.0, 2.0
    )
    expected = magnitude_soft_targets(wrench, 10.0, 2.0, sigma_mag=0.15)
    assert torch.equal(output.per_wrist_magnitude, expected)


def test_single_double_and_unequal_wrist_weighting() -> None:
    wrench = torch.zeros(3, 2, 6, dtype=torch.float64)
    wrench[:, 0, 0] = 10.0 * 2.0**0.5
    wrench[:, 1, 1] = 10.0 * 2.0**0.5
    probability = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [0.25, 0.75]], dtype=torch.float64
    )
    output = two_wrist_semantic_targets(wrench, probability, 10.0, 2.0)
    per = output.per_wrist_direction
    assert torch.equal(output.global_direction[0], per[0, 0])
    assert torch.allclose(output.global_direction[1], per[1].mean(dim=0))
    assert torch.allclose(
        output.global_direction[2], 0.25 * per[2, 0] + 0.75 * per[2, 1]
    )


@pytest.mark.parametrize(
    ("dtype", "tiny"),
    ((torch.float32, 1e-20), (torch.float64, 1e-200)),
)
def test_tiny_nonzero_contact_weights_are_normalized_by_true_sum(
    dtype: torch.dtype, tiny: float
) -> None:
    wrench = torch.zeros(1, 2, 6, dtype=dtype)
    wrench[0, 0, 0] = 10.0
    wrench[0, 1, 1] = 10.0
    probability = torch.tensor([[tiny, 0.0]], dtype=dtype)
    output = two_wrist_semantic_targets(wrench, probability, 10.0, 2.0)
    assert torch.allclose(
        output.global_direction, output.per_wrist_direction[:, 0], atol=1e-6
    )
    assert torch.allclose(
        output.global_magnitude, output.per_wrist_magnitude[:, 0], atol=1e-6
    )
    assert torch.allclose(
        output.global_direction.sum(dim=-1), torch.ones(1, dtype=dtype)
    )
    assert torch.allclose(
        output.global_magnitude.sum(dim=-1), torch.ones(1, dtype=dtype)
    )
    assert output.g_contact.item() > 0.0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_zero_contact_fallback_and_g_contact_formula(dtype: torch.dtype) -> None:
    wrench = torch.randn(2, 2, 6, dtype=dtype)
    probability = torch.tensor([[0.0, 0.0], [0.2, 0.7]], dtype=dtype)
    output = two_wrist_semantic_targets(wrench, probability, 2.0, 3.0)
    zero = torch.zeros(2, 6, dtype=dtype)
    assert torch.equal(
        output.global_direction[0], direction_soft_targets(zero, 2.0, 3.0)[0]
    )
    assert torch.equal(
        output.global_magnitude[0], magnitude_soft_targets(zero, 2.0, 3.0)[0]
    )
    expected = 1.0 - torch.prod(1.0 - probability, dim=-1, keepdim=True)
    assert torch.allclose(output.g_contact, expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cpu_outputs_are_finite_normalized_and_preserve_dtype_device(
    dtype: torch.dtype,
) -> None:
    detector = _detector(batch_size=4, dtype=dtype)
    wrench = torch.randn(4, 2, 6, dtype=dtype)
    detected = detector(wrench, torch.ones(4, 2, dtype=torch.bool))
    targets = two_wrist_semantic_targets(
        wrench, detected.contact_probability, 10.0, 2.0
    )

    for value in (detected.r_force, detected.r_torque, detected.strength):
        assert value.dtype == dtype
        assert value.device.type == "cpu"
        assert torch.isfinite(value).all()
    assert detected.contact_state.dtype == torch.bool
    assert detected.contact_probability.dtype == dtype
    assert detected.contact_probability.device.type == "cpu"
    for target in (
        targets.per_wrist_direction,
        targets.per_wrist_magnitude,
        targets.global_direction,
        targets.global_magnitude,
    ):
        assert target.dtype == dtype
        assert target.device.type == "cpu"
        assert torch.isfinite(target).all()
        assert torch.allclose(
            target.sum(dim=-1),
            torch.ones_like(target[..., 0]),
            atol=1e-6 if dtype == torch.float32 else 1e-12,
        )
    assert targets.g_contact.dtype == dtype
    assert targets.g_contact.device.type == "cpu"
    assert torch.isfinite(targets.g_contact).all()


def test_semantic_distributions_are_finite_and_normalized_float64() -> None:
    wrench = torch.randn(5, 2, 6, dtype=torch.float64)
    probability = torch.rand(5, 2, dtype=torch.float64)
    output = two_wrist_semantic_targets(wrench, probability, 4.0, 2.0)
    for target in (
        output.per_wrist_direction,
        output.per_wrist_magnitude,
        output.global_direction,
        output.global_magnitude,
    ):
        assert target.dtype == torch.float64
        assert torch.isfinite(target).all()
        assert torch.allclose(
            target.sum(dim=-1), torch.ones_like(target[..., 0]), atol=1e-12
        )


def test_quality_is_not_an_input_or_applied_twice() -> None:
    detector_parameters = inspect.signature(ContactDetector).parameters
    target_parameters = inspect.signature(two_wrist_semantic_targets).parameters
    assert "sensor_quality" not in detector_parameters
    assert "sensor_quality" not in target_parameters
    wrench = torch.randn(1, 2, 6, dtype=torch.float64)
    probability = torch.tensor([[0.3, 0.8]], dtype=torch.float64)
    output = two_wrist_semantic_targets(wrench, probability, 2.0, 3.0)
    expected = (probability[..., None] * output.per_wrist_direction).sum(
        dim=1
    ) / probability.sum(dim=1, keepdim=True)
    assert torch.allclose(output.global_direction, expected)


def test_package_level_public_api_exports() -> None:
    expected_force = {
        "TwoWristSemanticTargets",
        "direction_soft_targets",
        "two_wrist_semantic_targets",
    }
    expected_sensing = {"ContactDetector", "ContactDetectorOutput"}
    assert expected_force <= set(force_api.__all__)
    assert expected_sensing <= set(sensing_api.__all__)
    assert force_api.TwoWristSemanticTargets is TwoWristSemanticTargets
    assert force_api.direction_soft_targets is direction_soft_targets
    assert force_api.two_wrist_semantic_targets is two_wrist_semantic_targets
    assert sensing_api.ContactDetector is ContactDetector
    assert sensing_api.ContactDetectorOutput is ContactDetectorOutput


def test_new_modules_have_no_forbidden_import_or_independent_cross_target() -> None:
    root = Path(__file__).parents[1]
    for relative in (
        "somaforce_cross/sensing/contact_detector.py",
        "somaforce_cross/force/semantic_targets.py",
    ):
        source = (root / relative).read_text()
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        lowered = " ".join(imports).lower()
        for forbidden in ("isaac", "hdmi", "active_adaptation", "rsl_rl"):
            assert forbidden not in lowered
        identifiers = {
            node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert "p_cross" not in identifiers
        assert "cross_loss" not in identifiers
        assert "lambda_cross" not in identifiers
    assert all(
        "cross" not in field.lower() for field in TwoWristSemanticTargets._fields
    )
    assert (
        "sensor_quality" not in inspect.signature(two_wrist_semantic_targets).parameters
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_contact_semantics_cuda_smoke(dtype: torch.dtype) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    detector = _detector(device=device, dtype=dtype)
    wrench = torch.randn(2, 2, 6, device=device, dtype=dtype)
    detected = detector(wrench, torch.ones(2, 2, device=device, dtype=torch.bool))
    targets = two_wrist_semantic_targets(
        wrench, detected.contact_probability, 10.0, 2.0
    )
    assert detected.strength.device == device
    assert detected.strength.dtype == dtype
    assert targets.global_direction.device == device
    assert targets.global_direction.dtype == dtype
    assert torch.isfinite(targets.global_direction).all()
    assert torch.allclose(
        targets.global_direction.sum(dim=-1),
        torch.ones(2, device=device, dtype=dtype),
        atol=1e-6 if dtype == torch.float32 else 1e-12,
    )
