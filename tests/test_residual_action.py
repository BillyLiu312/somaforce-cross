from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from somaforce_cross.residual import (
    ContactGainRamp,
    ContactGatedResidual,
    ExecutedActionHistory,
    JointMarginLimiter,
    PerJointAuthority,
    ResidualActionComposer,
    VelocityMarginLimiter,
)


ARMS = [11, 12, 15, 16, 19, 20, 21, 22]
WAIST = [2, 5, 8]
LEGS = [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]


def _limiters(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    velocity_limit: float = 10.0,
) -> tuple[JointMarginLimiter, VelocityMarginLimiter]:
    default = torch.zeros(23, dtype=dtype, device=device)
    joint = JointMarginLimiter(
        default_joint_pos=default,
        joint_lower=torch.full((23,), -1.0, dtype=dtype, device=device),
        joint_upper=torch.full((23,), 1.0, dtype=dtype, device=device),
        action_scale=0.5,
        margin=0.1,
    )
    velocity = VelocityMarginLimiter(
        default_joint_pos=default,
        action_scale=0.5,
        velocity_limit=velocity_limit,
        dt=0.1,
    )
    return joint, velocity


@pytest.mark.parametrize(
    ("stage", "arms", "waist", "legs"),
    [
        (0, 0.00, 0.00, 0.00),
        (1, 0.10, 0.05, 0.04),
        (2, 0.18, 0.10, 0.08),
        (3, 0.25, 0.15, 0.12),
    ],
)
def test_per_joint_authority_exact_groups(
    stage: int, arms: float, waist: float, legs: float
) -> None:
    result = PerJointAuthority()(stage)
    assert result.shape == (23,)
    assert torch.equal(result[ARMS], torch.full((len(ARMS),), arms))
    assert torch.equal(result[WAIST], torch.full((len(WAIST),), waist))
    assert torch.equal(result[LEGS], torch.full((len(LEGS),), legs))
    assert sorted(ARMS + WAIST + LEGS) == list(range(23))


def test_per_joint_authority_per_environment_stages() -> None:
    authority = PerJointAuthority(dtype=torch.float64)
    stages = torch.tensor([0, 3, 1, 2])
    result = authority(stages)
    assert result.shape == (4, 23)
    assert result.dtype == torch.float64
    assert torch.equal(result[0], authority(0))
    assert torch.equal(result[1], authority(3))
    with pytest.raises(ValueError):
        authority(torch.tensor([0, 4]))
    with pytest.raises(TypeError):
        authority(torch.tensor([0.0, 1.0]))


def test_contact_gain_ramp_attack_release_and_partial_reset() -> None:
    ramp = ContactGainRamp(3, attack_step=0.2, release_step=0.3)
    target = torch.tensor([[1.0], [0.1], [0.7]])
    assert torch.equal(ramp(target), torch.tensor([[0.2], [0.1], [0.2]]))
    assert torch.equal(ramp(target), torch.tensor([[0.4], [0.1], [0.4]]))
    released = ramp(torch.zeros(3, 1))
    assert torch.allclose(released, torch.tensor([[0.1], [0.0], [0.1]]), atol=1e-7)
    ramp(torch.ones(3, 1))
    before = ramp.state.clone()
    ramp.reset(torch.tensor([0, 2]))
    assert ramp.state[0].item() == 0.0
    assert torch.equal(ramp.state[1], before[1])
    assert ramp.state[2].item() == 0.0


def test_contact_gain_ramp_requires_explicit_valid_steps_and_target() -> None:
    with pytest.raises(TypeError):
        ContactGainRamp(2)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        ContactGainRamp(2, attack_step=0.0, release_step=0.1)
    ramp = ContactGainRamp(2, attack_step=0.1, release_step=0.1)
    with pytest.raises(ValueError):
        ramp(torch.tensor([[0.0], [1.1]]))


def test_contact_gated_residual_tanh_bound_and_gains() -> None:
    raw = torch.tensor([[100.0] * 23, [-100.0] * 23])
    authority = torch.linspace(0.01, 0.23, 23)
    contact = torch.tensor([[0.5], [1.0]])
    safety = torch.stack((torch.ones(23), torch.zeros(23)))
    output = ContactGatedResidual()(raw, authority, contact, safety)
    assert torch.equal(output.delta_bounded, authority * torch.tanh(raw))
    assert torch.all(output.delta_bounded.abs() <= authority)
    assert torch.equal(output.delta_gated[0], 0.5 * output.delta_bounded[0])
    assert torch.count_nonzero(output.delta_gated[1]) == 0


def test_contact_gated_residual_contact_zero() -> None:
    raw = torch.randn(2, 23)
    output = ContactGatedResidual()(
        raw, torch.full((23,), 0.25), torch.zeros(2, 1), torch.ones(2, 1)
    )
    assert torch.count_nonzero(output.delta_gated) == 0


def test_joint_margin_interval_and_clamping() -> None:
    joint, _ = _limiters()
    a_nom = torch.zeros(1, 23)
    interval = joint.residual_interval(a_nom)
    assert torch.allclose(interval.lower, torch.full((1, 23), -1.8))
    assert torch.allclose(interval.upper, torch.full((1, 23), 1.8))
    residual = torch.full((1, 23), 3.0)
    assert torch.allclose(joint(a_nom, residual), interval.upper)


def test_joint_margin_outside_nominal_allows_zero_or_correction_only() -> None:
    joint, _ = _limiters()
    a_nom = torch.full((1, 23), 3.0)
    interval = joint.residual_interval(a_nom)
    assert torch.all(interval.upper == 0)
    assert torch.all(interval.lower < 0)
    zero = torch.zeros_like(a_nom)
    assert torch.equal(joint(a_nom, zero), zero)
    assert torch.equal(joint(a_nom, torch.ones_like(a_nom)), zero)
    assert torch.all(joint(a_nom, -torch.ones_like(a_nom)) < 0)


def test_velocity_interval_normal_and_correction_direction() -> None:
    _, velocity = _limiters(velocity_limit=1.0)
    zero = torch.zeros(1, 23)
    interval = velocity.residual_interval(zero, zero)
    assert torch.allclose(interval.lower, torch.full((1, 23), -0.2))
    assert torch.allclose(interval.upper, torch.full((1, 23), 0.2))
    assert torch.allclose(velocity(zero, torch.ones_like(zero), zero), interval.upper)

    a_nom = torch.ones(1, 23)
    outside = velocity.residual_interval(a_nom, zero)
    assert torch.allclose(outside.lower, torch.full((1, 23), -1.2))
    assert torch.all(outside.upper == 0)
    assert torch.equal(velocity(a_nom, zero, zero), zero)
    assert torch.equal(velocity(a_nom, torch.ones_like(a_nom), zero), zero)
    assert torch.all(velocity(a_nom, -torch.ones_like(a_nom), zero) < 0)


@pytest.mark.parametrize("zero_source", ["raw", "authority", "contact"])
def test_composer_zero_residual_is_bitwise_nominal(zero_source: str) -> None:
    joint, velocity = _limiters()
    composer = ResidualActionComposer(joint, velocity, action_low=-1.0, action_high=1.0)
    a_nom = torch.randn(4, 23) * 2.0
    raw = torch.randn(4, 23)
    authority = PerJointAuthority()(3)
    contact = torch.ones(4, 1)
    if zero_source == "raw":
        raw.zero_()
    elif zero_source == "authority":
        authority = torch.zeros_like(authority)
    else:
        contact.zero_()
    current = torch.zeros_like(a_nom)
    output = composer(a_nom, raw, authority, contact, torch.ones(4, 1), current)
    assert torch.count_nonzero(output.delta_safe) == 0
    assert torch.equal(output.a_total, a_nom)


def test_composer_order_bounds_and_does_not_mutate_inputs() -> None:
    joint, velocity = _limiters(velocity_limit=1.0)
    composer = ResidualActionComposer(joint, velocity, action_low=-0.15, action_high=0.15)
    a_nom = torch.zeros(2, 23)
    raw = torch.full((2, 23), 100.0)
    a_nom_before = a_nom.clone()
    raw_before = raw.clone()
    output = composer(
        a_nom,
        raw,
        PerJointAuthority()(3),
        torch.ones(2, 1),
        torch.ones(2, 23),
        torch.zeros_like(a_nom),
    )
    assert torch.equal(output.delta_bounded[:, ARMS], torch.full((2, len(ARMS)), 0.25))
    assert torch.all(output.delta_safe <= 0.15)
    assert torch.all(output.delta_safe >= -0.15)
    assert torch.equal(output.a_total, a_nom + output.delta_safe)
    assert torch.equal(a_nom, a_nom_before)
    assert torch.equal(raw, raw_before)


def test_composer_nonzero_residual_never_exceeds_group_authority() -> None:
    joint, velocity = _limiters()
    output = ResidualActionComposer(joint, velocity)(
        torch.zeros(3, 23),
        torch.randn(3, 23) * 10,
        PerJointAuthority()(3),
        torch.ones(3, 1),
        torch.ones(3, 1),
        torch.zeros(3, 23),
    )
    expected = PerJointAuthority()(3).expand_as(output.delta_safe)
    assert torch.all(output.delta_safe.abs() <= expected)
    assert torch.isfinite(output.a_total).all()


def test_action_bound_preserves_outside_nominal_and_allows_correction() -> None:
    joint, velocity = _limiters()
    composer = ResidualActionComposer(joint, velocity, action_low=-1.0, action_high=1.0)
    a_nom = torch.full((1, 23), 1.2)
    positive = composer(
        a_nom,
        torch.full_like(a_nom, 10.0),
        PerJointAuthority()(3),
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.full_like(a_nom, 0.6),
    )
    assert torch.equal(positive.a_total, a_nom)
    negative = composer(
        a_nom,
        torch.full_like(a_nom, -10.0),
        PerJointAuthority()(3),
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.full_like(a_nom, 0.6),
    )
    assert torch.all(negative.delta_safe <= 0)
    assert torch.all(negative.a_total <= a_nom)


def test_executed_action_history_records_total_newest_first_and_resets() -> None:
    history = ExecutedActionHistory(2)
    a_nom = torch.zeros(2, 23)
    a_total_1 = torch.full((2, 23), 0.2)
    a_total_2 = torch.full((2, 23), -0.1)
    history.push(a_total_1)
    history.push(a_total_2)
    assert torch.equal(history.storage[:, :, 0], a_total_2)
    assert torch.equal(history.storage[:, :, 1], a_total_1)
    assert not torch.equal(history.storage[:, :, 0], a_nom)
    history.reset([1])
    assert torch.count_nonzero(history.storage[1]) == 0
    assert torch.equal(history.storage[0, :, 0], a_total_2[0])


def test_invalid_shapes_finite_values_limits_and_dtype_are_rejected() -> None:
    default = torch.zeros(23)
    with pytest.raises(ValueError):
        JointMarginLimiter(
            default_joint_pos=default,
            joint_lower=torch.ones(23),
            joint_upper=-torch.ones(23),
            action_scale=0.5,
            margin=0.0,
        )
    with pytest.raises(ValueError):
        JointMarginLimiter(
            default_joint_pos=default,
            joint_lower=-torch.ones(23),
            joint_upper=torch.ones(23),
            action_scale=0.0,
            margin=0.0,
        )
    with pytest.raises(ValueError):
        VelocityMarginLimiter(
            default_joint_pos=default,
            action_scale=0.5,
            velocity_limit=1.0,
            dt=0.0,
        )
    joint, _ = _limiters()
    incompatible_velocity = VelocityMarginLimiter(
        default_joint_pos=default,
        action_scale=0.25,
        velocity_limit=1.0,
        dt=0.1,
    )
    with pytest.raises(ValueError):
        ResidualActionComposer(joint, incompatible_velocity)
    joint, velocity = _limiters()
    composer = ResidualActionComposer(joint, velocity)
    with pytest.raises(ValueError):
        composer(
            torch.zeros(2, 22),
            torch.zeros(2, 23),
            torch.zeros(23),
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.zeros(2, 23),
        )
    bad = torch.zeros(2, 23)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError):
        composer(
            torch.zeros(2, 23),
            bad,
            torch.zeros(23),
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.zeros(2, 23),
        )
    with pytest.raises(ValueError):
        composer(
            torch.zeros(2, 23, dtype=torch.float64),
            torch.zeros(2, 23, dtype=torch.float64),
            torch.zeros(23, dtype=torch.float64),
            torch.ones(2, 1, dtype=torch.float64),
            torch.ones(2, 1, dtype=torch.float64),
            torch.zeros(2, 23, dtype=torch.float64),
        )


def test_residual_components_support_float64() -> None:
    dtype = torch.float64
    joint, velocity = _limiters(dtype=dtype)
    composer = ResidualActionComposer(joint, velocity, action_low=-1.0, action_high=1.0)
    a_nom = torch.zeros(2, 23, dtype=dtype)
    output = composer(
        a_nom,
        torch.ones_like(a_nom),
        PerJointAuthority(dtype=dtype)(3),
        torch.ones(2, 1, dtype=dtype),
        torch.ones(2, 1, dtype=dtype),
        torch.zeros_like(a_nom),
    )
    assert output.a_total.dtype == dtype
    ramp = ContactGainRamp(2, attack_step=0.1, release_step=0.2, dtype=dtype)
    assert ramp(torch.ones(2, 1, dtype=dtype)).dtype == dtype
    history = ExecutedActionHistory(2, dtype=dtype)
    assert history.push(output.a_total).dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_residual_components_cuda_smoke() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    joint, velocity = _limiters(device=device)
    composer = ResidualActionComposer(joint, velocity).to(device)
    a_nom = torch.zeros(2, 23, device=device)
    output = composer(
        a_nom,
        torch.ones_like(a_nom),
        PerJointAuthority(device=device)(torch.tensor([2, 3], device=device)),
        torch.ones(2, 1, device=device),
        torch.ones(2, 1, device=device),
        torch.zeros_like(a_nom),
    )
    assert output.a_total.device == device
    ramp = ContactGainRamp(
        2, attack_step=0.1, release_step=0.2, device=device
    )
    assert ramp(torch.ones(2, 1, device=device)).device == device
    history = ExecutedActionHistory(2, device=device)
    assert history.push(output.a_total).device == device


def test_residual_package_has_no_forbidden_imports_via_ast() -> None:
    residual_dir = Path(__file__).resolve().parents[1] / "somaforce_cross" / "residual"
    forbidden = ("isaac", "hdmi", "active_adaptation")
    for path in sorted(residual_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.append(node.module or "")
                imported_names.extend(alias.name for alias in node.names)
        assert not any(
            token in name.lower()
            for name in imported_names
            for token in forbidden
        ), f"forbidden import in {path}: {imported_names}"
