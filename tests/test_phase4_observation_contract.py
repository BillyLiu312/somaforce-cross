from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from somaforce_cross.envs import (
    CriticObservationBundle,
    ForceSemanticPipeline,
    PolicyObservationBundle,
    SemanticTargetBundle,
    build_semantic_target_bundle,
)
from somaforce_cross.force import two_wrist_semantic_targets


def _policy_fields(
    batch_size: int = 2,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        "wrist_tokens": torch.randn(batch_size, 2, 16, 14, device=device),
        "proprio": torch.randn(batch_size, 64, device=device),
        "a_nom_history": torch.randn(batch_size, 23, 3, device=device),
        "previous_a_total": torch.randn(batch_size, 23, device=device),
    }


def _critic_fields(
    policy: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    batch_size = policy.shape[0]
    return {
        "policy": policy,
        "object_state": torch.randn(batch_size, 16, device=device),
        "physics_mismatch": torch.randn(batch_size, 39, device=device),
        "scaffold_mismatch": torch.randn(batch_size, 2, device=device),
        "sensor_mismatch": torch.randn(batch_size, 90, device=device),
        "progress_state": torch.randn(batch_size, 4, device=device),
        "contact_state": torch.randn(batch_size, 17, device=device),
        "stability_state": torch.randn(batch_size, 9, device=device),
    }


def test_canonical_assembly_binds_semantic_output_and_actor_gradients() -> None:
    torch.manual_seed(0)
    pipeline = ForceSemanticPipeline()
    wrist_tokens = torch.randn(3, 2, 16, 14, requires_grad=True)
    before = wrist_tokens.detach().clone()
    inputs = _policy_fields(batch_size=3)
    inputs["wrist_tokens"] = wrist_tokens

    assembly = pipeline.assemble_policy_observation(**inputs)
    bundle = assembly.bundle
    output = assembly.semantic_output

    assert bundle.z_cross is output.z_cross
    assert torch.equal(bundle.z_cross, output.z_cross)
    assert output.per_wrist_features.shape == (3, 2, 128)
    assert output.fused_features.shape == (3, 272)
    assert output.dir_logits.shape == (3, 13)
    assert output.p_dir.shape == (3, 13)
    assert output.mag_logits.shape == (3, 5)
    assert output.p_mag.shape == (3, 5)
    assert output.P_cross.shape == (3, 13, 5)
    assert output.z_joint.shape == (3, 65)
    assert output.z_cross.shape == (3, 64)
    assert torch.allclose(output.p_dir.sum(-1), torch.ones(3), atol=1e-6)
    assert torch.allclose(output.p_mag.sum(-1), torch.ones(3), atol=1e-6)
    assert torch.allclose(output.P_cross.sum((-1, -2)), torch.ones(3), atol=1e-6)
    assert torch.equal(output.z_joint, output.P_cross.reshape(3, 65))
    assert torch.equal(wrist_tokens.detach(), before)

    actor_input = bundle.residual_actor_input()
    assert torch.equal(actor_input[:, :64], output.z_cross)
    actor_input[:, :64].square().sum().backward()
    assert wrist_tokens.grad is not None
    assert torch.count_nonzero(wrist_tokens.grad) > 0
    for module in (
        pipeline.wrist_encoder,
        pipeline.semantic_heads,
        pipeline.cross_encoder,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_wrist_token_change_changes_canonically_derived_z_cross() -> None:
    torch.manual_seed(1)
    pipeline = ForceSemanticPipeline()
    inputs = _policy_fields()
    first = pipeline.assemble_policy_observation(**inputs)
    changed_inputs = dict(inputs)
    changed_tokens = inputs["wrist_tokens"].clone()
    changed_tokens[:, :, -1, :] += 3.0
    changed_inputs["wrist_tokens"] = changed_tokens
    second = pipeline.assemble_policy_observation(**changed_inputs)

    assert not torch.equal(
        first.semantic_output.z_cross, second.semantic_output.z_cross
    )
    assert not torch.equal(first.bundle.z_cross, second.bundle.z_cross)
    assert second.bundle.z_cross is second.semantic_output.z_cross


def test_direct_policy_construction_with_independent_z_cross_is_closed() -> None:
    with pytest.raises(TypeError, match="assemble_policy_observation"):
        PolicyObservationBundle(
            **_policy_fields(),
            z_cross=torch.randn(2, 64),
        )


def test_environment_policy_factory_uses_only_a_zero_semantic_placeholder() -> None:
    inputs = _policy_fields(batch_size=3)
    bundle = PolicyObservationBundle.with_zero_semantic_placeholder(**inputs)

    assert bundle.flatten().shape == (3, 668)
    assert torch.equal(bundle.z_cross, torch.zeros(3, 64))
    assert torch.equal(
        bundle.flatten()[:, :604],
        torch.cat(
            (
                inputs["wrist_tokens"].reshape(3, 448),
                inputs["proprio"],
                inputs["a_nom_history"].reshape(3, 69),
                inputs["previous_a_total"],
            ),
            dim=-1,
        ),
    )


def test_policy_exact_nonsemantic_field_order_with_sentinels() -> None:
    batch_size = 2
    torch.manual_seed(2)
    wrist_tokens = torch.randn(batch_size, 2, 16, 14)
    proprio = torch.arange(64, dtype=torch.float32).expand(batch_size, -1) + 1_000
    a_nom_history = (
        torch.arange(69, dtype=torch.float32)
        .reshape(1, 23, 3)
        .expand(batch_size, -1, -1)
        + 2_000
    )
    previous_a_total = (
        torch.arange(23, dtype=torch.float32).expand(batch_size, -1) + 3_000
    )
    assembly = ForceSemanticPipeline().assemble_policy_observation(
        wrist_tokens=wrist_tokens,
        proprio=proprio,
        a_nom_history=a_nom_history,
        previous_a_total=previous_a_total,
    )
    bundle = assembly.bundle
    z_cross = assembly.semantic_output.z_cross

    semantic_prefix = bundle.semantic_prefix()
    actor_input = bundle.residual_actor_input()
    policy = bundle.flatten()
    expected_actor = torch.cat(
        (z_cross, proprio, a_nom_history.reshape(batch_size, 69), previous_a_total),
        dim=-1,
    )
    expected_policy = torch.cat(
        (
            wrist_tokens.reshape(batch_size, 448),
            proprio,
            a_nom_history.reshape(batch_size, 69),
            previous_a_total,
            z_cross,
        ),
        dim=-1,
    )

    assert semantic_prefix.shape == (batch_size, 448)
    assert actor_input.shape == (batch_size, 220)
    assert policy.shape == (batch_size, 668)
    assert torch.equal(semantic_prefix, wrist_tokens.reshape(batch_size, 448))
    assert torch.equal(actor_input, expected_actor)
    assert torch.equal(policy, expected_policy)


def test_critic_exact_shape_and_policy_first_order() -> None:
    widths = (668, 16, 39, 2, 90, 4, 17, 9)
    values = tuple(
        torch.full((2, width), float(index), dtype=torch.float32)
        for index, width in enumerate(widths, start=1)
    )
    bundle = CriticObservationBundle(*values)
    critic = bundle.flatten()

    assert critic.shape == (2, 845)
    offset = 0
    for index, width in enumerate(widths, start=1):
        assert torch.equal(
            critic[:, offset : offset + width],
            torch.full((2, width), float(index)),
        )
        offset += width


def test_semantic_target_reuses_phase3c_outputs_and_exact_order() -> None:
    clean_wrench = torch.randn(4, 2, 6)
    contact_probability = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.2, 0.7], [1.0, 1.0]])
    F_scale = torch.tensor(10.0)
    M_scale = torch.tensor(2.0)
    expected = two_wrist_semantic_targets(
        clean_wrench, contact_probability, F_scale, M_scale
    )

    bundle = build_semantic_target_bundle(
        clean_wrench, contact_probability, F_scale, M_scale
    )
    target = bundle.flatten()

    assert isinstance(bundle, SemanticTargetBundle)
    assert target.shape == (4, 31)
    assert torch.equal(bundle.p_dir_target, expected.global_direction)
    assert torch.equal(bundle.p_mag_target, expected.global_magnitude)
    assert torch.equal(bundle.semantic_loss_weight, expected.g_contact)
    assert torch.equal(target[:, :12], clean_wrench.reshape(4, 12))
    assert torch.equal(target[:, 12:25], expected.global_direction)
    assert torch.equal(target[:, 25:30], expected.global_magnitude)
    assert torch.equal(target[:, 30:], expected.g_contact)


def test_privileged_and_clean_fields_cannot_change_policy_or_actor_input() -> None:
    policy_bundle = (
        ForceSemanticPipeline().assemble_policy_observation(**_policy_fields()).bundle
    )
    policy = policy_bundle.flatten()
    actor_input = policy_bundle.residual_actor_input()

    first_critic = CriticObservationBundle(**_critic_fields(policy)).flatten()
    changed_fields = _critic_fields(policy)
    for name in changed_fields.keys() - {"policy"}:
        changed_fields[name].add_(100.0)
    second_critic = CriticObservationBundle(**changed_fields).flatten()
    assert not torch.equal(first_critic, second_critic)
    assert torch.equal(policy_bundle.flatten(), policy)
    assert torch.equal(policy_bundle.residual_actor_input(), actor_input)

    first_target = build_semantic_target_bundle(
        torch.zeros(2, 2, 6),
        torch.zeros(2, 2),
        torch.tensor(1.0),
        torch.tensor(1.0),
    ).flatten()
    second_target = build_semantic_target_bundle(
        torch.ones(2, 2, 6),
        torch.ones(2, 2),
        torch.tensor(1.0),
        torch.tensor(1.0),
    ).flatten()
    assert not torch.equal(first_target, second_target)
    assert torch.equal(policy_bundle.flatten(), policy)
    assert torch.equal(policy_bundle.residual_actor_input(), actor_input)


def test_assembly_does_not_modify_caller_tensors() -> None:
    policy_fields = _policy_fields()
    policy_before = {name: value.clone() for name, value in policy_fields.items()}
    policy_bundle = (
        ForceSemanticPipeline().assemble_policy_observation(**policy_fields).bundle
    )
    policy = policy_bundle.flatten()
    policy_bundle.residual_actor_input()
    assert all(
        torch.equal(policy_fields[name], before)
        for name, before in policy_before.items()
    )

    critic_fields = _critic_fields(policy)
    critic_before = {name: value.clone() for name, value in critic_fields.items()}
    CriticObservationBundle(**critic_fields).flatten()
    assert all(
        torch.equal(critic_fields[name], before)
        for name, before in critic_before.items()
    )

    clean_wrench = torch.randn(2, 2, 6)
    contact_probability = torch.rand(2, 2)
    F_scale = torch.tensor(3.0)
    M_scale = torch.tensor(2.0)
    target_inputs = tuple(
        value.clone() for value in (clean_wrench, contact_probability, F_scale, M_scale)
    )
    build_semantic_target_bundle(clean_wrench, contact_probability, F_scale, M_scale)
    for value, before in zip(
        (clean_wrench, contact_probability, F_scale, M_scale),
        target_inputs,
        strict=True,
    ):
        assert torch.equal(value, before)


@pytest.mark.parametrize(
    ("field", "replacement", "error", "match"),
    (
        ("proprio", object(), TypeError, "torch.Tensor"),
        ("proprio", torch.zeros(2, 64, dtype=torch.float64), TypeError, "float32"),
        ("proprio", torch.zeros(2, 63), ValueError, "shape"),
        ("proprio", torch.zeros(3, 64), ValueError, "shape"),
        ("proprio", torch.full((2, 64), torch.inf), ValueError, "finite"),
    ),
)
def test_policy_rejects_invalid_inputs_early(
    field: str,
    replacement: object,
    error: type[Exception],
    match: str,
) -> None:
    inputs: dict[str, object] = _policy_fields()
    inputs[field] = replacement
    with pytest.raises(error, match=match):
        ForceSemanticPipeline().assemble_policy_observation(  # type: ignore[arg-type]
            **inputs
        )


def test_policy_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ForceSemanticPipeline().assemble_policy_observation(
            **_policy_fields(batch_size=0)
        )


def test_critic_and_target_reject_invalid_inputs() -> None:
    policy = (
        ForceSemanticPipeline()
        .assemble_policy_observation(**_policy_fields())
        .bundle.flatten()
    )
    critic_fields = _critic_fields(policy)
    critic_fields["sensor_mismatch"] = torch.zeros(2, 89)
    with pytest.raises(ValueError, match="sensor_mismatch.*shape"):
        CriticObservationBundle(**critic_fields)

    with pytest.raises(TypeError, match="contact_probability.*float32"):
        build_semantic_target_bundle(
            torch.zeros(2, 2, 6),
            torch.zeros(2, 2, dtype=torch.float64),
            torch.tensor(1.0),
            torch.tensor(1.0),
        )
    with pytest.raises(ValueError, match="contact_probability.*finite"):
        build_semantic_target_bundle(
            torch.zeros(2, 2, 6),
            torch.full((2, 2), torch.nan),
            torch.tensor(1.0),
            torch.tensor(1.0),
        )
    with pytest.raises(TypeError, match="F_scale.*torch.Tensor"):
        build_semantic_target_bundle(
            torch.zeros(2, 2, 6),
            torch.zeros(2, 2),
            1.0,  # type: ignore[arg-type]
            torch.tensor(1.0),
        )
    with pytest.raises(ValueError, match="M_scale.*positive"):
        build_semantic_target_bundle(
            torch.zeros(2, 2, 6),
            torch.zeros(2, 2),
            torch.tensor(1.0),
            torch.tensor(0.0),
        )


def test_public_bundle_field_names_are_exact() -> None:
    assert [field.name for field in fields(PolicyObservationBundle)] == [
        "wrist_tokens",
        "proprio",
        "a_nom_history",
        "previous_a_total",
        "z_cross",
    ]
    assert [field.name for field in fields(CriticObservationBundle)] == [
        "policy",
        "object_state",
        "physics_mismatch",
        "scaffold_mismatch",
        "sensor_mismatch",
        "progress_state",
        "contact_state",
        "stability_state",
    ]


def test_pipeline_rejects_a_partially_mixed_parameter_dtype() -> None:
    pipeline = ForceSemanticPipeline()
    parameter = pipeline.cross_encoder.mlp[2].weight
    parameter.data = parameter.data.to(torch.float64)

    with pytest.raises(
        ValueError,
        match=r"cross_encoder\.mlp\.2\.weight.*dtype torch\.float64.*torch\.float32",
    ):
        pipeline(torch.randn(2, 2, 16, 14))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_phase4_observation_cuda_float32_and_device_rejection() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    pipeline = ForceSemanticPipeline().to(device)
    assembly = pipeline.assemble_policy_observation(**_policy_fields(device=device))
    semantic = assembly.semantic_output
    policy_bundle = assembly.bundle
    policy = policy_bundle.flatten()
    critic = CriticObservationBundle(**_critic_fields(policy, device=device)).flatten()
    target = build_semantic_target_bundle(
        torch.randn(2, 2, 6, device=device),
        torch.rand(2, 2, device=device),
        torch.tensor(10.0, device=device),
        torch.tensor(2.0, device=device),
    ).flatten()
    assert semantic.z_cross.device == device
    assert policy.shape == (2, 668) and policy.device == device
    assert critic.shape == (2, 845) and critic.device == device
    assert target.shape == (2, 31) and target.device == device

    mixed = _policy_fields(device=device)
    mixed["proprio"] = torch.zeros(2, 64)
    with pytest.raises(ValueError, match="same device|must be on"):
        pipeline.assemble_policy_observation(**mixed)

    pipeline.cross_encoder.mlp[2].weight = torch.nn.Parameter(
        pipeline.cross_encoder.mlp[2].weight.detach().cpu()
    )
    with pytest.raises(
        ValueError,
        match=r"cross_encoder\.mlp\.2\.weight.*cpu.*cuda",
    ):
        pipeline(torch.randn(2, 2, 16, 14, device=device))
