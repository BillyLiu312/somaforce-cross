from __future__ import annotations

import torch
from torch import nn

from somaforce_cross.learning.actor_critic import (
    ACTION_DIM,
    ACTOR_INPUT_DIM,
    CRITIC_DIM,
    POLICY_DIM,
    PrivilegedCritic,
    ResidualActor,
    ResidualActorCritic,
)


def _observations(batch: int = 4) -> dict[str, torch.Tensor]:
    return {
        "policy": torch.randn(batch, POLICY_DIM),
        "critic": torch.randn(batch, CRITIC_DIM),
    }


def _linear_shapes(module: nn.Module) -> list[tuple[int, int]]:
    return [
        (layer.in_features, layer.out_features)
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def test_direct_network_shapes_and_finite_distribution() -> None:
    torch.manual_seed(1)
    actor = ResidualActor()
    critic = PrivilegedCritic()
    model = ResidualActorCritic()
    observations = _observations()

    assert _linear_shapes(actor) == [(220, 256), (256, 256), (256, 23)]
    assert _linear_shapes(critic) == [(845, 512), (512, 256), (256, 256), (256, 1)]
    assert not any(isinstance(module, nn.Tanh) for module in actor.modules())
    assert actor(torch.randn(4, ACTOR_INPUT_DIM)).shape == (4, ACTION_DIM)
    assert critic(observations["critic"]).shape == (4, 1)

    forward = model.actor_forward(observations["policy"])
    assert forward.actor_input.shape == (4, ACTOR_INPUT_DIM)
    assert forward.raw_residual.shape == (4, ACTION_DIM)
    assert torch.equal(forward.actor_input[:, 64:], observations["policy"][:, 448:604])
    actions = model.act(observations)
    assert actions.shape == (4, ACTION_DIM)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(model.get_actions_log_prob(actions)).all()
    assert torch.isfinite(model.action_std).all()


def test_actor_privilege_boundary_and_detached_environment_z_cross_slice() -> None:
    torch.manual_seed(2)
    model = ResidualActorCritic()
    observations = _observations()
    action = model.act_inference(observations)
    sampled_actions = torch.randn_like(action)
    first_log_prob = model.get_actions_log_prob(sampled_actions)

    critic_changed = dict(observations)
    critic_changed["critic"] = observations["critic"] + 1000.0
    assert torch.equal(action, model.act_inference(critic_changed))
    assert torch.equal(first_log_prob, model.get_actions_log_prob(sampled_actions))

    diagnostic_changed = dict(observations)
    policy = observations["policy"].clone()
    policy[:, 604:668] = torch.randn_like(policy[:, 604:668]) * 100.0
    diagnostic_changed["policy"] = policy
    assert torch.equal(action, model.act_inference(diagnostic_changed))
    assert torch.equal(first_log_prob, model.get_actions_log_prob(sampled_actions))


def test_actor_output_has_gradient_path_through_all_semantic_modules() -> None:
    torch.manual_seed(3)
    model = ResidualActorCritic()
    policy = torch.randn(3, POLICY_DIM)
    first = model.actor_forward(policy).raw_residual
    first[:, 0].sum().backward()
    first_gradient = model.semantic_pipeline.cross_encoder.mlp[0].weight.grad.clone()
    for parameter in model.parameters():
        parameter.grad = None

    second = model.actor_forward(policy).raw_residual
    second[:, 1].sum().backward()
    second_gradient = model.semantic_pipeline.cross_encoder.mlp[0].weight.grad
    assert second_gradient is not None
    assert not torch.allclose(first_gradient, second_gradient)
    for module in (
        model.semantic_pipeline.wrist_encoder,
        model.semantic_pipeline.semantic_heads,
        model.semantic_pipeline.cross_encoder,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_actor_and_critic_reject_wrong_contract_widths() -> None:
    model = ResidualActorCritic()
    with torch.no_grad():
        try:
            model.actor_forward(torch.zeros(2, POLICY_DIM - 1))
        except ValueError as exc:
            assert "668" in str(exc)
        else:
            raise AssertionError("policy width validation did not run")
    try:
        model.evaluate(
            {
                "policy": torch.zeros(2, POLICY_DIM),
                "critic": torch.zeros(2, CRITIC_DIM - 1),
            }
        )
    except ValueError as exc:
        assert "845" in str(exc)
    else:
        raise AssertionError("critic width validation did not run")
