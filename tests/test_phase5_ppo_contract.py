from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from somaforce_cross.learning.actor_critic import (
    ACTION_DIM,
    CRITIC_DIM,
    POLICY_DIM,
    ResidualActorCritic,
)
from somaforce_cross.learning.losses import semantic_losses
from somaforce_cross.learning.runner import record_policy_semantics
from somaforce_cross.learning.semantic_ppo import SemanticPPO


class _Storage:
    """Small RSL-RL storage-shaped test double with semantic target retention."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        observations: Mapping[str, torch.Tensor],
        actions_shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        assert training_type == "rl"
        assert actions_shape == (ACTION_DIM,)
        self.observations = {
            name: value.clone() for name, value in observations.items()
        }
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
        self.records: list[dict[str, Any]] = []
        self.cleared = False

    def add_transitions(self, transition: Any) -> None:
        assert hasattr(transition, "hidden_states")
        assert transition.hidden_states is None
        assert transition.observations is not None
        self.records.append(
            {
                "observations": {
                    name: value.clone()
                    for name, value in transition.observations.items()
                },
                "actions": transition.actions.clone(),
                "values": transition.values.clone(),
                "log_prob": transition.actions_log_prob.clone(),
                "mu": transition.action_mean.clone(),
                "sigma": transition.action_sigma.clone(),
                "rewards": transition.rewards.clone(),
            }
        )

    def compute_returns(self, last_values: torch.Tensor, *_: Any, **__: Any) -> None:
        assert torch.isfinite(last_values).all()
        assert last_values.requires_grad is False
        assert torch.is_inference(last_values) is False
        return None

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        assert len(self.records) == 1
        record = self.records[0]
        for _ in range(num_epochs):
            for _ in range(num_mini_batches):
                rewards = record["rewards"].unsqueeze(-1)
                yield (
                    record["observations"],
                    record["actions"],
                    record["values"],
                    torch.ones_like(rewards),
                    record["values"] + rewards,
                    record["log_prob"].unsqueeze(-1),
                    record["mu"],
                    record["sigma"],
                    (None, None),
                    None,
                )

    def clear(self) -> None:
        self.cleared = True


def _observations(batch: int = 8) -> dict[str, torch.Tensor]:
    p_dir = torch.softmax(torch.randn(batch, 13), dim=-1)
    p_mag = torch.softmax(torch.randn(batch, 5), dim=-1)
    return {
        "policy": torch.randn(batch, POLICY_DIM),
        "critic": torch.randn(batch, CRITIC_DIM),
        "semantic_target": torch.cat(
            (torch.randn(batch, 12), p_dir, p_mag, torch.ones(batch, 1)), dim=-1
        ),
    }


def test_obs_groups_semantic_storage_reforward_and_combined_gradient() -> None:
    torch.manual_seed(5)
    model = ResidualActorCritic()
    algorithm = SemanticPPO(model, storage_class=_Storage)
    observations = _observations()
    assert model.obs_groups == {"policy": ["policy"], "critic": ["critic"]}
    algorithm.init_storage(
        num_envs=8,
        num_transitions_per_env=32,
        observations=observations,
    )
    assert algorithm.storage is not None
    assert "semantic_target" in algorithm.storage.observations

    forward_calls = 0

    def count_forward(*_: Any) -> None:
        nonlocal forward_calls
        forward_calls += 1

    hook = model.semantic_pipeline.register_forward_hook(count_forward)
    actions = algorithm.act(observations)
    rollout_calls = forward_calls
    algorithm.process_env_step(
        observations,
        torch.ones(8),
        torch.zeros(8, dtype=torch.bool),
        {"time_outs": torch.zeros(8, dtype=torch.bool)},
    )
    with torch.inference_mode():
        terminal_observations = {
            name: value.clone() for name, value in observations.items()
        }
    for value in terminal_observations.values():
        assert torch.is_inference(value)
    algorithm.compute_returns(terminal_observations)
    update = algorithm.update()
    hook.remove()

    assert rollout_calls == 1
    assert forward_calls > rollout_calls
    assert algorithm.storage.records[0]["observations"]["semantic_target"].shape == (
        8,
        31,
    )
    assert actions.shape == (8, ACTION_DIM)
    for name in (
        "ppo_loss",
        "semantic_dir_loss",
        "semantic_mag_loss",
        "semantic_total",
        "semantic_entropy",
        "semantic_pipeline_grad_norm",
    ):
        assert name in update
        assert torch.isfinite(torch.tensor(update[name]))
    assert update["semantic_pipeline_has_nonzero_gradient"] is True
    assert update["semantic_pipeline_grad_norm"] > 0.0
    assert update["contact_bearing_minibatches"] == 24.0
    assert len(algorithm.last_gradient_diagnostics) == 24
    assert all(
        torch.isfinite(torch.tensor(item["ppo_grad_norm"]))
        and torch.isfinite(torch.tensor(item["aux_grad_norm"]))
        and item["ppo_grad_norm"] >= 0.0
        and item["aux_grad_norm"] >= 0.0
        for item in algorithm.last_gradient_diagnostics
    )
    assert any(
        item["ppo_grad_norm"] > 0.0 for item in algorithm.last_gradient_diagnostics
    )
    assert any(
        item["aux_grad_norm"] > 0.0 for item in algorithm.last_gradient_diagnostics
    )
    assert algorithm.storage.cleared is True


def test_semantic_target_changes_loss_but_never_actor_output() -> None:
    torch.manual_seed(6)
    model = ResidualActorCritic()
    observations = _observations(batch=3)
    actor_output = model.act_inference(observations)
    first_semantic = model.last_semantic_output
    first_loss = semantic_losses(
        first_semantic.p_dir, first_semantic.p_mag, observations["semantic_target"]
    )
    changed = dict(observations)
    changed_target = observations["semantic_target"].clone()
    changed_target[:, 12:25] = torch.roll(changed_target[:, 12:25], shifts=1, dims=-1)
    changed["semantic_target"] = changed_target
    assert torch.equal(actor_output, model.act_inference(changed))
    second_semantic = model.last_semantic_output
    second_loss = semantic_losses(
        second_semantic.p_dir, second_semantic.p_mag, changed_target
    )
    assert not torch.equal(first_loss.dir_loss, second_loss.dir_loss)


def test_runner_records_the_same_policy_forward_used_for_rollout() -> None:
    class _Environment:
        def __init__(self) -> None:
            self.record: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        def record_policy_semantics(
            self,
            p_dir: torch.Tensor,
            p_mag: torch.Tensor,
            semantic_target: torch.Tensor,
        ) -> None:
            self.record = (p_dir, p_mag, semantic_target)

    model = ResidualActorCritic()
    observations = _observations(batch=3)
    model.act_inference(observations)
    expected = model.last_semantic_output
    environment = _Environment()

    record_policy_semantics(environment, model, observations)

    assert environment.record is not None
    p_dir, p_mag, target = environment.record
    assert torch.equal(p_dir, expected.p_dir)
    assert torch.equal(p_mag, expected.p_mag)
    assert torch.equal(target, observations["semantic_target"])
    assert not p_dir.requires_grad and not p_mag.requires_grad
