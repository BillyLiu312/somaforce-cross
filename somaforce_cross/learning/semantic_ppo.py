"""RSL-RL 3.0.1-compatible PPO update with Phase 5 semantic supervision.

The RSL storage class is supplied by the delayed training factory.  Keeping
this module PyTorch-only preserves ordinary test and model imports without an
Isaac or RSL-RL runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn, optim

from somaforce_cross.learning.actor_critic import ACTION_DIM, ResidualActorCritic
from somaforce_cross.learning.losses import SemanticLossOutput, semantic_losses


@dataclass
class _Transition:
    observations: Mapping[str, torch.Tensor] | None = None
    hidden_states: object | None = None
    actions: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    values: torch.Tensor | None = None
    actions_log_prob: torch.Tensor | None = None
    action_mean: torch.Tensor | None = None
    action_sigma: torch.Tensor | None = None

    def clear(self) -> None:
        self.observations = None
        self.hidden_states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.values = None
        self.actions_log_prob = None
        self.action_mean = None
        self.action_sigma = None


class SemanticPPO:
    """Feed-forward PPO matching RSL-RL 3.0.1 storage and loss semantics.

    `storage_class` is deliberately injected by the runner after it checks the
    installed pinned RSL-RL API.  The class requires the RSL rollout interface
    (`add_transitions`, `compute_returns`, `mini_batch_generator`, and `clear`).
    """

    def __init__(
        self,
        policy: ResidualActorCritic,
        *,
        storage_class: type[Any],
        num_learning_epochs: int = 3,
        num_mini_batches: int = 8,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.001,
        learning_rate: float = 3.0e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = None,
        device: torch.device | str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
    ) -> None:
        if not isinstance(policy, ResidualActorCritic):
            raise TypeError("Phase 5 requires ResidualActorCritic")
        if not isinstance(storage_class, type):
            raise TypeError("storage_class must be an RSL-RL RolloutStorage class")
        if schedule != "fixed" or desired_kl is not None:
            raise ValueError("Phase 5 disables adaptive PPO KL scheduling")
        if not use_clipped_value_loss:
            raise ValueError("Phase 5 requires clipped value loss")
        if normalize_advantage_per_mini_batch:
            raise ValueError("Phase 5 disables per-mini-batch advantage normalization")
        if num_learning_epochs != 3 or num_mini_batches != 8:
            raise ValueError("Phase 5 PPO epoch and mini-batch counts are frozen")
        self.policy = policy.to(device)
        self.storage_class = storage_class
        self.device = torch.device(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.storage: Any | None = None
        self.transition = _Transition()
        self.last_gradient_diagnostics: tuple[dict[str, float], ...] = ()

    @staticmethod
    def _require_observations(observations: object) -> Mapping[str, torch.Tensor]:
        if not isinstance(observations, Mapping):
            raise TypeError("rollout observations must be a mapping")
        required = {"policy", "critic", "semantic_target"}
        missing = required - set(observations)
        if missing:
            raise KeyError(f"rollout observations missing {sorted(missing)}")
        validated: dict[str, torch.Tensor] = {}
        for name in required:
            value = observations[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"observations.{name} must be a torch.Tensor")
            validated[name] = value
        return validated

    def init_storage(
        self,
        *,
        num_envs: int,
        num_transitions_per_env: int,
        observations: object,
    ) -> None:
        observations = self._require_observations(observations)
        if num_transitions_per_env != 32:
            raise ValueError("Phase 5 num_steps_per_env must be 32")
        if not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.storage = self.storage_class(
            "rl",
            num_envs,
            num_transitions_per_env,
            observations,
            (ACTION_DIM,),
            self.device,
        )
        if "semantic_target" not in self.storage.observations.keys():
            raise AssertionError("RSL rollout storage lost semantic_target")

    def act(self, observations: object) -> torch.Tensor:
        observations = self._require_observations(observations)
        self.transition.actions = self.policy.act(observations).detach()
        self.transition.values = self.policy.evaluate(observations).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = observations
        return self.transition.actions

    def process_env_step(
        self,
        next_observations: object,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: Mapping[str, object],
    ) -> None:
        if self.storage is None:
            raise RuntimeError("storage has not been initialized")
        self._require_observations(next_observations)
        if self.transition.observations is None or self.transition.values is None:
            raise RuntimeError("act must precede process_env_step")
        if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1:
            raise ValueError("rewards must have shape [B]")
        if not isinstance(dones, torch.Tensor) or dones.ndim != 1:
            raise ValueError("dones must have shape [B]")
        if rewards.shape != dones.shape or not torch.isfinite(rewards).all():
            raise ValueError("rewards and dones must be finite matching vectors")
        self.policy.update_normalization(next_observations)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        time_outs = extras.get("time_outs")
        if time_outs is not None:
            if (
                not isinstance(time_outs, torch.Tensor)
                or time_outs.shape != rewards.shape
            ):
                raise ValueError("extras.time_outs must have shape [B]")
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs.unsqueeze(1).to(self.device), dim=1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, observations: object) -> None:
        if self.storage is None:
            raise RuntimeError("storage has not been initialized")
        observations = self._require_observations(observations)
        with torch.no_grad():
            last_values = self.policy.evaluate(observations)
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    @staticmethod
    def _semantic_gradient_norm(policy: ResidualActorCritic) -> tuple[float, bool]:
        squared_norm = 0.0
        has_nonzero_gradient = False
        for parameter in policy.semantic_pipeline.parameters():
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().norm().item())
            squared_norm += norm * norm
            has_nonzero_gradient |= norm > 0.0
        return squared_norm**0.5, has_nonzero_gradient

    @staticmethod
    def _objective_gradient_norm(
        objective: torch.Tensor,
        parameters: tuple[nn.Parameter, ...],
    ) -> float:
        """Inspect one objective without touching `.grad` or optimizer state."""
        gradients = torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        squared_norm = 0.0
        for gradient in gradients:
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    "non-finite semantic-pipeline diagnostic gradient"
                )
            norm = float(gradient.detach().norm().item())
            squared_norm += norm * norm
        result = squared_norm**0.5
        if not torch.isfinite(torch.tensor(result)):
            raise FloatingPointError("non-finite semantic-pipeline gradient norm")
        return result

    def update(self) -> dict[str, float | bool]:
        """Re-forward semantic features for every stored mini-batch with gradients."""
        if self.storage is None:
            raise RuntimeError("storage has not been initialized")
        sums = {
            "value_loss": 0.0,
            "surrogate_loss": 0.0,
            "entropy": 0.0,
            "ppo_loss": 0.0,
            "semantic_dir_loss": 0.0,
            "semantic_mag_loss": 0.0,
            "semantic_total": 0.0,
            "semantic_entropy": 0.0,
            "semantic_pipeline_grad_norm": 0.0,
        }
        gradient_diagnostics: list[dict[str, float]] = []
        any_semantic_gradient = False
        semantic_parameters = tuple(self.policy.semantic_pipeline.parameters())
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        updates = 0
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                raise AssertionError(
                    "Phase 5 must not normalize advantages per mini batch"
                )
            self.policy.act(
                obs_batch,
                masks=masks_batch,
                hidden_states=hidden_states_batch[0],
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                obs_batch,
                masks=masks_batch,
                hidden_states=hidden_states_batch[1],
            )
            entropy_batch = self.policy.entropy
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            semantic: SemanticLossOutput = semantic_losses(
                self.policy.last_semantic_output.p_dir,
                self.policy.last_semantic_output.p_mag,
                obs_batch["semantic_target"],
            )
            combined_loss = ppo_loss + semantic.total
            if not torch.isfinite(combined_loss):
                raise FloatingPointError("non-finite combined Phase 5 PPO objective")
            semantic_weight = obs_batch["semantic_target"][:, 30:31]
            if torch.any(semantic_weight > 0.0):
                # These autograd queries are diagnostic-only.  They run before the
                # one combined backward and never populate parameter `.grad`.
                ppo_gradient_norm = self._objective_gradient_norm(
                    ppo_loss, semantic_parameters
                )
                auxiliary_gradient_norm = self._objective_gradient_norm(
                    semantic.total, semantic_parameters
                )
                gradient_diagnostics.append(
                    {
                        "ppo_grad_norm": ppo_gradient_norm,
                        "aux_grad_norm": auxiliary_gradient_norm,
                    }
                )
            self.optimizer.zero_grad(set_to_none=True)
            combined_loss.backward()
            semantic_gradient_norm, nonzero_gradient = self._semantic_gradient_norm(
                self.policy
            )
            any_semantic_gradient |= nonzero_gradient
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            sums["value_loss"] += float(value_loss.detach().item())
            sums["surrogate_loss"] += float(surrogate_loss.detach().item())
            sums["entropy"] += float(entropy_batch.detach().mean().item())
            sums["ppo_loss"] += float(ppo_loss.detach().item())
            sums["semantic_dir_loss"] += float(semantic.dir_loss.detach().item())
            sums["semantic_mag_loss"] += float(semantic.mag_loss.detach().item())
            sums["semantic_total"] += float(semantic.total.detach().item())
            sums["semantic_entropy"] += float(semantic.entropy.detach().item())
            sums["semantic_pipeline_grad_norm"] += semantic_gradient_norm
            updates += 1
        if updates != self.num_learning_epochs * self.num_mini_batches:
            raise RuntimeError(
                "RSL rollout produced an unexpected number of PPO updates"
            )
        self.storage.clear()
        self.last_gradient_diagnostics = tuple(gradient_diagnostics)
        result: dict[str, float | bool] = {
            key: value / updates for key, value in sums.items()
        }
        result["semantic_pipeline_has_nonzero_gradient"] = any_semantic_gradient
        result["ppo_updates"] = float(updates)
        result["contact_bearing_minibatches"] = float(len(gradient_diagnostics))
        return result
