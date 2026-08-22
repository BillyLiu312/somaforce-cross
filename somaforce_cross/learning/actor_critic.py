"""Deployable actor and privileged critic for the bounded Phase 5 PPO path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Normal

from somaforce_cross.envs.observations import ForceSemanticOutput, ForceSemanticPipeline


POLICY_DIM = 668
CRITIC_DIM = 845
ACTOR_INPUT_DIM = 220
ACTION_DIM = 23
WRIST_TOKEN_END = 448
ACTOR_SUFFIX_START = 448
ACTOR_SUFFIX_END = 604
POLICY_Z_CROSS_START = 604
POLICY_Z_CROSS_END = 668
Z_CROSS_CLIP = 10.0
OBS_GROUPS = {"policy": ["policy"], "critic": ["critic"]}


def _require_tensor(
    value: object, *, name: str, width: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [B, {width}]")
    if value.shape[0] <= 0:
        raise ValueError(f"{name} requires a nonempty batch")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


class ResidualActor(nn.Module):
    """Direct Phase 5 MLP that emits unconstrained raw residual actions."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ACTOR_INPUT_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, ACTION_DIM),
        )

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        actor_input = _require_tensor(
            actor_input, name="actor_input", width=ACTOR_INPUT_DIM
        )
        return self.mlp(actor_input)


class PrivilegedCritic(nn.Module):
    """Direct privileged-state value MLP; its input is already normalized."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(CRITIC_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(self, critic_observation: torch.Tensor) -> torch.Tensor:
        critic_observation = _require_tensor(
            critic_observation, name="critic_observation", width=CRITIC_DIM
        )
        return self.mlp(critic_observation)


class ActorForward(NamedTuple):
    actor_input: torch.Tensor
    raw_residual: torch.Tensor
    semantic: ForceSemanticOutput


class ResidualActorCritic(nn.Module):
    """RSL-compatible policy with one checkpoint-owned semantic pipeline.

    The environment's final policy slice is a zero placeholder retained only to
    preserve the frozen observation widths.  Actor and critic both reconstruct
    `z_cross` from the normalized wrist-token prefix.  The critic receives a
    detached copy so value loss cannot supervise the semantic pipeline.
    """

    is_recurrent = False
    actor_obs_normalization = False
    critic_obs_normalization = False
    noise_std_type = "log"

    def __init__(self, *, init_noise_std: float = 1.0) -> None:
        super().__init__()
        if not isinstance(init_noise_std, (int, float)) or init_noise_std <= 0.0:
            raise ValueError("init_noise_std must be positive")
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.semantic_pipeline = ForceSemanticPipeline()
        self.actor = ResidualActor()
        self.critic = PrivilegedCritic()
        self.log_std = nn.Parameter(
            torch.log(torch.full((ACTION_DIM,), float(init_noise_std)))
        )
        self.distribution: Normal | None = None
        self._last_actor_forward: ActorForward | None = None
        self._last_actor_policy_observation: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _field(observations: object, name: str, width: int) -> torch.Tensor:
        if not isinstance(observations, Mapping):
            raise TypeError("observations must map named groups to tensors")
        try:
            value = observations[name]
        except KeyError as exc:
            raise KeyError(f"observations is missing {name!r}") from exc
        return _require_tensor(value, name=f"observations.{name}", width=width)

    def actor_forward(self, policy_observation: torch.Tensor) -> ActorForward:
        """Rebuild `z_cross` and create exactly the approved 220-D actor input."""
        policy_observation = _require_tensor(
            policy_observation, name="policy_observation", width=POLICY_DIM
        )
        wrist_tokens = policy_observation[:, :WRIST_TOKEN_END].reshape(-1, 2, 16, 14)
        semantic = self.semantic_pipeline(wrist_tokens)
        actor_input = torch.cat(
            (
                semantic.z_cross.clamp(-Z_CROSS_CLIP, Z_CROSS_CLIP),
                policy_observation[:, ACTOR_SUFFIX_START:ACTOR_SUFFIX_END],
            ),
            dim=-1,
        )
        if actor_input.shape[1] != ACTOR_INPUT_DIM:
            raise AssertionError("Phase 5 actor input width changed")
        raw_residual = self.actor(actor_input)
        return ActorForward(
            actor_input=actor_input,
            raw_residual=raw_residual,
            semantic=semantic,
        )

    def get_actor_obs(self, observations: object) -> torch.Tensor:
        policy_observation = self._field(observations, "policy", POLICY_DIM)
        forward = self.actor_forward(policy_observation)
        self._last_actor_forward = forward
        self._last_actor_policy_observation = policy_observation
        return forward.actor_input

    def get_critic_obs(self, observations: object) -> torch.Tensor:
        policy_observation = self._field(observations, "policy", POLICY_DIM)
        critic_observation = self._field(observations, "critic", CRITIC_DIM)
        if policy_observation.shape[0] != critic_observation.shape[0]:
            raise ValueError("policy and critic observation batches must match")

        if (
            self._last_actor_forward is not None
            and self._last_actor_policy_observation is policy_observation
        ):
            z_cross = self._last_actor_forward.semantic.z_cross
            self._last_actor_policy_observation = None
        else:
            wrist_tokens = policy_observation[:, :WRIST_TOKEN_END].reshape(
                -1, 2, 16, 14
            )
            with torch.no_grad():
                z_cross = self.semantic_pipeline(wrist_tokens).z_cross

        critic_with_policy_semantics = critic_observation.clone()
        critic_with_policy_semantics[:, POLICY_Z_CROSS_START:POLICY_Z_CROSS_END] = (
            z_cross.detach().clamp(-Z_CROSS_CLIP, Z_CROSS_CLIP)
        )
        return critic_with_policy_semantics

    def update_distribution(self, observations: object) -> None:
        self.get_actor_obs(observations)
        assert self._last_actor_forward is not None
        mean = self._last_actor_forward.raw_residual
        std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been created")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been created")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been created")
        return self.distribution.entropy().sum(dim=-1)

    @property
    def last_semantic_output(self) -> ForceSemanticOutput:
        if self._last_actor_forward is None:
            raise RuntimeError("actor forward has not run")
        return self._last_actor_forward.semantic

    def act(self, observations: object, **_: object) -> torch.Tensor:
        self.update_distribution(observations)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, observations: object) -> torch.Tensor:
        self.update_distribution(observations)
        return self.action_mean

    def evaluate(self, observations: object, **_: object) -> torch.Tensor:
        return self.critic(self.get_critic_obs(observations))

    def get_actions_log_prob(self, actions: object) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("action distribution has not been created")
        actions = _require_tensor(actions, name="actions", width=ACTION_DIM)
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, _: object) -> None:
        """Phase 4 FixedFieldNormalizer owns all approved normalization."""

    def reset(self, _: object | None = None) -> None:
        """Feed-forward policy has no recurrent state to reset."""
