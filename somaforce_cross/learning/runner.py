"""Explicit Phase 5 runner factory with delayed RSL-RL imports."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from somaforce_cross.learning.actor_critic import (
    ACTION_DIM,
    ACTOR_INPUT_DIM,
    CRITIC_DIM,
    POLICY_DIM,
    ResidualActorCritic,
)
from somaforce_cross.learning.config import Phase5Config, load_phase5_config
from somaforce_cross.learning.semantic_ppo import SemanticPPO


RSL_RL_VERSION = "3.0.1"


def _load_rsl_storage_class() -> type[Any]:
    """Load only the pinned RSL storage API after the Isaac app is active."""
    try:
        installed_version = importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("rsl-rl-lib is required for a Phase 5 rollout") from exc
    if installed_version != RSL_RL_VERSION:
        raise RuntimeError(
            f"Phase 5 requires rsl-rl-lib=={RSL_RL_VERSION}, found {installed_version}"
        )
    module = importlib.import_module("rsl_rl.storage")
    storage_class = getattr(module, "RolloutStorage")
    parameters = tuple(inspect.signature(storage_class.__init__).parameters)
    expected = (
        "self",
        "training_type",
        "num_envs",
        "num_transitions_per_env",
        "obs",
        "actions_shape",
        "device",
    )
    if parameters != expected:
        raise RuntimeError(
            "rsl-rl RolloutStorage.__init__ signature differs from 3.0.1: "
            f"{inspect.signature(storage_class.__init__)}"
        )
    return storage_class


def _require_observations(
    observations: object, *, num_envs: int
) -> Mapping[str, torch.Tensor]:
    if not isinstance(observations, Mapping):
        raise TypeError("environment observations must be a named tensor mapping")
    expected = {"policy": POLICY_DIM, "critic": CRITIC_DIM, "semantic_target": 31}
    if set(observations) != set(expected):
        raise ValueError(
            "Phase 5 environment observations must be policy/critic/semantic_target"
        )
    for name, width in expected.items():
        value = observations[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != (num_envs, width)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"invalid Phase 5 observation {name}")
    return observations


class Phase5Runner:
    """Bounded single-task rollout/update loop without stock runner class lookup."""

    def __init__(self, env: Any, algorithm: SemanticPPO, config: Phase5Config) -> None:
        self.env = env
        self.algorithm = algorithm
        self.config = config
        if not isinstance(env.num_envs, int) or env.num_envs <= 0:
            raise ValueError("environment must expose a positive integer num_envs")

    @staticmethod
    def _finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite {name}")

    def run(self, *, iterations: int) -> dict[str, Any]:
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be positive")
        observations, _ = self.env.reset()
        observations = _require_observations(observations, num_envs=self.env.num_envs)
        iteration_records: list[dict[str, Any]] = []
        steps_per_env = self.config.payload["ppo"]["num_steps_per_env"]
        for iteration in range(iterations):
            self.algorithm.init_storage(
                num_envs=self.env.num_envs,
                num_transitions_per_env=steps_per_env,
                observations=observations,
            )
            reward_sum = 0.0
            for _ in range(steps_per_env):
                with torch.inference_mode():
                    actions = self.algorithm.act(observations)
                    values = self.algorithm.transition.values
                    log_prob = self.algorithm.transition.actions_log_prob
                if values is None or log_prob is None:
                    raise AssertionError("PPO act did not record value/log probability")
                self._finite("action", actions)
                self._finite("value", values)
                self._finite("log_prob", log_prob)
                observations, rewards, terminated, time_outs, extras = self.env.step(
                    actions
                )
                observations = _require_observations(
                    observations, num_envs=self.env.num_envs
                )
                if not isinstance(rewards, torch.Tensor) or rewards.shape != (
                    self.env.num_envs,
                ):
                    raise ValueError("environment rewards must have shape [num_envs]")
                self._finite("reward", rewards)
                if not isinstance(terminated, torch.Tensor) or not isinstance(
                    time_outs, torch.Tensor
                ):
                    raise TypeError(
                        "environment terminated and time_outs must be tensors"
                    )
                dones = terminated.to(dtype=torch.bool) | time_outs.to(dtype=torch.bool)
                step_extras = dict(extras)
                step_extras["time_outs"] = time_outs
                self.algorithm.process_env_step(
                    observations, rewards, dones, step_extras
                )
                reward_sum += float(rewards.mean().item())
            self.algorithm.compute_returns(observations)
            update = self.algorithm.update()
            if not all(
                isinstance(value, bool) or torch.isfinite(torch.tensor(value)).item()
                for value in update.values()
            ):
                raise FloatingPointError("non-finite Phase 5 PPO update record")
            if not update["semantic_pipeline_has_nonzero_gradient"]:
                raise AssertionError("semantic pipeline did not receive a gradient")
            iteration_records.append(
                {
                    "mean_reward": reward_sum / steps_per_env,
                    "update": update,
                }
            )
        last_update = iteration_records[-1]["update"]
        return {
            "iterations": iterations,
            "num_steps_per_env": steps_per_env,
            "actor_input_dim": ACTOR_INPUT_DIM,
            "critic_input_dim": CRITIC_DIM,
            "action_dim": ACTION_DIM,
            "semantic_target_stored": True,
            "iteration_records": iteration_records,
            "last_update": last_update,
        }


def make_phase5_runner(
    env: Any,
    *,
    config_path: str | Path,
) -> Phase5Runner:
    """Construct the custom model/PPO path without `OnPolicyRunner` eval lookup."""
    config = load_phase5_config(config_path)
    storage_class = _load_rsl_storage_class()
    policy = ResidualActorCritic(
        init_noise_std=float(config.payload["ppo"]["init_noise_std"])
    )
    device = getattr(env, "device", "cpu")
    algorithm = SemanticPPO(
        policy,
        storage_class=storage_class,
        num_learning_epochs=int(config.payload["ppo"]["num_learning_epochs"]),
        num_mini_batches=int(config.payload["ppo"]["num_mini_batches"]),
        clip_param=float(config.payload["ppo"]["clip_param"]),
        gamma=float(config.payload["ppo"]["gamma"]),
        lam=float(config.payload["ppo"]["lam"]),
        value_loss_coef=float(config.payload["ppo"]["value_loss_coef"]),
        entropy_coef=float(config.payload["ppo"]["entropy_coef"]),
        learning_rate=float(config.payload["ppo"]["learning_rate"]),
        max_grad_norm=float(config.payload["ppo"]["max_grad_norm"]),
        use_clipped_value_loss=bool(config.payload["ppo"]["use_clipped_value_loss"]),
        schedule=str(config.payload["ppo"]["schedule"]),
        desired_kl=config.payload["ppo"]["desired_kl"],
        device=device,
        normalize_advantage_per_mini_batch=bool(
            config.payload["ppo"]["normalize_advantage_per_mini_batch"]
        ),
    )
    return Phase5Runner(env, algorithm, config)
