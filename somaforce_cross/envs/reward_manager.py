"""Contract-driven Phase 4B5 reward, done, and equal-episode logging."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from somaforce_cross.envs.numeric_contract import NumericContract, TASKS


def _float_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 1
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite float32 [B]")
    return value


def _bool_vector(
    value: torch.Tensor, name: str, batch: int, device: torch.device
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.shape != (batch,)
        or value.device != device
    ):
        raise ValueError(f"{name} must be bool [B] on the reward device")
    return value


def _float_matrix(
    value: torch.Tensor, name: str, batch: int, width: int, device: torch.device
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.shape != (batch, width)
        or value.device != device
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite float32 [B,{width}]")
    return value


@dataclass(frozen=True)
class RewardOutput:
    raw_terms: Mapping[str, torch.Tensor]
    weighted_terms: Mapping[str, torch.Tensor]
    total: torch.Tensor


class Phase4B5RewardManager:
    """Frozen shared formulas with terminal-only terminal rewards."""

    def __init__(
        self, contract: NumericContract, task: str, *, c0: bool = False
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"unsupported task: {task!r}")
        self.contract = contract
        self.task = task
        self.c0 = c0
        self.horizon = int(contract.payload["episodes"]["horizons"][task])
        self.constants = contract.payload["reward"]["constants"]
        self.weights = contract.payload["reward"]["weights"]
        self.aggregate_clip = contract.payload["reward"]["aggregate_clip"]

    def compute(
        self,
        *,
        progress_delta: torch.Tensor,
        expected_contact: torch.Tensor,
        contact_truth: torch.Tensor,
        wrench: torch.Tensor,
        stability_margin: torch.Tensor,
        delta_safe: torch.Tensor,
        authority: torch.Tensor,
        previous_delta_safe: torch.Tensor,
        success_latched: torch.Tensor,
        failure_latched: torch.Tensor,
        nonfinite: torch.Tensor,
        terminated: torch.Tensor,
        time_outs: torch.Tensor,
    ) -> RewardOutput:
        progress_delta = _float_vector(progress_delta, "progress_delta")
        batch, device = progress_delta.shape[0], progress_delta.device
        if self.c0:
            zero = torch.zeros(batch, dtype=torch.float32, device=device)
            return RewardOutput({"c0": zero}, {"c0": zero}, zero)
        expected = _float_matrix(expected_contact, "expected_contact", batch, 2, device)
        truth = _float_matrix(contact_truth, "contact_truth", batch, 2, device)
        if (
            wrench.shape != (batch, 2, 6)
            or wrench.dtype != torch.float32
            or wrench.device != device
            or not torch.isfinite(wrench).all()
        ):
            raise ValueError("wrench must be finite float32 [B,2,6]")
        stability = _float_vector(stability_margin, "stability_margin")
        delta = _float_matrix(delta_safe, "delta_safe", batch, 23, device)
        limit = _float_matrix(authority, "authority", batch, 23, device)
        previous = _float_matrix(
            previous_delta_safe, "previous_delta_safe", batch, 23, device
        )
        success = _bool_vector(success_latched, "success_latched", batch, device)
        failure = _bool_vector(failure_latched, "failure_latched", batch, device)
        invalid = _bool_vector(nonfinite, "nonfinite", batch, device)
        terminal = _bool_vector(terminated, "terminated", batch, device) | _bool_vector(
            time_outs, "time_outs", batch, device
        )
        c = self.constants
        q = torch.tensor(1.0 / self.horizon, dtype=torch.float32, device=device)
        force = torch.linalg.vector_norm(wrench[..., :3], dim=-1) / float(
            c["force_scale_N"]
        )
        moment = torch.linalg.vector_norm(wrench[..., 3:6], dim=-1) / float(
            c["moment_scale_Nm"]
        )
        s = torch.clamp(
            torch.sqrt(0.5 * (force.square() + moment.square())),
            0.0,
            float(c["force_s_clip"]),
        )
        terminal_nonfinite = terminal & invalid
        terminal_failure = terminal & failure & ~terminal_nonfinite
        terminal_success = terminal & success & ~terminal_failure & ~terminal_nonfinite
        raw = {
            "progress": torch.clamp(
                progress_delta,
                -float(c["progress_delta_clip"]),
                float(c["progress_delta_clip"]),
            ),
            "contact": q
            * (
                expected * (2.0 * truth - 1.0)
                - float(c["contact_unexpected_penalty"]) * (1.0 - expected) * truth
            ).mean(-1),
            "force": -q
            * (
                expected * torch.relu(s - float(c["force_expected_threshold"])).square()
                + (1.0 - expected) * s.square()
            ).mean(-1),
            "stability": -q
            * torch.clamp(
                torch.relu(
                    (float(c["stability_margin_m"]) - stability)
                    / float(c["stability_margin_m"])
                ).square(),
                0.0,
                float(c["stability_penalty_clip"]),
            ),
            "residual": -q
            * torch.where(
                limit > 0.0,
                (delta / torch.clamp_min(limit, float(c["residual_epsilon"]))).square(),
                torch.zeros_like(delta),
            ).mean(-1),
            "rate": -q
            * torch.clamp(
                (delta - previous) / float(c["rate_delta_scale"]),
                -float(c["rate_ratio_clip"]),
                float(c["rate_ratio_clip"]),
            )
            .square()
            .mean(-1),
            "terminal_success": terminal_success.to(torch.float32),
            "terminal_failure": terminal_failure.to(torch.float32),
            "nonfinite": terminal_nonfinite.to(torch.float32),
        }
        weighted = {
            name: value * float(self.weights[name]) for name, value in raw.items()
        }
        total = torch.clamp(
            torch.stack(tuple(weighted.values())).sum(0),
            float(self.aggregate_clip[0]),
            float(self.aggregate_clip[1]),
        )
        return RewardOutput(raw, weighted, total)


class EpisodeTermination:
    """Selected-row resettable latches with adapter-owned payload success."""

    def __init__(
        self,
        contract: NumericContract,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.contract = contract
        self.success = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.failure = torch.zeros_like(self.success)
        self.episode_invalid = torch.zeros_like(self.success)

    def reset(self, env_ids: torch.Tensor) -> None:
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.int64
            or env_ids.ndim != 1
            or env_ids.device != self.success.device
            or torch.any(env_ids < 0)
            or torch.any(env_ids >= self.success.shape[0])
            or torch.unique(env_ids).numel() != env_ids.numel()
        ):
            raise ValueError("env_ids must be unique in-range int64 indices")
        self.success[env_ids] = False
        self.failure[env_ids] = False
        self.episode_invalid[env_ids] = False

    def update(
        self,
        *,
        task: str,
        progress: torch.Tensor,
        adapter_task_success: torch.Tensor,
        failure: torch.Tensor,
        nonfinite: torch.Tensor,
        reference_exhausted: torch.Tensor,
        episode_steps: torch.Tensor,
        tare_ready: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if task not in TASKS:
            raise ValueError("unknown task")
        progress = _float_vector(progress, "progress")
        batch, device = progress.shape[0], progress.device
        task_success = _bool_vector(
            adapter_task_success, "adapter_task_success", batch, device
        )
        failure = _bool_vector(failure, "failure", batch, device)
        nonfinite = _bool_vector(nonfinite, "nonfinite", batch, device)
        exhausted = _bool_vector(
            reference_exhausted, "reference_exhausted", batch, device
        )
        ready = _bool_vector(tare_ready, "tare_ready", batch, device)
        if (
            episode_steps.dtype not in (torch.int32, torch.int64)
            or episode_steps.shape != (batch,)
            or episode_steps.device != device
        ):
            raise ValueError("episode_steps must be integer [B]")
        if task in ("push_door_hand", "push_box"):
            self.success |= (progress >= 1.0) | task_success
        else:
            self.success |= task_success
        self.failure |= failure
        tare = self.contract.payload["episodes"]["tare"]
        self.episode_invalid |= ~ready & (episode_steps >= int(tare["max_steps"]))
        terminated = nonfinite | self.failure
        timeout = (
            exhausted
            | (
                episode_steps
                >= int(self.contract.payload["episodes"]["horizons"][task])
            )
        ) & ~terminated
        return terminated, timeout


class EpisodeMetricLog:
    """Frozen episode schema with equal-episode pooled summaries."""

    def __init__(self, contract: NumericContract) -> None:
        self.contract = contract
        self.schema = contract.payload["logging"]
        self._episodes: list[dict[str, Any]] = []

    @staticmethod
    def _finite_scalar(value: object, name: str) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be a finite scalar")
            return float(value.item())
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{name} must be a finite scalar")
        number = float(value)
        if not torch.isfinite(torch.tensor(number)):
            raise ValueError(f"{name} must be a finite scalar")
        return number

    def add_episode(self, record: Mapping[str, object]) -> None:
        required = {
            "diagnostics",
            "evaluation_id",
            "family",
            "nominal",
            "outcome",
            "raw_reward_sums",
            "return",
            "seed",
            "stage",
            "steps",
            "task",
            "weighted_reward_sums",
        }
        if set(record) != required:
            raise ValueError("episode record has unknown or missing schema keys")
        if record["task"] not in TASKS or record["stage"] not in {"C1", "C2", "C3"}:
            raise ValueError("episode task/stage is invalid")
        if not isinstance(record["family"], str) or not record["family"]:
            raise ValueError("episode family must be non-empty")
        if record["evaluation_id"] is not None and (
            not isinstance(record["evaluation_id"], str) or not record["evaluation_id"]
        ):
            raise ValueError("evaluation_id must be null or non-empty")
        if not isinstance(record["nominal"], bool):
            raise ValueError("episode nominal must be bool")
        if any(
            isinstance(record[key], bool)
            or not isinstance(record[key], int)
            or record[key] < 0
            for key in ("seed", "steps")
        ):
            raise ValueError("episode seed/steps must be nonnegative integers")
        outcome = record["outcome"]
        if (
            not isinstance(outcome, Mapping)
            or set(outcome) != set(self.schema["outcome_keys"])
            or any(not isinstance(value, bool) for value in outcome.values())
        ):
            raise ValueError("episode outcome schema is invalid")
        normalized: dict[str, Any] = dict(record)
        for group, schema_key in (
            ("raw_reward_sums", "raw_reward_keys"),
            ("weighted_reward_sums", "weighted_reward_keys"),
            ("diagnostics", "diagnostic_keys"),
        ):
            values = record[group]
            if not isinstance(values, Mapping) or set(values) != set(
                self.schema[schema_key]
            ):
                raise ValueError(f"episode {group} schema is invalid")
            normalized[group] = {
                key: self._finite_scalar(value, f"{group}.{key}")
                for key, value in values.items()
            }
        normalized["return"] = self._finite_scalar(record["return"], "return")
        self._episodes.append(normalized)

    @staticmethod
    def _readonly_copy(value: object) -> object:
        if isinstance(value, Mapping):
            return MappingProxyType(
                {
                    copy.deepcopy(key): EpisodeMetricLog._readonly_copy(item)
                    for key, item in value.items()
                }
            )
        if isinstance(value, list | tuple):
            return tuple(EpisodeMetricLog._readonly_copy(item) for item in value)
        return copy.deepcopy(value)

    def snapshot(self) -> tuple[Mapping[str, object], ...]:
        """Return an immutable deep copy of completed episode records."""
        return tuple(
            self._readonly_copy(episode)  # type: ignore[return-value]
            for episode in self._episodes
        )

    def pooled(self, key: str) -> dict[str, object]:
        if not self._episodes:
            raise ValueError("cannot summarize empty episode log")
        values = [self._finite_scalar(episode[key], key) for episode in self._episodes]
        counts = {
            task: sum(episode["task"] == task for episode in self._episodes)
            for task in TASKS
        }
        return {"mean": sum(values) / len(values), "task_episode_counts": counts}


class RetentionEvaluator:
    """Apply frozen higher/lower metric directions without reward coupling."""

    def __init__(self, contract: NumericContract) -> None:
        self.contract = contract
        self.specification = contract.payload["retention"]

    def score(self, metrics: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        if set(metrics) != set(TASKS):
            raise ValueError("retention requires exactly four task records")
        task_scores: dict[str, float] = {}
        for task in TASKS:
            expected = self.specification[task]
            values = metrics[task]
            if set(values) != set(expected):
                raise ValueError(f"retention metrics for {task} are incomplete")
            scores: list[float] = []
            for name, spec in expected.items():
                value = EpisodeMetricLog._finite_scalar(values[name], f"{task}.{name}")
                baseline = float(spec["baseline"])
                if value < 0.0:
                    raise ValueError(
                        f"retention metric {task}.{name} must be nonnegative"
                    )
                ratio = (
                    value / baseline
                    if spec["direction"] == "higher"
                    else (1.0 if value == 0.0 else baseline / value)
                )
                scores.append(min(1.0, ratio))
            task_scores[task] = min(scores)
        return {"gate_score": min(task_scores.values()), "task_scores": task_scores}
