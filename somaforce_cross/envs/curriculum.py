"""Checkpointable contract-driven Phase 4B5 curriculum controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from somaforce_cross.envs.numeric_contract import NumericContract, STAGES, TASKS


@dataclass(frozen=True)
class StageDecision:
    stage: str
    promoted: bool
    rolled_back: bool


class Phase4B5Curriculum:
    """Per-task window gates, cadence, and atomic checkpoint restoration."""

    def __init__(
        self, contract: NumericContract, *, stage: str = "C1", transitions: int = 0
    ) -> None:
        if stage not in STAGES or transitions < 0:
            raise ValueError("stage/transitions are invalid")
        self.contract = contract
        self.stage = stage
        self.transitions = int(transitions)
        self.last_evaluation_transition = -1
        self._passes = 0
        self._rollbacks = 0
        self.episode_counters = {task: 0 for task in TASKS}
        self.nominal_episode_counters = {task: 0 for task in TASKS}

    def state_dict(self) -> dict[str, object]:
        return {
            "contract_hash": self.contract.sha256,
            "episode_counters": dict(self.episode_counters),
            "last_evaluation_transition": self.last_evaluation_transition,
            "nominal_episode_counters": dict(self.nominal_episode_counters),
            "passes": self._passes,
            "rollbacks": self._rollbacks,
            "stage": self.stage,
            "transitions": self.transitions,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "contract_hash",
            "episode_counters",
            "last_evaluation_transition",
            "nominal_episode_counters",
            "passes",
            "rollbacks",
            "stage",
            "transitions",
        }
        if set(state) != expected or state["contract_hash"] != self.contract.sha256:
            raise ValueError("curriculum checkpoint is not for this numeric contract")
        stage = state["stage"]
        if stage not in STAGES:
            raise ValueError("invalid curriculum stage")
        counters: dict[str, dict[str, int]] = {}
        for name in ("episode_counters", "nominal_episode_counters"):
            value = state[name]
            if not isinstance(value, Mapping) or set(value) != set(TASKS):
                raise ValueError(f"invalid {name}")
            prepared: dict[str, int] = {}
            for task, count in value.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"invalid {name}.{task}")
                prepared[task] = count
            counters[name] = prepared
        integers: dict[str, int] = {}
        for name in (
            "last_evaluation_transition",
            "passes",
            "rollbacks",
            "transitions",
        ):
            value = state[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (-1 if name == "last_evaluation_transition" else 0)
            ):
                raise ValueError(f"invalid {name}")
            integers[name] = value
        if integers["last_evaluation_transition"] > integers["transitions"]:
            raise ValueError("evaluation transition cannot exceed transitions")
        self.stage = str(stage)
        self.transitions = integers["transitions"]
        self.last_evaluation_transition = integers["last_evaluation_transition"]
        self._passes = integers["passes"]
        self._rollbacks = integers["rollbacks"]
        self.episode_counters = counters["episode_counters"]
        self.nominal_episode_counters = counters["nominal_episode_counters"]

    def complete_episode(self, task: str, *, nominal: bool = False) -> int:
        if task not in TASKS or not isinstance(nominal, bool):
            raise ValueError("task/nominal is invalid")
        index = self.episode_counters[task]
        self.episode_counters[task] += 1
        if nominal:
            self.nominal_episode_counters[task] += 1
        return index

    def _validate_metrics(self, metrics: Mapping[str, Mapping[str, object]]) -> None:
        if set(metrics) != set(TASKS):
            raise ValueError("window requires every task")
        windows = self.contract.payload["curriculum"]["windows"]
        for task, row in metrics.items():
            if set(row) != {
                "failure",
                "nominal_episodes",
                "nominal_retention",
                "stage_episodes",
                "success",
            }:
                raise ValueError(f"window schema for {task} is invalid")
            for key in ("stage_episodes", "nominal_episodes"):
                value = row[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{task}.{key} is invalid")
            if row["stage_episodes"] < int(windows["stage_episodes_per_task"]) or row[
                "nominal_episodes"
            ] < int(windows["nominal_episodes_per_task"]):
                raise ValueError("window has insufficient per-task episode evidence")
            for key in ("success", "failure", "nominal_retention"):
                value = row[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{task}.{key} is nonfinite or invalid")

    def evaluate_window(
        self, *, transitions: int, metrics: Mapping[str, Mapping[str, object]]
    ) -> StageDecision:
        if (
            isinstance(transitions, bool)
            or not isinstance(transitions, int)
            or transitions < 0
        ):
            raise ValueError("transitions is invalid")
        self._validate_metrics(metrics)
        cadence = int(
            self.contract.payload["curriculum"]["windows"][
                "evaluation_interval_transitions"
            ]
        )
        previous = self.last_evaluation_transition
        if (
            transitions <= previous
            or transitions < self.transitions
            or transitions % cadence != 0
        ):
            raise ValueError("evaluation transition violates cadence or monotonicity")
        if previous >= 0 and (transitions - previous) % cadence != 0:
            raise ValueError("evaluation cadence was skipped")
        self.transitions = transitions
        self.last_evaluation_transition = transitions
        if self.stage == "C3":
            return StageDecision("C3", False, False)
        gates = self.contract.payload["curriculum"]["gates"]
        current = gates[self.stage]
        rollback = gates["rollback"]
        rows = tuple(metrics.values())
        passed = transitions >= int(current["min_transitions"]) and all(
            float(row["success"]) >= float(current["success_min"])
            and float(row["failure"]) <= float(current["failure_max"])
            and float(row["nominal_retention"]) >= float(current["retention_min"])
            for row in rows
        )
        requires_rollback = self.stage == "C2" and any(
            float(row["success"])
            < float(current["success_min"]) - float(rollback["success_margin"])
            or float(row["failure"])
            > float(current["failure_max"]) + float(rollback["failure_margin"])
            or float(row["nominal_retention"]) < float(rollback["retention_min"])
            for row in rows
        )
        self._passes = self._passes + 1 if passed else 0
        self._rollbacks = self._rollbacks + 1 if requires_rollback else 0
        windows = self.contract.payload["curriculum"]["windows"]
        if self._passes >= int(windows["promotion_windows"]):
            self.stage = "C2" if self.stage == "C1" else "C3"
            self._passes = 0
            self._rollbacks = 0
            return StageDecision(self.stage, True, False)
        if self.stage == "C2" and self._rollbacks >= int(windows["rollback_windows"]):
            self.stage = "C1"
            self._passes = 0
            self._rollbacks = 0
            return StageDecision(self.stage, False, True)
        return StageDecision(self.stage, False, False)

    def sample_tasks(
        self, count: int, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be nonnegative")
        probabilities = self.contract.payload["curriculum"]["task_probability"]
        if tuple(probabilities[task] for task in TASKS) != (0.25, 0.25, 0.25, 0.25):
            raise ValueError("task balance contract was corrupted")
        return torch.arange(count, device=device, dtype=torch.int64) % len(TASKS)
