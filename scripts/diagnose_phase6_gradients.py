#!/usr/bin/env python3
"""Read-only C1/C2 diagnostic for the immutable Phase 6 segment 47 endpoint.

The module remains importable without Isaac Lab.  Workers import Isaac, RSL-RL,
and environment code only after AppLauncher has been constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "834691b5eef7cbcbe3bf439630f81a6a1be7844065593f73b409ad5422615ae1"
RESULT_MARKER = "PHASE6_DIAGNOSTIC_RESULT="
SUCCESS_MARKERS = ("JSON_WRITTEN", RESULT_MARKER, "ENV_CLOSED")
TASKS = ("push_door_hand", "push_box", "move_suitcase", "move_largebox")
GROUPS = ("all", "semantic_pipeline", "actor", "critic", "log_std")
OBJECTIVES = (
    "surrogate",
    "value_raw",
    "value_weighted",
    "entropy",
    "ppo",
    "semantic_dir",
    "semantic_mag",
    "semantic_total",
    "combined",
)
OUTCOME_MODES = ("residual", "scaffold_only")
PROBES = ("gradient", *OUTCOME_MODES)
AUTHORITY_CASES = (
    "c2_env_c1_authority",
    "c2_env_c2_authority",
    "scaffold_only",
)
SUITCASE_WINDOWS = ("contact", "lift", "carry", "set_down")
DRIFT_SEGMENTS = (47, 51, 52, 59, 79)
ORIGINAL_CHECKPOINTS = {
    47: "b7c7c495394882218d8c941f31cdce76e9b993ac2073891bb4a156128693348a",
    51: "6b4d2845d4194822aabd15c4d5d29b3980dfb5adc75bb3ac297a034516442ab5",
    52: "97f1770323636195ea83c2363b31c1d49a7d180a1a07f05416bd6e99e6eaccf5",
    59: "15ee9d673603050c65b003fd362b3765ee06dafc95f1845434d2deca297befa6",
    79: "ea7c015f82a5a349387e966f33de55a6f7d8efc2a26b17c08698e55fb3ee5b24",
}
REBOUND_CHECKPOINTS = {
    47: (
        "834691b5eef7cbcbe3bf439630f81a6a1be7844065593f73b409ad5422615ae1",
        "a20bfaf08bcaa4bc10d76dbc8c6ba54f2966d7113d20b7ff57283be6b0146143",
    ),
    51: (
        "b8f844d28d8653c714a3c56251eb4c5f845fe1ad6bf3c6f676bb67a7881be85b",
        "dbb47719ca8c34f4ce2b71b79771ab5ca56e4ae25834b6e80f5ce6195e7ee8dd",
    ),
    52: (
        "3f26f3b8ed35fae758f2ebf60cedd3a411cd8ef7fc5e8e9f4a16e5483b31799f",
        "262592ed2f564da4d685fb35598f3a6e7c8d89a495793ad19fd61c97451089a6",
    ),
    59: (
        "b57031a71062adcbe331b8d5ff8f5d23d25343ec796389476ce1ba8dc680f5ed",
        "2b88ac7cd8c81580951e89dd933ee57fa1503eb4f47f95efba104e416a982bd3",
    ),
    79: (
        "6573a5170f88e2aed90075d05e3acf31081fd144fdbcaed10b9806c36a085b04",
        "6b517addc09a1be56927be4b0ad58ba3ceb35ea043a7d3bc8ca6f957c1a57b2a",
    ),
}
REBOUND_ROOT = (
    REPO_ROOT
    / "outputs/phase6_gradient_diagnostics"
    / "phase6-diagnostic-source-rebind-2b7bfbc-20260817"
)
OUTCOME_RECORD_KEYS = {
    "diagnostics",
    "env_id",
    "failure",
    "family",
    "invalid",
    "mode",
    "raw_reward_sums",
    "return",
    "seed",
    "stage",
    "steps",
    "subset",
    "success",
    "task",
    "timeout",
    "weighted_reward_sums",
}
OUTCOME_DIAGNOSTICS = (
    "p50_wrench",
    "p95_wrench",
    "p99_wrench",
    "impulse",
    "p95_force_rate",
    "contact_fraction",
    "contact_loss",
    "sensor_quality",
    "saturation",
    "dropout",
    "arms_residual_norm",
    "waist_residual_norm",
    "legs_residual_norm",
    "semantic_entropy",
    "semantic_kl",
    "stability_margin",
)
OUTCOME_REWARD_KEYS = {
    "contact",
    "force",
    "nonfinite",
    "progress",
    "rate",
    "residual",
    "stability",
    "terminal_failure",
    "terminal_success",
}


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_checkpoint_sha256(value: str) -> bool:
    return value == CHECKPOINT_SHA256 or value in {
        checkpoint_sha256 for checkpoint_sha256, _ in REBOUND_CHECKPOINTS.values()
    }


def _original_checkpoint(segment: int) -> Path:
    return (
        REPO_ROOT
        / "outputs/phase6_learning_acceptance"
        / "phase6-learning-main-4gpu-64env-2442iter-20260812_185544"
        / f"segment_{segment:04d}"
        / "post_evaluation.pt"
    )


def _rebound_attempt(segment: int) -> Path:
    return REBOUND_ROOT / f"segment_{segment:04d}" / "attempt_0001"


def _rebound_checkpoint(segment: int) -> Path:
    return _rebound_attempt(segment) / "post_evaluation_mainrunner_rebound.pt"


def _rebind_record(segment: int) -> Path:
    return _rebound_attempt(segment) / "source_rebind_mainrunner.json"


def _diagnostic_checkpoint_binding(checkpoint: Path) -> dict[str, object]:
    """Require one of the five diagnostic-only, current-source-bound endpoints."""
    for segment, (rebound_sha256, record_sha256) in REBOUND_CHECKPOINTS.items():
        if checkpoint.resolve() != _rebound_checkpoint(segment).resolve():
            continue
        record_path = _rebind_record(segment)
        if (
            _sha256(checkpoint) != rebound_sha256
            or _sha256(record_path) != record_sha256
        ):
            raise ValueError("diagnostic rebound checkpoint or record SHA256 mismatch")
        record = _read_json(record_path)
        checkpoint_record = record.get("checkpoint")
        if (
            record.get("status") != "ok"
            or not isinstance(checkpoint_record, Mapping)
            or checkpoint_record.get("input_path") != str(_original_checkpoint(segment))
            or checkpoint_record.get("input_sha256") != ORIGINAL_CHECKPOINTS[segment]
            or checkpoint_record.get("output_path") != str(checkpoint)
            or checkpoint_record.get("output_sha256") != rebound_sha256
            or record.get("invariants")
            != {
                "payload_except_source_manifest": True,
                "source_manifest_only_change": True,
            }
        ):
            raise ValueError("diagnostic rebound record binding is invalid")
        return {
            "diagnostic_only": True,
            "original_path": str(_original_checkpoint(segment)),
            "original_sha256": ORIGINAL_CHECKPOINTS[segment],
            "rebind_record_path": str(record_path),
            "rebind_record_sha256": record_sha256,
            "rebound_path": str(checkpoint),
            "rebound_sha256": rebound_sha256,
            "segment": segment,
        }
    raise ValueError("diagnostic refuses a historical or unbound checkpoint input")


def _update_digest(digest: object, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=str):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _update_digest(digest, item)
    elif value is None:
        digest.update(b"none")
    else:
        digest.update(repr(value).encode("utf-8"))


def _hash_object(value: object) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _authority_case(*, authority_stage: str | None, probe: str) -> str:
    """Name the single-process diagnostic condition without changing runtime code."""
    if probe == "scaffold_only":
        return "scaffold_only"
    if authority_stage not in {"C1", "C2"}:
        raise ValueError("residual diagnostic requires C1 or C2 authority")
    return f"c2_env_{authority_stage.lower()}_authority"


def _bind_diagnostic_authority(
    environment: object, *, environment_stage: str, authority_stage: str | None
) -> dict[str, object]:
    """Override only this process' authority lookup while preserving its buffers.

    The residual environment owns the C2 mismatch/sensor rows through its normal
    curriculum stage.  Rebinding the instance method leaves the authority module
    state dict unchanged and cannot affect a checkpoint or production source.
    """
    if environment_stage != "C2":
        raise ValueError("authority discrimination fixes the environment at C2")
    if authority_stage not in {"C1", "C2"}:
        raise ValueError("authority discrimination requires C1 or C2 authority")
    authority = getattr(environment, "authority")
    before = _hash_object(authority.state_dict())
    selected = authority(1 if authority_stage == "C1" else 2).detach().clone()
    original = authority.forward

    def diagnostic_forward(_self: object, stage: object) -> torch.Tensor:
        # `_pre_physics_step` always requests the environment's C2 index.
        if stage != 2:
            raise AssertionError("diagnostic authority received a non-C2 environment")
        return selected

    authority.forward = types.MethodType(diagnostic_forward, authority)
    return {
        "environment_stage": environment_stage,
        "authority_stage": authority_stage,
        "authority_buffer_sha256_before": before,
        "authority_buffer_sha256_after": _hash_object(authority.state_dict()),
        "selected_authority_sha256": _hash_object(selected),
        "process_local_override": True,
        "original_forward_name": original.__name__,
    }


def _suitcase_time_series_record(
    *, environment: object, step: int
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Capture all fixed-horizon suitcase event signals before one control step."""
    adapter = environment.adapter
    signals = adapter.progress_signals()
    contact = signals.contact_truth.any(dim=-1)
    lifted = adapter.lift_latched
    exhausted = signals.reference_exhausted
    object_local = adapter.object.data.root_link_pos_w - environment.scene.env_origins
    set_down_error = torch.linalg.vector_norm(
        object_local - adapter.reference_positions[-1], dim=-1
    )
    windows = {
        "contact": contact,
        "lift": lifted,
        "carry": lifted & ~exhausted,
        "set_down": exhausted,
    }
    return (
        {
            "step": step,
            "contact_fraction": float(contact.float().mean().item()),
            "lift_fraction": float(lifted.float().mean().item()),
            "carry_fraction": float(windows["carry"].float().mean().item()),
            "set_down_fraction": float(exhausted.float().mean().item()),
            "reference_exhausted_fraction": float(exhausted.float().mean().item()),
            "set_down_error_mean": float(set_down_error.mean().item()),
            "window_counts": {
                name: int(mask.sum().item()) for name, mask in windows.items()
            },
        },
        windows,
    )


def _summarize_suitcase_windows(
    *,
    records: Sequence[Mapping[str, object]],
    window_diagnostics: Mapping[str, Mapping[str, object] | None],
) -> dict[str, object]:
    if len(records) != 472:
        raise ValueError("suitcase diagnostic requires the complete 472-step horizon")
    result: dict[str, object] = {}
    for window in SUITCASE_WINDOWS:
        active_steps = sum(
            int(record["window_counts"][window]) > 0 for record in records
        )
        result[window] = {
            "active_steps": active_steps,
            "available": active_steps > 0,
            "diagnostic": window_diagnostics[window],
        }
    return result


def _window_gradient_diagnostic(
    *,
    policy: torch.nn.Module,
    samples: Sequence[Mapping[str, object]],
    ppo: Mapping[str, object],
) -> dict[str, object] | None:
    """Differentiate a fixed event-window trace without an optimizer update."""
    if not samples:
        return None
    from somaforce_cross.learning.losses import semantic_losses

    keys = ("policy", "critic", "semantic_target")
    observations = {
        key: torch.cat([sample["observations"][key] for sample in samples], dim=0)
        for key in keys
    }
    actions = torch.cat([sample["actions"] for sample in samples], dim=0)
    rewards = torch.cat([sample["rewards"] for sample in samples], dim=0)
    policy.act(observations)
    log_prob = policy.get_actions_log_prob(actions)
    ratio = torch.exp(log_prob - log_prob.detach())
    advantages = rewards - rewards.mean()
    surrogate = torch.maximum(
        -advantages * ratio,
        -advantages
        * torch.clamp(
            ratio, 1.0 - float(ppo["clip_param"]), 1.0 + float(ppo["clip_param"])
        ),
    ).mean()
    value = policy.evaluate(observations)
    returns = rewards.unsqueeze(-1)
    value_raw = (value - returns).pow(2).mean()
    value_weighted = float(ppo["value_loss_coef"]) * value_raw
    entropy = -float(ppo["entropy_coef"]) * policy.entropy.mean()
    semantic = semantic_losses(
        policy.last_semantic_output.p_dir,
        policy.last_semantic_output.p_mag,
        observations["semantic_target"],
    )
    record = _minibatch_diagnostic(
        policy=policy,
        objectives={
            "surrogate": surrogate,
            "value_raw": value_raw,
            "value_weighted": value_weighted,
            "entropy": entropy,
            "ppo": surrogate + value_weighted + entropy,
            "semantic_dir": semantic.dir_loss,
            "semantic_mag": semantic.mag_loss,
            "semantic_total": semantic.total,
            "combined": surrogate + value_weighted + entropy + semantic.total,
        },
    )
    return {
        "sample_count": len(samples),
        "losses": record["losses"],
        "gradient_norms": record["gradient_norms"],
        "ppo_semantic_cosine": record["ppo_semantic_cosine"],
        "combined_norm": record["combined_norm"],
    }


def _signal_name(return_code: int | None) -> str | None:
    if return_code is None or return_code >= 0:
        return None
    return signal.Signals(-return_code).name


def _proc_hwm_kib(pid: int) -> int | None:
    try:
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _terminate_group(process: subprocess.Popen[str], *, grace_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=grace_s)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("diagnostic values must be nonempty and finite")
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "median": ordered[round((len(ordered) - 1) * 0.5)],
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
    }


def _safe_ratio(c2: float, c1: float) -> float | None:
    if not math.isfinite(c1) or not math.isfinite(c2) or c1 == 0.0:
        return None
    return c2 / c1


def _gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite read-only diagnostic gradient")
        total += float(gradient.detach().pow(2).sum().item())
    return total**0.5


def _cosine(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> float:
    dot = left_squared = right_squared = 0.0
    for first, second in zip(left, right, strict=True):
        if first is not None:
            left_squared += float(first.detach().pow(2).sum().item())
        if second is not None:
            right_squared += float(second.detach().pow(2).sum().item())
        if first is not None and second is not None:
            dot += float((first.detach() * second.detach()).sum().item())
    return (
        0.0
        if not left_squared or not right_squared
        else dot / math.sqrt(left_squared * right_squared)
    )


def _parameter_groups(
    policy: torch.nn.Module,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    groups = {
        "all": tuple(policy.parameters()),
        "semantic_pipeline": tuple(policy.semantic_pipeline.parameters()),
        "actor": tuple(policy.actor.parameters()),
        "critic": tuple(policy.critic.parameters()),
        "log_std": (policy.log_std,),
    }
    if any(not group for group in groups.values()):
        raise RuntimeError("diagnostic parameter group is empty")
    return groups


def _grads(
    objective: torch.Tensor, parameters: Sequence[torch.nn.Parameter]
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
    )


def _minibatch_diagnostic(
    *, policy: torch.nn.Module, objectives: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    if set(objectives) != set(OBJECTIVES):
        raise ValueError("diagnostic objective schema differs")
    groups = _parameter_groups(policy)
    all_gradients = {
        name: _grads(loss, groups["all"]) for name, loss in objectives.items()
    }
    parameter_indices = {
        id(parameter): index for index, parameter in enumerate(groups["all"])
    }

    def select(
        gradients: Sequence[torch.Tensor | None],
        parameters: Sequence[torch.nn.Parameter],
    ) -> tuple[torch.Tensor | None, ...]:
        return tuple(
            gradients[parameter_indices[id(parameter)]] for parameter in parameters
        )

    combined_norm = _gradient_norm(all_gradients["combined"])
    return {
        "losses": {
            name: float(loss.detach().item()) for name, loss in objectives.items()
        },
        "gradient_norms": {
            name: {
                group: _gradient_norm(select(all_gradients[name], parameters))
                for group, parameters in groups.items()
            }
            for name in objectives
        },
        "ppo_semantic_cosine": _cosine(
            all_gradients["ppo"], all_gradients["semantic_total"]
        ),
        "combined_norm": combined_norm,
        "preclip_pressure": combined_norm > 1.0,
    }


def _summarize_gradient_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(records) != 24:
        raise ValueError("diagnostic requires exactly 3 epochs x 8 mini-batches")
    objectives: dict[str, object] = {}
    for objective in OBJECTIVES:
        objectives[objective] = {
            group: _quantiles(
                [
                    float(record["gradient_norms"][objective][group])
                    for record in records
                ]
            )
            for group in GROUPS
        }
    return {
        "minibatches": 24,
        "records": [dict(record) for record in records],
        "losses": {
            objective: _quantiles(
                [float(record["losses"][objective]) for record in records]
            )
            for objective in OBJECTIVES
        },
        "objectives": objectives,
        "combined_norm": _quantiles(
            [float(record["combined_norm"]) for record in records]
        ),
        "ppo_semantic_cosine": _quantiles(
            [float(record["ppo_semantic_cosine"]) for record in records]
        ),
        "preclip_fraction": sum(bool(record["preclip_pressure"]) for record in records)
        / 24,
    }


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_outcome_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected: int,
    task: str,
    mode: str,
    stage: str,
) -> None:
    if len(rows) != expected:
        raise ValueError("paired outcome quota differs from the production schedule")
    for row in rows:
        if set(row) != OUTCOME_RECORD_KEYS:
            raise ValueError("paired outcome record differs from production schema")
        if (
            row["task"] != task
            or row["mode"] != mode
            or row["stage"] != stage
            or row["subset"] not in {"stage", "nominal"}
        ):
            raise ValueError("paired outcome identity is invalid")
        for name in ("success", "failure", "invalid", "timeout"):
            if not isinstance(row[name], bool):
                raise ValueError(f"paired outcome {name} is not boolean")
        diagnostics = row["diagnostics"]
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(
            OUTCOME_DIAGNOSTICS
        ):
            raise ValueError("paired outcome diagnostics differ from production schema")
        _finite_number(row["return"], name="outcome.return")
        for name in OUTCOME_DIAGNOSTICS:
            _finite_number(diagnostics[name], name=f"outcome.diagnostics.{name}")
        for name in ("raw_reward_sums", "weighted_reward_sums"):
            reward = row[name]
            if not isinstance(reward, Mapping) or set(reward) != OUTCOME_REWARD_KEYS:
                raise ValueError("paired outcome reward schema differs from production")
            for key, value in reward.items():
                _finite_number(value, name=f"outcome.{name}.{key}")
        for name in ("env_id", "seed", "steps"):
            if (
                isinstance(row[name], bool)
                or not isinstance(row[name], int)
                or int(row[name]) < 0
            ):
                raise ValueError(f"paired outcome {name} is invalid")
        if not isinstance(row["family"], str) or not row["family"]:
            raise ValueError("paired outcome family is invalid")
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("paired outcome contains repeated seeds")


def _aggregate_outcomes(
    rows: Sequence[Mapping[str, object]], *, task: str, mode: str, stage_name: str
) -> dict[str, object]:
    _validate_outcome_rows(rows, expected=384, task=task, mode=mode, stage=stage_name)
    stage = [row for row in rows if row["subset"] == "stage"]
    nominal = [row for row in rows if row["subset"] == "nominal"]
    if len(stage) != 256 or len(nominal) != 128:
        raise ValueError("paired outcome quotas are not 256 stage plus 128 nominal")

    def mean_value(rows_: Sequence[Mapping[str, object]], key: str) -> float:
        values = [_finite_number(row[key], name=key) for row in rows_]
        return sum(values) / len(values)

    diagnostics = [row["diagnostics"] for row in stage]
    assert all(isinstance(item, Mapping) for item in diagnostics)
    return {
        "mode": mode,
        "reward": mean_value(stage, "return"),
        "success": sum(bool(row["success"]) for row in stage) / len(stage),
        "failure": sum(bool(row["failure"]) for row in stage) / len(stage),
        "invalid": sum(bool(row["invalid"]) for row in stage) / len(stage),
        "timeout": sum(bool(row["timeout"]) for row in stage) / len(stage),
        "nominal_success": sum(bool(row["success"]) for row in nominal) / len(nominal),
        "nominal_invalid": sum(bool(row["invalid"]) for row in nominal),
        "nominal_saturation": sum(
            float(row["diagnostics"]["saturation"]) for row in nominal
        )
        / len(nominal),
        "diagnostics": {
            name: sum(float(item[name]) for item in diagnostics) / len(diagnostics)
            for name in OUTCOME_DIAGNOSTICS
        },
    }


def _paired_outcome_summary(
    residual_rows: Sequence[Mapping[str, object]],
    scaffold_rows: Sequence[Mapping[str, object]],
    *,
    task: str,
    stage: str,
) -> dict[str, object]:
    residual = _aggregate_outcomes(
        residual_rows, task=task, mode="residual", stage_name=stage
    )
    scaffold = _aggregate_outcomes(
        scaffold_rows, task=task, mode="scaffold_only", stage_name=stage
    )
    residual_seeds = sorted(int(row["seed"]) for row in residual_rows)
    scaffold_seeds = sorted(int(row["seed"]) for row in scaffold_rows)
    if residual_seeds != scaffold_seeds:
        raise ValueError("paired outcome modes do not use identical seeds")
    denominator = float(scaffold["nominal_success"])
    if denominator <= 0.0:
        raise ValueError("nominal scaffold success cannot define retention")
    return {
        "failure": residual["failure"],
        "invalid": residual["invalid"],
        "nominal_invalid": residual["nominal_invalid"],
        "nominal_residual_success": residual["nominal_success"],
        "nominal_saturation": residual["nominal_saturation"],
        "nominal_scaffold_success": scaffold["nominal_success"],
        "retention": float(residual["nominal_success"]) / denominator,
        "reward": residual["reward"],
        "success": residual["success"],
        "timeout": residual["timeout"],
        "residual": residual,
        "scaffold_only": scaffold,
        "seed_sha256": _hash_object(residual_seeds),
    }


def _support_decision(
    c1: Mapping[str, object], c2: Mapping[str, object]
) -> dict[str, object]:
    c1_suitcase = c1["tasks"]["move_suitcase"]
    c2_suitcase = c2["tasks"]["move_suitcase"]
    reward_ratio = _safe_ratio(
        float(c2_suitcase["reward"]), float(c1_suitcase["reward"])
    )
    outcome = {
        "success_delta": float(c2_suitcase["success"]) - float(c1_suitcase["success"]),
        "retention_delta": float(c2_suitcase["retention"])
        - float(c1_suitcase["retention"]),
        "failure_delta": float(c2_suitcase["failure"]) - float(c1_suitcase["failure"]),
        "reward_ratio": reward_ratio,
    }
    rollout_material = (
        outcome["success_delta"] <= -0.15
        or outcome["retention_delta"] <= -0.10
        or outcome["failure_delta"] >= 0.05
        or (reward_ratio is not None and reward_ratio <= 0.8)
    )
    ratio = _safe_ratio(
        float(c2["gradient"]["combined_norm"]["mean"]),
        float(c1["gradient"]["combined_norm"]["mean"]),
    )
    gradient_material = (
        (ratio is not None and (ratio >= 2.0 or ratio <= 0.5))
        or float(c2["gradient"]["ppo_semantic_cosine"]["mean"]) <= 0.0
        or float(c2["gradient"]["preclip_fraction"]) > 0.0
    )
    invariants = bool(c1["invariants_ok"]) and bool(c2["invariants_ok"])
    status = (
        "supported"
        if invariants and rollout_material and gradient_material
        else "not_supported"
        if invariants
        else "inconclusive"
    )
    return {
        "status": status,
        "c2_stage_sensitivity_supported": status == "supported",
        "invariants_ok": invariants,
        "move_suitcase": outcome,
        "rollout_material": rollout_material,
        "gradient_material": gradient_material,
        "combined_norm_ratio": ratio,
        "claim_boundary": "diagnostic only; no root-cause or learning-acceptance claim",
    }


def _worker_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=("C1", "C2"), required=True)
    parser.add_argument("--authority-stage", choices=("C1", "C2"))
    parser.add_argument("--probe", choices=PROBES, required=True)
    parser.add_argument("--timeout-s", type=int, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _plain_parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(mode,), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=("C1", "C2"))
    parser.add_argument("--authority-stage", choices=("C1", "C2"))
    if mode in {"rank-wrapper", "summarize-stage"}:
        parser.add_argument("--probe", choices=PROBES)
    parser.add_argument("--timeout-s", type=int, required=True)
    return parser


def _worker_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
        "--output-dir",
        str(args.output_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--stage",
        args.stage,
        "--probe",
        args.probe,
        "--timeout-s",
        str(args.timeout_s),
        "--headless",
    ]
    if args.authority_stage is not None:
        command.extend(("--authority-stage", args.authority_stage))
    return command


def _run_rank_wrapper(args: argparse.Namespace) -> int:
    rank, local_rank, world_size = (
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
    )
    if world_size != 4 or rank not in range(4):
        raise ValueError("diagnostic requires exactly four ranks")
    if args.probe not in PROBES:
        raise ValueError("diagnostic wrapper requires one explicit probe")
    rank_dir = args.output_dir / args.stage / args.probe / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)
    command = _worker_command(args)
    _atomic_json(
        rank_dir / "command.json",
        {
            "command": command,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
        },
    )
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def drain() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    markers: list[str] = []
    maximum_hwm: int | None = None
    timed_out = False
    log_path = rank_dir / "worker.log"
    with log_path.open("w", encoding="utf-8") as handle:
        done = False
        while not done:
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                line = ""
            if line is None:
                done = True
            elif line:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
                if line.startswith("JSON_WRITTEN"):
                    markers.append("JSON_WRITTEN")
                elif line.startswith(RESULT_MARKER):
                    markers.append(RESULT_MARKER.rstrip("="))
                elif line.startswith("ENV_CLOSED"):
                    markers.append("ENV_CLOSED")
            observed = _proc_hwm_kib(process.pid)
            if observed is not None:
                maximum_hwm = max(maximum_hwm or observed, observed)
            if process.poll() is None and time.monotonic() - started > args.timeout_s:
                timed_out = True
                _terminate_group(process)
    process.wait()
    result_path = rank_dir / "result.json"
    result = _read_json(result_path) if result_path.is_file() else None
    wrapper = {
        "command": command,
        "elapsed_s": round(time.monotonic() - started, 3),
        "exit_code": process.returncode if process.returncode >= 0 else None,
        "signal": _signal_name(process.returncode),
        "timed_out": timed_out,
        "timeout_kind": "wall_clock" if timed_out else None,
        "vmhwm_kib": maximum_hwm,
        "markers": markers,
        "marker_order_valid": markers == [item.rstrip("=") for item in SUCCESS_MARKERS],
        "result_json": str(result_path),
        "result_sha256": _sha256(result_path) if result_path.is_file() else None,
        "result_status": None if result is None else result.get("status"),
        "rank": rank,
        "world_size": world_size,
    }
    wrapper["passed"] = bool(
        result
        and result.get("status") == "ok"
        and wrapper["marker_order_valid"]
        and wrapper["exit_code"] == 0
        and not timed_out
    )
    _atomic_json(rank_dir / "wrapper.json", wrapper)
    if not wrapper["passed"]:
        raise RuntimeError("rank diagnostic wrapper evidence gate failed")
    return 0


def _source_hashes() -> dict[str, str]:
    paths = (
        REPO_ROOT / "scripts/train_phase6.py",
        REPO_ROOT / "somaforce_cross/learning/joint_runner.py",
        REPO_ROOT / "somaforce_cross/learning/semantic_ppo.py",
        Path(__file__).resolve(),
    )
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths}


def _protected_state(
    *,
    checkpoint: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    curriculum: object,
    rank_rng: object,
    normalizer: object,
    source_manifest: object,
) -> dict[str, object]:
    from somaforce_cross.learning.joint_runner import optimizer_step, state_dict_sha256

    normalizer_state = (
        normalizer if isinstance(normalizer, Mapping) else normalizer.state_dict()
    )
    return {
        "checkpoint": _sha256(checkpoint),
        "policy": state_dict_sha256(policy.state_dict()),
        "optimizer": state_dict_sha256(optimizer.state_dict()),
        "optimizer_step": optimizer_step(optimizer),
        "curriculum": _hash_object(curriculum),
        "rank_rng_payload": _hash_object(rank_rng),
        "normalizer": _hash_object(normalizer_state),
        "source_manifest": _hash_object(source_manifest),
    }


def _gradient_probe(
    *,
    algorithm: object,
    observations: Mapping[str, torch.Tensor],
    ppo: Mapping[str, object],
) -> list[dict[str, object]]:
    from somaforce_cross.learning.losses import semantic_losses

    policy = algorithm.policy
    records: list[dict[str, object]] = []
    for batch in algorithm.storage.mini_batch_generator(
        int(ppo["num_mini_batches"]), int(ppo["num_learning_epochs"])
    ):
        obs, actions, target_values, advantages, returns, old_log_prob, *_ = batch
        policy.act(obs)
        log_prob = policy.get_actions_log_prob(actions)
        ratio = torch.exp(log_prob - torch.squeeze(old_log_prob))
        surrogate = torch.maximum(
            -torch.squeeze(advantages) * ratio,
            -torch.squeeze(advantages)
            * torch.clamp(
                ratio, 1.0 - float(ppo["clip_param"]), 1.0 + float(ppo["clip_param"])
            ),
        ).mean()
        value = policy.evaluate(obs)
        value_clipped = target_values + (value - target_values).clamp(
            -float(ppo["clip_param"]), float(ppo["clip_param"])
        )
        value_raw = torch.maximum(
            (value - returns).pow(2), (value_clipped - returns).pow(2)
        ).mean()
        value_weighted = float(ppo["value_loss_coef"]) * value_raw
        entropy = -float(ppo["entropy_coef"]) * policy.entropy.mean()
        ppo_loss = surrogate + value_weighted + entropy
        semantic = semantic_losses(
            policy.last_semantic_output.p_dir,
            policy.last_semantic_output.p_mag,
            obs["semantic_target"],
        )
        records.append(
            _minibatch_diagnostic(
                policy=policy,
                objectives={
                    "surrogate": surrogate,
                    "value_raw": value_raw,
                    "value_weighted": value_weighted,
                    "entropy": entropy,
                    "ppo": ppo_loss,
                    "semantic_dir": semantic.dir_loss,
                    "semantic_mag": semantic.mag_loss,
                    "semantic_total": semantic.total,
                    "combined": ppo_loss + semantic.total,
                },
            )
        )
    return records


def _run_outcome_mode(
    *,
    args: argparse.Namespace,
    task: object,
    rank: int,
    policy: torch.nn.Module,
    mode: str,
    rank_dir: Path,
    keep_environment_open: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object], object | None]:
    from somaforce_cross.learning.joint_runner import (
        paired_evaluation_rank_seed,
        paired_evaluation_schedule,
        validate_paired_evaluation_schedule,
    )
    from scripts.train_phase6 import (
        _build_environment,
        _validate_completed_episode_record,
    )

    if mode not in OUTCOME_MODES:
        raise ValueError("diagnostic outcome mode is invalid")
    output_dir = rank_dir / "outcome" / mode
    output_dir.mkdir(parents=True, exist_ok=False)
    evaluation_seed = 20262806
    rank_seed = paired_evaluation_rank_seed(evaluation_seed, rank)
    args.runtime_mode = mode
    environment = _build_environment(args, task=task, seed=rank_seed)
    records: list[dict[str, object]] = []
    control_steps = 0
    try:
        authority_override = None
        if mode == "residual":
            authority_override = _bind_diagnostic_authority(
                environment,
                environment_stage=args.stage,
                authority_stage=args.authority_stage or args.stage,
            )
        schedule = paired_evaluation_schedule(
            task=task.task,
            stage=args.stage,
            evaluation_seed=evaluation_seed,
            rank=rank,
            num_envs=64,
            stage_quota=256,
            nominal_quota=128,
            nominal_probe=environment.evaluation_nominal_probe,
        )
        validate_paired_evaluation_schedule(schedule)
        environment.bind_evaluation_schedule(schedule, mode=mode)
        normalizer_before = (
            None
            if environment.normalizer is None
            else _hash_object(environment.normalizer.state_dict())
        )
        if mode == "residual" and normalizer_before is None:
            raise RuntimeError("residual outcome probe has no fixed normalizer")
        observations, _ = environment.reset()
        horizons = {
            "push_door_hand": 508,
            "push_box": 792,
            "move_suitcase": 472,
            "move_largebox": 199,
        }
        max_steps = horizons[task.task] * (256 // 64 + 128 // 64 + 2)
        while len(records) < 384:
            with torch.inference_mode():
                actions = policy.act_inference(observations)
                observations, _, _, _, _ = environment.step(actions)
            control_steps += 1
            records.extend(
                _validate_completed_episode_record(
                    record,
                    task=task.task,
                    mode=mode,
                    stage=args.stage,
                )
                for record in environment.drain_evaluation_completions()
            )
            _atomic_json(
                output_dir / "progress.json",
                {
                    "completed_episodes": len(records),
                    "control_steps": control_steps,
                    "mode": mode,
                    "rank": rank,
                    "stage": args.stage,
                    "status": "running",
                },
            )
            if len(records) > 384 or control_steps > max_steps:
                raise RuntimeError("paired outcome probe did not complete its quota")
        normalizer_after = (
            None
            if environment.normalizer is None
            else _hash_object(environment.normalizer.state_dict())
        )
        if normalizer_before != normalizer_after:
            raise AssertionError("paired outcome probe mutated the fixed normalizer")
        seeds = [int(record["seed"]) for record in records]
        manifest = {
            "authority_override": authority_override,
            "control_steps": control_steps,
            "mode": mode,
            "normalizer_sha256": normalizer_before,
            "row_count": len(records),
            "schedule_sha256": schedule["schedule_sha256"],
            "seed_sha256": _hash_object(sorted(seeds)),
            "subset_counts": {
                subset: sum(record["subset"] == subset for record in records)
                for subset in ("stage", "nominal")
            },
        }
        _atomic_json(
            output_dir / "completed_episode_records.json",
            {**manifest, "records": records},
        )
        _atomic_json(
            output_dir / "progress.json",
            {
                "completed_episodes": len(records),
                "control_steps": control_steps,
                "mode": mode,
                "rank": rank,
                "stage": args.stage,
                "status": "complete",
            },
        )
        if keep_environment_open:
            kept_environment = environment
            environment = None
            return records, manifest, kept_environment
        return records, manifest, None
    finally:
        if environment is not None:
            environment.close()


def _worker_main(values: list[str]) -> int:
    args = _worker_parser().parse_args(values)
    if (
        args.timeout_s <= 0
        or not args.checkpoint.is_file()
        or not _accepted_checkpoint_sha256(_sha256(args.checkpoint))
    ):
        raise ValueError("diagnostic checkpoint or timeout is invalid")
    checkpoint_binding = _diagnostic_checkpoint_binding(args.checkpoint)
    rank, local_rank, world_size = (
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
    )
    rank_dir = args.output_dir / args.stage / args.probe / f"rank_{rank}"
    launcher = environment = None
    result: dict[str, object]
    try:
        if world_size != 4 or rank not in range(4):
            raise ValueError("diagnostic requires exactly four ranks")
        from isaaclab.app import AppLauncher

        args.device = f"cuda:{local_rank}"
        args.num_envs = 64
        torch.cuda.set_device(local_rank)
        launcher = AppLauncher(args)
        from somaforce_cross.learning.actor_critic import ResidualActorCritic
        from somaforce_cross.learning.joint_runner import (
            TransitionMetricCollector,
            load_learning_acceptance_config,
            load_phase6_config,
            load_phase6_task_roster,
            phase6_metrics_schema,
            process_group_timeout,
            restore_learning_checkpoint,
            restore_rank_rng_state,
        )
        from somaforce_cross.learning.runner import _load_rsl_storage_class
        from somaforce_cross.learning.semantic_ppo import SemanticPPO
        from scripts.train_phase6 import (
            _build_environment,
            _current_mainrunner_source_manifest,
        )

        config = load_phase6_config(REPO_ROOT / "configs/phase6_joint_training_v1.json")
        acceptance = load_learning_acceptance_config(
            REPO_ROOT / "configs/phase6_learning_acceptance_v1.json"
        )
        roster = load_phase6_task_roster(
            REPO_ROOT / "configs/phase6_task_roster_v1.json", phase6_config=config
        )
        policy = ResidualActorCritic()
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(config.payload["ppo"]["learning_rate"])
        )
        source_manifest = _current_mainrunner_source_manifest()
        restored = restore_learning_checkpoint(
            args.checkpoint,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=policy,
            optimizer=optimizer,
            source_manifest=source_manifest,
            expected_iteration=1465,
            device="cpu",
        )
        checkpoint_sha = _sha256(args.checkpoint)
        if restored["pre_evaluation"] is not False:
            raise ValueError("diagnostic checkpoint must be a completed endpoint")
        if args.stage == "C2" and checkpoint_sha != CHECKPOINT_SHA256:
            raise ValueError("C2 authority comparison is fixed to segment 47")
        if args.stage == "C2" and restored["curriculum"]["stage"] != "C2":
            raise ValueError("segment 47 endpoint must retain its C2 curriculum")
        torch.distributed.init_process_group(
            "nccl", init_method="env://", timeout=process_group_timeout(config)
        )
        restore_rank_rng_state(
            restored["rank_rng"],
            rank=rank,
            local_device_index=local_rank,
            cuda_state_count=world_size,
        )
        runtime_rng_start = _hash_object(
            {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(local_rank),
            }
        )
        policy = policy.to(f"cuda:{local_rank}")
        policy.eval()
        task = roster.tasks[rank]
        if args.probe in OUTCOME_MODES:
            args.runtime_mode = args.probe
            outcome_rows, outcome_manifest, _ = _run_outcome_mode(
                args=args,
                task=task,
                rank=rank,
                policy=policy,
                mode=args.probe,
                rank_dir=rank_dir,
                keep_environment_open=False,
            )
            runtime_rng_end = _hash_object(
                {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(local_rank),
                }
            )
            result = {
                "status": "ok",
                "probe": args.probe,
                "rank": rank,
                "task": task.task,
                "stage": args.stage,
                "outcome": _aggregate_outcomes(
                    outcome_rows,
                    task=task.task,
                    mode=args.probe,
                    stage_name=args.stage,
                ),
                "outcome_manifest": outcome_manifest,
                "checkpoint": {
                    "path": str(args.checkpoint),
                    "sha256_before": _sha256(args.checkpoint),
                    "sha256_after": _sha256(args.checkpoint),
                    "binding": checkpoint_binding,
                },
                "invariants": {
                    "checkpoint_unchanged": True,
                    "runtime_rng_start": runtime_rng_start,
                    "runtime_rng_end": runtime_rng_end,
                },
                "source_hashes": _source_hashes(),
            }
            _atomic_json(rank_dir / "result.json", result)
            print("JSON_WRITTEN", flush=True)
            print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
            print("ENV_CLOSED", flush=True)
            torch.distributed.destroy_process_group()
            launcher.app.close()
            print("APP_CLOSED", flush=True)
            return 0
        args.runtime_mode = "residual"
        environment = _build_environment(args, task=task, seed=20260806 + rank)
        environment.set_curriculum_stage(args.stage)
        authority_override = _bind_diagnostic_authority(
            environment,
            environment_stage=args.stage,
            authority_stage=args.authority_stage or args.stage,
        )
        if environment.normalizer is None:
            raise RuntimeError("environment did not own its fixed normalizer")
        protected_before = _protected_state(
            checkpoint=args.checkpoint,
            policy=policy,
            optimizer=optimizer,
            curriculum=restored["curriculum"],
            rank_rng=restored["rank_rng"],
            normalizer=environment.normalizer,
            source_manifest=source_manifest,
        )
        algorithm = SemanticPPO(
            policy,
            storage_class=_load_rsl_storage_class(),
            num_learning_epochs=int(config.payload["ppo"]["num_learning_epochs"]),
            num_mini_batches=int(config.payload["ppo"]["num_mini_batches"]),
            clip_param=float(config.payload["ppo"]["clip_param"]),
            gamma=float(config.payload["ppo"]["gamma"]),
            lam=float(config.payload["ppo"]["lam"]),
            value_loss_coef=float(config.payload["ppo"]["value_loss_coef"]),
            entropy_coef=float(config.payload["ppo"]["entropy_coef"]),
            learning_rate=float(config.payload["ppo"]["learning_rate"]),
            max_grad_norm=float(config.payload["ppo"]["max_grad_norm"]),
            use_clipped_value_loss=bool(
                config.payload["ppo"]["use_clipped_value_loss"]
            ),
            schedule=str(config.payload["ppo"]["schedule"]),
            desired_kl=config.payload["ppo"]["desired_kl"],
            device=policy.log_std.device,
            normalize_advantage_per_mini_batch=bool(
                config.payload["ppo"]["normalize_advantage_per_mini_batch"]
            ),
            gradient_sync_hook=None,
        )
        observations, _ = environment.reset()
        algorithm.init_storage(
            num_envs=64, num_transitions_per_env=32, observations=observations
        )
        collector = TransitionMetricCollector(phase6_metrics_schema(config))
        reward_samples: list[float] = []
        suitcase_time_series: list[dict[str, object]] = []
        suitcase_window_samples: dict[str, list[dict[str, object]]] = {
            window: [] for window in SUITCASE_WINDOWS
        }
        gradient_observations: Mapping[str, torch.Tensor] | None = None
        rollout_steps = 472 if task.task == "move_suitcase" else 32
        for step in range(rollout_steps):
            if task.task == "move_suitcase":
                time_record, suitcase_windows = _suitcase_time_series_record(
                    environment=environment, step=step
                )
                suitcase_time_series.append(time_record)
            with torch.inference_mode():
                actions = (
                    algorithm.act(observations)
                    if step < 32
                    else policy.act_inference(observations)
                )
            signals = environment.adapter.progress_signals()
            collector.observe(
                observed_wrench=environment._observed_wrench,
                contact_fraction=signals.contact_truth.mean(dim=-1),
                stability_margin=signals.stability_margin.squeeze(-1),
                residual_arms_norm=torch.linalg.vector_norm(
                    environment._delta_safe[:, environment.authority.ARMS], dim=-1
                ),
                residual_waist_norm=torch.linalg.vector_norm(
                    environment._delta_safe[:, environment.authority.WAIST], dim=-1
                ),
                residual_legs_norm=torch.linalg.vector_norm(
                    environment._delta_safe[:, environment.authority.LEGS], dim=-1
                ),
                saturation_fraction=environment._saturation_mask.float().mean(dim=-1),
                p_cross=policy.last_semantic_output.P_cross,
            )
            next_obs, step_rewards, terminated, time_outs, extras = environment.step(
                actions
            )
            if step < 32:
                algorithm.process_env_step(
                    next_obs,
                    step_rewards,
                    terminated.to(torch.bool) | time_outs.to(torch.bool),
                    {**dict(extras), "time_outs": time_outs},
                )
                if step == 31:
                    gradient_observations = next_obs
            reward_samples.extend(float(value) for value in step_rewards.detach().cpu())
            if task.task == "move_suitcase":
                for window, mask in suitcase_windows.items():
                    samples = suitcase_window_samples[window]
                    if len(samples) >= 32 or not bool(mask.any().item()):
                        continue
                    index = int(mask.nonzero(as_tuple=False)[0].item())
                    samples.append(
                        {
                            "actions": actions[index : index + 1].detach().clone(),
                            "observations": {
                                key: value[index : index + 1].detach().clone()
                                for key, value in observations.items()
                            },
                            "rewards": step_rewards[index : index + 1].detach().clone(),
                        }
                    )
            _atomic_json(
                rank_dir / "progress.json",
                {
                    "rank": rank,
                    "stage": args.stage,
                    "state": "rollout",
                    "control_steps": step + 1,
                    "transitions": min(step + 1, 32) * 64,
                },
            )
            observations = next_obs
        if gradient_observations is None:
            raise AssertionError("gradient rollout did not collect 32 transitions")
        algorithm.compute_returns(gradient_observations)
        records = _gradient_probe(
            algorithm=algorithm,
            observations=gradient_observations,
            ppo=config.payload["ppo"],
        )
        gradient_summary = _summarize_gradient_records(records)
        suitcase_trace: dict[str, object] | None = None
        if task.task == "move_suitcase":
            time_series_path = rank_dir / "suitcase_event_time_series.json"
            _atomic_json(
                time_series_path,
                {
                    "case": _authority_case(
                        authority_stage=args.authority_stage or args.stage,
                        probe=args.probe,
                    ),
                    "environment_stage": args.stage,
                    "horizon_steps": 472,
                    "records": suitcase_time_series,
                },
            )
            suitcase_trace = {
                "path": str(time_series_path),
                "sha256": _sha256(time_series_path),
                "horizon_steps": 472,
                "windows": _summarize_suitcase_windows(
                    records=suitcase_time_series,
                    window_diagnostics={
                        window: _window_gradient_diagnostic(
                            policy=policy,
                            samples=samples,
                            ppo=config.payload["ppo"],
                        )
                        for window, samples in suitcase_window_samples.items()
                    },
                ),
            }
        gradient_rollout = {
            "mean_reward": sum(reward_samples) / len(reward_samples),
            "metrics": collector.transition_metrics(),
            "p_cross_trace": collector.p_cross_trace(),
        }
        runtime_rng_end = _hash_object(
            {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(local_rank),
            }
        )
        protected_after = _protected_state(
            checkpoint=args.checkpoint,
            policy=policy,
            optimizer=optimizer,
            curriculum=restored["curriculum"],
            rank_rng=restored["rank_rng"],
            normalizer=environment.normalizer,
            source_manifest=source_manifest,
        )
        if protected_before != protected_after:
            raise AssertionError("read-only diagnostic mutated protected state")
        result = {
            "status": "ok",
            "probe": "gradient",
            "rank": rank,
            "task": task.task,
            "stage": args.stage,
            "transitions": 2048,
            "gradient": gradient_summary,
            "gradient_rollout": gradient_rollout,
            "authority_override": authority_override,
            "suitcase_event_time_series": suitcase_trace,
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256_before": protected_before["checkpoint"],
                "sha256_after": protected_after["checkpoint"],
                "binding": checkpoint_binding,
            },
            "invariants": {
                "before": protected_before,
                "after": protected_after,
                "unchanged": True,
                "runtime_rng_start": runtime_rng_start,
                "runtime_rng_end": runtime_rng_end,
            },
            "source_hashes": _source_hashes(),
        }
    except BaseException as exc:
        result = {
            "status": "error",
            "rank": rank,
            "stage": args.stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    environment_ready_to_close = environment is not None
    if environment is None and result["status"] == "ok":
        result = {
            **result,
            "status": "error",
            "close_exception_type": "MissingEnvironment",
            "close_exception_message": "successful worker has no final environment",
        }
    _atomic_json(rank_dir / "result.json", result)
    print("JSON_WRITTEN", flush=True)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    environment_closed = False
    if environment_ready_to_close:
        try:
            assert environment is not None
            environment.close()
            environment = None
            environment_closed = True
        except BaseException as close_exc:
            result = {
                **result,
                "status": "error",
                "close_exception_type": type(close_exc).__name__,
                "close_exception_message": str(close_exc),
            }
    if environment_closed and result["status"] == "ok":
        print("ENV_CLOSED", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if launcher is not None:
        launcher.app.close()
        print("APP_CLOSED", flush=True)
    return 0 if result["status"] == "ok" else 1


def _run_plan(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError("diagnostic attempt refuses to overwrite output")
    if (
        args.timeout_s <= 0
        or not args.checkpoint.is_file()
        or not _accepted_checkpoint_sha256(_sha256(args.checkpoint))
    ):
        raise ValueError("diagnostic plan checkpoint or timeout is invalid")
    binding = _diagnostic_checkpoint_binding(args.checkpoint)
    plan = {
        "run_id": args.output_dir.name,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            **binding,
        },
        "environment_stage": "C2",
        "authority_cases": list(AUTHORITY_CASES),
        "stages": ["C1", "C2"],
        "read_only": True,
        "authority_override": {
            "scope": "diagnostic process instance only",
            "forbidden_mutations": [
                "checkpoint_curriculum",
                "policy",
                "optimizer",
                "normalizer",
                "rank_rng_payload",
                "production_source",
            ],
        },
        "gradient_probe": {
            "world_size": 4,
            "num_envs_per_rank": 64,
            "steps": 32,
            "transitions_per_task": 2048,
            "epochs": 3,
            "minibatches": 8,
        },
        "outcome_probe": {
            "stage_episodes_per_task": 256,
            "nominal_episodes_per_task": 128,
            "modes": ["residual", "scaffold_only"],
            "deterministic_actor_mean": True,
        },
        "suitcase_event_trace": {
            "horizon_steps": 472,
            "windows": list(SUITCASE_WINDOWS),
            "window_loss_gradient_samples": 32,
        },
        "c1_nominal_checkpoint_drift": {
            "segments": list(DRIFT_SEGMENTS),
            "rebound_checkpoints": {
                str(segment): {
                    "diagnostic_only": True,
                    "original_sha256": ORIGINAL_CHECKPOINTS[segment],
                    "rebind_record_sha256": REBOUND_CHECKPOINTS[segment][1],
                    "sha256": REBOUND_CHECKPOINTS[segment][0],
                }
                for segment in DRIFT_SEGMENTS
            },
            "fixed_schedule_and_seeds": True,
        },
    }
    _atomic_json(args.output_dir / "plan.json", plan)
    _atomic_json(args.output_dir / "progress.json", {"status": "planned", "stages": []})
    return 0


def _strict_restore_catalog(args: argparse.Namespace) -> int:
    """CPU-only strict restore of five diagnostic-only rebound endpoints."""
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )
    from scripts.train_phase6 import _current_mainrunner_source_manifest

    config = load_phase6_config(REPO_ROOT / "configs/phase6_joint_training_v1.json")
    acceptance = load_learning_acceptance_config(
        REPO_ROOT / "configs/phase6_learning_acceptance_v1.json"
    )
    roster = load_phase6_task_roster(
        REPO_ROOT / "configs/phase6_task_roster_v1.json", phase6_config=config
    )
    source_manifest = _current_mainrunner_source_manifest()
    restored_rows: list[dict[str, object]] = []
    for segment in DRIFT_SEGMENTS:
        checkpoint = _rebound_checkpoint(segment)
        binding = _diagnostic_checkpoint_binding(checkpoint)
        before = _sha256(checkpoint)
        if before != REBOUND_CHECKPOINTS[segment][0]:
            raise ValueError(f"segment {segment} checkpoint SHA256 mismatch")
        summary = _read_json(
            _original_checkpoint(segment).parent / "segment_summary.json"
        )
        iteration = summary.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise ValueError(f"segment {segment} summary iteration is invalid")
        policy = ResidualActorCritic()
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(config.payload["ppo"]["learning_rate"])
        )
        restored = restore_learning_checkpoint(
            checkpoint,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=policy,
            optimizer=optimizer,
            source_manifest=source_manifest,
            expected_iteration=iteration,
            device="cpu",
        )
        after = _sha256(checkpoint)
        if before != after:
            raise AssertionError("strict restore changed an immutable checkpoint")
        restored_rows.append(
            {
                "segment": segment,
                "path": str(checkpoint),
                "binding": binding,
                "sha256_before": before,
                "sha256_after": after,
                "iteration": restored["iteration"],
                "stage": restored["curriculum"]["stage"],
                "strict_restore": True,
            }
        )
    _atomic_json(
        args.output_dir / "strict_restore.json",
        {"checkpoints": restored_rows, "status": "ok"},
    )
    return 0


def _stage_probe_records(stage_dir: Path) -> dict[str, list[dict[str, object]]]:
    records: dict[str, list[dict[str, object]]] = {}
    for probe in PROBES:
        probe_dir = stage_dir / probe
        expected_dirs = {f"rank_{rank}" for rank in range(4)}
        if (
            not probe_dir.is_dir()
            or {path.name for path in probe_dir.iterdir()} != expected_dirs
        ):
            raise ValueError("stage probe directories are missing or duplicated")
        rows: list[dict[str, object]] = []
        for rank in range(4):
            rank_dir = probe_dir / f"rank_{rank}"
            required = {
                "command.json",
                "progress.json",
                "result.json",
                "wrapper.json",
                "worker.log",
            }
            if {path.name for path in rank_dir.iterdir()} < required:
                raise ValueError("stage probe evidence is incomplete")
            wrapper = _read_json(rank_dir / "wrapper.json")
            if (
                wrapper.get("passed") is not True
                or wrapper.get("exit_code") != 0
                or wrapper.get("signal") is not None
                or wrapper.get("timed_out") is not False
                or wrapper.get("timeout_kind") is not None
                or wrapper.get("marker_order_valid") is not True
            ):
                raise ValueError("stage probe wrapper is not a clean completion")
            result = _read_json(rank_dir / "result.json")
            if (
                result.get("status") != "ok"
                or result.get("probe") != probe
                or result.get("rank") != rank
            ):
                raise ValueError("stage probe result does not match wrapper identity")
            rows.append(result)
        if [row.get("task") for row in rows] != list(TASKS):
            raise ValueError("stage probe roster ordering differs from admitted tasks")
        records[probe] = rows
    return records


def _paired_stage_outcome(
    *, stage_dir: Path, task: str, stage: str, rank: int
) -> dict[str, object]:
    loaded: dict[str, dict[str, object]] = {}
    for mode in OUTCOME_MODES:
        path = (
            stage_dir
            / mode
            / f"rank_{rank}"
            / "outcome"
            / mode
            / "completed_episode_records.json"
        )
        loaded[mode] = _read_json(path)
        if loaded[mode].get("row_count") != 384:
            raise ValueError("paired mode has an incomplete episode quota")
    if (
        loaded["residual"].get("schedule_sha256")
        != loaded["scaffold_only"].get("schedule_sha256")
        or loaded["residual"].get("seed_sha256")
        != loaded["scaffold_only"].get("seed_sha256")
        or loaded["residual"].get("subset_counts")
        != loaded["scaffold_only"].get("subset_counts")
    ):
        raise ValueError("paired residual/scaffold schedule, seed, or quota differs")
    residual = loaded["residual"].get("records")
    scaffold = loaded["scaffold_only"].get("records")
    if not isinstance(residual, list) or not isinstance(scaffold, list):
        raise ValueError("paired episode records are invalid")
    return _paired_outcome_summary(residual, scaffold, task=task, stage=stage)


def _summarize_stage(args: argparse.Namespace) -> int:
    assert args.stage is not None
    stage_dir = args.output_dir / args.stage
    probe_records = _stage_probe_records(stage_dir)
    rows = probe_records["gradient"]
    starts = {str(row["invariants"]["runtime_rng_start"]) for row in rows}
    authority_stages = {
        str(row["authority_override"]["authority_stage"]) for row in rows
    }
    summary = {
        "stage": args.stage,
        "environment_stage": args.stage,
        "authority_stage": next(iter(authority_stages)),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
        },
        "transitions": sum(int(row["transitions"]) for row in rows),
        "task_balance": {
            str(row["task"]): int(row["transitions"]) / 8192 for row in rows
        },
        "gradient": {
            "by_task": {str(row["task"]): row["gradient"] for row in rows},
            "combined_norm": _quantiles(
                [float(row["gradient"]["combined_norm"]["mean"]) for row in rows]
            ),
            "ppo_semantic_cosine": _quantiles(
                [float(row["gradient"]["ppo_semantic_cosine"]["mean"]) for row in rows]
            ),
            "preclip_fraction": sum(
                float(row["gradient"]["preclip_fraction"]) for row in rows
            )
            / 4,
        },
        "gradient_rollout": {str(row["task"]): row["gradient_rollout"] for row in rows},
        "tasks": {
            str(row["task"]): _paired_stage_outcome(
                stage_dir=stage_dir,
                task=str(row["task"]),
                stage=args.stage,
                rank=int(row["rank"]),
            )
            for row in rows
        },
        "probe_wrappers": {probe: 4 for probe in PROBES},
        "invariants_ok": all(bool(row["invariants"]["unchanged"]) for row in rows),
        "runtime_rng_start_by_rank": [
            row["invariants"]["runtime_rng_start"] for row in rows
        ],
        "source_hashes": _source_hashes(),
    }
    if (
        summary["transitions"] != 8192
        or any(value != 0.25 for value in summary["task_balance"].values())
        or len(starts) != 4
        or len(authority_stages) != 1
    ):
        raise ValueError("stage transition balance or rank RNG evidence is invalid")
    _atomic_json(stage_dir / "stage_summary.json", summary)
    progress = _read_json(args.output_dir / "progress.json")
    completed = list(progress.get("stages", []))
    if args.stage in completed:
        raise ValueError("diagnostic progress refuses a duplicate stage")
    completed.append(args.stage)
    _atomic_json(
        args.output_dir / "progress.json",
        {"status": "running", "stages": completed},
    )
    return 0


def _summarize_authority(args: argparse.Namespace) -> int:
    """Compare the three fixed C2 conditions after all wrappers closed cleanly."""
    root = args.output_dir / "cases"
    summaries = {
        case: _read_json(root / case / "C2" / "stage_summary.json")
        for case in AUTHORITY_CASES
    }
    c1 = summaries["c2_env_c1_authority"]
    c2 = summaries["c2_env_c2_authority"]
    scaffold = summaries["scaffold_only"]
    if (
        c1["checkpoint"] != c2["checkpoint"]
        or c1["checkpoint"] != scaffold["checkpoint"]
        or c1["environment_stage"] != "C2"
        or c2["environment_stage"] != "C2"
        or scaffold["environment_stage"] != "C2"
        or c1["authority_stage"] != "C1"
        or c2["authority_stage"] != "C2"
    ):
        raise ValueError("authority comparison did not keep C2 environment fixed")
    summary = {
        "conditions": summaries,
        "comparison": _support_decision(c1, c2),
        "scaffold_only": scaffold,
        "invariants_ok": all(
            bool(item["invariants_ok"]) for item in summaries.values()
        ),
        "claim_boundary": (
            "read-only authority discriminator only; it does not modify training "
            "and does not establish Phase 6 acceptance"
        ),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    _atomic_json(
        args.output_dir / "progress.json",
        {"status": "complete", "authority_cases": list(AUTHORITY_CASES)},
    )
    return 0


def _summarize_checkpoint_drift(args: argparse.Namespace) -> int:
    """Join non-overwriting C1 nominal runs without inferring a cause."""
    root = args.output_dir / "checkpoint_drift"
    summaries = {
        str(segment): _read_json(
            root / f"segment_{segment:04d}" / "C1" / "stage_summary.json"
        )
        for segment in DRIFT_SEGMENTS
    }
    reference = summaries["47"]
    for segment, summary in summaries.items():
        if summary["environment_stage"] != "C1" or summary["authority_stage"] != "C1":
            raise ValueError(f"segment {segment} did not use fixed C1 nominal settings")
        for task in TASKS:
            if (
                summary["tasks"][task]["seed_sha256"]
                != reference["tasks"][task]["seed_sha256"]
            ):
                raise ValueError("checkpoint drift comparison changed nominal seeds")
    _atomic_json(
        root / "summary.json",
        {
            "checkpoints": summaries,
            "fixed_c1_nominal_schedule_and_seeds": True,
            "claim_boundary": "checkpoint drift localization only; no optimizer change",
        },
    )
    return 0


def _summarize_suite(args: argparse.Namespace) -> int:
    c1, c2 = (
        _read_json(args.output_dir / "C1" / "stage_summary.json"),
        _read_json(args.output_dir / "C2" / "stage_summary.json"),
    )
    if c1["runtime_rng_start_by_rank"] != c2["runtime_rng_start_by_rank"]:
        raise AssertionError("C1/C2 did not start from identical per-rank RNG inputs")
    if c1["checkpoint"] != c2["checkpoint"]:
        raise AssertionError("C1/C2 did not use the same immutable checkpoint")
    _atomic_json(
        args.output_dir / "progress.json",
        {"status": "complete", "stages": ["C1", "C2"]},
    )
    _atomic_json(
        args.output_dir / "summary.json",
        {
            "C1": c1,
            "C2": c2,
            "comparison": _support_decision(c1, c2),
            "paired_rng_inputs_equal": True,
            "claim_boundary": (
                "bounded diagnostic only; does not prove root cause or "
                "Phase 6 learning acceptance"
            ),
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if "--mode" not in values:
        raise SystemExit("--mode is required")
    mode = values[values.index("--mode") + 1]
    if mode == "worker":
        return _worker_main(values)
    if mode == "rank-wrapper":
        return _run_rank_wrapper(_plain_parser(mode).parse_args(values))
    if mode == "plan":
        return _run_plan(_plain_parser(mode).parse_args(values))
    if mode == "strict-restore":
        return _strict_restore_catalog(_plain_parser(mode).parse_args(values))
    if mode == "summarize-stage":
        return _summarize_stage(_plain_parser(mode).parse_args(values))
    if mode == "summarize-authority":
        return _summarize_authority(_plain_parser(mode).parse_args(values))
    if mode == "summarize-checkpoint-drift":
        return _summarize_checkpoint_drift(_plain_parser(mode).parse_args(values))
    if mode == "summarize-suite":
        return _summarize_suite(_plain_parser(mode).parse_args(values))
    raise SystemExit(f"unsupported mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
