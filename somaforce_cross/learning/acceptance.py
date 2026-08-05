"""Pure-PyTorch contracts and metrics for the Phase 5 learning acceptance."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from somaforce_cross.learning.losses import semantic_losses, split_semantic_target


ACCEPTANCE_CONTRACT_VERSION = "phase5_learning_acceptance_v4"
PHASE4_CANONICAL_SHA256 = (
    "214b5328f0705b467f0fe305ec5eec78dc91f3ced163cf1d410ea00d43889ab3"
)
PHASE5_RAW_SHA256 = "ac3ca17ef7b0f5cd800c2c967c7242eec94a2fdaf59eb8ba465e8f3697d9f200"
PHASE5_CANONICAL_SHA256 = (
    "e95eca5f3c1332965967555f81cd33a78f9144280d0aa446e5eefaf5b83ae1f8"
)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the contract's stable JSON encoding."""
    if not isinstance(payload, Mapping):
        raise TypeError("learning acceptance contract must be an object")
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("learning acceptance contract is not canonical JSON") from exc


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _exact_keys(value: object, path: str, expected: set[str]) -> Mapping[str, Any]:
    mapping = _mapping(value, path)
    if set(mapping) != expected:
        raise ValueError(
            f"{path} keys differ: missing={sorted(expected - set(mapping))} "
            f"unknown={sorted(set(mapping) - expected)}"
        )
    return mapping


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _finite(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def validate_learning_acceptance_config(payload: Mapping[str, Any]) -> None:
    """Reject any attempt to alter the bounded single-task acceptance contract."""
    root = _exact_keys(
        payload,
        "root",
        {
            "bindings",
            "checkpoints",
            "contract_version",
            "evaluation",
            "probe",
            "run",
            "thresholds",
        },
    )
    if root["contract_version"] != ACCEPTANCE_CONTRACT_VERSION:
        raise ValueError("unexpected learning acceptance contract version")
    bindings = _exact_keys(
        root["bindings"],
        "bindings",
        {"phase4_canonical_sha256", "phase5_canonical_sha256", "phase5_raw_sha256"},
    )
    if bindings != {
        "phase4_canonical_sha256": PHASE4_CANONICAL_SHA256,
        "phase5_canonical_sha256": PHASE5_CANONICAL_SHA256,
        "phase5_raw_sha256": PHASE5_RAW_SHA256,
    }:
        raise ValueError("learning acceptance must bind the frozen Phase 4/5 contracts")
    run = _exact_keys(
        root["run"],
        "run",
        {
            "episode_horizon",
            "evaluation_seed",
            "iterations",
            "num_envs",
            "num_steps_per_env",
            "semantic_probe_seed",
            "stage",
            "task",
            "train_seed",
            "transitions",
        },
    )
    expected_run = {
        "episode_horizon": 508,
        "evaluation_seed": 20262801,
        "iterations": 2442,
        "num_envs": 64,
        "num_steps_per_env": 32,
        "semantic_probe_seed": 20261801,
        "stage": "C1",
        "task": "push_door_hand",
        "train_seed": 20260801,
        "transitions": 5001216,
    }
    if run != expected_run:
        raise ValueError("learning acceptance run parameters are frozen")
    if (
        run["iterations"] * run["num_envs"] * run["num_steps_per_env"]
        != run["transitions"]
    ):
        raise ValueError("learning acceptance transition arithmetic is inconsistent")
    checkpoints = _exact_keys(
        root["checkpoints"],
        "checkpoints",
        {"interval_transitions", "strict_same_source_restore"},
    )
    if checkpoints != {
        "interval_transitions": 250000,
        "strict_same_source_restore": True,
    }:
        raise ValueError("learning acceptance checkpoint policy is frozen")
    probe = _exact_keys(
        root["probe"], "probe", {"loss_history_windows", "samples", "zero_residual"}
    )
    if probe != {"loss_history_windows": 3, "samples": 4096, "zero_residual": True}:
        raise ValueError("learning acceptance semantic probe is frozen")
    evaluation = _exact_keys(
        root["evaluation"],
        "evaluation",
        {"c1_episodes", "deterministic_actor_mean", "nominal_paired_episodes"},
    )
    if evaluation != {
        "c1_episodes": 256,
        "deterministic_actor_mean": True,
        "nominal_paired_episodes": 128,
    }:
        raise ValueError("learning acceptance evaluation is frozen")
    thresholds = _mapping(root["thresholds"], "thresholds")
    expected_thresholds = {
        "aux_gradient_min",
        "contact_bearing_rms_gradient_ratio_max",
        "contact_bearing_saturation_window_max",
        "contact_bearing_training_transitions_min",
        "evaluation_episode_invalid_max",
        "evaluation_failure_max",
        "evaluation_success_min",
        "nominal_retention_ratio_min",
        "prediction_entropy_target_ratio_min",
        "prediction_max_probability_allowance",
        "prediction_max_probability_cap",
        "residual_contact_fraction_min",
        "residual_contact_norm_min",
        "residual_mean_norm_min",
        "semantic_loss_final_to_initial_max",
        "target_aggregate_entropy_min",
    }
    if set(thresholds) != expected_thresholds:
        raise ValueError("learning acceptance thresholds are incomplete")
    for key, value in thresholds.items():
        _finite(value, f"thresholds.{key}")


@dataclass(frozen=True)
class LearningAcceptanceConfig:
    """Validated contract with separate raw and canonical identities."""

    payload: Mapping[str, Any]
    raw_sha256: str
    canonical_sha256: str


def load_learning_acceptance_config(path: str | Path) -> LearningAcceptanceConfig:
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load learning acceptance config {source}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("learning acceptance config root must be an object")
    validate_learning_acceptance_config(payload)
    return LearningAcceptanceConfig(
        payload=payload,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=canonical_sha256(payload),
    )


def source_sha256(paths: Sequence[str | Path]) -> dict[str, str]:
    """Hash a deterministic, explicit source allowlist for strict resume checks."""
    result: dict[str, str] = {}
    for item in sorted(str(Path(path)) for path in paths):
        source = Path(item)
        if not source.is_file():
            raise FileNotFoundError(f"source allowlist path is missing: {source}")
        result[item] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


def source_allowlist_sha256(values: Mapping[str, str]) -> str:
    if not values or any(len(value) != 64 for value in values.values()):
        raise ValueError("source hashes must be a nonempty SHA256 mapping")
    return hashlib.sha256(canonical_json_bytes(values)).hexdigest()


class FixedSemanticProbe:
    """Retain exactly the fixed number of contact-bearing semantic samples."""

    def __init__(self, samples: int) -> None:
        self.samples = _positive_int(samples, "samples")
        self._policy: list[torch.Tensor] = []
        self._target: list[torch.Tensor] = []
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def add(self, policy: torch.Tensor, semantic_target: torch.Tensor) -> int:
        if (
            policy.ndim != 2
            or policy.shape[1] != 668
            or not torch.isfinite(policy).all()
        ):
            raise ValueError("probe policy must be finite [B, 668]")
        fields = split_semantic_target(semantic_target)
        if policy.shape[0] != semantic_target.shape[0]:
            raise ValueError("probe policy and target batches must match")
        selected = fields.semantic_loss_weight.squeeze(1) > 0.0
        remaining = self.samples - self._count
        if remaining <= 0 or not selected.any():
            return 0
        indices = selected.nonzero(as_tuple=False).flatten()[:remaining]
        self._policy.append(policy.index_select(0, indices).detach().cpu().clone())
        self._target.append(
            semantic_target.index_select(0, indices).detach().cpu().clone()
        )
        accepted = int(indices.numel())
        self._count += accepted
        return accepted

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._count != self.samples:
            raise ValueError(
                f"semantic probe requires exactly {self.samples} samples, found {self._count}"
            )
        policy = torch.cat(self._policy, dim=0)
        target = torch.cat(self._target, dim=0)
        if policy.shape != (self.samples, 668) or target.shape != (self.samples, 31):
            raise AssertionError("fixed probe accumulation changed its tensor contract")
        if not torch.all(split_semantic_target(target).semantic_loss_weight > 0.0):
            raise AssertionError("fixed probe retained a zero-weight sample")
        return policy, target


def _entropy(distribution: torch.Tensor) -> float:
    aggregate = distribution.mean(dim=0)
    value = -(aggregate * aggregate.clamp_min(1.0e-8).log()).sum()
    result = float(value.item())
    if not math.isfinite(result):
        raise FloatingPointError("non-finite aggregate entropy")
    return result


def semantic_probe_metrics(
    p_dir_pred: torch.Tensor, p_mag_pred: torch.Tensor, semantic_target: torch.Tensor
) -> dict[str, Any]:
    """Summarize loss and class-distribution gates on the unchanged fixed probe."""
    fields = split_semantic_target(semantic_target)
    if not torch.all(fields.semantic_loss_weight > 0.0):
        raise ValueError("semantic probe requires only positive-weight samples")
    output = semantic_losses(p_dir_pred, p_mag_pred, semantic_target)
    values = {
        "semantic_dir_loss": float(output.dir_loss.detach().item()),
        "semantic_mag_loss": float(output.mag_loss.detach().item()),
        "semantic_total": float(output.total.detach().item()),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite semantic probe loss")
    target_entropy = 0.5 * (
        _entropy(fields.p_dir_target) + _entropy(fields.p_mag_target)
    )
    prediction_entropy = 0.5 * (_entropy(p_dir_pred) + _entropy(p_mag_pred))
    target_max_probability = max(
        float(fields.p_dir_target.mean(dim=0).max().item()),
        float(fields.p_mag_target.mean(dim=0).max().item()),
    )
    prediction_max_probability = max(
        float(p_dir_pred.mean(dim=0).max().item()),
        float(p_mag_pred.mean(dim=0).max().item()),
    )
    return {
        "samples": int(semantic_target.shape[0]),
        **values,
        "target_aggregate_entropy": target_entropy,
        "prediction_aggregate_entropy": prediction_entropy,
        "target_max_aggregate_class_probability": target_max_probability,
        "prediction_max_aggregate_class_probability": prediction_max_probability,
    }


def gradient_summary(records: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    """Aggregate RMS acceptance metrics and the diagnostic-only p95 ratio."""
    if not records:
        raise ValueError("gradient diagnostics are empty")
    pairs: list[tuple[float, float]] = []
    ratios: list[float] = []
    for index, record in enumerate(records):
        ppo = _finite(record.get("ppo_grad_norm"), f"gradient[{index}].ppo_grad_norm")
        auxiliary = _finite(
            record.get("aux_grad_norm"), f"gradient[{index}].aux_grad_norm"
        )
        if ppo < 0.0 or auxiliary < 0.0:
            raise ValueError("gradient norms must be nonnegative")
        pairs.append((ppo, auxiliary))
        if ppo > 0.0:
            ratios.append(auxiliary / ppo)
    if not pairs or not ratios:
        raise ValueError(
            "contact-bearing gradient diagnostics require nonzero PPO gradient"
        )
    ppo_rms = math.sqrt(sum(ppo * ppo for ppo, _ in pairs) / len(pairs))
    auxiliary_rms = math.sqrt(sum(aux * aux for _, aux in pairs) / len(pairs))
    ordered = sorted(ratios)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    if ppo_rms <= 0.0 or auxiliary_rms <= 0.0:
        raise ValueError("PPO and auxiliary gradient RMS must both be nonzero")
    return {
        "contact_bearing_minibatches": len(pairs),
        "ppo_gradient_rms": ppo_rms,
        "auxiliary_gradient_rms": auxiliary_rms,
        "rms_aux_to_ppo": auxiliary_rms / ppo_rms,
        "ratio_p95": p95,
    }


def validate_learning_gates(
    metrics: Mapping[str, Any], config: LearningAcceptanceConfig
) -> dict[str, bool]:
    """Validate final learning metrics and return named pass bits or raise."""
    thresholds = config.payload["thresholds"]
    initial = _mapping(metrics["initial_probe"], "initial_probe")
    recent = metrics["recent_probe"]
    if (
        not isinstance(recent, Sequence)
        or len(recent) != config.payload["probe"]["loss_history_windows"]
    ):
        raise ValueError(
            "recent probe must contain exactly the final three evaluations"
        )
    for name in ("semantic_dir_loss", "semantic_mag_loss", "semantic_total"):
        baseline = _finite(initial.get(name), f"initial_probe.{name}")
        if baseline <= 0.0:
            raise ValueError(f"initial_probe.{name} must be positive")
        median = sorted(
            _finite(item.get(name), f"recent_probe.{name}") for item in recent
        )[1]
        if median > baseline * float(thresholds["semantic_loss_final_to_initial_max"]):
            raise AssertionError(f"semantic learning gate failed for {name}")
    final_probe = _mapping(metrics["final_probe"], "final_probe")
    if _finite(
        metrics["training_contact_transitions"], "training_contact_transitions"
    ) < float(thresholds["contact_bearing_training_transitions_min"]):
        raise AssertionError("insufficient contact-bearing training transitions")
    gradients = _mapping(metrics["gradients"], "gradients")
    if _finite(
        gradients.get("auxiliary_gradient_rms"), "gradients.auxiliary_gradient_rms"
    ) <= float(thresholds["aux_gradient_min"]):
        raise AssertionError("auxiliary semantic gradient is zero")
    if _finite(gradients.get("ppo_gradient_rms"), "gradients.ppo_gradient_rms") <= 0.0:
        raise AssertionError("PPO semantic-pipeline gradient is zero")
    if _finite(gradients.get("rms_aux_to_ppo"), "gradients.rms_aux_to_ppo") > float(
        thresholds["contact_bearing_rms_gradient_ratio_max"]
    ):
        raise AssertionError("RMS auxiliary/PPO gradient ratio exceeds the contract")
    # RMS is the acceptance metric; p95 remains a finite diagnostic for reports.
    _finite(gradients.get("ratio_p95"), "gradients.ratio_p95")
    target_entropy = _finite(
        final_probe.get("target_aggregate_entropy"),
        "final_probe.target_aggregate_entropy",
    )
    prediction_entropy = _finite(
        final_probe.get("prediction_aggregate_entropy"),
        "final_probe.prediction_aggregate_entropy",
    )
    if target_entropy <= float(thresholds["target_aggregate_entropy_min"]):
        raise AssertionError("semantic target aggregate entropy is too small")
    if prediction_entropy < target_entropy * float(
        thresholds["prediction_entropy_target_ratio_min"]
    ):
        raise AssertionError("semantic prediction aggregate entropy collapsed")
    target_max = _finite(
        final_probe.get("target_max_aggregate_class_probability"),
        "final_probe.target_max_aggregate_class_probability",
    )
    prediction_max = _finite(
        final_probe.get("prediction_max_aggregate_class_probability"),
        "final_probe.prediction_max_aggregate_class_probability",
    )
    max_allowed = max(
        float(thresholds["prediction_max_probability_cap"]),
        target_max + float(thresholds["prediction_max_probability_allowance"]),
    )
    if prediction_max > max_allowed:
        raise AssertionError("semantic prediction maximum class probability collapsed")
    return {
        "contact_transitions": True,
        "semantic_loss_medians": True,
        "gradient_norms": True,
        "gradient_ratios": True,
        "semantic_entropy": True,
    }


def validate_final_evaluation(
    metrics: Mapping[str, Any], config: LearningAcceptanceConfig
) -> dict[str, bool]:
    """Validate deterministic C1, nominal retention, action, and residual gates."""
    thresholds = config.payload["thresholds"]
    outcome = _mapping(metrics["outcome"], "outcome")
    if _finite(outcome.get("success"), "outcome.success") < float(
        thresholds["evaluation_success_min"]
    ):
        raise AssertionError("C1 deterministic success gate failed")
    if _finite(outcome.get("failure"), "outcome.failure") > float(
        thresholds["evaluation_failure_max"]
    ):
        raise AssertionError("C1 deterministic failure gate failed")
    if int(outcome.get("episode_invalid", -1)) > int(
        thresholds["evaluation_episode_invalid_max"]
    ):
        raise AssertionError("C1 deterministic evaluation has invalid episodes")
    retention = _mapping(metrics["retention"], "retention")
    _finite(retention.get("median_progress"), "retention.median_progress")
    _finite(
        retention.get("scaffold_only_median_progress"),
        "retention.scaffold_only_median_progress",
    )
    if _finite(
        retention.get("trained_to_scaffold_ratio"),
        "retention.trained_to_scaffold_ratio",
    ) < float(thresholds["nominal_retention_ratio_min"]):
        raise AssertionError("nominal retention ratio gate failed")
    action = _mapping(metrics["action"], "action")
    saturation_fraction = _finite(
        action.get("contact_saturation_fraction"), "action.contact_saturation_fraction"
    )
    if not 0.0 <= saturation_fraction <= 1.0:
        raise AssertionError(
            "contact-bearing action saturation fraction must be in [0, 1]"
        )
    if _finite(
        action.get("contact_saturation_window_max"),
        "action.contact_saturation_window_max",
    ) > float(thresholds["contact_bearing_saturation_window_max"]):
        raise AssertionError("contact-bearing action saturation window gate failed")
    residual = _mapping(metrics["residual"], "residual")
    if _finite(
        residual.get("contact_mean_delta_safe_norm"),
        "residual.contact_mean_delta_safe_norm",
    ) < float(thresholds["residual_mean_norm_min"]):
        raise AssertionError("contact-bearing residual mean norm gate failed")
    if _finite(
        residual.get("contact_norm_fraction_ge_min"),
        "residual.contact_norm_fraction_ge_min",
    ) < float(thresholds["residual_contact_fraction_min"]):
        raise AssertionError(
            "contact-bearing residual non-collapse fraction gate failed"
        )
    return {
        "c1_outcome": True,
        "nominal_retention": True,
        "action_saturation": True,
        "residual_noncollapse": True,
    }
