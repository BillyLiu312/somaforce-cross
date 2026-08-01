"""Canonical Phase 5 PPO contract loading without runtime dependencies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PHASE5_CONTRACT_VERSION = "phase5_ppo_v1"
PHASE4_CONTRACT_VERSION = "phase4b5_numeric_v2"
PHASE4_CANONICAL_SHA256 = (
    "214b5328f0705b467f0fe305ec5eec78dc91f3ced163cf1d410ea00d43889ab3"
)
PHASE4_SOURCE_COMMIT = "e578c4f8bb617fbbf8a0021aa6dd0ada9fbd5329"


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode the Phase 5 contract with its fixed canonical JSON encoding."""
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 5 contract must be a JSON object")
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Phase 5 contract is not canonical JSON") from exc


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _exact_keys(value: object, path: str, expected: set[str]) -> Mapping[str, Any]:
    mapping = _mapping(value, path)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{path} keys differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )
    return mapping


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _positive(value: object, path: str) -> float:
    result = _finite_number(value, path)
    if result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _dimensions(value: object, path: str, expected: tuple[int, ...]) -> None:
    if not isinstance(value, list) or tuple(value) != expected:
        raise ValueError(f"{path} must be {list(expected)}")


def validate_phase5_config(payload: Mapping[str, Any]) -> None:
    """Validate every frozen Phase 5 field before any runner can use it."""
    root = _exact_keys(
        payload,
        "root",
        {
            "contract_version",
            "network",
            "observation",
            "phase4",
            "ppo",
            "semantic_loss",
        },
    )
    if root["contract_version"] != PHASE5_CONTRACT_VERSION:
        raise ValueError("unexpected Phase 5 contract version")

    phase4 = _exact_keys(
        root["phase4"],
        "phase4",
        {"canonical_sha256", "contract_version", "source_commit"},
    )
    if (
        phase4["contract_version"] != PHASE4_CONTRACT_VERSION
        or phase4["canonical_sha256"] != PHASE4_CANONICAL_SHA256
        or phase4["source_commit"] != PHASE4_SOURCE_COMMIT
    ):
        raise ValueError("Phase 5 must bind the approved Phase 4 source contract")

    network = _exact_keys(
        root["network"],
        "network",
        {
            "action_dim",
            "actor",
            "actor_input_dim",
            "critic",
            "critic_input_dim",
            "policy_dim",
            "semantic_pipeline",
        },
    )
    for key, expected in (
        ("action_dim", 23),
        ("actor_input_dim", 220),
        ("critic_input_dim", 845),
        ("policy_dim", 668),
    ):
        if network[key] != expected:
            raise ValueError(f"network.{key} must be {expected}")
    _dimensions(network["actor"], "network.actor", (220, 256, 256, 23))
    _dimensions(network["critic"], "network.critic", (845, 512, 256, 256, 1))
    semantic_pipeline = _exact_keys(
        network["semantic_pipeline"],
        "network.semantic_pipeline",
        {"cross_encoder", "p_dir_dim", "p_mag_dim", "wrist_tokens"},
    )
    _dimensions(
        semantic_pipeline["cross_encoder"],
        "network.semantic_pipeline.cross_encoder",
        (65, 128, 64),
    )
    _dimensions(
        semantic_pipeline["wrist_tokens"],
        "network.semantic_pipeline.wrist_tokens",
        (2, 16, 14),
    )
    if semantic_pipeline["p_dir_dim"] != 13 or semantic_pipeline["p_mag_dim"] != 5:
        raise ValueError("semantic pipeline dimensions are frozen")

    observation = _exact_keys(
        root["observation"],
        "observation",
        {
            "actor_suffix",
            "critic_dim",
            "environment_z_cross_diagnostic",
            "obs_groups",
            "policy_dim",
            "semantic_target_dim",
            "wrist_token_prefix",
        },
    )
    for key, expected in (
        ("policy_dim", 668),
        ("critic_dim", 845),
        ("semantic_target_dim", 31),
    ):
        if observation[key] != expected:
            raise ValueError(f"observation.{key} must be {expected}")
    _dimensions(
        observation["wrist_token_prefix"], "observation.wrist_token_prefix", (0, 448)
    )
    _dimensions(observation["actor_suffix"], "observation.actor_suffix", (448, 604))
    _dimensions(
        observation["environment_z_cross_diagnostic"],
        "observation.environment_z_cross_diagnostic",
        (604, 668),
    )
    if observation["obs_groups"] != {"policy": ["policy"], "critic": ["critic"]}:
        raise ValueError("Phase 5 observation groups are frozen")

    ppo = _exact_keys(
        root["ppo"],
        "ppo",
        {
            "clip_param",
            "desired_kl",
            "entropy_coef",
            "gamma",
            "init_noise_std",
            "lam",
            "learning_rate",
            "max_grad_norm",
            "noise_std_type",
            "normalize_advantage_per_mini_batch",
            "num_learning_epochs",
            "num_mini_batches",
            "num_steps_per_env",
            "ppo_epochs",
            "schedule",
            "use_clipped_value_loss",
            "value_loss_coef",
        },
    )
    expected_numbers = {
        "clip_param": 0.2,
        "entropy_coef": 0.001,
        "gamma": 0.99,
        "init_noise_std": 1.0,
        "lam": 0.95,
        "learning_rate": 3.0e-4,
        "max_grad_norm": 1.0,
        "value_loss_coef": 1.0,
    }
    for key, expected in expected_numbers.items():
        if _finite_number(ppo[key], f"ppo.{key}") != expected:
            raise ValueError(f"ppo.{key} must be {expected}")
    for key, expected in (
        ("num_steps_per_env", 32),
        ("num_learning_epochs", 3),
        ("ppo_epochs", 3),
        ("num_mini_batches", 8),
    ):
        if _integer(ppo[key], f"ppo.{key}") != expected:
            raise ValueError(f"ppo.{key} must be {expected}")
    if (
        ppo["schedule"] != "fixed"
        or ppo["desired_kl"] is not None
        or ppo["noise_std_type"] != "log"
        or ppo["normalize_advantage_per_mini_batch"] is not False
        or ppo["use_clipped_value_loss"] is not True
    ):
        raise ValueError(
            "Phase 5 PPO scheduler, normalization, or noise contract is invalid"
        )

    semantic = _exact_keys(
        root["semantic_loss"],
        "semantic_loss",
        {
            "clean_wrench_dim",
            "eps",
            "lambda_cross",
            "lambda_dir",
            "lambda_mag",
            "p_dir_dim",
            "p_mag_dim",
            "reduction",
            "semantic_loss_weight_dim",
        },
    )
    for key, expected in (
        ("clean_wrench_dim", 12),
        ("p_dir_dim", 13),
        ("p_mag_dim", 5),
        ("semantic_loss_weight_dim", 1),
    ):
        if semantic[key] != expected:
            raise ValueError(f"semantic_loss.{key} must be {expected}")
    if (
        _positive(semantic["eps"], "semantic_loss.eps") != 1.0e-8
        or _finite_number(semantic["lambda_dir"], "semantic_loss.lambda_dir") != 0.05
        or _finite_number(semantic["lambda_mag"], "semantic_loss.lambda_mag") != 0.05
        or _finite_number(semantic["lambda_cross"], "semantic_loss.lambda_cross") != 0.0
        or semantic["reduction"]
        != "sum(weight * per_sample_kl) / clamp(sum(weight), min=1)"
    ):
        raise ValueError("Phase 5 semantic-loss contract is invalid")


@dataclass(frozen=True)
class Phase5Config:
    """Validated Phase 5 JSON with separate raw and canonical identities."""

    payload: Mapping[str, Any]
    raw_sha256: str
    canonical_sha256: str


def load_phase5_config(path: str | Path) -> Phase5Config:
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load Phase 5 config {source}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 5 contract root must be an object")
    validate_phase5_config(payload)
    return Phase5Config(
        payload=payload,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=canonical_sha256(payload),
    )
