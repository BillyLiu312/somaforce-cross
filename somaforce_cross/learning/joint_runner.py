"""Pure-PyTorch contracts and coordination primitives for Phase 6 joint PPO."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as distributed
from torch import nn

from somaforce_cross.learning.actor_critic import (
    ACTION_DIM,
    ACTOR_INPUT_DIM,
    CRITIC_DIM,
    POLICY_DIM,
    ResidualActorCritic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE6_CONTRACT_VERSION = "phase6_joint_training_v1"
PHASE6_ROSTER_VERSION = "phase6_task_roster_v1"
PHASE6_CHECKPOINT_VERSION = "phase6_joint_checkpoint_v1"
PHASE6_LEARNING_ACCEPTANCE_VERSION = "phase6_learning_acceptance_v1"
PHASE6_LEARNING_CHECKPOINT_VERSION = "phase6_learning_checkpoint_v1"
PHASE6_LEARNING_CONFIG = REPO_ROOT / "configs/phase6_learning_acceptance_v1.json"
_LEARNING_PHASE6_RAW_SHA256 = (
    "f26f8e118decb13066362b7a8612b1e2c5fbba37050a38b7ab3880f796c2131d"
)
_LEARNING_PHASE6_CANONICAL_SHA256 = (
    "8e733479baada2334636106ee352aeb8ab8bf8dc7ad4ce3b39b844019c74df42"
)
_LEARNING_ROSTER_RAW_SHA256 = (
    "8ddcdd05f423138eb855d9c7cb3478ffb373f718484f0a948512f9bce1492fdb"
)
_LEARNING_ROSTER_CANONICAL_SHA256 = (
    "14dfe15db96daa6c8d0bac93965fcdca3d9e815a1ac69a7adaa44d9f0db45bc4"
)
_LEARNING_PHASE5_CHECKPOINT_SHA256 = (
    "46d76e722e0422bfa1f00e5d815d80f4a328b0cf169f7104a16c6fbdeebf723a"
)

_PHASE4_BINDING = {
    "canonical_sha256": "214b5328f0705b467f0fe305ec5eec78dc91f3ced163cf1d410ea00d43889ab3",
    "contract_version": "phase4b5_numeric_v2",
    "raw_sha256": "689e02f9eb3fae1959abb001f7917e91908980377299275c843f04e17b0f930f",
}
_PHASE5_BINDING = {
    "canonical_sha256": "e95eca5f3c1332965967555f81cd33a78f9144280d0aa446e5eefaf5b83ae1f8",
    "contract_version": "phase5_ppo_v1",
    "raw_sha256": "ac3ca17ef7b0f5cd800c2c967c7242eec94a2fdaf59eb8ba465e8f3697d9f200",
}
_PHASE5_ACCEPTANCE_BINDING = {
    "canonical_sha256": "28907947392ba5cc8e663c7219c13b3043a4c93ba6563fcdc49e88f12531dc50",
    "contract_version": "phase5_learning_acceptance_v4",
    "raw_sha256": "cf2d4a2d78babb632775fe58a41bce83421d50b4a30c57adeb9cc5170c1f80fb",
}
_INITIAL_CHECKPOINT = {
    "path": (
        "outputs/phase5_learning_acceptance/phase5_learning_v3_20260805/"
        "learning_64env_2442iter/checkpoints/"
        "iteration_0002442_transitions_5001216.pt"
    ),
    "sha256": "46d76e722e0422bfa1f00e5d815d80f4a328b0cf169f7104a16c6fbdeebf723a",
}
_PPO = {
    "clip_param": 0.2,
    "desired_kl": None,
    "entropy_coef": 0.001,
    "gamma": 0.99,
    "init_noise_std": 1.0,
    "lam": 0.95,
    "learning_rate": 0.0003,
    "max_grad_norm": 1.0,
    "noise_std_type": "log",
    "normalize_advantage_per_mini_batch": False,
    "num_learning_epochs": 3,
    "num_mini_batches": 8,
    "num_steps_per_env": 32,
    "ppo_epochs": 3,
    "schedule": "fixed",
    "use_clipped_value_loss": True,
    "value_loss_coef": 1.0,
}
_METRICS = {
    "episode_metric_names": [
        "return",
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
    ],
    "force_norm_formula": "mean(norm(observed_wrench_physical[:, :, :3], dim=-1), dim=-1)",
    "force_rate_excludes_first_control_step": True,
    "force_rate_formula": "abs(force_norm[t]-force_norm[t-1])/0.02",
    "impulse_dt_s": 0.02,
    "p_cross_shape": [13, 5],
    "p_cross_trace_reduction": "env_mean_per_control_step",
    "ppo_metric_names": [
        "ppo_loss",
        "value_loss",
        "surrogate_loss",
        "entropy",
        "semantic_dir_loss",
        "semantic_mag_loss",
        "semantic_total",
        "semantic_entropy",
        "semantic_pipeline_grad_norm",
    ],
    "transition_metric_names": [
        "reward",
        "ppo_loss",
        "value_loss",
        "surrogate_loss",
        "entropy",
        "semantic_dir_loss",
        "semantic_mag_loss",
        "semantic_total",
        "semantic_entropy",
        "semantic_pipeline_grad_norm",
        "force_p95_N",
        "force_p99_N",
        "force_impulse_Ns_mean",
        "force_rate_p95_N_per_s",
        "contact_fraction",
        "stability_margin_mean_m",
        "residual_arms_mean_norm",
        "residual_waist_mean_norm",
        "residual_legs_mean_norm",
        "saturation_fraction",
        "p_cross_entropy_mean",
    ],
}
_CURRENT_TASKS = (
    "push_door_hand",
    "push_box",
    "move_suitcase",
    "move_largebox",
)
_TASK_IDENTITIES = {
    "push_door_hand": {
        "adapter_identity": (
            "somaforce_cross.envs.task_adapters.push_door_hand.PushDoorHandTaskAdapter"
        ),
        "mismatch_owner": "Phase4B5MismatchSampler:push_door_hand",
        "retention_owner": "phase4b5_numeric_v2.retention.push_door_hand",
        "task_spec_identity": "hdmi_push_door_hand_teacher_v1",
    },
    "push_box": {
        "adapter_identity": (
            "somaforce_cross.envs.task_adapters.push_box.PushBoxTaskAdapter"
        ),
        "mismatch_owner": "Phase4B5MismatchSampler:push_box",
        "retention_owner": "phase4b5_numeric_v2.retention.push_box",
        "task_spec_identity": "hdmi_push_box_teacher_v1",
    },
    "move_suitcase": {
        "adapter_identity": (
            "somaforce_cross.envs.task_adapters.move_payload.MovePayloadTaskAdapter"
        ),
        "mismatch_owner": "Phase4B5MismatchSampler:move_suitcase",
        "retention_owner": "phase4b5_numeric_v2.retention.move_suitcase",
        "task_spec_identity": "hdmi_move_suitcase_teacher_v1",
    },
    "move_largebox": {
        "adapter_identity": (
            "somaforce_cross.envs.task_adapters.move_payload.MovePayloadTaskAdapter"
        ),
        "mismatch_owner": "Phase4B5MismatchSampler:move_largebox",
        "retention_owner": "phase4b5_numeric_v2.retention.move_largebox",
        "task_spec_identity": "hdmi_move_largebox_teacher_v1",
    },
}


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode a Phase 6 JSON object without accepting non-finite values."""
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 6 JSON payload must be an object")
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Phase 6 JSON payload is not canonical") from exc


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _exact_keys(value: object, path: str, expected: set[str]) -> Mapping[str, Any]:
    result = _mapping(value, path)
    actual = set(result)
    if actual != expected:
        raise ValueError(
            f"{path} keys differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )
    return result


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a nonnegative integer")
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{path} must be a SHA256 digest") from exc
    return value


def _finite(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _validate_bindings(value: object) -> None:
    bindings = _exact_keys(
        value,
        "bindings",
        {
            "phase4",
            "phase5",
            "phase5_initial_checkpoint",
            "phase5_learning_acceptance",
            "roster_contract_version",
        },
    )
    for name, expected in (
        ("phase4", _PHASE4_BINDING),
        ("phase5", _PHASE5_BINDING),
        ("phase5_learning_acceptance", _PHASE5_ACCEPTANCE_BINDING),
    ):
        bound = _exact_keys(bindings[name], f"bindings.{name}", set(expected))
        if dict(bound) != expected:
            raise ValueError(f"bindings.{name} does not match the admitted contract")
    checkpoint = _exact_keys(
        bindings["phase5_initial_checkpoint"],
        "bindings.phase5_initial_checkpoint",
        {"path", "sha256"},
    )
    if dict(checkpoint) != _INITIAL_CHECKPOINT:
        raise ValueError("Phase 6 must start from the admitted Phase 5 checkpoint")
    if bindings["roster_contract_version"] != PHASE6_ROSTER_VERSION:
        raise ValueError("Phase 6 requires the admitted roster schema version")


def validate_phase6_config(payload: Mapping[str, Any]) -> None:
    """Validate every frozen Phase 6 algorithm field before runtime startup."""
    canonical_json_bytes(payload)
    root = _exact_keys(
        payload,
        "root",
        {
            "bindings",
            "checkpoint",
            "contract_version",
            "curriculum",
            "dimensions",
            "metrics",
            "ppo",
            "runtime",
            "scheduler",
        },
    )
    if root["contract_version"] != PHASE6_CONTRACT_VERSION:
        raise ValueError("unexpected Phase 6 contract version")
    _validate_bindings(root["bindings"])
    dimensions = _exact_keys(
        root["dimensions"],
        "dimensions",
        {"action", "actor_input", "critic", "policy", "semantic_target"},
    )
    if dict(dimensions) != {
        "action": ACTION_DIM,
        "actor_input": ACTOR_INPUT_DIM,
        "critic": CRITIC_DIM,
        "policy": POLICY_DIM,
        "semantic_target": 31,
    }:
        raise ValueError("Phase 6 must preserve all Phase 5 tensor dimensions")
    metrics = _exact_keys(root["metrics"], "metrics", set(_METRICS))
    if dict(metrics) != _METRICS:
        raise ValueError("Phase 6 metrics schema is frozen")
    ppo = _exact_keys(root["ppo"], "ppo", set(_PPO))
    for name, value in ppo.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            _finite(value, f"ppo.{name}")
    if dict(ppo) != _PPO:
        raise ValueError("Phase 6 may not change the Phase 5 PPO contract")
    runtime = _exact_keys(
        root["runtime"],
        "runtime",
        {
            "backend",
            "init_method",
            "process_group_timeout_s",
            "required_visible_gpus",
            "world_size",
        },
    )
    if dict(runtime) != {
        "backend": "nccl",
        "init_method": "env://",
        "process_group_timeout_s": 180,
        "required_visible_gpus": 4,
        "world_size": 4,
    }:
        raise ValueError("Phase 6 production runtime must be four-rank NCCL env://")
    checkpoint = _exact_keys(
        root["checkpoint"],
        "checkpoint",
        {
            "atomic_writer_rank",
            "checkpoint_version",
            "persist_rank_rng_states",
            "resume_strict",
            "verify_hash_each_iteration",
        },
    )
    if dict(checkpoint) != {
        "atomic_writer_rank": 0,
        "checkpoint_version": PHASE6_CHECKPOINT_VERSION,
        "persist_rank_rng_states": True,
        "resume_strict": True,
        "verify_hash_each_iteration": True,
    }:
        raise ValueError("Phase 6 checkpoint policy is frozen")
    curriculum = _exact_keys(
        root["curriculum"],
        "curriculum",
        {
            "evaluation_interval_transitions",
            "nominal_atom",
            "nominal_episodes_per_task",
            "promotion_windows",
            "rollback_windows",
            "stage_episodes_per_task",
        },
    )
    atom = _exact_keys(
        curriculum["nominal_atom"], "curriculum.nominal_atom", {"C1", "C2", "C3"}
    )
    if dict(curriculum) != {
        "evaluation_interval_transitions": 250000,
        "nominal_atom": {"C1": 0.7, "C2": 0.4, "C3": 0.3},
        "nominal_episodes_per_task": 128,
        "promotion_windows": 3,
        "rollback_windows": 2,
        "stage_episodes_per_task": 256,
    } or dict(atom) != {"C1": 0.7, "C2": 0.4, "C3": 0.3}:
        raise ValueError("Phase 6 curriculum policy is frozen")
    scheduler = _exact_keys(
        root["scheduler"],
        "scheduler",
        {"schedule_formula", "slot_length_source", "supercycle_windows_formula"},
    )
    if dict(scheduler) != {
        "schedule_formula": (
            "task_index=(window_index*world_size+rank+cycle_index)%task_count"
        ),
        "slot_length_source": "num_envs*num_steps_per_env",
        "supercycle_windows_formula": "task_count/gcd(task_count,world_size)",
    }:
        raise ValueError("Phase 6 scheduler formula is frozen")


@dataclass(frozen=True)
class Phase6Config:
    payload: Mapping[str, Any]
    raw_sha256: str
    canonical_sha256: str


def _verify_phase6_inputs(config: Phase6Config, *, repo_root: Path) -> None:
    expected_files = (
        ("configs/phase4b5_numeric_contract.json", _PHASE4_BINDING["raw_sha256"]),
        ("configs/phase5_ppo_v1.json", _PHASE5_BINDING["raw_sha256"]),
        (
            "configs/phase5_learning_acceptance_v1.json",
            _PHASE5_ACCEPTANCE_BINDING["raw_sha256"],
        ),
        (_INITIAL_CHECKPOINT["path"], _INITIAL_CHECKPOINT["sha256"]),
    )
    for relative, expected in expected_files:
        source = repo_root / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"Phase 6 binding checksum mismatch: {relative}")


def load_phase6_config(
    path: str | Path, *, repo_root: str | Path = REPO_ROOT
) -> Phase6Config:
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load Phase 6 config {source}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 6 config root must be an object")
    validate_phase6_config(payload)
    result = Phase6Config(
        payload=payload,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=canonical_sha256(payload),
    )
    _verify_phase6_inputs(result, repo_root=Path(repo_root))
    return result


def _verify_checksum_manifest(path: Path, expected_sha256: str) -> None:
    checksum_file = path / "SHA256SUMS"
    if not checksum_file.is_file() or sha256_file(checksum_file) != expected_sha256:
        raise ValueError(f"artifact SHA256SUMS mismatch: {checksum_file}")
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"invalid SHA256SUMS line in {checksum_file}")
        digest, name = fields
        target = path / name.lstrip(" *")
        if not target.is_file() or _sha256(digest, "artifact digest") != sha256_file(
            target
        ):
            raise ValueError(f"artifact member checksum mismatch: {target}")
        checked += 1
    if checked == 0:
        raise ValueError("artifact SHA256SUMS must contain entries")


@dataclass(frozen=True)
class RosterTask:
    task: str
    artifact_path: Path
    artifact_sha256sums_sha256: str
    adapter_identity: str
    task_spec_identity: str
    mismatch_owner: str
    retention_owner: str


@dataclass(frozen=True)
class Phase6Roster:
    tasks: tuple[RosterTask, ...]
    raw_sha256: str
    canonical_sha256: str
    phase6_canonical_sha256: str


def load_phase6_task_roster(
    path: str | Path,
    *,
    phase6_config: Phase6Config,
    repo_root: str | Path = REPO_ROOT,
) -> Phase6Roster:
    source = Path(path)
    root_path = Path(repo_root).resolve()
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load Phase 6 roster {source}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 6 roster root must be an object")
    canonical_json_bytes(payload)
    root = _exact_keys(
        payload,
        "roster",
        {"contract_version", "phase6_canonical_sha256", "tasks"},
    )
    if root["contract_version"] != PHASE6_ROSTER_VERSION:
        raise ValueError("unexpected Phase 6 roster version")
    if root["phase6_canonical_sha256"] != phase6_config.canonical_sha256:
        raise ValueError("roster does not bind this Phase 6 algorithm contract")
    values = root["tasks"]
    if not isinstance(values, list) or len(values) != len(_CURRENT_TASKS):
        raise ValueError("Phase 6 v1 roster must contain exactly four tasks")
    parsed: list[RosterTask] = []
    expected_item_keys = {
        "adapter_identity",
        "admitted",
        "artifact_absolute_path",
        "artifact_repo_relative_path",
        "artifact_sha256sums_sha256",
        "mismatch_owner",
        "retention_owner",
        "task",
        "task_spec_identity",
    }
    for index, item in enumerate(values):
        row = _exact_keys(item, f"tasks[{index}]", expected_item_keys)
        task = row["task"]
        if task != _CURRENT_TASKS[index] or task not in _TASK_IDENTITIES:
            raise ValueError("Phase 6 task roster order or identity is not admitted")
        if row["admitted"] is not True:
            raise ValueError(f"task {task} is not admitted")
        identity = _TASK_IDENTITIES[task]
        for name, expected in identity.items():
            if row[name] != expected:
                raise ValueError(f"task {task} has an unknown {name}")
        relative = row["artifact_repo_relative_path"]
        absolute = row["artifact_absolute_path"]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"task {task} artifact path must be repository relative")
        artifact = (root_path / relative).resolve()
        if not isinstance(absolute, str) or Path(absolute).resolve() != artifact:
            raise ValueError(
                f"task {task} absolute artifact path does not match roster"
            )
        digest = _sha256(row["artifact_sha256sums_sha256"], f"tasks[{index}].sha")
        _verify_checksum_manifest(artifact, digest)
        manifest_path = artifact / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid artifact manifest: {manifest_path}") from exc
        manifest_task = manifest.get("task", "push_door_hand")
        if (
            manifest_task != task
            or manifest.get("contract_version") != row["task_spec_identity"]
        ):
            raise ValueError(f"task {task} artifact task-spec identity mismatch")
        parsed.append(
            RosterTask(
                task=task,
                artifact_path=artifact,
                artifact_sha256sums_sha256=digest,
                adapter_identity=str(row["adapter_identity"]),
                task_spec_identity=str(row["task_spec_identity"]),
                mismatch_owner=str(row["mismatch_owner"]),
                retention_owner=str(row["retention_owner"]),
            )
        )
    if len({item.task for item in parsed}) != len(parsed):
        raise ValueError("Phase 6 roster contains duplicate tasks")
    return Phase6Roster(
        tasks=tuple(parsed),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=canonical_sha256(payload),
        phase6_canonical_sha256=str(root["phase6_canonical_sha256"]),
    )


@dataclass(frozen=True)
class LearningAcceptanceConfig:
    """严格绑定 Phase 6、roster 与 Phase 5 初始 checkpoint 的合同。"""

    payload: Mapping[str, Any]
    raw_sha256: str
    canonical_sha256: str


def _validate_learning_bindings(bindings: object, *, repo_root: Path) -> None:
    expected = {
        "phase6_canonical_sha256": _LEARNING_PHASE6_CANONICAL_SHA256,
        "phase6_raw_sha256": _LEARNING_PHASE6_RAW_SHA256,
        "phase5_initial_checkpoint_sha256": _LEARNING_PHASE5_CHECKPOINT_SHA256,
        "roster_canonical_sha256": _LEARNING_ROSTER_CANONICAL_SHA256,
        "roster_raw_sha256": _LEARNING_ROSTER_RAW_SHA256,
    }
    values = _exact_keys(bindings, "learning.bindings", set(expected))
    if dict(values) != expected:
        raise ValueError("learning acceptance bindings are not admitted")
    files = {
        "configs/phase6_joint_training_v1.json": expected["phase6_raw_sha256"],
        "configs/phase6_task_roster_v1.json": expected["roster_raw_sha256"],
        _INITIAL_CHECKPOINT["path"]: expected["phase5_initial_checkpoint_sha256"],
    }
    for relative, digest in files.items():
        source = repo_root / relative
        if not source.is_file() or sha256_file(source) != digest:
            raise ValueError(
                f"learning acceptance binding checksum mismatch: {relative}"
            )


def validate_learning_acceptance_config(
    payload: Mapping[str, Any], *, repo_root: str | Path = REPO_ROOT
) -> None:
    """Validate the frozen production learning contract and all arithmetic."""
    canonical_json_bytes(payload)
    root = _exact_keys(
        payload,
        "learning",
        {
            "bindings",
            "checkpoint",
            "contract_version",
            "crossing",
            "curriculum",
            "evaluation",
            "final_acceptance",
            "profiles",
            "runtime",
        },
    )
    if root["contract_version"] != PHASE6_LEARNING_ACCEPTANCE_VERSION:
        raise ValueError("unexpected Phase 6 learning acceptance version")
    _validate_learning_bindings(root["bindings"], repo_root=Path(repo_root))
    runtime = _exact_keys(
        root["runtime"],
        "learning.runtime",
        {
            "evaluation_seed",
            "global_transitions_per_iteration",
            "initial_stage",
            "num_envs_per_rank",
            "num_steps_per_env",
            "optimizer_steps_per_iteration",
            "train_seed",
            "world_size",
        },
    )
    expected_runtime = {
        "evaluation_seed": 20262806,
        "global_transitions_per_iteration": 8192,
        "initial_stage": "C1",
        "num_envs_per_rank": 64,
        "num_steps_per_env": 32,
        "optimizer_steps_per_iteration": 24,
        "train_seed": 20260806,
        "world_size": 4,
    }
    if dict(runtime) != expected_runtime:
        raise ValueError("Phase 6 learning runtime arithmetic or seed changed")
    if (
        runtime["world_size"]
        * runtime["num_envs_per_rank"]
        * runtime["num_steps_per_env"]
        != runtime["global_transitions_per_iteration"]
        or runtime["optimizer_steps_per_iteration"] != 3 * 8
    ):
        raise ValueError("Phase 6 learning per-iteration arithmetic is invalid")
    profiles = _exact_keys(root["profiles"], "learning.profiles", {"main", "pilot"})
    for name, expected in {
        "pilot": (31, 253952, 63488, 744),
        "main": (2442, 20004864, 5001216, 58608),
    }.items():
        row = _exact_keys(
            profiles[name],
            f"learning.profiles.{name}",
            {
                "global_transitions",
                "optimizer_steps",
                "per_task_transitions",
                "total_iterations",
            },
        )
        if dict(row) != dict(
            zip(
                (
                    "total_iterations",
                    "global_transitions",
                    "per_task_transitions",
                    "optimizer_steps",
                ),
                expected,
                strict=True,
            )
        ):
            raise ValueError(f"Phase 6 {name} profile arithmetic is invalid")
        if (
            row["global_transitions"]
            != row["total_iterations"] * runtime["global_transitions_per_iteration"]
            or row["per_task_transitions"] * 4 != row["global_transitions"]
            or row["optimizer_steps"]
            != row["total_iterations"] * runtime["optimizer_steps_per_iteration"]
        ):
            raise ValueError(f"Phase 6 {name} transition balance is invalid")
    crossing = _exact_keys(
        root["crossing"],
        "learning.crossing",
        {
            "actual_transitions_per_iteration",
            "formula",
            "interval_transitions",
            "record_logical_and_actual",
            "reject_duplicate_or_skipped",
        },
    )
    if dict(crossing) != {
        "actual_transitions_per_iteration": 8192,
        "formula": "target_iteration=ceil(logical_crossing/8192)",
        "interval_transitions": 250000,
        "record_logical_and_actual": True,
        "reject_duplicate_or_skipped": True,
    }:
        raise ValueError("Phase 6 crossing contract is frozen")
    checkpoint = _exact_keys(
        root["checkpoint"],
        "learning.checkpoint",
        {
            "checkpoint_version",
            "latest_pointer",
            "persist_rank_rng_states",
            "post_evaluation_atomic",
            "pre_evaluation_required",
            "resume_strict",
        },
    )
    if dict(checkpoint) != {
        "checkpoint_version": PHASE6_LEARNING_CHECKPOINT_VERSION,
        "latest_pointer": "latest.json",
        "persist_rank_rng_states": True,
        "post_evaluation_atomic": True,
        "pre_evaluation_required": True,
        "resume_strict": True,
    }:
        raise ValueError("Phase 6 learning checkpoint policy is frozen")
    evaluation = _exact_keys(
        root["evaluation"],
        "learning.evaluation",
        {
            "deterministic_actor_mean",
            "evaluation_seed",
            "forbid_gradient_optimizer_normalizer_update",
            "horizons",
            "logical_interval_transitions",
            "nominal_atom",
            "nominal_pair_seeds_per_task",
            "paired_modes",
            "pair_seeds_per_task",
            "rank_zero_aggregate",
            "stage",
            "stage_pair_seed_base",
        },
    )
    if (
        evaluation["evaluation_seed"] != 20262806
        or evaluation["stage_pair_seed_base"] != 20262806
    ):
        raise ValueError("Phase 6 evaluation seed is frozen")
    if (
        evaluation["logical_interval_transitions"] != 250000
        or evaluation["pair_seeds_per_task"] != 256
        or evaluation["nominal_pair_seeds_per_task"] != 128
    ):
        raise ValueError("Phase 6 evaluation quota is frozen")
    if evaluation["paired_modes"] != ["residual", "scaffold_only"] or not all(
        evaluation[name] is True
        for name in (
            "deterministic_actor_mean",
            "forbid_gradient_optimizer_normalizer_update",
            "rank_zero_aggregate",
        )
    ):
        raise ValueError("Phase 6 evaluation isolation is frozen")
    if dict(evaluation["horizons"]) != {
        "move_largebox": 199,
        "move_suitcase": 472,
        "push_box": 792,
        "push_door_hand": 508,
    }:
        raise ValueError("Phase 6 evaluation horizons are frozen")
    curriculum = _exact_keys(
        root["curriculum"],
        "learning.curriculum",
        {
            "gates",
            "promotion_eligibility",
            "promotion_windows",
            "rollback",
            "rollback_windows",
        },
    )
    if (
        curriculum["promotion_windows"] != 3
        or curriculum["rollback_windows"] != 2
        or curriculum["promotion_eligibility"] != {"C1": 5000000, "C2": 10000000}
    ):
        raise ValueError("Phase 6 curriculum eligibility is frozen")
    final = _mapping(root["final_acceptance"], "learning.final_acceptance")
    required_final = {
        "final_stage",
        "final_iteration",
        "task_transition_fraction",
        "nominal_retention_min",
        "nominal_invalid_max",
        "nominal_saturation_max",
        "residual_mean_norm_min",
        "contact_bearing_residual_fraction_min",
        "benefit_task_count_min",
        "benefit_mismatch_family_count_min",
        "benefit_success_delta_min",
        "benefit_progress_delta_min",
        "benefit_force_p95_ratio_max",
        "stability_degradation_max",
        "task_normalized_reward_share_max",
        "task_semantic_total_share_max",
        "semantic_total_sum_zero_share",
    }
    if (
        set(final) != required_final
        or final["final_stage"] != "C3"
        or final["final_iteration"] != 2442
    ):
        raise ValueError("Phase 6 final acceptance contract is frozen")


def load_learning_acceptance_config(
    path: str | Path = PHASE6_LEARNING_CONFIG, *, repo_root: str | Path = REPO_ROOT
) -> LearningAcceptanceConfig:
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to load Phase 6 learning acceptance {source}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 6 learning acceptance root must be an object")
    validate_learning_acceptance_config(payload, repo_root=repo_root)
    return LearningAcceptanceConfig(
        payload, hashlib.sha256(raw).hexdigest(), canonical_sha256(payload)
    )


def learning_transition_arithmetic(
    *,
    iterations: int,
    world_size: int = 4,
    num_envs_per_rank: int = 64,
    num_steps_per_env: int = 32,
) -> dict[str, int]:
    iterations = _positive_int(iterations, "iterations")
    global_transitions = transition_accounting(
        world_size=world_size,
        num_envs=num_envs_per_rank,
        num_steps_per_env=num_steps_per_env,
        iterations=iterations,
    )
    return {
        "iterations": iterations,
        "global_transitions": global_transitions,
        "per_task_transitions": global_transitions // 4,
        "optimizer_steps": iterations * 24,
    }


@dataclass(frozen=True)
class CrossingRecord:
    logical_transitions: int
    target_iteration: int
    actual_transitions: int


@dataclass(frozen=True)
class LearningSegment:
    segment_index: int
    start_iteration: int
    end_iteration: int
    start_transitions: int
    end_transitions: int
    crossing: CrossingRecord


def crossing_target_iteration(
    logical_crossing: int, *, transitions_per_iteration: int = 8192
) -> int:
    logical_crossing = _positive_int(logical_crossing, "logical_crossing")
    transitions_per_iteration = _positive_int(
        transitions_per_iteration, "transitions_per_iteration"
    )
    return (
        logical_crossing + transitions_per_iteration - 1
    ) // transitions_per_iteration


def plan_learning_segments(
    *,
    total_iterations: int,
    interval_transitions: int = 250000,
    transitions_per_iteration: int = 8192,
) -> tuple[LearningSegment, ...]:
    total_iterations = _positive_int(total_iterations, "total_iterations")
    interval_transitions = _positive_int(interval_transitions, "interval_transitions")
    transitions_per_iteration = _positive_int(
        transitions_per_iteration, "transitions_per_iteration"
    )
    records: list[LearningSegment] = []
    previous_iteration = 0
    previous_transitions = 0
    logical = interval_transitions
    index = 0
    while True:
        target = crossing_target_iteration(
            logical, transitions_per_iteration=transitions_per_iteration
        )
        if target > total_iterations:
            break
        actual = target * transitions_per_iteration
        records.append(
            LearningSegment(
                index,
                previous_iteration,
                target,
                previous_transitions,
                actual,
                CrossingRecord(logical, target, actual),
            )
        )
        previous_iteration, previous_transitions = target, actual
        logical += interval_transitions
        index += 1
    if not records or records[-1].end_iteration != total_iterations:
        raise ValueError("learning total iterations must end at a recorded crossing")
    if len({item.crossing.logical_transitions for item in records}) != len(records):
        raise ValueError("learning segment plan contains duplicate crossings")
    if tuple(item.crossing.target_iteration for item in records) != tuple(
        sorted(item.crossing.target_iteration for item in records)
    ):
        raise ValueError("learning segment plan skips or reorders a crossing")
    return tuple(records)


def next_learning_crossing(
    actual_transitions: int, *, interval_transitions: int = 250000
) -> int | None:
    actual_transitions = _nonnegative_int(actual_transitions, "actual_transitions")
    interval_transitions = _positive_int(interval_transitions, "interval_transitions")
    candidate = (
        (actual_transitions // interval_transitions) + 1
    ) * interval_transitions
    return candidate


@dataclass(frozen=True)
class PairedSeed:
    seed: int
    residual_mode: str = "residual"
    scaffold_mode: str = "scaffold_only"


@dataclass(frozen=True)
class PairedEvaluationSlot:
    """One deterministic environment-row episode in a paired evaluation."""

    env_id: int
    episode_index: int
    seed: int
    subset: str
    nominal: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "env_id": self.env_id,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "subset": self.subset,
            "nominal": self.nominal,
        }


def paired_evaluation_rank_seed(evaluation_seed: int, rank: int) -> int:
    """Return the globally disjoint sampler base seed for one evaluation rank."""
    for value, name in ((evaluation_seed, "evaluation_seed"), (rank, "rank")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    return evaluation_seed + rank * 1_000_000_000


def paired_evaluation_schedule(
    *,
    task: str,
    stage: str,
    evaluation_seed: int,
    rank: int,
    num_envs: int,
    stage_quota: int,
    nominal_quota: int,
    nominal_probe: Callable[[int, int], bool],
) -> dict[str, object]:
    """Build a row-balanced schedule whose nominal rows are sampler-proven."""
    if not isinstance(task, str) or not task:
        raise ValueError("schedule task must be non-empty")
    if stage not in {"C1", "C2", "C3"}:
        raise ValueError("schedule stage must be C1, C2, or C3")
    for value, name in (
        (evaluation_seed, "evaluation_seed"),
        (rank, "rank"),
        (num_envs, "num_envs"),
        (stage_quota, "stage_quota"),
        (nominal_quota, "nominal_quota"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if num_envs <= 0 or stage_quota <= 0 or nominal_quota <= 0:
        raise ValueError("schedule dimensions must be positive")
    if stage_quota % num_envs or nominal_quota % num_envs:
        raise ValueError("stage and nominal quotas must divide num_envs")
    if not callable(nominal_probe):
        raise TypeError("nominal_probe must be callable")
    stage_per_row = stage_quota // num_envs
    nominal_per_row = nominal_quota // num_envs
    task_base = paired_evaluation_rank_seed(evaluation_seed, rank)
    rows: dict[str, list[dict[str, object]]] = {}
    for env_id in range(num_envs):
        row: list[dict[str, object]] = []
        for episode_index in range(stage_per_row):
            row.append(
                PairedEvaluationSlot(
                    env_id,
                    episode_index,
                    task_base + env_id + 1_000_003 * episode_index,
                    "stage",
                    bool(nominal_probe(env_id, episode_index)),
                ).as_dict()
            )
        accepted = 0
        candidate = stage_per_row
        scanned = 0
        while accepted < nominal_per_row and scanned < 4096:
            scanned += 1
            if nominal_probe(env_id, candidate):
                row.append(
                    PairedEvaluationSlot(
                        env_id,
                        candidate,
                        task_base + env_id + 1_000_003 * candidate,
                        "nominal",
                        True,
                    ).as_dict()
                )
                accepted += 1
            candidate += 1
        if accepted != nominal_per_row:
            raise ValueError("nominal sampler did not provide the requested quota")
        rows[str(env_id)] = row
    batches = []
    for subset, count in (("stage", stage_per_row), ("nominal", nominal_per_row)):
        for offset in range(count):
            batches.append(
                {
                    "subset": subset,
                    "slots": [
                        rows[str(env_id)][
                            offset if subset == "stage" else stage_per_row + offset
                        ]
                        for env_id in range(num_envs)
                    ],
                }
            )
    canonical = {
        "evaluation_seed": evaluation_seed,
        "num_envs": num_envs,
        "nominal_quota": nominal_quota,
        "rank": rank,
        "stage": stage,
        "stage_quota": stage_quota,
        "task": task,
        "task_base": task_base,
        "rows": rows,
    }
    schedule = {
        **canonical,
        "batches": batches,
        "schedule_sha256": canonical_sha256(canonical),
    }
    validate_paired_evaluation_schedule(schedule)
    return schedule


def validate_paired_evaluation_schedule(schedule: Mapping[str, object]) -> None:
    """Reject quota, seed, nominal-authenticity, or row-order mutations."""
    required = {
        "batches",
        "evaluation_seed",
        "nominal_quota",
        "num_envs",
        "rank",
        "rows",
        "schedule_sha256",
        "stage",
        "stage_quota",
        "task",
        "task_base",
    }
    if not isinstance(schedule, Mapping) or set(schedule) != required:
        raise ValueError("paired evaluation schedule schema is not strict")
    if not isinstance(schedule["rows"], Mapping):
        raise ValueError("paired evaluation schedule rows are invalid")
    num_envs = schedule["num_envs"]
    stage_quota = schedule["stage_quota"]
    nominal_quota = schedule["nominal_quota"]
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or not isinstance(stage_quota, int)
        or stage_quota <= 0
        or stage_quota % num_envs
        or not isinstance(nominal_quota, int)
        or nominal_quota <= 0
        or nominal_quota % num_envs
    ):
        raise ValueError("paired evaluation schedule quotas are invalid")
    expected_sha = canonical_sha256(
        {
            key: schedule[key]
            for key in (
                "evaluation_seed",
                "num_envs",
                "nominal_quota",
                "rank",
                "stage",
                "stage_quota",
                "task",
                "task_base",
                "rows",
            )
        }
    )
    if schedule["schedule_sha256"] != expected_sha:
        raise ValueError("paired evaluation schedule hash is invalid")
    if schedule["task_base"] != paired_evaluation_rank_seed(
        schedule["evaluation_seed"], schedule["rank"]
    ):
        raise ValueError("paired evaluation task base is invalid")
    rows = schedule["rows"]
    if set(rows) != {str(index) for index in range(num_envs)}:
        raise ValueError("paired evaluation schedule rows are incomplete")
    all_slots: list[Mapping[str, object]] = []
    stage_per_row = stage_quota // num_envs
    nominal_per_row = nominal_quota // num_envs
    for env_id in range(num_envs):
        row = rows[str(env_id)]
        if not isinstance(row, list) or len(row) != stage_per_row + nominal_per_row:
            raise ValueError("paired evaluation row quota is invalid")
        for offset, slot in enumerate(row):
            if not isinstance(slot, Mapping) or set(slot) != {
                "env_id",
                "episode_index",
                "nominal",
                "seed",
                "subset",
            }:
                raise ValueError("paired evaluation slot schema is invalid")
            if slot["env_id"] != env_id:
                raise ValueError("paired evaluation slot row identity changed")
            expected_subset = "stage" if offset < stage_per_row else "nominal"
            if slot["subset"] != expected_subset:
                raise ValueError("paired evaluation subset order changed")
            if expected_subset == "nominal" and slot["nominal"] is not True:
                raise ValueError("nominal schedule slot was not sampler-proven")
            episode_index = slot["episode_index"]
            if not isinstance(episode_index, int) or episode_index < 0:
                raise ValueError("paired evaluation episode index is invalid")
            expected_seed = (
                int(schedule["task_base"]) + env_id + 1_000_003 * episode_index
            )
            if slot["seed"] != expected_seed:
                raise ValueError("paired evaluation seed formula changed")
            all_slots.append(slot)
    if len({int(slot["seed"]) for slot in all_slots}) != len(all_slots):
        raise ValueError("paired evaluation seeds are repeated")
    stage_batches = stage_per_row
    nominal_batches = nominal_per_row
    batches = schedule["batches"]
    if not isinstance(batches, list) or len(batches) != stage_batches + nominal_batches:
        raise ValueError("paired evaluation batches are incomplete")
    expected_batches = [
        ("stage", offset, stage_per_row) for offset in range(stage_batches)
    ] + [("nominal", offset, nominal_per_row) for offset in range(nominal_batches)]
    for batch, (subset, offset, per_row) in zip(batches, expected_batches, strict=True):
        if not isinstance(batch, Mapping) or set(batch) != {"slots", "subset"}:
            raise ValueError("paired evaluation batch schema is invalid")
        if batch["subset"] != subset or not isinstance(batch["slots"], list):
            raise ValueError("paired evaluation batch order changed")
        if len(batch["slots"]) != num_envs:
            raise ValueError("paired evaluation batch row count is invalid")
        expected_index = offset if subset == "stage" else stage_per_row + offset
        for env_id, slot in enumerate(batch["slots"]):
            if slot != rows[str(env_id)][expected_index]:
                raise ValueError(
                    "paired evaluation batch slot differs from row schedule"
                )


def paired_evaluation_plan(
    roster: Phase6Roster,
    *,
    stage: str = "C1",
    evaluation_seed: int = 20262806,
    stage_quota: int = 256,
    nominal_quota: int = 128,
) -> dict[str, object]:
    """Create equal, seed-paired residual/scaffold episodes per roster task."""
    if stage not in {"C1", "C2", "C3"}:
        raise ValueError("paired evaluation stage must be C1, C2, or C3")
    evaluation_seed = _nonnegative_int(evaluation_seed, "evaluation_seed")
    stage_quota = _positive_int(stage_quota, "stage_quota")
    nominal_quota = _positive_int(nominal_quota, "nominal_quota")
    horizon = {
        "push_door_hand": 508,
        "push_box": 792,
        "move_suitcase": 472,
        "move_largebox": 199,
    }
    tasks: dict[str, object] = {}
    for task_index, task in enumerate(roster.tasks):
        if task.task not in horizon:
            raise ValueError(f"evaluation horizon is missing for {task.task}")
        stage_seeds = tuple(
            PairedSeed(evaluation_seed + task_index * 1000000 + seed)
            for seed in range(stage_quota)
        )
        nominal_seeds = tuple(
            PairedSeed(evaluation_seed + 100000000 + task_index * 1000000 + seed)
            for seed in range(nominal_quota)
        )
        tasks[task.task] = {
            "rank": task_index,
            "stage": stage,
            "horizon": horizon[task.task],
            "stage_pairs": tuple(stage_seeds),
            "nominal_pairs": tuple(nominal_seeds),
        }
    return {
        "stage": stage,
        "evaluation_seed": evaluation_seed,
        "tasks": tasks,
        "pair_quota": {"stage": stage_quota, "nominal": nominal_quota},
        "modes": ("residual", "scaffold_only"),
        "rank_assignment": "roster_order",
    }


def validate_paired_evaluation_plan(
    plan: Mapping[str, object], *, roster: Phase6Roster
) -> None:
    root = _exact_keys(
        plan,
        "paired_evaluation",
        {"evaluation_seed", "modes", "pair_quota", "rank_assignment", "stage", "tasks"},
    )
    if root["stage"] not in {"C1", "C2", "C3"} or root["modes"] != (
        "residual",
        "scaffold_only",
    ):
        raise ValueError("paired evaluation plan mode or stage is invalid")
    tasks = _exact_keys(
        root["tasks"], "paired_evaluation.tasks", {task.task for task in roster.tasks}
    )
    global_pair_seeds: dict[str, set[int]] = {
        "stage_pairs": set(),
        "nominal_pairs": set(),
    }
    for index, task in enumerate(roster.tasks):
        record = _exact_keys(
            tasks[task.task],
            f"paired_evaluation.{task.task}",
            {"horizon", "nominal_pairs", "rank", "stage", "stage_pairs"},
        )
        if record["rank"] != index or record["stage"] != root["stage"]:
            raise ValueError("paired evaluation rank assignment is invalid")
        if record["horizon"] not in {199, 472, 792, 508}:
            raise ValueError("paired evaluation horizon is invalid")
        for name in ("stage_pairs", "nominal_pairs"):
            pairs = record[name]
            if not isinstance(pairs, (tuple, list)) or not pairs:
                raise ValueError("paired evaluation quota is empty")
            seeds: list[int] = []
            for pair in pairs:
                if isinstance(pair, PairedSeed):
                    seed = pair.seed
                    if (
                        pair.residual_mode != "residual"
                        or pair.scaffold_mode != "scaffold_only"
                    ):
                        raise ValueError("paired seed modes are not actor-clean")
                elif isinstance(pair, Mapping):
                    expected = _exact_keys(
                        pair, "paired_seed", {"residual_mode", "scaffold_mode", "seed"}
                    )
                    if (
                        expected["residual_mode"] != "residual"
                        or expected["scaffold_mode"] != "scaffold_only"
                    ):
                        raise ValueError("paired seed modes are not actor-clean")
                    seed = _nonnegative_int(expected["seed"], "paired_seed.seed")
                else:
                    raise TypeError("paired seed record is invalid")
                seeds.append(seed)
            if len(seeds) != len(set(seeds)):
                raise ValueError("paired evaluation seed is repeated")
            if global_pair_seeds[name].intersection(seeds):
                raise ValueError("paired evaluation seed is reused across tasks")
            global_pair_seeds[name].update(seeds)


def _task_gate(stage: str) -> dict[str, float]:
    if stage == "C1":
        return {"success_min": 0.75, "failure_max": 0.05, "retention_min": 0.95}
    if stage in {"C2", "C3"}:
        return {"success_min": 0.65, "failure_max": 0.08, "retention_min": 0.95}
    raise ValueError("stage must be C1, C2, or C3")


def _window_passes(
    stage: str, task_metrics: Mapping[str, Mapping[str, object]]
) -> bool:
    gate = _task_gate(stage)
    if not task_metrics:
        raise ValueError("curriculum window has no task metrics")
    for task, metrics in task_metrics.items():
        success = _finite(metrics.get("success"), f"{task}.success")
        failure = _finite(metrics.get("failure"), f"{task}.failure")
        retention = _finite(metrics.get("retention"), f"{task}.retention")
        if (
            success < gate["success_min"]
            or failure > gate["failure_max"]
            or retention < gate["retention_min"]
        ):
            return False
    return True


def _window_rolls_back(
    stage: str, task_metrics: Mapping[str, Mapping[str, object]]
) -> bool:
    gate = _task_gate(stage)
    for task, metrics in task_metrics.items():
        success = _finite(metrics.get("success"), f"{task}.success")
        failure = _finite(metrics.get("failure"), f"{task}.failure")
        retention = _finite(metrics.get("retention"), f"{task}.retention")
        if (
            success < gate["success_min"] - 0.15
            or failure > gate["failure_max"] + 0.05
            or retention < 0.90
        ):
            return True
    return False


def _validate_p_cross_trace(value: object, *, path: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain one trace entry per control step")
    for index, step in enumerate(value):
        if (
            not isinstance(step, list)
            or len(step) != 13
            or any(not isinstance(row, list) or len(row) != 5 for row in step)
        ):
            raise ValueError(f"{path}[{index}] must have shape [13, 5]")
        flattened = [_finite(cell, f"{path}[{index}]") for row in step for cell in row]
        if not math.isclose(sum(flattened), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise ValueError(f"{path}[{index}] must sum to one")


def validate_phase6_joint_metrics(
    metrics: object,
    *,
    config: Phase6Config,
    roster: Phase6Roster,
    expected_global_transitions: int,
) -> dict[str, object]:
    """Reject incomplete, synthetic, or cross-rank-divergent Phase 6 metrics."""
    root = _exact_keys(metrics, "metrics", {"pooled", "task_records"})
    schema = phase6_metrics_schema(config)
    task_records = _exact_keys(
        root["task_records"],
        "metrics.task_records",
        {task.task for task in roster.tasks},
    )
    transition_names = set(schema["transition_metric_names"])
    episode_names = set(schema["episode_metric_names"])
    transition_counts: dict[str, int] = {}
    normalized_records: dict[str, dict[str, object]] = {}
    for task in roster.tasks:
        record = _exact_keys(
            task_records[task.task],
            f"metrics.task_records.{task.task}",
            {
                "episode_count",
                "episode_metrics",
                "p_cross_trace",
                "task_transitions",
                "transition_metrics",
            },
        )
        episode_count = _nonnegative_int(
            record["episode_count"], f"{task.task}.episode_count"
        )
        episodes = _exact_keys(
            record["episode_metrics"], f"{task.task}.episode_metrics", episode_names
        )
        for name, value in episodes.items():
            if episode_count == 0:
                if value is not None:
                    raise ValueError(
                        f"{task.task}.episode_metrics.{name} must be null without episodes"
                    )
            else:
                _finite(value, f"{task.task}.episode_metrics.{name}")
        transition = _exact_keys(
            record["transition_metrics"],
            f"{task.task}.transition_metrics",
            transition_names,
        )
        for name, value in transition.items():
            _finite(value, f"{task.task}.transition_metrics.{name}")
        _validate_p_cross_trace(record["p_cross_trace"], path=f"{task.task}.P_cross")
        transition_counts[task.task] = _positive_int(
            record["task_transitions"], f"{task.task}.task_transitions"
        )
        normalized_records[task.task] = dict(record)
    assert_transition_balance(
        transition_counts, expected_total=expected_global_transitions
    )
    pooled = _exact_keys(
        root["pooled"],
        "metrics.pooled",
        {"episode_mean", "task_episode_counts", "transition_mean"},
    )
    expected_pooled = pooled_macro_metrics(normalized_records)["pooled"]
    if dict(pooled) != expected_pooled:
        raise ValueError("pooled Phase 6 metrics do not match task records")
    return {
        "pooled": dict(pooled),
        "task_records": normalized_records,
    }


@dataclass(frozen=True)
class ScheduleSlot:
    cycle_index: int
    window_index: int
    rank: int
    task_index: int


class JointTaskScheduler:
    """Assign homogeneous task slots independently of task and rank counts."""

    def __init__(self, *, task_count: int, world_size: int) -> None:
        self.task_count = _positive_int(task_count, "task_count")
        self.world_size = _positive_int(world_size, "world_size")
        self.common_divisor = math.gcd(self.task_count, self.world_size)
        self.supercycle_windows = self.task_count // self.common_divisor
        self.slots_per_task = self.world_size // self.common_divisor

    def assignment(
        self, *, cycle_index: int, window_index: int, rank: int
    ) -> ScheduleSlot:
        cycle = _nonnegative_int(cycle_index, "cycle_index")
        window = _nonnegative_int(window_index, "window_index")
        if window >= self.supercycle_windows:
            raise ValueError("window_index exceeds the current supercycle")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < self.world_size
        ):
            raise ValueError("rank is outside the world size")
        task_index = (window * self.world_size + rank + cycle) % self.task_count
        return ScheduleSlot(cycle, window, rank, task_index)

    def supercycle_slot_counts(self, *, cycle_index: int = 0) -> dict[int, int]:
        counts = {index: 0 for index in range(self.task_count)}
        for window in range(self.supercycle_windows):
            for rank in range(self.world_size):
                counts[
                    self.assignment(
                        cycle_index=cycle_index, window_index=window, rank=rank
                    ).task_index
                ] += 1
        if set(counts.values()) != {self.slots_per_task}:
            raise AssertionError("scheduler supercycle is not task balanced")
        return counts

    def next_slot(self, *, cycle_index: int, window_index: int) -> tuple[int, int]:
        self.assignment(cycle_index=cycle_index, window_index=window_index, rank=0)
        next_window = window_index + 1
        if next_window == self.supercycle_windows:
            return cycle_index + 1, 0
        return cycle_index, next_window

    def expected_task_transitions(
        self,
        *,
        cycle_index: int,
        window_index: int,
        slot_transitions: int,
    ) -> dict[int, int]:
        per_slot = _positive_int(slot_transitions, "slot_transitions")
        self.assignment(cycle_index=cycle_index, window_index=window_index, rank=0)
        counts = {index: 0 for index in range(self.task_count)}
        for cycle in range(cycle_index + 1):
            final_window = (
                window_index if cycle == cycle_index else self.supercycle_windows - 1
            )
            for window in range(final_window + 1):
                for rank in range(self.world_size):
                    task = self.assignment(
                        cycle_index=cycle, window_index=window, rank=rank
                    ).task_index
                    counts[task] += per_slot
        return counts


def transition_accounting(
    *, world_size: int, num_envs: int, num_steps_per_env: int, iterations: int
) -> int:
    return (
        _positive_int(world_size, "world_size")
        * _positive_int(num_envs, "num_envs")
        * _positive_int(num_steps_per_env, "num_steps_per_env")
        * _positive_int(iterations, "iterations")
    )


def assert_transition_balance(
    task_transitions: Mapping[str, int], *, expected_total: int
) -> None:
    if not task_transitions:
        raise ValueError("task transition accounting is empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in task_transitions.values()
    ):
        raise ValueError("task transition accounting must be nonnegative integers")
    if sum(task_transitions.values()) != _positive_int(
        expected_total, "expected_total"
    ):
        raise ValueError("task transitions do not sum to the global transition count")
    values = set(task_transitions.values())
    if len(values) != 1:
        raise ValueError("joint task transitions are not balanced")


def phase6_metrics_schema(config: Phase6Config) -> Mapping[str, Any]:
    """Return the already-validated frozen Phase 6 metrics schema."""
    metrics = _mapping(config.payload["metrics"], "metrics")
    if dict(metrics) != _METRICS:
        raise ValueError("Phase 6 metrics schema is frozen")
    return metrics


def _finite_vector(value: object, *, name: str, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (
        not value.is_floating_point()
        or value.shape != (batch_size,)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite [{batch_size}] floating tensor")
    return value


class TransitionMetricCollector:
    """Collect Phase 6 transition metrics from pre-step environment state."""

    def __init__(self, metric_schema: Mapping[str, Any]) -> None:
        if dict(metric_schema) != _METRICS:
            raise ValueError("Phase 6 transition metrics require the frozen schema")
        self._dt = float(metric_schema["impulse_dt_s"])
        self._force_norm_samples: list[float] = []
        self._force_rate_samples: list[float] = []
        self._p_cross_trace: list[list[list[float]]] = []
        self._previous_force_norm: torch.Tensor | None = None
        self._num_envs: int | None = None
        self._sums = {
            "contact_fraction": 0.0,
            "force_impulse_Ns_mean": 0.0,
            "p_cross_entropy_mean": 0.0,
            "residual_arms_mean_norm": 0.0,
            "residual_legs_mean_norm": 0.0,
            "residual_waist_mean_norm": 0.0,
            "saturation_fraction": 0.0,
            "stability_margin_mean_m": 0.0,
        }
        self._transitions = 0

    @property
    def transitions(self) -> int:
        return self._transitions

    def observe(
        self,
        *,
        observed_wrench: object,
        contact_fraction: object,
        stability_margin: object,
        residual_arms_norm: object,
        residual_waist_norm: object,
        residual_legs_norm: object,
        saturation_fraction: object,
        p_cross: object,
    ) -> None:
        if (
            not isinstance(observed_wrench, torch.Tensor)
            or not observed_wrench.is_floating_point()
            or observed_wrench.ndim != 3
            or observed_wrench.shape[1:] != (2, 6)
            or observed_wrench.shape[0] <= 0
            or not torch.isfinite(observed_wrench).all()
        ):
            raise ValueError("observed_wrench must be a finite [B, 2, 6] tensor")
        batch_size = observed_wrench.shape[0]
        if self._num_envs is None:
            self._num_envs = batch_size
        elif self._num_envs != batch_size:
            raise ValueError("transition metric batch size changed during a task slot")
        vectors = {
            "contact_fraction": _finite_vector(
                contact_fraction, name="contact_fraction", batch_size=batch_size
            ),
            "stability_margin_mean_m": _finite_vector(
                stability_margin, name="stability_margin", batch_size=batch_size
            ),
            "residual_arms_mean_norm": _finite_vector(
                residual_arms_norm, name="residual_arms_norm", batch_size=batch_size
            ),
            "residual_waist_mean_norm": _finite_vector(
                residual_waist_norm, name="residual_waist_norm", batch_size=batch_size
            ),
            "residual_legs_mean_norm": _finite_vector(
                residual_legs_norm, name="residual_legs_norm", batch_size=batch_size
            ),
            "saturation_fraction": _finite_vector(
                saturation_fraction,
                name="saturation_fraction",
                batch_size=batch_size,
            ),
        }
        if (
            not isinstance(p_cross, torch.Tensor)
            or not p_cross.is_floating_point()
            or p_cross.shape != (batch_size, 13, 5)
            or not torch.isfinite(p_cross).all()
            or not torch.allclose(
                p_cross.sum(dim=(-2, -1)),
                torch.ones(batch_size, device=p_cross.device, dtype=p_cross.dtype),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ):
            raise ValueError("P_cross must be finite [B, 13, 5] distributions")
        force_norm = torch.linalg.vector_norm(observed_wrench[..., :3], dim=-1).mean(
            dim=-1
        )
        if self._previous_force_norm is not None:
            if self._previous_force_norm.shape != force_norm.shape:
                raise ValueError("force-rate batch shape changed during a task slot")
            force_rate = (force_norm - self._previous_force_norm).abs() / self._dt
            self._force_rate_samples.extend(force_rate.detach().cpu().tolist())
        self._previous_force_norm = force_norm.detach().clone()
        entropy = -(
            p_cross * p_cross.clamp_min(torch.finfo(p_cross.dtype).eps).log()
        ).sum(dim=(-2, -1))
        self._force_norm_samples.extend(force_norm.detach().cpu().tolist())
        self._p_cross_trace.append(p_cross.detach().mean(dim=0).cpu().tolist())
        self._sums["force_impulse_Ns_mean"] += float(
            (force_norm.sum() * self._dt).detach().item()
        )
        self._sums["p_cross_entropy_mean"] += float(entropy.detach().sum().item())
        for name, value in vectors.items():
            self._sums[name] += float(value.detach().sum().item())
        self._transitions += batch_size

    def transition_metrics(self) -> dict[str, float]:
        if self._transitions <= 0 or not self._force_rate_samples:
            raise ValueError("transition metrics require at least two control steps")
        force = torch.tensor(self._force_norm_samples, dtype=torch.float64)
        force_rate = torch.tensor(self._force_rate_samples, dtype=torch.float64)
        result = {
            "force_p95_N": float(torch.quantile(force, 0.95).item()),
            "force_p99_N": float(torch.quantile(force, 0.99).item()),
            "force_rate_p95_N_per_s": float(torch.quantile(force_rate, 0.95).item()),
        }
        for name, value in self._sums.items():
            denominator = (
                self._num_envs if name == "force_impulse_Ns_mean" else self._transitions
            )
            assert denominator is not None
            result[name] = value / denominator
        return result

    def raw_samples(self) -> dict[str, object]:
        """Return gather-only values needed for exact cross-rank quantiles."""
        return {
            "force_norm_N": list(self._force_norm_samples),
            "force_rate_N_per_s": list(self._force_rate_samples),
        }

    def p_cross_trace(self) -> list[list[list[float]]]:
        return [[list(row) for row in step] for step in self._p_cross_trace]


def episode_metrics_from_snapshot(
    snapshot: Sequence[Mapping[str, object]], *, metric_schema: Mapping[str, Any]
) -> tuple[int, dict[str, float | None]]:
    """Pool completed episodes only; an empty snapshot has no observed value."""
    if dict(metric_schema) != _METRICS:
        raise ValueError("Phase 6 episode metrics require the frozen schema")
    names = tuple(str(name) for name in metric_schema["episode_metric_names"])
    if not snapshot:
        return 0, {name: None for name in names}
    values = {name: [] for name in names}
    for index, record in enumerate(snapshot):
        if not isinstance(record, Mapping):
            raise TypeError(f"episode snapshot {index} must be a mapping")
        diagnostics = _mapping(
            record.get("diagnostics"), f"episode[{index}].diagnostics"
        )
        for name in names:
            source = record.get("return") if name == "return" else diagnostics.get(name)
            values[name].append(_finite(source, f"episode[{index}].{name}"))
    return len(snapshot), {
        name: sum(entries) / len(entries) for name, entries in values.items()
    }


def pooled_macro_metrics(
    task_records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Pool transition metrics equally by task and episode metrics by episode."""
    if not task_records:
        raise ValueError("joint metrics require at least one task")
    transition_counts: list[int] = []
    episode_counts: dict[str, int] = {}
    transition_rows: list[Mapping[str, object]] = []
    episode_rows: list[Mapping[str, object]] = []
    for task, record in task_records.items():
        row = _exact_keys(
            record,
            f"metrics.{task}",
            {
                "episode_count",
                "episode_metrics",
                "p_cross_trace",
                "task_transitions",
                "transition_metrics",
            },
        )
        _validate_p_cross_trace(row["p_cross_trace"], path=f"metrics.{task}.P_cross")
        transition_counts.append(
            _positive_int(row["task_transitions"], f"{task}.task_transitions")
        )
        episode_counts[task] = _nonnegative_int(
            row["episode_count"], f"{task}.episode_count"
        )
        transition_rows.append(
            _mapping(row["transition_metrics"], f"{task}.transition")
        )
        episode_rows.append(_mapping(row["episode_metrics"], f"{task}.episode"))
    if len(set(transition_counts)) != 1:
        raise ValueError("pooled transition metrics require equal task transitions")
    transition_keys = set(transition_rows[0])
    episode_keys = set(episode_rows[0])
    if any(set(row) != transition_keys for row in transition_rows) or any(
        set(row) != episode_keys for row in episode_rows
    ):
        raise ValueError("task metric schemas differ")
    transition_mean = {
        key: sum(_finite(row[key], f"transition.{key}") for row in transition_rows)
        / len(transition_rows)
        for key in sorted(transition_keys)
    }
    total_episodes = sum(episode_counts.values())
    episode_mean: dict[str, float | None] = {}
    for key in sorted(episode_keys):
        if not total_episodes:
            episode_mean[key] = None
            continue
        weighted_sum = 0.0
        for task, row in zip(task_records, episode_rows, strict=True):
            count = episode_counts[task]
            if count == 0:
                if row[key] is not None:
                    raise ValueError(
                        f"episode.{key} must be null when episode_count is zero"
                    )
                continue
            weighted_sum += _finite(row[key], f"episode.{key}") * count
        episode_mean[key] = weighted_sum / total_episodes
    return {
        "task_records": {task: dict(record) for task, record in task_records.items()},
        "pooled": {
            "episode_mean": episode_mean,
            "task_episode_counts": episode_counts,
            "transition_mean": transition_mean,
        },
    }


@dataclass(frozen=True)
class CurriculumDecision:
    stage: str
    promoted: bool
    rolled_back: bool
    crossing: int


class JointCurriculum:
    """Rank-zero-owned Phase 6 global curriculum with strict persistence."""

    def __init__(self, *, stage: str = "C1", transitions: int = 0) -> None:
        if stage not in {"C1", "C2", "C3"}:
            raise ValueError("joint curriculum stage must be C1, C2, or C3")
        self.stage = stage
        self.transitions = _nonnegative_int(transitions, "transitions")
        self.last_evaluation_transition = 0
        self.passes = 0
        self.rollbacks = 0

    @staticmethod
    def evaluation_crossings(previous: int, current: int) -> tuple[int, ...]:
        previous = _nonnegative_int(previous, "previous")
        current = _nonnegative_int(current, "current")
        if current < previous:
            raise ValueError("global transitions cannot move backwards")
        cadence = 250000
        return tuple(
            crossing
            for crossing in range(
                ((previous // cadence) + 1) * cadence, current + 1, cadence
            )
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "last_evaluation_transition": self.last_evaluation_transition,
            "passes": self.passes,
            "rollbacks": self.rollbacks,
            "stage": self.stage,
            "transitions": self.transitions,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        required = {
            "last_evaluation_transition",
            "passes",
            "rollbacks",
            "stage",
            "transitions",
        }
        if set(state) != required or state["stage"] not in {"C1", "C2", "C3"}:
            raise ValueError("joint curriculum checkpoint schema is invalid")
        values = {
            key: _nonnegative_int(state[key], f"curriculum.{key}")
            for key in required - {"stage"}
        }
        if values["last_evaluation_transition"] > values["transitions"]:
            raise ValueError("joint curriculum evaluation exceeds transitions")
        self.stage = str(state["stage"])
        self.transitions = values["transitions"]
        self.last_evaluation_transition = values["last_evaluation_transition"]
        self.passes = values["passes"]
        self.rollbacks = values["rollbacks"]

    def evaluate(
        self, *, crossing: int, passed: bool, rollback: bool
    ) -> CurriculumDecision:
        crossing = _positive_int(crossing, "crossing")
        if crossing % 250000 or crossing <= self.last_evaluation_transition:
            raise ValueError("joint curriculum evaluation crossing is invalid")
        self.transitions = crossing
        self.last_evaluation_transition = crossing
        self.passes = self.passes + 1 if passed else 0
        self.rollbacks = self.rollbacks + 1 if rollback else 0
        promoted = False
        rolled_back = False
        if self.stage != "C3" and self.passes >= 3:
            self.stage = "C2" if self.stage == "C1" else "C3"
            self.passes = 0
            self.rollbacks = 0
            promoted = True
        elif self.stage == "C2" and self.rollbacks >= 2:
            self.stage = "C1"
            self.passes = 0
            self.rollbacks = 0
            rolled_back = True
        return CurriculumDecision(self.stage, promoted, rolled_back, crossing)

    def evaluate_window(
        self,
        *,
        crossing: int,
        task_metrics: Mapping[str, Mapping[str, object]],
        actual_transitions: int | None = None,
    ) -> CurriculumDecision:
        """Apply learning-acceptance eligibility, promotion, and adjacent rollback."""
        crossing = _positive_int(crossing, "crossing")
        if crossing % 250000 or crossing <= self.last_evaluation_transition:
            raise ValueError("learning evaluation crossing is invalid")
        if actual_transitions is None:
            actual_transitions = crossing
        actual_transitions = _positive_int(actual_transitions, "actual_transitions")
        if actual_transitions < crossing:
            raise ValueError("curriculum actual transitions precede logical crossing")
        passed = _window_passes(self.stage, task_metrics)
        rollback = _window_rolls_back(self.stage, task_metrics)
        self.transitions = actual_transitions
        self.last_evaluation_transition = crossing
        if not passed:
            self.passes = 0
        elif self.stage != "C3" and (
            (self.stage == "C1" and crossing >= 5000000)
            or (self.stage == "C2" and crossing >= 10000000)
        ):
            self.passes += 1
        else:
            self.passes = 0
        if rollback and self.stage != "C1":
            self.rollbacks += 1
        else:
            self.rollbacks = 0
        promoted = False
        rolled_back = False
        if self.stage == "C1" and self.passes >= 3:
            self.stage = "C2"
            self.passes = 0
            self.rollbacks = 0
            promoted = True
        elif self.stage == "C2" and self.passes >= 3:
            self.stage = "C3"
            self.passes = 0
            self.rollbacks = 0
            promoted = True
        elif self.stage == "C2" and self.rollbacks >= 2:
            self.stage = "C1"
            self.passes = 0
            self.rollbacks = 0
            rolled_back = True
        elif self.stage == "C3" and self.rollbacks >= 2:
            self.stage = "C2"
            self.passes = 0
            self.rollbacks = 0
            rolled_back = True
        return CurriculumDecision(self.stage, promoted, rolled_back, crossing)


def broadcast_curriculum_state(
    curriculum: JointCurriculum,
    *,
    rank: int,
    device: torch.device | str,
    process_group: Any | None = None,
) -> JointCurriculum:
    if not distributed.is_initialized():
        raise RuntimeError("curriculum broadcast requires an initialized process group")
    payload: list[object] = [curriculum.state_dict() if rank == 0 else None]
    distributed.broadcast_object_list(
        payload, src=0, group=process_group, device=device
    )
    if not isinstance(payload[0], Mapping):
        raise ValueError("rank-zero curriculum broadcast is invalid")
    restored = JointCurriculum()
    restored.load_state_dict(payload[0])
    return restored


class GradientAverager:
    """All-reduce every PPO gradient after backward and before clipping."""

    def __init__(
        self,
        *,
        world_size: int,
        allow_test_backend: bool = False,
        process_group: Any | None = None,
    ) -> None:
        self.world_size = _positive_int(world_size, "world_size")
        self.allow_test_backend = allow_test_backend
        self.process_group = process_group

    def __call__(self, policy: nn.Module) -> None:
        if not distributed.is_initialized():
            raise RuntimeError("gradient synchronization requires a process group")
        backend = distributed.get_backend(self.process_group)
        if backend != "nccl" and not self.allow_test_backend:
            raise RuntimeError(
                "Phase 6 production gradient synchronization requires NCCL"
            )
        if distributed.get_world_size(self.process_group) != self.world_size:
            raise RuntimeError(
                "gradient synchronization world size differs from contract"
            )
        for name, parameter in policy.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"gradient is missing for {name}")
            if (
                gradient.shape != parameter.shape
                or gradient.dtype != parameter.dtype
                or gradient.device != parameter.device
            ):
                raise RuntimeError(
                    f"gradient shape, dtype, or device mismatch for {name}"
                )
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"non-finite gradient for {name}")
            distributed.all_reduce(
                gradient, op=distributed.ReduceOp.SUM, group=self.process_group
            )
            gradient.div_(self.world_size)


def _hash_value(value: object, digest: hashlib._Hash) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=str):
            _hash_value(str(key), digest)
            _hash_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _hash_value(item, digest)
    elif value is None:
        digest.update(b"none")
    else:
        digest.update(repr(value).encode("utf-8"))


def state_dict_sha256(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    _hash_value(state, digest)
    return digest.hexdigest()


def optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    steps: set[int] = set()
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError("optimizer step tensor is invalid")
            steps.add(int(value.item()))
        elif isinstance(value, int) and not isinstance(value, bool):
            steps.add(value)
        else:
            raise ValueError("optimizer step is invalid")
    if not steps:
        return 0
    if len(steps) != 1 or next(iter(steps)) < 0:
        raise ValueError("optimizer parameter steps differ")
    return next(iter(steps))


def assert_matching_hash_records(records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        raise ValueError("synchronization records are empty")
    expected = _exact_keys(records[0], "hash_record", {"optimizer", "policy", "step"})
    for index, record in enumerate(records[1:], start=1):
        if _exact_keys(record, f"hash_record[{index}]", set(expected)) != expected:
            raise RuntimeError("rank policy, optimizer, or step hashes diverged")


def synchronize_iteration_hashes(
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    process_group: Any | None = None,
) -> dict[str, object]:
    if not distributed.is_initialized():
        raise RuntimeError("iteration hash synchronization requires a process group")
    record = {
        "policy": state_dict_sha256(policy.state_dict()),
        "optimizer": state_dict_sha256(optimizer.state_dict()),
        "step": optimizer_step(optimizer),
    }
    encoded = bytes.fromhex(record["policy"] + record["optimizer"]) + int(
        record["step"]
    ).to_bytes(8, "big", signed=False)
    local = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
    gathered = [
        torch.empty_like(local)
        for _ in range(distributed.get_world_size(process_group))
    ]
    distributed.all_gather(gathered, local, group=process_group)
    records = [
        {
            "policy": bytes(item[:32].cpu().tolist()).hex(),
            "optimizer": bytes(item[32:64].cpu().tolist()).hex(),
            "step": int.from_bytes(bytes(item[64:].cpu().tolist()), "big"),
        }
        for item in gathered
    ]
    assert_matching_hash_records(records)
    return record


def load_phase5_policy_only(
    policy: ResidualActorCritic,
    checkpoint: str | Path,
    *,
    expected_sha256: str = _INITIAL_CHECKPOINT["sha256"],
    device: torch.device | str = "cpu",
) -> None:
    source = Path(checkpoint)
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError("Phase 5 initialization checkpoint checksum mismatch")
    value = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(value, Mapping) or "policy" not in value:
        raise ValueError("Phase 5 initialization checkpoint has no policy state")
    policy.load_state_dict(value["policy"], strict=True)


def rank_rng_state() -> dict[str, object]:
    return {
        "cpu": torch.get_rng_state().cpu(),
        "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def validate_rank_rng_states(
    rank_rng: Mapping[str, object], *, cuda_state_count: int = 4
) -> None:
    """Validate the exact four-rank RNG checkpoint boundary without restoring it."""
    if cuda_state_count <= 0:
        raise ValueError("CUDA RNG state count must be positive")
    expected_ranks = {str(rank) for rank in range(4)}
    if set(rank_rng) != expected_ranks:
        raise ValueError("learning checkpoint must persist RNG state for ranks 0..3")
    for rank in range(4):
        state = _exact_keys(rank_rng[str(rank)], f"rank_rng.{rank}", {"cpu", "cuda"})
        cpu = state["cpu"]
        if (
            not isinstance(cpu, torch.Tensor)
            or cpu.dtype != torch.uint8
            or cpu.ndim != 1
            or cpu.numel() <= 0
        ):
            raise ValueError(f"rank_rng.{rank}.cpu must be a nonempty uint8 state")
        cuda = state["cuda"]
        if not isinstance(cuda, list) or len(cuda) != cuda_state_count:
            raise ValueError(
                f"rank_rng.{rank}.cuda must contain {cuda_state_count} device states"
            )
        for device_index, device_state in enumerate(cuda):
            if (
                not isinstance(device_state, torch.Tensor)
                or device_state.dtype != torch.uint8
                or device_state.ndim != 1
                or device_state.numel() <= 0
            ):
                raise ValueError(
                    f"rank_rng.{rank}.cuda[{device_index}] must be a nonempty uint8 state"
                )


def restore_rank_rng_state(
    rank_rng: Mapping[str, object],
    *,
    rank: int,
    local_device_index: int,
    cuda_state_count: int = 4,
) -> None:
    """Restore one rank's CPU and local CUDA RNG before stochastic PPO work."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in range(4):
        raise ValueError("rank RNG restore requires a rank in 0..3")
    if (
        isinstance(local_device_index, bool)
        or not isinstance(local_device_index, int)
        or local_device_index < 0
        or local_device_index >= cuda_state_count
    ):
        raise ValueError("rank RNG restore local CUDA device index is invalid")
    validate_rank_rng_states(rank_rng, cuda_state_count=cuda_state_count)
    state = _exact_keys(rank_rng[str(rank)], f"rank_rng.{rank}", {"cpu", "cuda"})
    cpu_state = state["cpu"]
    cuda_states = state["cuda"]
    assert isinstance(cpu_state, torch.Tensor)
    assert isinstance(cuda_states, list)
    torch.set_rng_state(cpu_state.detach().cpu().contiguous())
    if not torch.cuda.is_available() or torch.cuda.device_count() != cuda_state_count:
        raise RuntimeError(
            "rank RNG restore requires exactly the contracted CUDA devices"
        )
    cuda_state = cuda_states[local_device_index]
    assert isinstance(cuda_state, torch.Tensor)
    torch.cuda.set_rng_state(
        cuda_state.detach().cpu().contiguous(), device=local_device_index
    )


def _checkpoint_contracts(
    config: Phase6Config, roster: Phase6Roster
) -> dict[str, object]:
    return {
        "phase6": {
            "canonical_sha256": config.canonical_sha256,
            "contract_version": PHASE6_CONTRACT_VERSION,
            "raw_sha256": config.raw_sha256,
        },
        "phase5_bindings": dict(config.payload["bindings"]),
        "roster": {
            "artifact_sha256sums": {
                task.task: task.artifact_sha256sums_sha256 for task in roster.tasks
            },
            "canonical_sha256": roster.canonical_sha256,
            "raw_sha256": roster.raw_sha256,
            "version": PHASE6_ROSTER_VERSION,
        },
    }


def joint_checkpoint_payload(
    *,
    config: Phase6Config,
    roster: Phase6Roster,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    cycle_index: int,
    window_index: int,
    slot_transitions: int,
    task_transitions: Mapping[str, int],
    curriculum: JointCurriculum,
    rank_rng: Mapping[str, object],
    metrics: Mapping[str, object],
    hashes: Mapping[str, object],
) -> dict[str, object]:
    scheduler = JointTaskScheduler(
        task_count=len(roster.tasks),
        world_size=int(config.payload["runtime"]["world_size"]),
    )
    expected = scheduler.expected_task_transitions(
        cycle_index=cycle_index,
        window_index=window_index,
        slot_transitions=slot_transitions,
    )
    expected_named = {
        roster.tasks[index].task: value for index, value in expected.items()
    }
    if dict(task_transitions) != expected_named:
        raise ValueError(
            "checkpoint task transitions do not match completed schedule slots"
        )
    assert_transition_balance(
        task_transitions, expected_total=sum(expected_named.values())
    )
    validated_metrics = validate_phase6_joint_metrics(
        metrics,
        config=config,
        roster=roster,
        expected_global_transitions=sum(expected_named.values()),
    )
    if set(rank_rng) != {str(rank) for rank in range(scheduler.world_size)}:
        raise ValueError("checkpoint must persist every rank RNG state")
    hashes = _exact_keys(hashes, "hashes", {"optimizer", "policy", "step"})
    if hashes["step"] != optimizer_step(optimizer):
        raise ValueError("checkpoint optimizer step differs from synchronized state")
    return {
        "checkpoint_version": PHASE6_CHECKPOINT_VERSION,
        "contracts": _checkpoint_contracts(config, roster),
        "curriculum": curriculum.state_dict(),
        "iteration": _positive_int(iteration, "iteration"),
        "metrics": validated_metrics,
        "optimizer": optimizer.state_dict(),
        "optimizer_sha256": hashes["optimizer"],
        "optimizer_step": hashes["step"],
        "policy": policy.state_dict(),
        "policy_sha256": hashes["policy"],
        "rank_rng": dict(rank_rng),
        "schedule": {
            "cycle_index": cycle_index,
            "formula": config.payload["scheduler"]["schedule_formula"],
            "slot_transitions": slot_transitions,
            "window_index": window_index,
        },
        "task_transitions": dict(task_transitions),
    }


def atomic_torch_save(path: str | Path, payload: Mapping[str, object]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)
    return sha256_file(target)


def restore_joint_checkpoint(
    path: str | Path,
    *,
    config: Phase6Config,
    roster: Phase6Roster,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_cycle_index: int,
    expected_window_index: int,
    device: torch.device | str,
) -> dict[str, object]:
    value = torch.load(Path(path), map_location=device, weights_only=False)
    required = {
        "checkpoint_version",
        "contracts",
        "curriculum",
        "iteration",
        "metrics",
        "optimizer",
        "optimizer_sha256",
        "optimizer_step",
        "policy",
        "policy_sha256",
        "rank_rng",
        "schedule",
        "task_transitions",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("joint checkpoint schema is not strict")
    if value["checkpoint_version"] != PHASE6_CHECKPOINT_VERSION:
        raise ValueError("joint checkpoint version mismatch")
    if value["contracts"] != _checkpoint_contracts(config, roster):
        raise ValueError("joint checkpoint contract or roster binding mismatch")
    schedule = _exact_keys(
        value["schedule"],
        "checkpoint.schedule",
        {"cycle_index", "formula", "slot_transitions", "window_index"},
    )
    scheduler = JointTaskScheduler(
        task_count=len(roster.tasks),
        world_size=int(config.payload["runtime"]["world_size"]),
    )
    next_cycle, next_window = scheduler.next_slot(
        cycle_index=_nonnegative_int(schedule["cycle_index"], "cycle_index"),
        window_index=_nonnegative_int(schedule["window_index"], "window_index"),
    )
    if (expected_cycle_index, expected_window_index) != (next_cycle, next_window):
        raise ValueError("joint resume rejects repeated or skipped scheduler windows")
    expected = scheduler.expected_task_transitions(
        cycle_index=int(schedule["cycle_index"]),
        window_index=int(schedule["window_index"]),
        slot_transitions=_positive_int(
            schedule["slot_transitions"], "slot_transitions"
        ),
    )
    expected_named = {
        roster.tasks[index].task: count for index, count in expected.items()
    }
    if value["task_transitions"] != expected_named:
        raise ValueError("joint resume rejects unbalanced task transition counters")
    validate_phase6_joint_metrics(
        value["metrics"],
        config=config,
        roster=roster,
        expected_global_transitions=sum(expected_named.values()),
    )
    policy.load_state_dict(value["policy"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    if state_dict_sha256(policy.state_dict()) != value["policy_sha256"]:
        raise ValueError("joint checkpoint policy hash mismatch")
    if state_dict_sha256(optimizer.state_dict()) != value["optimizer_sha256"]:
        raise ValueError("joint checkpoint optimizer hash mismatch")
    if optimizer_step(optimizer) != value["optimizer_step"]:
        raise ValueError("joint checkpoint optimizer step mismatch")
    restored = JointCurriculum()
    restored.load_state_dict(_mapping(value["curriculum"], "checkpoint.curriculum"))
    if restored.transitions != sum(expected_named.values()):
        raise ValueError("joint checkpoint curriculum transitions are inconsistent")
    return dict(value)


def process_group_timeout(config: Phase6Config) -> timedelta:
    return timedelta(seconds=int(config.payload["runtime"]["process_group_timeout_s"]))


def _learning_contracts(
    config: Phase6Config,
    roster: Phase6Roster,
    acceptance: LearningAcceptanceConfig,
    phase5_checkpoint_sha256: str,
) -> dict[str, object]:
    if phase5_checkpoint_sha256 != _LEARNING_PHASE5_CHECKPOINT_SHA256:
        raise ValueError(
            "learning checkpoint must bind the admitted Phase 5 checkpoint"
        )
    return {
        "phase6": {
            "canonical_sha256": config.canonical_sha256,
            "raw_sha256": config.raw_sha256,
            "contract_version": PHASE6_CONTRACT_VERSION,
        },
        "roster": {
            "canonical_sha256": roster.canonical_sha256,
            "raw_sha256": roster.raw_sha256,
            "contract_version": PHASE6_ROSTER_VERSION,
        },
        "learning_acceptance": {
            "canonical_sha256": acceptance.canonical_sha256,
            "raw_sha256": acceptance.raw_sha256,
            "contract_version": PHASE6_LEARNING_ACCEPTANCE_VERSION,
        },
        "phase5_checkpoint_sha256": phase5_checkpoint_sha256,
    }


def learning_checkpoint_payload(
    *,
    config: Phase6Config,
    roster: Phase6Roster,
    acceptance: LearningAcceptanceConfig,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    task_transitions: Mapping[str, int],
    curriculum: JointCurriculum,
    next_crossing: Mapping[str, object] | None,
    evaluation_history: Sequence[Mapping[str, object]],
    rank_rng: Mapping[str, object],
    source_manifest: Mapping[str, str],
    metrics: Mapping[str, object],
    pre_evaluation: bool,
    phase5_checkpoint_sha256: str = _LEARNING_PHASE5_CHECKPOINT_SHA256,
) -> dict[str, object]:
    iteration = _positive_int(iteration, "learning checkpoint.iteration")
    arithmetic = learning_transition_arithmetic(iterations=iteration)
    expected_tasks = {
        task.task: arithmetic["per_task_transitions"] for task in roster.tasks
    }
    if dict(task_transitions) != expected_tasks:
        raise ValueError(
            "learning checkpoint task transitions are not exactly balanced"
        )
    validate_rank_rng_states(rank_rng)
    source_manifest = _exact_keys(
        source_manifest, "learning checkpoint.source_manifest", set(source_manifest)
    )
    if not source_manifest or any(
        not isinstance(value, str) or len(value) != 64
        for value in source_manifest.values()
    ):
        raise ValueError(
            "learning checkpoint source manifest must contain SHA256 values"
        )
    if not isinstance(pre_evaluation, bool):
        raise TypeError("learning checkpoint pre_evaluation must be bool")
    hashes = {
        "policy": state_dict_sha256(policy.state_dict()),
        "optimizer": state_dict_sha256(optimizer.state_dict()),
        "step": optimizer_step(optimizer),
    }
    if hashes["step"] != arithmetic["optimizer_steps"]:
        raise ValueError("learning checkpoint optimizer step arithmetic is invalid")
    return {
        "checkpoint_version": PHASE6_LEARNING_CHECKPOINT_VERSION,
        "contracts": _learning_contracts(
            config, roster, acceptance, phase5_checkpoint_sha256
        ),
        "evaluation_history": [dict(item) for item in evaluation_history],
        "iteration": iteration,
        "metrics": dict(metrics),
        "next_crossing": None if next_crossing is None else dict(next_crossing),
        "optimizer": optimizer.state_dict(),
        "optimizer_sha256": hashes["optimizer"],
        "optimizer_step": hashes["step"],
        "policy": policy.state_dict(),
        "policy_sha256": hashes["policy"],
        "pre_evaluation": pre_evaluation,
        "rank_rng": dict(rank_rng),
        "source_manifest": dict(source_manifest),
        "task_transitions": dict(task_transitions),
        "actual_global_transitions": arithmetic["global_transitions"],
        "actual_per_task_transitions": arithmetic["per_task_transitions"],
        "curriculum": curriculum.state_dict(),
    }


def atomic_learning_checkpoint(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    latest_path: str | Path | None = None,
) -> str:
    if payload.get("checkpoint_version") != PHASE6_LEARNING_CHECKPOINT_VERSION:
        raise ValueError("only phase6_learning_checkpoint_v1 may be written")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"learning checkpoint already exists: {target}")
    if latest_path is not None and Path(latest_path).exists():
        raise FileExistsError(f"learning latest pointer already exists: {latest_path}")
    digest = atomic_torch_save(path, payload)
    if latest_path is not None and not payload.get("pre_evaluation", True):
        latest = Path(latest_path)
        latest.parent.mkdir(parents=True, exist_ok=True)
        temporary = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {"checkpoint": str(Path(path)), "sha256": digest}, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, latest)
    return digest


def restore_learning_checkpoint(
    path: str | Path,
    *,
    config: Phase6Config,
    roster: Phase6Roster,
    acceptance: LearningAcceptanceConfig,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_manifest: Mapping[str, str],
    expected_iteration: int | None = None,
    expected_next_crossing: Mapping[str, object] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    value = torch.load(Path(path), map_location=device, weights_only=False)
    required = {
        "actual_global_transitions",
        "actual_per_task_transitions",
        "checkpoint_version",
        "contracts",
        "curriculum",
        "evaluation_history",
        "iteration",
        "metrics",
        "next_crossing",
        "optimizer",
        "optimizer_sha256",
        "optimizer_step",
        "policy",
        "policy_sha256",
        "pre_evaluation",
        "rank_rng",
        "source_manifest",
        "task_transitions",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("learning checkpoint schema is not strict")
    if value["checkpoint_version"] != PHASE6_LEARNING_CHECKPOINT_VERSION:
        raise ValueError("learning checkpoint version mismatch")
    if value["contracts"] != _learning_contracts(
        config, roster, acceptance, _LEARNING_PHASE5_CHECKPOINT_SHA256
    ):
        raise ValueError("learning checkpoint contract binding mismatch")
    if dict(value["source_manifest"]) != dict(source_manifest):
        raise ValueError("learning checkpoint source manifest mismatch")
    iteration = _positive_int(value["iteration"], "checkpoint.iteration")
    if expected_iteration is not None and iteration != _positive_int(
        expected_iteration, "expected_iteration"
    ):
        raise ValueError("learning resume iteration mismatch")
    arithmetic = learning_transition_arithmetic(iterations=iteration)
    if (
        value["actual_global_transitions"] != arithmetic["global_transitions"]
        or value["actual_per_task_transitions"] != arithmetic["per_task_transitions"]
    ):
        raise ValueError("learning checkpoint transition arithmetic mismatch")
    expected_tasks = {
        task.task: arithmetic["per_task_transitions"] for task in roster.tasks
    }
    if value["task_transitions"] != expected_tasks:
        raise ValueError("learning checkpoint task balance mismatch")
    validate_rank_rng_states(value["rank_rng"])
    if expected_next_crossing is not None and value["next_crossing"] != dict(
        expected_next_crossing
    ):
        raise ValueError("learning resume crossing mismatch")
    policy.load_state_dict(value["policy"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    if (
        state_dict_sha256(policy.state_dict()) != value["policy_sha256"]
        or state_dict_sha256(optimizer.state_dict()) != value["optimizer_sha256"]
    ):
        raise ValueError("learning checkpoint policy or optimizer hash mismatch")
    if optimizer_step(optimizer) != value["optimizer_step"]:
        raise ValueError("learning checkpoint optimizer step mismatch")
    restored = JointCurriculum()
    restored.load_state_dict(_mapping(value["curriculum"], "checkpoint.curriculum"))
    if restored.transitions != arithmetic["global_transitions"]:
        raise ValueError("learning checkpoint curriculum transition mismatch")
    return dict(value)


def validate_learning_evaluation_summary(
    summary: Mapping[str, object],
    *,
    roster: Phase6Roster,
    expected_iteration: int,
    expected_stage_quota: int = 256,
    expected_nominal_quota: int = 128,
) -> dict[str, object]:
    """Require all rank/episode close evidence before curriculum mutation."""
    if isinstance(summary, Mapping) and "schedule_sha256" not in summary:
        root = _exact_keys(
            summary,
            "learning.evaluation_summary.legacy",
            {"iteration", "rank_results", "status", "task_metrics"},
        )
        if root["status"] != "ok" or root["iteration"] != expected_iteration:
            raise ValueError(
                "learning evaluation summary status or iteration is incomplete"
            )
        results = root["rank_results"]
        if not isinstance(results, list) or len(results) != 4:
            raise ValueError("learning evaluation requires four rank results")
        seen: set[str] = set()
        for index, result in enumerate(results):
            row = _exact_keys(
                result,
                f"legacy.rank_results[{index}]",
                {"episodes", "rank", "status", "task"},
            )
            if (
                row["rank"] != index
                or row["status"] != "ok"
                or row["task"] in seen
                or row["task"] not in {task.task for task in roster.tasks}
                or row["episodes"]
                != (expected_stage_quota + expected_nominal_quota) * 2
            ):
                raise ValueError("learning evaluation episode quota is incomplete")
            seen.add(str(row["task"]))
        if seen != {task.task for task in roster.tasks} or not isinstance(
            root["task_metrics"], Mapping
        ):
            raise ValueError("learning evaluation did not cover every roster task")
        return {
            "rank_results": results,
            "task_metrics": dict(root["task_metrics"]),
            "iteration": expected_iteration,
            "status": "ok",
        }
    root = _exact_keys(
        summary,
        "learning.evaluation_summary",
        {"iteration", "rank_results", "schedule_sha256", "status", "task_metrics"},
    )
    if root["status"] != "ok" or root["iteration"] != expected_iteration:
        raise ValueError(
            "learning evaluation summary status or iteration is incomplete"
        )
    results = root["rank_results"]
    if not isinstance(results, list) or len(results) != 4:
        raise ValueError("learning evaluation requires four rank results")
    seen: set[str] = set()
    schedule_sha = root["schedule_sha256"]
    if not isinstance(schedule_sha, str) or len(schedule_sha) != 64:
        raise ValueError("learning evaluation schedule hash is invalid")
    rank_schedule_hashes: list[str] = []
    for index, result in enumerate(results):
        row = _exact_keys(
            result,
            f"rank_results[{index}]",
            {
                "control_steps",
                "episodes",
                "mode_counts",
                "rank",
                "schedule_sha256",
                "status",
                "subset_counts",
                "task",
                "unique_seeds",
            },
        )
        if row["rank"] != index or row["status"] != "ok" or row["task"] in seen:
            raise ValueError("learning evaluation rank result is incomplete")
        if (
            row["task"] not in {task.task for task in roster.tasks}
            or row["episodes"] != (expected_stage_quota + expected_nominal_quota) * 2
            or not isinstance(row["schedule_sha256"], str)
            or len(row["schedule_sha256"]) != 64
            or row["unique_seeds"] != expected_stage_quota + expected_nominal_quota
            or row["mode_counts"]
            != {
                "residual": expected_stage_quota + expected_nominal_quota,
                "scaffold_only": expected_stage_quota + expected_nominal_quota,
            }
            or row["subset_counts"]
            != {
                "nominal": expected_nominal_quota * 2,
                "stage": expected_stage_quota * 2,
            }
        ):
            raise ValueError("learning evaluation episode quota is incomplete")
        if (
            isinstance(row["control_steps"], bool)
            or not isinstance(row["control_steps"], int)
            or row["control_steps"] <= 0
        ):
            raise ValueError("learning evaluation control-step evidence is incomplete")
        rank_schedule_hashes.append(str(row["schedule_sha256"]))
        seen.add(str(row["task"]))
    aggregate = hashlib.sha256(
        json.dumps(rank_schedule_hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if schedule_sha != aggregate:
        raise ValueError("learning evaluation aggregate schedule hash is invalid")
    if seen != {task.task for task in roster.tasks} or not isinstance(
        root["task_metrics"], Mapping
    ):
        raise ValueError("learning evaluation did not cover every roster task")
    return {
        "rank_results": results,
        "schedule_sha256": schedule_sha,
        "task_metrics": dict(root["task_metrics"]),
        "iteration": expected_iteration,
        "status": "ok",
    }


def validate_learning_final_acceptance(
    metrics: Mapping[str, object], *, config: LearningAcceptanceConfig
) -> dict[str, bool]:
    """Validate C3 closeout, equal-task accounting, retention and two benefits."""
    final = _mapping(config.payload["final_acceptance"], "final_acceptance")
    if (
        metrics.get("stage") != final["final_stage"]
        or metrics.get("iteration") != final["final_iteration"]
    ):
        raise AssertionError("Phase 6 final stage or iteration gate failed")
    arithmetic = learning_transition_arithmetic(iterations=2442)
    if (
        metrics.get("global_transitions") != arithmetic["global_transitions"]
        or metrics.get("optimizer_steps") != arithmetic["optimizer_steps"]
    ):
        raise AssertionError("Phase 6 final arithmetic gate failed")
    tasks = _mapping(metrics.get("tasks"), "final.tasks")
    if set(tasks) != set(_CURRENT_TASKS):
        raise ValueError("Phase 6 final metrics do not cover exactly four tasks")
    task_fractions: list[float] = []
    benefits: list[Mapping[str, object]] = []
    families: set[str] = set()
    reward_shares: list[float] = []
    semantic_shares: list[float] = []
    for task, row_value in tasks.items():
        row = _mapping(row_value, f"final.tasks.{task}")
        fraction = _finite(
            row.get("transition_fraction"), f"{task}.transition_fraction"
        )
        task_fractions.append(fraction)
        if not math.isclose(
            fraction,
            float(final["task_transition_fraction"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise AssertionError("Phase 6 task transition fraction is not exactly 0.25")
        if (
            _finite(row.get("nominal_retention"), f"{task}.nominal_retention")
            < float(final["nominal_retention_min"])
            or int(row.get("nominal_invalid", -1)) != int(final["nominal_invalid_max"])
            or _finite(row.get("nominal_saturation"), f"{task}.nominal_saturation")
            > float(final["nominal_saturation_max"])
        ):
            raise AssertionError(f"Phase 6 nominal gate failed for {task}")
        if _finite(row.get("residual_mean_norm"), f"{task}.residual_mean_norm") < float(
            final["residual_mean_norm_min"]
        ) or _finite(
            row.get("contact_bearing_residual_fraction"),
            f"{task}.contact_bearing_residual_fraction",
        ) < float(final["contact_bearing_residual_fraction_min"]):
            raise AssertionError(f"Phase 6 residual gate failed for {task}")
        reward_shares.append(
            _finite(
                row.get("normalized_reward_share"), f"{task}.normalized_reward_share"
            )
        )
        semantic_shares.append(
            _finite(row.get("semantic_total_share"), f"{task}.semantic_total_share")
        )
        benefit = row.get("benefit")
        if isinstance(benefit, Mapping) and bool(benefit.get("passes")):
            benefits.append(benefit)
            families.add(str(benefit.get("mismatch_family")))
    if max(reward_shares) > float(final["task_normalized_reward_share_max"]) or max(
        semantic_shares
    ) > float(final["task_semantic_total_share_max"]):
        raise AssertionError("Phase 6 task dominance share gate failed")
    if len(benefits) < int(final["benefit_task_count_min"]) or len(families) < int(
        final["benefit_mismatch_family_count_min"]
    ):
        raise AssertionError(
            "Phase 6 requires two tasks and two mismatch families with benefit"
        )
    for benefit in benefits:
        success_delta = _finite(
            benefit.get("success_delta", 0.0), "benefit.success_delta"
        )
        progress_delta = _finite(
            benefit.get("progress_delta", 0.0), "benefit.progress_delta"
        )
        force_ratio = _finite(
            benefit.get("force_p95_ratio", 1.0), "benefit.force_p95_ratio"
        )
        if not (success_delta >= 0.05 or progress_delta >= 0.05 or force_ratio <= 0.95):
            raise AssertionError("Phase 6 residual benefit threshold failed")
        if _finite(
            benefit.get("stability_degradation", 1.0), "benefit.stability_degradation"
        ) > float(final["stability_degradation_max"]):
            raise AssertionError("Phase 6 stability nondominance gate failed")
    return {
        "c3": True,
        "arithmetic": True,
        "retention": True,
        "nondominance": True,
        "two_benefits": True,
    }
