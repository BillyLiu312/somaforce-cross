#!/usr/bin/env python3
"""Bounded four-rank Phase 6 joint-training worker and evidence wrapper."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_path = str(REPO_ROOT)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

PHASE6_CONFIG = REPO_ROOT / "configs/phase6_joint_training_v1.json"
PHASE6_ROSTER = REPO_ROOT / "configs/phase6_task_roster_v1.json"
PHASE6_LEARNING_CONFIG = REPO_ROOT / "configs/phase6_learning_acceptance_v1.json"
RESULT_MARKER = "PHASE6_JOINT_RESULT="
LEARNING_RESULT_MARKER = "PHASE6_LEARNING_RESULT="
PROGRESS_MARKERS = ("JSON_WRITTEN", RESULT_MARKER, "ENV_CLOSED")
ISAAC_PYTHON = "/opt/miniconda3/envs/isaaclab/bin/python"
LEGACY_PRE_EVALUATION_CHECKPOINT = (
    REPO_ROOT
    / "outputs/phase6_learning_acceptance"
    / "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940"
    / "segment_0000/train/pre_evaluation.pt"
)
LEGACY_PRE_EVALUATION_SHA256 = (
    "26abccc4735722bd2feee7c2a683536212c5385d326caab52715fc75a72f69c0"
)
WORKER_CRITICAL_FUNCTIONS = (
    "_worker_main",
    "_build_environment",
    "_new_algorithm",
    "_mean_iteration_value",
    "_local_metrics",
    "_merge_rank_metrics",
    "_run_paired_evaluation_rank",
)
EVALUATION_DEVICEFIX_FUNCTION = "_run_paired_evaluation_rank"
EVALUATION_DEVICEFIX_SNAPSHOT_SHA256 = (
    "03ce3112971522194f75226b4a1143f9cb36193d191c73a3f5c80f593c2b19b9"
)
EVALUATION_DEVICEFIX_INPUT_SHA256 = (
    "9fc413f84afcbc4c669d3b7c5c4cfc34b7a77d6ae160b7516b5a73023673ef9f"
)
STEPBOUNDARYFIX_SNAPSHOT_SHA256 = (
    "a190e822d09e3315b59f1c52df11cecd58905f9b5d66a67ad6c039f67185d487"
)
STEPBOUNDARYFIX_INPUT_SHA256 = (
    "598ebe961594e70cabbf3379d947a4893ae8df01d9efc2b4833cbb108668e1cb"
)
PAIREDBATCH_INPUT_SHA256 = (
    "f81a54e9cae84ba4a00c96320ac171bb8706d2b6db912ac9b42e1177989c8cb6"
)
PAIREDBATCH_TRAIN_SNAPSHOT_SHA256 = (
    "dea0ff25ab7515ea5b4aade3a6681f98f6470a8776f4bbbf15946670864019fc"
)
PAIREDBATCH_JOINT_SNAPSHOT_SHA256 = (
    "3168196c654dc8773cad8630eb49362f2d0bb6b678de43c578552949f202be0e"
)
PAIREDBATCH_ENV_SNAPSHOT_SHA256 = (
    "f44444d1c37c4d6f7594f5b246c91a4453e906471cb22bcd6921e961336df98f"
)
RANKSEED_INPUT_SHA256 = (
    "1467b99946b427092aa94faa1713137a5d02745f5dc38b283d10fa64f2d48388"
)
RANKSEED_TRAIN_SNAPSHOT_SHA256 = (
    "90c3c12c8eb0b5657be888e954544b8543fcd35c356c42e86bce57f39b4926bf"
)
RANKSEED_JOINT_SNAPSHOT_SHA256 = (
    "e3de1ab066ac39b57875a8b5b59357e74f50d7f2f516caee7eb88b6f258c7290"
)
RANKSEED_ENV_SNAPSHOT_SHA256 = (
    "6a1cde838dee627d93b2eb2560906eb6638cf0e0ec60b6e892181336711d764a"
)
EPISODEORDER_INPUT_SHA256 = (
    "468aa578faa8ce6968db4061f0173d9fc9101846bfc7100b51d2ff20a249bbd4"
)
EPISODEORDER_TRAIN_SNAPSHOT_SHA256 = (
    "0392bd63128c96121f2ed4d15a98e5eeef8213b59ad9a9ae32a655d5f2c23833"
)
EPISODEORDER_JOINT_SNAPSHOT_SHA256 = (
    "78b95fb4cc5148dbf68a69e4fc68eaf5713dd2e624975a053618a4585a1e6e65"
)
EPISODEORDER_ENV_SNAPSHOT_SHA256 = (
    "6a1cde838dee627d93b2eb2560906eb6638cf0e0ec60b6e892181336711d764a"
)
MODEPROCESS_INPUT_SHA256 = (
    "1952b9844633ea950d177267ad6bd685451339a429c38d4b348d00457acd0732"
)
MODEPROCESS_TRAIN_SNAPSHOT_SHA256 = (
    "2c4322e6a7db2b2dd7bd430f179126ed4d0fefc22d13f89404e47be99545a6c9"
)
MODEPROCESS_JOINT_SNAPSHOT_SHA256 = (
    "78b95fb4cc5148dbf68a69e4fc68eaf5713dd2e624975a053618a4585a1e6e65"
)
MODEPROCESS_ENV_SNAPSHOT_SHA256 = (
    "86aa99b303290a3b746af65936baf93b93fcb00ae99ce6632721df997185293f"
)
WARNINGPOLICY_INPUT_SHA256 = (
    "0359129f3ae5de92d1cdea4d1c42ae12b69fe2d03999705d9d68a00c47fb1b92"
)
WARNINGPOLICY_TRAIN_SNAPSHOT_SHA256 = (
    "f63931eeb92c254495d871d7b38960b3675a13ffb1f1c53afa407f8b76f3d25f"
)
WARNINGPOLICY_JOINT_SNAPSHOT_SHA256 = (
    "78b95fb4cc5148dbf68a69e4fc68eaf5713dd2e624975a053618a4585a1e6e65"
)
WARNINGPOLICY_ENV_SNAPSHOT_SHA256 = (
    "86aa99b303290a3b746af65936baf93b93fcb00ae99ce6632721df997185293f"
)
SUMMARYSCHEMA_INPUT_SHA256 = (
    "bf61602f13e3049de51fb1c05f390d879e84e3393e7a063cd386f733987d3e65"
)
SUMMARYSCHEMA_INPUT_RECORD_SHA256 = (
    "60d13ecf71a5b8d5fc7055c0faf6077b1fad68b00406838b1fe75e910725a171"
)
SUMMARYSCHEMA_TRAIN_SNAPSHOT_SHA256 = (
    "1d37f7dcb9d4ffe6d42d03dee207377058639bf07058d83d43e8557695e0a0c8"
)
SUMMARYSCHEMA_JOINT_SNAPSHOT_SHA256 = (
    "78b95fb4cc5148dbf68a69e4fc68eaf5713dd2e624975a053618a4585a1e6e65"
)
SUMMARYSCHEMA_ENV_SNAPSHOT_SHA256 = (
    "86aa99b303290a3b746af65936baf93b93fcb00ae99ce6632721df997185293f"
)
SUMMARYSCHEMA_FAILED_FINALIZE_MANIFEST_SHA256 = (
    "85ed76182518a3ed476091447790a05b9a96da1d8891bef8cdc5d23e4f5437f1"
)
MAINRUNNER_REBIND_INPUT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/phase6_learning_acceptance"
    / "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940"
    / "segment_0000/recovery/post_evaluation.pt"
)
MAINRUNNER_REBIND_INPUT_CHECKPOINT_SHA256 = (
    "5f6e5c0def21ab736b1ed7d57ad21c64cdd33592827a2f57223cc02c2fb23946"
)
MAINRUNNER_REBIND_INPUT_RECORD = (
    MAINRUNNER_REBIND_INPUT_CHECKPOINT.parent / "source_rebind_summaryschema.json"
)
MAINRUNNER_REBIND_INPUT_RECORD_SHA256 = (
    "8948f275d45a17761fe5c77de83c934b168190f2b19e8881fb583068cdd7ca76"
)
MAINRUNNER_REBIND_ATTEMPT = (
    REPO_ROOT
    / "outputs/phase6_learning_acceptance/mainrunner_readiness_rebind_20260808"
    / "attempt_0001"
)
MAINRUNNER_SNAPSHOT_DIR = MAINRUNNER_REBIND_INPUT_CHECKPOINT.parent / "source_snapshot"
MAINRUNNER_SNAPSHOT_SHA256 = {
    "train_phase6_pre_mainrunner.py": (
        "3db545b1cf8e3436e658151b29b91679e09056bf8bd9e981f58589f36650c2dc"
    ),
    "joint_runner_pre_mainrunner.py": (
        "4cf86dda401494a0793351ce99de775872d52d91647184f0a5a30702e7b16d2c"
    ),
    "residual_env_pre_mainrunner.py": (
        "86aa99b303290a3b746af65936baf93b93fcb00ae99ce6632721df997185293f"
    ),
}
PRODUCTION_SOURCE_PATHS = (
    REPO_ROOT / "configs/phase6_learning_acceptance_v1.json",
    REPO_ROOT / "configs/phase6_joint_training_v1.json",
    REPO_ROOT / "configs/phase6_task_roster_v1.json",
    REPO_ROOT / "somaforce_cross/learning/acceptance.py",
    REPO_ROOT / "somaforce_cross/learning/joint_runner.py",
    REPO_ROOT / "scripts/train_phase6.py",
    REPO_ROOT / "somaforce_cross/envs/residual_env.py",
)
HARNESS_REBIND_FUNCTIONS = (
    "_rank_wrapper_parser",
    "_worker_command",
    "_production_summary_parser",
    "_legacy_recovery_summary_parser",
    "_harness_rebind_parser",
    "_recovery_evaluate_parser",
    "_command_argument",
    "_validate_production_command",
    "_validate_production_train_result",
    "_validate_production_evaluation_result",
    "_run_production_summary",
    "_run_production",
    "_run_stage_command",
    "_run_finalize",
    "_current_production_source_manifest",
    "_ast_function_dumps",
    "_semantic_equal",
    "_legacy_paths",
    "_validate_legacy_stage_command",
    "_validate_legacy_checkpoint_and_results",
    "_validate_legacy_central_markers",
    "_run_legacy_recovery_summary",
    "_validate_rebind_record",
    "_run_harness_rebind",
    "_run_recovery_evaluate",
    "main",
)


class _EvaluationComplete(Exception):
    """Internal control flow used after an isolated evaluation child closes."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_mainrunner_source_manifest() -> dict[str, str]:
    """Return the one immutable seven-file production source binding."""
    from somaforce_cross.learning.acceptance import source_sha256

    manifest = source_sha256(PRODUCTION_SOURCE_PATHS)
    if set(manifest) != {str(path) for path in PRODUCTION_SOURCE_PATHS}:
        raise AssertionError(
            "production source manifest must contain exactly seven files"
        )
    return manifest


def _source_manifest_digest(manifest: Mapping[str, str]) -> str:
    from somaforce_cross.learning.acceptance import source_allowlist_sha256

    return source_allowlist_sha256(manifest)


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
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=grace_s)


def _last_result_marker(log: Path) -> dict[str, Any] | None:
    if not log.is_file():
        return None
    for line in reversed(
        log.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if line.startswith(RESULT_MARKER):
            value = json.loads(line[len(RESULT_MARKER) :])
            if not isinstance(value, dict):
                raise ValueError("Phase 6 result marker must contain an object")
            return value
    return None


def _persist_worker_result(path: Path, result: Mapping[str, Any]) -> None:
    _atomic_json(path, result)
    print("JSON_WRITTEN", flush=True)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)


def _worker_warning_counts(log_text: str) -> dict[str, int]:
    return {
        "headless_glfw": log_text.count("GLFW"),
        "kvdb_lock": log_text.lower().count("kvdb") + log_text.lower().count("lock"),
        "multiple_installable_client_drivers": log_text.count(
            "Multiple Installable Client Drivers"
        ),
    }


def _parent_close_verified(
    *,
    result: Mapping[str, Any] | None,
    exit_code: int | None,
    timed_out: bool,
    markers: list[str],
    log_text: str,
) -> bool:
    return (
        result is not None
        and result.get("status") == "ok"
        and exit_code == 0
        and not timed_out
        and markers == ["JSON_WRITTEN", RESULT_MARKER.rstrip("="), "ENV_CLOSED"]
        and markers[-1] == "ENV_CLOSED"
        and "ENV_CLOSED" in log_text
    )


def _write_progress(path: Path, **fields: Any) -> None:
    _atomic_json(path, {"version": "phase6_progress_v1", **fields})


class _EvaluationProgress:
    """Atomic monotonic per-rank progress with a bounded ETA estimate."""

    def __init__(
        self,
        *,
        path: Path,
        rank: int,
        world_size: int,
        task: str,
        device: str,
        total_episodes: int,
        schedule_sha256: str,
        checkpoint_sha256: str,
        source_rebind_sha256: str,
    ) -> None:
        self.path = path
        self.rank = rank
        self.world_size = world_size
        self.task = task
        self.device = device
        self.total_episodes = total_episodes
        self.schedule_sha256 = schedule_sha256
        self.checkpoint_sha256 = checkpoint_sha256
        self.source_rebind_sha256 = source_rebind_sha256
        self.started = time.monotonic()
        self.last_write = 0.0
        self.completed = 0
        self.control_steps = 0
        self.mode_counts = {"residual": 0, "scaffold_only": 0}
        self.subset_counts = {"stage": 0, "nominal": 0}
        self.update(status="initializing", force=True)

    def update(
        self,
        *,
        status: str,
        completed: int | None = None,
        mode_counts: Mapping[str, int] | None = None,
        subset_counts: Mapping[str, int] | None = None,
        control_steps: int | None = None,
        force: bool = False,
    ) -> None:
        if completed is not None:
            if completed < self.completed:
                raise ValueError("evaluation progress completed count regressed")
            self.completed = completed
        if mode_counts is not None:
            self.mode_counts = dict(mode_counts)
        if subset_counts is not None:
            self.subset_counts = dict(subset_counts)
        if control_steps is not None:
            self.control_steps = control_steps
        now = time.monotonic()
        if not force and self.completed == 0 and now - self.last_write < 10.0:
            return
        if not force and self.completed > 0 and now - self.last_write < 10.0:
            return
        elapsed = max(now - self.started, 1.0e-9)
        rate = self.completed / elapsed if self.completed else 0.0
        eta = (
            None if rate == 0.0 else max(self.total_episodes - self.completed, 0) / rate
        )
        self.last_write = now
        _write_progress(
            self.path,
            schema_version="phase6_paired_evaluation_progress_v1",
            status=status,
            rank=self.rank,
            world_size=self.world_size,
            task=self.task,
            mode=self.mode_counts.get("active", ""),
            subset="",
            device=self.device,
            completed_episodes=self.completed,
            total_episodes=self.total_episodes,
            mode_counters={
                key: int(value)
                for key, value in self.mode_counts.items()
                if key != "active"
            },
            subset_counters={
                key: int(value) for key, value in self.subset_counts.items()
            },
            control_steps=self.control_steps,
            elapsed_s=elapsed,
            episodes_per_s=rate,
            eta_s=eta,
            schedule_sha256=self.schedule_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            source_rebind_sha256=self.source_rebind_sha256,
            optimizer_steps=0,
            normalizer_updates=0,
            last_update_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


def _aggregate_evaluation_progress(output_dir: Path, world_size: int = 4) -> None:
    rank_progress = []
    for rank in range(world_size):
        progress = _read_json(output_dir / f"rank_{rank}/progress.json")
        if progress is not None:
            rank_progress.append(progress)
    if not rank_progress:
        return
    unfinished = [
        item for item in rank_progress if item["status"] not in {"complete", "error"}
    ]
    eta_values = [item["eta_s"] for item in unfinished if item["eta_s"] is not None]
    root = dict(rank_progress[0])
    root.update(
        {
            "rank": 0,
            "status": "complete" if not unfinished else "running",
            "rank_progress": rank_progress,
            "eta_s": max(eta_values) if eta_values else None,
            "completed_episodes": sum(
                item["completed_episodes"] for item in rank_progress
            ),
            "total_episodes": sum(item["total_episodes"] for item in rank_progress),
        }
    )
    _write_progress(output_dir / "progress.json", **root)


def _aggregate_modeprocess_progress(attempt_dir: Path, world_size: int = 4) -> None:
    """Atomically aggregate the two isolated evaluation process lifecycles."""
    modes = ("residual", "scaffold_only")
    mode_progress = {
        mode: _read_json(attempt_dir / mode / "progress.json") for mode in modes
    }
    available = {
        mode: progress
        for mode, progress in mode_progress.items()
        if progress is not None
    }
    if not available:
        return
    first = next(iter(available.values()))
    assert first is not None
    per_mode_total = int(first["total_episodes"])
    all_ranks = [
        rank_progress
        for progress in available.values()
        for rank_progress in progress.get("rank_progress", [])
    ]
    if any(
        item.get("optimizer_steps") != 0 or item.get("normalizer_updates") != 0
        for item in all_ranks
    ):
        raise ValueError("modeprocess progress reports an evaluation update")
    completed = sum(
        int(progress["completed_episodes"]) for progress in available.values()
    )
    complete = len(available) == len(modes) and all(
        progress["status"] == "complete" for progress in available.values()
    )
    active_mode = next(
        (
            mode
            for mode in modes
            if mode not in available or available[mode]["status"] != "complete"
        ),
        "",
    )
    eta_values = [
        progress["eta_s"]
        for progress in available.values()
        if progress["status"] != "complete" and progress.get("eta_s") is not None
    ]
    _write_progress(
        attempt_dir / "progress.json",
        schema_version="phase6_modeprocess_evaluation_progress_v1",
        status="complete" if complete else "running",
        active_mode=active_mode,
        mode_offsets={"residual": 0, "scaffold_only": per_mode_total},
        mode_progress=available,
        rank_progress=all_ranks,
        completed_episodes=completed,
        total_episodes=per_mode_total * 2,
        eta_s=max(eta_values) if eta_values else None,
        optimizer_steps=0,
        normalizer_updates=0,
        last_update_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def _rank_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rank-wrapper",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "smoke_4gpu_2env_1iter",
            "smoke_4gpu_64env_2iter",
            "production_segment",
        ),
        required=True,
    )
    parser.add_argument(
        "--worker-mode",
        choices=("worker", "train-segment", "evaluate"),
        default="worker",
    )
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    parser.add_argument("--segment-start", type=int, default=0)
    parser.add_argument("--segment-end", type=int)
    parser.add_argument("--stage", choices=("C1", "C2", "C3"), default="C1")
    parser.add_argument("--evaluation-seed", type=int, default=20262806)
    parser.add_argument("--paired-num-envs", type=int)
    parser.add_argument("--paired-stage-quota", type=int, default=256)
    parser.add_argument("--paired-nominal-quota", type=int, default=128)
    parser.add_argument("--paired-mode", choices=("residual", "scaffold_only"))
    parser.add_argument("--progress-stall-timeout-s", type=int, default=0)
    parser.add_argument("--source-rebind", type=Path)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    return parser


def _production_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("production",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("pilot", "main"), required=True)
    parser.add_argument(
        "--train-timeout-s",
        type=int,
        default=int(os.getenv("PHASE6_TRAIN_TIMEOUT_S", "3600")),
    )
    parser.add_argument(
        "--evaluation-timeout-s",
        type=int,
        default=int(os.getenv("PHASE6_EVALUATION_TIMEOUT_S", "14400")),
    )
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--source-rebind", type=Path)
    parser.add_argument("--max-segments", type=int, default=1)
    return parser


def _summarize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("summarize",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke_4gpu_2env_1iter", "smoke_4gpu_64env_2iter"),
        required=True,
    )
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("worker", "train-segment", "evaluate"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "smoke_4gpu_2env_1iter",
            "smoke_4gpu_64env_2iter",
            "production_segment",
        ),
        required=True,
    )
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    parser.add_argument("--segment-start", type=int, default=0)
    parser.add_argument("--segment-end", type=int)
    parser.add_argument("--stage", choices=("C1", "C2", "C3"), default="C1")
    parser.add_argument("--evaluation-seed", type=int, default=20262806)
    parser.add_argument("--paired-num-envs", type=int)
    parser.add_argument("--paired-stage-quota", type=int, default=256)
    parser.add_argument("--paired-nominal-quota", type=int, default=128)
    parser.add_argument("--paired-mode", choices=("residual", "scaffold_only"))
    parser.add_argument("--progress-stall-timeout-s", type=int, default=0)
    parser.add_argument("--source-rebind", type=Path)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _profile(profile: str) -> tuple[int, int]:
    values = {
        "smoke_4gpu_2env_1iter": (2, 1),
        "smoke_4gpu_64env_2iter": (64, 2),
    }
    try:
        return values[profile]
    except KeyError as exc:
        raise ValueError(f"unknown Phase 6 bounded profile: {profile}") from exc


def _validate_worker_paths(args: argparse.Namespace) -> None:
    if args.phase6_config.resolve() != PHASE6_CONFIG.resolve():
        raise ValueError("worker requires configs/phase6_joint_training_v1.json")
    if args.roster.resolve() != PHASE6_ROSTER.resolve():
        raise ValueError("worker requires configs/phase6_task_roster_v1.json")
    if args.timeout_s <= 0 or args.cycle_index < 0 or args.window_index < 0:
        raise ValueError("worker timeout, cycle, and window values must be nonnegative")
    if args.mode in {"train-segment", "evaluate"}:
        if args.profile != "production_segment":
            raise ValueError(
                "production workers require the production_segment profile"
            )
        if (
            args.segment_start < 0
            or args.segment_end is None
            or args.segment_end <= args.segment_start
        ):
            raise ValueError("production worker segment bounds are invalid")
        if args.acceptance_config.resolve() != PHASE6_LEARNING_CONFIG.resolve():
            raise ValueError("production worker requires the Phase 6 learning contract")
    if args.mode == "evaluate":
        num_envs = args.paired_num_envs or 64
        if (
            isinstance(num_envs, bool)
            or num_envs <= 0
            or args.paired_stage_quota <= 0
            or args.paired_nominal_quota <= 0
            or args.paired_stage_quota % num_envs
            or args.paired_nominal_quota % num_envs
            or args.paired_mode not in {"residual", "scaffold_only"}
            or args.progress_stall_timeout_s <= 0
            or args.progress_stall_timeout_s >= args.timeout_s
        ):
            raise ValueError(
                "paired evaluation dimensions, mode, or watchdog are invalid"
            )


def _worker_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        args.worker_mode,
        "--output-dir",
        str(args.output_dir),
        "--profile",
        args.profile,
        "--timeout-s",
        str(args.timeout_s),
        "--cycle-index",
        str(args.cycle_index),
        "--window-index",
        str(args.window_index),
        "--phase6-config",
        str(args.phase6_config),
        "--roster",
        str(args.roster),
        "--headless",
    ]
    if args.worker_mode in {"train-segment", "evaluate"}:
        if args.profile != "production_segment" or args.segment_end is None:
            raise ValueError("production wrapper requires complete segment arguments")
        command.extend(
            (
                "--acceptance-config",
                str(args.acceptance_config),
                "--segment-start",
                str(args.segment_start),
                "--segment-end",
                str(args.segment_end),
                "--stage",
                args.stage,
                "--evaluation-seed",
                str(args.evaluation_seed),
            )
        )
        if args.worker_mode == "evaluate":
            command.extend(
                (
                    "--paired-stage-quota",
                    str(args.paired_stage_quota),
                    "--paired-nominal-quota",
                    str(args.paired_nominal_quota),
                    "--paired-mode",
                    str(args.paired_mode),
                    "--progress-stall-timeout-s",
                    str(args.progress_stall_timeout_s),
                )
            )
            if args.paired_num_envs is not None:
                command.extend(("--paired-num-envs", str(args.paired_num_envs)))
        if args.source_rebind is not None:
            command.extend(("--source-rebind", str(args.source_rebind)))
    if args.resume is not None:
        command.extend(("--resume", str(args.resume)))
    return command


def _run_rank_wrapper(args: argparse.Namespace) -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank_dir = args.output_dir / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    command = _worker_command(args)
    _atomic_json(
        rank_dir / "command.json",
        {
            "command": command,
            "local_rank": local_rank,
            "rank": rank,
            "world_size": world_size,
        },
    )
    log_path = rank_dir / "worker.log"
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=os.environ.copy(),
        preexec_fn=os.setsid,
    )
    markers: list[str] = []
    maximum_hwm: int | None = None
    progress_path = rank_dir / "progress.json"
    last_progress_mtime_ns: int | None = None
    last_progress_change = started
    with log_path.open("w", encoding="utf-8") as handle:

        def drain() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
                for marker in PROGRESS_MARKERS:
                    if line.startswith(marker):
                        markers.append(marker.rstrip("="))
                        break

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        timed_out = False
        timeout_kind: str | None = None
        try:
            while process.poll() is None:
                _aggregate_rank_wrapper_progress(args, rank=rank, world_size=world_size)
                observed = _proc_hwm_kib(process.pid)
                if observed is not None:
                    maximum_hwm = max(maximum_hwm or observed, observed)
                if progress_path.is_file():
                    progress_mtime_ns = progress_path.stat().st_mtime_ns
                    if progress_mtime_ns != last_progress_mtime_ns:
                        last_progress_mtime_ns = progress_mtime_ns
                        last_progress_change = time.monotonic()
                if (
                    args.progress_stall_timeout_s > 0
                    and time.monotonic() - last_progress_change
                    > args.progress_stall_timeout_s
                ):
                    timed_out = True
                    timeout_kind = "progress_stall"
                    _terminate_group(process)
                    break
                if time.monotonic() - started > args.timeout_s:
                    timed_out = True
                    timeout_kind = "wall_clock"
                    _terminate_group(process)
                    break
                time.sleep(0.2)
            process.wait(timeout=35.0)
        finally:
            reader.join(timeout=35.0)
    if rank == 0:
        _aggregate_rank_wrapper_progress(args, rank=rank, world_size=world_size)
    result_path = rank_dir / "result.json"
    result_error: str | None = None
    try:
        result = _read_json(result_path)
        marker_result = _last_result_marker(log_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = None
        marker_result = None
        result_error = f"{type(exc).__name__}: {exc}"
    if marker_result is None:
        result_error = result_error or "PHASE6_JOINT_RESULT is missing"
    elif result is not None and marker_result != result:
        result_error = "PHASE6_JOINT_RESULT does not match result.json"
    exit_code = (
        process.returncode
        if process.returncode is not None and process.returncode >= 0
        else None
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    warning_counts = _worker_warning_counts(log_text)
    marker_order_valid = markers == [
        "JSON_WRITTEN",
        RESULT_MARKER.rstrip("="),
        "ENV_CLOSED",
    ]
    last_marker = markers[-1] if markers else None
    close_verification = None
    if _parent_close_verified(
        result=result,
        exit_code=exit_code,
        timed_out=timed_out,
        markers=markers,
        log_text=log_text,
    ):
        close_verification = "parent_verified_exit0"
    wrapper = {
        "command": command,
        "close_verification": close_verification,
        "elapsed_s": round(time.monotonic() - started, 3),
        "exit_code": exit_code,
        "last_progress_marker": last_marker,
        "local_rank": local_rank,
        "log": str(log_path),
        "marker_order_valid": marker_order_valid,
        "markers": markers,
        "passed": close_verification == "parent_verified_exit0",
        "rank": rank,
        "result_error": result_error,
        "result_json": str(result_path),
        "result_status": None if result is None else result.get("status"),
        "signal": _signal_name(process.returncode),
        "timed_out": timed_out,
        "timeout_kind": timeout_kind,
        "vmhwm_kib": maximum_hwm,
        "world_size": world_size,
        "warning_counts": warning_counts,
    }
    _atomic_json(rank_dir / "wrapper.json", wrapper)
    if not wrapper["passed"]:
        raise RuntimeError(f"Phase 6 wrapper evidence gate failed: {rank_dir}")
    return 0


def _aggregate_rank_wrapper_progress(
    args: argparse.Namespace, *, rank: int, world_size: int
) -> None:
    if rank != 0 or args.worker_mode != "evaluate":
        return
    _aggregate_evaluation_progress(args.output_dir, world_size)
    if args.paired_mode is not None:
        _aggregate_modeprocess_progress(args.output_dir.parent, world_size)


def _require_exact_mapping(
    value: object, *, name: str, expected: set[str]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} has unknown or missing schema keys")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA256 digest") from exc
    return value


def _validate_summary_command(
    command: object,
    *,
    rank: int,
    profile: str,
    output_dir: Path,
) -> Mapping[str, object]:
    payload = _require_exact_mapping(
        command,
        name=f"rank_{rank}.command",
        expected={"command", "local_rank", "rank", "world_size"},
    )
    if payload["rank"] != rank or payload["local_rank"] != rank:
        raise ValueError("rank command metadata does not match rank directory")
    if payload["world_size"] != 4 or not isinstance(payload["command"], list):
        raise ValueError("rank command does not describe four-rank execution")
    values = payload["command"]
    if not all(isinstance(item, str) for item in values):
        raise ValueError("rank command must contain string arguments")
    if (
        "--mode" not in values
        or values[values.index("--mode") + 1] != "worker"
        or "--profile" not in values
        or values[values.index("--profile") + 1] != profile
        or "--output-dir" not in values
        or Path(values[values.index("--output-dir") + 1]) != output_dir
    ):
        raise ValueError("rank command does not bind this Phase 6 profile")
    return payload


def _validate_summary_wrapper(
    wrapper: object,
    *,
    command: Mapping[str, object],
    rank: int,
    rank_dir: Path,
) -> Mapping[str, object]:
    payload = _require_exact_mapping(
        wrapper,
        name=f"rank_{rank}.wrapper",
        expected={
            "close_verification",
            "command",
            "elapsed_s",
            "exit_code",
            "last_progress_marker",
            "local_rank",
            "log",
            "marker_order_valid",
            "markers",
            "passed",
            "rank",
            "result_error",
            "result_json",
            "result_status",
            "signal",
            "timed_out",
            "timeout_kind",
            "vmhwm_kib",
            "warning_counts",
            "world_size",
        },
    )
    if (
        payload["command"] != command["command"]
        or payload["rank"] != rank
        or payload["local_rank"] != rank
        or payload["world_size"] != 4
        or payload["result_json"] != str(rank_dir / "result.json")
        or payload["log"] != str(rank_dir / "worker.log")
        or payload["close_verification"] != "parent_verified_exit0"
        or payload["exit_code"] != 0
        or payload["signal"] is not None
        or payload["timed_out"] is not False
        or payload["timeout_kind"] is not None
        or payload["passed"] is not True
        or payload["result_error"] is not None
        or payload["result_status"] != "ok"
        or payload["marker_order_valid"] is not True
        or payload["last_progress_marker"] != "ENV_CLOSED"
        or payload["markers"] != ["JSON_WRITTEN", "PHASE6_JOINT_RESULT", "ENV_CLOSED"]
    ):
        raise ValueError("rank wrapper close evidence is invalid")
    warnings = _require_exact_mapping(
        payload["warning_counts"],
        name=f"rank_{rank}.wrapper.warning_counts",
        expected={
            "headless_glfw",
            "kvdb_lock",
            "multiple_installable_client_drivers",
        },
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in warnings.values()
    ):
        raise ValueError("rank wrapper warning counts are invalid")
    log_text = (rank_dir / "worker.log").read_text(encoding="utf-8", errors="replace")
    observed_warnings = _worker_warning_counts(log_text)
    if dict(warnings) != observed_warnings:
        raise ValueError("rank wrapper warning counts do not match worker log")
    if observed_warnings["multiple_installable_client_drivers"]:
        raise RuntimeError(
            "worker log reports multiple Vulkan installable client drivers"
        )
    return payload


def _validate_summary_result(
    result: object,
    *,
    rank: int,
    profile: str,
    config: object,
    roster: object,
    local_transitions: int,
    global_transitions: int,
    expected_task_transitions: Mapping[str, int],
) -> Mapping[str, object]:
    from somaforce_cross.learning.joint_runner import (
        JointCurriculum,
        JointTaskScheduler,
        Phase6Config,
        Phase6Roster,
        validate_phase6_joint_metrics,
    )

    if not isinstance(config, Phase6Config) or not isinstance(roster, Phase6Roster):
        raise TypeError("Phase 6 summary received invalid contracts")
    payload = _require_exact_mapping(
        result,
        name=f"rank_{rank}.result",
        expected={
            "checkpoint",
            "contracts",
            "curriculum",
            "global_transitions",
            "local_transitions",
            "metrics",
            "optimizer_sha256",
            "optimizer_step",
            "per_task_transitions",
            "policy_sha256",
            "profile",
            "rank",
            "schedule",
            "status",
            "task_fraction",
        },
    )
    if (
        payload["rank"] != rank
        or payload["profile"] != profile
        or payload["status"] != "ok"
        or payload["local_transitions"] != local_transitions
        or payload["global_transitions"] != global_transitions
    ):
        raise ValueError("rank result accounting or status is invalid")
    checkpoint = _require_exact_mapping(
        payload["checkpoint"],
        name=f"rank_{rank}.result.checkpoint",
        expected={"path", "sha256"},
    )
    if not isinstance(checkpoint["path"], str):
        raise ValueError("rank result checkpoint path is invalid")
    _require_sha256(checkpoint["sha256"], name="checkpoint.sha256")
    contracts = _require_exact_mapping(
        payload["contracts"],
        name=f"rank_{rank}.result.contracts",
        expected={
            "phase6_canonical_sha256",
            "phase6_raw_sha256",
            "roster_canonical_sha256",
            "roster_raw_sha256",
        },
    )
    if dict(contracts) != {
        "phase6_canonical_sha256": config.canonical_sha256,
        "phase6_raw_sha256": config.raw_sha256,
        "roster_canonical_sha256": roster.canonical_sha256,
        "roster_raw_sha256": roster.raw_sha256,
    }:
        raise ValueError("rank result contract binding is invalid")
    schedule = _require_exact_mapping(
        payload["schedule"],
        name=f"rank_{rank}.result.schedule",
        expected={"cycle_index", "task", "task_index", "window_index"},
    )
    if not all(
        isinstance(schedule[name], int) and not isinstance(schedule[name], bool)
        for name in ("cycle_index", "task_index", "window_index")
    ):
        raise ValueError("rank schedule coordinates are invalid")
    expected_slot = JointTaskScheduler(
        task_count=len(roster.tasks), world_size=4
    ).assignment(
        cycle_index=schedule["cycle_index"],
        window_index=schedule["window_index"],
        rank=rank,
    )
    if (
        schedule["task_index"] != expected_slot.task_index
        or schedule["task"] != roster.tasks[expected_slot.task_index].task
    ):
        raise ValueError("rank schedule task assignment is invalid")
    if payload["per_task_transitions"] != dict(expected_task_transitions):
        raise ValueError("rank result task transitions are invalid")
    expected_fraction = {
        task: transitions / global_transitions
        for task, transitions in expected_task_transitions.items()
    }
    if payload["task_fraction"] != expected_fraction:
        raise ValueError("rank result task fractions are invalid")
    validate_phase6_joint_metrics(
        payload["metrics"],
        config=config,
        roster=roster,
        expected_global_transitions=global_transitions,
    )
    _require_sha256(payload["policy_sha256"], name="policy_sha256")
    _require_sha256(payload["optimizer_sha256"], name="optimizer_sha256")
    if isinstance(payload["optimizer_step"], bool) or not isinstance(
        payload["optimizer_step"], int
    ):
        raise ValueError("rank optimizer step is invalid")
    curriculum = JointCurriculum()
    curriculum.load_state_dict(payload["curriculum"])
    if curriculum.transitions != global_transitions:
        raise ValueError("rank curriculum transition count is invalid")
    return payload


def summarize_phase6_run(args: argparse.Namespace) -> dict[str, object]:
    """Strictly assemble evidence only after every torchrun wrapper has exited."""
    from somaforce_cross.learning.joint_runner import (
        JointTaskScheduler,
        assert_matching_hash_records,
        load_phase6_config,
        load_phase6_task_roster,
        sha256_file,
    )

    if args.phase6_config.resolve() != PHASE6_CONFIG.resolve():
        raise ValueError("summary requires configs/phase6_joint_training_v1.json")
    if args.roster.resolve() != PHASE6_ROSTER.resolve():
        raise ValueError("summary requires configs/phase6_task_roster_v1.json")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    num_envs, iterations = _profile(args.profile)
    local_transitions = (
        num_envs * int(config.payload["ppo"]["num_steps_per_env"]) * iterations
    )
    global_transitions = 4 * local_transitions
    scheduler = JointTaskScheduler(task_count=len(roster.tasks), world_size=4)
    schedule_values: tuple[int, int] | None = None
    results: list[Mapping[str, object]] = []
    wrappers: list[Mapping[str, object]] = []
    checkpoint: Mapping[str, object] | None = None
    for rank in range(4):
        rank_dir = args.output_dir / f"rank_{rank}"
        command = _validate_summary_command(
            _read_json(rank_dir / "command.json"),
            rank=rank,
            profile=args.profile,
            output_dir=args.output_dir,
        )
        wrapper = _validate_summary_wrapper(
            _read_json(rank_dir / "wrapper.json"),
            command=command,
            rank=rank,
            rank_dir=rank_dir,
        )
        raw_result = _read_json(rank_dir / "result.json")
        if raw_result is None:
            raise ValueError(f"rank {rank} has no result.json")
        raw_schedule = raw_result.get("schedule")
        if not isinstance(raw_schedule, Mapping):
            raise ValueError("rank result has no schedule")
        coordinates = (
            raw_schedule.get("cycle_index"),
            raw_schedule.get("window_index"),
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in coordinates
        ):
            raise ValueError("rank result schedule coordinates are invalid")
        if schedule_values is None:
            schedule_values = (int(coordinates[0]), int(coordinates[1]))
        elif schedule_values != coordinates:
            raise ValueError("rank results do not share one scheduler window")
        expected_named = {
            roster.tasks[index].task: count
            for index, count in scheduler.expected_task_transitions(
                cycle_index=schedule_values[0],
                window_index=schedule_values[1],
                slot_transitions=local_transitions,
            ).items()
        }
        result = _validate_summary_result(
            raw_result,
            rank=rank,
            profile=args.profile,
            config=config,
            roster=roster,
            local_transitions=local_transitions,
            global_transitions=global_transitions,
            expected_task_transitions=expected_named,
        )
        if checkpoint is None:
            checkpoint = result["checkpoint"]
        elif checkpoint != result["checkpoint"]:
            raise ValueError("rank results do not agree on the joint checkpoint")
        results.append(result)
        wrappers.append(wrapper)
    if checkpoint is None:
        raise AssertionError("Phase 6 summary did not inspect any checkpoint")
    checkpoint_path = Path(str(checkpoint["path"]))
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != checkpoint["sha256"]
    ):
        raise ValueError("Phase 6 checkpoint checksum is invalid")
    hashes = [
        {
            "optimizer": result["optimizer_sha256"],
            "policy": result["policy_sha256"],
            "step": result["optimizer_step"],
        }
        for result in results
    ]
    assert_matching_hash_records(hashes)
    if any(result["metrics"] != results[0]["metrics"] for result in results[1:]):
        raise ValueError("rank results do not agree on pooled Phase 6 metrics")
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("Phase 6 checkpoint payload is invalid")
    if (
        checkpoint_payload.get("policy_sha256") != results[0]["policy_sha256"]
        or checkpoint_payload.get("optimizer_sha256") != results[0]["optimizer_sha256"]
        or checkpoint_payload.get("optimizer_step") != results[0]["optimizer_step"]
        or checkpoint_payload.get("task_transitions")
        != results[0]["per_task_transitions"]
        or checkpoint_payload.get("metrics") != results[0]["metrics"]
    ):
        raise ValueError(
            "checkpoint hashes, accounting, or metrics do not match results"
        )
    if set(checkpoint_payload.get("rank_rng", {})) != {"0", "1", "2", "3"}:
        raise ValueError("checkpoint does not persist four rank RNG states")
    summary = {
        "checkpoint": dict(checkpoint),
        "metrics": results[0]["metrics"],
        "profile": args.profile,
        "rank_results": [dict(result) for result in results],
        "status": "ok",
        "wrapper_summaries": [
            {
                "close_verification": wrapper["close_verification"],
                "exit_code": wrapper["exit_code"],
                "rank": wrapper["rank"],
                "signal": wrapper["signal"],
                "timed_out": wrapper["timed_out"],
                "warning_counts": wrapper["warning_counts"],
            }
            for wrapper in wrappers
        ],
    }
    _atomic_json(args.output_dir / "suite_summary.json", summary)
    return summary


def _run_summarize(args: argparse.Namespace) -> int:
    summary = summarize_phase6_run(args)
    print("PHASE6_JOINT_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _production_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("production-summarize",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--paired-stage-quota", type=int, default=256)
    parser.add_argument("--paired-nominal-quota", type=int, default=128)
    parser.add_argument("--source-rebind", type=Path)
    return parser


def _mainrunner_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mainrunner-rebind",), required=True)
    parser.add_argument(
        "--input-checkpoint", type=Path, default=MAINRUNNER_REBIND_INPUT_CHECKPOINT
    )
    parser.add_argument(
        "--input-source-record", type=Path, default=MAINRUNNER_REBIND_INPUT_RECORD
    )
    parser.add_argument("--attempt-dir", type=Path, default=MAINRUNNER_REBIND_ATTEMPT)
    parser.add_argument(
        "--source-snapshot-dir", type=Path, default=MAINRUNNER_SNAPSHOT_DIR
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=MAINRUNNER_REBIND_INPUT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-record-sha256", default=MAINRUNNER_REBIND_INPUT_RECORD_SHA256
    )
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _mainrunner_finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mainrunner-finalize",), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--segment-dir", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--logical-crossing", type=int, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _mainrunner_evaluation_merge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("mainrunner-evaluation-merge",), required=True
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--stage", choices=("C1", "C2", "C3"), required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--paired-stage-quota", type=int, default=256)
    parser.add_argument("--paired-nominal-quota", type=int, default=128)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _legacy_recovery_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy-recovery-summary",), required=True)
    parser.add_argument("--central-train-log", type=Path, required=True)
    parser.add_argument("--train-command", type=Path, required=True)
    parser.add_argument("--summary-command", type=Path, required=True)
    parser.add_argument(
        "--result", dest="results", type=Path, action="append", required=True
    )
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _harness_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("harness-rebind",), required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--legacy-summary", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _recovery_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("recovery-evaluate",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--evaluation-timeout-s", type=int, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _evaluation_devicefix_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("evaluation-devicefix-rebind",), required=True
    )
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _devicefix_recovery_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("devicefix-recovery-evaluate",), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--evaluation-timeout-s", type=int, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _stepboundary_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stepboundary-rebind",), required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _stepboundary_recovery_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("stepboundary-recovery-evaluate",), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--evaluation-timeout-s", type=int, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _pairedbatch_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pairedbatch-rebind",), required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _pairedbatch_recovery_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("pairedbatch-recovery-evaluate",), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--evaluation-timeout-s", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _rankseed_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rankseed-rebind",), required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _rankseed_recovery_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("rankseed-recovery-evaluate",), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--evaluation-timeout-s", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _episodeorder_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("episodeorder-rebind",), required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _episodeorder_recovery_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("episodeorder-recovery-evaluate",), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--evaluation-timeout-s", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _modeprocess_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("modeprocess-rebind",), required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _modeprocess_recovery_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("modeprocess-recovery-evaluate",), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _warningpolicy_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("warningpolicy-rebind",), required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _warningpolicy_recovery_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("warningpolicy-recovery-evaluate",), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _summaryschema_rebind_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("summaryschema-rebind",), required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind-output", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _summaryschema_finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("summaryschema-finalize",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebound-checkpoint", type=Path, required=True)
    parser.add_argument("--source-rebind", type=Path, required=True)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
    parser.add_argument(
        "--acceptance-config", type=Path, default=PHASE6_LEARNING_CONFIG
    )
    return parser


def _finalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("finalize",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    return parser


def _command_argument(values: list[str], name: str) -> str:
    if name not in values:
        raise ValueError(f"command is missing {name}")
    index = values.index(name) + 1
    if index >= len(values):
        raise ValueError(f"command has no value for {name}")
    return values[index]


def _validate_production_command(
    command: object,
    *,
    rank: int,
    output_dir: Path,
    worker_mode: str,
    iteration: int,
) -> Mapping[str, object]:
    payload = _require_exact_mapping(
        command,
        name=f"production.rank_{rank}.command",
        expected={"command", "local_rank", "rank", "world_size"},
    )
    if (
        payload["rank"] != rank
        or payload["local_rank"] != rank
        or payload["world_size"] != 4
        or not isinstance(payload["command"], list)
        or not all(isinstance(value, str) for value in payload["command"])
    ):
        raise ValueError("production rank command metadata is invalid")
    values = list(payload["command"])
    if (
        _command_argument(values, "--mode") != worker_mode
        or _command_argument(values, "--profile") != "production_segment"
        or Path(_command_argument(values, "--output-dir")) != output_dir
        or _command_argument(values, "--acceptance-config")
        != str(PHASE6_LEARNING_CONFIG)
        or _command_argument(values, "--phase6-config") != str(PHASE6_CONFIG)
        or _command_argument(values, "--roster") != str(PHASE6_ROSTER)
        or "--headless" not in values
    ):
        raise ValueError("production rank command does not bind the frozen contracts")
    start = int(_command_argument(values, "--segment-start"))
    end = int(_command_argument(values, "--segment-end"))
    if worker_mode == "train-segment":
        if end != iteration or start < 0 or start >= end:
            raise ValueError("production train command segment bounds are invalid")
    elif start != iteration or end != iteration + 1 or "--resume" not in values:
        raise ValueError("production evaluation command must consume one checkpoint")
    return payload


def _validate_production_train_result(
    result: object,
    *,
    rank: int,
    config: object,
    roster: object,
    expected_global: int,
    expected_per_task: int,
    expected_local_transitions: int,
    expected_cycle_index: int,
    expected_window_index: int,
) -> Mapping[str, object]:
    from somaforce_cross.learning.joint_runner import (
        JointCurriculum,
        Phase6Config,
        Phase6Roster,
        JointTaskScheduler,
        validate_phase6_joint_metrics,
    )

    if not isinstance(config, Phase6Config) or not isinstance(roster, Phase6Roster):
        raise TypeError("production summary requires validated Phase 6 contracts")
    payload = _require_exact_mapping(
        result,
        name=f"production.rank_{rank}.result",
        expected={
            "checkpoint",
            "contracts",
            "curriculum",
            "global_transitions",
            "local_transitions",
            "metrics",
            "optimizer_sha256",
            "optimizer_step",
            "per_task_transitions",
            "policy_sha256",
            "profile",
            "rank",
            "schedule",
            "status",
            "task_fraction",
        },
    )
    expected_tasks = {task.task: expected_per_task for task in roster.tasks}
    if (
        payload["rank"] != rank
        or payload["profile"] != "production_segment"
        or payload["status"] != "ok"
        or payload["global_transitions"] != expected_global
        or payload["local_transitions"] != expected_local_transitions
        or payload["per_task_transitions"] != expected_tasks
        or payload["task_fraction"] != {task: 0.25 for task in expected_tasks}
        or payload["optimizer_step"] != (expected_global // 8192) * 24
    ):
        raise ValueError("production rank result accounting is invalid")
    contracts = _require_exact_mapping(
        payload["contracts"],
        name=f"production.rank_{rank}.contracts",
        expected={
            "phase6_canonical_sha256",
            "phase6_raw_sha256",
            "roster_canonical_sha256",
            "roster_raw_sha256",
        },
    )
    if dict(contracts) != {
        "phase6_canonical_sha256": config.canonical_sha256,
        "phase6_raw_sha256": config.raw_sha256,
        "roster_canonical_sha256": roster.canonical_sha256,
        "roster_raw_sha256": roster.raw_sha256,
    }:
        raise ValueError("production rank result contract binding is invalid")
    schedule = _require_exact_mapping(
        payload["schedule"],
        name=f"production.rank_{rank}.schedule",
        expected={"cycle_index", "task", "task_index", "window_index"},
    )
    if (
        any(
            isinstance(schedule[name], bool) or not isinstance(schedule[name], int)
            for name in ("cycle_index", "task_index", "window_index")
        )
        or schedule["cycle_index"] != expected_cycle_index
        or schedule["window_index"] != expected_window_index
    ):
        raise ValueError("production rank result task assignment is invalid")
    expected_slot = JointTaskScheduler(
        task_count=len(roster.tasks), world_size=4
    ).assignment(
        cycle_index=expected_cycle_index,
        window_index=expected_window_index,
        rank=rank,
    )
    if (
        schedule["task_index"] != expected_slot.task_index
        or schedule["task"] != roster.tasks[expected_slot.task_index].task
    ):
        raise ValueError("production rank result task assignment is invalid")
    checkpoint = _require_exact_mapping(
        payload["checkpoint"],
        name=f"production.rank_{rank}.checkpoint",
        expected={"path", "sha256"},
    )
    if not isinstance(checkpoint["path"], str):
        raise ValueError("production rank checkpoint path is invalid")
    _require_sha256(checkpoint["sha256"], name="production.checkpoint.sha256")
    _require_sha256(payload["policy_sha256"], name="production.policy_sha256")
    _require_sha256(payload["optimizer_sha256"], name="production.optimizer_sha256")
    validate_phase6_joint_metrics(
        payload["metrics"],
        config=config,
        roster=roster,
        expected_global_transitions=expected_global,
    )
    curriculum = JointCurriculum()
    curriculum.load_state_dict(payload["curriculum"])
    if curriculum.transitions != expected_global:
        raise ValueError("production rank curriculum accounting is invalid")
    return payload


def _validate_production_evaluation_result(
    result: object,
    *,
    rank: int,
    roster: object,
    stage_quota: int = 256,
    nominal_quota: int = 128,
) -> Mapping[str, object]:
    from somaforce_cross.learning.joint_runner import (
        Phase6Roster,
        paired_evaluation_plan,
    )

    if not isinstance(roster, Phase6Roster):
        raise TypeError("production evaluation requires a validated roster")
    payload = _require_exact_mapping(
        result,
        name=f"production.evaluation.rank_{rank}.result",
        expected={
            "deterministic_actor_mean",
            "episodes",
            "horizon",
            "control_steps",
            "mode_counts",
            "normalizer_updates",
            "optimizer_steps",
            "paired_modes",
            "rank",
            "schedule_sha256",
            "subset_counts",
            "status",
            "task",
            "unique_seeds",
            "world_size",
        },
    )
    plan = paired_evaluation_plan(
        roster, stage_quota=stage_quota, nominal_quota=nominal_quota
    )
    task = roster.tasks[rank].task
    mode_counts = _require_exact_mapping(
        payload["mode_counts"],
        name=f"production.evaluation.rank_{rank}.mode_counts",
        expected={"residual", "scaffold_only"},
    )
    subset_counts = _require_exact_mapping(
        payload["subset_counts"],
        name=f"production.evaluation.rank_{rank}.subset_counts",
        expected={"stage", "nominal"},
    )
    _require_sha256(payload["schedule_sha256"], name="production.schedule_sha256")
    if (
        payload["rank"] != rank
        or payload["task"] != task
        or payload["status"] != "ok"
        or payload["world_size"] != 4
        or payload["episodes"] != (stage_quota + nominal_quota) * 2
        or payload["control_steps"] <= 0
        or payload["horizon"] != plan["tasks"][task]["horizon"]
        or payload["deterministic_actor_mean"] is not True
        or payload["optimizer_steps"] != 0
        or payload["normalizer_updates"] != 0
        or payload["paired_modes"] != ["residual", "scaffold_only"]
        or mode_counts
        != {
            "residual": stage_quota + nominal_quota,
            "scaffold_only": stage_quota + nominal_quota,
        }
        or subset_counts != {"stage": stage_quota * 2, "nominal": nominal_quota * 2}
        or payload["unique_seeds"] != stage_quota + nominal_quota
    ):
        raise ValueError("production paired evaluation result is invalid")
    return payload


def _run_production_summary(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        optimizer_step,
        restore_learning_checkpoint,
        state_dict_sha256,
        validate_learning_evaluation_summary,
    )

    if args.iteration <= 0:
        raise ValueError("production summary iteration must be positive")
    config = load_phase6_config(PHASE6_CONFIG)
    roster = load_phase6_task_roster(PHASE6_ROSTER, phase6_config=config)
    acceptance = load_learning_acceptance_config(PHASE6_LEARNING_CONFIG)
    rank_results: list[Mapping[str, object]] = []
    wrappers: list[Mapping[str, object]] = []
    tasks: set[str] = set()
    expected_global = args.iteration * 8192
    expected_per_task = expected_global // 4
    expected_segment: tuple[int, int, int, int] | None = None
    checkpoint: Mapping[str, object] | None = None
    for rank in range(4):
        rank_dir = args.output_dir / f"rank_{rank}"
        command = _validate_production_command(
            _read_json(rank_dir / "command.json"),
            rank=rank,
            output_dir=args.output_dir,
            worker_mode="evaluate" if args.evaluation else "train-segment",
            iteration=args.iteration,
        )
        wrapper = _validate_summary_wrapper(
            _read_json(rank_dir / "wrapper.json"),
            command=command,
            rank=rank,
            rank_dir=rank_dir,
        )
        if not isinstance(wrapper["vmhwm_kib"], int) or wrapper["vmhwm_kib"] <= 0:
            raise ValueError("production wrapper must persist a positive VmHWM")
        result = _read_json(rank_dir / "result.json")
        if result is None:
            raise RuntimeError(f"production rank_{rank} result is missing")
        if args.evaluation:
            _validate_production_evaluation_result(
                result,
                rank=rank,
                roster=roster,
                stage_quota=args.paired_stage_quota,
                nominal_quota=args.paired_nominal_quota,
            )
            task = result["task"]
        else:
            values = list(command["command"])
            segment_start = int(_command_argument(values, "--segment-start"))
            segment_end = int(_command_argument(values, "--segment-end"))
            cycle_index = int(_command_argument(values, "--cycle-index"))
            window_index = int(_command_argument(values, "--window-index"))
            segment = (segment_start, segment_end, cycle_index, window_index)
            if expected_segment is None:
                expected_segment = segment
            elif segment != expected_segment:
                raise ValueError("production ranks disagree on segment schedule")
            record = _validate_production_train_result(
                result,
                rank=rank,
                config=config,
                roster=roster,
                expected_global=expected_global,
                expected_per_task=expected_per_task,
                expected_local_transitions=(segment_end - segment_start) * 64 * 32,
                expected_cycle_index=cycle_index,
                expected_window_index=window_index,
            )
            task = record["schedule"]["task"]
            if checkpoint is None:
                checkpoint = record["checkpoint"]
            elif checkpoint != record["checkpoint"]:
                raise ValueError(
                    "production ranks disagree on pre-evaluation checkpoint"
                )
        if not isinstance(task, str) or task in tasks:
            raise ValueError("production ranks must cover four distinct roster tasks")
        tasks.add(task)
        rank_results.append(result)
        wrappers.append(wrapper)
    if tasks != {task.task for task in roster.tasks}:
        raise ValueError("production summary does not cover every roster task")
    if not args.evaluation:
        if checkpoint is None:
            raise AssertionError("production train summary has no checkpoint")
        checkpoint_path = Path(str(checkpoint["path"]))
        if (
            not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint["sha256"]
        ):
            raise ValueError("production pre-evaluation checkpoint checksum is invalid")
        source_manifest = _current_mainrunner_source_manifest()
        policy = ResidualActorCritic()
        optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
        restored = restore_learning_checkpoint(
            checkpoint_path,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=policy,
            optimizer=optimizer,
            source_manifest=source_manifest,
            expected_iteration=args.iteration,
            device="cpu",
        )
        if (
            restored["pre_evaluation"] is not True
            or restored["policy_sha256"] != rank_results[0]["policy_sha256"]
            or restored["optimizer_sha256"] != rank_results[0]["optimizer_sha256"]
            or restored["optimizer_step"] != rank_results[0]["optimizer_step"]
            or restored["metrics"] != rank_results[0]["metrics"]
            or state_dict_sha256(policy.state_dict()) != restored["policy_sha256"]
            or state_dict_sha256(optimizer.state_dict()) != restored["optimizer_sha256"]
            or optimizer_step(optimizer) != restored["optimizer_step"]
        ):
            raise ValueError(
                "production checkpoint payload does not match rank results"
            )
        if any(
            result["metrics"] != rank_results[0]["metrics"]
            for result in rank_results[1:]
        ):
            raise ValueError("production ranks disagree on metrics")
        summary: dict[str, object] = {
            "checkpoint": dict(checkpoint),
            "iteration": args.iteration,
            "rank_results": [dict(result) for result in rank_results],
            "source_manifest_sha256": _source_manifest_digest(source_manifest),
            "status": "ok",
            "task_metrics": dict(restored["metrics"]),
        }
        if args.source_rebind is not None:
            summary["source_rebind"] = {
                "path": str(args.source_rebind),
                "sha256": _sha256(args.source_rebind),
            }
        summary_path = args.output_dir / "segment_summary.json"
    else:
        schedule_hashes = [str(result["schedule_sha256"]) for result in rank_results]
        summary = {
            "iteration": args.iteration,
            "rank_results": [
                {
                    "control_steps": result["control_steps"],
                    "episodes": result["episodes"],
                    "mode_counts": result["mode_counts"],
                    "rank": result["rank"],
                    "schedule_sha256": result["schedule_sha256"],
                    "status": result["status"],
                    "subset_counts": result["subset_counts"],
                    "task": result["task"],
                    "unique_seeds": result["unique_seeds"],
                }
                for result in rank_results
            ],
            "schedule_sha256": hashlib.sha256(
                json.dumps(schedule_hashes, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "status": "ok",
            "task_metrics": {
                str(result["task"]): {
                    "control_steps": result["control_steps"],
                    "episodes": result["episodes"],
                    "mode_counts": result["mode_counts"],
                    "schedule_sha256": result["schedule_sha256"],
                    "subset_counts": result["subset_counts"],
                    "unique_seeds": result["unique_seeds"],
                }
                for result in rank_results
            },
        }
        summary = validate_learning_evaluation_summary(
            summary,
            roster=roster,
            expected_iteration=args.iteration,
            expected_stage_quota=args.paired_stage_quota,
            expected_nominal_quota=args.paired_nominal_quota,
        )
        summary_path = args.output_dir / "evaluation_summary.json"
    _atomic_json(summary_path, summary)
    _atomic_json(
        args.output_dir / "production_evidence_summary.json",
        {
            "iteration": args.iteration,
            "rank_wrappers": [dict(wrapper) for wrapper in wrappers],
            "status": "ok",
            "topology": "per_rank_wrapper",
            "worker_mode": "evaluate" if args.evaluation else "train-segment",
        },
    )
    print("PHASE6_LEARNING_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _run_finalize(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
        validate_learning_evaluation_summary,
    )

    config = load_phase6_config(PHASE6_CONFIG)
    roster = load_phase6_task_roster(PHASE6_ROSTER, phase6_config=config)
    acceptance = load_learning_acceptance_config(PHASE6_LEARNING_CONFIG)
    evaluation = _read_json(args.evaluation_summary)
    if evaluation is None:
        raise RuntimeError("cannot finalize a missing paired evaluation")
    evaluation = validate_learning_evaluation_summary(
        evaluation, roster=roster, expected_iteration=args.iteration
    )
    checkpoint = args.pre_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"pre-evaluation checkpoint is missing: {checkpoint}")
    source_manifest = _current_mainrunner_source_manifest()
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=source_manifest,
        expected_iteration=args.iteration,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("finalize requires a pre-evaluation checkpoint")
    payload = dict(restored)
    payload["pre_evaluation"] = False
    payload["evaluation_history"] = [evaluation]
    post_checkpoint = args.output_dir / "post_evaluation.pt"
    latest = args.output_dir / "latest.json"
    digest = atomic_learning_checkpoint(post_checkpoint, payload, latest_path=latest)
    latest_payload = _read_json(latest)
    if latest_payload != {"checkpoint": str(post_checkpoint), "sha256": digest}:
        raise RuntimeError("finalize latest checkpoint record is invalid")
    print(
        "PHASE6_LEARNING_FINALIZED=" + json.dumps(latest_payload, sort_keys=True),
        flush=True,
    )
    return 0


def _current_production_source_manifest() -> dict[str, str]:
    return _current_mainrunner_source_manifest()


def _mainrunner_ast_source_proof(snapshot_dir: Path) -> dict[str, object]:
    """Bind preserved pre-mainrunner sources separately from the current source."""
    current_paths = {
        "train_phase6": REPO_ROOT / "scripts/train_phase6.py",
        "joint_runner": REPO_ROOT / "somaforce_cross/learning/joint_runner.py",
        "residual_env": REPO_ROOT / "somaforce_cross/envs/residual_env.py",
    }
    snapshot_paths = {
        "train_phase6": snapshot_dir / "train_phase6_pre_mainrunner.py",
        "joint_runner": snapshot_dir / "joint_runner_pre_mainrunner.py",
        "residual_env": snapshot_dir / "residual_env_pre_mainrunner.py",
    }
    snapshots: dict[str, object] = {}
    current: dict[str, object] = {}
    for name, snapshot in snapshot_paths.items():
        expected = MAINRUNNER_SNAPSHOT_SHA256[snapshot.name]
        if not snapshot.is_file() or _sha256(snapshot) != expected:
            raise ValueError(f"mainrunner frozen snapshot differs: {snapshot}")
        current_path = current_paths[name]
        old_tree = ast.parse(
            snapshot.read_text(encoding="utf-8"), filename=str(snapshot)
        )
        new_tree = ast.parse(
            current_path.read_text(encoding="utf-8"), filename=str(current_path)
        )
        old_functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in old_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        new_functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in new_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        snapshots[name] = {
            "path": str(snapshot),
            "sha256": expected,
            "ast_sha256": hashlib.sha256(
                ast.dump(old_tree, include_attributes=False).encode("utf-8")
            ).hexdigest(),
        }
        current[name] = {
            "path": str(current_path),
            "sha256": _sha256(current_path),
            "ast_sha256": hashlib.sha256(
                ast.dump(new_tree, include_attributes=False).encode("utf-8")
            ).hexdigest(),
            "changed_existing_functions": sorted(
                name
                for name in old_functions.keys() & new_functions.keys()
                if old_functions[name] != new_functions[name]
            ),
            "added_functions": sorted(set(new_functions) - set(old_functions)),
            "removed_functions": sorted(set(old_functions) - set(new_functions)),
        }
    return {"current": current, "snapshots": snapshots}


def _mainrunner_rebind_invariants(
    old_payload: Mapping[str, object], new_payload: Mapping[str, object]
) -> dict[str, bool]:
    protected = set(old_payload) | set(new_payload)
    protected.discard("source_manifest")
    return {
        "payload_except_source_manifest": all(
            _semantic_equal(old_payload.get(name), new_payload.get(name))
            for name in protected
        ),
        "source_manifest_only_change": (
            old_payload.get("source_manifest") != new_payload.get("source_manifest")
            and set(old_payload.get("source_manifest", {}))
            == set(new_payload.get("source_manifest", {}))
        ),
    }


def _run_mainrunner_rebind(args: argparse.Namespace) -> int:
    """Rebind a completed pilot checkpoint on CPU without Isaac or CUDA workers."""
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    attempt = args.attempt_dir.resolve()
    output_checkpoint = attempt / "post_evaluation_mainrunner_rebound.pt"
    output_record = attempt / "source_rebind_mainrunner.json"
    if attempt.exists() or output_checkpoint.exists() or output_record.exists():
        raise FileExistsError("mainrunner rebind refuses to overwrite an attempt")
    for path, expected, name in (
        (args.input_checkpoint, args.expected_checkpoint_sha256, "input checkpoint"),
        (args.input_source_record, args.expected_record_sha256, "input source record"),
    ):
        _require_sha256(expected, name=f"{name} SHA256")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"mainrunner {name} checksum mismatch")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    if (
        config.raw_sha256
        != "f26f8e118decb13066362b7a8612b1e2c5fbba37050a38b7ab3880f796c2131d"
        or config.canonical_sha256
        != "8e733479baada2334636106ee352aeb8ab8bf8dc7ad4ce3b39b844019c74df42"
        or roster.raw_sha256
        != "8ddcdd05f423138eb855d9c7cb3478ffb373f718484f0a948512f9bce1492fdb"
        or roster.canonical_sha256
        != "14dfe15db96daa6c8d0bac93965fcdca3d9e815a1ac69a7adaa44d9f0db45bc4"
        or acceptance.raw_sha256
        != "42d855f5aaf2676d5604ba476aa7210ae9b0391adb046d99941d240b3576c063"
        or acceptance.canonical_sha256
        != "b542b5475c5111a97961a1f4fe4e970dfe9af7de97ebcfa55c51c1826c756f85"
    ):
        raise ValueError("mainrunner rebind contract hash mismatch")
    old_payload_value = torch.load(
        args.input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("mainrunner input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    old_manifest = old_payload.get("source_manifest")
    if not isinstance(old_manifest, Mapping) or set(old_manifest) != {
        str(path) for path in PRODUCTION_SOURCE_PATHS
    }:
        raise ValueError("mainrunner input checkpoint lacks the seven-file manifest")
    input_record = _read_json(args.input_source_record)
    if (
        input_record is None
        or input_record.get("status") != "ok"
        or input_record.get("new_source_manifest") != old_manifest
    ):
        raise ValueError("mainrunner input source record does not bind the checkpoint")
    proof = _mainrunner_ast_source_proof(args.source_snapshot_dir)
    current_manifest = _current_mainrunner_source_manifest()
    old_policy = ResidualActorCritic()
    old_optimizer = torch.optim.Adam(old_policy.parameters(), lr=3.0e-4)
    old_restored = restore_learning_checkpoint(
        args.input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=old_policy,
        optimizer=old_optimizer,
        source_manifest=dict(old_manifest),
        expected_iteration=31,
        device="cpu",
    )
    if old_restored["pre_evaluation"] is not False:
        raise ValueError("mainrunner rebind requires a completed post-evaluation pilot")
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = current_manifest
    invariants = _mainrunner_rebind_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError("mainrunner rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=current_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("mainrunner rebound checkpoint did not strictly restore")
    record = {
        "checkpoint": {
            "input_path": str(args.input_checkpoint.resolve()),
            "input_sha256": _sha256(args.input_checkpoint),
            "output_path": str(output_checkpoint),
            "output_sha256": new_sha256,
        },
        "contracts": {
            "learning_acceptance": {
                "canonical_sha256": acceptance.canonical_sha256,
                "raw_sha256": acceptance.raw_sha256,
            },
            "phase6": {
                "canonical_sha256": config.canonical_sha256,
                "raw_sha256": config.raw_sha256,
            },
            "roster": {
                "canonical_sha256": roster.canonical_sha256,
                "raw_sha256": roster.raw_sha256,
            },
        },
        "current_source_proof": proof["current"],
        "input_source_record": {
            "path": str(args.input_source_record.resolve()),
            "sha256": _sha256(args.input_source_record),
        },
        "invariants": invariants,
        "new_source_manifest": current_manifest,
        "new_source_manifest_sha256": _source_manifest_digest(current_manifest),
        "old_source_manifest": dict(old_manifest),
        "old_source_manifest_sha256": _source_manifest_digest(old_manifest),
        "source_snapshots": proof["snapshots"],
        "status": "ok",
    }
    _atomic_json(output_record, record)
    print("PHASE6_MAINRUNNER_REBIND=" + json.dumps(record, sort_keys=True), flush=True)
    return 0


def _validate_mainrunner_rebind_record(
    record_path: Path,
    *,
    config: object,
    roster: object,
    acceptance: object,
    expected_checkpoint: Path | None = None,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        LearningAcceptanceConfig,
        Phase6Config,
        Phase6Roster,
        restore_learning_checkpoint,
    )

    if not isinstance(config, Phase6Config) or not isinstance(roster, Phase6Roster):
        raise TypeError("mainrunner rebind contracts are invalid")
    if not isinstance(acceptance, LearningAcceptanceConfig):
        raise TypeError("mainrunner acceptance contract is invalid")
    record = _read_json(record_path)
    required = {
        "checkpoint",
        "contracts",
        "current_source_proof",
        "input_source_record",
        "invariants",
        "new_source_manifest",
        "new_source_manifest_sha256",
        "old_source_manifest",
        "old_source_manifest_sha256",
        "source_snapshots",
        "status",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("mainrunner source rebind record schema is invalid")
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="mainrunner.rebind.checkpoint",
        expected={"input_path", "input_sha256", "output_path", "output_sha256"},
    )
    input_checkpoint = Path(str(checkpoint["input_path"]))
    output = Path(str(checkpoint["output_path"]))
    input_record = _require_exact_mapping(
        record["input_source_record"],
        name="mainrunner.rebind.input_source_record",
        expected={"path", "sha256"},
    )
    expected_paths = {str(path) for path in PRODUCTION_SOURCE_PATHS}
    old_manifest = record["old_source_manifest"]
    new_manifest = record["new_source_manifest"]
    if (
        not isinstance(old_manifest, Mapping)
        or not isinstance(new_manifest, Mapping)
        or set(old_manifest) != expected_paths
        or set(new_manifest) != expected_paths
        or any(
            _require_sha256(value, name="mainrunner.rebind.source_manifest") is None
            for value in (*old_manifest.values(), *new_manifest.values())
        )
    ):
        raise ValueError("mainrunner source rebind manifests are invalid")
    if (
        record["status"] != "ok"
        or input_checkpoint.resolve() != MAINRUNNER_REBIND_INPUT_CHECKPOINT.resolve()
        or checkpoint["input_sha256"] != MAINRUNNER_REBIND_INPUT_CHECKPOINT_SHA256
        or not input_checkpoint.is_file()
        or _sha256(input_checkpoint) != checkpoint["input_sha256"]
        or Path(str(input_record["path"])).resolve()
        != MAINRUNNER_REBIND_INPUT_RECORD.resolve()
        or input_record["sha256"] != MAINRUNNER_REBIND_INPUT_RECORD_SHA256
        or not MAINRUNNER_REBIND_INPUT_RECORD.is_file()
        or _sha256(MAINRUNNER_REBIND_INPUT_RECORD) != input_record["sha256"]
        or not output.is_file()
        or _sha256(output) != checkpoint["output_sha256"]
        or expected_checkpoint is not None
        and output.resolve() != expected_checkpoint.resolve()
        or dict(new_manifest) != _current_mainrunner_source_manifest()
        or record["new_source_manifest_sha256"] != _source_manifest_digest(new_manifest)
        or record["old_source_manifest_sha256"] != _source_manifest_digest(old_manifest)
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("mainrunner source rebind record binding is invalid")
    source_record = _read_json(MAINRUNNER_REBIND_INPUT_RECORD)
    input_payload = torch.load(input_checkpoint, map_location="cpu", weights_only=False)
    proof = _mainrunner_ast_source_proof(MAINRUNNER_SNAPSHOT_DIR)
    if (
        source_record is None
        or source_record.get("status") != "ok"
        or source_record.get("new_source_manifest") != dict(old_manifest)
        or not isinstance(input_payload, Mapping)
        or input_payload.get("source_manifest") != dict(old_manifest)
        or record["source_snapshots"] != proof["snapshots"]
        or record["current_source_proof"] != proof["current"]
    ):
        raise ValueError("mainrunner source rebind provenance is invalid")
    input_policy = ResidualActorCritic()
    input_optimizer = torch.optim.Adam(input_policy.parameters(), lr=3.0e-4)
    input_restored = restore_learning_checkpoint(
        input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=input_policy,
        optimizer=input_optimizer,
        source_manifest=dict(old_manifest),
        expected_iteration=31,
        device="cpu",
    )
    if (
        input_restored["pre_evaluation"] is not False
        or not isinstance(input_restored["evaluation_history"], list)
        or not input_restored["evaluation_history"]
    ):
        raise ValueError("mainrunner input checkpoint is not a completed pilot")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_mainrunner_source_manifest(),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not False or not all(
        _mainrunner_rebind_invariants(input_restored, restored).values()
    ):
        raise ValueError("mainrunner rebound checkpoint is not post-evaluation")
    return dict(record)


def _current_pairedbatch_source_manifest() -> dict[str, str]:
    from somaforce_cross.learning.acceptance import source_sha256

    return source_sha256(
        (
            REPO_ROOT / "configs/phase6_learning_acceptance_v1.json",
            REPO_ROOT / "configs/phase6_joint_training_v1.json",
            REPO_ROOT / "configs/phase6_task_roster_v1.json",
            REPO_ROOT / "somaforce_cross/learning/acceptance.py",
            REPO_ROOT / "somaforce_cross/learning/joint_runner.py",
            REPO_ROOT / "somaforce_cross/envs/residual_env.py",
            REPO_ROOT / "scripts/train_phase6.py",
        )
    )


def _ast_function_dumps(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    if set(found) != set(names):
        raise ValueError("worker-critical function set changed during harness rebind")
    return found


def _semantic_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if (
            not isinstance(left, Mapping)
            or not isinstance(right, Mapping)
            or set(left) != set(right)
        ):
            return False
        return all(_semantic_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _semantic_equal(first, second) for first, second in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _current_evaluation_crossing(actual_global_transitions: int) -> dict[str, int]:
    if (
        isinstance(actual_global_transitions, bool)
        or not isinstance(actual_global_transitions, int)
        or actual_global_transitions <= 0
    ):
        raise ValueError("evaluation crossing transitions must be a positive integer")
    logical = ((actual_global_transitions - 1) // 250000) * 250000
    if logical <= 0:
        raise ValueError("evaluation crossing has not reached the first boundary")
    target_iteration = (logical + 8191) // 8192
    crossing_actual = target_iteration * 8192
    if crossing_actual != actual_global_transitions:
        raise ValueError("evaluation crossing is not an iteration endpoint")
    return {
        "logical_transitions": logical,
        "target_iteration": target_iteration,
        "actual_transitions": crossing_actual,
    }


def _legacy_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    checkpoint = LEGACY_PRE_EVALUATION_CHECKPOINT
    segment_dir = checkpoint.parent.parent
    expected_results = tuple(
        checkpoint.parent / f"rank_{rank}" / "result.json" for rank in range(4)
    )
    if (
        args.pre_checkpoint.resolve() != checkpoint.resolve()
        or args.central_train_log.resolve() != (segment_dir / "train.log").resolve()
        or args.train_command.resolve()
        != (segment_dir / "train.command.json").resolve()
        or args.summary_command.resolve()
        != (segment_dir / "summary.command.json").resolve()
        or tuple(path.resolve() for path in args.results)
        != tuple(path.resolve() for path in expected_results)
    ):
        raise ValueError("legacy recovery is restricted to the preserved pilot segment")
    return checkpoint, segment_dir, checkpoint.parent, expected_results


def _validate_legacy_stage_command(
    path: Path, *, stage: str, mode: str, output_dir: Path
) -> Mapping[str, object]:
    payload = _require_exact_mapping(
        _read_json(path),
        name=f"legacy.{stage}.command",
        expected={"command", "stage", "timeout_s"},
    )
    if (
        payload["stage"] != stage
        or not isinstance(payload["timeout_s"], int)
        or payload["timeout_s"] <= 0
        or not isinstance(payload["command"], list)
        or not all(isinstance(item, str) for item in payload["command"])
    ):
        raise ValueError("legacy stage command schema is invalid")
    command = list(payload["command"])
    if (
        _command_argument(command, "--mode") != mode
        or Path(_command_argument(command, "--output-dir")) != output_dir
    ):
        raise ValueError("legacy stage command does not bind the preserved segment")
    return payload


def _validate_legacy_checkpoint_and_results(
    *,
    checkpoint: Path,
    expected_sha256: str,
    results: tuple[Path, ...],
    config: object,
    roster: object,
    acceptance: object,
) -> tuple[dict[str, object], list[Mapping[str, object]]]:
    from somaforce_cross.learning.joint_runner import (
        JointCurriculum,
        Phase6Config,
        Phase6Roster,
        sha256_file,
        validate_phase6_joint_metrics,
    )

    if not isinstance(config, Phase6Config) or not isinstance(roster, Phase6Roster):
        raise TypeError("legacy recovery requires validated Phase 6 contracts")
    if (
        not isinstance(expected_sha256, str)
        or expected_sha256 != LEGACY_PRE_EVALUATION_SHA256
    ):
        raise ValueError("legacy recovery expected checkpoint SHA is not frozen")
    if sha256_file(checkpoint) != expected_sha256:
        raise ValueError("legacy recovery pre-evaluation checkpoint SHA differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("legacy recovery checkpoint payload is invalid")
    payload = dict(payload)
    expected_contracts = {
        "phase6": {
            "canonical_sha256": config.canonical_sha256,
            "raw_sha256": config.raw_sha256,
            "contract_version": "phase6_joint_training_v1",
        },
        "roster": {
            "canonical_sha256": roster.canonical_sha256,
            "raw_sha256": roster.raw_sha256,
            "contract_version": "phase6_task_roster_v1",
        },
        "learning_acceptance": {
            "canonical_sha256": acceptance.canonical_sha256,
            "raw_sha256": acceptance.raw_sha256,
            "contract_version": "phase6_learning_acceptance_v1",
        },
        "phase5_checkpoint_sha256": "46d76e722e0422bfa1f00e5d815d80f4a328b0cf169f7104a16c6fbdeebf723a",
    }
    expected_tasks = {task.task: 63488 for task in roster.tasks}
    if (
        payload.get("checkpoint_version") != "phase6_learning_checkpoint_v1"
        or payload.get("pre_evaluation") is not True
        or payload.get("iteration") != 31
        or payload.get("actual_global_transitions") != 253952
        or payload.get("actual_per_task_transitions") != 63488
        or payload.get("task_transitions") != expected_tasks
        or payload.get("optimizer_step") != 744
        or payload.get("contracts") != expected_contracts
        or set(payload.get("rank_rng", {})) != {"0", "1", "2", "3"}
    ):
        raise ValueError("legacy checkpoint does not satisfy the frozen pilot contract")
    snapshot = checkpoint.parent.parent / "recovery/source_snapshot/train_phase6.py"
    if _sha256(snapshot) != payload.get("source_manifest", {}).get(
        str(REPO_ROOT / "scripts/train_phase6.py")
    ):
        raise ValueError("legacy source snapshot does not bind the checkpoint manifest")
    current_manifest = _current_production_source_manifest()
    old_manifest = payload.get("source_manifest")
    if not isinstance(old_manifest, Mapping) or set(old_manifest) != set(
        current_manifest
    ):
        raise ValueError("legacy checkpoint source manifest keys are invalid")
    script_key = str(REPO_ROOT / "scripts/train_phase6.py")
    if any(
        old_manifest[key] != current_manifest[key]
        for key in old_manifest
        if key != script_key
    ):
        raise ValueError("a non-harness source changed since the preserved pilot")
    normalized: list[Mapping[str, object]] = []
    common_hashes: dict[str, object] | None = None
    for rank, path in enumerate(results):
        result = _validate_production_train_result(
            _read_json(path),
            rank=rank,
            config=config,
            roster=roster,
            expected_global=253952,
            expected_per_task=63488,
            expected_local_transitions=63488,
            expected_cycle_index=0,
            expected_window_index=0,
        )
        if (
            result["checkpoint"] != {"path": str(checkpoint), "sha256": expected_sha256}
            or result["metrics"] != payload["metrics"]
            or result["policy_sha256"] != payload["policy_sha256"]
            or result["optimizer_sha256"] != payload["optimizer_sha256"]
            or result["optimizer_step"] != payload["optimizer_step"]
        ):
            raise ValueError("legacy rank result does not match preserved checkpoint")
        validate_phase6_joint_metrics(
            result["metrics"],
            config=config,
            roster=roster,
            expected_global_transitions=253952,
        )
        record_hashes = {
            "policy": result["policy_sha256"],
            "optimizer": result["optimizer_sha256"],
            "step": result["optimizer_step"],
        }
        if common_hashes is None:
            common_hashes = record_hashes
        elif common_hashes != record_hashes:
            raise ValueError("legacy rank hash records disagree")
        curriculum = JointCurriculum()
        curriculum.load_state_dict(result["curriculum"])
        if curriculum.transitions != 253952:
            raise ValueError(
                "legacy rank curriculum does not match preserved checkpoint"
            )
        normalized.append(result)
    if payload["policy_sha256"] != payload.get("policy_sha256") or payload[
        "optimizer_sha256"
    ] != payload.get("optimizer_sha256"):
        raise AssertionError("unreachable checkpoint hash guard")
    return payload, normalized


def _validate_legacy_central_markers(
    log_text: str, *, rank_results: list[Mapping[str, object]]
) -> dict[str, int]:
    line_positions: dict[str, list[int]] = {"JSON_WRITTEN": [], "ENV_CLOSED": []}
    offset = 0
    for line in log_text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if content in line_positions:
            line_positions[content].append(offset)
        offset += len(line)
    json_positions = line_positions["JSON_WRITTEN"]
    close_positions = line_positions["ENV_CLOSED"]
    result_positions: list[int] = []
    search_from = 0
    while True:
        position = log_text.find(RESULT_MARKER, search_from)
        if position < 0:
            break
        result_positions.append(position)
        search_from = position + len(RESULT_MARKER)
    if (
        len(json_positions) != 4
        or len(result_positions) != 4
        or len(close_positions) != 4
        or max(json_positions) >= result_positions[0]
    ):
        raise RuntimeError("legacy centralized close marker evidence is incomplete")
    decoder = json.JSONDecoder()
    marker_results: list[Mapping[str, object]] = []
    result_ends: list[int] = []
    for position in result_positions:
        try:
            value, end = decoder.raw_decode(log_text, position + len(RESULT_MARKER))
        except json.JSONDecodeError as exc:
            raise RuntimeError("legacy centralized result JSON is invalid") from exc
        if not isinstance(value, Mapping):
            raise RuntimeError("legacy centralized result JSON must be an object")
        marker_results.append(value)
        result_ends.append(end)
    first_close = close_positions[0]
    boundaries = result_positions[1:] + [first_close]
    if (
        any(
            end > boundary
            or any(character not in " \t\r\n" for character in log_text[end:boundary])
            for end, boundary in zip(result_ends, boundaries)
        )
        or result_ends[-1] > first_close
    ):
        raise RuntimeError("legacy centralized result marker order is invalid")
    by_rank: dict[int, Mapping[str, object]] = {}
    for result in rank_results:
        rank = result.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank in by_rank:
            raise RuntimeError("legacy centralized rank results are invalid")
        by_rank[rank] = result
    marker_by_rank: dict[int, Mapping[str, object]] = {}
    for result in marker_results:
        rank = result.get("rank")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank in marker_by_rank
        ):
            raise RuntimeError("legacy centralized result ranks are not unique")
        marker_by_rank[rank] = result
    if (
        set(by_rank) != {0, 1, 2, 3}
        or set(marker_by_rank) != {0, 1, 2, 3}
        or any(marker_by_rank[rank] != by_rank[rank] for rank in by_rank)
    ):
        raise RuntimeError("legacy centralized result markers do not match rank JSON")
    warning_counts = _worker_warning_counts(log_text)
    if warning_counts["multiple_installable_client_drivers"] != 0:
        raise RuntimeError("legacy centralized log reports multiple Vulkan ICDs")
    return warning_counts


def _run_legacy_recovery_summary(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    if len(args.results) != 4:
        raise ValueError("legacy recovery requires exactly four rank result files")
    checkpoint, _, train_dir, results = _legacy_paths(args)
    if (
        args.summary_output.resolve()
        != (checkpoint.parent.parent / "recovery/legacy_train_summary.json").resolve()
    ):
        raise ValueError(
            "legacy recovery summary output is fixed for the preserved pilot"
        )
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_legacy_stage_command(
        args.train_command,
        stage="train-segment",
        mode="train-segment",
        output_dir=train_dir,
    )
    summary_command = _validate_legacy_stage_command(
        args.summary_command,
        stage="summarize",
        mode="production-summarize",
        output_dir=train_dir,
    )
    command = list(summary_command["command"])
    if _command_argument(command, "--iteration") != "31":
        raise ValueError("legacy summary did not start at the pilot crossing")
    checkpoint_payload, rank_results = _validate_legacy_checkpoint_and_results(
        checkpoint=checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        results=results,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    warning_counts = _validate_legacy_central_markers(
        args.central_train_log.read_text(encoding="utf-8", errors="replace"),
        rank_results=rank_results,
    )
    summary = {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": args.expected_checkpoint_sha256,
        },
        "evidence_topology": "legacy_central_torchrun",
        "global_transitions": 253952,
        "iteration": 31,
        "optimizer_sha256": checkpoint_payload["optimizer_sha256"],
        "optimizer_step": 744,
        "parent_exit_proven_by_stage_advancement": True,
        "per_rank_vmhwm_available": False,
        "policy_sha256": checkpoint_payload["policy_sha256"],
        "rank_results": [dict(result) for result in rank_results],
        "status": "ok",
        "task_transitions": dict(checkpoint_payload["task_transitions"]),
        "warning_counts": warning_counts,
    }
    _atomic_json(args.summary_output, summary)
    print("PHASE6_LEGACY_RECOVERY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _validate_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    record = _require_exact_mapping(
        _read_json(record_path),
        name="source_rebind",
        expected={
            "allowed_harness_functions",
            "checkpoint",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "source_snapshot",
            "status",
            "worker_critical_ast_equality",
        },
    )
    if (
        record["status"] != "ok"
        or tuple(record["allowed_harness_functions"]) != HARNESS_REBIND_FUNCTIONS
        or not isinstance(record["worker_critical_ast_equality"], Mapping)
        or set(record["worker_critical_ast_equality"]) != set(WORKER_CRITICAL_FUNCTIONS)
        or not all(record["worker_critical_ast_equality"].values())
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("source rebind record does not prove a harness-only change")
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="source_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    if (
        checkpoint["old_path"] != str(LEGACY_PRE_EVALUATION_CHECKPOINT)
        or checkpoint["old_sha256"] != LEGACY_PRE_EVALUATION_SHA256
        or checkpoint["new_path"] != str(rebound_checkpoint)
        or _sha256(rebound_checkpoint) != checkpoint["new_sha256"]
        or record["source_snapshot"]
        != {
            "path": str(
                LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent
                / "recovery/source_snapshot/train_phase6.py"
            ),
            "sha256": "70b35871b72d8cc2d841fc75c04578207c2a3aa96083240ce4b953a4961aace2",
        }
    ):
        raise ValueError("source rebind record does not bind the preserved pilot")
    source_manifest = _current_production_source_manifest()
    if record["new_source_manifest"] != source_manifest:
        raise ValueError("source rebind record does not bind current production source")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=source_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("rebound checkpoint is not a pre-evaluation checkpoint")
    return dict(record)


def _run_harness_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    if (
        args.pre_checkpoint.resolve() != LEGACY_PRE_EVALUATION_CHECKPOINT.resolve()
        or args.expected_checkpoint_sha256 != LEGACY_PRE_EVALUATION_SHA256
        or args.source_snapshot.resolve()
        != (
            LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent
            / "recovery/source_snapshot/train_phase6.py"
        ).resolve()
        or args.rebound_checkpoint.resolve()
        != (
            LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent
            / "recovery/pre_evaluation_harness_rebound.pt"
        ).resolve()
        or args.source_rebind_output.resolve()
        != (
            LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent
            / "recovery/source_rebind.json"
        ).resolve()
        or args.legacy_summary.resolve()
        != (
            LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent
            / "recovery/legacy_train_summary.json"
        ).resolve()
    ):
        raise ValueError(
            "harness rebind is restricted to the preserved pilot checkpoint"
        )
    legacy_summary = _read_json(args.legacy_summary)
    if (
        legacy_summary is None
        or legacy_summary.get("status") != "ok"
        or legacy_summary.get("evidence_topology") != "legacy_central_torchrun"
        or legacy_summary.get("checkpoint")
        != {
            "path": str(LEGACY_PRE_EVALUATION_CHECKPOINT),
            "sha256": LEGACY_PRE_EVALUATION_SHA256,
        }
    ):
        raise ValueError("harness rebind requires a verified legacy recovery summary")
    if (
        _sha256(args.source_snapshot)
        != "70b35871b72d8cc2d841fc75c04578207c2a3aa96083240ce4b953a4961aace2"
    ):
        raise ValueError("harness rebind source snapshot SHA differs")
    if _sha256(args.pre_checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("harness rebind pre-evaluation checkpoint SHA differs")
    if args.rebound_checkpoint.exists() or args.source_rebind_output.exists():
        raise FileExistsError("harness rebind refuses to overwrite recovery evidence")
    old_function_dumps = _ast_function_dumps(
        args.source_snapshot, WORKER_CRITICAL_FUNCTIONS
    )
    new_function_dumps = _ast_function_dumps(
        Path(__file__).resolve(), WORKER_CRITICAL_FUNCTIONS
    )
    function_equality = {
        name: old_function_dumps[name] == new_function_dumps[name]
        for name in WORKER_CRITICAL_FUNCTIONS
    }
    if not all(function_equality.values()):
        raise ValueError("harness rebind changed a worker-critical function AST")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    old_payload_value = torch.load(
        args.pre_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("harness rebind pre-evaluation payload is invalid")
    old_payload = dict(old_payload_value)
    old_manifest = old_payload.get("source_manifest")
    new_manifest = _current_production_source_manifest()
    script_key = str(REPO_ROOT / "scripts/train_phase6.py")
    if (
        not isinstance(old_manifest, Mapping)
        or set(old_manifest) != set(new_manifest)
        or old_manifest.get(script_key) != _sha256(args.source_snapshot)
        or any(
            old_manifest[key] != new_manifest[key]
            for key in old_manifest
            if key != script_key
        )
    ):
        raise ValueError(
            "harness rebind source manifest is not limited to train_phase6.py"
        )
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariant_names = (
        "policy",
        "optimizer",
        "policy_sha256",
        "optimizer_sha256",
        "optimizer_step",
        "rank_rng",
        "iteration",
        "actual_global_transitions",
        "actual_per_task_transitions",
        "task_transitions",
        "curriculum",
        "metrics",
        "next_crossing",
        "evaluation_history",
        "contracts",
        "pre_evaluation",
    )
    invariants = {
        name: _semantic_equal(old_payload[name], new_payload[name])
        for name in invariant_names
    }
    invariants["payload_except_source_manifest"] = _semantic_equal(
        {key: value for key, value in old_payload.items() if key != "source_manifest"},
        {key: value for key, value in new_payload.items() if key != "source_manifest"},
    )
    if not all(invariants.values()):
        raise AssertionError("harness rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(args.rebound_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        args.rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("harness rebound checkpoint did not strictly restore")
    record = {
        "allowed_harness_functions": list(HARNESS_REBIND_FUNCTIONS),
        "checkpoint": {
            "new_path": str(args.rebound_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(args.pre_checkpoint),
            "old_sha256": args.expected_checkpoint_sha256,
        },
        "invariants": invariants,
        "new_source_manifest": new_manifest,
        "old_source_manifest": dict(old_manifest),
        "source_snapshot": {
            "path": str(args.source_snapshot),
            "sha256": _sha256(args.source_snapshot),
        },
        "status": "ok",
        "worker_critical_ast_equality": function_equality,
    }
    _atomic_json(args.source_rebind_output, record)
    print("PHASE6_HARNESS_REBIND=" + json.dumps(record, sort_keys=True), flush=True)
    return 0


def _source_manifest_with_train_sha256(train_sha256: str) -> dict[str, str]:
    _require_sha256(train_sha256, name="train_phase6.py SHA256")
    manifest = _current_production_source_manifest()
    manifest[str(REPO_ROOT / "scripts/train_phase6.py")] = train_sha256
    return manifest


def _top_level_function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise ValueError(f"{path} must contain exactly one {name} function")
    return functions[0]


def _policy_constructor_assignment(function: ast.FunctionDef) -> ast.Assign:
    assignments = [
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "policy"
    ]
    if len(assignments) != 1:
        raise ValueError("paired evaluation must assign policy exactly once")
    return assignments[0]


def _evaluation_devicefix_ast_proof(
    source_snapshot: Path, current_source: Path
) -> dict[str, object]:
    old_functions = _ast_function_dumps(source_snapshot, WORKER_CRITICAL_FUNCTIONS)
    new_functions = _ast_function_dumps(current_source, WORKER_CRITICAL_FUNCTIONS)
    other_critical = {
        name: old_functions[name] == new_functions[name]
        for name in WORKER_CRITICAL_FUNCTIONS
        if name != EVALUATION_DEVICEFIX_FUNCTION
    }
    old_function = _top_level_function(source_snapshot, EVALUATION_DEVICEFIX_FUNCTION)
    new_function = _top_level_function(current_source, EVALUATION_DEVICEFIX_FUNCTION)
    old_assignment = _policy_constructor_assignment(old_function)
    new_assignment = _policy_constructor_assignment(new_function)
    old_value = old_assignment.value
    if not (
        isinstance(old_value, ast.Call)
        and isinstance(old_value.func, ast.Name)
        and old_value.func.id == "ResidualActorCritic"
        and not old_value.args
        and not old_value.keywords
    ):
        raise ValueError("evaluation snapshot policy constructor is not frozen")
    new_value = new_assignment.value
    if not (
        isinstance(new_value, ast.Call)
        and isinstance(new_value.func, ast.Attribute)
        and new_value.func.attr == "to"
        and isinstance(new_value.func.value, ast.Call)
        and isinstance(new_value.func.value.func, ast.Name)
        and new_value.func.value.func.id == "ResidualActorCritic"
        and not new_value.func.value.args
        and not new_value.func.value.keywords
        and not new_value.args
        and len(new_value.keywords) == 1
        and new_value.keywords[0].arg == "device"
        and isinstance(new_value.keywords[0].value, ast.Call)
        and isinstance(new_value.keywords[0].value.func, ast.Attribute)
        and isinstance(new_value.keywords[0].value.func.value, ast.Name)
        and new_value.keywords[0].value.func.value.id == "torch"
        and new_value.keywords[0].value.func.attr == "device"
        and len(new_value.keywords[0].value.args) == 1
        and isinstance(new_value.keywords[0].value.args[0], ast.Attribute)
        and isinstance(new_value.keywords[0].value.args[0].value, ast.Name)
        and new_value.keywords[0].value.args[0].value.id == "args"
        and new_value.keywords[0].value.args[0].attr == "device"
        and not new_value.keywords[0].value.keywords
    ):
        raise ValueError("evaluation device fix is not the approved policy migration")
    load_calls = [
        node
        for node in ast.walk(new_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "policy"
        and node.func.attr == "load_state_dict"
    ]
    if len(load_calls) != 1 or new_assignment.lineno >= load_calls[0].lineno:
        raise ValueError("evaluation policy must migrate before checkpoint loading")
    normalized = copy.deepcopy(new_function)
    _policy_constructor_assignment(normalized).value = copy.deepcopy(old_value)
    non_device_ast_equal = ast.dump(normalized, include_attributes=False) == ast.dump(
        old_function, include_attributes=False
    )
    proof = {
        "device_argument": "torch.device(args.device)",
        "evaluation_function": EVALUATION_DEVICEFIX_FUNCTION,
        "load_follows_device_move": True,
        "new_policy_constructor": "ResidualActorCritic().to(device=torch.device(args.device))",
        "non_device_ast_equal": non_device_ast_equal,
        "other_worker_critical_ast_equality": other_critical,
        "old_policy_constructor": "ResidualActorCritic()",
    }
    if not all(other_critical.values()) or not non_device_ast_equal:
        raise ValueError(
            "evaluation device fix changed more than policy device placement"
        )
    return proof


def _evaluation_devicefix_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    snapshot = recovery / "source_snapshot/train_phase6_pre_devicefix.py"
    input_checkpoint = recovery / "pre_evaluation_harness_rebound.pt"
    input_record = recovery / "source_rebind.json"
    output_checkpoint = recovery / "pre_evaluation_devicefix_rebound.pt"
    output_record = recovery / "source_rebind_devicefix.json"
    if (
        args.source_snapshot.resolve() != snapshot.resolve()
        or args.pre_checkpoint.resolve() != input_checkpoint.resolve()
        or args.source_rebind.resolve() != input_record.resolve()
        or args.rebound_checkpoint.resolve() != output_checkpoint.resolve()
        or args.source_rebind_output.resolve() != output_record.resolve()
    ):
        raise ValueError("evaluation devicefix rebind paths are fixed for this pilot")
    return snapshot, input_checkpoint, input_record, output_checkpoint, output_record


def _validate_devicefix_input_record(
    *,
    record_path: Path,
    input_checkpoint: Path,
    snapshot_manifest: Mapping[str, str],
) -> dict[str, object]:
    record = _require_exact_mapping(
        _read_json(record_path),
        name="devicefix.input_source_rebind",
        expected={
            "allowed_harness_functions",
            "checkpoint",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "source_snapshot",
            "status",
            "worker_critical_ast_equality",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="devicefix.input_source_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    if (
        record["status"] != "ok"
        or tuple(record["allowed_harness_functions"]) != HARNESS_REBIND_FUNCTIONS
        or checkpoint["new_path"] != str(input_checkpoint)
        or checkpoint["new_sha256"] != EVALUATION_DEVICEFIX_INPUT_SHA256
        or _sha256(input_checkpoint) != EVALUATION_DEVICEFIX_INPUT_SHA256
        or record["new_source_manifest"] != dict(snapshot_manifest)
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
        or not isinstance(record["worker_critical_ast_equality"], Mapping)
        or not all(record["worker_critical_ast_equality"].values())
    ):
        raise ValueError("devicefix input does not bind the admitted harness rebind")
    return dict(record)


def _evaluation_devicefix_invariants(
    old_payload: Mapping[str, object], new_payload: Mapping[str, object]
) -> dict[str, bool]:
    names = (
        "policy",
        "optimizer",
        "policy_sha256",
        "optimizer_sha256",
        "optimizer_step",
        "rank_rng",
        "iteration",
        "actual_global_transitions",
        "actual_per_task_transitions",
        "task_transitions",
        "curriculum",
        "metrics",
        "next_crossing",
        "evaluation_history",
        "contracts",
        "pre_evaluation",
    )
    invariants = {
        name: _semantic_equal(old_payload[name], new_payload[name]) for name in names
    }
    invariants["payload_except_source_manifest"] = _semantic_equal(
        {key: value for key, value in old_payload.items() if key != "source_manifest"},
        {key: value for key, value in new_payload.items() if key != "source_manifest"},
    )
    return invariants


def _run_evaluation_devicefix_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    snapshot, input_checkpoint, input_record, output_checkpoint, output_record = (
        _evaluation_devicefix_paths(args)
    )
    if (
        _sha256(snapshot) != EVALUATION_DEVICEFIX_SNAPSHOT_SHA256
        or _sha256(input_checkpoint) != EVALUATION_DEVICEFIX_INPUT_SHA256
    ):
        raise ValueError("evaluation devicefix inputs differ from the frozen evidence")
    if output_checkpoint.exists() or output_record.exists():
        raise FileExistsError(
            "evaluation devicefix rebind refuses to overwrite evidence"
        )
    snapshot_manifest = _source_manifest_with_train_sha256(_sha256(snapshot))
    _validate_devicefix_input_record(
        record_path=input_record,
        input_checkpoint=input_checkpoint,
        snapshot_manifest=snapshot_manifest,
    )
    proof = _evaluation_devicefix_ast_proof(snapshot, Path(__file__).resolve())
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("evaluation devicefix input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    if old_payload.get("source_manifest") != snapshot_manifest:
        raise ValueError("evaluation devicefix input source manifest is invalid")
    new_manifest = _current_production_source_manifest()
    script_key = str(REPO_ROOT / "scripts/train_phase6.py")
    if (
        set(old_payload["source_manifest"]) != set(new_manifest)
        or any(
            old_payload["source_manifest"][key] != new_manifest[key]
            for key in old_payload["source_manifest"]
            if key != script_key
        )
        or old_payload["source_manifest"][script_key] == new_manifest[script_key]
    ):
        raise ValueError("evaluation devicefix source delta is not script-only")
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariants = _evaluation_devicefix_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError(
            "evaluation devicefix rebind altered protected checkpoint state"
        )
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("evaluation devicefix checkpoint did not strictly restore")
    record = {
        "checkpoint": {
            "new_path": str(output_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(input_checkpoint),
            "old_sha256": EVALUATION_DEVICEFIX_INPUT_SHA256,
        },
        "device_fix": proof,
        "invariants": invariants,
        "new_source_manifest": new_manifest,
        "old_source_manifest": dict(old_payload["source_manifest"]),
        "prior_harness_rebind": {
            "path": str(input_record),
            "sha256": _sha256(input_record),
        },
        "source_snapshot": {
            "path": str(snapshot),
            "sha256": _sha256(snapshot),
        },
        "status": "ok",
    }
    _atomic_json(output_record, record)
    print(
        "PHASE6_EVALUATION_DEVICEFIX_REBIND=" + json.dumps(record, sort_keys=True),
        flush=True,
    )
    return 0


def _validate_evaluation_devicefix_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    snapshot = recovery / "source_snapshot/train_phase6_pre_devicefix.py"
    input_checkpoint = recovery / "pre_evaluation_harness_rebound.pt"
    input_record = recovery / "source_rebind.json"
    record = _require_exact_mapping(
        _read_json(record_path),
        name="evaluation_devicefix_rebind",
        expected={
            "checkpoint",
            "device_fix",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_harness_rebind",
            "source_snapshot",
            "status",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="evaluation_devicefix_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshot_manifest = _source_manifest_with_train_sha256(_sha256(snapshot))
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": _sha256(rebound_checkpoint),
            "old_path": str(input_checkpoint),
            "old_sha256": EVALUATION_DEVICEFIX_INPUT_SHA256,
        }
        or record["source_snapshot"]
        != {
            "path": str(snapshot),
            "sha256": EVALUATION_DEVICEFIX_SNAPSHOT_SHA256,
        }
        or record["prior_harness_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["old_source_manifest"] != snapshot_manifest
        or record["new_source_manifest"] != _current_production_source_manifest()
        or record["device_fix"]
        != _evaluation_devicefix_ast_proof(snapshot, Path(__file__).resolve())
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("evaluation devicefix rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_production_source_manifest(),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("evaluation devicefix checkpoint is not pre-evaluation")
    return dict(record)


def _torch_context_name(node: ast.With) -> str | None:
    if len(node.items) != 1:
        return None
    expression = node.items[0].context_expr
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and isinstance(expression.func.value, ast.Name)
        and expression.func.value.id == "torch"
        and not expression.args
        and not expression.keywords
    ):
        return None
    return expression.func.attr


def _pair_loop(function: ast.FunctionDef) -> ast.For:
    loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "pair"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "pairs"
    ]
    if len(loops) != 1:
        raise ValueError("paired evaluation must contain exactly one pair loop")
    return loops[0]


def _environment_context_free(function: ast.FunctionDef, *, method: str) -> bool:
    parents = {
        id(child): parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "environment"
        and node.func.attr == method
    ]
    if len(calls) != 1:
        return False
    current: ast.AST | None = calls[0]
    while current is not None:
        current = parents.get(id(current))
        if isinstance(current, ast.With) and _torch_context_name(current) in {
            "inference_mode",
            "no_grad",
        }:
            return False
    return True


def _stepboundary_ast_proof(
    source_snapshot: Path, current_source: Path
) -> dict[str, object]:
    old_functions = _ast_function_dumps(source_snapshot, WORKER_CRITICAL_FUNCTIONS)
    new_functions = _ast_function_dumps(current_source, WORKER_CRITICAL_FUNCTIONS)
    other_critical = {
        name: old_functions[name] == new_functions[name]
        for name in WORKER_CRITICAL_FUNCTIONS
        if name != EVALUATION_DEVICEFIX_FUNCTION
    }
    old_function = _top_level_function(source_snapshot, EVALUATION_DEVICEFIX_FUNCTION)
    new_function = _top_level_function(current_source, EVALUATION_DEVICEFIX_FUNCTION)
    old_policy = _policy_constructor_assignment(old_function)
    new_policy = _policy_constructor_assignment(new_function)
    devicefix_policy_preserved = ast.dump(
        old_policy.value, include_attributes=False
    ) == ast.dump(new_policy.value, include_attributes=False)
    load_calls = [
        node
        for node in ast.walk(new_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "policy"
        and node.func.attr == "load_state_dict"
    ]
    if not devicefix_policy_preserved or len(load_calls) != 1:
        raise ValueError("stepboundary rebind changed the approved device placement")
    old_pair = _pair_loop(old_function)
    new_pair = _pair_loop(new_function)
    if len(old_pair.body) != 3 or len(new_pair.body) != 3:
        raise ValueError("paired evaluation loop shape changed")
    old_context = old_pair.body[1]
    new_horizon = new_pair.body[1]
    if not (
        isinstance(old_context, ast.With)
        and _torch_context_name(old_context) == "inference_mode"
        and len(old_context.body) == 1
        and isinstance(old_context.body[0], ast.For)
        and isinstance(new_horizon, ast.For)
        and len(new_horizon.body) == 2
        and isinstance(new_horizon.body[0], ast.With)
        and _torch_context_name(new_horizon.body[0]) == "no_grad"
        and len(new_horizon.body[0].body) == 1
    ):
        raise ValueError("stepboundary rebind does not isolate the policy forward")
    old_horizon = old_context.body[0]
    new_no_grad = new_horizon.body[0]
    action_ast_equal = ast.dump(
        old_horizon.body[0], include_attributes=False
    ) == ast.dump(new_no_grad.body[0], include_attributes=False)
    step_ast_equal = ast.dump(
        old_horizon.body[1], include_attributes=False
    ) == ast.dump(new_horizon.body[1], include_attributes=False)
    horizon_control_ast_equal = ast.dump(
        old_horizon.target, include_attributes=False
    ) == ast.dump(new_horizon.target, include_attributes=False) and ast.dump(
        old_horizon.iter, include_attributes=False
    ) == ast.dump(new_horizon.iter, include_attributes=False)
    reset_ast_equal = ast.dump(old_pair.body[0], include_attributes=False) == ast.dump(
        new_pair.body[0], include_attributes=False
    )
    episode_count_ast_equal = ast.dump(
        old_pair.body[2], include_attributes=False
    ) == ast.dump(new_pair.body[2], include_attributes=False)
    normalized = copy.deepcopy(new_function)
    normalized_pair = _pair_loop(normalized)
    normalized_pair.body[1] = copy.deepcopy(old_context)
    non_stepboundary_ast_equal = ast.dump(
        normalized, include_attributes=False
    ) == ast.dump(old_function, include_attributes=False)
    no_grad_contexts = [
        node
        for node in ast.walk(new_function)
        if isinstance(node, ast.With) and _torch_context_name(node) == "no_grad"
    ]
    inference_contexts = [
        node
        for node in ast.walk(new_function)
        if isinstance(node, ast.With) and _torch_context_name(node) == "inference_mode"
    ]
    proof = {
        "action_ast_equal": action_ast_equal,
        "approved_differences": [
            "ResidualActorCritic().to(device=torch.device(args.device))",
            "policy.act_inference under torch.no_grad only",
        ],
        "devicefix_policy_preserved": devicefix_policy_preserved,
        "environment_reset_outside_torch_context": _environment_context_free(
            new_function, method="reset"
        ),
        "environment_step_outside_torch_context": _environment_context_free(
            new_function, method="step"
        ),
        "evaluation_function": EVALUATION_DEVICEFIX_FUNCTION,
        "episode_count_ast_equal": episode_count_ast_equal,
        "horizon_control_ast_equal": horizon_control_ast_equal,
        "inference_mode_context_count": len(inference_contexts),
        "load_follows_device_move": new_policy.lineno < load_calls[0].lineno,
        "no_grad_context_count": len(no_grad_contexts),
        "no_grad_body_is_policy_forward_only": action_ast_equal
        and len(new_no_grad.body) == 1,
        "non_stepboundary_ast_equal": non_stepboundary_ast_equal,
        "other_worker_critical_ast_equality": other_critical,
        "reset_ast_equal": reset_ast_equal,
        "step_ast_equal": step_ast_equal,
    }
    if (
        not all(other_critical.values())
        or not devicefix_policy_preserved
        or not action_ast_equal
        or not step_ast_equal
        or not horizon_control_ast_equal
        or not reset_ast_equal
        or not episode_count_ast_equal
        or not non_stepboundary_ast_equal
        or len(no_grad_contexts) != 1
        or inference_contexts
        or not proof["environment_reset_outside_torch_context"]
        or not proof["environment_step_outside_torch_context"]
        or not proof["load_follows_device_move"]
    ):
        raise ValueError("stepboundary rebind changed more than the approved context")
    return proof


def _stepboundary_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    snapshot = recovery / "source_snapshot/train_phase6_pre_stepboundaryfix.py"
    input_checkpoint = recovery / "pre_evaluation_devicefix_rebound.pt"
    input_record = recovery / "source_rebind_devicefix.json"
    output_checkpoint = recovery / "pre_evaluation_stepboundary_rebound.pt"
    output_record = recovery / "source_rebind_stepboundary.json"
    if (
        args.source_snapshot.resolve() != snapshot.resolve()
        or args.pre_checkpoint.resolve() != input_checkpoint.resolve()
        or args.source_rebind.resolve() != input_record.resolve()
        or args.rebound_checkpoint.resolve() != output_checkpoint.resolve()
        or args.source_rebind_output.resolve() != output_record.resolve()
    ):
        raise ValueError("stepboundary rebind paths are fixed for this pilot")
    return snapshot, input_checkpoint, input_record, output_checkpoint, output_record


def _validate_stepboundary_input_record(
    *,
    record_path: Path,
    input_checkpoint: Path,
    snapshot_manifest: Mapping[str, str],
) -> dict[str, object]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    devicefix_snapshot = recovery / "source_snapshot/train_phase6_pre_devicefix.py"
    harness_checkpoint = recovery / "pre_evaluation_harness_rebound.pt"
    harness_record = recovery / "source_rebind.json"
    record = _require_exact_mapping(
        _read_json(record_path),
        name="stepboundary.input_source_rebind",
        expected={
            "checkpoint",
            "device_fix",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_harness_rebind",
            "source_snapshot",
            "status",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="stepboundary.input_source_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    devicefix_manifest = _source_manifest_with_train_sha256(
        EVALUATION_DEVICEFIX_SNAPSHOT_SHA256
    )
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(input_checkpoint),
            "new_sha256": STEPBOUNDARYFIX_INPUT_SHA256,
            "old_path": str(harness_checkpoint),
            "old_sha256": EVALUATION_DEVICEFIX_INPUT_SHA256,
        }
        or _sha256(input_checkpoint) != STEPBOUNDARYFIX_INPUT_SHA256
        or record["source_snapshot"]
        != {
            "path": str(devicefix_snapshot),
            "sha256": EVALUATION_DEVICEFIX_SNAPSHOT_SHA256,
        }
        or record["prior_harness_rebind"]
        != {"path": str(harness_record), "sha256": _sha256(harness_record)}
        or record["old_source_manifest"] != devicefix_manifest
        or record["new_source_manifest"] != dict(snapshot_manifest)
        or record["device_fix"]
        != _evaluation_devicefix_ast_proof(
            devicefix_snapshot,
            Path(
                record_path.parent
                / "source_snapshot/train_phase6_pre_stepboundaryfix.py"
            ),
        )
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("stepboundary input does not bind the devicefix evidence")
    return dict(record)


def _run_stepboundary_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    snapshot, input_checkpoint, input_record, output_checkpoint, output_record = (
        _stepboundary_paths(args)
    )
    if (
        _sha256(snapshot) != STEPBOUNDARYFIX_SNAPSHOT_SHA256
        or _sha256(input_checkpoint) != STEPBOUNDARYFIX_INPUT_SHA256
    ):
        raise ValueError("stepboundary inputs differ from the frozen evidence")
    if output_checkpoint.exists() or output_record.exists():
        raise FileExistsError("stepboundary rebind refuses to overwrite evidence")
    snapshot_manifest = _source_manifest_with_train_sha256(_sha256(snapshot))
    _validate_stepboundary_input_record(
        record_path=input_record,
        input_checkpoint=input_checkpoint,
        snapshot_manifest=snapshot_manifest,
    )
    proof = _stepboundary_ast_proof(snapshot, Path(__file__).resolve())
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("stepboundary input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    if old_payload.get("source_manifest") != snapshot_manifest:
        raise ValueError("stepboundary input source manifest is invalid")
    new_manifest = _current_production_source_manifest()
    script_key = str(REPO_ROOT / "scripts/train_phase6.py")
    if (
        set(old_payload["source_manifest"]) != set(new_manifest)
        or any(
            old_payload["source_manifest"][key] != new_manifest[key]
            for key in old_payload["source_manifest"]
            if key != script_key
        )
        or old_payload["source_manifest"][script_key] == new_manifest[script_key]
    ):
        raise ValueError("stepboundary source delta is not script-only")
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariants = _evaluation_devicefix_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError("stepboundary rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("stepboundary checkpoint did not strictly restore")
    record = {
        "checkpoint": {
            "new_path": str(output_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(input_checkpoint),
            "old_sha256": STEPBOUNDARYFIX_INPUT_SHA256,
        },
        "invariants": invariants,
        "new_source_manifest": new_manifest,
        "old_source_manifest": dict(old_payload["source_manifest"]),
        "prior_devicefix_rebind": {
            "path": str(input_record),
            "sha256": _sha256(input_record),
        },
        "source_snapshot": {"path": str(snapshot), "sha256": _sha256(snapshot)},
        "status": "ok",
        "step_boundary_fix": proof,
    }
    _atomic_json(output_record, record)
    print(
        "PHASE6_STEPBOUNDARY_REBIND=" + json.dumps(record, sort_keys=True),
        flush=True,
    )
    return 0


def _validate_stepboundary_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    snapshot = recovery / "source_snapshot/train_phase6_pre_stepboundaryfix.py"
    validated_source = recovery / "source_snapshot/train_phase6_pre_pairedbatchfix.py"
    input_checkpoint = recovery / "pre_evaluation_devicefix_rebound.pt"
    input_record = recovery / "source_rebind_devicefix.json"
    record = _require_exact_mapping(
        _read_json(record_path),
        name="stepboundary_rebind",
        expected={
            "checkpoint",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_devicefix_rebind",
            "source_snapshot",
            "status",
            "step_boundary_fix",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="stepboundary_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    historical_manifest = dict(record["new_source_manifest"])
    historical_train_key = str(REPO_ROOT / "scripts/train_phase6.py")
    prior_train_manifest = dict(record["old_source_manifest"])
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": _sha256(rebound_checkpoint),
            "old_path": str(input_checkpoint),
            "old_sha256": STEPBOUNDARYFIX_INPUT_SHA256,
        }
        or record["source_snapshot"]
        != {"path": str(snapshot), "sha256": STEPBOUNDARYFIX_SNAPSHOT_SHA256}
        or record["prior_devicefix_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or prior_train_manifest.get(historical_train_key) != _sha256(snapshot)
        or historical_manifest.get(historical_train_key) != _sha256(validated_source)
        or record["step_boundary_fix"]
        != _stepboundary_ast_proof(snapshot, validated_source)
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("stepboundary rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=historical_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("stepboundary checkpoint is not pre-evaluation")
    return dict(record)


def _module_ast_sha256(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return hashlib.sha256(
        ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _validate_pairedbatch_frozen_attempt(manifest_path: Path) -> dict[str, object]:
    """Prove the cancelled attempt has not been altered before a retry rebind."""
    payload = _require_exact_mapping(
        _read_json(manifest_path),
        name="pairedbatch.frozen_attempt_manifest",
        expected={
            "attempt",
            "attempt_relative",
            "files",
            "outer_log",
            "schema_version",
            "status",
        },
    )
    if (
        payload["schema_version"] != "phase6_cancelled_attempt_frozen_manifest_v1"
        or payload["status"] != "cancelled_before_result"
        or not isinstance(payload["files"], list)
        or not payload["files"]
        or not isinstance(payload["outer_log"], Mapping)
    ):
        raise ValueError("frozen cancelled attempt manifest is invalid")
    entries = [*payload["files"], payload["outer_log"]]
    for entry in entries:
        record = _require_exact_mapping(
            entry,
            name="pairedbatch.frozen_attempt_entry",
            expected={
                "absolute_path",
                "mtime_ns",
                "relative_path",
                "sha256",
                "size_bytes",
            },
        )
        path = Path(str(record["absolute_path"]))
        if (
            not path.is_file()
            or str(path.resolve()) != str(path)
            or str(path.relative_to(REPO_ROOT)) != record["relative_path"]
            or path.stat().st_size != record["size_bytes"]
            or path.stat().st_mtime_ns != record["mtime_ns"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError("frozen cancelled attempt evidence changed")
    return dict(payload)


def _pairedbatch_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_pairedbatchfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_pairedbatchfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_pairedbatchfix.py"
    input_checkpoint = recovery / "pre_evaluation_stepboundary_rebound.pt"
    input_record = recovery / "source_rebind_stepboundary.json"
    output_checkpoint = recovery / "pre_evaluation_pairedbatch_rebound.pt"
    output_record = recovery / "source_rebind_pairedbatch.json"
    manifest = recovery / "evaluation_stepboundary_attempt_0002_frozen_manifest.json"
    if (
        args.source_snapshot.resolve() != train_snapshot.resolve()
        or args.pre_checkpoint.resolve() != input_checkpoint.resolve()
        or args.source_rebind.resolve() != input_record.resolve()
        or args.rebound_checkpoint.resolve() != output_checkpoint.resolve()
        or args.source_rebind_output.resolve() != output_record.resolve()
    ):
        raise ValueError("pairedbatch rebind paths are fixed for the preserved pilot")
    return (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        manifest,
    )


def _pairedbatch_ast_proof_for_sources(
    train_snapshot: Path,
    joint_snapshot: Path,
    env_snapshot: Path,
    *,
    candidate_train: Path,
    candidate_joint: Path,
    candidate_env: Path,
) -> dict[str, object]:
    old_functions = _ast_function_dumps(train_snapshot, WORKER_CRITICAL_FUNCTIONS)
    new_functions = _ast_function_dumps(candidate_train, WORKER_CRITICAL_FUNCTIONS)
    return {
        "approved_changed_paths": [
            str(candidate_train),
            str(candidate_joint),
            str(candidate_env),
        ],
        "module_ast_sha256": {
            "joint_runner_current": _module_ast_sha256(candidate_joint),
            "joint_runner_snapshot": _module_ast_sha256(joint_snapshot),
            "residual_env_current": _module_ast_sha256(candidate_env),
            "residual_env_snapshot": _module_ast_sha256(env_snapshot),
            "train_phase6_current": _module_ast_sha256(candidate_train),
            "train_phase6_snapshot": _module_ast_sha256(train_snapshot),
        },
        "worker_critical_ast_changed": {
            name: old_functions[name] != new_functions[name]
            for name in WORKER_CRITICAL_FUNCTIONS
        },
        "worker_critical_functions": list(WORKER_CRITICAL_FUNCTIONS),
    }


def _pairedbatch_ast_proof(
    train_snapshot: Path, joint_snapshot: Path, env_snapshot: Path
) -> dict[str, object]:
    return _pairedbatch_ast_proof_for_sources(
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        candidate_train=Path(__file__).resolve(),
        candidate_joint=REPO_ROOT / "somaforce_cross/learning/joint_runner.py",
        candidate_env=REPO_ROOT / "somaforce_cross/envs/residual_env.py",
    )


def _run_pairedbatch_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        manifest_path,
    ) = _pairedbatch_paths(args)
    expected_snapshots = {
        train_snapshot: PAIREDBATCH_TRAIN_SNAPSHOT_SHA256,
        joint_snapshot: PAIREDBATCH_JOINT_SNAPSHOT_SHA256,
        env_snapshot: PAIREDBATCH_ENV_SNAPSHOT_SHA256,
        input_checkpoint: PAIREDBATCH_INPUT_SHA256,
    }
    if any(_sha256(path) != digest for path, digest in expected_snapshots.items()):
        raise ValueError("pairedbatch rebind inputs differ from frozen evidence")
    if output_checkpoint.exists() or output_record.exists():
        raise FileExistsError("pairedbatch rebind refuses to overwrite evidence")
    frozen_attempt = _validate_pairedbatch_frozen_attempt(manifest_path)
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    prior = _validate_stepboundary_rebind_record(
        record_path=input_record,
        rebound_checkpoint=input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("pairedbatch input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    if old_payload.get("source_manifest") != prior["new_source_manifest"]:
        raise ValueError("pairedbatch input checkpoint source manifest is invalid")
    new_manifest = _current_pairedbatch_source_manifest()
    allowed_changed = {
        str(REPO_ROOT / "scripts/train_phase6.py"),
        str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py"),
        str(REPO_ROOT / "somaforce_cross/envs/residual_env.py"),
    }
    old_manifest = dict(old_payload["source_manifest"])
    if any(
        old_manifest[key] != new_manifest[key]
        for key in old_manifest
        if key not in allowed_changed
    ):
        raise ValueError("pairedbatch rebind changed a frozen source manifest entry")
    if str(REPO_ROOT / "somaforce_cross/envs/residual_env.py") in old_manifest:
        raise ValueError("pairedbatch input unexpectedly already binds residual_env.py")
    proof = _pairedbatch_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariants = _evaluation_devicefix_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError("pairedbatch rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("pairedbatch rebound checkpoint did not strictly restore")
    record = {
        "approved_changed_paths": proof["approved_changed_paths"],
        "checkpoint": {
            "new_path": str(output_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(input_checkpoint),
            "old_sha256": PAIREDBATCH_INPUT_SHA256,
        },
        "frozen_attempt_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "status": frozen_attempt["status"],
        },
        "invariants": invariants,
        "new_source_manifest": new_manifest,
        "old_source_manifest": old_manifest,
        "prior_stepboundary_rebind": {
            "path": str(input_record),
            "sha256": _sha256(input_record),
        },
        "source_snapshots": {
            "joint_runner": {
                "path": str(joint_snapshot),
                "sha256": _sha256(joint_snapshot),
            },
            "residual_env": {
                "path": str(env_snapshot),
                "sha256": _sha256(env_snapshot),
            },
            "train_phase6": {
                "path": str(train_snapshot),
                "sha256": _sha256(train_snapshot),
            },
        },
        "status": "ok",
        "worker_ast_scope": proof,
    }
    _atomic_json(output_record, record)
    print("PHASE6_PAIREDBATCH_REBIND=" + json.dumps(record, sort_keys=True), flush=True)
    return 0


def _validate_pairedbatch_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    manifest = recovery / "evaluation_stepboundary_attempt_0002_frozen_manifest.json"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_pairedbatchfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_pairedbatchfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_pairedbatchfix.py"
    input_checkpoint = recovery / "pre_evaluation_stepboundary_rebound.pt"
    input_record = recovery / "source_rebind_stepboundary.json"
    record = _require_exact_mapping(
        _read_json(record_path),
        name="pairedbatch_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "frozen_attempt_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_stepboundary_rebind",
            "source_snapshots",
            "status",
            "worker_ast_scope",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="pairedbatch_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshots = _require_exact_mapping(
        record["source_snapshots"],
        name="pairedbatch_rebind.source_snapshots",
        expected={"joint_runner", "residual_env", "train_phase6"},
    )
    expected_snapshots = {
        "joint_runner": (joint_snapshot, PAIREDBATCH_JOINT_SNAPSHOT_SHA256),
        "residual_env": (env_snapshot, PAIREDBATCH_ENV_SNAPSHOT_SHA256),
        "train_phase6": (train_snapshot, PAIREDBATCH_TRAIN_SNAPSHOT_SHA256),
    }
    if any(
        snapshots[name] != {"path": str(path), "sha256": digest}
        for name, (path, digest) in expected_snapshots.items()
    ):
        raise ValueError("pairedbatch source snapshot record is invalid")
    frozen = _validate_pairedbatch_frozen_attempt(manifest)
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": _sha256(rebound_checkpoint),
            "old_path": str(input_checkpoint),
            "old_sha256": PAIREDBATCH_INPUT_SHA256,
        }
        or record["frozen_attempt_manifest"]
        != {
            "path": str(manifest),
            "sha256": _sha256(manifest),
            "status": frozen["status"],
        }
        or record["prior_stepboundary_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["new_source_manifest"] != _current_pairedbatch_source_manifest()
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
        or record["worker_ast_scope"]
        != _pairedbatch_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
    ):
        raise ValueError("pairedbatch rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_pairedbatch_source_manifest(),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("pairedbatch checkpoint is not pre-evaluation")
    return dict(record)


def _validate_rankseed_failed_smoke_manifest(manifest_path: Path) -> dict[str, object]:
    """Reject any post-freeze mutation of the failed rank-seed smoke evidence."""
    payload = _require_exact_mapping(
        _read_json(manifest_path),
        name="rankseed.failed_smoke_manifest",
        expected={
            "attempt",
            "attempt_relative",
            "files",
            "outer_log",
            "schema_version",
            "status",
        },
    )
    if (
        payload["schema_version"] != "phase6_rankseed_failed_smoke_frozen_manifest_v1"
        or payload["status"] != "failed_before_episode"
        or not isinstance(payload["files"], list)
        or len(payload["files"]) != 20
        or not isinstance(payload["outer_log"], Mapping)
    ):
        raise ValueError("rankseed failed smoke manifest is invalid")
    attempt = Path(str(payload["attempt"]))
    if (
        not attempt.is_dir()
        or str(attempt.resolve()) != str(attempt)
        or str(attempt.relative_to(REPO_ROOT)) != payload["attempt_relative"]
    ):
        raise ValueError("rankseed failed smoke attempt path is invalid")
    entries = [*payload["files"], payload["outer_log"]]
    for entry in entries:
        record = _require_exact_mapping(
            entry,
            name="rankseed.failed_smoke_entry",
            expected={
                "absolute_path",
                "mtime_ns",
                "relative_path",
                "sha256",
                "size_bytes",
            },
        )
        path = Path(str(record["absolute_path"]))
        if (
            not path.is_file()
            or str(path.resolve()) != str(path)
            or str(path.relative_to(REPO_ROOT)) != record["relative_path"]
            or path.stat().st_size != record["size_bytes"]
            or path.stat().st_mtime_ns != record["mtime_ns"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError("rankseed failed smoke evidence changed")
    return dict(payload)


def _rankseed_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_rankseedfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_rankseedfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_rankseedfix.py"
    input_checkpoint = recovery / "pre_evaluation_pairedbatch_rebound.pt"
    input_record = recovery / "source_rebind_pairedbatch.json"
    output_checkpoint = recovery / "pre_evaluation_rankseed_rebound.pt"
    output_record = recovery / "source_rebind_rankseed.json"
    if (
        args.source_snapshot.resolve() != train_snapshot.resolve()
        or args.pre_checkpoint.resolve() != input_checkpoint.resolve()
        or args.source_rebind.resolve() != input_record.resolve()
        or args.rebound_checkpoint.resolve() != output_checkpoint.resolve()
        or args.source_rebind_output.resolve() != output_record.resolve()
    ):
        raise ValueError("rankseed rebind paths are fixed for the preserved pilot")
    return (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        recovery / "evaluation_pairedbatch_smoke_0003_frozen_manifest.json",
    )


def _rankseed_ast_proof_for_sources(
    train_snapshot: Path,
    joint_snapshot: Path,
    env_snapshot: Path,
    *,
    candidate_train: Path,
    candidate_joint: Path,
    candidate_env: Path,
) -> dict[str, object]:
    current_train = candidate_train
    current_joint = candidate_joint
    current_env = candidate_env
    old_functions = _ast_function_dumps(train_snapshot, WORKER_CRITICAL_FUNCTIONS)
    new_functions = _ast_function_dumps(current_train, WORKER_CRITICAL_FUNCTIONS)
    helper = _top_level_function(current_joint, "paired_evaluation_rank_seed")
    schedule = _top_level_function(current_joint, "paired_evaluation_schedule")
    runner = _top_level_function(current_train, "_run_paired_evaluation_rank")
    helper_return = [node for node in helper.body if isinstance(node, ast.Return)]
    helper_formula = (
        len(helper_return) == 1
        and isinstance(helper_return[0].value, ast.BinOp)
        and isinstance(helper_return[0].value.op, ast.Add)
        and isinstance(helper_return[0].value.left, ast.Name)
        and helper_return[0].value.left.id == "evaluation_seed"
        and isinstance(helper_return[0].value.right, ast.BinOp)
        and isinstance(helper_return[0].value.right.op, ast.Mult)
        and isinstance(helper_return[0].value.right.left, ast.Name)
        and helper_return[0].value.right.left.id == "rank"
        and isinstance(helper_return[0].value.right.right, ast.Constant)
        and helper_return[0].value.right.right.value == 1_000_000_000
    )
    helper_validates_inputs = any(
        isinstance(node, ast.For) for node in ast.walk(helper)
    ) and any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        for node in ast.walk(helper)
    )
    schedule_calls_helper = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "task_base"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "paired_evaluation_rank_seed"
        for node in ast.walk(schedule)
    )
    runner_calls_helper = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "rank_seed"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "paired_evaluation_rank_seed"
        for node in ast.walk(runner)
    )
    environment_seeds = [
        keyword.value
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_environment"
        for keyword in node.keywords
        if keyword.arg == "seed"
    ]
    runner_uses_rank_seed = (
        len(environment_seeds) == 1
        and isinstance(environment_seeds[0], ast.Name)
        and environment_seeds[0].id == "rank_seed"
    )
    runner_has_no_formula = not any(
        isinstance(node, ast.Constant) and node.value == 1_000_000_000
        for node in ast.walk(runner)
    )
    other_critical = {
        name: old_functions[name] == new_functions[name]
        for name in WORKER_CRITICAL_FUNCTIONS
        if name != "_run_paired_evaluation_rank"
    }
    proof = {
        "approved_differences": [
            "paired_evaluation_rank_seed shared helper",
            "paired_evaluation_schedule task_base uses shared helper",
            "_run_paired_evaluation_rank builds both modes with rank_seed",
            "rankseed rebind and recovery harness",
        ],
        "helper_formula": helper_formula,
        "helper_validates_inputs": helper_validates_inputs,
        "other_worker_critical_ast_equality": other_critical,
        "residual_env_byte_equality": _sha256(current_env) == _sha256(env_snapshot),
        "runner_calls_helper": runner_calls_helper,
        "runner_has_no_rank_formula": runner_has_no_formula,
        "runner_uses_rank_seed": runner_uses_rank_seed,
        "schedule_calls_helper": schedule_calls_helper,
        "source_snapshots": {
            "joint_runner": _sha256(joint_snapshot),
            "residual_env": _sha256(env_snapshot),
            "train_phase6": _sha256(train_snapshot),
        },
    }
    if (
        not helper_formula
        or not helper_validates_inputs
        or not schedule_calls_helper
        or not runner_calls_helper
        or not runner_uses_rank_seed
        or not runner_has_no_formula
        or not proof["residual_env_byte_equality"]
        or not all(other_critical.values())
    ):
        raise ValueError("rankseed rebind changed more than the approved seed wiring")
    return proof


def _rankseed_ast_proof(
    train_snapshot: Path, joint_snapshot: Path, env_snapshot: Path
) -> dict[str, object]:
    return _rankseed_ast_proof_for_sources(
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        candidate_train=Path(__file__).resolve(),
        candidate_joint=REPO_ROOT / "somaforce_cross/learning/joint_runner.py",
        candidate_env=REPO_ROOT / "somaforce_cross/envs/residual_env.py",
    )


def _validate_historical_pairedbatch_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    pair_train = recovery / "source_snapshot/train_phase6_pre_pairedbatchfix.py"
    pair_joint = recovery / "source_snapshot/joint_runner_pre_pairedbatchfix.py"
    pair_env = recovery / "source_snapshot/residual_env_pre_pairedbatchfix.py"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_rankseedfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_rankseedfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_rankseedfix.py"
    stepboundary_checkpoint = recovery / "pre_evaluation_stepboundary_rebound.pt"
    stepboundary_record = recovery / "source_rebind_stepboundary.json"
    frozen_manifest = (
        recovery / "evaluation_stepboundary_attempt_0002_frozen_manifest.json"
    )
    record = _require_exact_mapping(
        _read_json(record_path),
        name="rankseed.historical_pairedbatch_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "frozen_attempt_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_stepboundary_rebind",
            "source_snapshots",
            "status",
            "worker_ast_scope",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="rankseed.historical_pairedbatch_checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshots = _require_exact_mapping(
        record["source_snapshots"],
        name="rankseed.historical_pairedbatch_snapshots",
        expected={"joint_runner", "residual_env", "train_phase6"},
    )
    expected_snapshots = {
        "joint_runner": (pair_joint, PAIREDBATCH_JOINT_SNAPSHOT_SHA256),
        "residual_env": (pair_env, PAIREDBATCH_ENV_SNAPSHOT_SHA256),
        "train_phase6": (pair_train, PAIREDBATCH_TRAIN_SNAPSHOT_SHA256),
    }
    historical_manifest = dict(record["new_source_manifest"])
    expected_historical = {
        str(REPO_ROOT / "scripts/train_phase6.py"): RANKSEED_TRAIN_SNAPSHOT_SHA256,
        str(
            REPO_ROOT / "somaforce_cross/learning/joint_runner.py"
        ): RANKSEED_JOINT_SNAPSHOT_SHA256,
        str(
            REPO_ROOT / "somaforce_cross/envs/residual_env.py"
        ): RANKSEED_ENV_SNAPSHOT_SHA256,
    }
    frozen = _validate_pairedbatch_frozen_attempt(frozen_manifest)
    historical_proof = _pairedbatch_ast_proof_for_sources(
        pair_train,
        pair_joint,
        pair_env,
        candidate_train=train_snapshot,
        candidate_joint=joint_snapshot,
        candidate_env=env_snapshot,
    )
    historical_proof["approved_changed_paths"] = [
        str(REPO_ROOT / "scripts/train_phase6.py"),
        str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py"),
        str(REPO_ROOT / "somaforce_cross/envs/residual_env.py"),
    ]
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": RANKSEED_INPUT_SHA256,
            "old_path": str(stepboundary_checkpoint),
            "old_sha256": PAIREDBATCH_INPUT_SHA256,
        }
        or _sha256(rebound_checkpoint) != RANKSEED_INPUT_SHA256
        or any(
            snapshots[name] != {"path": str(path), "sha256": digest}
            for name, (path, digest) in expected_snapshots.items()
        )
        or any(
            historical_manifest.get(path) != digest
            for path, digest in expected_historical.items()
        )
        or record["prior_stepboundary_rebind"]
        != {"path": str(stepboundary_record), "sha256": _sha256(stepboundary_record)}
        or record["frozen_attempt_manifest"]
        != {
            "path": str(frozen_manifest),
            "sha256": _sha256(frozen_manifest),
            "status": frozen["status"],
        }
        or record["worker_ast_scope"] != historical_proof
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("historical pairedbatch rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=historical_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("historical pairedbatch checkpoint is not pre-evaluation")
    return dict(record)


def _run_rankseed_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        failed_manifest,
    ) = _rankseed_paths(args)
    expected_snapshots = {
        train_snapshot: RANKSEED_TRAIN_SNAPSHOT_SHA256,
        joint_snapshot: RANKSEED_JOINT_SNAPSHOT_SHA256,
        env_snapshot: RANKSEED_ENV_SNAPSHOT_SHA256,
        input_checkpoint: RANKSEED_INPUT_SHA256,
    }
    if any(_sha256(path) != digest for path, digest in expected_snapshots.items()):
        raise ValueError("rankseed rebind inputs differ from frozen evidence")
    if output_checkpoint.exists() or output_record.exists():
        raise FileExistsError("rankseed rebind refuses to overwrite evidence")
    failed_smoke = _validate_rankseed_failed_smoke_manifest(failed_manifest)
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    prior = _validate_historical_pairedbatch_rebind_record(
        record_path=input_record,
        rebound_checkpoint=input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("rankseed input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    if old_payload.get("source_manifest") != prior["new_source_manifest"]:
        raise ValueError("rankseed input checkpoint source manifest is invalid")
    new_manifest = _current_pairedbatch_source_manifest()
    allowed_changed = {
        str(REPO_ROOT / "scripts/train_phase6.py"),
        str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py"),
    }
    old_manifest = dict(old_payload["source_manifest"])
    env_key = str(REPO_ROOT / "somaforce_cross/envs/residual_env.py")
    if (
        set(old_manifest) != set(new_manifest)
        or old_manifest.get(env_key) != new_manifest.get(env_key)
        or any(
            old_manifest[key] != new_manifest[key]
            for key in old_manifest
            if key not in allowed_changed
        )
    ):
        raise ValueError(
            "rankseed source delta is not limited to train and joint runner"
        )
    proof = _rankseed_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariants = _evaluation_devicefix_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError("rankseed rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("rankseed rebound checkpoint did not strictly restore")
    record = {
        "approved_changed_paths": sorted(allowed_changed),
        "checkpoint": {
            "new_path": str(output_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(input_checkpoint),
            "old_sha256": RANKSEED_INPUT_SHA256,
        },
        "failed_smoke_manifest": {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed_smoke["status"],
        },
        "invariants": invariants,
        "new_source_manifest": new_manifest,
        "old_source_manifest": old_manifest,
        "prior_pairedbatch_rebind": {
            "path": str(input_record),
            "sha256": _sha256(input_record),
        },
        "source_snapshots": {
            "joint_runner": {
                "path": str(joint_snapshot),
                "sha256": _sha256(joint_snapshot),
            },
            "residual_env": {
                "path": str(env_snapshot),
                "sha256": _sha256(env_snapshot),
            },
            "train_phase6": {
                "path": str(train_snapshot),
                "sha256": _sha256(train_snapshot),
            },
        },
        "status": "ok",
        "rankseed_fix": proof,
    }
    _atomic_json(output_record, record)
    print("PHASE6_RANKSEED_REBIND=" + json.dumps(record, sort_keys=True), flush=True)
    return 0


def _validate_rankseed_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_rankseedfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_rankseedfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_rankseedfix.py"
    input_checkpoint = recovery / "pre_evaluation_pairedbatch_rebound.pt"
    input_record = recovery / "source_rebind_pairedbatch.json"
    failed_manifest = (
        recovery / "evaluation_pairedbatch_smoke_0003_frozen_manifest.json"
    )
    record = _require_exact_mapping(
        _read_json(record_path),
        name="rankseed_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "failed_smoke_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_pairedbatch_rebind",
            "rankseed_fix",
            "source_snapshots",
            "status",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="rankseed_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshots = _require_exact_mapping(
        record["source_snapshots"],
        name="rankseed_rebind.source_snapshots",
        expected={"joint_runner", "residual_env", "train_phase6"},
    )
    expected_snapshots = {
        "joint_runner": (joint_snapshot, RANKSEED_JOINT_SNAPSHOT_SHA256),
        "residual_env": (env_snapshot, RANKSEED_ENV_SNAPSHOT_SHA256),
        "train_phase6": (train_snapshot, RANKSEED_TRAIN_SNAPSHOT_SHA256),
    }
    failed = _validate_rankseed_failed_smoke_manifest(failed_manifest)
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": _sha256(rebound_checkpoint),
            "old_path": str(input_checkpoint),
            "old_sha256": RANKSEED_INPUT_SHA256,
        }
        or record["approved_changed_paths"]
        != sorted(
            (
                str(REPO_ROOT / "scripts/train_phase6.py"),
                str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py"),
            )
        )
        or any(
            snapshots[name] != {"path": str(path), "sha256": digest}
            for name, (path, digest) in expected_snapshots.items()
        )
        or record["failed_smoke_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["prior_pairedbatch_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["new_source_manifest"] != _current_pairedbatch_source_manifest()
        or record["rankseed_fix"]
        != _rankseed_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("rankseed rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_pairedbatch_source_manifest(),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("rankseed checkpoint is not pre-evaluation")
    return dict(record)


def _validate_episodeorder_failed_smoke_manifest(
    manifest_path: Path,
) -> dict[str, object]:
    """Reject mutation of the partial-episode rank-seed smoke evidence."""
    payload = _require_exact_mapping(
        _read_json(manifest_path),
        name="episodeorder.failed_smoke_manifest",
        expected={
            "attempt",
            "attempt_relative",
            "files",
            "outer_log",
            "schema_version",
            "status",
        },
    )
    if (
        payload["schema_version"]
        != "phase6_episodeorder_failed_smoke_frozen_manifest_v1"
        or payload["status"] != "failed_after_partial_episodes"
        or not isinstance(payload["files"], list)
        or len(payload["files"]) != 17
        or not isinstance(payload["outer_log"], Mapping)
    ):
        raise ValueError("episodeorder failed smoke manifest is invalid")
    attempt = Path(str(payload["attempt"]))
    if (
        not attempt.is_dir()
        or str(attempt.resolve()) != str(attempt)
        or str(attempt.relative_to(REPO_ROOT)) != payload["attempt_relative"]
    ):
        raise ValueError("episodeorder failed smoke attempt path is invalid")
    entries = [*payload["files"], payload["outer_log"]]
    for entry in entries:
        record = _require_exact_mapping(
            entry,
            name="episodeorder.failed_smoke_entry",
            expected={
                "absolute_path",
                "mtime_ns",
                "relative_path",
                "sha256",
                "size_bytes",
            },
        )
        path = Path(str(record["absolute_path"]))
        if (
            not path.is_file()
            or str(path.resolve()) != str(path)
            or str(path.relative_to(REPO_ROOT)) != record["relative_path"]
            or path.stat().st_size != record["size_bytes"]
            or path.stat().st_mtime_ns != record["mtime_ns"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError("episodeorder failed smoke evidence changed")
    return dict(payload)


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"{path} must contain exactly one {class_name}")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError(f"{class_name} must contain exactly one {method_name}")
    return methods[0]


def _evaluation_metadata_commit_lines(function: ast.FunctionDef) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            value = target.value
            if isinstance(value, ast.Attribute) and value.attr in {
                "_evaluation_subset",
                "_evaluation_parked",
            }:
                lines.append(node.lineno)
    return lines


def _self_method_identity(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    owner = call.func.value
    if (
        isinstance(owner, ast.Attribute)
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "self"
    ):
        return f"self.{owner.attr}.{call.func.attr}"
    if isinstance(owner, ast.Name) and owner.id == "self":
        return f"self.{call.func.attr}"
    return None


def _episodeorder_ast_proof_for_sources(
    train_snapshot: Path,
    joint_snapshot: Path,
    env_snapshot: Path,
    *,
    candidate_train: Path,
    candidate_joint: Path,
    candidate_env: Path,
) -> dict[str, object]:
    current_train = candidate_train
    current_joint = candidate_joint
    current_env = candidate_env
    old_functions = _ast_function_dumps(train_snapshot, WORKER_CRITICAL_FUNCTIONS)
    new_functions = _ast_function_dumps(current_train, WORKER_CRITICAL_FUNCTIONS)
    old_reset = _class_method(
        env_snapshot, "SomaForceResidualEnv", "_reset_scaffold_only_idx"
    )
    new_reset = _class_method(
        current_env, "SomaForceResidualEnv", "_reset_scaffold_only_idx"
    )
    completion_calls = [
        node
        for node in ast.walk(new_reset)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_record_completed_episodes"
    ]
    reset_metric_calls = [
        node
        for node in ast.walk(new_reset)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_reset_episode_metrics"
    ]
    self_calls = [node for node in ast.walk(new_reset) if isinstance(node, ast.Call)]
    sample_calls = [
        node
        for node in self_calls
        if _self_method_identity(node) == "self.mismatch_sampler.sample"
    ]
    commit_lines = _evaluation_metadata_commit_lines(new_reset)
    completion_line = completion_calls[0].lineno if len(completion_calls) == 1 else -1
    required_reset_plan = {
        "self.parameter_store.validate_rows",
        "self.random_stream.validate_seeds",
        "self.scaffold_state.validate_reset",
        "self.action_sink.validate_reset",
        "self.scaffold_runtime.validate",
        "self.adapter.validate_reset",
        "self.adapter.validate_runtime_parameters",
    }
    reset_plan_calls = [
        node
        for node in self_calls
        if _self_method_identity(node) in required_reset_plan
        and node.lineno < completion_line
    ]
    reset_plan_identities = [_self_method_identity(node) for node in reset_plan_calls]
    applied_readback_calls = [
        node
        for node in self_calls
        if _self_method_identity(node) == "self.parameter_store.validate_rows"
        and node.lineno > completion_line
        and [
            argument.id if isinstance(argument, ast.Name) else None
            for argument in node.args
        ]
        == ["ids", "applied_physics", "applied_scaffold", "applied_sensor"]
        and not node.keywords
    ]
    proof = {
        "approved_differences": [
            "residual_env next-slot metadata staged until completion recording",
            "episodeorder rebind and recovery harness",
        ],
        "joint_runner_byte_equality": _sha256(current_joint) == _sha256(joint_snapshot),
        "old_reset_ast_differs": ast.dump(old_reset, include_attributes=False)
        != ast.dump(new_reset, include_attributes=False),
        "applied_readback_validation_follows_record": len(applied_readback_calls) == 1
        and completion_line > 0,
        "record_before_next_slot_commit": bool(commit_lines)
        and completion_line > 0
        and completion_line < min(commit_lines),
        "record_before_reset_metrics": len(reset_metric_calls) == 1
        and completion_line > 0
        and completion_line < reset_metric_calls[0].lineno,
        "reset_plan_validation_precedes_record": len(sample_calls) == 1
        and sample_calls[0].lineno < completion_line
        and len(reset_plan_calls) == len(required_reset_plan)
        and set(reset_plan_identities) == required_reset_plan,
        "source_snapshots": {
            "joint_runner": _sha256(joint_snapshot),
            "residual_env": _sha256(env_snapshot),
            "train_phase6": _sha256(train_snapshot),
        },
        "staged_subset_and_parked_values": all(
            any(
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else (node.target,)
                    )
                )
                for node in ast.walk(new_reset)
            )
            for name in ("staged_evaluation_subsets", "staged_evaluation_parked")
        ),
        "worker_critical_ast_equality": {
            name: old_functions[name] == new_functions[name]
            for name in WORKER_CRITICAL_FUNCTIONS
        },
    }
    if (
        not proof["joint_runner_byte_equality"]
        or not proof["old_reset_ast_differs"]
        or not proof["record_before_next_slot_commit"]
        or not proof["record_before_reset_metrics"]
        or not proof["reset_plan_validation_precedes_record"]
        or not proof["applied_readback_validation_follows_record"]
        or not proof["staged_subset_and_parked_values"]
        or not all(proof["worker_critical_ast_equality"].values())
    ):
        raise ValueError("episodeorder rebind changed more than reset metadata order")
    return proof


def _episodeorder_ast_proof(
    train_snapshot: Path, joint_snapshot: Path, env_snapshot: Path
) -> dict[str, object]:
    return _episodeorder_ast_proof_for_sources(
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        candidate_train=Path(__file__).resolve(),
        candidate_joint=REPO_ROOT / "somaforce_cross/learning/joint_runner.py",
        candidate_env=REPO_ROOT / "somaforce_cross/envs/residual_env.py",
    )


def _episodeorder_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_episodeorderfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_episodeorderfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_episodeorderfix.py"
    input_checkpoint = recovery / "pre_evaluation_rankseed_rebound.pt"
    input_record = recovery / "source_rebind_rankseed.json"
    output_checkpoint = recovery / "pre_evaluation_episodeorder_rebound.pt"
    output_record = recovery / "source_rebind_episodeorder.json"
    if (
        args.source_snapshot.resolve() != train_snapshot.resolve()
        or args.pre_checkpoint.resolve() != input_checkpoint.resolve()
        or args.source_rebind.resolve() != input_record.resolve()
        or args.rebound_checkpoint.resolve() != output_checkpoint.resolve()
        or args.source_rebind_output.resolve() != output_record.resolve()
    ):
        raise ValueError("episodeorder rebind paths are fixed for the preserved pilot")
    return (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        recovery / "evaluation_rankseed_smoke_0004_frozen_manifest.json",
    )


def _validate_historical_rankseed_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_rankseedfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_rankseedfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_rankseedfix.py"
    validated_train = recovery / "source_snapshot/train_phase6_pre_episodeorderfix.py"
    validated_joint = recovery / "source_snapshot/joint_runner_pre_episodeorderfix.py"
    validated_env = recovery / "source_snapshot/residual_env_pre_episodeorderfix.py"
    input_checkpoint = recovery / "pre_evaluation_pairedbatch_rebound.pt"
    input_record = recovery / "source_rebind_pairedbatch.json"
    failed_manifest = (
        recovery / "evaluation_pairedbatch_smoke_0003_frozen_manifest.json"
    )
    record = _require_exact_mapping(
        _read_json(record_path),
        name="episodeorder.historical_rankseed_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "failed_smoke_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_pairedbatch_rebind",
            "rankseed_fix",
            "source_snapshots",
            "status",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="episodeorder.historical_rankseed_checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshots = _require_exact_mapping(
        record["source_snapshots"],
        name="episodeorder.historical_rankseed_snapshots",
        expected={"joint_runner", "residual_env", "train_phase6"},
    )
    expected_snapshots = {
        "joint_runner": (joint_snapshot, RANKSEED_JOINT_SNAPSHOT_SHA256),
        "residual_env": (env_snapshot, RANKSEED_ENV_SNAPSHOT_SHA256),
        "train_phase6": (train_snapshot, RANKSEED_TRAIN_SNAPSHOT_SHA256),
    }
    expected_old = {
        str(REPO_ROOT / "scripts/train_phase6.py"): RANKSEED_TRAIN_SNAPSHOT_SHA256,
        str(
            REPO_ROOT / "somaforce_cross/learning/joint_runner.py"
        ): RANKSEED_JOINT_SNAPSHOT_SHA256,
        str(
            REPO_ROOT / "somaforce_cross/envs/residual_env.py"
        ): RANKSEED_ENV_SNAPSHOT_SHA256,
    }
    expected_new = {
        str(REPO_ROOT / "scripts/train_phase6.py"): EPISODEORDER_TRAIN_SNAPSHOT_SHA256,
        str(
            REPO_ROOT / "somaforce_cross/learning/joint_runner.py"
        ): EPISODEORDER_JOINT_SNAPSHOT_SHA256,
        str(
            REPO_ROOT / "somaforce_cross/envs/residual_env.py"
        ): EPISODEORDER_ENV_SNAPSHOT_SHA256,
    }
    failed = _validate_rankseed_failed_smoke_manifest(failed_manifest)
    historical_proof = _rankseed_ast_proof_for_sources(
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        candidate_train=validated_train,
        candidate_joint=validated_joint,
        candidate_env=validated_env,
    )
    prior = _validate_historical_pairedbatch_rebind_record(
        record_path=input_record,
        rebound_checkpoint=input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": EPISODEORDER_INPUT_SHA256,
            "old_path": str(input_checkpoint),
            "old_sha256": RANKSEED_INPUT_SHA256,
        }
        or _sha256(rebound_checkpoint) != EPISODEORDER_INPUT_SHA256
        or any(
            snapshots[name] != {"path": str(path), "sha256": digest}
            for name, (path, digest) in expected_snapshots.items()
        )
        or any(
            record["old_source_manifest"].get(path) != digest
            for path, digest in expected_old.items()
        )
        or any(
            record["new_source_manifest"].get(path) != digest
            for path, digest in expected_new.items()
        )
        or record["prior_pairedbatch_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["failed_smoke_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["rankseed_fix"] != historical_proof
        or not prior
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("historical rankseed rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=dict(record["new_source_manifest"]),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("historical rankseed checkpoint is not pre-evaluation")
    return dict(record)


def _run_episodeorder_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        failed_manifest,
    ) = _episodeorder_paths(args)
    expected = {
        train_snapshot: EPISODEORDER_TRAIN_SNAPSHOT_SHA256,
        joint_snapshot: EPISODEORDER_JOINT_SNAPSHOT_SHA256,
        env_snapshot: EPISODEORDER_ENV_SNAPSHOT_SHA256,
        input_checkpoint: EPISODEORDER_INPUT_SHA256,
    }
    if any(_sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("episodeorder rebind inputs differ from frozen evidence")
    if output_checkpoint.exists() or output_record.exists():
        raise FileExistsError("episodeorder rebind refuses to overwrite evidence")
    failed = _validate_episodeorder_failed_smoke_manifest(failed_manifest)
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    prior = _validate_historical_rankseed_rebind_record(
        record_path=input_record,
        rebound_checkpoint=input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("episodeorder input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    if old_payload.get("source_manifest") != prior["new_source_manifest"]:
        raise ValueError("episodeorder input checkpoint source manifest is invalid")
    new_manifest = _current_pairedbatch_source_manifest()
    allowed_changed = {
        str(REPO_ROOT / "scripts/train_phase6.py"),
        str(REPO_ROOT / "somaforce_cross/envs/residual_env.py"),
    }
    old_manifest = dict(old_payload["source_manifest"])
    joint_key = str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py")
    if (
        set(old_manifest) != set(new_manifest)
        or old_manifest.get(joint_key) != new_manifest.get(joint_key)
        or any(
            old_manifest[key] != new_manifest[key]
            for key in old_manifest
            if key not in allowed_changed
        )
        or any(old_manifest[key] == new_manifest[key] for key in allowed_changed)
    ):
        raise ValueError("episodeorder source delta is not limited to train and env")
    proof = _episodeorder_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariants = _evaluation_devicefix_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError("episodeorder rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("episodeorder checkpoint did not strictly restore")
    record = {
        "approved_changed_paths": sorted(allowed_changed),
        "checkpoint": {
            "new_path": str(output_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(input_checkpoint),
            "old_sha256": EPISODEORDER_INPUT_SHA256,
        },
        "episode_order_fix": proof,
        "failed_smoke_manifest": {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        },
        "invariants": invariants,
        "new_source_manifest": new_manifest,
        "old_source_manifest": old_manifest,
        "prior_rankseed_rebind": {
            "path": str(input_record),
            "sha256": _sha256(input_record),
        },
        "source_snapshots": {
            "joint_runner": {
                "path": str(joint_snapshot),
                "sha256": _sha256(joint_snapshot),
            },
            "residual_env": {
                "path": str(env_snapshot),
                "sha256": _sha256(env_snapshot),
            },
            "train_phase6": {
                "path": str(train_snapshot),
                "sha256": _sha256(train_snapshot),
            },
        },
        "status": "ok",
    }
    _atomic_json(output_record, record)
    print(
        "PHASE6_EPISODEORDER_REBIND=" + json.dumps(record, sort_keys=True), flush=True
    )
    return 0


def _validate_episodeorder_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_episodeorderfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_episodeorderfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_episodeorderfix.py"
    input_checkpoint = recovery / "pre_evaluation_rankseed_rebound.pt"
    input_record = recovery / "source_rebind_rankseed.json"
    failed_manifest = recovery / "evaluation_rankseed_smoke_0004_frozen_manifest.json"
    record = _require_exact_mapping(
        _read_json(record_path),
        name="episodeorder_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "episode_order_fix",
            "failed_smoke_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_rankseed_rebind",
            "source_snapshots",
            "status",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="episodeorder_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshots = _require_exact_mapping(
        record["source_snapshots"],
        name="episodeorder_rebind.source_snapshots",
        expected={"joint_runner", "residual_env", "train_phase6"},
    )
    failed = _validate_episodeorder_failed_smoke_manifest(failed_manifest)
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": _sha256(rebound_checkpoint),
            "old_path": str(input_checkpoint),
            "old_sha256": EPISODEORDER_INPUT_SHA256,
        }
        or record["approved_changed_paths"]
        != sorted(
            (
                str(REPO_ROOT / "scripts/train_phase6.py"),
                str(REPO_ROOT / "somaforce_cross/envs/residual_env.py"),
            )
        )
        or snapshots
        != {
            "joint_runner": {
                "path": str(joint_snapshot),
                "sha256": EPISODEORDER_JOINT_SNAPSHOT_SHA256,
            },
            "residual_env": {
                "path": str(env_snapshot),
                "sha256": EPISODEORDER_ENV_SNAPSHOT_SHA256,
            },
            "train_phase6": {
                "path": str(train_snapshot),
                "sha256": EPISODEORDER_TRAIN_SNAPSHOT_SHA256,
            },
        }
        or record["failed_smoke_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["prior_rankseed_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["new_source_manifest"] != _current_pairedbatch_source_manifest()
        or record["episode_order_fix"]
        != _episodeorder_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("episodeorder rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_pairedbatch_source_manifest(),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("episodeorder checkpoint is not pre-evaluation")
    return dict(record)


def _validate_modeprocess_failed_smoke_manifest(
    manifest_path: Path,
) -> dict[str, object]:
    payload = _require_exact_mapping(
        _read_json(manifest_path),
        name="modeprocess.failed_smoke_manifest",
        expected={
            "attempt",
            "attempt_relative",
            "files",
            "outer_log",
            "schema_version",
            "status",
        },
    )
    if (
        payload["schema_version"]
        != "phase6_modeprocess_failed_smoke_frozen_manifest_v1"
        or payload["status"] != "timeout_during_residual_cleanup_before_scaffold"
        or not isinstance(payload["files"], list)
        or len(payload["files"]) != 15
        or not isinstance(payload["outer_log"], Mapping)
    ):
        raise ValueError("modeprocess failed smoke manifest is invalid")
    attempt = Path(str(payload["attempt"]))
    if (
        not attempt.is_dir()
        or str(attempt.resolve()) != str(attempt)
        or str(attempt.relative_to(REPO_ROOT)) != payload["attempt_relative"]
    ):
        raise ValueError("modeprocess failed smoke attempt path is invalid")
    for entry in [*payload["files"], payload["outer_log"]]:
        record = _require_exact_mapping(
            entry,
            name="modeprocess.failed_smoke_entry",
            expected={
                "absolute_path",
                "mtime_ns",
                "relative_path",
                "sha256",
                "size_bytes",
            },
        )
        path = Path(str(record["absolute_path"]))
        if (
            not path.is_file()
            or str(path.resolve()) != str(path)
            or str(path.relative_to(REPO_ROOT)) != record["relative_path"]
            or path.stat().st_size != record["size_bytes"]
            or path.stat().st_mtime_ns != record["mtime_ns"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError("modeprocess failed smoke evidence changed")
    return dict(payload)


def _modeprocess_ast_proof(
    train_snapshot: Path,
    joint_snapshot: Path,
    env_snapshot: Path,
    *,
    candidate_train: Path | None = None,
    candidate_joint: Path | None = None,
    candidate_env: Path | None = None,
) -> dict[str, object]:
    current_train = candidate_train or Path(__file__).resolve()
    current_joint = candidate_joint or (
        REPO_ROOT / "somaforce_cross/learning/joint_runner.py"
    )
    current_env = candidate_env or REPO_ROOT / "somaforce_cross/envs/residual_env.py"
    old_functions = _ast_function_dumps(train_snapshot, WORKER_CRITICAL_FUNCTIONS)
    new_functions = _ast_function_dumps(current_train, WORKER_CRITICAL_FUNCTIONS)
    evaluator = _top_level_function(current_train, "_run_paired_evaluation_rank")
    worker = _top_level_function(current_train, "_worker_main")
    recovery = _top_level_function(current_train, "_run_modeprocess_recovery_evaluate")
    build_calls = [
        node
        for node in ast.walk(evaluator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_environment"
    ]
    close_calls = [
        node
        for node in ast.walk(evaluator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
    ]
    paired_mode_reads = [
        node
        for node in ast.walk(evaluator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "args"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "paired_mode"
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value is None
        and not node.keywords
    ]
    direct_mode_reads = [
        node
        for node in ast.walk(evaluator)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
        and node.attr == "paired_mode"
    ]
    mode_rejection = any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "mode"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Set)
        and {
            item.value
            for item in node.comparators[0].elts
            if isinstance(item, ast.Constant)
        }
        == {"residual", "scaffold_only"}
        for node in ast.walk(evaluator)
    )
    progress_rebindings = [
        node
        for node in ast.walk(evaluator)
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "progress"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "progress"
        )
        or (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "progress"
        )
        or (
            isinstance(node, ast.NamedExpr)
            and isinstance(node.target, ast.Name)
            and node.target.id == "progress"
        )
    ]
    barrier_calls = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
        and node.func.value.attr == "distributed"
        and node.func.attr == "barrier"
    ]
    torchrun_strings = [
        node.value
        for node in ast.walk(recovery)
        if isinstance(node, ast.Constant) and node.value == "torch.distributed.run"
    ]
    proof = {
        "approved_differences": [
            "one paired mode per Isaac worker process",
            "rank-ordered evaluation cleanup barriers",
            "modeprocess rebind, merge, progress, and recovery harness",
        ],
        "evaluator_build_environment_count": len(build_calls),
        "evaluator_close_call_count": len(close_calls),
        "evaluator_reads_required_paired_mode": len(paired_mode_reads) == 1
        and len(direct_mode_reads) == 0,
        "evaluator_rejects_invalid_paired_mode": mode_rejection,
        "evaluator_preserves_progress_parameter": not progress_rebindings,
        "joint_runner_byte_equality": _sha256(current_joint) == _sha256(joint_snapshot),
        "residual_env_byte_equality": _sha256(current_env) == _sha256(env_snapshot),
        "source_snapshots": {
            "joint_runner": _sha256(joint_snapshot),
            "residual_env": _sha256(env_snapshot),
            "train_phase6": _sha256(train_snapshot),
        },
        "two_sequential_torchrun_lifecycles": len(torchrun_strings) == 1,
        "worker_cleanup_has_barriers": len(barrier_calls) >= 2,
        "worker_critical_ast_equality": {
            name: old_functions[name] == new_functions[name]
            for name in WORKER_CRITICAL_FUNCTIONS
            if name not in {"_run_paired_evaluation_rank", "_worker_main"}
        },
    }
    if (
        proof["evaluator_build_environment_count"] != 1
        or proof["evaluator_close_call_count"] != 0
        or not proof["evaluator_reads_required_paired_mode"]
        or not proof["evaluator_rejects_invalid_paired_mode"]
        or not proof["evaluator_preserves_progress_parameter"]
        or not proof["joint_runner_byte_equality"]
        or not proof["residual_env_byte_equality"]
        or not proof["two_sequential_torchrun_lifecycles"]
        or not proof["worker_cleanup_has_barriers"]
        or not all(proof["worker_critical_ast_equality"].values())
    ):
        raise ValueError(
            "modeprocess rebind changed more than approved evaluation ownership"
        )
    return proof


def _modeprocess_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_modeprocessfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_modeprocessfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_modeprocessfix.py"
    input_checkpoint = recovery / "pre_evaluation_episodeorder_rebound.pt"
    input_record = recovery / "source_rebind_episodeorder.json"
    output_checkpoint = recovery / "pre_evaluation_modeprocess_rebound.pt"
    output_record = recovery / "source_rebind_modeprocess.json"
    if (
        args.source_snapshot.resolve() != train_snapshot.resolve()
        or args.pre_checkpoint.resolve() != input_checkpoint.resolve()
        or args.source_rebind.resolve() != input_record.resolve()
        or args.rebound_checkpoint.resolve() != output_checkpoint.resolve()
        or args.source_rebind_output.resolve() != output_record.resolve()
    ):
        raise ValueError("modeprocess rebind paths are fixed for the preserved pilot")
    return (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        recovery / "evaluation_episodeorder_smoke_0005_frozen_manifest.json",
    )


def _validate_historical_episodeorder_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    episode_train = recovery / "source_snapshot/train_phase6_pre_episodeorderfix.py"
    episode_joint = recovery / "source_snapshot/joint_runner_pre_episodeorderfix.py"
    episode_env = recovery / "source_snapshot/residual_env_pre_episodeorderfix.py"
    validated_train = recovery / "source_snapshot/train_phase6_pre_modeprocessfix.py"
    validated_joint = recovery / "source_snapshot/joint_runner_pre_modeprocessfix.py"
    validated_env = recovery / "source_snapshot/residual_env_pre_modeprocessfix.py"
    input_checkpoint = recovery / "pre_evaluation_rankseed_rebound.pt"
    input_record = recovery / "source_rebind_rankseed.json"
    failed_manifest = recovery / "evaluation_rankseed_smoke_0004_frozen_manifest.json"
    record = _require_exact_mapping(
        _read_json(record_path),
        name="modeprocess.historical_episodeorder_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "episode_order_fix",
            "failed_smoke_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_rankseed_rebind",
            "source_snapshots",
            "status",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="modeprocess.historical_episodeorder_checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshots = _require_exact_mapping(
        record["source_snapshots"],
        name="modeprocess.historical_episodeorder_snapshots",
        expected={"joint_runner", "residual_env", "train_phase6"},
    )
    failed = _validate_episodeorder_failed_smoke_manifest(failed_manifest)
    historical_proof = _episodeorder_ast_proof_for_sources(
        episode_train,
        episode_joint,
        episode_env,
        candidate_train=validated_train,
        candidate_joint=validated_joint,
        candidate_env=validated_env,
    )
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": MODEPROCESS_INPUT_SHA256,
            "old_path": str(input_checkpoint),
            "old_sha256": EPISODEORDER_INPUT_SHA256,
        }
        or _sha256(rebound_checkpoint) != MODEPROCESS_INPUT_SHA256
        or snapshots
        != {
            "joint_runner": {
                "path": str(episode_joint),
                "sha256": EPISODEORDER_JOINT_SNAPSHOT_SHA256,
            },
            "residual_env": {
                "path": str(episode_env),
                "sha256": EPISODEORDER_ENV_SNAPSHOT_SHA256,
            },
            "train_phase6": {
                "path": str(episode_train),
                "sha256": EPISODEORDER_TRAIN_SNAPSHOT_SHA256,
            },
        }
        or record["failed_smoke_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["prior_rankseed_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["episode_order_fix"] != historical_proof
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("historical episodeorder rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=dict(record["new_source_manifest"]),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("historical episodeorder checkpoint is not pre-evaluation")
    return dict(record)


def _run_modeprocess_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        failed_manifest,
    ) = _modeprocess_paths(args)
    expected = {
        train_snapshot: MODEPROCESS_TRAIN_SNAPSHOT_SHA256,
        joint_snapshot: MODEPROCESS_JOINT_SNAPSHOT_SHA256,
        env_snapshot: MODEPROCESS_ENV_SNAPSHOT_SHA256,
        input_checkpoint: MODEPROCESS_INPUT_SHA256,
    }
    if any(_sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("modeprocess rebind inputs differ from frozen evidence")
    if output_checkpoint.exists() or output_record.exists():
        raise FileExistsError("modeprocess rebind refuses to overwrite evidence")
    failed = _validate_modeprocess_failed_smoke_manifest(failed_manifest)
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    prior = _validate_historical_episodeorder_rebind_record(
        record_path=input_record,
        rebound_checkpoint=input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("modeprocess input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    if old_payload.get("source_manifest") != prior["new_source_manifest"]:
        raise ValueError("modeprocess input checkpoint source manifest is invalid")
    new_manifest = _current_pairedbatch_source_manifest()
    old_manifest = dict(old_payload["source_manifest"])
    script_key = str(REPO_ROOT / "scripts/train_phase6.py")
    if (
        set(old_manifest) != set(new_manifest)
        or any(
            old_manifest[key] != new_manifest[key]
            for key in old_manifest
            if key != script_key
        )
        or old_manifest[script_key] == new_manifest[script_key]
    ):
        raise ValueError("modeprocess source delta is not script-only")
    proof = _modeprocess_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariants = _evaluation_devicefix_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError("modeprocess rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("modeprocess checkpoint did not strictly restore")
    record = {
        "approved_changed_paths": [script_key],
        "checkpoint": {
            "new_path": str(output_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(input_checkpoint),
            "old_sha256": MODEPROCESS_INPUT_SHA256,
        },
        "failed_smoke_manifest": {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        },
        "invariants": invariants,
        "modeprocess_fix": proof,
        "new_source_manifest": new_manifest,
        "old_source_manifest": old_manifest,
        "prior_episodeorder_rebind": {
            "path": str(input_record),
            "sha256": _sha256(input_record),
        },
        "source_snapshots": {
            "joint_runner": {
                "path": str(joint_snapshot),
                "sha256": _sha256(joint_snapshot),
            },
            "residual_env": {
                "path": str(env_snapshot),
                "sha256": _sha256(env_snapshot),
            },
            "train_phase6": {
                "path": str(train_snapshot),
                "sha256": _sha256(train_snapshot),
            },
        },
        "status": "ok",
    }
    _atomic_json(output_record, record)
    print("PHASE6_MODEPROCESS_REBIND=" + json.dumps(record, sort_keys=True), flush=True)
    return 0


def _validate_modeprocess_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_modeprocessfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_modeprocessfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_modeprocessfix.py"
    input_checkpoint = recovery / "pre_evaluation_episodeorder_rebound.pt"
    input_record = recovery / "source_rebind_episodeorder.json"
    failed_manifest = (
        recovery / "evaluation_episodeorder_smoke_0005_frozen_manifest.json"
    )
    record = _require_exact_mapping(
        _read_json(record_path),
        name="modeprocess_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "failed_smoke_manifest",
            "invariants",
            "modeprocess_fix",
            "new_source_manifest",
            "old_source_manifest",
            "prior_episodeorder_rebind",
            "source_snapshots",
            "status",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="modeprocess_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    failed = _validate_modeprocess_failed_smoke_manifest(failed_manifest)
    if (
        record["status"] != "ok"
        or record["approved_changed_paths"]
        != [str(REPO_ROOT / "scripts/train_phase6.py")]
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": _sha256(rebound_checkpoint),
            "old_path": str(input_checkpoint),
            "old_sha256": MODEPROCESS_INPUT_SHA256,
        }
        or record["failed_smoke_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["prior_episodeorder_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["new_source_manifest"] != _current_pairedbatch_source_manifest()
        or record["modeprocess_fix"]
        != _modeprocess_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("modeprocess rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_pairedbatch_source_manifest(),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("modeprocess checkpoint is not pre-evaluation")
    return dict(record)


def _warningpolicy_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_warningpolicyfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_warningpolicyfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_warningpolicyfix.py"
    input_checkpoint = recovery / "pre_evaluation_modeprocess_rebound.pt"
    input_record = recovery / "source_rebind_modeprocess.json"
    output_checkpoint = recovery / "pre_evaluation_warningpolicy_rebound.pt"
    output_record = recovery / "source_rebind_warningpolicy.json"
    if (
        args.source_snapshot.resolve() != train_snapshot.resolve()
        or args.pre_checkpoint.resolve() != input_checkpoint.resolve()
        or args.source_rebind.resolve() != input_record.resolve()
        or args.rebound_checkpoint.resolve() != output_checkpoint.resolve()
        or args.source_rebind_output.resolve() != output_record.resolve()
    ):
        raise ValueError("warningpolicy rebind paths are fixed for the preserved pilot")
    return (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        recovery / "evaluation_modeprocess_smoke_0006_frozen_manifest.json",
    )


def _validate_warningpolicy_failed_smoke_manifest(
    manifest_path: Path,
) -> dict[str, object]:
    payload = _require_exact_mapping(
        _read_json(manifest_path),
        name="warningpolicy.failed_smoke_manifest",
        expected={
            "attempt",
            "attempt_relative",
            "files",
            "outer_log",
            "schema_version",
            "status",
        },
    )
    if (
        payload["schema_version"]
        != "phase6_warningpolicy_failed_smoke_frozen_manifest_v1"
        or payload["status"] != "business_complete_merge_rejected_warning_policy"
        or not isinstance(payload["files"], list)
        or len(payload["files"]) != 47
        or not isinstance(payload["outer_log"], Mapping)
    ):
        raise ValueError("warningpolicy failed smoke manifest is invalid")
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    attempt = recovery / "evaluation_modeprocess_smoke_0006"
    if Path(str(payload["attempt"])) != attempt or payload["attempt_relative"] != str(
        attempt.relative_to(REPO_ROOT)
    ):
        raise ValueError("warningpolicy failed smoke attempt path is invalid")
    expected_outer = {
        "sha256": "0d27848f1dbdac1abc779a0f099c841725fec9d5d28e13f60c2e98b086ad9c48",
        "size_bytes": 3581,
        "mtime_ns": 1786089813482411044,
    }
    outer = _require_exact_mapping(
        payload["outer_log"],
        name="warningpolicy.failed_smoke_outer_log",
        expected={
            "absolute_path",
            "mtime_ns",
            "relative_path",
            "sha256",
            "size_bytes",
        },
    )
    if any(outer[key] != value for key, value in expected_outer.items()):
        raise ValueError("warningpolicy failed smoke outer log differs from evidence")
    for entry in [*payload["files"], outer]:
        record = _require_exact_mapping(
            entry,
            name="warningpolicy.failed_smoke_entry",
            expected={
                "absolute_path",
                "mtime_ns",
                "relative_path",
                "sha256",
                "size_bytes",
            },
        )
        path = Path(str(record["absolute_path"]))
        if (
            not path.is_file()
            or str(path.resolve()) != str(path)
            or str(path.relative_to(REPO_ROOT)) != record["relative_path"]
            or path.stat().st_size != record["size_bytes"]
            or path.stat().st_mtime_ns != record["mtime_ns"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError("warningpolicy failed smoke evidence changed")
    return dict(payload)


def _warningpolicy_ast_proof(
    train_snapshot: Path,
    joint_snapshot: Path,
    env_snapshot: Path,
    *,
    candidate_train: Path | None = None,
    candidate_joint: Path | None = None,
    candidate_env: Path | None = None,
) -> dict[str, object]:
    current_train = candidate_train or Path(__file__).resolve()
    current_joint = candidate_joint or (
        REPO_ROOT / "somaforce_cross/learning/joint_runner.py"
    )
    current_env = candidate_env or REPO_ROOT / "somaforce_cross/envs/residual_env.py"
    old_tree = ast.parse(train_snapshot.read_text(encoding="utf-8"))
    new_tree = ast.parse(current_train.read_text(encoding="utf-8"))
    old_functions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in old_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    new_functions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in new_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    allowed_changed = {"_validate_modeprocess_wrapper", "main"}
    allowed_added = {
        "_warningpolicy_rebind_parser",
        "_warningpolicy_recovery_evaluate_parser",
        "_warningpolicy_paths",
        "_validate_warningpolicy_failed_smoke_manifest",
        "_warningpolicy_ast_proof",
        "_validate_historical_modeprocess_rebind_record",
        "_run_warningpolicy_rebind",
        "_validate_warningpolicy_rebind_record",
        "_run_warningpolicy_recovery_evaluate",
    }
    changed = {
        name
        for name in old_functions.keys() & new_functions.keys()
        if old_functions[name] != new_functions[name]
    }
    added = set(new_functions) - set(old_functions)
    removed = set(old_functions) - set(new_functions)
    wrapper = _top_level_function(current_train, "_validate_modeprocess_wrapper")
    warning_calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_worker_warning_counts"
    ]
    vulkan_fatal = any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Subscript)
        and isinstance(node.left.value, ast.Name)
        and node.left.value.id == "warnings"
        and isinstance(node.left.slice, ast.Constant)
        and node.left.slice.value == "multiple_installable_client_drivers"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Gt)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == 0
        for node in ast.walk(wrapper)
    )
    preserved = (
        "_run_paired_evaluation_rank",
        "_worker_main",
        "_merge_modeprocess_evaluation",
        "_aggregate_modeprocess_progress",
        "_worker_warning_counts",
    )
    proof = {
        "approved_differences": [
            "modeprocess wrapper warning policy",
            "warningpolicy rebind, validator, and recovery harness",
        ],
        "changed_existing_functions": sorted(changed),
        "added_functions": sorted(added),
        "joint_runner_byte_equality": _sha256(current_joint) == _sha256(joint_snapshot),
        "residual_env_byte_equality": _sha256(current_env) == _sha256(env_snapshot),
        "modeprocess_wrapper_recounts_worker_log": len(warning_calls) == 1,
        "only_multiple_vulkan_icd_is_fatal": vulkan_fatal,
        "preserved_function_ast": {
            name: old_functions[name] == new_functions[name] for name in preserved
        },
        "worker_critical_ast_equality": {
            name: _ast_function_dumps(train_snapshot, WORKER_CRITICAL_FUNCTIONS)[name]
            == _ast_function_dumps(current_train, WORKER_CRITICAL_FUNCTIONS)[name]
            for name in WORKER_CRITICAL_FUNCTIONS
        },
    }
    if (
        changed - allowed_changed
        or added != allowed_added
        or removed
        or not proof["joint_runner_byte_equality"]
        or not proof["residual_env_byte_equality"]
        or not proof["modeprocess_wrapper_recounts_worker_log"]
        or not proof["only_multiple_vulkan_icd_is_fatal"]
        or not all(proof["preserved_function_ast"].values())
        or not all(proof["worker_critical_ast_equality"].values())
    ):
        raise ValueError("warningpolicy rebind changed more than warning policy")
    return proof


def _validate_historical_modeprocess_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    modeprocess_train = recovery / "source_snapshot/train_phase6_pre_modeprocessfix.py"
    modeprocess_joint = recovery / "source_snapshot/joint_runner_pre_modeprocessfix.py"
    modeprocess_env = recovery / "source_snapshot/residual_env_pre_modeprocessfix.py"
    validated_train = recovery / "source_snapshot/train_phase6_pre_warningpolicyfix.py"
    validated_joint = recovery / "source_snapshot/joint_runner_pre_warningpolicyfix.py"
    validated_env = recovery / "source_snapshot/residual_env_pre_warningpolicyfix.py"
    input_checkpoint = recovery / "pre_evaluation_episodeorder_rebound.pt"
    input_record = recovery / "source_rebind_episodeorder.json"
    failed_manifest = (
        recovery / "evaluation_episodeorder_smoke_0005_frozen_manifest.json"
    )
    record = _require_exact_mapping(
        _read_json(record_path),
        name="warningpolicy.historical_modeprocess_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "failed_smoke_manifest",
            "invariants",
            "modeprocess_fix",
            "new_source_manifest",
            "old_source_manifest",
            "prior_episodeorder_rebind",
            "source_snapshots",
            "status",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="warningpolicy.historical_modeprocess_checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshots = _require_exact_mapping(
        record["source_snapshots"],
        name="warningpolicy.historical_modeprocess_snapshots",
        expected={"joint_runner", "residual_env", "train_phase6"},
    )
    if (
        _sha256(validated_train) != WARNINGPOLICY_TRAIN_SNAPSHOT_SHA256
        or _sha256(validated_joint) != WARNINGPOLICY_JOINT_SNAPSHOT_SHA256
        or _sha256(validated_env) != WARNINGPOLICY_ENV_SNAPSHOT_SHA256
        or _sha256(REPO_ROOT / "somaforce_cross/learning/joint_runner.py")
        != _sha256(validated_joint)
        or _sha256(REPO_ROOT / "somaforce_cross/envs/residual_env.py")
        != _sha256(validated_env)
    ):
        raise ValueError("historical modeprocess endpoint snapshots differ")
    failed = _validate_modeprocess_failed_smoke_manifest(failed_manifest)
    historical_proof = _modeprocess_ast_proof(
        modeprocess_train,
        modeprocess_joint,
        modeprocess_env,
        candidate_train=validated_train,
    )
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("historical modeprocess input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    script_key = str(REPO_ROOT / "scripts/train_phase6.py")
    joint_key = str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py")
    env_key = str(REPO_ROOT / "somaforce_cross/envs/residual_env.py")
    if (
        record["status"] != "ok"
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": WARNINGPOLICY_INPUT_SHA256,
            "old_path": str(input_checkpoint),
            "old_sha256": MODEPROCESS_INPUT_SHA256,
        }
        or _sha256(rebound_checkpoint) != WARNINGPOLICY_INPUT_SHA256
        or snapshots
        != {
            "joint_runner": {
                "path": str(modeprocess_joint),
                "sha256": MODEPROCESS_JOINT_SNAPSHOT_SHA256,
            },
            "residual_env": {
                "path": str(modeprocess_env),
                "sha256": MODEPROCESS_ENV_SNAPSHOT_SHA256,
            },
            "train_phase6": {
                "path": str(modeprocess_train),
                "sha256": MODEPROCESS_TRAIN_SNAPSHOT_SHA256,
            },
        }
        or record["failed_smoke_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["prior_episodeorder_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["old_source_manifest"] != old_payload["source_manifest"]
        or record["new_source_manifest"].get(script_key)
        != WARNINGPOLICY_TRAIN_SNAPSHOT_SHA256
        or record["new_source_manifest"].get(joint_key)
        != WARNINGPOLICY_JOINT_SNAPSHOT_SHA256
        or record["new_source_manifest"].get(env_key)
        != WARNINGPOLICY_ENV_SNAPSHOT_SHA256
        or record["modeprocess_fix"] != historical_proof
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("historical modeprocess rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=dict(record["new_source_manifest"]),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("historical modeprocess checkpoint is not pre-evaluation")
    return dict(record)


def _run_warningpolicy_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        failed_manifest,
    ) = _warningpolicy_paths(args)
    expected = {
        train_snapshot: WARNINGPOLICY_TRAIN_SNAPSHOT_SHA256,
        joint_snapshot: WARNINGPOLICY_JOINT_SNAPSHOT_SHA256,
        env_snapshot: WARNINGPOLICY_ENV_SNAPSHOT_SHA256,
        input_checkpoint: WARNINGPOLICY_INPUT_SHA256,
    }
    if any(_sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("warningpolicy rebind inputs differ from frozen evidence")
    if output_checkpoint.exists() or output_record.exists():
        raise FileExistsError("warningpolicy rebind refuses to overwrite evidence")
    failed = _validate_warningpolicy_failed_smoke_manifest(failed_manifest)
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    prior = _validate_historical_modeprocess_rebind_record(
        record_path=input_record,
        rebound_checkpoint=input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("warningpolicy input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    if old_payload.get("source_manifest") != prior["new_source_manifest"]:
        raise ValueError("warningpolicy input checkpoint source manifest is invalid")
    new_manifest = _current_pairedbatch_source_manifest()
    old_manifest = dict(old_payload["source_manifest"])
    script_key = str(REPO_ROOT / "scripts/train_phase6.py")
    if (
        set(old_manifest) != set(new_manifest)
        or any(
            old_manifest[key] != new_manifest[key]
            for key in old_manifest
            if key != script_key
        )
        or old_manifest[script_key] == new_manifest[script_key]
    ):
        raise ValueError("warningpolicy source delta is not script-only")
    proof = _warningpolicy_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariants = _evaluation_devicefix_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError("warningpolicy rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("warningpolicy checkpoint did not strictly restore")
    record = {
        "approved_changed_paths": [script_key],
        "checkpoint": {
            "new_path": str(output_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(input_checkpoint),
            "old_sha256": WARNINGPOLICY_INPUT_SHA256,
        },
        "failed_smoke_manifest": {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        },
        "invariants": invariants,
        "new_source_manifest": new_manifest,
        "old_source_manifest": old_manifest,
        "prior_modeprocess_rebind": {
            "path": str(input_record),
            "sha256": _sha256(input_record),
        },
        "source_snapshots": {
            "joint_runner": {
                "path": str(joint_snapshot),
                "sha256": _sha256(joint_snapshot),
            },
            "residual_env": {
                "path": str(env_snapshot),
                "sha256": _sha256(env_snapshot),
            },
            "train_phase6": {
                "path": str(train_snapshot),
                "sha256": _sha256(train_snapshot),
            },
        },
        "status": "ok",
        "warningpolicy_fix": proof,
    }
    _atomic_json(output_record, record)
    print(
        "PHASE6_WARNINGPOLICY_REBIND=" + json.dumps(record, sort_keys=True), flush=True
    )
    return 0


def _validate_warningpolicy_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_warningpolicyfix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_warningpolicyfix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_warningpolicyfix.py"
    input_checkpoint = recovery / "pre_evaluation_modeprocess_rebound.pt"
    input_record = recovery / "source_rebind_modeprocess.json"
    failed_manifest = (
        recovery / "evaluation_modeprocess_smoke_0006_frozen_manifest.json"
    )
    record = _require_exact_mapping(
        _read_json(record_path),
        name="warningpolicy_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "failed_smoke_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_modeprocess_rebind",
            "source_snapshots",
            "status",
            "warningpolicy_fix",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="warningpolicy_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    snapshots = _require_exact_mapping(
        record["source_snapshots"],
        name="warningpolicy_rebind.source_snapshots",
        expected={"joint_runner", "residual_env", "train_phase6"},
    )
    failed = _validate_warningpolicy_failed_smoke_manifest(failed_manifest)
    if (
        record["status"] != "ok"
        or record["approved_changed_paths"]
        != [str(REPO_ROOT / "scripts/train_phase6.py")]
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": _sha256(rebound_checkpoint),
            "old_path": str(input_checkpoint),
            "old_sha256": WARNINGPOLICY_INPUT_SHA256,
        }
        or snapshots
        != {
            "joint_runner": {
                "path": str(joint_snapshot),
                "sha256": WARNINGPOLICY_JOINT_SNAPSHOT_SHA256,
            },
            "residual_env": {
                "path": str(env_snapshot),
                "sha256": WARNINGPOLICY_ENV_SNAPSHOT_SHA256,
            },
            "train_phase6": {
                "path": str(train_snapshot),
                "sha256": WARNINGPOLICY_TRAIN_SNAPSHOT_SHA256,
            },
        }
        or record["failed_smoke_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["prior_modeprocess_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["new_source_manifest"] != _current_pairedbatch_source_manifest()
        or record["warningpolicy_fix"]
        != _warningpolicy_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("warningpolicy rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_pairedbatch_source_manifest(),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("warningpolicy checkpoint is not pre-evaluation")
    return dict(record)


def _pairedbatch_source_manifest_for_sources(
    train_source: Path, joint_source: Path, env_source: Path
) -> dict[str, str]:
    from somaforce_cross.learning.acceptance import source_sha256

    manifest = source_sha256(
        (
            REPO_ROOT / "configs/phase6_learning_acceptance_v1.json",
            REPO_ROOT / "configs/phase6_joint_training_v1.json",
            REPO_ROOT / "configs/phase6_task_roster_v1.json",
            REPO_ROOT / "somaforce_cross/learning/acceptance.py",
        )
    )
    manifest[str(REPO_ROOT / "scripts/train_phase6.py")] = _sha256(train_source)
    manifest[str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py")] = _sha256(
        joint_source
    )
    manifest[str(REPO_ROOT / "somaforce_cross/envs/residual_env.py")] = _sha256(
        env_source
    )
    return manifest


def _summaryschema_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_summaryschemafix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_summaryschemafix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_summaryschemafix.py"
    input_checkpoint = recovery / "pre_evaluation_warningpolicy_rebound.pt"
    input_record = recovery / "source_rebind_warningpolicy.json"
    output_checkpoint = recovery / "pre_evaluation_summaryschema_rebound.pt"
    output_record = recovery / "source_rebind_summaryschema.json"
    if (
        args.source_snapshot.resolve() != train_snapshot.resolve()
        or args.pre_checkpoint.resolve() != input_checkpoint.resolve()
        or args.source_rebind.resolve() != input_record.resolve()
        or args.rebound_checkpoint.resolve() != output_checkpoint.resolve()
        or args.source_rebind_output.resolve() != output_record.resolve()
    ):
        raise ValueError("summaryschema rebind paths are fixed for the preserved pilot")
    return (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        recovery / "evaluation_warningpolicy_attempt_0007_frozen_manifest.json",
    )


def _validate_summaryschema_failed_finalize_manifest(
    manifest_path: Path,
) -> dict[str, object]:
    payload = _require_exact_mapping(
        _read_json(manifest_path),
        name="summaryschema.failed_finalize_manifest",
        expected={
            "attempt",
            "attempt_relative",
            "files",
            "outer_log",
            "schema_version",
            "status",
        },
    )
    if (
        _sha256(manifest_path) != SUMMARYSCHEMA_FAILED_FINALIZE_MANIFEST_SHA256
        or payload["schema_version"]
        != "phase6_summaryschema_failed_finalize_frozen_manifest_v1"
        or payload["status"]
        != "paired_evaluation_complete_finalize_rejected_summary_schema"
        or not isinstance(payload["files"], list)
        or len(payload["files"]) != 50
        or not isinstance(payload["outer_log"], Mapping)
    ):
        raise ValueError("summaryschema failed finalize manifest is invalid")
    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    attempt = recovery / "evaluation_warningpolicy_attempt_0007"
    if Path(str(payload["attempt"])) != attempt or payload["attempt_relative"] != str(
        attempt.relative_to(REPO_ROOT)
    ):
        raise ValueError("summaryschema failed finalize attempt path is invalid")
    expected_outer = {
        "sha256": "aaa2c15e23f917b1f19eefc43d14d67e54722dec1d541ebd306d751e8c4b8870",
        "size_bytes": 3542,
        "mtime_ns": 1786097733659214104,
    }
    outer = _require_exact_mapping(
        payload["outer_log"],
        name="summaryschema.failed_finalize_outer_log",
        expected={
            "absolute_path",
            "mtime_ns",
            "relative_path",
            "sha256",
            "size_bytes",
        },
    )
    if any(outer[key] != value for key, value in expected_outer.items()):
        raise ValueError("summaryschema failed finalize outer log differs")
    for entry in [*payload["files"], outer]:
        record = _require_exact_mapping(
            entry,
            name="summaryschema.failed_finalize_entry",
            expected={
                "absolute_path",
                "mtime_ns",
                "relative_path",
                "sha256",
                "size_bytes",
            },
        )
        path = Path(str(record["absolute_path"]))
        if (
            not path.is_file()
            or str(path.resolve()) != str(path)
            or str(path.relative_to(REPO_ROOT)) != record["relative_path"]
            or path.stat().st_size != record["size_bytes"]
            or path.stat().st_mtime_ns != record["mtime_ns"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError("summaryschema failed finalize evidence changed")
    return dict(payload)


def _summaryschema_ast_proof(
    train_snapshot: Path,
    joint_snapshot: Path,
    env_snapshot: Path,
    *,
    candidate_train: Path | None = None,
    candidate_joint: Path | None = None,
    candidate_env: Path | None = None,
) -> dict[str, object]:
    current_train = candidate_train or Path(__file__).resolve()
    current_joint = candidate_joint or (
        REPO_ROOT / "somaforce_cross/learning/joint_runner.py"
    )
    current_env = candidate_env or REPO_ROOT / "somaforce_cross/envs/residual_env.py"
    old_train_tree = ast.parse(train_snapshot.read_text(encoding="utf-8"))
    new_train_tree = ast.parse(current_train.read_text(encoding="utf-8"))
    old_train_functions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in old_train_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    new_train_functions = {
        node.name: ast.dump(node, include_attributes=False)
        for node in new_train_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    old_joint_tree = ast.parse(joint_snapshot.read_text(encoding="utf-8"))
    new_joint_tree = ast.parse(current_joint.read_text(encoding="utf-8"))
    old_joint_functions = {
        node.name: node
        for node in old_joint_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    new_joint_functions = {
        node.name: node
        for node in new_joint_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    changed_train = {
        name
        for name in old_train_functions.keys() & new_train_functions.keys()
        if old_train_functions[name] != new_train_functions[name]
    }
    added_train = set(new_train_functions) - set(old_train_functions)
    removed_train = set(old_train_functions) - set(new_train_functions)
    changed_joint = {
        name
        for name in old_joint_functions.keys() & new_joint_functions.keys()
        if ast.dump(old_joint_functions[name], include_attributes=False)
        != ast.dump(new_joint_functions[name], include_attributes=False)
    }
    old_summary = old_joint_functions["validate_learning_evaluation_summary"]
    new_summary = new_joint_functions["validate_learning_evaluation_summary"]
    old_returns = sorted(
        (
            node
            for node in ast.walk(old_summary)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        ),
        key=lambda node: node.lineno,
    )
    new_returns = sorted(
        (
            node
            for node in ast.walk(new_summary)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        ),
        key=lambda node: node.lineno,
    )
    normalized_summary = copy.deepcopy(new_summary)
    normalized_returns = sorted(
        (
            node
            for node in ast.walk(normalized_summary)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        ),
        key=lambda node: node.lineno,
    )
    schedule_indexes = [
        index
        for index, key in enumerate(normalized_returns[-1].value.keys)
        if isinstance(key, ast.Constant) and key.value == "schedule_sha256"
    ]
    if len(schedule_indexes) == 1:
        index = schedule_indexes[0]
        del normalized_returns[-1].value.keys[index]
        del normalized_returns[-1].value.values[index]
    modern_keys = {
        key.value
        for key in new_returns[-1].value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    legacy_keys = {
        key.value
        for key in new_returns[0].value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    old_modeprocess = _top_level_function(train_snapshot, "_modeprocess_ast_proof")
    new_modeprocess = _top_level_function(current_train, "_modeprocess_ast_proof")
    normalized_modeprocess = copy.deepcopy(new_modeprocess)
    old_assignments = {
        target.id: node
        for node in old_modeprocess.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id in {"current_train", "current_joint", "current_env"}
    }
    new_assignments = {
        target.id: node
        for node in new_modeprocess.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id in {"current_train", "current_joint", "current_env"}
    }
    normalized_assignments = {
        target.id: index
        for index, node in enumerate(normalized_modeprocess.body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and target.id in {"current_train", "current_joint", "current_env"}
    }
    for name in ("current_joint", "current_env"):
        if name in normalized_assignments and name in old_assignments:
            normalized_modeprocess.body[normalized_assignments[name]] = copy.deepcopy(
                old_assignments[name]
            )
    normalized_modeprocess.args.kwonlyargs = normalized_modeprocess.args.kwonlyargs[:1]
    normalized_modeprocess.args.kw_defaults = normalized_modeprocess.args.kw_defaults[
        :1
    ]
    endpoint_routing_only = (
        [argument.arg for argument in new_modeprocess.args.kwonlyargs]
        == ["candidate_train", "candidate_joint", "candidate_env"]
        and all(
            isinstance(default, ast.Constant) and default.value is None
            for default in new_modeprocess.args.kw_defaults
        )
        and set(old_assignments) == set(new_assignments) == set(normalized_assignments)
        and ast.dump(new_assignments["current_train"], include_attributes=False)
        == ast.dump(old_assignments["current_train"], include_attributes=False)
        and all(
            isinstance(new_assignments[name].value, ast.BoolOp)
            and isinstance(new_assignments[name].value.op, ast.Or)
            and len(new_assignments[name].value.values) == 2
            and isinstance(new_assignments[name].value.values[0], ast.Name)
            and new_assignments[name].value.values[0].id
            == {"current_joint": "candidate_joint", "current_env": "candidate_env"}[
                name
            ]
            and ast.dump(
                new_assignments[name].value.values[1], include_attributes=False
            )
            == ast.dump(old_assignments[name].value, include_attributes=False)
            for name in ("current_joint", "current_env")
        )
        and ast.dump(normalized_modeprocess, include_attributes=False)
        == ast.dump(old_modeprocess, include_attributes=False)
    )
    allowed_added = {
        "_pairedbatch_source_manifest_for_sources",
        "_summaryschema_ast_proof",
        "_summaryschema_finalize_parser",
        "_summaryschema_paths",
        "_summaryschema_rebind_parser",
        "_run_summaryschema_finalize",
        "_run_summaryschema_rebind",
        "_validate_historical_warningpolicy_rebind_record",
        "_validate_summaryschema_failed_finalize_manifest",
        "_validate_summaryschema_rebind_record",
    }
    preserved = (
        "_aggregate_modeprocess_progress",
        "_merge_modeprocess_evaluation",
        "_run_paired_evaluation_rank",
        "_run_warningpolicy_recovery_evaluate",
    )
    proof = {
        "approved_differences": [
            "modern summary retains validated aggregate schedule hash",
            "summaryschema rebind and finalize-only recovery harness",
        ],
        "added_train_functions": sorted(added_train),
        "changed_joint_functions": sorted(changed_joint),
        "changed_train_functions": sorted(changed_train),
        "modeprocess_proof_endpoint_routing_only": endpoint_routing_only,
        "joint_runner_only_adds_modern_schedule_sha256": (
            len(old_returns) == len(new_returns) == len(normalized_returns) == 2
            and len(schedule_indexes) == 1
            and legacy_keys == {"rank_results", "task_metrics", "iteration", "status"}
            and modern_keys
            == {
                "rank_results",
                "schedule_sha256",
                "task_metrics",
                "iteration",
                "status",
            }
            and ast.dump(normalized_summary, include_attributes=False)
            == ast.dump(old_summary, include_attributes=False)
        ),
        "joint_runner_byte_change_is_required": _sha256(current_joint)
        != _sha256(joint_snapshot),
        "residual_env_byte_equality": _sha256(current_env) == _sha256(env_snapshot),
        "preserved_function_ast": {
            name: old_train_functions[name] == new_train_functions[name]
            for name in preserved
        },
        "worker_critical_ast_equality": {
            name: _ast_function_dumps(train_snapshot, WORKER_CRITICAL_FUNCTIONS)[name]
            == _ast_function_dumps(current_train, WORKER_CRITICAL_FUNCTIONS)[name]
            for name in WORKER_CRITICAL_FUNCTIONS
        },
    }
    if (
        changed_train != {"_modeprocess_ast_proof", "_warningpolicy_ast_proof", "main"}
        or added_train != allowed_added
        or removed_train
        or changed_joint != {"validate_learning_evaluation_summary"}
        or set(old_joint_functions) != set(new_joint_functions)
        or not proof["joint_runner_only_adds_modern_schedule_sha256"]
        or not proof["modeprocess_proof_endpoint_routing_only"]
        or not proof["joint_runner_byte_change_is_required"]
        or not proof["residual_env_byte_equality"]
        or not all(proof["preserved_function_ast"].values())
        or not all(proof["worker_critical_ast_equality"].values())
    ):
        raise ValueError("summaryschema rebind changed more than summary schema")
    return proof


def _validate_historical_warningpolicy_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    old_train = recovery / "source_snapshot/train_phase6_pre_warningpolicyfix.py"
    old_joint = recovery / "source_snapshot/joint_runner_pre_warningpolicyfix.py"
    old_env = recovery / "source_snapshot/residual_env_pre_warningpolicyfix.py"
    validated_train = recovery / "source_snapshot/train_phase6_pre_summaryschemafix.py"
    validated_joint = recovery / "source_snapshot/joint_runner_pre_summaryschemafix.py"
    validated_env = recovery / "source_snapshot/residual_env_pre_summaryschemafix.py"
    input_checkpoint = recovery / "pre_evaluation_modeprocess_rebound.pt"
    input_record = recovery / "source_rebind_modeprocess.json"
    failed_manifest = (
        recovery / "evaluation_modeprocess_smoke_0006_frozen_manifest.json"
    )
    expected = {
        validated_train: SUMMARYSCHEMA_TRAIN_SNAPSHOT_SHA256,
        validated_joint: SUMMARYSCHEMA_JOINT_SNAPSHOT_SHA256,
        validated_env: SUMMARYSCHEMA_ENV_SNAPSHOT_SHA256,
        rebound_checkpoint: SUMMARYSCHEMA_INPUT_SHA256,
    }
    if any(_sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("historical warningpolicy endpoint differs from snapshots")
    record = _require_exact_mapping(
        _read_json(record_path),
        name="summaryschema.historical_warningpolicy_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "failed_smoke_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_modeprocess_rebind",
            "source_snapshots",
            "status",
            "warningpolicy_fix",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="summaryschema.historical_warningpolicy_checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    failed = _validate_warningpolicy_failed_smoke_manifest(failed_manifest)
    historical_manifest = _pairedbatch_source_manifest_for_sources(
        validated_train, validated_joint, validated_env
    )
    historical_proof = _warningpolicy_ast_proof(
        old_train,
        old_joint,
        old_env,
        candidate_train=validated_train,
        candidate_joint=validated_joint,
        candidate_env=validated_env,
    )
    input_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(input_payload_value, Mapping):
        raise ValueError("historical warningpolicy input checkpoint is invalid")
    input_payload = dict(input_payload_value)
    rebound_payload_value = torch.load(
        rebound_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(rebound_payload_value, Mapping):
        raise ValueError("historical warningpolicy rebound checkpoint is invalid")
    rebound_payload = dict(rebound_payload_value)
    if (
        _sha256(record_path) != SUMMARYSCHEMA_INPUT_RECORD_SHA256
        or _sha256(input_checkpoint) != WARNINGPOLICY_INPUT_SHA256
        or record["status"] != "ok"
        or record["approved_changed_paths"]
        != [str(REPO_ROOT / "scripts/train_phase6.py")]
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": SUMMARYSCHEMA_INPUT_SHA256,
            "old_path": str(input_checkpoint),
            "old_sha256": WARNINGPOLICY_INPUT_SHA256,
        }
        or record["old_source_manifest"] != input_payload.get("source_manifest")
        or record["new_source_manifest"] != rebound_payload.get("source_manifest")
        or rebound_payload.get("source_manifest") != historical_manifest
        or record["prior_modeprocess_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["failed_smoke_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["warningpolicy_fix"] != historical_proof
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("historical warningpolicy rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=historical_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("historical warningpolicy checkpoint is not pre-evaluation")
    return dict(record)


def _run_summaryschema_rebind(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    (
        train_snapshot,
        joint_snapshot,
        env_snapshot,
        input_checkpoint,
        input_record,
        output_checkpoint,
        output_record,
        failed_manifest,
    ) = _summaryschema_paths(args)
    expected = {
        train_snapshot: SUMMARYSCHEMA_TRAIN_SNAPSHOT_SHA256,
        joint_snapshot: SUMMARYSCHEMA_JOINT_SNAPSHOT_SHA256,
        env_snapshot: SUMMARYSCHEMA_ENV_SNAPSHOT_SHA256,
        input_checkpoint: SUMMARYSCHEMA_INPUT_SHA256,
        input_record: SUMMARYSCHEMA_INPUT_RECORD_SHA256,
    }
    if any(_sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("summaryschema rebind inputs differ from frozen evidence")
    if output_checkpoint.exists() or output_record.exists():
        raise FileExistsError("summaryschema rebind refuses to overwrite evidence")
    failed = _validate_summaryschema_failed_finalize_manifest(failed_manifest)
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    prior = _validate_historical_warningpolicy_rebind_record(
        record_path=input_record,
        rebound_checkpoint=input_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    old_payload_value = torch.load(
        input_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(old_payload_value, Mapping):
        raise ValueError("summaryschema input checkpoint is invalid")
    old_payload = dict(old_payload_value)
    if old_payload.get("source_manifest") != prior["new_source_manifest"]:
        raise ValueError("summaryschema input checkpoint source manifest is invalid")
    old_manifest = dict(old_payload["source_manifest"])
    new_manifest = _current_pairedbatch_source_manifest()
    script_key = str(REPO_ROOT / "scripts/train_phase6.py")
    joint_key = str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py")
    approved = [script_key, joint_key]
    changed = sorted(
        key
        for key in set(old_manifest) | set(new_manifest)
        if old_manifest.get(key) != new_manifest.get(key)
    )
    if (
        set(old_manifest) != set(new_manifest)
        or changed != sorted(approved)
        or any(old_manifest[key] == new_manifest[key] for key in approved)
    ):
        raise ValueError("summaryschema source delta is not train-and-joint only")
    proof = _summaryschema_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
    new_payload = dict(old_payload)
    new_payload["source_manifest"] = new_manifest
    invariants = _evaluation_devicefix_invariants(old_payload, new_payload)
    if not all(invariants.values()):
        raise AssertionError("summaryschema rebind altered protected checkpoint state")
    new_sha256 = atomic_learning_checkpoint(output_checkpoint, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        output_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    if not _semantic_equal(restored, new_payload):
        raise AssertionError("summaryschema checkpoint did not strictly restore")
    record = {
        "approved_changed_paths": approved,
        "checkpoint": {
            "new_path": str(output_checkpoint),
            "new_sha256": new_sha256,
            "old_path": str(input_checkpoint),
            "old_sha256": SUMMARYSCHEMA_INPUT_SHA256,
        },
        "failed_finalize_manifest": {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        },
        "invariants": invariants,
        "new_source_manifest": new_manifest,
        "old_source_manifest": old_manifest,
        "prior_warningpolicy_rebind": {
            "path": str(input_record),
            "sha256": _sha256(input_record),
        },
        "source_snapshots": {
            "joint_runner": {
                "path": str(joint_snapshot),
                "sha256": _sha256(joint_snapshot),
            },
            "residual_env": {
                "path": str(env_snapshot),
                "sha256": _sha256(env_snapshot),
            },
            "train_phase6": {
                "path": str(train_snapshot),
                "sha256": _sha256(train_snapshot),
            },
        },
        "status": "ok",
        "summaryschema_fix": proof,
    }
    _atomic_json(output_record, record)
    print(
        "PHASE6_SUMMARYSCHEMA_REBIND=" + json.dumps(record, sort_keys=True),
        flush=True,
    )
    return 0


def _validate_summaryschema_rebind_record(
    *,
    record_path: Path,
    rebound_checkpoint: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    train_snapshot = recovery / "source_snapshot/train_phase6_pre_summaryschemafix.py"
    joint_snapshot = recovery / "source_snapshot/joint_runner_pre_summaryschemafix.py"
    env_snapshot = recovery / "source_snapshot/residual_env_pre_summaryschemafix.py"
    input_checkpoint = recovery / "pre_evaluation_warningpolicy_rebound.pt"
    input_record = recovery / "source_rebind_warningpolicy.json"
    failed_manifest = (
        recovery / "evaluation_warningpolicy_attempt_0007_frozen_manifest.json"
    )
    record = _require_exact_mapping(
        _read_json(record_path),
        name="summaryschema_rebind",
        expected={
            "approved_changed_paths",
            "checkpoint",
            "failed_finalize_manifest",
            "invariants",
            "new_source_manifest",
            "old_source_manifest",
            "prior_warningpolicy_rebind",
            "source_snapshots",
            "status",
            "summaryschema_fix",
        },
    )
    checkpoint = _require_exact_mapping(
        record["checkpoint"],
        name="summaryschema_rebind.checkpoint",
        expected={"new_path", "new_sha256", "old_path", "old_sha256"},
    )
    failed = _validate_summaryschema_failed_finalize_manifest(failed_manifest)
    approved = [
        str(REPO_ROOT / "scripts/train_phase6.py"),
        str(REPO_ROOT / "somaforce_cross/learning/joint_runner.py"),
    ]
    if (
        record["status"] != "ok"
        or record["approved_changed_paths"] != approved
        or checkpoint
        != {
            "new_path": str(rebound_checkpoint),
            "new_sha256": _sha256(rebound_checkpoint),
            "old_path": str(input_checkpoint),
            "old_sha256": SUMMARYSCHEMA_INPUT_SHA256,
        }
        or record["failed_finalize_manifest"]
        != {
            "path": str(failed_manifest),
            "sha256": _sha256(failed_manifest),
            "status": failed["status"],
        }
        or record["prior_warningpolicy_rebind"]
        != {"path": str(input_record), "sha256": _sha256(input_record)}
        or record["new_source_manifest"] != _current_pairedbatch_source_manifest()
        or record["summaryschema_fix"]
        != _summaryschema_ast_proof(train_snapshot, joint_snapshot, env_snapshot)
        or not isinstance(record["invariants"], Mapping)
        or not all(record["invariants"].values())
    ):
        raise ValueError("summaryschema rebind record is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_pairedbatch_source_manifest(),
        expected_iteration=31,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("summaryschema checkpoint is not pre-evaluation")
    return dict(record)


def _run_summaryschema_finalize(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
        validate_learning_evaluation_summary,
    )

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    checkpoint = recovery / "pre_evaluation_summaryschema_rebound.pt"
    record = recovery / "source_rebind_summaryschema.json"
    original_attempt = recovery / "evaluation_warningpolicy_attempt_0007"
    finalized_attempt = recovery / "evaluation_summaryschema_finalize_0008"
    post_checkpoint = recovery / "post_evaluation.pt"
    latest = recovery / "latest.json"
    if (
        args.output_dir.resolve() != recovery.resolve()
        or args.rebound_checkpoint.resolve() != checkpoint.resolve()
        or args.source_rebind.resolve() != record.resolve()
    ):
        raise ValueError(
            "summaryschema finalize paths are fixed for the preserved pilot"
        )
    if finalized_attempt.exists() or post_checkpoint.exists() or latest.exists():
        raise FileExistsError("summaryschema finalize refuses to overwrite evidence")
    _validate_summaryschema_failed_finalize_manifest(
        recovery / "evaluation_warningpolicy_attempt_0007_frozen_manifest.json"
    )
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_summaryschema_rebind_record(
        record_path=record,
        rebound_checkpoint=checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    finalized_attempt.mkdir(parents=True)
    command = {
        "command": [
            ISAAC_PYTHON,
            "-m",
            "scripts.train_phase6",
            "--mode",
            "summaryschema-finalize",
            "--output-dir",
            str(recovery),
            "--rebound-checkpoint",
            str(checkpoint),
            "--source-rebind",
            str(record),
        ],
        "stage": "summaryschema-finalize-only",
        "timeout_s": 900,
    }
    _atomic_json(finalized_attempt / "finalize_summaryschema.command.json", command)
    try:
        shutil.copytree(original_attempt, finalized_attempt, dirs_exist_ok=True)
        summary = _merge_modeprocess_evaluation(
            attempt=finalized_attempt,
            roster=roster,
            stage_quota=256,
            nominal_quota=128,
        )
        if set(summary) != {
            "iteration",
            "rank_results",
            "schedule_sha256",
            "status",
            "task_metrics",
        }:
            raise ValueError("summaryschema recovery summary has an invalid schema")
        round_tripped = validate_learning_evaluation_summary(
            json.loads(json.dumps(summary)), roster=roster, expected_iteration=31
        )
        if round_tripped["schedule_sha256"] != summary["schedule_sha256"]:
            raise ValueError("summaryschema recovery summary lost its schedule hash")
        policy = ResidualActorCritic()
        optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
        restored = restore_learning_checkpoint(
            checkpoint,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=policy,
            optimizer=optimizer,
            source_manifest=_current_pairedbatch_source_manifest(),
            expected_iteration=31,
            device="cpu",
        )
        if restored["pre_evaluation"] is not True:
            raise ValueError(
                "summaryschema finalize requires a pre-evaluation checkpoint"
            )
        payload = dict(restored)
        payload["pre_evaluation"] = False
        payload["evaluation_history"] = [round_tripped]
        protected = (
            "actual_global_transitions",
            "actual_per_task_transitions",
            "contracts",
            "curriculum",
            "iteration",
            "metrics",
            "next_crossing",
            "optimizer",
            "optimizer_sha256",
            "optimizer_step",
            "policy",
            "policy_sha256",
            "rank_rng",
            "source_manifest",
            "task_transitions",
        )
        if not all(
            _semantic_equal(payload[name], restored[name]) for name in protected
        ):
            raise AssertionError("summaryschema finalization altered training state")
        digest = atomic_learning_checkpoint(
            post_checkpoint, payload, latest_path=latest
        )
        latest_payload = _read_json(latest)
        if latest_payload != {"checkpoint": str(post_checkpoint), "sha256": digest}:
            raise RuntimeError("summaryschema latest checkpoint record is invalid")
        post_policy = ResidualActorCritic()
        post_optimizer = torch.optim.Adam(post_policy.parameters(), lr=3.0e-4)
        post_restored = restore_learning_checkpoint(
            post_checkpoint,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=post_policy,
            optimizer=post_optimizer,
            source_manifest=_current_pairedbatch_source_manifest(),
            expected_iteration=31,
            device="cpu",
        )
        if not _semantic_equal(post_restored, payload):
            raise AssertionError(
                "summaryschema post-evaluation checkpoint did not restore"
            )
        _validate_summaryschema_failed_finalize_manifest(
            recovery / "evaluation_warningpolicy_attempt_0007_frozen_manifest.json"
        )
        log = "PHASE6_SUMMARYSCHEMA_FINALIZED=" + json.dumps(
            latest_payload, sort_keys=True
        )
        (finalized_attempt / "finalize_summaryschema.log").write_text(
            log + "\n", encoding="utf-8"
        )
        _atomic_json(
            finalized_attempt / "finalize_summaryschema.status.json",
            {
                "exit_code": 0,
                "signal": None,
                "timeout": False,
                "vmhwm_kib": _proc_hwm_kib(os.getpid()),
            },
        )
        print(log, flush=True)
        return 0
    except Exception:
        (finalized_attempt / "finalize_summaryschema.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        _atomic_json(
            finalized_attempt / "finalize_summaryschema.status.json",
            {
                "exit_code": 1,
                "signal": None,
                "timeout": False,
                "vmhwm_kib": _proc_hwm_kib(os.getpid()),
            },
        )
        raise


def _validate_modeprocess_result(
    result: object,
    *,
    rank: int,
    roster: object,
    mode: str,
    stage_quota: int,
    nominal_quota: int,
) -> Mapping[str, object]:
    expected_keys = {
        "completed_seed_sha256",
        "control_steps",
        "deterministic_actor_mean",
        "episodes",
        "horizon",
        "mode",
        "mode_counts",
        "normalizer_updates",
        "optimizer_steps",
        "rank",
        "schedule_sha256",
        "status",
        "subset_counts",
        "task",
        "unique_seeds",
        "world_size",
    }
    payload = _require_exact_mapping(
        result,
        name=f"modeprocess.{mode}.rank_{rank}.result",
        expected=expected_keys
        | (
            {"episode_evidence"}
            if isinstance(result, Mapping) and "episode_evidence" in result
            else set()
        ),
    )
    expected_per_mode = stage_quota + nominal_quota
    expected_mode_counts = {
        "residual": expected_per_mode if mode == "residual" else 0,
        "scaffold_only": expected_per_mode if mode == "scaffold_only" else 0,
    }
    task = roster.tasks[rank].task
    if (
        payload["mode"] != mode
        or payload["rank"] != rank
        or payload["task"] != task
        or payload["status"] != "ok"
        or payload["world_size"] != 4
        or payload["episodes"] != expected_per_mode
        or payload["control_steps"] <= 0
        or payload["deterministic_actor_mean"] is not True
        or payload["optimizer_steps"] != 0
        or payload["normalizer_updates"] != 0
        or payload["mode_counts"] != expected_mode_counts
        or payload["subset_counts"] != {"stage": stage_quota, "nominal": nominal_quota}
        or payload["unique_seeds"] != expected_per_mode
    ):
        raise ValueError("modeprocess single-mode result is invalid")
    _require_sha256(payload["schedule_sha256"], name="modeprocess.schedule_sha256")
    _require_sha256(
        payload["completed_seed_sha256"], name="modeprocess.completed_seed_sha256"
    )
    if "episode_evidence" in payload:
        evidence = _require_exact_mapping(
            payload["episode_evidence"],
            name="modeprocess.episode_evidence",
            expected={"path", "row_count", "schedule_sha256", "seed_sha256", "sha256"},
        )
        if (
            not isinstance(evidence["path"], str)
            or evidence["row_count"] != expected_per_mode
            or evidence["schedule_sha256"] != payload["schedule_sha256"]
            or evidence["seed_sha256"] != payload["completed_seed_sha256"]
        ):
            raise ValueError("modeprocess completed episode evidence is invalid")
        _require_sha256(evidence["sha256"], name="modeprocess.episode_evidence.sha")
    return payload


def _validate_modeprocess_wrapper(
    wrapper: object,
    *,
    rank: int,
    rank_dir: Path,
    mode: str,
) -> Mapping[str, object]:
    payload = _require_exact_mapping(
        wrapper,
        name=f"modeprocess.{mode}.rank_{rank}.wrapper",
        expected={
            "close_verification",
            "command",
            "elapsed_s",
            "exit_code",
            "last_progress_marker",
            "local_rank",
            "log",
            "marker_order_valid",
            "markers",
            "passed",
            "rank",
            "result_error",
            "result_json",
            "result_status",
            "signal",
            "timed_out",
            "timeout_kind",
            "vmhwm_kib",
            "warning_counts",
            "world_size",
        },
    )
    command = payload["command"]
    if not isinstance(command, list) or "--paired-mode" not in command:
        raise ValueError("modeprocess wrapper command is incomplete")
    mode_index = command.index("--paired-mode") + 1
    if (
        mode_index >= len(command)
        or command[mode_index] != mode
        or payload["rank"] != rank
        or payload["world_size"] != 4
        or payload["exit_code"] != 0
        or payload["signal"] is not None
        or payload["timed_out"] is not False
        or payload["timeout_kind"] is not None
        or payload["close_verification"] != "parent_verified_exit0"
        or payload["passed"] is not True
        or payload["last_progress_marker"] != "ENV_CLOSED"
        or payload["markers"] != ["JSON_WRITTEN", "PHASE6_JOINT_RESULT", "ENV_CLOSED"]
        or payload["result_status"] != "ok"
        or not isinstance(payload["vmhwm_kib"], int)
        or payload["vmhwm_kib"] <= 0
    ):
        raise ValueError("modeprocess wrapper close or timeout evidence is invalid")
    warnings = _require_exact_mapping(
        payload["warning_counts"],
        name="modeprocess.wrapper.warning_counts",
        expected={"headless_glfw", "kvdb_lock", "multiple_installable_client_drivers"},
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in warnings.values()
    ):
        raise ValueError("modeprocess wrapper warning counts are invalid")
    worker_log = rank_dir / "worker.log"
    if not worker_log.is_file():
        raise ValueError("modeprocess wrapper worker log is missing")
    observed_warnings = _worker_warning_counts(
        worker_log.read_text(encoding="utf-8", errors="replace")
    )
    if dict(warnings) != observed_warnings:
        raise ValueError("modeprocess wrapper warning counts do not match worker log")
    if warnings["multiple_installable_client_drivers"] > 0:
        raise RuntimeError("modeprocess worker reports multiple Vulkan ICDs")
    result = _read_json(rank_dir / "result.json")
    marker = _last_result_marker(worker_log)
    if result is None or marker != result:
        raise ValueError("modeprocess wrapper result marker is invalid")
    return payload


def _merge_modeprocess_evaluation(
    *,
    attempt: Path,
    roster: object,
    stage_quota: int,
    nominal_quota: int,
) -> dict[str, object]:
    from somaforce_cross.learning.joint_runner import (
        validate_learning_evaluation_summary,
    )

    modes = ("residual", "scaffold_only")
    per_mode: dict[str, list[Mapping[str, object]]] = {mode: [] for mode in modes}
    for mode in modes:
        for rank in range(4):
            rank_dir = attempt / mode / f"rank_{rank}"
            _validate_modeprocess_wrapper(
                _read_json(rank_dir / "wrapper.json"),
                rank=rank,
                rank_dir=rank_dir,
                mode=mode,
            )
            result = _validate_modeprocess_result(
                _read_json(rank_dir / "result.json"),
                rank=rank,
                roster=roster,
                mode=mode,
                stage_quota=stage_quota,
                nominal_quota=nominal_quota,
            )
            per_mode[mode].append(result)
    combined: list[dict[str, object]] = []
    expected_per_mode = stage_quota + nominal_quota
    for rank, (residual, scaffold) in enumerate(
        zip(per_mode["residual"], per_mode["scaffold_only"], strict=True)
    ):
        if (
            residual["task"] != scaffold["task"]
            or residual["schedule_sha256"] != scaffold["schedule_sha256"]
            or residual["completed_seed_sha256"] != scaffold["completed_seed_sha256"]
        ):
            raise ValueError("modeprocess modes disagree on task schedule or seeds")
        combined.append(
            {
                "control_steps": int(residual["control_steps"])
                + int(scaffold["control_steps"]),
                "episodes": expected_per_mode * 2,
                "mode_counts": {
                    "residual": expected_per_mode,
                    "scaffold_only": expected_per_mode,
                },
                "rank": rank,
                "schedule_sha256": residual["schedule_sha256"],
                "status": "ok",
                "subset_counts": {
                    "stage": stage_quota * 2,
                    "nominal": nominal_quota * 2,
                },
                "task": residual["task"],
                "unique_seeds": expected_per_mode,
            }
        )
    schedule_hashes = [str(result["schedule_sha256"]) for result in combined]
    summary = validate_learning_evaluation_summary(
        {
            "iteration": 31,
            "rank_results": combined,
            "schedule_sha256": hashlib.sha256(
                json.dumps(schedule_hashes, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "status": "ok",
            "task_metrics": {
                str(result["task"]): {
                    "control_steps": result["control_steps"],
                    "episodes": result["episodes"],
                    "mode_counts": result["mode_counts"],
                    "schedule_sha256": result["schedule_sha256"],
                    "subset_counts": result["subset_counts"],
                    "unique_seeds": result["unique_seeds"],
                }
                for result in combined
            },
        },
        roster=roster,
        expected_iteration=31,
        expected_stage_quota=stage_quota,
        expected_nominal_quota=nominal_quota,
    )
    _atomic_json(attempt / "evaluation_summary.json", summary)
    _aggregate_modeprocess_progress(attempt)
    return summary


def _read_mainrunner_episode_evidence(
    *,
    rank_dir: Path,
    result: Mapping[str, object],
    task: str,
    mode: str,
    stage: str,
    stage_quota: int,
    nominal_quota: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evidence = result.get("episode_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("mainrunner evaluation result lacks episode evidence")
    expected = {"path", "row_count", "schedule_sha256", "seed_sha256", "sha256"}
    if set(evidence) != expected:
        raise ValueError("mainrunner episode evidence schema is invalid")
    path = rank_dir / "completed_episode_records.json"
    if (
        evidence["path"] != str(path)
        or not path.is_file()
        or evidence["sha256"] != _sha256(path)
        or evidence["row_count"] != stage_quota + nominal_quota
        or evidence["schedule_sha256"] != result["schedule_sha256"]
        or evidence["seed_sha256"] != result["completed_seed_sha256"]
    ):
        raise ValueError("mainrunner episode evidence binding is invalid")
    payload = _read_json(path)
    if payload is None or set(payload) != {
        "records",
        "row_count",
        "schedule_sha256",
        "seed_sha256",
        "subset_counts",
    }:
        raise ValueError("mainrunner completed episode evidence file is invalid")
    if (
        payload["row_count"] != evidence["row_count"]
        or payload["schedule_sha256"] != evidence["schedule_sha256"]
        or payload["seed_sha256"] != evidence["seed_sha256"]
        or payload["subset_counts"] != {"stage": stage_quota, "nominal": nominal_quota}
        or not isinstance(payload["records"], list)
    ):
        raise ValueError("mainrunner completed episode evidence metadata is invalid")
    records = [
        _validate_completed_episode_record(record, task=task, mode=mode, stage=stage)
        for record in payload["records"]
        if isinstance(record, Mapping)
    ]
    if len(records) != len(payload["records"]):
        raise ValueError("mainrunner completed episode evidence record is invalid")
    seeds = sorted(int(record["seed"]) for record in records)
    if (
        len(seeds) != len(set(seeds))
        or hashlib.sha256(
            json.dumps(seeds, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        != payload["seed_sha256"]
    ):
        raise ValueError(
            "mainrunner completed episode evidence seed binding is invalid"
        )
    return records, {
        "path": str(path),
        "row_count": payload["row_count"],
        "schedule_sha256": payload["schedule_sha256"],
        "seed_sha256": payload["seed_sha256"],
        "sha256": _sha256(path),
    }


def _mean_record_value(records: list[dict[str, object]], key: str) -> float:
    if not records:
        raise ValueError(f"mainrunner record group {key} is empty")
    return sum(_finite_json_number(record[key], name=key) for record in records) / len(
        records
    )


def _mean_record_flag(records: list[dict[str, object]], key: str) -> float:
    if not records or any(not isinstance(record[key], bool) for record in records):
        raise ValueError(f"mainrunner record flag {key} is invalid")
    return sum(bool(record[key]) for record in records) / len(records)


def _mainrunner_task_metrics(
    *,
    residual_records: list[dict[str, object]],
    scaffold_records: list[dict[str, object]],
) -> dict[str, object]:
    residual_stage = [
        record for record in residual_records if record["subset"] == "stage"
    ]
    residual_nominal = [
        record for record in residual_records if record["subset"] == "nominal"
    ]
    scaffold_nominal = [
        record for record in scaffold_records if record["subset"] == "nominal"
    ]
    if not residual_stage or not residual_nominal or not scaffold_nominal:
        raise ValueError("mainrunner task metrics require stage and nominal records")
    nominal_scaffold_success = _mean_record_flag(scaffold_nominal, "success")
    if nominal_scaffold_success <= 0.0:
        raise ValueError("mainrunner nominal scaffold success cannot define retention")
    families: dict[str, dict[str, list[dict[str, object]]]] = {}
    for mode, records in (
        ("residual", residual_records),
        ("scaffold_only", scaffold_records),
    ):
        for record in records:
            family = str(record["family"])
            families.setdefault(family, {}).setdefault(mode, []).append(record)
    family_metrics: dict[str, object] = {}
    for family, by_mode in sorted(families.items()):
        if set(by_mode) != {"residual", "scaffold_only"}:
            raise ValueError("mainrunner family is not paired across modes")
        family_metrics[family] = {
            mode: {
                "failure": _mean_record_flag(records, "failure"),
                "return": _mean_record_value(records, "return"),
                "success": _mean_record_flag(records, "success"),
                "p95_wrench": sum(
                    float(record["diagnostics"]["p95_wrench"]) for record in records
                )
                / len(records),
            }
            for mode, records in by_mode.items()
        }
    return {
        "family_metrics": family_metrics,
        "failure": _mean_record_flag(residual_stage, "failure"),
        "nominal_residual_success": _mean_record_flag(residual_nominal, "success"),
        "nominal_scaffold_success": nominal_scaffold_success,
        "retention": _mean_record_flag(residual_nominal, "success")
        / nominal_scaffold_success,
        "success": _mean_record_flag(residual_stage, "success"),
    }


def _merge_mainrunner_evaluation(
    *,
    evaluation_dir: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    iteration: int,
    stage: str,
    roster: object,
    source_manifest: Mapping[str, str],
    source_rebind: Path,
    stage_quota: int,
    nominal_quota: int,
) -> dict[str, object]:
    from somaforce_cross.learning.joint_runner import (
        Phase6Roster,
        validate_learning_evaluation_summary,
    )

    if not isinstance(roster, Phase6Roster):
        raise TypeError("mainrunner evaluation roster is invalid")
    if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("mainrunner evaluation checkpoint binding is invalid")
    modes = ("residual", "scaffold_only")
    per_mode: dict[str, list[Mapping[str, object]]] = {mode: [] for mode in modes}
    episode_records: dict[tuple[str, int], list[dict[str, object]]] = {}
    evidence_manifests: dict[str, dict[str, object]] = {mode: {} for mode in modes}
    for mode in modes:
        for rank, task in enumerate(roster.tasks):
            rank_dir = evaluation_dir / mode / f"rank_{rank}"
            _validate_modeprocess_wrapper(
                _read_json(rank_dir / "wrapper.json"),
                rank=rank,
                rank_dir=rank_dir,
                mode=mode,
            )
            result = _validate_modeprocess_result(
                _read_json(rank_dir / "result.json"),
                rank=rank,
                roster=roster,
                mode=mode,
                stage_quota=stage_quota,
                nominal_quota=nominal_quota,
            )
            records, evidence = _read_mainrunner_episode_evidence(
                rank_dir=rank_dir,
                result=result,
                task=task.task,
                mode=mode,
                stage=stage,
                stage_quota=stage_quota,
                nominal_quota=nominal_quota,
            )
            per_mode[mode].append(result)
            episode_records[(mode, rank)] = records
            evidence_manifests[mode][str(rank)] = evidence
    rank_results: list[dict[str, object]] = []
    task_metrics: dict[str, object] = {}
    for rank, task in enumerate(roster.tasks):
        residual = per_mode["residual"][rank]
        scaffold = per_mode["scaffold_only"][rank]
        residual_records = episode_records[("residual", rank)]
        scaffold_records = episode_records[("scaffold_only", rank)]
        if (
            residual["schedule_sha256"] != scaffold["schedule_sha256"]
            or residual["completed_seed_sha256"] != scaffold["completed_seed_sha256"]
            or sorted(int(record["seed"]) for record in residual_records)
            != sorted(int(record["seed"]) for record in scaffold_records)
        ):
            raise ValueError("mainrunner paired modes disagree on completed row seeds")
        rank_results.append(
            {
                "control_steps": int(residual["control_steps"])
                + int(scaffold["control_steps"]),
                "episodes": (stage_quota + nominal_quota) * 2,
                "mode_counts": {
                    "residual": stage_quota + nominal_quota,
                    "scaffold_only": stage_quota + nominal_quota,
                },
                "rank": rank,
                "schedule_sha256": residual["schedule_sha256"],
                "status": "ok",
                "subset_counts": {
                    "stage": stage_quota * 2,
                    "nominal": nominal_quota * 2,
                },
                "task": task.task,
                "unique_seeds": stage_quota + nominal_quota,
            }
        )
        task_metrics[task.task] = _mainrunner_task_metrics(
            residual_records=residual_records, scaffold_records=scaffold_records
        )
    schedule_hashes = [str(result["schedule_sha256"]) for result in rank_results]
    compact = validate_learning_evaluation_summary(
        {
            "iteration": iteration,
            "rank_results": rank_results,
            "schedule_sha256": hashlib.sha256(
                json.dumps(schedule_hashes, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "status": "ok",
            "task_metrics": task_metrics,
        },
        roster=roster,
        expected_iteration=iteration,
        expected_stage_quota=stage_quota,
        expected_nominal_quota=nominal_quota,
    )
    return {
        **compact,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        "curriculum_task_metrics": task_metrics,
        "evidence_manifests": evidence_manifests,
        "source_manifest_sha256": _source_manifest_digest(source_manifest),
        "source_rebind": {"path": str(source_rebind), "sha256": _sha256(source_rebind)},
        "stage": stage,
    }


def _run_mainrunner_evaluation_merge(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    output = args.evaluation_dir / "evaluation_summary.json"
    if output.exists():
        raise FileExistsError("mainrunner evaluation summary already exists")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_mainrunner_rebind_record(
        args.source_rebind, config=config, roster=roster, acceptance=acceptance
    )
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        args.pre_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_mainrunner_source_manifest(),
        expected_iteration=args.iteration,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("mainrunner evaluation requires a pre-evaluation checkpoint")
    summary = _merge_mainrunner_evaluation(
        evaluation_dir=args.evaluation_dir,
        checkpoint=args.pre_checkpoint,
        checkpoint_sha256=_sha256(args.pre_checkpoint),
        iteration=args.iteration,
        stage=args.stage,
        roster=roster,
        source_manifest=_current_mainrunner_source_manifest(),
        source_rebind=args.source_rebind,
        stage_quota=args.paired_stage_quota,
        nominal_quota=args.paired_nominal_quota,
    )
    _atomic_json(output, summary)
    print(
        "PHASE6_MAINRUNNER_EVALUATION=" + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    return 0


def _validate_mainrunner_evaluation_summary(
    summary: Mapping[str, object],
    *,
    iteration: int,
    stage: str,
    checkpoint: Path,
    source_manifest: Mapping[str, str],
    source_rebind: Path,
) -> dict[str, object]:
    required = {
        "checkpoint",
        "curriculum_task_metrics",
        "evidence_manifests",
        "iteration",
        "rank_results",
        "schedule_sha256",
        "source_manifest_sha256",
        "source_rebind",
        "stage",
        "status",
        "task_metrics",
    }
    if set(summary) != required or summary["status"] != "ok":
        raise ValueError("mainrunner evaluation summary schema is invalid")
    checkpoint_record = _require_exact_mapping(
        summary["checkpoint"],
        name="mainrunner.evaluation.checkpoint",
        expected={"path", "sha256"},
    )
    source_rebind_record = _require_exact_mapping(
        summary["source_rebind"],
        name="mainrunner.evaluation.source_rebind",
        expected={"path", "sha256"},
    )
    if (
        summary["iteration"] != iteration
        or summary["stage"] != stage
        or checkpoint_record["path"] != str(checkpoint)
        or checkpoint_record["sha256"] != _sha256(checkpoint)
        or source_rebind_record["path"] != str(source_rebind)
        or source_rebind_record["sha256"] != _sha256(source_rebind)
        or summary["source_manifest_sha256"] != _source_manifest_digest(source_manifest)
        or summary["task_metrics"] != summary["curriculum_task_metrics"]
        or not isinstance(summary["curriculum_task_metrics"], Mapping)
        or set(summary["curriculum_task_metrics"])
        != {"push_door_hand", "push_box", "move_suitcase", "move_largebox"}
    ):
        raise ValueError("mainrunner evaluation summary binding is invalid")
    for task, metrics in summary["curriculum_task_metrics"].items():
        if not isinstance(metrics, Mapping) or not {
            "success",
            "failure",
            "retention",
            "family_metrics",
        }.issubset(metrics):
            raise ValueError(f"mainrunner curriculum task metrics are invalid: {task}")
        for name in ("success", "failure", "retention"):
            _finite_json_number(metrics[name], name=f"{task}.{name}")
        if (
            not isinstance(metrics["family_metrics"], Mapping)
            or not metrics["family_metrics"]
        ):
            raise ValueError(f"mainrunner family metrics are invalid: {task}")
    return dict(summary)


def _mainrunner_command_argument(values: list[str], name: str) -> str:
    if values.count(name) != 1:
        raise ValueError(f"mainrunner command must contain exactly one {name}")
    return _command_argument(values, name)


def _mainrunner_command_path(values: list[str], name: str) -> Path:
    return Path(_mainrunner_command_argument(values, name)).resolve()


def _validate_mainrunner_stage_evidence(log_path: Path, *, stage: str) -> list[str]:
    command = _require_exact_mapping(
        _read_json(log_path.with_suffix(".command.json")),
        name=f"mainrunner.{stage}.command",
        expected={"command", "stage", "timeout_s"},
    )
    status = _require_exact_mapping(
        _read_json(log_path.with_suffix(".status.json")),
        name=f"mainrunner.{stage}.status",
        expected={"elapsed_s", "exit_code", "signal", "stage", "timeout", "vmhwm_kib"},
    )
    if (
        not log_path.is_file()
        or command["stage"] != stage
        or isinstance(command["timeout_s"], bool)
        or not isinstance(command["timeout_s"], int)
        or command["timeout_s"] <= 0
        or not isinstance(command["command"], list)
        or not all(isinstance(value, str) for value in command["command"])
        or status["stage"] != stage
        or _finite_json_number(status["elapsed_s"], name=f"{stage}.elapsed_s") < 0.0
        or status["exit_code"] != 0
        or status["signal"] is not None
        or status["timeout"] is not False
        or isinstance(status["vmhwm_kib"], bool)
        or not isinstance(status["vmhwm_kib"], int)
        or status["vmhwm_kib"] <= 0
    ):
        raise ValueError(f"mainrunner {stage} stage evidence is invalid")
    return list(command["command"])


def _validate_mainrunner_worker_stage_command(
    values: list[str],
    *,
    worker_mode: str,
    output_dir: Path,
    segment: object,
    resume_checkpoint: Path,
    source_rebind: Path,
    paired_mode: str | None = None,
) -> None:
    if len(values) < 7 or values[:6] != [
        str(ISAAC_PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
    ]:
        raise ValueError("mainrunner distributed stage command prefix is invalid")
    if (
        Path(values[6]).resolve() != Path(__file__).resolve()
        or _mainrunner_command_argument(values, "--mode") != "rank-wrapper"
        or _mainrunner_command_argument(values, "--worker-mode") != worker_mode
        or _mainrunner_command_path(values, "--output-dir") != output_dir.resolve()
        or _mainrunner_command_argument(values, "--profile") != "production_segment"
        or _mainrunner_command_path(values, "--resume") != resume_checkpoint.resolve()
        or _mainrunner_command_path(values, "--source-rebind")
        != source_rebind.resolve()
        or _mainrunner_command_path(values, "--acceptance-config")
        != PHASE6_LEARNING_CONFIG.resolve()
        or _mainrunner_command_path(values, "--phase6-config")
        != PHASE6_CONFIG.resolve()
        or _mainrunner_command_path(values, "--roster") != PHASE6_ROSTER.resolve()
        or "--headless" in values
    ):
        raise ValueError("mainrunner distributed stage command binding is invalid")
    expected_end = int(getattr(segment, "end_iteration"))
    if worker_mode == "train-segment":
        if (
            int(_mainrunner_command_argument(values, "--segment-start"))
            != int(getattr(segment, "start_iteration"))
            or int(_mainrunner_command_argument(values, "--segment-end"))
            != expected_end
        ):
            raise ValueError("mainrunner train command segment boundary is invalid")
        if paired_mode is not None or "--paired-mode" in values:
            raise ValueError("mainrunner train command has paired-mode metadata")
        return
    if (
        worker_mode != "evaluate"
        or int(_mainrunner_command_argument(values, "--segment-start")) != expected_end
        or int(_mainrunner_command_argument(values, "--segment-end"))
        != expected_end + 1
        or paired_mode not in {"residual", "scaffold_only"}
        or _mainrunner_command_argument(values, "--paired-mode") != paired_mode
    ):
        raise ValueError("mainrunner evaluation command binding is invalid")


def _validate_mainrunner_utility_stage_command(values: list[str], *, mode: str) -> None:
    if (
        len(values) < 3
        or values[0] != str(ISAAC_PYTHON)
        or Path(values[1]).resolve() != Path(__file__).resolve()
        or _mainrunner_command_argument(values, "--mode") != mode
    ):
        raise ValueError("mainrunner utility stage command is invalid")


def _validate_mainrunner_completed_train(
    *,
    segment_dir: Path,
    segment: object,
    resume_checkpoint: Path,
    source_rebind: Path,
    config: object,
    roster: object,
    acceptance: object,
    source_manifest: Mapping[str, str],
) -> tuple[Path, dict[str, object]]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        optimizer_step,
        restore_learning_checkpoint,
        state_dict_sha256,
    )

    train_dir = segment_dir / "train"
    pre_checkpoint = train_dir / "pre_evaluation.pt"
    if not train_dir.is_dir() or not pre_checkpoint.is_file():
        raise FileNotFoundError("mainrunner completed train checkpoint is missing")
    train_command = _validate_mainrunner_stage_evidence(
        segment_dir / "train.log", stage="train-segment"
    )
    _validate_mainrunner_worker_stage_command(
        train_command,
        worker_mode="train-segment",
        output_dir=train_dir,
        segment=segment,
        resume_checkpoint=resume_checkpoint,
        source_rebind=source_rebind,
    )
    expected_global = int(getattr(segment, "end_iteration")) * 8192
    expected_per_task = expected_global // 4
    expected_local_transitions = (
        (
            int(getattr(segment, "end_iteration"))
            - int(getattr(segment, "start_iteration"))
        )
        * 64
        * 32
    )
    rank_results: list[Mapping[str, object]] = []
    wrappers: list[Mapping[str, object]] = []
    checkpoint: Mapping[str, object] | None = None
    tasks: set[str] = set()
    for rank in range(4):
        rank_dir = train_dir / f"rank_{rank}"
        command = _validate_production_command(
            _read_json(rank_dir / "command.json"),
            rank=rank,
            output_dir=train_dir,
            worker_mode="train-segment",
            iteration=int(getattr(segment, "end_iteration")),
        )
        values = list(command["command"])
        if (
            _mainrunner_command_path(values, "--resume") != resume_checkpoint.resolve()
            or _mainrunner_command_path(values, "--source-rebind")
            != source_rebind.resolve()
        ):
            raise ValueError("mainrunner train rank command does not bind resume")
        wrapper = _validate_summary_wrapper(
            _read_json(rank_dir / "wrapper.json"),
            command=command,
            rank=rank,
            rank_dir=rank_dir,
        )
        if not isinstance(wrapper["vmhwm_kib"], int) or wrapper["vmhwm_kib"] <= 0:
            raise ValueError("mainrunner train rank wrapper VmHWM is invalid")
        result = _validate_production_train_result(
            _read_json(rank_dir / "result.json"),
            rank=rank,
            config=config,
            roster=roster,
            expected_global=expected_global,
            expected_per_task=expected_per_task,
            expected_local_transitions=expected_local_transitions,
            expected_cycle_index=int(getattr(segment, "segment_index")),
            expected_window_index=0,
        )
        result_checkpoint = _require_exact_mapping(
            result["checkpoint"],
            name=f"mainrunner.train.rank_{rank}.checkpoint",
            expected={"path", "sha256"},
        )
        if (
            Path(str(result_checkpoint["path"])).resolve() != pre_checkpoint.resolve()
            or _sha256(pre_checkpoint) != result_checkpoint["sha256"]
            or result["schedule"]["cycle_index"]
            != int(getattr(segment, "segment_index"))
        ):
            raise ValueError("mainrunner train rank checkpoint binding is invalid")
        if checkpoint is None:
            checkpoint = result_checkpoint
        elif checkpoint != result_checkpoint:
            raise ValueError("mainrunner ranks disagree on pre-evaluation checkpoint")
        tasks.add(str(result["schedule"]["task"]))
        rank_results.append(result)
        wrappers.append(wrapper)
    if tasks != {task.task for task in roster.tasks}:
        raise ValueError("mainrunner train ranks do not cover the roster")
    resume_policy = ResidualActorCritic()
    resume_optimizer = torch.optim.Adam(resume_policy.parameters(), lr=3.0e-4)
    resume_restored = restore_learning_checkpoint(
        resume_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=resume_policy,
        optimizer=resume_optimizer,
        source_manifest=source_manifest,
        expected_iteration=int(getattr(segment, "start_iteration")),
        device="cpu",
    )
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        pre_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=source_manifest,
        expected_iteration=int(getattr(segment, "end_iteration")),
        device="cpu",
    )
    if (
        restored["pre_evaluation"] is not True
        or restored["actual_global_transitions"] != expected_global
        or restored["actual_per_task_transitions"] != expected_per_task
        or restored["policy_sha256"] != rank_results[0]["policy_sha256"]
        or restored["optimizer_sha256"] != rank_results[0]["optimizer_sha256"]
        or restored["optimizer_step"] != rank_results[0]["optimizer_step"]
        or restored["metrics"] != rank_results[0]["metrics"]
        or state_dict_sha256(policy.state_dict()) != restored["policy_sha256"]
        or state_dict_sha256(optimizer.state_dict()) != restored["optimizer_sha256"]
        or optimizer_step(optimizer) != restored["optimizer_step"]
        or any(
            result["metrics"] != rank_results[0]["metrics"]
            for result in rank_results[1:]
        )
    ):
        raise ValueError("mainrunner pre-evaluation checkpoint restore is invalid")
    if (
        resume_restored["pre_evaluation"] is not False
        or not isinstance(resume_restored["evaluation_history"], list)
        or not all(
            isinstance(item, Mapping) for item in resume_restored["evaluation_history"]
        )
        or not isinstance(restored["evaluation_history"], list)
        or not all(isinstance(item, Mapping) for item in restored["evaluation_history"])
        or not _semantic_equal(
            resume_restored["evaluation_history"], restored["evaluation_history"]
        )
    ):
        raise ValueError("mainrunner resume evaluation history is not continuous")
    summary_command = _validate_mainrunner_stage_evidence(
        segment_dir / "summary.log", stage="summarize"
    )
    _validate_mainrunner_utility_stage_command(
        summary_command, mode="production-summarize"
    )
    if (
        _mainrunner_command_path(summary_command, "--output-dir") != train_dir.resolve()
        or int(_mainrunner_command_argument(summary_command, "--iteration"))
        != int(getattr(segment, "end_iteration"))
        or _mainrunner_command_path(summary_command, "--source-rebind")
        != source_rebind.resolve()
    ):
        raise ValueError("mainrunner train summary command binding is invalid")
    train_summary = _read_json(train_dir / "segment_summary.json")
    expected_summary = {
        "checkpoint": {"path": str(pre_checkpoint), "sha256": _sha256(pre_checkpoint)},
        "iteration": int(getattr(segment, "end_iteration")),
        "rank_results": [dict(result) for result in rank_results],
        "source_manifest_sha256": _source_manifest_digest(source_manifest),
        "source_rebind": {"path": str(source_rebind), "sha256": _sha256(source_rebind)},
        "status": "ok",
        "task_metrics": dict(restored["metrics"]),
    }
    evidence_summary = _read_json(train_dir / "production_evidence_summary.json")
    if train_summary != expected_summary or evidence_summary != {
        "iteration": int(getattr(segment, "end_iteration")),
        "rank_wrappers": [dict(wrapper) for wrapper in wrappers],
        "status": "ok",
        "topology": "per_rank_wrapper",
        "worker_mode": "train-segment",
    }:
        raise ValueError("mainrunner train summary binding is invalid")
    return pre_checkpoint, restored


def _validate_mainrunner_completed_evaluation(
    *,
    segment_dir: Path,
    segment: object,
    pre_checkpoint: Path,
    stage: str,
    source_rebind: Path,
    roster: object,
    source_manifest: Mapping[str, str],
) -> dict[str, object]:
    evaluation_dir = segment_dir / "evaluation"
    if not evaluation_dir.is_dir():
        raise FileNotFoundError("mainrunner completed evaluation directory is missing")
    for mode in ("residual", "scaffold_only"):
        mode_dir = evaluation_dir / mode
        if not mode_dir.is_dir():
            raise FileNotFoundError(
                f"mainrunner completed {mode} evaluation is missing"
            )
        command = _validate_mainrunner_stage_evidence(
            segment_dir / f"evaluation_{mode}.log",
            stage=f"paired-evaluate-{mode}",
        )
        _validate_mainrunner_worker_stage_command(
            command,
            worker_mode="evaluate",
            output_dir=mode_dir,
            segment=segment,
            resume_checkpoint=pre_checkpoint,
            source_rebind=source_rebind,
            paired_mode=mode,
        )
    summary_command = _validate_mainrunner_stage_evidence(
        segment_dir / "evaluation-summary.log", stage="evaluation-summary"
    )
    _validate_mainrunner_utility_stage_command(
        summary_command, mode="mainrunner-evaluation-merge"
    )
    if (
        _mainrunner_command_path(summary_command, "--evaluation-dir")
        != evaluation_dir.resolve()
        or _mainrunner_command_path(summary_command, "--pre-checkpoint")
        != pre_checkpoint.resolve()
        or int(_mainrunner_command_argument(summary_command, "--iteration"))
        != int(getattr(segment, "end_iteration"))
        or _mainrunner_command_argument(summary_command, "--stage") != stage
        or _mainrunner_command_path(summary_command, "--source-rebind")
        != source_rebind.resolve()
    ):
        raise ValueError("mainrunner evaluation summary command binding is invalid")
    summary_value = _read_json(evaluation_dir / "evaluation_summary.json")
    if summary_value is None:
        raise FileNotFoundError("mainrunner completed evaluation summary is missing")
    summary = _validate_mainrunner_evaluation_summary(
        summary_value,
        iteration=int(getattr(segment, "end_iteration")),
        stage=stage,
        checkpoint=pre_checkpoint,
        source_manifest=source_manifest,
        source_rebind=source_rebind,
    )
    rebuilt = _merge_mainrunner_evaluation(
        evaluation_dir=evaluation_dir,
        checkpoint=pre_checkpoint,
        checkpoint_sha256=_sha256(pre_checkpoint),
        iteration=int(getattr(segment, "end_iteration")),
        stage=stage,
        roster=roster,
        source_manifest=source_manifest,
        source_rebind=source_rebind,
        stage_quota=256,
        nominal_quota=128,
    )
    if summary != rebuilt:
        raise ValueError("mainrunner evaluation summary SHA or business rows changed")
    return summary


def _validate_mainrunner_completed_finalize(
    *,
    run_dir: Path,
    segment_dir: Path,
    segment: object,
    pre_checkpoint: Path,
    pre_restored: Mapping[str, object],
    evaluation: Mapping[str, object],
    source_rebind: Path,
    config: object,
    roster: object,
    acceptance: object,
    source_manifest: Mapping[str, str],
) -> dict[str, object]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        JointCurriculum,
        restore_learning_checkpoint,
        validate_learning_final_acceptance,
    )

    command = _validate_mainrunner_stage_evidence(
        segment_dir / "finalize.log", stage="finalize-checkpoint"
    )
    _validate_mainrunner_utility_stage_command(command, mode="mainrunner-finalize")
    if (
        _mainrunner_command_path(command, "--run-dir") != run_dir.resolve()
        or _mainrunner_command_path(command, "--segment-dir") != segment_dir.resolve()
        or _mainrunner_command_path(command, "--pre-checkpoint")
        != pre_checkpoint.resolve()
        or _mainrunner_command_path(command, "--evaluation-summary")
        != (segment_dir / "evaluation" / "evaluation_summary.json").resolve()
        or int(_mainrunner_command_argument(command, "--iteration"))
        != int(getattr(segment, "end_iteration"))
        or int(_mainrunner_command_argument(command, "--segment-index"))
        != int(getattr(segment, "segment_index"))
        or int(_mainrunner_command_argument(command, "--logical-crossing"))
        != int(getattr(segment, "crossing").logical_transitions)
        or _mainrunner_command_path(command, "--source-rebind")
        != source_rebind.resolve()
    ):
        raise ValueError("mainrunner finalize command binding is invalid")
    post_checkpoint = segment_dir / "post_evaluation.pt"
    if not post_checkpoint.is_file():
        raise FileNotFoundError("mainrunner post-evaluation checkpoint is missing")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        post_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=source_manifest,
        expected_iteration=int(getattr(segment, "end_iteration")),
        device="cpu",
    )
    curriculum = JointCurriculum()
    curriculum.load_state_dict(pre_restored["curriculum"])
    decision = curriculum.evaluate_window(
        crossing=int(getattr(segment, "crossing").logical_transitions),
        task_metrics=evaluation["curriculum_task_metrics"],
        actual_transitions=int(pre_restored["actual_global_transitions"]),
    )
    expected_history = [*pre_restored["evaluation_history"], dict(evaluation)]
    if (
        restored["pre_evaluation"] is not False
        or restored["actual_global_transitions"]
        != int(getattr(segment, "end_transitions"))
        or restored["actual_per_task_transitions"]
        != int(getattr(segment, "end_transitions")) // 4
        or restored["curriculum"] != curriculum.state_dict()
        or not _semantic_equal(restored["evaluation_history"], expected_history)
        or restored["next_crossing"] != pre_restored["next_crossing"]
    ):
        raise ValueError("mainrunner post-evaluation checkpoint chain is invalid")
    digest = _sha256(post_checkpoint)
    expected_latest = {
        "actual_global_transitions": int(getattr(segment, "end_transitions")),
        "checkpoint": str(post_checkpoint),
        "checkpoint_sha256": digest,
        "curriculum_stage": decision.stage,
        "iteration": int(getattr(segment, "end_iteration")),
        "logical_crossing": int(getattr(segment, "crossing").logical_transitions),
        "segment": int(getattr(segment, "segment_index")),
        "source_manifest_sha256": _source_manifest_digest(source_manifest),
    }
    segment_latest = _read_json(segment_dir / "latest.json")
    if segment_latest != expected_latest:
        raise ValueError("mainrunner segment latest binding is invalid")
    final_acceptance: dict[str, bool] | None = None
    if (
        int(getattr(segment, "end_iteration")) == 2442
        and restored["actual_global_transitions"] == 20004864
        and restored["actual_per_task_transitions"] == 5001216
        and restored["optimizer_step"] == 58608
        and decision.stage == "C3"
    ):
        final_acceptance = validate_learning_final_acceptance(
            {
                "global_transitions": restored["actual_global_transitions"],
                "iteration": int(getattr(segment, "end_iteration")),
                "optimizer_steps": restored["optimizer_step"],
                "stage": decision.stage,
                "tasks": evaluation["curriculum_task_metrics"],
            },
            config=acceptance,
        )
    summary = _read_json(segment_dir / "segment_summary.json")
    expected_summary = {
        "checkpoint": {"path": str(post_checkpoint), "sha256": digest},
        "curriculum_decision": {
            "crossing": decision.crossing,
            "promoted": decision.promoted,
            "rolled_back": decision.rolled_back,
            "stage": decision.stage,
        },
        "evaluation_summary": {
            "path": str(segment_dir / "evaluation" / "evaluation_summary.json"),
            "sha256": _sha256(segment_dir / "evaluation" / "evaluation_summary.json"),
        },
        "evidence_manifests": evaluation["evidence_manifests"],
        "final_acceptance": final_acceptance,
        "iteration": int(getattr(segment, "end_iteration")),
        "logical_crossing": int(getattr(segment, "crossing").logical_transitions),
        "status": "final_acceptance_complete"
        if final_acceptance is not None
        else "segment_complete",
        "task_metrics": evaluation["curriculum_task_metrics"],
        "source_manifest_sha256": _source_manifest_digest(source_manifest),
    }
    if summary != expected_summary:
        raise ValueError("mainrunner segment summary binding is invalid")
    return expected_latest


def _validate_mainrunner_completed_segment(
    *,
    run_dir: Path,
    segment: object,
    resume_checkpoint: Path,
    source_rebind: Path,
    config: object,
    roster: object,
    acceptance: object,
    source_manifest: Mapping[str, str],
) -> dict[str, object]:
    segment_dir = run_dir / f"segment_{int(getattr(segment, 'segment_index')):04d}"
    if not segment_dir.is_dir():
        raise FileNotFoundError("mainrunner completed segment directory is missing")
    pre_checkpoint, pre_restored = _validate_mainrunner_completed_train(
        segment_dir=segment_dir,
        segment=segment,
        resume_checkpoint=resume_checkpoint,
        source_rebind=source_rebind,
        config=config,
        roster=roster,
        acceptance=acceptance,
        source_manifest=source_manifest,
    )
    stage = str(pre_restored["curriculum"]["stage"])
    evaluation = _validate_mainrunner_completed_evaluation(
        segment_dir=segment_dir,
        segment=segment,
        pre_checkpoint=pre_checkpoint,
        stage=stage,
        source_rebind=source_rebind,
        roster=roster,
        source_manifest=source_manifest,
    )
    return _validate_mainrunner_completed_finalize(
        run_dir=run_dir,
        segment_dir=segment_dir,
        segment=segment,
        pre_checkpoint=pre_checkpoint,
        pre_restored=pre_restored,
        evaluation=evaluation,
        source_rebind=source_rebind,
        config=config,
        roster=roster,
        acceptance=acceptance,
        source_manifest=source_manifest,
    )


def _validate_mainrunner_completed_segments(
    run_dir: Path,
    *,
    current_iteration: int,
    target_iteration: int,
    source_rebind: Path,
    config: object,
    roster: object,
    acceptance: object,
) -> None:
    from somaforce_cross.learning.joint_runner import plan_learning_segments

    segments = plan_learning_segments(total_iterations=target_iteration)
    endpoints = {segment.end_iteration for segment in segments}
    if current_iteration not in endpoints:
        raise ValueError("mainrunner resume iteration is not a segment boundary")
    source_manifest = _current_mainrunner_source_manifest()
    rebind = _validate_mainrunner_rebind_record(
        source_rebind, config=config, roster=roster, acceptance=acceptance
    )
    bootstrap_checkpoint = Path(str(rebind["checkpoint"]["output_path"]))
    if (run_dir / "segment_0000").exists():
        raise ValueError("mainrunner must not materialize pilot bootstrap segment_0000")
    completed = [
        segment
        for segment in segments
        if segment.segment_index > 0 and segment.end_iteration <= current_iteration
    ]
    expected_dirs = {f"segment_{segment.segment_index:04d}" for segment in completed}
    observed_dirs = (
        {path.name for path in run_dir.glob("segment_*")} if run_dir.is_dir() else set()
    )
    if observed_dirs != expected_dirs:
        raise ValueError("mainrunner completed segment directory set is invalid")
    latest_records: list[dict[str, object]] = []
    previous_checkpoint = bootstrap_checkpoint
    for index, segment in enumerate(completed):
        latest_records.append(
            _validate_mainrunner_completed_segment(
                run_dir=run_dir,
                segment=segment,
                resume_checkpoint=previous_checkpoint,
                source_rebind=source_rebind,
                config=config,
                roster=roster,
                acceptance=acceptance,
                source_manifest=source_manifest,
            )
        )
        previous_checkpoint = Path(str(latest_records[index]["checkpoint"]))
    root_latest = run_dir / "latest.json"
    history_dir = run_dir / "latest_history"
    expected_history = {
        f"segment_{record['segment']:04d}.json": record
        for record in latest_records[:-1]
    }
    observed_history = (
        {path.name for path in history_dir.glob("segment_*.json") if path.is_file()}
        if history_dir.is_dir()
        else set()
    )
    if observed_history != set(expected_history):
        raise ValueError("mainrunner latest history set is invalid")
    for name, expected in expected_history.items():
        if _read_json(history_dir / name) != expected:
            raise ValueError("mainrunner latest history binding is invalid")
    if not latest_records:
        if root_latest.exists() or history_dir.exists():
            raise ValueError(
                "mainrunner bootstrap cannot fabricate local latest evidence"
            )
        return
    if _read_json(root_latest) != latest_records[-1]:
        raise ValueError("mainrunner root latest binding is invalid")


def _write_mainrunner_root_latest(run_dir: Path, payload: Mapping[str, object]) -> None:
    latest = run_dir / "latest.json"
    if latest.exists():
        previous = _read_json(latest)
        if previous is None or not isinstance(previous.get("segment"), int):
            raise ValueError("existing mainrunner latest pointer is invalid")
        if previous["segment"] >= payload["segment"]:
            raise FileExistsError(
                "mainrunner latest refuses to move backwards or overwrite"
            )
        history = run_dir / "latest_history" / f"segment_{previous['segment']:04d}.json"
        if history.exists():
            raise FileExistsError("mainrunner latest history already exists")
        _atomic_json(history, previous)
    _atomic_json(latest, payload)


def _run_mainrunner_finalize(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        JointCurriculum,
        atomic_learning_checkpoint,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
        validate_learning_final_acceptance,
    )

    if args.segment_index < 0 or args.iteration <= 0:
        raise ValueError("mainrunner finalize segment or iteration is invalid")
    post_checkpoint = args.segment_dir / "post_evaluation.pt"
    segment_latest = args.segment_dir / "latest.json"
    segment_summary = args.segment_dir / "segment_summary.json"
    if post_checkpoint.exists() or segment_latest.exists() or segment_summary.exists():
        raise FileExistsError(
            "mainrunner finalize refuses to overwrite segment evidence"
        )
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_mainrunner_rebind_record(
        args.source_rebind, config=config, roster=roster, acceptance=acceptance
    )
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        args.pre_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_mainrunner_source_manifest(),
        expected_iteration=args.iteration,
        device="cpu",
    )
    if restored["pre_evaluation"] is not True:
        raise ValueError("mainrunner finalize requires a pre-evaluation checkpoint")
    evaluation_value = _read_json(args.evaluation_summary)
    if evaluation_value is None:
        raise FileNotFoundError("mainrunner evaluation summary is missing")
    evaluation = _validate_mainrunner_evaluation_summary(
        evaluation_value,
        iteration=args.iteration,
        stage=str(restored["curriculum"]["stage"]),
        checkpoint=args.pre_checkpoint,
        source_manifest=_current_mainrunner_source_manifest(),
        source_rebind=args.source_rebind,
    )
    expected_crossing = restored["next_crossing"]
    if (
        not isinstance(expected_crossing, Mapping)
        or expected_crossing.get("logical_transitions") != args.logical_crossing
        or expected_crossing.get("target_iteration") != args.iteration
        or expected_crossing.get("actual_transitions")
        != restored["actual_global_transitions"]
    ):
        raise ValueError("mainrunner finalize checkpoint crossing is invalid")
    curriculum = JointCurriculum()
    curriculum.load_state_dict(restored["curriculum"])
    decision = curriculum.evaluate_window(
        crossing=args.logical_crossing,
        task_metrics=evaluation["curriculum_task_metrics"],
        actual_transitions=int(restored["actual_global_transitions"]),
    )
    payload = dict(restored)
    payload["pre_evaluation"] = False
    payload["curriculum"] = curriculum.state_dict()
    payload["evaluation_history"] = [
        *restored["evaluation_history"],
        dict(evaluation),
    ]
    protected = set(restored) - {"pre_evaluation", "curriculum", "evaluation_history"}
    if not all(_semantic_equal(payload[name], restored[name]) for name in protected):
        raise AssertionError("mainrunner finalization altered protected training state")
    digest = atomic_learning_checkpoint(post_checkpoint, payload)
    post_policy = ResidualActorCritic()
    post_optimizer = torch.optim.Adam(post_policy.parameters(), lr=3.0e-4)
    post_restored = restore_learning_checkpoint(
        post_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=post_policy,
        optimizer=post_optimizer,
        source_manifest=_current_mainrunner_source_manifest(),
        expected_iteration=args.iteration,
        device="cpu",
    )
    if not _semantic_equal(post_restored, payload):
        raise AssertionError(
            "mainrunner post-evaluation checkpoint did not strictly restore"
        )
    latest_payload = {
        "actual_global_transitions": payload["actual_global_transitions"],
        "checkpoint": str(post_checkpoint),
        "checkpoint_sha256": digest,
        "curriculum_stage": decision.stage,
        "iteration": args.iteration,
        "logical_crossing": args.logical_crossing,
        "segment": args.segment_index,
        "source_manifest_sha256": _source_manifest_digest(
            _current_mainrunner_source_manifest()
        ),
    }
    _atomic_json(segment_latest, latest_payload)
    final_acceptance: dict[str, bool] | None = None
    if (
        args.iteration == 2442
        and payload["actual_global_transitions"] == 20004864
        and payload["actual_per_task_transitions"] == 5001216
        and payload["optimizer_step"] == 58608
        and decision.stage == "C3"
    ):
        final_acceptance = validate_learning_final_acceptance(
            {
                "global_transitions": payload["actual_global_transitions"],
                "iteration": args.iteration,
                "optimizer_steps": payload["optimizer_step"],
                "stage": decision.stage,
                "tasks": evaluation["curriculum_task_metrics"],
            },
            config=acceptance,
        )
    _atomic_json(
        segment_summary,
        {
            "checkpoint": {"path": str(post_checkpoint), "sha256": digest},
            "curriculum_decision": {
                "crossing": decision.crossing,
                "promoted": decision.promoted,
                "rolled_back": decision.rolled_back,
                "stage": decision.stage,
            },
            "evaluation_summary": {
                "path": str(args.evaluation_summary),
                "sha256": _sha256(args.evaluation_summary),
            },
            "evidence_manifests": evaluation["evidence_manifests"],
            "final_acceptance": final_acceptance,
            "iteration": args.iteration,
            "logical_crossing": args.logical_crossing,
            "status": "final_acceptance_complete"
            if final_acceptance is not None
            else "segment_complete",
            "task_metrics": evaluation["curriculum_task_metrics"],
            "source_manifest_sha256": _source_manifest_digest(
                _current_mainrunner_source_manifest()
            ),
        },
    )
    _write_mainrunner_root_latest(args.run_dir, latest_payload)
    print(
        "PHASE6_MAINRUNNER_FINALIZED=" + json.dumps(latest_payload, sort_keys=True),
        flush=True,
    )
    return 0


def _run_modeprocess_recovery_evaluate(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    checkpoint = recovery / "pre_evaluation_modeprocess_rebound.pt"
    record = recovery / "source_rebind_modeprocess.json"
    attempt = recovery / (
        "evaluation_modeprocess_smoke_0006"
        if args.smoke
        else "evaluation_modeprocess_attempt_0006"
    )
    if (
        args.output_dir.resolve() != recovery.resolve()
        or args.rebound_checkpoint.resolve() != checkpoint.resolve()
        or args.source_rebind.resolve() != record.resolve()
    ):
        raise ValueError("modeprocess recovery paths are invalid")
    if attempt.exists() or (
        not args.smoke
        and any(
            (recovery / name).exists() for name in ("post_evaluation.pt", "latest.json")
        )
    ):
        raise FileExistsError("modeprocess recovery refuses to overwrite evidence")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_modeprocess_rebind_record(
        record_path=record,
        rebound_checkpoint=checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    if args.smoke:
        num_envs, stage_quota, nominal_quota = 2, 2, 2
        stall_timeout_s, worker_timeout_s, stage_timeout_s = 180, 600, 900
    else:
        num_envs, stage_quota, nominal_quota = 64, 256, 128
        stall_timeout_s, worker_timeout_s, stage_timeout_s = 300, 12600, 13200
    if not stall_timeout_s < worker_timeout_s < stage_timeout_s:
        raise ValueError("modeprocess timeout nesting is invalid")
    for paired_mode in ("residual", "scaffold_only"):
        mode_dir = attempt / paired_mode
        command = [
            ISAAC_PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(Path(__file__).resolve()),
            "--mode",
            "rank-wrapper",
            "--worker-mode",
            "evaluate",
            "--output-dir",
            str(mode_dir),
            "--profile",
            "production_segment",
            "--timeout-s",
            str(worker_timeout_s),
            "--progress-stall-timeout-s",
            str(stall_timeout_s),
            "--paired-mode",
            paired_mode,
            "--segment-start",
            "31",
            "--segment-end",
            "32",
            "--resume",
            str(checkpoint),
            "--source-rebind",
            str(record),
            "--paired-num-envs",
            str(num_envs),
            "--paired-stage-quota",
            str(stage_quota),
            "--paired-nominal-quota",
            str(nominal_quota),
            "--acceptance-config",
            str(args.acceptance_config),
            "--phase6-config",
            str(args.phase6_config),
            "--roster",
            str(args.roster),
        ]
        _run_stage_command(
            command,
            log_path=attempt / f"paired_evaluation_modeprocess_{paired_mode}.log",
            timeout_s=stage_timeout_s,
            stage=f"modeprocess-{paired_mode}",
        )
        _aggregate_modeprocess_progress(attempt)
    summary = _merge_modeprocess_evaluation(
        attempt=attempt,
        roster=roster,
        stage_quota=stage_quota,
        nominal_quota=nominal_quota,
    )
    if args.smoke:
        return 0
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "finalize",
            "--output-dir",
            str(recovery),
            "--pre-checkpoint",
            str(checkpoint),
            "--evaluation-summary",
            str(attempt / "evaluation_summary.json"),
            "--iteration",
            "31",
        ],
        log_path=attempt / "finalize_modeprocess.log",
        timeout_s=stage_timeout_s,
        stage="modeprocess-finalize",
    )
    if summary["status"] != "ok":
        raise AssertionError("modeprocess summary is not complete")
    return 0


def _run_warningpolicy_recovery_evaluate(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    checkpoint = recovery / "pre_evaluation_warningpolicy_rebound.pt"
    record = recovery / "source_rebind_warningpolicy.json"
    attempt = recovery / (
        "evaluation_warningpolicy_smoke_0007"
        if args.smoke
        else "evaluation_warningpolicy_attempt_0007"
    )
    if (
        args.output_dir.resolve() != recovery.resolve()
        or args.rebound_checkpoint.resolve() != checkpoint.resolve()
        or args.source_rebind.resolve() != record.resolve()
    ):
        raise ValueError("warningpolicy recovery paths are invalid")
    if attempt.exists() or (
        not args.smoke
        and any(
            (recovery / name).exists() for name in ("post_evaluation.pt", "latest.json")
        )
    ):
        raise FileExistsError("warningpolicy recovery refuses to overwrite evidence")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_warningpolicy_rebind_record(
        record_path=record,
        rebound_checkpoint=checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    if args.smoke:
        num_envs, stage_quota, nominal_quota = 2, 2, 2
        stall_timeout_s, worker_timeout_s, stage_timeout_s = 180, 600, 900
    else:
        num_envs, stage_quota, nominal_quota = 64, 256, 128
        stall_timeout_s, worker_timeout_s, stage_timeout_s = 300, 12600, 13200
    if not stall_timeout_s < worker_timeout_s < stage_timeout_s:
        raise ValueError("warningpolicy timeout nesting is invalid")
    for paired_mode in ("residual", "scaffold_only"):
        mode_dir = attempt / paired_mode
        command = [
            ISAAC_PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(Path(__file__).resolve()),
            "--mode",
            "rank-wrapper",
            "--worker-mode",
            "evaluate",
            "--output-dir",
            str(mode_dir),
            "--profile",
            "production_segment",
            "--timeout-s",
            str(worker_timeout_s),
            "--progress-stall-timeout-s",
            str(stall_timeout_s),
            "--paired-mode",
            paired_mode,
            "--segment-start",
            "31",
            "--segment-end",
            "32",
            "--resume",
            str(checkpoint),
            "--source-rebind",
            str(record),
            "--paired-num-envs",
            str(num_envs),
            "--paired-stage-quota",
            str(stage_quota),
            "--paired-nominal-quota",
            str(nominal_quota),
            "--acceptance-config",
            str(args.acceptance_config),
            "--phase6-config",
            str(args.phase6_config),
            "--roster",
            str(args.roster),
        ]
        _run_stage_command(
            command,
            log_path=attempt / f"paired_evaluation_warningpolicy_{paired_mode}.log",
            timeout_s=stage_timeout_s,
            stage=f"warningpolicy-{paired_mode}",
        )
        _aggregate_modeprocess_progress(attempt)
    summary = _merge_modeprocess_evaluation(
        attempt=attempt,
        roster=roster,
        stage_quota=stage_quota,
        nominal_quota=nominal_quota,
    )
    if args.smoke:
        return 0
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "finalize",
            "--output-dir",
            str(recovery),
            "--pre-checkpoint",
            str(checkpoint),
            "--evaluation-summary",
            str(attempt / "evaluation_summary.json"),
            "--iteration",
            "31",
        ],
        log_path=attempt / "finalize_warningpolicy.log",
        timeout_s=stage_timeout_s,
        stage="warningpolicy-finalize",
    )
    if summary["status"] != "ok":
        raise AssertionError("warningpolicy summary is not complete")
    return 0


def _run_episodeorder_recovery_evaluate(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    checkpoint = recovery / "pre_evaluation_episodeorder_rebound.pt"
    record = recovery / "source_rebind_episodeorder.json"
    attempt = recovery / (
        "evaluation_episodeorder_smoke_0005"
        if args.smoke
        else "evaluation_episodeorder_attempt_0005"
    )
    if (
        args.evaluation_timeout_s <= 0
        or args.output_dir.resolve() != recovery.resolve()
        or args.rebound_checkpoint.resolve() != checkpoint.resolve()
        or args.source_rebind.resolve() != record.resolve()
    ):
        raise ValueError("episodeorder recovery paths or timeout are invalid")
    if attempt.exists() or (
        not args.smoke
        and any(
            (recovery / name).exists() for name in ("post_evaluation.pt", "latest.json")
        )
    ):
        raise FileExistsError("episodeorder recovery refuses to overwrite evidence")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_episodeorder_rebind_record(
        record_path=record,
        rebound_checkpoint=checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    num_envs, stage_quota, nominal_quota = (2, 2, 2) if args.smoke else (64, 256, 128)
    evaluation_command = [
        ISAAC_PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        str(Path(__file__).resolve()),
        "--mode",
        "rank-wrapper",
        "--worker-mode",
        "evaluate",
        "--output-dir",
        str(attempt),
        "--profile",
        "production_segment",
        "--timeout-s",
        str(args.evaluation_timeout_s),
        "--segment-start",
        "31",
        "--segment-end",
        "32",
        "--resume",
        str(checkpoint),
        "--source-rebind",
        str(record),
        "--paired-num-envs",
        str(num_envs),
        "--paired-stage-quota",
        str(stage_quota),
        "--paired-nominal-quota",
        str(nominal_quota),
        "--acceptance-config",
        str(args.acceptance_config),
        "--phase6-config",
        str(args.phase6_config),
        "--roster",
        str(args.roster),
    ]
    _run_stage_command(
        evaluation_command,
        log_path=attempt / "paired_evaluation_episodeorder.log",
        timeout_s=args.evaluation_timeout_s,
        stage="paired-evaluate-episodeorder-smoke"
        if args.smoke
        else "paired-evaluate-episodeorder",
    )
    evaluation_summary = attempt / "evaluation_summary.json"
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "production-summarize",
            "--output-dir",
            str(attempt),
            "--iteration",
            "31",
            "--evaluation",
            "--paired-stage-quota",
            str(stage_quota),
            "--paired-nominal-quota",
            str(nominal_quota),
        ],
        log_path=attempt / "evaluation_summary_episodeorder.log",
        timeout_s=args.evaluation_timeout_s,
        stage="evaluation-summary-episodeorder-smoke"
        if args.smoke
        else "evaluation-summary-episodeorder",
    )
    if args.smoke:
        return 0
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "finalize",
            "--output-dir",
            str(recovery),
            "--pre-checkpoint",
            str(checkpoint),
            "--evaluation-summary",
            str(evaluation_summary),
            "--iteration",
            "31",
        ],
        log_path=attempt / "finalize_episodeorder.log",
        timeout_s=args.evaluation_timeout_s,
        stage="finalize-checkpoint-episodeorder",
    )
    return 0


def _run_rankseed_recovery_evaluate(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    checkpoint = recovery / "pre_evaluation_rankseed_rebound.pt"
    record = recovery / "source_rebind_rankseed.json"
    attempt = recovery / (
        "evaluation_rankseed_smoke_0004"
        if args.smoke
        else "evaluation_rankseed_attempt_0004"
    )
    if (
        args.evaluation_timeout_s <= 0
        or args.output_dir.resolve() != recovery.resolve()
        or args.rebound_checkpoint.resolve() != checkpoint.resolve()
        or args.source_rebind.resolve() != record.resolve()
    ):
        raise ValueError("rankseed recovery paths or timeout are invalid")
    if attempt.exists() or (
        not args.smoke
        and any(
            (recovery / name).exists() for name in ("post_evaluation.pt", "latest.json")
        )
    ):
        raise FileExistsError("rankseed recovery refuses to overwrite evidence")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_rankseed_rebind_record(
        record_path=record,
        rebound_checkpoint=checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    num_envs, stage_quota, nominal_quota = (2, 2, 2) if args.smoke else (64, 256, 128)
    evaluation_command = [
        ISAAC_PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        str(Path(__file__).resolve()),
        "--mode",
        "rank-wrapper",
        "--worker-mode",
        "evaluate",
        "--output-dir",
        str(attempt),
        "--profile",
        "production_segment",
        "--timeout-s",
        str(args.evaluation_timeout_s),
        "--segment-start",
        "31",
        "--segment-end",
        "32",
        "--resume",
        str(checkpoint),
        "--source-rebind",
        str(record),
        "--paired-num-envs",
        str(num_envs),
        "--paired-stage-quota",
        str(stage_quota),
        "--paired-nominal-quota",
        str(nominal_quota),
        "--acceptance-config",
        str(args.acceptance_config),
        "--phase6-config",
        str(args.phase6_config),
        "--roster",
        str(args.roster),
    ]
    _run_stage_command(
        evaluation_command,
        log_path=attempt / "paired_evaluation_rankseed.log",
        timeout_s=args.evaluation_timeout_s,
        stage="paired-evaluate-rankseed-smoke"
        if args.smoke
        else "paired-evaluate-rankseed",
    )
    evaluation_summary = attempt / "evaluation_summary.json"
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "production-summarize",
            "--output-dir",
            str(attempt),
            "--iteration",
            "31",
            "--evaluation",
            "--paired-stage-quota",
            str(stage_quota),
            "--paired-nominal-quota",
            str(nominal_quota),
        ],
        log_path=attempt / "evaluation_summary_rankseed.log",
        timeout_s=args.evaluation_timeout_s,
        stage="evaluation-summary-rankseed-smoke"
        if args.smoke
        else "evaluation-summary-rankseed",
    )
    if args.smoke:
        return 0
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "finalize",
            "--output-dir",
            str(recovery),
            "--pre-checkpoint",
            str(checkpoint),
            "--evaluation-summary",
            str(evaluation_summary),
            "--iteration",
            "31",
        ],
        log_path=attempt / "finalize_rankseed.log",
        timeout_s=args.evaluation_timeout_s,
        stage="finalize-checkpoint-rankseed",
    )
    return 0


def _run_pairedbatch_recovery_evaluate(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    checkpoint = recovery / "pre_evaluation_pairedbatch_rebound.pt"
    record = recovery / "source_rebind_pairedbatch.json"
    attempt_name = (
        "evaluation_pairedbatch_smoke_0003"
        if args.smoke
        else "evaluation_pairedbatch_attempt_0003"
    )
    attempt = recovery / attempt_name
    if (
        args.evaluation_timeout_s <= 0
        or args.output_dir.resolve() != recovery.resolve()
        or args.rebound_checkpoint.resolve() != checkpoint.resolve()
        or args.source_rebind.resolve() != record.resolve()
    ):
        raise ValueError("pairedbatch recovery paths or timeout are invalid")
    if attempt.exists() or (
        not args.smoke
        and any(
            (recovery / name).exists() for name in ("post_evaluation.pt", "latest.json")
        )
    ):
        raise FileExistsError("pairedbatch recovery refuses to overwrite evidence")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_pairedbatch_rebind_record(
        record_path=record,
        rebound_checkpoint=checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    num_envs, stage_quota, nominal_quota = (2, 2, 2) if args.smoke else (64, 256, 128)
    evaluation_command = [
        ISAAC_PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        str(Path(__file__).resolve()),
        "--mode",
        "rank-wrapper",
        "--worker-mode",
        "evaluate",
        "--output-dir",
        str(attempt),
        "--profile",
        "production_segment",
        "--timeout-s",
        str(args.evaluation_timeout_s),
        "--segment-start",
        "31",
        "--segment-end",
        "32",
        "--resume",
        str(checkpoint),
        "--source-rebind",
        str(record),
        "--paired-num-envs",
        str(num_envs),
        "--paired-stage-quota",
        str(stage_quota),
        "--paired-nominal-quota",
        str(nominal_quota),
        "--acceptance-config",
        str(args.acceptance_config),
        "--phase6-config",
        str(args.phase6_config),
        "--roster",
        str(args.roster),
    ]
    _run_stage_command(
        evaluation_command,
        log_path=attempt / "paired_evaluation_pairedbatch.log",
        timeout_s=args.evaluation_timeout_s,
        stage="paired-evaluate-pairedbatch-smoke"
        if args.smoke
        else "paired-evaluate-pairedbatch",
    )
    evaluation_summary = attempt / "evaluation_summary.json"
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "production-summarize",
            "--output-dir",
            str(attempt),
            "--iteration",
            "31",
            "--evaluation",
            "--paired-stage-quota",
            str(stage_quota),
            "--paired-nominal-quota",
            str(nominal_quota),
        ],
        log_path=attempt / "evaluation_summary_pairedbatch.log",
        timeout_s=args.evaluation_timeout_s,
        stage="evaluation-summary-pairedbatch-smoke"
        if args.smoke
        else "evaluation-summary-pairedbatch",
    )
    if args.smoke:
        return 0
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "finalize",
            "--output-dir",
            str(recovery),
            "--pre-checkpoint",
            str(checkpoint),
            "--evaluation-summary",
            str(evaluation_summary),
            "--iteration",
            "31",
        ],
        log_path=attempt / "finalize_pairedbatch.log",
        timeout_s=args.evaluation_timeout_s,
        stage="finalize-checkpoint-pairedbatch",
    )
    return 0


def _run_stepboundary_recovery_evaluate(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    attempt = recovery / "evaluation_stepboundary_attempt_0002"
    checkpoint = recovery / "pre_evaluation_stepboundary_rebound.pt"
    record = recovery / "source_rebind_stepboundary.json"
    if (
        args.evaluation_timeout_s <= 0
        or args.output_dir.resolve() != recovery.resolve()
        or args.rebound_checkpoint.resolve() != checkpoint.resolve()
        or args.source_rebind.resolve() != record.resolve()
    ):
        raise ValueError("stepboundary recovery paths or timeout are invalid")
    if (
        attempt.exists()
        or (recovery / "post_evaluation.pt").exists()
        or (recovery / "latest.json").exists()
    ):
        raise FileExistsError("stepboundary recovery refuses to overwrite evidence")
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_stepboundary_rebind_record(
        record_path=record,
        rebound_checkpoint=checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    evaluation_command = [
        ISAAC_PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        str(Path(__file__).resolve()),
        "--mode",
        "rank-wrapper",
        "--worker-mode",
        "evaluate",
        "--output-dir",
        str(attempt),
        "--profile",
        "production_segment",
        "--timeout-s",
        str(args.evaluation_timeout_s),
        "--segment-start",
        "31",
        "--segment-end",
        "32",
        "--resume",
        str(checkpoint),
        "--acceptance-config",
        str(args.acceptance_config),
        "--phase6-config",
        str(args.phase6_config),
        "--roster",
        str(args.roster),
    ]
    _run_stage_command(
        evaluation_command,
        log_path=attempt / "paired_evaluation_stepboundary.log",
        timeout_s=args.evaluation_timeout_s,
        stage="paired-evaluate-stepboundary",
    )
    evaluation_summary = attempt / "evaluation_summary.json"
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "production-summarize",
            "--output-dir",
            str(attempt),
            "--iteration",
            "31",
            "--evaluation",
        ],
        log_path=attempt / "evaluation_summary_stepboundary.log",
        timeout_s=args.evaluation_timeout_s,
        stage="evaluation-summary-stepboundary",
    )
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "finalize",
            "--output-dir",
            str(recovery),
            "--pre-checkpoint",
            str(checkpoint),
            "--evaluation-summary",
            str(evaluation_summary),
            "--iteration",
            "31",
        ],
        log_path=attempt / "finalize_stepboundary.log",
        timeout_s=args.evaluation_timeout_s,
        stage="finalize-checkpoint-stepboundary",
    )
    return 0


def _run_devicefix_recovery_evaluate(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    recovery = LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery"
    attempt = recovery / "evaluation_devicefix_attempt_0001"
    checkpoint = recovery / "pre_evaluation_devicefix_rebound.pt"
    record = recovery / "source_rebind_devicefix.json"
    if (
        args.evaluation_timeout_s <= 0
        or args.output_dir.resolve() != recovery.resolve()
        or args.rebound_checkpoint.resolve() != checkpoint.resolve()
        or args.source_rebind.resolve() != record.resolve()
    ):
        raise ValueError("devicefix recovery evaluation paths or timeout are invalid")
    if (
        attempt.exists()
        or (recovery / "post_evaluation.pt").exists()
        or (recovery / "latest.json").exists()
    ):
        raise FileExistsError(
            "devicefix recovery refuses to overwrite evaluation evidence"
        )
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_evaluation_devicefix_rebind_record(
        record_path=record,
        rebound_checkpoint=checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    evaluation_command = [
        ISAAC_PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        str(Path(__file__).resolve()),
        "--mode",
        "rank-wrapper",
        "--worker-mode",
        "evaluate",
        "--output-dir",
        str(attempt),
        "--profile",
        "production_segment",
        "--timeout-s",
        str(args.evaluation_timeout_s),
        "--segment-start",
        "31",
        "--segment-end",
        "32",
        "--resume",
        str(checkpoint),
        "--acceptance-config",
        str(args.acceptance_config),
        "--phase6-config",
        str(args.phase6_config),
        "--roster",
        str(args.roster),
    ]
    _run_stage_command(
        evaluation_command,
        log_path=attempt / "paired_evaluation_devicefix.log",
        timeout_s=args.evaluation_timeout_s,
        stage="paired-evaluate-devicefix",
    )
    evaluation_summary = attempt / "evaluation_summary.json"
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "production-summarize",
            "--output-dir",
            str(attempt),
            "--iteration",
            "31",
            "--evaluation",
        ],
        log_path=attempt / "evaluation_summary_devicefix.log",
        timeout_s=args.evaluation_timeout_s,
        stage="evaluation-summary-devicefix",
    )
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "finalize",
            "--output-dir",
            str(recovery),
            "--pre-checkpoint",
            str(checkpoint),
            "--evaluation-summary",
            str(evaluation_summary),
            "--iteration",
            "31",
        ],
        log_path=attempt / "finalize_devicefix.log",
        timeout_s=args.evaluation_timeout_s,
        stage="finalize-checkpoint-devicefix",
    )
    return 0


def _run_recovery_evaluate(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
    )

    if args.evaluation_timeout_s <= 0:
        raise ValueError("recovery evaluation timeout must be positive")
    if (
        args.rebound_checkpoint.resolve()
        != (
            LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent
            / "recovery/pre_evaluation_harness_rebound.pt"
        ).resolve()
        or args.source_rebind.resolve()
        != (
            LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent
            / "recovery/source_rebind.json"
        ).resolve()
        or args.output_dir.resolve()
        != (LEGACY_PRE_EVALUATION_CHECKPOINT.parent.parent / "recovery").resolve()
    ):
        raise ValueError(
            "recovery evaluation is restricted to the rebound pilot checkpoint"
        )
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    _validate_rebind_record(
        record_path=args.source_rebind,
        rebound_checkpoint=args.rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    if (args.output_dir / "latest.json").exists():
        raise FileExistsError(
            "recovery evaluation refuses to advance an existing latest.json"
        )
    evaluation_dir = args.output_dir / "evaluation"
    evaluation_command = [
        ISAAC_PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        str(Path(__file__).resolve()),
        "--mode",
        "rank-wrapper",
        "--worker-mode",
        "evaluate",
        "--output-dir",
        str(evaluation_dir),
        "--profile",
        "production_segment",
        "--timeout-s",
        str(args.evaluation_timeout_s),
        "--segment-start",
        "31",
        "--segment-end",
        "32",
        "--resume",
        str(args.rebound_checkpoint),
        "--acceptance-config",
        str(args.acceptance_config),
        "--phase6-config",
        str(args.phase6_config),
        "--roster",
        str(args.roster),
    ]
    _run_stage_command(
        evaluation_command,
        log_path=args.output_dir / "paired_evaluation.log",
        timeout_s=args.evaluation_timeout_s,
        stage="paired-evaluate",
    )
    evaluation_summary = evaluation_dir / "evaluation_summary.json"
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "production-summarize",
            "--output-dir",
            str(evaluation_dir),
            "--iteration",
            "31",
            "--evaluation",
        ],
        log_path=args.output_dir / "evaluation_summary.log",
        timeout_s=args.evaluation_timeout_s,
        stage="evaluation-summary",
    )
    _run_stage_command(
        [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--mode",
            "finalize",
            "--output-dir",
            str(args.output_dir),
            "--pre-checkpoint",
            str(args.rebound_checkpoint),
            "--evaluation-summary",
            str(evaluation_summary),
            "--iteration",
            "31",
        ],
        log_path=args.output_dir / "finalize.log",
        timeout_s=args.evaluation_timeout_s,
        stage="finalize-checkpoint",
    )
    return 0


def _run_stage_command(
    command: list[str], *, log_path: Path, timeout_s: int, stage: str
) -> None:
    if timeout_s <= 0:
        raise ValueError(f"{stage} timeout must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_path = log_path.with_suffix(".command.json")
    status_path = log_path.with_suffix(".status.json")
    if log_path.exists() or command_path.exists() or status_path.exists():
        raise FileExistsError(f"Phase 6 {stage} evidence already exists: {log_path}")
    _atomic_json(
        command_path, {"command": command, "stage": stage, "timeout_s": timeout_s}
    )
    started = time.monotonic()
    maximum_hwm: int | None = None
    timed_out = False
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                preexec_fn=os.setsid,
            )
            while process.poll() is None:
                observed = _proc_hwm_kib(process.pid)
                if observed is not None:
                    maximum_hwm = max(maximum_hwm or observed, observed)
                if time.monotonic() - started > timeout_s:
                    timed_out = True
                    _terminate_group(process)
                    break
                time.sleep(0.2)
            return_code = process.wait(timeout=35.0)
    except BaseException:
        if process is not None and process.poll() is None:
            _terminate_group(process)
        raise
    finally:
        return_code = None if process is None else process.returncode
        _atomic_json(
            status_path,
            {
                "elapsed_s": round(time.monotonic() - started, 3),
                "exit_code": return_code
                if return_code is None or return_code >= 0
                else None,
                "signal": _signal_name(return_code),
                "stage": stage,
                "timeout": timed_out,
                "vmhwm_kib": maximum_hwm,
            },
        )
    if timed_out:
        raise RuntimeError(f"Phase 6 {stage} timed out after {timeout_s}s")
    if return_code != 0:
        raise RuntimeError(
            f"Phase 6 {stage} failed with exit={return_code}: {log_path}"
        )


def _update_mainrunner_progress(
    path: Path,
    *,
    target_iteration: int,
    current_iteration: int,
    segment: int,
    stage: str,
    active_substage: str,
    active_mode: str | None,
    completed_rows: int,
    control_steps: int,
    eta_s: float | None,
    error_state: str | None,
) -> None:
    previous = _read_json(path)
    if previous is not None:
        if (
            previous.get("target_iteration") != target_iteration
            or previous.get("current_iteration", -1) > current_iteration
            or previous.get("segment", -1) > segment
            or previous.get("completed_rows", -1) > completed_rows
            or previous.get("control_steps", -1) > control_steps
        ):
            raise ValueError("mainrunner progress is not monotonic")
    _atomic_json(
        path,
        {
            "active_mode": active_mode,
            "active_substage": active_substage,
            "completed_rows": completed_rows,
            "control_steps": control_steps,
            "current_iteration": current_iteration,
            "error_state": error_state,
            "eta_s": eta_s,
            "last_update_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "segment": segment,
            "stage": stage,
            "target_iteration": target_iteration,
        },
    )


def _resolve_mainrunner_resume(
    resume: Path,
    *,
    config: object,
    roster: object,
    acceptance: object,
) -> tuple[Path, dict[str, object]]:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import restore_learning_checkpoint

    pointer = resume / "latest.json" if resume.is_dir() else resume
    checkpoint = pointer
    if pointer.suffix == ".json":
        latest = _read_json(pointer)
        required = {
            "actual_global_transitions",
            "checkpoint",
            "checkpoint_sha256",
            "curriculum_stage",
            "iteration",
            "logical_crossing",
            "segment",
            "source_manifest_sha256",
        }
        if latest is None or set(latest) != required:
            raise ValueError("mainrunner latest pointer schema is invalid")
        checkpoint = Path(str(latest["checkpoint"]))
        if (
            not checkpoint.is_file()
            or _sha256(checkpoint) != latest["checkpoint_sha256"]
            or latest["source_manifest_sha256"]
            != _source_manifest_digest(_current_mainrunner_source_manifest())
        ):
            raise ValueError("mainrunner latest checkpoint binding is invalid")
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=_current_mainrunner_source_manifest(),
        device="cpu",
    )
    if restored["pre_evaluation"] is not False:
        raise ValueError(
            "mainrunner resume requires a completed post-evaluation checkpoint"
        )
    return checkpoint, restored


def _run_production(args: argparse.Namespace) -> int:
    from somaforce_cross.learning.joint_runner import (
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        plan_learning_segments,
    )

    if args.max_segments <= 0:
        raise ValueError("mainrunner max_segments must be a positive integer")
    if args.profile == "pilot" and args.resume is not None:
        raise ValueError("pilot production run cannot resume")
    if args.profile == "main" and (args.resume is None or args.source_rebind is None):
        raise ValueError("main production requires resume checkpoint and source rebind")
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    config = load_phase6_config(args.phase6_config)
    roster = load_phase6_task_roster(args.roster, phase6_config=config)
    if args.profile == "main":
        assert args.source_rebind is not None
        _validate_mainrunner_rebind_record(
            args.source_rebind, config=config, roster=roster, acceptance=acceptance
        )
    profile = acceptance.payload["profiles"][args.profile]
    target_iteration = int(profile["total_iterations"])
    segments = plan_learning_segments(total_iterations=target_iteration)
    if args.resume is None:
        raise ValueError(
            "production bootstrap without a completed checkpoint is disabled"
        )
    resume, restored = _resolve_mainrunner_resume(
        args.resume, config=config, roster=roster, acceptance=acceptance
    )
    current_iteration = int(restored["iteration"])
    curriculum_state = restored["curriculum"]
    if not isinstance(curriculum_state, Mapping):
        raise ValueError("mainrunner resume curriculum is invalid")
    stage = str(curriculum_state["stage"])
    if current_iteration >= target_iteration:
        raise ValueError(
            "mainrunner resume is already at or beyond the target iteration"
        )
    if args.profile == "main":
        assert args.source_rebind is not None
        _validate_mainrunner_completed_segments(
            args.output_dir,
            current_iteration=current_iteration,
            target_iteration=target_iteration,
            source_rebind=args.source_rebind,
            config=config,
            roster=roster,
            acceptance=acceptance,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    previous_progress = _read_json(progress_path)
    completed_rows = (
        int(previous_progress.get("completed_rows", 0)) if previous_progress else 0
    )
    control_steps = (
        int(previous_progress.get("control_steps", 0)) if previous_progress else 0
    )
    active_segment = max(
        (
            segment.segment_index
            for segment in segments
            if segment.end_iteration <= current_iteration
        ),
        default=-1,
    )
    _update_mainrunner_progress(
        progress_path,
        target_iteration=target_iteration,
        current_iteration=current_iteration,
        segment=active_segment,
        stage=stage,
        active_substage="ready",
        active_mode=None,
        completed_rows=completed_rows,
        control_steps=control_steps,
        eta_s=None,
        error_state=None,
    )
    completed_segments = 0
    print(f"PHASE6_TRAIN_TIMEOUT_S={args.train_timeout_s}", flush=True)
    print(f"PHASE6_EVALUATION_TIMEOUT_S={args.evaluation_timeout_s}", flush=True)
    try:
        for segment in segments:
            if segment.end_iteration <= current_iteration:
                continue
            if segment.start_iteration != current_iteration:
                raise ValueError(
                    "mainrunner resume does not align with a segment boundary"
                )
            if completed_segments >= args.max_segments:
                break
            segment_dir = args.output_dir / f"segment_{segment.segment_index:04d}"
            if segment_dir.exists():
                raise FileExistsError(
                    "mainrunner refuses to overwrite an existing segment"
                )
            segment_dir.mkdir(parents=True)
            active_segment = segment.segment_index
            _update_mainrunner_progress(
                progress_path,
                target_iteration=target_iteration,
                current_iteration=current_iteration,
                segment=segment.segment_index,
                stage=stage,
                active_substage="train",
                active_mode=None,
                completed_rows=completed_rows,
                control_steps=control_steps,
                eta_s=None,
                error_state=None,
            )
            train_dir = segment_dir / "train"
            train_command = [
                ISAAC_PYTHON,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=4",
                str(Path(__file__).resolve()),
                "--mode",
                "rank-wrapper",
                "--worker-mode",
                "train-segment",
                "--output-dir",
                str(train_dir),
                "--profile",
                "production_segment",
                "--timeout-s",
                str(args.train_timeout_s),
                "--segment-start",
                str(segment.start_iteration),
                "--segment-end",
                str(segment.end_iteration),
                "--cycle-index",
                str(segment.segment_index),
                "--window-index",
                "0",
                "--resume",
                str(resume),
                "--source-rebind",
                str(args.source_rebind),
                "--acceptance-config",
                str(args.acceptance_config),
                "--phase6-config",
                str(args.phase6_config),
                "--roster",
                str(args.roster),
            ]
            _run_stage_command(
                train_command,
                log_path=segment_dir / "train.log",
                timeout_s=args.train_timeout_s,
                stage="train-segment",
            )
            _run_stage_command(
                [
                    ISAAC_PYTHON,
                    str(Path(__file__).resolve()),
                    "--mode",
                    "production-summarize",
                    "--output-dir",
                    str(train_dir),
                    "--iteration",
                    str(segment.end_iteration),
                    "--source-rebind",
                    str(args.source_rebind),
                ],
                log_path=segment_dir / "summary.log",
                timeout_s=args.train_timeout_s,
                stage="summarize",
            )
            evaluation_dir = segment_dir / "evaluation"
            worker_timeout = max(2, args.evaluation_timeout_s - 300)
            stall_timeout = min(300, worker_timeout - 1)
            for paired_mode in ("residual", "scaffold_only"):
                _update_mainrunner_progress(
                    progress_path,
                    target_iteration=target_iteration,
                    current_iteration=current_iteration,
                    segment=segment.segment_index,
                    stage=stage,
                    active_substage="evaluation",
                    active_mode=paired_mode,
                    completed_rows=completed_rows,
                    control_steps=control_steps,
                    eta_s=None,
                    error_state=None,
                )
                _run_stage_command(
                    [
                        ISAAC_PYTHON,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nnodes=1",
                        "--nproc_per_node=4",
                        str(Path(__file__).resolve()),
                        "--mode",
                        "rank-wrapper",
                        "--worker-mode",
                        "evaluate",
                        "--output-dir",
                        str(evaluation_dir / paired_mode),
                        "--profile",
                        "production_segment",
                        "--timeout-s",
                        str(worker_timeout),
                        "--progress-stall-timeout-s",
                        str(stall_timeout),
                        "--paired-mode",
                        paired_mode,
                        "--segment-start",
                        str(segment.end_iteration),
                        "--segment-end",
                        str(segment.end_iteration + 1),
                        "--resume",
                        str(train_dir / "pre_evaluation.pt"),
                        "--source-rebind",
                        str(args.source_rebind),
                        "--acceptance-config",
                        str(args.acceptance_config),
                        "--phase6-config",
                        str(args.phase6_config),
                        "--roster",
                        str(args.roster),
                    ],
                    log_path=segment_dir / f"evaluation_{paired_mode}.log",
                    timeout_s=args.evaluation_timeout_s,
                    stage=f"paired-evaluate-{paired_mode}",
                )
            evaluation_summary = evaluation_dir / "evaluation_summary.json"
            _run_stage_command(
                [
                    ISAAC_PYTHON,
                    str(Path(__file__).resolve()),
                    "--mode",
                    "mainrunner-evaluation-merge",
                    "--evaluation-dir",
                    str(evaluation_dir),
                    "--pre-checkpoint",
                    str(train_dir / "pre_evaluation.pt"),
                    "--iteration",
                    str(segment.end_iteration),
                    "--stage",
                    stage,
                    "--source-rebind",
                    str(args.source_rebind),
                ],
                log_path=segment_dir / "evaluation-summary.log",
                timeout_s=args.evaluation_timeout_s,
                stage="evaluation-summary",
            )
            _run_stage_command(
                [
                    ISAAC_PYTHON,
                    str(Path(__file__).resolve()),
                    "--mode",
                    "mainrunner-finalize",
                    "--run-dir",
                    str(args.output_dir),
                    "--segment-dir",
                    str(segment_dir),
                    "--pre-checkpoint",
                    str(train_dir / "pre_evaluation.pt"),
                    "--evaluation-summary",
                    str(evaluation_summary),
                    "--iteration",
                    str(segment.end_iteration),
                    "--segment-index",
                    str(segment.segment_index),
                    "--logical-crossing",
                    str(segment.crossing.logical_transitions),
                    "--source-rebind",
                    str(args.source_rebind),
                ],
                log_path=segment_dir / "finalize.log",
                timeout_s=args.evaluation_timeout_s,
                stage="finalize-checkpoint",
            )
            resume, restored = _resolve_mainrunner_resume(
                segment_dir / "post_evaluation.pt",
                config=config,
                roster=roster,
                acceptance=acceptance,
            )
            current_iteration = int(restored["iteration"])
            stage = str(restored["curriculum"]["stage"])
            completed_segments += 1
            completed_rows += 4 * 2 * (256 + 128)
            _update_mainrunner_progress(
                progress_path,
                target_iteration=target_iteration,
                current_iteration=current_iteration,
                segment=segment.segment_index,
                stage=stage,
                active_substage="segment_complete",
                active_mode=None,
                completed_rows=completed_rows,
                control_steps=control_steps,
                eta_s=None,
                error_state=None,
            )
    except BaseException as exc:
        _update_mainrunner_progress(
            progress_path,
            target_iteration=target_iteration,
            current_iteration=current_iteration,
            segment=active_segment,
            stage=stage,
            active_substage="error",
            active_mode=None,
            completed_rows=completed_rows,
            control_steps=control_steps,
            eta_s=None,
            error_state=f"{type(exc).__name__}: {exc}",
        )
        raise
    return 0


def _build_environment(args: argparse.Namespace, *, task: Any, seed: int) -> Any:
    from somaforce_cross.envs.residual_env import SomaForceResidualEnv
    from somaforce_cross.envs.residual_env_cfg import SomaForceResidualEnvCfg
    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
    from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg

    spec = get_hdmi_task_spec(task.task)
    cfg = SomaForceResidualEnvCfg()
    cfg.seed = seed
    cfg.sim.device = args.device
    cfg.artifact_dir = str(task.artifact_path)
    cfg.scene = make_scene_cfg(
        task.artifact_path,
        args.num_envs,
        spec,
        rigid_object_mass=spec.nominal_object_mass or 8.0,
    )
    cfg.smoke_profile.task = task.task
    cfg.smoke_profile.runtime_mode = getattr(args, "runtime_mode", "residual")
    cfg.smoke_profile.scaffold_stage = getattr(args, "stage", "C1")
    cfg.smoke_profile.numeric_contract_path = str(
        REPO_ROOT / "configs/phase4b5_numeric_contract.json"
    )
    cfg.smoke_profile.episode_length_steps = int(
        {
            "push_door_hand": 508,
            "push_box": 792,
            "move_suitcase": 472,
            "move_largebox": 199,
        }[task.task]
    )
    cfg.__post_init__()
    environment = SomaForceResidualEnv(cfg)
    if environment.is_residual:
        environment.set_curriculum_stage(args.stage)
    return environment


def _new_algorithm(config: Any, *, device: torch.device, world_size: int) -> Any:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import GradientAverager
    from somaforce_cross.learning.runner import _load_rsl_storage_class
    from somaforce_cross.learning.semantic_ppo import SemanticPPO

    ppo = config.payload["ppo"]
    policy = ResidualActorCritic(init_noise_std=float(ppo["init_noise_std"]))
    algorithm = SemanticPPO(
        policy,
        storage_class=_load_rsl_storage_class(),
        num_learning_epochs=int(ppo["num_learning_epochs"]),
        num_mini_batches=int(ppo["num_mini_batches"]),
        clip_param=float(ppo["clip_param"]),
        gamma=float(ppo["gamma"]),
        lam=float(ppo["lam"]),
        value_loss_coef=float(ppo["value_loss_coef"]),
        entropy_coef=float(ppo["entropy_coef"]),
        learning_rate=float(ppo["learning_rate"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
        use_clipped_value_loss=bool(ppo["use_clipped_value_loss"]),
        schedule=str(ppo["schedule"]),
        desired_kl=ppo["desired_kl"],
        device=device,
        normalize_advantage_per_mini_batch=bool(
            ppo["normalize_advantage_per_mini_batch"]
        ),
        gradient_sync_hook=GradientAverager(world_size=world_size),
    )
    return algorithm


def _mean_iteration_value(run: Mapping[str, object], name: str) -> float:
    records = run.get("iteration_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Phase 5 run has no iteration records")
    values: list[float] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("Phase 5 iteration record is invalid")
        source = record.get("update") if name != "reward" else record
        source_name = name if name != "reward" else "mean_reward"
        if not isinstance(source, Mapping) or source_name not in source:
            raise ValueError(f"Phase 5 iteration record is missing {name}")
        value = source[source_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Phase 5 iteration {index} has invalid {name}")
        value = float(value)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"Phase 5 iteration {index} has non-finite {name}")
        values.append(value)
    return sum(values) / len(values)


def _local_metrics(
    *,
    task: str,
    run: Mapping[str, object],
    transition_offset: int,
    collector: Any,
    episode_snapshot: object,
    metric_schema: Mapping[str, object],
) -> dict[str, object]:
    from somaforce_cross.learning.joint_runner import episode_metrics_from_snapshot

    if (
        isinstance(transition_offset, bool)
        or not isinstance(transition_offset, int)
        or transition_offset < 0
    ):
        raise ValueError("local metrics transition offset is invalid")
    if not isinstance(episode_snapshot, tuple):
        raise TypeError("EpisodeMetricLog.snapshot() must return a tuple")
    transition_metrics = dict(collector.transition_metrics())
    transition_metrics["reward"] = _mean_iteration_value(run, "reward")
    for name in metric_schema["ppo_metric_names"]:
        if not isinstance(name, str):
            raise ValueError("Phase 6 PPO metric name is invalid")
        transition_metrics[name] = _mean_iteration_value(run, name)
    if set(transition_metrics) != set(metric_schema["transition_metric_names"]):
        raise ValueError("local Phase 6 transition metric schema is incomplete")
    episode_count, episode_metrics = episode_metrics_from_snapshot(
        episode_snapshot, metric_schema=metric_schema
    )
    transitions = collector.transitions
    if run.get("transition_count") != transition_offset + transitions:
        raise ValueError("runner transition count differs from collected metrics")
    return {
        "raw_samples": collector.raw_samples(),
        "record": {
            "episode_count": episode_count,
            "episode_metrics": episode_metrics,
            "p_cross_trace": collector.p_cross_trace(),
            "task_transitions": transitions,
            "transition_metrics": transition_metrics,
        },
        "task": task,
    }


def _finite_sample_list(value: object, *, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    samples: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain finite numbers")
        number = float(item)
        if not torch.isfinite(torch.tensor(number)):
            raise ValueError(f"{name} must contain finite numbers")
        samples.append(number)
    return samples


def _merge_rank_metrics(
    rows: list[object], *, metric_schema: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    expected_row_keys = {"raw_samples", "record", "task"}
    for item in rows:
        if not isinstance(item, Mapping) or set(item) != expected_row_keys:
            raise ValueError("distributed task metrics are invalid")
        task = item["task"]
        record = item["record"]
        raw_samples = item["raw_samples"]
        if (
            not isinstance(task, str)
            or not isinstance(record, Mapping)
            or not isinstance(raw_samples, Mapping)
        ):
            raise ValueError("distributed task metric record is invalid")
        grouped.setdefault(task, []).append(item)
    transition_names = tuple(
        str(name) for name in metric_schema["transition_metric_names"]
    )
    episode_names = tuple(str(name) for name in metric_schema["episode_metric_names"])
    merged: dict[str, Mapping[str, object]] = {}
    for task, rows_for_task in grouped.items():
        records = [row["record"] for row in rows_for_task]
        if not all(isinstance(record, Mapping) for record in records):
            raise ValueError("distributed task metrics are invalid")
        records = [record for record in records if isinstance(record, Mapping)]
        expected_record_keys = {
            "episode_count",
            "episode_metrics",
            "p_cross_trace",
            "task_transitions",
            "transition_metrics",
        }
        if any(set(record) != expected_record_keys for record in records):
            raise ValueError("distributed task metric schema is invalid")
        transitions = [int(record["task_transitions"]) for record in records]
        if any(value <= 0 for value in transitions):
            raise ValueError("distributed task transition count is invalid")
        transition_rows = [record["transition_metrics"] for record in records]
        if any(
            not isinstance(row, Mapping) or set(row) != set(transition_names)
            for row in transition_rows
        ):
            raise ValueError("distributed transition metric schema is invalid")
        transition_metrics = {
            name: sum(
                float(row[name]) * count
                for row, count in zip(transition_rows, transitions, strict=True)
            )
            / sum(transitions)
            for name in transition_names
        }
        force_samples: list[float] = []
        force_rate_samples: list[float] = []
        for row, transitions_for_row in zip(rows_for_task, transitions, strict=True):
            samples = row["raw_samples"]
            assert isinstance(samples, Mapping)
            if set(samples) != {"force_norm_N", "force_rate_N_per_s"}:
                raise ValueError("distributed raw metric schema is invalid")
            force_values = _finite_sample_list(
                samples["force_norm_N"], name="force_norm_N"
            )
            if len(force_values) != transitions_for_row:
                raise ValueError("force sample count differs from task transitions")
            force_samples.extend(force_values)
            force_rate_samples.extend(
                _finite_sample_list(
                    samples["force_rate_N_per_s"], name="force_rate_N_per_s"
                )
            )
        force = torch.tensor(force_samples, dtype=torch.float64)
        force_rate = torch.tensor(force_rate_samples, dtype=torch.float64)
        transition_metrics["force_p95_N"] = float(torch.quantile(force, 0.95).item())
        transition_metrics["force_p99_N"] = float(torch.quantile(force, 0.99).item())
        transition_metrics["force_rate_p95_N_per_s"] = float(
            torch.quantile(force_rate, 0.95).item()
        )
        episode_counts = [int(record["episode_count"]) for record in records]
        if any(value < 0 for value in episode_counts):
            raise ValueError("distributed episode count is invalid")
        episode_rows = [record["episode_metrics"] for record in records]
        if any(
            not isinstance(row, Mapping) or set(row) != set(episode_names)
            for row in episode_rows
        ):
            raise ValueError("distributed episode metric schema is invalid")
        total_episodes = sum(episode_counts)
        episode_metrics: dict[str, float | None] = {}
        for name in episode_names:
            if total_episodes == 0:
                if any(row[name] is not None for row in episode_rows):
                    raise ValueError(
                        "zero episode count must persist null episode metrics"
                    )
                episode_metrics[name] = None
                continue
            weighted = 0.0
            for row, count in zip(episode_rows, episode_counts, strict=True):
                if count == 0:
                    if row[name] is not None:
                        raise ValueError("zero episode count must persist null metrics")
                    continue
                if isinstance(row[name], bool) or not isinstance(
                    row[name], (int, float)
                ):
                    raise ValueError("episode metric must be finite")
                weighted += float(row[name]) * count
            episode_metrics[name] = weighted / total_episodes
        traces = [record["p_cross_trace"] for record in records]
        if any(not isinstance(trace, list) or not trace for trace in traces):
            raise ValueError("distributed P_cross trace is invalid")
        trace_tensors = [torch.tensor(trace, dtype=torch.float64) for trace in traces]
        reference_shape = trace_tensors[0].shape
        if (
            reference_shape[1:] != (13, 5)
            or any(trace.shape != reference_shape for trace in trace_tensors)
            or any(not torch.isfinite(trace).all() for trace in trace_tensors)
        ):
            raise ValueError("distributed P_cross trace shape or finiteness is invalid")
        trace = torch.stack(trace_tensors).mean(dim=0)
        if not torch.allclose(
            trace.sum(dim=(-2, -1)),
            torch.ones(trace.shape[0], dtype=trace.dtype),
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise ValueError("distributed P_cross trace rows must sum to one")
        merged[task] = {
            "episode_count": total_episodes,
            "episode_metrics": episode_metrics,
            "p_cross_trace": trace.tolist(),
            "task_transitions": sum(transitions),
            "transition_metrics": transition_metrics,
        }
    return merged


def _canonical_task_metrics(
    task_metrics: Mapping[str, Mapping[str, object]], *, roster: object
) -> dict[str, Mapping[str, object]]:
    """Order task records exactly as the admitted roster before float pooling."""
    tasks = getattr(roster, "tasks", None)
    if not isinstance(tasks, tuple):
        raise TypeError("task metrics roster is invalid")
    task_names = [getattr(task, "task", None) for task in tasks]
    if not all(isinstance(name, str) and name for name in task_names) or len(
        task_names
    ) != len(set(task_names)):
        raise ValueError("task metrics roster has duplicate or invalid tasks")
    if set(task_metrics) != set(task_names):
        raise ValueError("task metrics keys do not match the roster")
    return {name: task_metrics[name] for name in task_names}


_EVALUATION_DIAGNOSTIC_KEYS = {
    "arms_residual_norm",
    "contact_fraction",
    "contact_loss",
    "dropout",
    "impulse",
    "legs_residual_norm",
    "p50_wrench",
    "p95_force_rate",
    "p95_wrench",
    "p99_wrench",
    "saturation",
    "semantic_entropy",
    "semantic_kl",
    "sensor_quality",
    "stability_margin",
    "waist_residual_norm",
}
_EVALUATION_REWARD_KEYS = {
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


def _finite_json_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON-safe finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a JSON-safe finite number")
    return number


def _validate_completed_episode_record(
    record: Mapping[str, object],
    *,
    task: str,
    mode: str,
    stage: str,
) -> dict[str, object]:
    expected = {
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
    if set(record) != expected:
        raise ValueError("completed episode record has unknown or missing fields")
    if (
        record["task"] != task
        or record["mode"] != mode
        or record["stage"] != stage
        or record["subset"] not in {"stage", "nominal"}
        or not isinstance(record["family"], str)
        or not record["family"]
    ):
        raise ValueError("completed episode record task/mode/stage is invalid")
    for name in ("env_id", "seed", "steps"):
        if (
            isinstance(record[name], bool)
            or not isinstance(record[name], int)
            or record[name] < 0
        ):
            raise ValueError(f"completed episode record {name} is invalid")
    for name in ("success", "failure", "timeout", "invalid"):
        if not isinstance(record[name], bool):
            raise ValueError(f"completed episode record {name} is invalid")
    normalized: dict[str, object] = dict(record)
    normalized["return"] = _finite_json_number(record["return"], name="record.return")
    for name in ("raw_reward_sums", "weighted_reward_sums"):
        values = record[name]
        if not isinstance(values, Mapping) or set(values) != _EVALUATION_REWARD_KEYS:
            raise ValueError(f"completed episode record {name} schema is invalid")
        normalized[name] = {
            key: _finite_json_number(value, name=f"record.{name}.{key}")
            for key, value in values.items()
        }
    diagnostics = record["diagnostics"]
    if (
        not isinstance(diagnostics, Mapping)
        or set(diagnostics) != _EVALUATION_DIAGNOSTIC_KEYS
    ):
        raise ValueError("completed episode record diagnostics schema is invalid")
    normalized["diagnostics"] = {
        key: _finite_json_number(value, name=f"record.diagnostics.{key}")
        for key, value in diagnostics.items()
    }
    try:
        json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("completed episode record is not JSON-safe") from exc
    return normalized


def _persist_completed_episode_records(
    *,
    rank_dir: Path,
    records: list[Mapping[str, object]],
    task: str,
    mode: str,
    stage: str,
    schedule_sha256: str,
    expected_stage_quota: int,
    expected_nominal_quota: int,
) -> dict[str, object]:
    path = rank_dir / "completed_episode_records.json"
    if path.exists():
        raise FileExistsError(f"completed episode evidence already exists: {path}")
    normalized = [
        _validate_completed_episode_record(record, task=task, mode=mode, stage=stage)
        for record in records
    ]
    if len(normalized) != expected_stage_quota + expected_nominal_quota:
        raise ValueError("completed episode record quota is incomplete")
    seeds = [int(record["seed"]) for record in normalized]
    if len(seeds) != len(set(seeds)):
        raise ValueError("completed episode records contain repeated seeds")
    subsets = {subset: 0 for subset in ("stage", "nominal")}
    for record in normalized:
        subsets[str(record["subset"])] += 1
    if subsets != {"stage": expected_stage_quota, "nominal": expected_nominal_quota}:
        raise ValueError("completed episode record subset quota is incomplete")
    seed_sha256 = hashlib.sha256(
        json.dumps(sorted(seeds), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "records": normalized,
        "row_count": len(normalized),
        "schedule_sha256": schedule_sha256,
        "seed_sha256": seed_sha256,
        "subset_counts": subsets,
    }
    _atomic_json(path, payload)
    return {"path": str(path), "sha256": _sha256(path), **payload}


def _run_paired_evaluation_rank(
    args: argparse.Namespace,
    *,
    config: Any,
    roster: Any,
    rank: int,
    world_size: int,
    progress: _EvaluationProgress | None = None,
) -> tuple[dict[str, Any], Any]:
    """Run paired episodes from row-owned slots without mutating PPO state."""
    from somaforce_cross.learning.actor_critic import ResidualActorCritic
    from somaforce_cross.learning.joint_runner import (
        paired_evaluation_plan,
        paired_evaluation_rank_seed,
        paired_evaluation_schedule,
        validate_paired_evaluation_schedule,
    )

    if args.resume is None:
        raise ValueError("paired evaluation requires a pre-evaluation checkpoint")
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("checkpoint_version") != "phase6_learning_checkpoint_v1"
        or checkpoint.get("pre_evaluation") is not True
    ):
        raise ValueError(
            "paired evaluation requires phase6_learning_checkpoint_v1 pre-evaluation input"
        )
    policy = ResidualActorCritic().to(device=torch.device(args.device))
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    task = roster.tasks[rank]
    num_envs = int(
        getattr(args, "paired_num_envs", None) or getattr(args, "num_envs", 64)
    )
    stage_quota = int(getattr(args, "paired_stage_quota", 256))
    nominal_quota = int(getattr(args, "paired_nominal_quota", 128))
    stage = str(getattr(args, "stage", "C1"))
    evaluation_seed = int(getattr(args, "evaluation_seed", 20262806))
    rank_seed = paired_evaluation_rank_seed(evaluation_seed, rank)
    plan = paired_evaluation_plan(
        roster,
        stage=stage,
        evaluation_seed=evaluation_seed,
        stage_quota=stage_quota,
        nominal_quota=nominal_quota,
    )
    task_plan = plan["tasks"][task.task]
    horizon = int(task_plan["horizon"])
    expected_per_mode = stage_quota + nominal_quota
    mode = getattr(args, "paired_mode", None)
    if mode not in {"residual", "scaffold_only"}:
        raise ValueError("paired evaluation requires exactly one paired mode")
    environment: Any | None = None
    completed_records: list[Mapping[str, object]] = []
    mode_counts = {"residual": 0, "scaffold_only": 0}
    subset_counts = {"stage": 0, "nominal": 0}
    unique_seeds: set[int] = set()
    seen_mode_seeds: set[tuple[str, int]] = set()
    control_steps = 0
    schedule: Mapping[str, object] | None = None

    def update_progress(status: str, *, force: bool = False) -> None:
        if progress is None:
            return
        progress.update(
            status=status,
            completed=len(completed_records),
            mode_counts={**mode_counts, "active": mode},
            subset_counts=subset_counts,
            control_steps=control_steps,
            force=force,
        )

    def policy_action(observations: object) -> torch.Tensor:
        with torch.no_grad():
            return policy.act_inference(observations)

    try:
        args.runtime_mode = mode
        environment = _build_environment(args, task=task, seed=rank_seed)
        if not all(
            hasattr(environment, attribute)
            for attribute in (
                "evaluation_nominal_probe",
                "bind_evaluation_schedule",
                "drain_evaluation_completions",
            )
        ):
            raise TypeError("paired evaluation environment lacks schedule ownership")
        schedule = paired_evaluation_schedule(
            task=task.task,
            stage=stage,
            evaluation_seed=evaluation_seed,
            rank=rank,
            num_envs=num_envs,
            stage_quota=stage_quota,
            nominal_quota=nominal_quota,
            nominal_probe=environment.evaluation_nominal_probe,
        )
        validate_paired_evaluation_schedule(schedule)
        if progress is not None:
            progress.schedule_sha256 = str(schedule["schedule_sha256"])
        environment.bind_evaluation_schedule(schedule, mode=mode)
        update_progress("environment_ready", force=True)
        observations, _ = environment.reset()
        max_steps = max(
            horizon * (stage_quota // num_envs + nominal_quota // num_envs + 2),
            horizon,
        )
        while len(completed_records) < expected_per_mode:
            actions = policy_action(observations)
            observations, _, _, _, _ = environment.step(actions)
            control_steps += 1
            newly_completed = environment.drain_evaluation_completions()
            for record in newly_completed:
                if record.get("mode") != mode:
                    raise ValueError("evaluation completion mode is out of order")
                subset = record.get("subset")
                seed = record.get("seed")
                if subset not in {"stage", "nominal"} or not isinstance(seed, int):
                    raise ValueError("evaluation completion record is invalid")
                completed_records.append(record)
                mode_counts[mode] += 1
                subset_counts[str(subset)] += 1
                if (mode, seed) in seen_mode_seeds:
                    raise ValueError(
                        "paired evaluation seed was repeated within a mode"
                    )
                seen_mode_seeds.add((mode, seed))
                unique_seeds.add(seed)
            update_progress("running")
            if control_steps > max_steps:
                raise RuntimeError("paired evaluation did not complete its row quota")
        update_progress("rollout_complete", force=True)
    except BaseException:
        if progress is not None:
            update_progress("error", force=True)
        raise
    episode_evidence: dict[str, object] | None = None
    if hasattr(args, "output_dir"):
        if schedule is None:
            raise AssertionError("paired evaluation schedule was not initialized")
        episode_evidence = _persist_completed_episode_records(
            rank_dir=Path(args.output_dir) / f"rank_{rank}",
            records=completed_records,
            task=task.task,
            mode=mode,
            stage=stage,
            schedule_sha256=str(schedule["schedule_sha256"]),
            expected_stage_quota=stage_quota,
            expected_nominal_quota=nominal_quota,
        )
    result = {
        "episodes": len(completed_records),
        "rank": rank,
        "status": "ok",
        "task": task.task,
        "world_size": world_size,
        "deterministic_actor_mean": True,
        "optimizer_steps": 0,
        "normalizer_updates": 0,
        "mode": mode,
        "horizon": horizon,
        "schedule_sha256": str(schedule["schedule_sha256"]),
        "mode_counts": mode_counts,
        "subset_counts": subset_counts,
        "unique_seeds": len(unique_seeds),
        "completed_seed_sha256": hashlib.sha256(
            json.dumps(sorted(unique_seeds), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "control_steps": control_steps,
    }
    if episode_evidence is not None:
        result["episode_evidence"] = {
            key: episode_evidence[key]
            for key in ("path", "row_count", "schedule_sha256", "seed_sha256", "sha256")
        }
    return result, environment


def _worker_main(argv: list[str]) -> int:
    args = _worker_parser().parse_args(argv)
    _validate_worker_paths(args)
    from isaaclab.app import AppLauncher

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.profile == "production_segment":
        num_envs = args.paired_num_envs if args.mode == "evaluate" else 64
        num_envs = num_envs or 64
        if args.segment_end is None or args.segment_end <= args.segment_start:
            raise ValueError("production segment bounds are invalid")
        iterations = args.segment_end - args.segment_start
    else:
        num_envs, iterations = _profile(args.profile)
    args.num_envs = num_envs
    args.device = f"cuda:{local_rank}"
    rank_dir = args.output_dir / f"rank_{rank}"
    launcher: Any | None = None
    environment: Any | None = None
    process_group_started = False
    status = "error"
    result: dict[str, Any]
    try:
        if torch.cuda.device_count() < 4 or local_rank >= torch.cuda.device_count():
            raise RuntimeError("Phase 6 requires four visible CUDA devices")
        torch.cuda.set_device(local_rank)
        launcher = AppLauncher(args)
        from somaforce_cross.learning.joint_runner import (
            JointCurriculum,
            JointTaskScheduler,
            TransitionMetricCollector,
            atomic_torch_save,
            atomic_learning_checkpoint,
            broadcast_curriculum_state,
            joint_checkpoint_payload,
            learning_checkpoint_payload,
            load_phase5_policy_only,
            load_learning_acceptance_config,
            load_phase6_config,
            load_phase6_task_roster,
            phase6_metrics_schema,
            pooled_macro_metrics,
            process_group_timeout,
            rank_rng_state,
            restore_rank_rng_state,
            sha256_file,
            synchronize_iteration_hashes,
            transition_accounting,
            restore_learning_checkpoint,
            validate_phase6_joint_metrics,
        )
        from somaforce_cross.learning.runner import Phase5Runner

        config = load_phase6_config(args.phase6_config)
        is_production = args.mode in {"train-segment", "evaluate"}
        acceptance = (
            load_learning_acceptance_config(args.acceptance_config)
            if is_production
            else None
        )
        source_manifest = None
        if is_production:
            source_manifest = _current_mainrunner_source_manifest()
        if world_size != int(config.payload["runtime"]["world_size"]):
            raise RuntimeError(
                "torch distributed world size differs from Phase 6 contract"
            )
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=process_group_timeout(config),
        )
        process_group_started = True
        roster = load_phase6_task_roster(args.roster, phase6_config=config)
        if is_production and args.source_rebind is not None:
            if acceptance is None:
                raise AssertionError("production worker has no acceptance contract")
            _validate_mainrunner_rebind_record(
                args.source_rebind,
                config=config,
                roster=roster,
                acceptance=acceptance,
            )
        if args.mode == "evaluate":
            source_rebind_sha256 = (
                _sha256(args.source_rebind)
                if args.source_rebind is not None and args.source_rebind.is_file()
                else _sha256(Path(__file__).resolve())
            )
            progress = _EvaluationProgress(
                path=rank_dir / "progress.json",
                rank=rank,
                world_size=world_size,
                task=roster.tasks[rank].task,
                device=args.device,
                total_episodes=args.paired_stage_quota + args.paired_nominal_quota,
                schedule_sha256="pending",
                checkpoint_sha256=_sha256(args.resume),
                source_rebind_sha256=source_rebind_sha256,
            )
            result, environment = _run_paired_evaluation_rank(
                args,
                config=config,
                roster=roster,
                rank=rank,
                world_size=world_size,
                progress=progress,
            )
            status = "ok"
            raise _EvaluationComplete
        scheduler = JointTaskScheduler(
            task_count=len(roster.tasks), world_size=world_size
        )
        slot = scheduler.assignment(
            cycle_index=args.cycle_index, window_index=args.window_index, rank=rank
        )
        task = roster.tasks[slot.task_index]
        _write_progress(
            rank_dir / "progress.json",
            cycle_index=args.cycle_index,
            iteration=0,
            rank=rank,
            state="app_and_process_group_ready",
            task=task.task,
            transitions=0,
            window_index=args.window_index,
        )
        algorithm = _new_algorithm(
            config, device=torch.device(args.device), world_size=world_size
        )
        checkpoint_path = (
            REPO_ROOT / config.payload["bindings"]["phase5_initial_checkpoint"]["path"]
        )
        curriculum = JointCurriculum(stage=args.stage)
        previous_task_transitions: dict[str, int] = {
            item.task: 0 for item in roster.tasks
        }
        resume_evaluation_history: list[dict[str, object]] = []
        if is_production and args.resume is not None:
            assert acceptance is not None and source_manifest is not None
            restored = restore_learning_checkpoint(
                args.resume,
                config=config,
                roster=roster,
                acceptance=acceptance,
                policy=algorithm.policy,
                optimizer=algorithm.optimizer,
                source_manifest=source_manifest,
                expected_iteration=args.segment_start,
                device=args.device,
            )
            curriculum.load_state_dict(restored["curriculum"])
            previous_task_transitions = {
                str(name): int(value)
                for name, value in restored["task_transitions"].items()
            }
            if not isinstance(restored["evaluation_history"], list) or not all(
                isinstance(item, Mapping) for item in restored["evaluation_history"]
            ):
                raise ValueError("production resume evaluation history is invalid")
            resume_evaluation_history = [
                dict(item) for item in restored["evaluation_history"]
            ]
            restore_rank_rng_state(
                restored["rank_rng"],
                rank=rank,
                local_device_index=local_rank,
                cuda_state_count=world_size,
            )
        else:
            load_phase5_policy_only(
                algorithm.policy, checkpoint_path, device=args.device
            )
        environment = _build_environment(args, task=task, seed=20260805 + rank)
        runner = Phase5Runner(environment, algorithm, config)
        metric_schema = phase6_metrics_schema(config)
        collector = TransitionMetricCollector(metric_schema)
        last_hash: dict[str, object] = {}

        def step_observe(_: Mapping[str, torch.Tensor], __: torch.Tensor) -> None:
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
                p_cross=algorithm.policy.last_semantic_output.P_cross,
            )

        def iteration_observe(
            iteration: int,
            transitions: int,
            update: Mapping[str, float | bool],
            _: tuple[dict[str, float], ...],
        ) -> None:
            nonlocal last_hash
            last_hash = synchronize_iteration_hashes(
                policy=algorithm.policy,
                optimizer=algorithm.optimizer,
                device=args.device,
            )
            _write_progress(
                rank_dir / "progress.json",
                cycle_index=args.cycle_index,
                iteration=iteration,
                optimizer_step=last_hash["step"],
                rank=rank,
                state="iteration_hash_verified",
                task=task.task,
                transitions=transitions,
                window_index=args.window_index,
            )

        run = runner.run(
            iterations=iterations,
            iteration_offset=args.segment_start if is_production else 0,
            transition_offset=(
                args.segment_start
                * num_envs
                * int(config.payload["ppo"]["num_steps_per_env"])
                if is_production
                else 0
            ),
            step_observer=step_observe,
            iteration_observer=iteration_observe,
        )
        expected_step = 24 * (args.segment_end if is_production else iterations)
        if algorithm.optimizer.state and int(last_hash["step"]) != expected_step:
            raise AssertionError("Phase 6 bounded optimizer step differs from profile")
        if environment.episode_metric_log is None:
            raise RuntimeError("Phase 6 residual environment has no episode metric log")
        local = _local_metrics(
            task=task.task,
            run=run,
            transition_offset=previous_task_transitions[task.task],
            collector=collector,
            episode_snapshot=environment.episode_metric_log.snapshot(),
            metric_schema=metric_schema,
        )
        gathered_metrics: list[object] = [None] * world_size
        torch.distributed.all_gather_object(gathered_metrics, local)
        gathered_rng: list[object] = [None] * world_size
        torch.distributed.all_gather_object(gathered_rng, rank_rng_state())
        global_transitions = transition_accounting(
            world_size=world_size,
            num_envs=num_envs,
            num_steps_per_env=int(config.payload["ppo"]["num_steps_per_env"]),
            iterations=iterations,
        )
        if is_production:
            global_transitions = (
                int(args.segment_end)
                * world_size
                * num_envs
                * int(config.payload["ppo"]["num_steps_per_env"])
            )
        if not is_production and JointCurriculum.evaluation_crossings(
            0, global_transitions
        ):
            raise RuntimeError(
                "bounded profile unexpectedly crossed curriculum evaluation"
            )
        rank_zero_metrics: list[object] = [None]
        if rank == 0:
            task_metrics = _merge_rank_metrics(
                gathered_metrics, metric_schema=metric_schema
            )
            if is_production:
                task_metrics = {
                    task_name: {
                        **record,
                        "task_transitions": int(record["task_transitions"])
                        + previous_task_transitions[task_name],
                    }
                    for task_name, record in task_metrics.items()
                }
            task_metrics = _canonical_task_metrics(task_metrics, roster=roster)
            pooled = pooled_macro_metrics(task_metrics)
            rank_zero_metrics[0] = {
                "pooled": pooled["pooled"],
                "task_records": task_metrics,
            }
        torch.distributed.broadcast_object_list(
            rank_zero_metrics, src=0, device=args.device
        )
        metrics = rank_zero_metrics[0]
        if not isinstance(metrics, Mapping):
            raise RuntimeError("rank-zero Phase 6 metric broadcast is invalid")
        metrics = validate_phase6_joint_metrics(
            metrics,
            config=config,
            roster=roster,
            expected_global_transitions=global_transitions,
        )
        rank_zero_checkpoint: list[object] = [None]
        if rank == 0:
            task_transitions = {
                name: int(record["task_transitions"])
                for name, record in task_metrics.items()
            }
            curriculum.transitions = global_transitions
            rank_rng = {str(index): value for index, value in enumerate(gathered_rng)}
            if is_production:
                assert acceptance is not None and source_manifest is not None
                next_crossing = _current_evaluation_crossing(global_transitions)
                payload = learning_checkpoint_payload(
                    config=config,
                    roster=roster,
                    acceptance=acceptance,
                    policy=algorithm.policy,
                    optimizer=algorithm.optimizer,
                    iteration=int(args.segment_end),
                    task_transitions=task_transitions,
                    curriculum=curriculum,
                    next_crossing=next_crossing,
                    evaluation_history=resume_evaluation_history,
                    rank_rng=rank_rng,
                    source_manifest=source_manifest,
                    metrics=metrics,
                    pre_evaluation=True,
                )
                checkpoint = args.output_dir / "pre_evaluation.pt"
            else:
                payload = joint_checkpoint_payload(
                    config=config,
                    roster=roster,
                    policy=algorithm.policy,
                    optimizer=algorithm.optimizer,
                    iteration=iterations,
                    cycle_index=args.cycle_index,
                    window_index=args.window_index,
                    slot_transitions=num_envs
                    * int(config.payload["ppo"]["num_steps_per_env"])
                    * iterations,
                    task_transitions=task_transitions,
                    curriculum=curriculum,
                    rank_rng=rank_rng,
                    metrics=metrics,
                    hashes=last_hash,
                )
                checkpoint = (
                    args.output_dir
                    / "checkpoints"
                    / f"cycle_{args.cycle_index:04d}_window_{args.window_index:04d}.pt"
                )
            rank_zero_checkpoint[0] = {
                "path": str(checkpoint),
                "sha256": (
                    atomic_learning_checkpoint(checkpoint, payload)
                    if is_production
                    else atomic_torch_save(checkpoint, payload)
                ),
            }
        torch.distributed.broadcast_object_list(
            rank_zero_checkpoint, src=0, device=args.device
        )
        checkpoint_record = rank_zero_checkpoint[0]
        if (
            not isinstance(checkpoint_record, Mapping)
            or not isinstance(checkpoint_record.get("path"), str)
            or sha256_file(Path(str(checkpoint_record["path"])))
            != checkpoint_record.get("sha256")
        ):
            raise RuntimeError(
                "rank-zero checkpoint broadcast or shared-file verification failed"
            )
        curriculum = broadcast_curriculum_state(
            curriculum, rank=rank, device=args.device
        )
        task_transitions = {
            name: int(record["task_transitions"])
            for name, record in metrics["task_records"].items()
        }
        task_fraction = {
            name: value / global_transitions for name, value in task_transitions.items()
        }
        result = {
            "checkpoint": dict(checkpoint_record),
            "contracts": {
                "phase6_canonical_sha256": config.canonical_sha256,
                "phase6_raw_sha256": config.raw_sha256,
                "roster_canonical_sha256": roster.canonical_sha256,
                "roster_raw_sha256": roster.raw_sha256,
            },
            "curriculum": curriculum.state_dict(),
            "global_transitions": global_transitions,
            "local_transitions": collector.transitions,
            "metrics": metrics,
            "optimizer_sha256": last_hash["optimizer"],
            "optimizer_step": expected_step,
            "per_task_transitions": task_transitions,
            "policy_sha256": last_hash["policy"],
            "profile": args.profile,
            "rank": rank,
            "schedule": {
                "cycle_index": args.cycle_index,
                "task": task.task,
                "task_index": slot.task_index,
                "window_index": args.window_index,
            },
            "status": "ok",
            "task_fraction": task_fraction,
        }
        status = "ok"
    except _EvaluationComplete:
        pass
    except BaseException as exc:
        result = {
            "exception_message": str(exc),
            "exception_type": type(exc).__name__,
            "profile": args.profile,
            "rank": rank,
            "status": "error",
            "traceback": traceback.format_exc(),
        }
    _persist_worker_result(rank_dir / "result.json", result)
    try:
        if args.mode == "evaluate" and status == "ok":
            if not process_group_started:
                raise RuntimeError("evaluation cleanup requires a process group")
            torch.distributed.barrier()
            for close_rank in range(world_size):
                if rank == close_rank:
                    if environment is None:
                        raise RuntimeError(
                            "evaluation worker has no environment to close"
                        )
                    environment.close()
                    environment = None
                    print("ENV_CLOSED", flush=True)
                torch.distributed.barrier()
            if progress is not None:
                progress.update(status="complete", force=True)
        elif environment is not None:
            environment.close()
            print("ENV_CLOSED", flush=True)
    finally:
        if process_group_started and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if launcher is not None:
            launcher.app.close()
    return 0 if status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if "--mode" not in values:
        raise SystemExit("--mode {rank-wrapper,worker} is required")
    mode_index = values.index("--mode") + 1
    if mode_index >= len(values):
        raise SystemExit("--mode requires a value")
    if values[mode_index] == "rank-wrapper":
        return _run_rank_wrapper(_rank_wrapper_parser().parse_args(values))
    if values[mode_index] == "production":
        return _run_production(_production_parser().parse_args(values))
    if values[mode_index] == "summarize":
        return _run_summarize(_summarize_parser().parse_args(values))
    if values[mode_index] == "production-summarize":
        return _run_production_summary(_production_summary_parser().parse_args(values))
    if values[mode_index] == "mainrunner-rebind":
        return _run_mainrunner_rebind(_mainrunner_rebind_parser().parse_args(values))
    if values[mode_index] == "mainrunner-evaluation-merge":
        return _run_mainrunner_evaluation_merge(
            _mainrunner_evaluation_merge_parser().parse_args(values)
        )
    if values[mode_index] == "mainrunner-finalize":
        return _run_mainrunner_finalize(
            _mainrunner_finalize_parser().parse_args(values)
        )
    if values[mode_index] == "legacy-recovery-summary":
        return _run_legacy_recovery_summary(
            _legacy_recovery_summary_parser().parse_args(values)
        )
    if values[mode_index] == "harness-rebind":
        return _run_harness_rebind(_harness_rebind_parser().parse_args(values))
    if values[mode_index] == "evaluation-devicefix-rebind":
        return _run_evaluation_devicefix_rebind(
            _evaluation_devicefix_rebind_parser().parse_args(values)
        )
    if values[mode_index] == "stepboundary-rebind":
        return _run_stepboundary_rebind(
            _stepboundary_rebind_parser().parse_args(values)
        )
    if values[mode_index] == "pairedbatch-rebind":
        return _run_pairedbatch_rebind(_pairedbatch_rebind_parser().parse_args(values))
    if values[mode_index] == "rankseed-rebind":
        return _run_rankseed_rebind(_rankseed_rebind_parser().parse_args(values))
    if values[mode_index] == "episodeorder-rebind":
        return _run_episodeorder_rebind(
            _episodeorder_rebind_parser().parse_args(values)
        )
    if values[mode_index] == "modeprocess-rebind":
        return _run_modeprocess_rebind(_modeprocess_rebind_parser().parse_args(values))
    if values[mode_index] == "warningpolicy-rebind":
        return _run_warningpolicy_rebind(
            _warningpolicy_rebind_parser().parse_args(values)
        )
    if values[mode_index] == "summaryschema-rebind":
        return _run_summaryschema_rebind(
            _summaryschema_rebind_parser().parse_args(values)
        )
    if values[mode_index] == "recovery-evaluate":
        return _run_recovery_evaluate(_recovery_evaluate_parser().parse_args(values))
    if values[mode_index] == "devicefix-recovery-evaluate":
        return _run_devicefix_recovery_evaluate(
            _devicefix_recovery_evaluate_parser().parse_args(values)
        )
    if values[mode_index] == "stepboundary-recovery-evaluate":
        return _run_stepboundary_recovery_evaluate(
            _stepboundary_recovery_evaluate_parser().parse_args(values)
        )
    if values[mode_index] == "pairedbatch-recovery-evaluate":
        return _run_pairedbatch_recovery_evaluate(
            _pairedbatch_recovery_evaluate_parser().parse_args(values)
        )
    if values[mode_index] == "rankseed-recovery-evaluate":
        return _run_rankseed_recovery_evaluate(
            _rankseed_recovery_evaluate_parser().parse_args(values)
        )
    if values[mode_index] == "episodeorder-recovery-evaluate":
        return _run_episodeorder_recovery_evaluate(
            _episodeorder_recovery_evaluate_parser().parse_args(values)
        )
    if values[mode_index] == "modeprocess-recovery-evaluate":
        return _run_modeprocess_recovery_evaluate(
            _modeprocess_recovery_evaluate_parser().parse_args(values)
        )
    if values[mode_index] == "warningpolicy-recovery-evaluate":
        return _run_warningpolicy_recovery_evaluate(
            _warningpolicy_recovery_evaluate_parser().parse_args(values)
        )
    if values[mode_index] == "summaryschema-finalize":
        return _run_summaryschema_finalize(
            _summaryschema_finalize_parser().parse_args(values)
        )
    if values[mode_index] == "finalize":
        return _run_finalize(_finalize_parser().parse_args(values))
    if values[mode_index] in {"worker", "train-segment", "evaluate"}:
        return _worker_main(values)
    raise SystemExit("unknown Phase 6 mode")


if __name__ == "__main__":
    raise SystemExit(main())
