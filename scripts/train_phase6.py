#!/usr/bin/env python3
"""Bounded four-rank Phase 6 joint-training worker and evidence wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
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
PHASE6_CONFIG = REPO_ROOT / "configs/phase6_joint_training_v1.json"
PHASE6_ROSTER = REPO_ROOT / "configs/phase6_task_roster_v1.json"
RESULT_MARKER = "PHASE6_JOINT_RESULT="
PROGRESS_MARKERS = ("JSON_WRITTEN", RESULT_MARKER, "ENV_CLOSED")


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


def _rank_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rank-wrapper",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke_4gpu_2env_1iter", "smoke_4gpu_64env_2iter"),
        required=True,
    )
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--phase6-config", type=Path, default=PHASE6_CONFIG)
    parser.add_argument("--roster", type=Path, default=PHASE6_ROSTER)
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
    parser.add_argument("--mode", choices=("worker",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke_4gpu_2env_1iter", "smoke_4gpu_64env_2iter"),
        required=True,
    )
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--resume", type=Path)
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


def _worker_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
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
        try:
            while process.poll() is None:
                observed = _proc_hwm_kib(process.pid)
                if observed is not None:
                    maximum_hwm = max(maximum_hwm or observed, observed)
                if time.monotonic() - started > args.timeout_s:
                    timed_out = True
                    _terminate_group(process)
                    break
                time.sleep(0.2)
            process.wait(timeout=35.0)
        finally:
            reader.join(timeout=35.0)
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
        "vmhwm_kib": maximum_hwm,
        "world_size": world_size,
        "warning_counts": warning_counts,
    }
    _atomic_json(rank_dir / "wrapper.json", wrapper)
    if not wrapper["passed"]:
        raise RuntimeError(f"Phase 6 wrapper evidence gate failed: {rank_dir}")
    return 0


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
    cfg.smoke_profile.runtime_mode = "residual"
    cfg.smoke_profile.scaffold_stage = "C1"
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
    environment.set_curriculum_stage("C1")
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
    collector: Any,
    episode_snapshot: object,
    metric_schema: Mapping[str, object],
) -> dict[str, object]:
    from somaforce_cross.learning.joint_runner import episode_metrics_from_snapshot

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
    if run.get("transition_count") != transitions:
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


def _worker_main(argv: list[str]) -> int:
    args = _worker_parser().parse_args(argv)
    _validate_worker_paths(args)
    from isaaclab.app import AppLauncher

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
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
            broadcast_curriculum_state,
            joint_checkpoint_payload,
            load_phase5_policy_only,
            load_phase6_config,
            load_phase6_task_roster,
            phase6_metrics_schema,
            pooled_macro_metrics,
            process_group_timeout,
            rank_rng_state,
            sha256_file,
            synchronize_iteration_hashes,
            transition_accounting,
            validate_phase6_joint_metrics,
        )
        from somaforce_cross.learning.runner import Phase5Runner

        config = load_phase6_config(args.phase6_config)
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
        environment = _build_environment(args, task=task, seed=20260805 + rank)
        algorithm = _new_algorithm(
            config, device=torch.device(args.device), world_size=world_size
        )
        checkpoint_path = (
            REPO_ROOT / config.payload["bindings"]["phase5_initial_checkpoint"]["path"]
        )
        load_phase5_policy_only(algorithm.policy, checkpoint_path, device=args.device)
        curriculum = JointCurriculum()
        if args.resume is not None:
            raise ValueError(
                "bounded Phase 6 smoke profiles do not permit in-slot resume"
            )
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
            step_observer=step_observe,
            iteration_observer=iteration_observe,
        )
        expected_step = 24 * iterations
        if algorithm.optimizer.state and int(last_hash["step"]) != expected_step:
            raise AssertionError("Phase 6 bounded optimizer step differs from profile")
        if environment.episode_metric_log is None:
            raise RuntimeError("Phase 6 residual environment has no episode metric log")
        local = _local_metrics(
            task=task.task,
            run=run,
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
        if JointCurriculum.evaluation_crossings(0, global_transitions):
            raise RuntimeError(
                "bounded profile unexpectedly crossed curriculum evaluation"
            )
        rank_zero_metrics: list[object] = [None]
        if rank == 0:
            task_metrics = _merge_rank_metrics(
                gathered_metrics, metric_schema=metric_schema
            )
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
                "sha256": atomic_torch_save(checkpoint, payload),
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
        if environment is not None:
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
    if values[mode_index] == "summarize":
        return _run_summarize(_summarize_parser().parse_args(values))
    if values[mode_index] == "worker":
        return _worker_main(values)
    raise SystemExit("unknown Phase 6 mode")


if __name__ == "__main__":
    raise SystemExit(main())
