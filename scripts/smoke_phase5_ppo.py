#!/usr/bin/env python3
"""Bounded real-Isaac Phase 5 PPO smoke with persistent worker evidence."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE5_CONFIG = REPO_ROOT / "configs/phase5_ppo_v1.json"
PHASE4_CONTRACT = REPO_ROOT / "configs/phase4b5_numeric_contract.json"
RESULT_MARKER = "PHASE5_RESULT="
PROGRESS_MARKERS = ("JSON_WRITTEN", RESULT_MARKER, "ENV_CLOSED", "APP_CLOSED")
CASES = ((2, 1), (64, 2))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _last_result_marker(log: Path) -> dict[str, Any] | None:
    if not log.is_file():
        return None
    for line in reversed(
        log.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if line.startswith(RESULT_MARKER):
            value = json.loads(line[len(RESULT_MARKER) :])
            if not isinstance(value, dict):
                raise ValueError("PHASE5_RESULT must encode a JSON object")
            return value
    return None


def _persist_worker_result(result_json: Path, result: dict[str, Any]) -> None:
    _atomic_json(result_json, result)
    print("JSON_WRITTEN", flush=True)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)


def _close_worker_resources(*, env: Any | None, launcher: Any | None) -> None:
    try:
        if env is not None:
            env.close()
            print("ENV_CLOSED", flush=True)
    finally:
        if launcher is not None:
            launcher.app.close()
            print("APP_CLOSED", flush=True)


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


def _signal_name(return_code: int | None) -> str | None:
    if return_code is None or return_code >= 0:
        return None
    return signal.Signals(-return_code).name


def _terminate_group(process: subprocess.Popen[str], *, grace_s: float) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=grace_s)


def _run_worker_case(
    *,
    command: list[str],
    result_json: Path,
    log: Path,
    wrapper_json: Path,
    timeout_s: int,
    label: str,
) -> dict[str, Any]:
    """Execute one child process group and require persisted result/close evidence."""
    log.parent.mkdir(parents=True, exist_ok=True)
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
    markers: list[str] = []
    maximum_hwm: int | None = None
    with log.open("w", encoding="utf-8") as handle:

        def drain() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
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
                if time.monotonic() - started > timeout_s:
                    timed_out = True
                    _terminate_group(process, grace_s=20.0)
                    break
                time.sleep(0.2)
            process.wait(timeout=25.0)
        finally:
            reader.join(timeout=25.0)
    elapsed_s = round(time.monotonic() - started, 3)
    result_error: str | None = None
    try:
        result = _read_json(result_json)
        marker_result = _last_result_marker(log)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = None
        marker_result = None
        result_error = f"{type(exc).__name__}: {exc}"
    if marker_result is None:
        result_error = result_error or "PHASE5_RESULT is missing"
    elif result is not None and marker_result != result:
        result_error = "PHASE5_RESULT does not match result.json"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    exit_code = (
        process.returncode
        if process.returncode is not None and process.returncode >= 0
        else None
    )
    env_closed = "ENV_CLOSED" in markers and "ENV_CLOSED" in log_text
    app_closed = "APP_CLOSED" in markers and "APP_CLOSED" in log_text
    last_marker = markers[-1] if markers else None
    marker_order_valid = markers in (
        ["JSON_WRITTEN", RESULT_MARKER.rstrip("="), "ENV_CLOSED"],
        ["JSON_WRITTEN", RESULT_MARKER.rstrip("="), "ENV_CLOSED", "APP_CLOSED"],
    )
    close_verification: str | None = None
    if app_closed and env_closed:
        close_verification = "child_marker"
    elif env_closed and exit_code == 0 and last_marker == "ENV_CLOSED":
        close_verification = "parent_verified_exit0"
    passed = (
        not timed_out
        and exit_code == 0
        and result is not None
        and result.get("status") == "ok"
        and result_error is None
        and marker_order_valid
        and close_verification is not None
    )
    wrapper = {
        "label": label,
        "command": command,
        "result_json": str(result_json),
        "log": str(log),
        "exit_code": exit_code,
        "signal": _signal_name(process.returncode),
        "timed_out": timed_out,
        "elapsed_s": elapsed_s,
        "vmhwm_kib": maximum_hwm,
        "markers": markers,
        "last_progress_marker": last_marker,
        "marker_order_valid": marker_order_valid,
        "result_status": None if result is None else result.get("status"),
        "result_error": result_error,
        "close_verification": close_verification,
        "passed": passed,
    }
    _atomic_json(wrapper_json, wrapper)
    if not passed:
        raise RuntimeError(f"Phase 5 worker evidence gate failed: {wrapper_json}")
    return wrapper


def _worker_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker",), required=True)
    parser.add_argument("--task", choices=("push_door_hand",), required=True)
    parser.add_argument("--stage", choices=("C1",), required=True)
    parser.add_argument("--num-envs", choices=(2, 64), type=int, required=True)
    parser.add_argument("--iterations", choices=(1, 2), type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, default=PHASE5_CONFIG)
    parser.add_argument("--phase4-contract", type=Path, default=PHASE4_CONTRACT)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("suite",), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/phase5_ppo_smoke")
    )
    return parser


def _build_environment(args: argparse.Namespace, holder: dict[str, Any]) -> Any:
    from somaforce_cross.envs.residual_env import SomaForceResidualEnv
    from somaforce_cross.envs.residual_env_cfg import SomaForceResidualEnvCfg
    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
    from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg

    artifact = REPO_ROOT / "artifacts/scaffolds/hdmi_push_door_hand/v1"
    spec = get_hdmi_task_spec("push_door_hand")
    cfg = SomaForceResidualEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.artifact_dir = str(artifact)
    cfg.scene = make_scene_cfg(
        artifact,
        args.num_envs,
        spec,
        rigid_object_mass=spec.nominal_object_mass or 8.0,
    )
    cfg.smoke_profile.task = "push_door_hand"
    cfg.smoke_profile.runtime_mode = "residual"
    cfg.smoke_profile.scaffold_stage = "C1"
    cfg.smoke_profile.numeric_contract_path = str(args.phase4_contract.resolve())
    cfg.__post_init__()
    env = SomaForceResidualEnv(cfg)
    holder["env"] = env
    return env


def _worker_main(argv: list[str]) -> int:
    args = _worker_parser().parse_args(argv)
    if args.phase5_config.resolve() != PHASE5_CONFIG.resolve():
        raise ValueError("worker requires configs/phase5_ppo_v1.json")
    if args.phase4_contract.resolve() != PHASE4_CONTRACT.resolve():
        raise ValueError("worker requires configs/phase4b5_numeric_contract.json")
    expected_iterations = {2: 1, 64: 2}[args.num_envs]
    if args.iterations != expected_iterations:
        raise ValueError(
            "bounded case iteration count does not match the Phase 5 contract"
        )
    from isaaclab.app import AppLauncher

    launcher: Any | None = None
    holder: dict[str, Any] = {}
    status = "error"
    try:
        launcher = AppLauncher(args)
        repo_path = str(REPO_ROOT)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from somaforce_cross.learning.config import load_phase5_config
        from somaforce_cross.learning.runner import make_phase5_runner

        environment = _build_environment(args, holder)
        runner = make_phase5_runner(environment, config_path=args.phase5_config)
        training = runner.run(iterations=args.iterations)
        update = training["last_update"]
        required_update = (
            "ppo_loss",
            "semantic_dir_loss",
            "semantic_mag_loss",
            "semantic_total",
            "semantic_entropy",
            "semantic_pipeline_grad_norm",
        )
        if not all(name in update for name in required_update):
            raise AssertionError("Phase 5 update record is incomplete")
        if not update["semantic_pipeline_has_nonzero_gradient"]:
            raise AssertionError("Phase 5 semantic gradient diagnostic is zero")
        config = load_phase5_config(args.phase5_config)
        result = {
            "status": "ok",
            "task": args.task,
            "stage": args.stage,
            "num_envs": args.num_envs,
            "iterations": args.iterations,
            "phase5_config_raw_sha256": config.raw_sha256,
            "phase5_config_canonical_sha256": config.canonical_sha256,
            "phase4_contract_canonical_sha256": config.payload["phase4"][
                "canonical_sha256"
            ],
            "actor_input_dim": training["actor_input_dim"],
            "critic_input_dim": training["critic_input_dim"],
            "action_dim": training["action_dim"],
            "semantic_target_stored": training["semantic_target_stored"],
            "training": training,
        }
        status = "ok"
    except BaseException as exc:
        result = {
            "status": "error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    _persist_worker_result(args.result_json, result)
    _close_worker_resources(env=holder.get("env"), launcher=launcher)
    return 0 if status == "ok" else 1


def _suite_main(argv: list[str]) -> int:
    args = _suite_parser().parse_args(argv)
    run_dir = args.output_dir / time.strftime("run_%Y%m%dT%H%M%S")
    cases: list[dict[str, Any]] = []
    for num_envs, iterations in CASES:
        base = run_dir / f"push_door_hand_C1_{num_envs}env_{iterations}iter"
        command = [
            args.python,
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--task",
            "push_door_hand",
            "--stage",
            "C1",
            "--num-envs",
            str(num_envs),
            "--iterations",
            str(iterations),
            "--result-json",
            str(base / "result.json"),
            "--phase5-config",
            str(PHASE5_CONFIG),
            "--phase4-contract",
            str(PHASE4_CONTRACT),
            "--headless",
            "--device",
            "cuda:0",
        ]
        wrapper = _run_worker_case(
            command=command,
            result_json=base / "result.json",
            log=base / "worker.log",
            wrapper_json=base / "wrapper.json",
            timeout_s=args.timeout_s,
            label=f"phase5-push_door_hand-C1-{num_envs}env-{iterations}iter",
        )
        result = _read_json(base / "result.json")
        if result is None:
            raise AssertionError("worker result disappeared after wrapper pass")
        cases.append({"result": result, "wrapper": wrapper})
    summary = {
        "status": "ok",
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "all_exit_zero": True,
            "all_signal_null": True,
            "all_timed_out_false": True,
            "all_parent_verified": all(
                case["wrapper"]["close_verification"]
                in ("child_marker", "parent_verified_exit0")
                for case in cases
            ),
        },
    }
    summary_path = run_dir / "summary.json"
    _atomic_json(summary_path, summary)
    print(json.dumps({"status": "ok", "summary": str(summary_path)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if "--mode" not in values:
        raise SystemExit("--mode {suite,worker} is required")
    index = values.index("--mode") + 1
    if index >= len(values):
        raise SystemExit("--mode requires a value")
    if values[index] == "worker":
        return _worker_main(values)
    if values[index] == "suite":
        return _suite_main(values)
    raise SystemExit(f"unsupported mode: {values[index]}")


if __name__ == "__main__":
    raise SystemExit(main())
