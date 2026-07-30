#!/usr/bin/env python3
"""Bounded persistent evidence harness for Phase 4B5B2B1 Isaac probes."""

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
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXISTING_C0_SMOKE = REPO_ROOT / "scripts/smoke_phase4_multitask_env.py"
NUMERIC_CONTRACT = REPO_ROOT / "configs/phase4b5_numeric_contract.json"
TASKS = ("push_door_hand", "push_box", "move_suitcase", "move_largebox")
RUNTIME_ENV_COUNTS = (2, 64)
RESULT_MARKER = "B5_RUNTIME_RESULT="
PROGRESS_MARKERS = ("JSON_WRITTEN", "ENV_CLOSED", "APP_CLOSED", RESULT_MARKER)
RESULT_PROGRESS_MARKER = RESULT_MARKER.rstrip("=")


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


def _last_json_marker(log: Path) -> dict[str, Any] | None:
    if not log.is_file():
        return None
    for line in reversed(
        log.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if not line.startswith(RESULT_MARKER):
            continue
        value = json.loads(line[len(RESULT_MARKER) :])
        if not isinstance(value, dict):
            raise ValueError("B5_RUNTIME_RESULT must encode a JSON object")
        return value
    return None


def _persist_worker_result(result_json: Path, result: dict[str, Any]) -> None:
    """Persist one business result before either Isaac resource is closed."""
    _atomic_json(result_json, result)
    print("JSON_WRITTEN", flush=True)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)


def _close_worker_resources(*, env: Any | None, launcher: Any | None) -> None:
    """Close the environment before SimulationApp and emit only completed closes."""
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
    """Run one child in its own process group and persist wrapper evidence."""
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
    elapsed = round(time.monotonic() - started, 3)
    result: dict[str, Any] | None
    result_error: str | None = None
    try:
        result = _read_json(result_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = None
        result_error = f"{type(exc).__name__}: {exc}"
    marker_result: dict[str, Any] | None
    try:
        marker_result = _last_json_marker(log)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        marker_result = None
        result_error = result_error or f"{type(exc).__name__}: {exc}"
    if marker_result is None:
        result_error = result_error or "B5_RUNTIME_RESULT is missing"
    elif result is not None and result != marker_result:
        result_error = "B5_RUNTIME_RESULT does not match result JSON"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    exit_code = (
        process.returncode
        if process.returncode is not None and process.returncode >= 0
        else None
    )
    close_source: str | None = None
    env_closed = "ENV_CLOSED" in markers and "ENV_CLOSED" in log_text
    app_closed = "APP_CLOSED" in markers and "APP_CLOSED" in log_text
    last_marker = markers[-1] if markers else None
    marker_order_valid = markers in (
        ["JSON_WRITTEN", RESULT_PROGRESS_MARKER, "ENV_CLOSED"],
        ["JSON_WRITTEN", RESULT_PROGRESS_MARKER, "ENV_CLOSED", "APP_CLOSED"],
    )
    if not app_closed and env_closed and exit_code == 0 and last_marker == "ENV_CLOSED":
        close_source = "parent_verified_exit0"
    elif app_closed and env_closed:
        close_source = "child_marker"
    passed = (
        not timed_out
        and exit_code == 0
        and result is not None
        and result.get("status") == "ok"
        and result_error is None
        and "JSON_WRITTEN" in markers
        and env_closed
        and marker_order_valid
        and close_source is not None
    )
    wrapper = {
        "label": label,
        "command": command,
        "result_json": str(result_json),
        "log": str(log),
        "exit_code": exit_code,
        "signal": _signal_name(process.returncode),
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "vmhwm_kib": maximum_hwm,
        "last_progress_marker": last_marker,
        "markers": markers,
        "marker_order_valid": marker_order_valid,
        "result_status": None if result is None else result.get("status"),
        "result_error": result_error,
        "close_verification": close_source,
        "passed": passed,
    }
    _atomic_json(wrapper_json, wrapper)
    if not passed:
        raise RuntimeError(f"worker evidence gate failed: {wrapper_json}")
    return wrapper


def _worker_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker",), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--selected-env-ids", type=int, nargs="+", required=True)
    parser.add_argument("--stage", choices=("C1", "C2", "C3"), default="C1")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--contract", type=Path, default=NUMERIC_CONTRACT)
    parser.add_argument("--result-json", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _tensor_sha256(value: Any) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _runtime_probe(
    args: argparse.Namespace, environment_holder: dict[str, Any]
) -> dict[str, Any]:
    import torch

    from somaforce_cross.envs.residual_env import SomaForceResidualEnv
    from somaforce_cross.envs.residual_env_cfg import SomaForceResidualEnvCfg
    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
    from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg

    if args.num_envs not in RUNTIME_ENV_COUNTS:
        raise ValueError("runtime probe supports only 2 or 64 environments")
    if len(set(args.selected_env_ids)) != len(args.selected_env_ids):
        raise ValueError("selected env ids must be unique")
    if min(args.selected_env_ids) < 0 or max(args.selected_env_ids) >= args.num_envs:
        raise ValueError("selected env ids are outside the batch")
    expected_ids = (0,) if args.num_envs == 2 else (0, 7, 63)
    if tuple(args.selected_env_ids) != expected_ids:
        raise ValueError(f"selected env ids must be {expected_ids}")
    contract_path = args.contract.expanduser().resolve()
    if contract_path != NUMERIC_CONTRACT.resolve():
        raise ValueError("runtime probe requires the explicit repository v2 contract")

    spec = get_hdmi_task_spec(args.task)
    artifact = REPO_ROOT / "artifacts/scaffolds" / f"hdmi_{args.task}" / "v1"
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
    profile = cfg.smoke_profile
    profile.task = args.task
    profile.runtime_mode = "scaffold_only"
    profile.scaffold_stage = args.stage
    profile.numeric_contract_path = str(contract_path)
    cfg.__post_init__()

    env = SomaForceResidualEnv(cfg)
    environment_holder["env"] = env
    env.reset()
    selected = torch.tensor(args.selected_env_ids, device=env.device, dtype=torch.long)
    mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    mask[selected] = False
    before = {
        name: value.clone() for name, value in env.environment_owned_state().items()
    }
    env._reset_idx(selected)
    after = env.environment_owned_state()
    changed = [
        name
        for name, value in before.items()
        if not torch.equal(value[mask], after[name][mask])
    ]
    if changed:
        raise AssertionError(f"unselected rows changed: {changed}")
    applied_physics = env.adapter.runtime_parameter_readback(selected)
    applied_scaffold = env.scaffold_runtime.readback(selected)
    applied_sensor = env._sensor_parameter_readback(selected)
    if not torch.equal(env.parameter_store.physics_mismatch[selected], applied_physics):
        raise AssertionError("critic physics row is not the applied readback")
    if not torch.equal(
        env.parameter_store.scaffold_mismatch[selected], applied_scaffold
    ):
        raise AssertionError("critic scaffold row is not the runtime readback")
    if not torch.equal(env.parameter_store.sensor_mismatch[selected], applied_sensor):
        raise AssertionError("critic sensor row is not the public sensor readback")
    if env.task_spec.task == "push_door_hand" and (
        torch.count_nonzero(applied_physics[:, 0:3])
        or torch.count_nonzero(applied_physics[:, 5:11])
    ):
        raise AssertionError("door deferred fields are not bitwise zero")
    reset_source = (REPO_ROOT / "somaforce_cross/envs/residual_env.py").read_text(
        encoding="utf-8"
    )
    reset_source = reset_source[
        reset_source.index("def _reset_scaffold_only_idx") : reset_source.index(
            "def _owned_sensor_state"
        )
    ]
    if "self.sim.step" in reset_source or "self.sim.forward" in reset_source:
        raise AssertionError("runtime reset contains a global physics advance")
    return {
        "task": args.task,
        "stage": args.stage,
        "num_envs": args.num_envs,
        "selected_env_ids": args.selected_env_ids,
        "unselected_envs": int(mask.sum()),
        "selected_reset_bitwise": True,
        "checked_fields": sorted(before),
        "requested_applied_storage_separate": {
            "physics": env.requested_physics.data_ptr()
            != env.parameter_store.physics_mismatch.data_ptr(),
            "scaffold": env.requested_scaffold.data_ptr()
            != env.parameter_store.scaffold_mismatch.data_ptr(),
            "sensor": env.requested_sensor.data_ptr()
            != env.parameter_store.sensor_mismatch.data_ptr(),
        },
        "requested_sha256": {
            "physics": _tensor_sha256(env.requested_physics[selected]),
            "scaffold": _tensor_sha256(env.requested_scaffold[selected]),
            "sensor": _tensor_sha256(env.requested_sensor[selected]),
        },
        "applied_store_sha256": {
            "physics": _tensor_sha256(env.parameter_store.physics_mismatch[selected]),
            "scaffold": _tensor_sha256(env.parameter_store.scaffold_mismatch[selected]),
            "sensor": _tensor_sha256(env.parameter_store.sensor_mismatch[selected]),
        },
        "applied_readback_sha256": {
            "physics": _tensor_sha256(applied_physics),
            "scaffold": _tensor_sha256(applied_scaffold),
            "sensor": _tensor_sha256(applied_sensor),
        },
        "runtime_delay": applied_scaffold[:, 0].to(torch.int64).cpu().tolist(),
        "runtime_alpha": applied_scaffold[:, 1].cpu().tolist(),
        "reset_has_no_global_advance": True,
    }


def _worker_main(argv: list[str]) -> int:
    parser = _worker_parser()
    args = parser.parse_args(argv)
    from isaaclab.app import AppLauncher

    launcher: Any | None = None
    environment_holder: dict[str, Any] = {}
    result: dict[str, Any]
    status = "error"
    try:
        launcher = AppLauncher(args)
        repo_path = str(REPO_ROOT)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        import somaforce_cross

        package_file = Path(somaforce_cross.__file__).resolve()
        package_root = (REPO_ROOT / "somaforce_cross").resolve()
        if package_file != package_root and package_root not in package_file.parents:
            raise ImportError(
                f"somaforce_cross resolved outside repository: {package_file}"
            )
        result = {
            "package_file": str(package_file),
            **_runtime_probe(args, environment_holder),
        }
        status = "ok"
    except BaseException as exc:
        result = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    result = {"status": status, **result}
    _persist_worker_result(args.result_json, result)
    _close_worker_resources(env=environment_holder.get("env"), launcher=launcher)
    return 0 if status == "ok" else 1


def _suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("suite",), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase4b5b2a"))
    parser.add_argument("--skip-c0", action="store_true")
    parser.add_argument("--runtime-task", choices=TASKS)
    parser.add_argument("--runtime-num-envs", choices=RUNTIME_ENV_COUNTS, type=int)
    return parser


def _runtime_cases(args: argparse.Namespace) -> tuple[tuple[str, int], ...]:
    tasks = (args.runtime_task,) if args.runtime_task is not None else TASKS
    counts = (
        (args.runtime_num_envs,)
        if args.runtime_num_envs is not None
        else RUNTIME_ENV_COUNTS
    )
    return tuple((task, count) for task in tasks for count in counts)


def _c0_result_from_log(log: Path, *, mode: str) -> dict[str, Any]:
    for line in reversed(
        log.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("mode") == mode:
            return value
    raise ValueError("existing C0 runner emitted no mode JSON")


def _run_c0_external(
    *,
    command: list[str],
    result_json: Path,
    log: Path,
    wrapper_json: Path,
    timeout_s: int,
    label: str,
    expected_mode: str,
) -> dict[str, Any]:
    """Persist C0 runner output while verifying its documented finalizers."""
    try:
        wrapper = _run_worker_case(
            command=command,
            result_json=result_json,
            log=log,
            wrapper_json=wrapper_json,
            timeout_s=timeout_s,
            label=label,
        )
    except RuntimeError:
        # Existing C0 runners do not receive result-json, so synthesize their
        # result below after the bounded process has finished and rewrite wrapper.
        wrapper = _read_json(wrapper_json)
        if wrapper is None:
            raise
    result: dict[str, Any]
    try:
        output = _c0_result_from_log(log, mode=expected_mode)
        result = {"status": "ok", "runner": output}
    except BaseException as exc:
        result = {
            "status": "error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    _atomic_json(result_json, result)
    source = EXISTING_C0_SMOKE.read_text(encoding="utf-8")
    finalizers = "env.close()" in source and "simulation_app.close()" in source
    wrapper["result_status"] = result["status"]
    wrapper["result_error"] = (
        None if result["status"] == "ok" else result["exception_message"]
    )
    wrapper["close_verification"] = (
        "existing_runner_finally_source_and_exit0"
        if finalizers and wrapper["exit_code"] == 0 and not wrapper["timed_out"]
        else None
    )
    wrapper["passed"] = (
        wrapper["exit_code"] == 0
        and not wrapper["timed_out"]
        and result["status"] == "ok"
        and wrapper["close_verification"] is not None
    )
    _atomic_json(wrapper_json, wrapper)
    if not wrapper["passed"]:
        raise RuntimeError(f"C0 evidence gate failed: {wrapper_json}")
    return wrapper


def _c0_cases(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for task in TASKS:
        for count in (1, 64):
            base = run_dir / "c0" / f"{task}_{count}"
            common = [
                "--task",
                task,
                "--num-envs",
                str(count),
                "--steps",
                "6",
                "--headless",
                "--device",
                "cuda:0",
            ]
            golden = _run_c0_external(
                command=[
                    args.python,
                    str(EXISTING_C0_SMOKE),
                    "--mode",
                    "golden",
                    *common,
                ],
                result_json=base / "golden.result.json",
                log=base / "golden.log",
                wrapper_json=base / "golden.wrapper.json",
                timeout_s=args.timeout_s,
                label=f"c0-golden-{task}-{count}",
                expected_mode="golden",
            )
            golden_result = _read_json(base / "golden.result.json")
            assert golden_result is not None
            residual = _run_c0_external(
                command=[
                    args.python,
                    str(EXISTING_C0_SMOKE),
                    "--mode",
                    "residual",
                    *common,
                ],
                result_json=base / "residual.result.json",
                log=base / "residual.log",
                wrapper_json=base / "residual.wrapper.json",
                timeout_s=args.timeout_s,
                label=f"c0-residual-{task}-{count}",
                expected_mode="residual",
            )
            residual_result = _read_json(base / "residual.result.json")
            assert residual_result is not None
            compare = _run_c0_external(
                command=[
                    args.python,
                    str(EXISTING_C0_SMOKE),
                    "--mode",
                    "compare",
                    "--golden-trace",
                    golden_result["runner"]["trace"],
                    "--residual-trace",
                    residual_result["runner"]["trace"],
                    "--headless",
                ],
                result_json=base / "compare.result.json",
                log=base / "compare.log",
                wrapper_json=base / "compare.wrapper.json",
                timeout_s=args.timeout_s,
                label=f"c0-compare-{task}-{count}",
                expected_mode="compare",
            )
            compare_result = _read_json(base / "compare.result.json")
            assert compare_result is not None
            if compare_result["runner"].get("first_divergence") is not None:
                raise AssertionError(f"C0 first divergence: {task}/{count}")
            evidence.append(
                {
                    "task": task,
                    "num_envs": count,
                    "golden": golden,
                    "residual": residual,
                    "compare": compare,
                    "first_divergence": None,
                }
            )
    return evidence


def _suite_main(argv: list[str]) -> int:
    args = _suite_parser().parse_args(argv)
    run_dir = args.output_dir / time.strftime("run_%Y%m%dT%H%M%S")
    evidence: dict[str, Any] = {"runtime": [], "c0": []}
    for task, count in _runtime_cases(args):
        selected = (0,) if count == 2 else (0, 7, 63)
        base = run_dir / "runtime" / f"{task}_{count}"
        command = [
            args.python,
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--task",
            task,
            "--num-envs",
            str(count),
            "--selected-env-ids",
            *(str(value) for value in selected),
            "--stage",
            "C1",
            "--contract",
            str(NUMERIC_CONTRACT),
            "--result-json",
            str(base / "result.json"),
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
            label=f"runtime-{task}-{count}",
        )
        result = _read_json(base / "result.json")
        assert result is not None
        evidence["runtime"].append({"wrapper": wrapper, "result": result})
    if not args.skip_c0:
        evidence["c0"] = _c0_cases(args, run_dir)
    evidence["summary"] = {
        "all_exit_zero": True,
        "all_signal_null": True,
        "all_timed_out_false": True,
        "all_status_ok": True,
        "runtime_cases": len(evidence["runtime"]),
        "c0_cases": len(evidence["c0"]),
    }
    evidence["status"] = "ok"
    summary_path = run_dir / "summary.json"
    _atomic_json(summary_path, evidence)
    print(json.dumps({"status": "ok", "summary": str(summary_path)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if "--mode" not in values:
        raise SystemExit("--mode {suite,worker} is required")
    mode_index = values.index("--mode") + 1
    if mode_index >= len(values):
        raise SystemExit("--mode requires a value")
    if values[mode_index] == "worker":
        return _worker_main(values)
    if values[mode_index] == "suite":
        return _suite_main(values)
    raise SystemExit(f"unsupported mode: {values[mode_index]}")


if __name__ == "__main__":
    raise SystemExit(main())
