#!/usr/bin/env python3
"""Persistent, contact-bearing single-task Phase 5 learning acceptance."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE4_CONTRACT = REPO_ROOT / "configs/phase4b5_numeric_contract.json"
PHASE5_CONFIG = REPO_ROOT / "configs/phase5_ppo_v1.json"
ACCEPTANCE_CONFIG = REPO_ROOT / "configs/phase5_learning_acceptance_v1.json"
RESULT_MARKER = "PHASE5_LEARNING_RESULT="
PROGRESS_MARKER = "PHASE5_PROGRESS="
PROGRESS_MARKERS = ("JSON_WRITTEN", RESULT_MARKER, "ENV_CLOSED", "APP_CLOSED")
SEMANTIC_PROBE_VERSION = "phase5_semantic_probe_v1"
SEMANTIC_PROBE_KEYS = {
    "policy",
    "semantic_target",
    "collection",
    "contracts",
    "source_allowlist_sha256",
    "artifact_sha256",
}
SOURCE_ALLOWLIST = (
    "configs/phase5_learning_acceptance_v1.json",
    "somaforce_cross/learning/__init__.py",
    "somaforce_cross/learning/acceptance.py",
    "somaforce_cross/learning/losses.py",
    "somaforce_cross/learning/runner.py",
    "somaforce_cross/learning/semantic_ppo.py",
    "scripts/train_phase5.py",
)


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


def _git_diff_sha256() -> str:
    result = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _git_state() -> dict[str, str]:
    def command(*args: str) -> str:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()

    return {
        "branch": command("git", "branch", "--show-current"),
        "head": command("git", "rev-parse", "HEAD"),
        "origin_v1": command("git", "rev-parse", "origin/v1"),
        "diff_sha256": _git_diff_sha256(),
        "status_porcelain": command("git", "status", "--porcelain=v1"),
    }


def _last_result_marker(log: Path) -> dict[str, Any] | None:
    if not log.is_file():
        return None
    for line in reversed(
        log.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if line.startswith(RESULT_MARKER):
            value = json.loads(line[len(RESULT_MARKER) :])
            if not isinstance(value, dict):
                raise ValueError("learning result marker must be a JSON object")
            return value
    return None


def _persist_worker_result(result_json: Path, result: dict[str, Any]) -> None:
    _atomic_json(result_json, result)
    print("JSON_WRITTEN", flush=True)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)


def _write_progress(
    args: argparse.Namespace,
    *,
    started: float,
    state: str,
    iteration: int = 0,
    transitions: int = 0,
    contact_bearing_transitions: int = 0,
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {
        "version": "phase5_progress_v1",
        "case": args.case,
        "state": state,
        "iteration": iteration,
        "transitions": transitions,
        "contact_bearing_transitions": contact_bearing_transitions,
        "elapsed_s": round(time.monotonic() - started, 3),
        **fields,
    }
    _atomic_json(args.case_dir / "progress.json", payload)
    print(PROGRESS_MARKER + json.dumps(payload, sort_keys=True), flush=True)


def _write_final_evaluation_progress(
    args: argparse.Namespace,
    *,
    started: float,
    state: str,
    phase: str,
    episodes_completed: int,
    episodes_total: int,
    batch_index: int = 0,
    batches_completed: int = 0,
    batches_total: int = 0,
) -> None:
    """Persist observable final-evaluation phase and episode progress."""
    _write_progress(
        args,
        started=started,
        state=state,
        phase=phase,
        episodes_completed=episodes_completed,
        episodes_total=episodes_total,
        batch_index=batch_index,
        batches_completed=batches_completed,
        batches_total=batches_total,
    )


def _persist_final_metrics(
    args: argparse.Namespace,
    *,
    final_metrics: Mapping[str, Any],
    context: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    """Atomically preserve final metrics before their acceptance gate can fail."""

    def mapping(value: object, name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be an object")
        return value

    payload = {
        "outcome": dict(mapping(final_metrics["outcome"], "final_metrics.outcome")),
        "retention": dict(
            mapping(final_metrics["retention"], "final_metrics.retention")
        ),
        "action": dict(mapping(final_metrics["action"], "final_metrics.action")),
        "residual": dict(mapping(final_metrics["residual"], "final_metrics.residual")),
        "acceptance_thresholds": dict(context["acceptance"].payload["thresholds"]),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "contracts": dict(context["contracts"]),
        "source_allowlist_sha256": context["source_allowlist_sha256"],
        "git": _git_state(),
    }
    _atomic_json(args.case_dir / "final_metrics.json", payload)
    return payload


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
    base: Path,
    timeout_s: int,
    label: str,
) -> dict[str, Any]:
    """Run one isolated Isaac process group and verify durable close evidence."""
    base.mkdir(parents=True, exist_ok=True)
    _atomic_json(base / "command.json", {"command": command, "label": label})
    result_json = base / "result.json"
    log = base / "worker.log"
    wrapper_json = base / "wrapper.json"
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
                    _terminate_group(process, grace_s=30.0)
                    break
                time.sleep(0.2)
            process.wait(timeout=35.0)
        finally:
            reader.join(timeout=35.0)
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
        result_error = result_error or "PHASE5_LEARNING_RESULT is missing"
    elif result is not None and marker_result != result:
        result_error = "PHASE5_LEARNING_RESULT does not match result.json"
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
        raise RuntimeError(
            f"Phase 5 learning worker evidence gate failed: {wrapper_json}"
        )
    return wrapper


def _worker_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker",), required=True)
    parser.add_argument(
        "--case",
        choices=("contact_probe", "semantic_probe", "learning", "final_evaluation"),
        required=True,
    )
    parser.add_argument("--num-envs", choices=(2, 64), type=int, required=True)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--semantic-probe-artifact", type=Path)
    parser.add_argument("--phase4-contract", type=Path, default=PHASE4_CONTRACT)
    parser.add_argument("--phase5-config", type=Path, default=PHASE5_CONFIG)
    parser.add_argument("--acceptance-config", type=Path, default=ACCEPTANCE_CONFIG)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("suite",), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/phase5_learning_acceptance")
    )
    parser.add_argument("--run-id")
    parser.add_argument("--only-case", choices=("semantic_probe",))
    parser.add_argument("--contact-timeout-s", type=int, default=3600)
    parser.add_argument("--learning-timeout-s", type=int, default=28800)
    parser.add_argument("--evaluation-timeout-s", type=int, default=7200)
    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    if args.phase4_contract.resolve() != PHASE4_CONTRACT.resolve():
        raise ValueError("worker requires the frozen Phase 4 numeric contract")
    if args.phase5_config.resolve() != PHASE5_CONFIG.resolve():
        raise ValueError("worker requires configs/phase5_ppo_v1.json")
    if args.acceptance_config.resolve() != ACCEPTANCE_CONFIG.resolve():
        raise ValueError("worker requires configs/phase5_learning_acceptance_v1.json")
    if args.case == "learning" and args.semantic_probe_artifact is None:
        raise ValueError("learning worker requires --semantic-probe-artifact")
    if args.case != "learning" and args.semantic_probe_artifact is not None:
        raise ValueError("--semantic-probe-artifact is only valid for learning")


def _close_worker_resources(
    *, env: Any | None, launcher: Any | None, env_already_closed: bool = False
) -> None:
    try:
        if env is not None:
            env.close()
            print("ENV_CLOSED", flush=True)
        elif env_already_closed:
            print("ENV_CLOSED", flush=True)
    finally:
        if launcher is not None:
            launcher.app.close()
            print("APP_CLOSED", flush=True)


def _seed_everything(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _contract_context(args: argparse.Namespace) -> dict[str, Any]:
    from somaforce_cross.envs.numeric_contract import (
        canonical_sha256 as phase4_canonical,
    )
    from somaforce_cross.learning.acceptance import (
        load_learning_acceptance_config,
        source_allowlist_sha256,
        source_sha256,
    )
    from somaforce_cross.learning.config import load_phase5_config

    phase4_payload = json.loads(args.phase4_contract.read_text(encoding="utf-8"))
    phase5 = load_phase5_config(args.phase5_config)
    acceptance = load_learning_acceptance_config(args.acceptance_config)
    source_paths = [REPO_ROOT / path for path in SOURCE_ALLOWLIST]
    source_hashes = source_sha256(source_paths)
    return {
        "acceptance": acceptance,
        "contracts": {
            "phase4_raw_sha256": _sha256(args.phase4_contract),
            "phase4_canonical_sha256": phase4_canonical(phase4_payload),
            "phase5_raw_sha256": phase5.raw_sha256,
            "phase5_canonical_sha256": phase5.canonical_sha256,
            "learning_acceptance_raw_sha256": acceptance.raw_sha256,
            "learning_acceptance_canonical_sha256": acceptance.canonical_sha256,
        },
        "source_hashes": source_hashes,
        "source_allowlist_sha256": source_allowlist_sha256(source_hashes),
    }


def _build_environment(args: argparse.Namespace, *, seed: int) -> Any:
    from somaforce_cross.envs.residual_env import SomaForceResidualEnv
    from somaforce_cross.envs.residual_env_cfg import SomaForceResidualEnvCfg
    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
    from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg

    artifact = REPO_ROOT / "artifacts/scaffolds/hdmi_push_door_hand/v1"
    spec = get_hdmi_task_spec("push_door_hand")
    cfg = SomaForceResidualEnvCfg()
    cfg.seed = seed
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
    cfg.smoke_profile.episode_length_steps = 508
    cfg.__post_init__()
    if cfg.smoke_profile.episode_length_steps != 508:
        raise AssertionError(
            "learning acceptance must not use the 8-step smoke horizon"
        )
    env = SomaForceResidualEnv(cfg)
    env.set_curriculum_stage("C1")
    return env


def _semantic_probe_bytes(payload: Mapping[str, Any], torch: Any) -> bytes:
    normalized = dict(payload)
    normalized["artifact_sha256"] = ""
    buffer = io.BytesIO()
    torch.save(normalized, buffer)
    return buffer.getvalue()


def _semantic_probe_digest(payload: Mapping[str, Any], torch: Any) -> str:
    return hashlib.sha256(_semantic_probe_bytes(payload, torch)).hexdigest()


def _save_semantic_probe_artifact(
    path: Path,
    *,
    policy: Any,
    semantic_target: Any,
    collection: Mapping[str, Any],
    context: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "policy": policy.detach().cpu().clone(),
        "semantic_target": semantic_target.detach().cpu().clone(),
        "collection": dict(collection),
        "contracts": dict(context["contracts"]),
        "source_allowlist_sha256": context["source_allowlist_sha256"],
        "artifact_sha256": "",
    }
    payload["artifact_sha256"] = _semantic_probe_digest(payload, torch)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "artifact_sha256": payload["artifact_sha256"],
    }


def _load_semantic_probe_artifact(
    path: Path, *, context: Mapping[str, Any], torch: Any
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"semantic probe artifact is missing: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or set(value) != SEMANTIC_PROBE_KEYS:
        raise ValueError("semantic probe artifact schema mismatch")
    if value["artifact_sha256"] != _semantic_probe_digest(value, torch):
        raise ValueError("semantic probe artifact hash mismatch")
    if value["contracts"] != context["contracts"]:
        raise ValueError("semantic probe artifact contract mismatch")
    if value["source_allowlist_sha256"] != context["source_allowlist_sha256"]:
        raise ValueError("semantic probe artifact source mismatch")
    policy = value["policy"]
    semantic_target = value["semantic_target"]
    if not isinstance(policy, torch.Tensor) or policy.shape != (4096, 668):
        raise ValueError("semantic probe policy shape mismatch")
    if not isinstance(semantic_target, torch.Tensor) or semantic_target.shape != (
        4096,
        31,
    ):
        raise ValueError("semantic probe target shape mismatch")
    if not torch.isfinite(policy).all() or not torch.isfinite(semantic_target).all():
        raise ValueError("semantic probe artifact contains nonfinite values")
    if not torch.all(semantic_target[:, 30] > 0.0):
        raise ValueError("semantic probe artifact contains zero-weight samples")
    collection = value["collection"]
    if not isinstance(collection, Mapping):
        raise ValueError("semantic probe collection stats are missing")
    if (
        collection.get("samples") != 4096
        or collection.get("num_envs") != 64
        or collection.get("seed") != 20261801
        or not isinstance(collection.get("control_steps"), int)
        or collection["control_steps"] > 2048
    ):
        raise ValueError("semantic probe collection stats mismatch")
    return {
        "policy": policy.clone(),
        "semantic_target": semantic_target.clone(),
        "collection": dict(collection),
        "artifact_sha256": value["artifact_sha256"],
        "sha256": _sha256(path),
    }


class _ContactTracker:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.transitions = 0
        self.env_ids: set[int] = set()

    def observe(self, observations: Mapping[str, Any], _: Any) -> None:
        target = observations["semantic_target"]
        mask = target[:, 30] > 0.0
        self.transitions += int(mask.sum().item())
        self.env_ids.update(
            int(item) for item in mask.nonzero(as_tuple=False).flatten()
        )

    def result(self) -> dict[str, int]:
        return {
            "contact_bearing_transitions": self.transitions,
            "envs_with_valid_contact": len(self.env_ids),
            "num_envs": self.num_envs,
        }


class _ActionResidualMetrics:
    """Collect final deterministic residual statistics on contact-bearing steps."""

    def __init__(self, num_envs: int) -> None:
        self.contact_saturation: list[float] = []
        self.contact_norms: list[float] = []
        self.per_env_saturation: list[list[float]] = [[] for _ in range(num_envs)]

    def observe(self, env: Any, raw_residual: Any, semantic_target: Any) -> None:
        import torch

        contact = semantic_target[:, 30] > 0.0
        if not contact.any():
            return
        saturation = (torch.abs(torch.tanh(raw_residual)) >= 0.98).float().mean(dim=1)
        norms = torch.linalg.vector_norm(env._delta_safe, dim=1)
        for env_id in contact.nonzero(as_tuple=False).flatten().tolist():
            value = float(saturation[env_id].item())
            self.contact_saturation.append(value)
            self.per_env_saturation[env_id].append(value)
            self.contact_norms.append(float(norms[env_id].item()))

    def result(self) -> dict[str, float | int]:
        if not self.contact_saturation or not self.contact_norms:
            raise AssertionError(
                "final evaluation has no contact-bearing action samples"
            )
        windows: list[float] = []
        for values in self.per_env_saturation:
            if not values:
                continue
            width = min(1000, len(values))
            windows.extend(
                sum(values[index : index + width]) / width
                for index in range(len(values) - width + 1)
            )
        if not windows:
            raise AssertionError("final evaluation has no saturation windows")
        return {
            "contact_samples": len(self.contact_saturation),
            "contact_saturation_fraction": sum(self.contact_saturation)
            / len(self.contact_saturation),
            "contact_saturation_window_max": max(windows),
            "contact_mean_delta_safe_norm": sum(self.contact_norms)
            / len(self.contact_norms),
            "contact_norm_fraction_ge_min": sum(
                value >= 0.005 for value in self.contact_norms
            )
            / len(self.contact_norms),
        }


def _collect_fixed_probe(
    env: Any,
    *,
    samples: int,
    max_steps: int,
    on_progress: Any | None = None,
) -> tuple[Any, Any, dict[str, int]]:
    import torch

    from somaforce_cross.learning.acceptance import FixedSemanticProbe

    probe = FixedSemanticProbe(samples)
    observations, _ = env.reset()
    for control_step in range(max_steps):
        probe.add(observations["policy"], observations["semantic_target"])
        if probe.count == samples:
            if on_progress is not None:
                on_progress(control_steps=control_step, samples=probe.count)
            policy, target = probe.tensors()
            return policy, target, {"control_steps": control_step, "samples": samples}
        actions = torch.zeros(env.num_envs, 23, device=env.device, dtype=torch.float32)
        observations, rewards, terminated, time_outs, _ = env.step(actions)
        values = (rewards, observations["policy"], observations["semantic_target"])
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("non-finite fixed semantic probe value")
        if terminated.dtype != torch.bool or time_outs.dtype != torch.bool:
            raise TypeError("probe done flags must be boolean")
        if on_progress is not None and (control_step + 1) % 32 == 0:
            on_progress(control_steps=control_step + 1, samples=probe.count)
    raise AssertionError(
        f"fixed semantic probe did not collect {samples} valid contact samples"
    )


def _probe_metrics(policy: Any, probe_policy: Any, probe_target: Any) -> dict[str, Any]:
    import torch

    from somaforce_cross.learning.acceptance import semantic_probe_metrics

    device = next(policy.parameters()).device
    dir_batches: list[Any] = []
    mag_batches: list[Any] = []
    with torch.inference_mode():
        for start in range(0, probe_policy.shape[0], 256):
            observations = {"policy": probe_policy[start : start + 256].to(device)}
            policy.act_inference(observations)
            semantic = policy.last_semantic_output
            dir_batches.append(semantic.p_dir.detach().cpu())
            mag_batches.append(semantic.p_mag.detach().cpu())
    result = semantic_probe_metrics(
        torch.cat(dir_batches, dim=0), torch.cat(mag_batches, dim=0), probe_target
    )
    if result["samples"] != 4096:
        raise AssertionError("checkpoint semantic probe sample count changed")
    return result


def _rng_state(torch: Any) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().cpu(),
        "cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(torch: Any, state: Mapping[str, Any]) -> None:
    if set(state) != {"cpu", "cuda"}:
        raise ValueError("checkpoint RNG state is incomplete")
    torch.set_rng_state(state["cpu"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.to("cuda") for value in state["cuda"]])


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any], torch: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return _sha256(path)


def _checkpoint_payload(
    *,
    runner: Any,
    iteration: int,
    transitions: int,
    context: Mapping[str, Any],
    probe_policy: Any,
    probe_target: Any,
    torch: Any,
) -> dict[str, Any]:
    return {
        "checkpoint_version": "phase5_learning_acceptance_v1",
        "policy": runner.algorithm.policy.state_dict(),
        "optimizer": runner.algorithm.optimizer.state_dict(),
        "iteration": iteration,
        "transitions": transitions,
        "rng": _rng_state(torch),
        "contracts": context["contracts"],
        "source_sha256": context["source_hashes"],
        "source_allowlist_sha256": context["source_allowlist_sha256"],
        "runtime_parameters": context["acceptance"].payload["run"],
        "fixed_semantic_probe": {
            "policy": probe_policy,
            "semantic_target": probe_target,
        },
    }


def _restore_checkpoint(
    path: Path, *, runner: Any, context: Mapping[str, Any], torch: Any
) -> tuple[int, int, Any, Any]:
    value = torch.load(path, map_location=runner.algorithm.device, weights_only=False)
    required = {
        "checkpoint_version",
        "policy",
        "optimizer",
        "iteration",
        "transitions",
        "rng",
        "contracts",
        "source_sha256",
        "source_allowlist_sha256",
        "runtime_parameters",
        "fixed_semantic_probe",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(
            "checkpoint schema is not a strict learning acceptance checkpoint"
        )
    if value["checkpoint_version"] != "phase5_learning_acceptance_v1":
        raise ValueError("checkpoint version mismatch")
    if value["contracts"] != context["contracts"]:
        raise ValueError("checkpoint contract hashes differ from the current source")
    if (
        value["source_sha256"] != context["source_hashes"]
        or value["source_allowlist_sha256"] != context["source_allowlist_sha256"]
    ):
        raise ValueError(
            "strict same-source checkpoint recovery rejected source mismatch"
        )
    if value["runtime_parameters"] != context["acceptance"].payload["run"]:
        raise ValueError(
            "checkpoint runtime parameters differ from the frozen acceptance"
        )
    if not isinstance(value["iteration"], int) or not isinstance(
        value["transitions"], int
    ):
        raise ValueError("checkpoint iteration/transitions must be integers")
    runner.algorithm.policy.load_state_dict(value["policy"], strict=True)
    runner.algorithm.optimizer.load_state_dict(value["optimizer"])
    _restore_rng_state(torch, value["rng"])
    probe = value["fixed_semantic_probe"]
    if not isinstance(probe, Mapping) or set(probe) != {"policy", "semantic_target"}:
        raise ValueError("checkpoint fixed semantic probe is incomplete")
    return (
        value["iteration"],
        value["transitions"],
        probe["policy"].cpu(),
        probe["semantic_target"].cpu(),
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _save_checkpoint_and_evaluation(
    *,
    case_dir: Path,
    runner: Any,
    iteration: int,
    transitions: int,
    crossed_threshold: int,
    context: Mapping[str, Any],
    probe_policy: Any,
    probe_target: Any,
    torch: Any,
) -> dict[str, Any]:
    probe = _probe_metrics(runner.algorithm.policy, probe_policy, probe_target)
    stem = f"iteration_{iteration:07d}_transitions_{transitions:07d}"
    checkpoint = case_dir / "checkpoints" / f"{stem}.pt"
    checkpoint_sha = _atomic_checkpoint(
        checkpoint,
        _checkpoint_payload(
            runner=runner,
            iteration=iteration,
            transitions=transitions,
            context=context,
            probe_policy=probe_policy,
            probe_target=probe_target,
            torch=torch,
        ),
        torch,
    )
    record = {
        "iteration": iteration,
        "transitions": transitions,
        "crossed_transition_threshold": crossed_threshold,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "semantic_probe": probe,
        "contracts": context["contracts"],
        "source_allowlist_sha256": context["source_allowlist_sha256"],
        "git": _git_state(),
    }
    _atomic_json(case_dir / "evaluations" / f"{stem}.json", record)
    return record


def _run_contact_probe(
    args: argparse.Namespace, *, context: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    from somaforce_cross.learning.runner import make_phase5_runner

    expected = {2: 16, 64: 16}
    if args.iterations != expected[args.num_envs]:
        raise ValueError("contact probe requires exactly 16 iterations")
    _seed_everything(torch, int(context["acceptance"].payload["run"]["train_seed"]))
    env = _build_environment(
        args, seed=int(context["acceptance"].payload["run"]["train_seed"])
    )
    try:
        runner = make_phase5_runner(env, config_path=args.phase5_config)
        tracker = _ContactTracker(args.num_envs)
        training = runner.run(iterations=args.iterations, step_observer=tracker.observe)
        contact = tracker.result()
        required_envs = 2 if args.num_envs == 2 else 48
        required_transitions = 16 if args.num_envs == 2 else 512
        if contact["envs_with_valid_contact"] < required_envs:
            raise AssertionError(
                "contact probe did not reach the required environment rows"
            )
        if contact["contact_bearing_transitions"] < required_transitions:
            raise AssertionError(
                "contact probe did not reach the required contact samples"
            )
        return {
            "case": "contact_probe",
            "contact": contact,
            "training": training,
            "episode_horizon": 508,
        }
    finally:
        env.close()


def _run_semantic_probe(
    args: argparse.Namespace, *, context: Mapping[str, Any], started: float
) -> dict[str, Any]:
    import torch

    run = context["acceptance"].payload["run"]
    probe_config = context["acceptance"].payload["probe"]
    if args.num_envs != 64 or args.iterations != 0:
        raise ValueError("semantic probe requires 64 environments and zero iterations")
    seed = int(run["semantic_probe_seed"])
    _seed_everything(torch, seed)
    _write_progress(args, started=started, state="env_build_start")
    env = _build_environment(args, seed=seed)
    try:
        _write_progress(args, started=started, state="env_build_done")

        def on_progress(*, control_steps: int, samples: int) -> None:
            _write_progress(
                args,
                started=started,
                state="probe_collecting",
                transitions=control_steps * args.num_envs,
                contact_bearing_transitions=samples,
                control_steps=control_steps,
                samples=samples,
            )

        policy, target, collection = _collect_fixed_probe(
            env,
            samples=int(probe_config["samples"]),
            max_steps=2048,
            on_progress=on_progress,
        )
        artifact = _save_semantic_probe_artifact(
            args.case_dir / "semantic_probe.pt",
            policy=policy,
            semantic_target=target,
            collection={
                **collection,
                "num_envs": args.num_envs,
                "seed": seed,
            },
            context=context,
            torch=torch,
        )
        _write_progress(
            args,
            started=started,
            state="semantic_probe_artifact_done",
            transitions=collection["control_steps"] * args.num_envs,
            contact_bearing_transitions=collection["samples"],
            control_steps=collection["control_steps"],
            samples=collection["samples"],
            artifact=artifact,
        )
        return {
            "case": "semantic_probe",
            "episode_horizon": 508,
            "collection": {
                **collection,
                "num_envs": args.num_envs,
                "seed": seed,
            },
            "artifact": artifact,
        }
    finally:
        env.close()


def _run_learning(
    args: argparse.Namespace, *, context: Mapping[str, Any], started: float
) -> dict[str, Any]:
    import torch

    from somaforce_cross.learning.acceptance import (
        gradient_summary,
        validate_learning_gates,
    )
    from somaforce_cross.learning.runner import make_phase5_runner

    run = context["acceptance"].payload["run"]
    case_dir = args.case_dir
    _write_progress(args, started=started, state="env_build_start")
    _seed_everything(torch, int(run["train_seed"]))
    env = _build_environment(args, seed=int(run["train_seed"]))
    try:
        _write_progress(args, started=started, state="env_build_done")
        probe = _load_semantic_probe_artifact(
            args.semantic_probe_artifact, context=context, torch=torch
        )
        probe_policy = probe["policy"]
        probe_target = probe["semantic_target"]
        probe_collection = probe["collection"]
        _write_progress(
            args,
            started=started,
            state="probe_load_done",
            probe_artifact=str(args.semantic_probe_artifact),
            probe_artifact_sha256=probe["sha256"],
        )
        runner = make_phase5_runner(env, config_path=args.phase5_config)
        iteration_offset = 0
        transition_offset = 0
        if args.resume is not None:
            (
                iteration_offset,
                transition_offset,
                probe_policy,
                probe_target,
            ) = _restore_checkpoint(
                args.resume, runner=runner, context=context, torch=torch
            )
        if iteration_offset > int(run["iterations"]) or transition_offset > int(
            run["transitions"]
        ):
            raise ValueError("resume checkpoint exceeds the frozen training budget")
        initial_record = _save_checkpoint_and_evaluation(
            case_dir=case_dir,
            runner=runner,
            iteration=iteration_offset,
            transitions=transition_offset,
            crossed_threshold=transition_offset,
            context=context,
            probe_policy=probe_policy,
            probe_target=probe_target,
            torch=torch,
        )
        evaluations = [initial_record]
        _write_progress(
            args,
            started=started,
            state="initial_checkpoint_done",
            iteration=iteration_offset,
            transitions=transition_offset,
            checkpoint=initial_record["checkpoint"],
            checkpoint_sha256=initial_record["checkpoint_sha256"],
        )
        gradient_records: list[dict[str, float]] = []
        tracker = _ContactTracker(args.num_envs)
        interval = int(
            context["acceptance"].payload["checkpoints"]["interval_transitions"]
        )
        next_threshold = ((transition_offset // interval) + 1) * interval

        def on_iteration(
            iteration: int,
            transitions: int,
            update: Mapping[str, float | bool],
            diagnostics: tuple[dict[str, float], ...],
        ) -> None:
            nonlocal next_threshold
            gradient_records.extend(diagnostics)
            for diagnostic in diagnostics:
                _append_jsonl(
                    case_dir / "gradient_diagnostics.jsonl",
                    {"iteration": iteration, "transitions": transitions, **diagnostic},
                )
            _write_progress(
                args,
                started=started,
                state="iteration_done",
                iteration=iteration,
                transitions=transitions,
                contact_bearing_transitions=tracker.transitions,
            )
            while transitions >= next_threshold:
                evaluations.append(
                    _save_checkpoint_and_evaluation(
                        case_dir=case_dir,
                        runner=runner,
                        iteration=iteration,
                        transitions=transitions,
                        crossed_threshold=next_threshold,
                        context=context,
                        probe_policy=probe_policy,
                        probe_target=probe_target,
                        torch=torch,
                    )
                )
                next_threshold += interval

        remaining_iterations = int(run["iterations"]) - iteration_offset
        training = runner.run(
            iterations=remaining_iterations,
            iteration_offset=iteration_offset,
            transition_offset=transition_offset,
            step_observer=tracker.observe,
            iteration_observer=on_iteration,
        )
        if training["transition_count"] != int(run["transitions"]):
            raise AssertionError(
                "learning run transition count differs from the frozen total"
            )
        if not gradient_records:
            raise AssertionError(
                "learning run had no contact-bearing gradient diagnostics"
            )
        gradients = gradient_summary(gradient_records)
        final_record = evaluations[-1]
        learning_metrics = {
            "training_contact_transitions": tracker.transitions,
            "initial_probe": initial_record["semantic_probe"],
            "recent_probe": [item["semantic_probe"] for item in evaluations[-3:]],
            "final_probe": final_record["semantic_probe"],
            "gradients": gradients,
        }
        learning_gates = validate_learning_gates(
            learning_metrics, context["acceptance"]
        )
        return {
            "case": "learning",
            "episode_horizon": 508,
            "probe_collection": probe_collection,
            "training": training,
            "contact": tracker.result(),
            "gradients": gradients,
            "evaluations": evaluations,
            "learning_gates": learning_gates,
            "final_checkpoint": {
                "path": final_record["checkpoint"],
                "sha256": final_record["checkpoint_sha256"],
            },
        }
    finally:
        env.close()


def _new_policy_from_checkpoint(checkpoint: Path, *, device: str, torch: Any) -> Any:
    from somaforce_cross.learning.actor_critic import ResidualActorCritic

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = ResidualActorCritic().to(device)
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval()
    return policy


def _run_episodes(
    env: Any,
    policy: Any,
    *,
    episodes: int,
    zero_residual: bool,
    action_metrics: _ActionResidualMetrics | None = None,
    on_progress: Any | None = None,
) -> list[dict[str, Any]]:
    import torch

    from somaforce_cross.learning.runner import record_policy_semantics

    if env.episode_metric_log is None:
        raise AssertionError("residual C1 evaluation must expose episode metrics")
    observations, _ = env.reset()
    records: list[dict[str, Any]] = []
    known = len(env.episode_metric_log._episodes)
    max_steps = episodes * 508 * 2
    for _ in range(max_steps):
        target = observations["semantic_target"]
        with torch.inference_mode():
            policy_raw = policy.act_inference(observations)
            record_policy_semantics(env, policy, observations)
        raw = (
            torch.zeros(env.num_envs, 23, device=env.device, dtype=torch.float32)
            if zero_residual
            else policy_raw.clone()
        )
        observations, rewards, _, _, _ = env.step(raw)
        if not torch.isfinite(rewards).all():
            raise FloatingPointError("non-finite deterministic evaluation reward")
        if action_metrics is not None:
            action_metrics.observe(env, raw, target)
        new = env.episode_metric_log._episodes[known:]
        if new:
            records.extend(new)
            known += len(new)
            if on_progress is not None:
                on_progress(len(records))
            if len(records) >= episodes:
                return records[:episodes]
    raise AssertionError(
        "deterministic evaluation did not finish the required episodes"
    )


def _nominal_indices(env: Any, *, start: int = 0) -> Any:
    import torch

    if env.mismatch_sampler is None:
        raise AssertionError("nominal pairing requires the C1 mismatch sampler")
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    chosen = torch.empty_like(env_ids)
    unresolved = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    for candidate in range(start, start + 4096):
        indices = torch.full_like(env_ids, candidate)
        sample = env.mismatch_sampler.sample(
            task="push_door_hand", stage="C1", env_ids=env_ids, episode_indices=indices
        )
        take = unresolved & sample.nominal
        chosen[take] = candidate
        unresolved &= ~take
        if not unresolved.any():
            return chosen
    raise AssertionError(
        "could not deterministically construct nominal paired episodes"
    )


def _run_nominal_batch(
    env: Any,
    policy: Any,
    *,
    zero_residual: bool,
    episode_indices: Any,
    on_progress: Any | None = None,
) -> list[dict[str, Any]]:
    import torch

    from somaforce_cross.learning.runner import record_policy_semantics

    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.episode_index.copy_(episode_indices)
    env._reset_idx(ids)
    if env.episode_metric_log is None:
        raise AssertionError("nominal pairing requires episode metric logging")
    wanted = {int(value) for value in env._episode_seed.tolist()}
    records: dict[int, dict[str, Any]] = {}
    known = len(env.episode_metric_log._episodes)
    reported = 0
    observations = env._last_observations
    for _ in range(1016):
        with torch.inference_mode():
            policy_raw = policy.act_inference(observations)
            record_policy_semantics(env, policy, observations)
        raw = (
            torch.zeros(env.num_envs, 23, device=env.device, dtype=torch.float32)
            if zero_residual
            else policy_raw.clone()
        )
        observations, rewards, _, _, _ = env.step(raw)
        if not torch.isfinite(rewards).all():
            raise FloatingPointError("non-finite paired nominal reward")
        for record in env.episode_metric_log._episodes[known:]:
            seed = int(record["seed"])
            if seed in wanted:
                if not record["nominal"]:
                    raise AssertionError(
                        "paired evaluation reset did not retain nominal rows"
                    )
                records[seed] = record
        known = len(env.episode_metric_log._episodes)
        if on_progress is not None and len(records) > reported:
            reported = len(records)
            on_progress(reported)
        if set(records) == wanted:
            return [records[int(seed)] for seed in sorted(wanted)]
    raise AssertionError("paired nominal episode batch did not complete")


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("median requires at least one value")
    midpoint = count // 2
    return (
        ordered[midpoint]
        if count % 2
        else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
    )


def _run_final_evaluation(
    args: argparse.Namespace,
    *,
    context: Mapping[str, Any],
    started: float,
    resource_state: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from somaforce_cross.learning.acceptance import validate_final_evaluation

    if args.checkpoint is None or not args.checkpoint.is_file():
        raise ValueError("final evaluation requires the learning final checkpoint")
    run = context["acceptance"].payload["run"]
    _seed_everything(torch, int(run["evaluation_seed"]))
    policy = _new_policy_from_checkpoint(
        args.checkpoint, device=args.device, torch=torch
    )
    c1_total = int(context["acceptance"].payload["evaluation"]["c1_episodes"])
    _write_final_evaluation_progress(
        args,
        started=started,
        state="c1_evaluation_start",
        phase="c1",
        episodes_completed=0,
        episodes_total=c1_total,
    )
    env = _build_environment(args, seed=int(run["evaluation_seed"]))
    try:
        action_metrics = _ActionResidualMetrics(env.num_envs)
        c1_records = _run_episodes(
            env,
            policy,
            episodes=c1_total,
            zero_residual=False,
            action_metrics=action_metrics,
            on_progress=lambda completed: _write_final_evaluation_progress(
                args,
                started=started,
                state="c1_evaluation_progress",
                phase="c1",
                episodes_completed=completed,
                episodes_total=c1_total,
            ),
        )
        outcome = {
            "episodes": len(c1_records),
            "success": sum(bool(record["outcome"]["success"]) for record in c1_records)
            / len(c1_records),
            "failure": sum(bool(record["outcome"]["failure"]) for record in c1_records)
            / len(c1_records),
            "episode_invalid": sum(
                bool(record["outcome"]["episode_invalid"]) for record in c1_records
            ),
        }
        _write_final_evaluation_progress(
            args,
            started=started,
            state="c1_evaluation_done",
            phase="c1",
            episodes_completed=len(c1_records),
            episodes_total=c1_total,
        )

        required = int(
            context["acceptance"].payload["evaluation"]["nominal_paired_episodes"]
        )
        batch_size = int(context["acceptance"].payload["run"]["num_envs"])
        batches_total = (required + batch_size - 1) // batch_size
        _write_final_evaluation_progress(
            args,
            started=started,
            state="nominal_pair_evaluation_start",
            phase="nominal_paired",
            episodes_completed=0,
            episodes_total=required,
            batches_total=batches_total,
        )
        trained_records: list[dict[str, Any]] = []
        paired_indices: list[Any] = []
        nominal_cursor = 0
        while len(trained_records) < required:
            batch_index = len(paired_indices) + 1
            _write_final_evaluation_progress(
                args,
                started=started,
                state="nominal_pair_batch_constructing",
                phase="nominal_paired",
                episodes_completed=len(trained_records),
                episodes_total=required,
                batch_index=batch_index,
                batches_completed=batch_index - 1,
                batches_total=batches_total,
            )
            indices = _nominal_indices(env, start=nominal_cursor)
            paired_indices.append(indices.cpu())
            nominal_cursor = int(indices.max().item()) + 1
            _write_final_evaluation_progress(
                args,
                started=started,
                state="nominal_pair_batch_constructed",
                phase="nominal_paired",
                episodes_completed=len(trained_records),
                episodes_total=required,
                batch_index=batch_index,
                batches_completed=batch_index - 1,
                batches_total=batches_total,
            )
            base_completed = len(trained_records)
            batch_records = _run_nominal_batch(
                env,
                policy,
                zero_residual=False,
                episode_indices=indices,
                on_progress=lambda completed: _write_final_evaluation_progress(
                    args,
                    started=started,
                    state="nominal_pair_batch_progress",
                    phase="nominal_paired",
                    episodes_completed=base_completed + completed,
                    episodes_total=required,
                    batch_index=batch_index,
                    batches_completed=batch_index - 1,
                    batches_total=batches_total,
                ),
            )
            trained_records.extend(batch_records)
            _write_final_evaluation_progress(
                args,
                started=started,
                state="nominal_pair_batch_done",
                phase="nominal_paired",
                episodes_completed=len(trained_records),
                episodes_total=required,
                batch_index=batch_index,
                batches_completed=batch_index,
                batches_total=batches_total,
            )
        _write_final_evaluation_progress(
            args,
            started=started,
            state="nominal_pair_evaluation_done",
            phase="nominal_paired",
            episodes_completed=len(trained_records),
            episodes_total=required,
            batches_completed=len(paired_indices),
            batches_total=batches_total,
        )

        _write_final_evaluation_progress(
            args,
            started=started,
            state="scaffold_only_pair_evaluation_start",
            phase="scaffold_only_paired",
            episodes_completed=0,
            episodes_total=required,
            batches_total=len(paired_indices),
        )
        scaffold_records: list[dict[str, Any]] = []
        for batch_index, indices in enumerate(paired_indices, start=1):
            _write_final_evaluation_progress(
                args,
                started=started,
                state="scaffold_only_pair_batch_start",
                phase="scaffold_only_paired",
                episodes_completed=len(scaffold_records),
                episodes_total=required,
                batch_index=batch_index,
                batches_completed=batch_index - 1,
                batches_total=len(paired_indices),
            )
            base_completed = len(scaffold_records)
            batch_records = _run_nominal_batch(
                env,
                policy,
                zero_residual=True,
                episode_indices=indices.to(env.device),
                on_progress=lambda completed: _write_final_evaluation_progress(
                    args,
                    started=started,
                    state="scaffold_only_pair_batch_progress",
                    phase="scaffold_only_paired",
                    episodes_completed=base_completed + completed,
                    episodes_total=required,
                    batch_index=batch_index,
                    batches_completed=batch_index - 1,
                    batches_total=len(paired_indices),
                ),
            )
            scaffold_records.extend(batch_records)
            _write_final_evaluation_progress(
                args,
                started=started,
                state="scaffold_only_pair_batch_done",
                phase="scaffold_only_paired",
                episodes_completed=len(scaffold_records),
                episodes_total=required,
                batch_index=batch_index,
                batches_completed=batch_index,
                batches_total=len(paired_indices),
            )
        _write_final_evaluation_progress(
            args,
            started=started,
            state="scaffold_only_pair_evaluation_done",
            phase="scaffold_only_paired",
            episodes_completed=len(scaffold_records),
            episodes_total=required,
            batches_completed=len(paired_indices),
            batches_total=len(paired_indices),
        )
        trained_records = trained_records[:required]
        scaffold_records = scaffold_records[:required]
        trained_progress = [
            float(record["raw_reward_sums"]["progress"]) for record in trained_records
        ]
        scaffold_progress = [
            float(record["raw_reward_sums"]["progress"]) for record in scaffold_records
        ]
        median_trained = _median(trained_progress)
        median_scaffold = _median(scaffold_progress)
        if median_scaffold <= 0.0:
            raise AssertionError(
                "scaffold-only paired nominal median progress is nonpositive"
            )
        retention = {
            "episodes": required,
            "median_progress": median_trained,
            "scaffold_only_median_progress": median_scaffold,
            "trained_to_scaffold_ratio": median_trained / median_scaffold,
        }
        final_metrics = {
            "outcome": outcome,
            "retention": retention,
            "action": action_metrics.result(),
            "residual": action_metrics.result(),
        }
        _persist_final_metrics(
            args,
            final_metrics=final_metrics,
            context=context,
            checkpoint=args.checkpoint,
        )
        gates = validate_final_evaluation(final_metrics, context["acceptance"])
        return {"case": "final_evaluation", **final_metrics, "final_gates": gates}
    finally:
        env.close()
        resource_state["env_already_closed"] = True


def _worker_main(argv: list[str]) -> int:
    args = _worker_parser().parse_args(argv)
    _validate_paths(args)
    from isaaclab.app import AppLauncher

    started = time.monotonic()
    launcher: Any | None = None
    holder: dict[str, Any] = {}
    status = "error"
    try:
        launcher = AppLauncher(args)
        _write_progress(args, started=started, state="app_launched")
        repo_path = str(REPO_ROOT)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        context = _contract_context(args)
        if (
            context["contracts"]["phase4_canonical_sha256"]
            != context["acceptance"].payload["bindings"]["phase4_canonical_sha256"]
        ):
            raise AssertionError("Phase 4 canonical contract binding changed")
        if args.case == "contact_probe":
            payload = _run_contact_probe(args, context=context)
        elif args.case == "semantic_probe":
            payload = _run_semantic_probe(args, context=context, started=started)
        elif args.case == "learning":
            payload = _run_learning(args, context=context, started=started)
        else:
            payload = _run_final_evaluation(
                args,
                context=context,
                started=started,
                resource_state=holder,
            )
        holder["env_already_closed"] = True
        result = {
            "status": "ok",
            "case": args.case,
            "contracts": context["contracts"],
            "source_sha256": context["source_hashes"],
            "source_allowlist_sha256": context["source_allowlist_sha256"],
            "git": _git_state(),
            "payload": payload,
        }
        status = "ok"
    except BaseException as exc:
        result = {
            "status": "error",
            "case": args.case,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "git": _git_state(),
        }
    _persist_worker_result(args.result_json, result)
    _close_worker_resources(
        env=holder.get("env"),
        launcher=launcher,
        env_already_closed=bool(holder.get("env_already_closed")),
    )
    return 0 if status == "ok" else 1


def _worker_command(
    args: argparse.Namespace,
    *,
    case: str,
    num_envs: int,
    iterations: int,
    base: Path,
    checkpoint: Path | None = None,
    semantic_probe_artifact: Path | None = None,
) -> list[str]:
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
        "--case",
        case,
        "--num-envs",
        str(num_envs),
        "--iterations",
        str(iterations),
        "--result-json",
        str(base / "result.json"),
        "--case-dir",
        str(base),
        "--phase4-contract",
        str(PHASE4_CONTRACT),
        "--phase5-config",
        str(PHASE5_CONFIG),
        "--acceptance-config",
        str(ACCEPTANCE_CONFIG),
        "--headless",
        "--device",
        "cuda:0",
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    if semantic_probe_artifact is not None:
        command.extend(("--semantic-probe-artifact", str(semantic_probe_artifact)))
    return command


def _suite_main(argv: list[str]) -> int:
    args = _suite_parser().parse_args(argv)
    run_id = args.run_id or time.strftime("run_%Y%m%dT%H%M%S") + f"_{os.getpid()}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    summary_path = run_dir / "summary.json"
    try:
        if args.only_case == "semantic_probe":
            definitions = (
                ("semantic_probe", "semantic_probe", 64, 0, args.contact_timeout_s),
            )
        else:
            definitions = (
                (
                    "contact_probe_2env_16iter",
                    "contact_probe",
                    2,
                    16,
                    args.contact_timeout_s,
                ),
                (
                    "contact_probe_64env_16iter",
                    "contact_probe",
                    64,
                    16,
                    args.contact_timeout_s,
                ),
                ("semantic_probe", "semantic_probe", 64, 0, args.contact_timeout_s),
                (
                    "learning_64env_2442iter",
                    "learning",
                    64,
                    2442,
                    args.learning_timeout_s,
                ),
            )
        semantic_probe_artifact: Path | None = None
        for name, case, num_envs, iterations, timeout_s in definitions:
            base = run_dir / name
            wrapper = _run_worker_case(
                command=_worker_command(
                    args,
                    case=case,
                    num_envs=num_envs,
                    iterations=iterations,
                    base=base,
                    semantic_probe_artifact=semantic_probe_artifact,
                ),
                base=base,
                timeout_s=timeout_s,
                label=f"phase5-learning-{name}",
            )
            result = _read_json(base / "result.json")
            if result is None:
                raise AssertionError("passed worker result disappeared")
            cases.append({"name": name, "wrapper": wrapper, "result": result})
            if case == "semantic_probe":
                artifact_path = (
                    result.get("payload", {}).get("artifact", {}).get("path")
                )
                if (
                    not isinstance(artifact_path, str)
                    or not Path(artifact_path).is_file()
                ):
                    raise AssertionError("semantic probe artifact disappeared")
                semantic_probe_artifact = Path(artifact_path)
        if args.only_case == "semantic_probe":
            summary = {
                "status": "ok",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "cases": cases,
                "git": _git_state(),
            }
            _atomic_json(summary_path, summary)
            print(
                json.dumps(
                    {"status": "ok", "summary": str(summary_path)}, sort_keys=True
                )
            )
            return 0
        final_checkpoint = Path(
            cases[-1]["result"]["payload"]["final_checkpoint"]["path"]
        )
        evaluation_base = run_dir / "final_evaluation_256c1_128nominal"
        wrapper = _run_worker_case(
            command=_worker_command(
                args,
                case="final_evaluation",
                num_envs=64,
                iterations=0,
                base=evaluation_base,
                checkpoint=final_checkpoint,
            ),
            base=evaluation_base,
            timeout_s=args.evaluation_timeout_s,
            label="phase5-learning-final-evaluation",
        )
        result = _read_json(evaluation_base / "result.json")
        if result is None:
            raise AssertionError("passed final evaluation result disappeared")
        cases.append({"name": "final_evaluation", "wrapper": wrapper, "result": result})
        summary = {
            "status": "ok",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "cases": cases,
            "git": _git_state(),
        }
        _atomic_json(summary_path, summary)
        print(
            json.dumps({"status": "ok", "summary": str(summary_path)}, sort_keys=True)
        )
        return 0
    except BaseException as exc:
        _atomic_json(
            summary_path,
            {
                "status": "error",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "cases": cases,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "git": _git_state(),
            },
        )
        raise


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
