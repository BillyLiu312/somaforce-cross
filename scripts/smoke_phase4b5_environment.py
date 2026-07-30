#!/usr/bin/env python3
"""Bounded Isaac evidence for the non-C0 Phase 4 environment path."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from smoke_phase4b5_runtime import (
    NUMERIC_CONTRACT,
    TASKS,
    _atomic_json,
    _persist_worker_result,
    _read_json,
    _run_worker_case,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ENV_COUNTS = (2, 64)


def _worker_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker",), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--num-envs", choices=RUNTIME_ENV_COUNTS, type=int, required=True
    )
    parser.add_argument("--selected-env-ids", type=int, nargs="+", required=True)
    parser.add_argument("--stage", choices=("C1",), default="C1")
    parser.add_argument(
        "--runtime-mode", choices=("scaffold_only", "residual"), default="residual"
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--contract", type=Path, default=NUMERIC_CONTRACT)
    parser.add_argument("--result-json", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("suite",), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/phase4b5_environment")
    )
    return parser


def _expected_selected_ids(num_envs: int) -> tuple[int, ...]:
    if num_envs == 2:
        return (0,)
    if num_envs == 64:
        return (0, 7, 63)
    raise ValueError("only 2-env and 64-env cases are supported")


def _validate_case_args(args: argparse.Namespace) -> None:
    selected = tuple(args.selected_env_ids)
    if selected != _expected_selected_ids(args.num_envs):
        raise ValueError(
            f"selected env ids for {args.num_envs} envs must be "
            f"{_expected_selected_ids(args.num_envs)}"
        )
    if args.contract.expanduser().resolve() != NUMERIC_CONTRACT.resolve():
        raise ValueError("worker requires configs/phase4b5_numeric_contract.json")
    if args.stage != "C1":
        raise ValueError("worker requires the explicit C1 stage")


def _require_finite(
    name: str, value: Any, *, shape: tuple[int, ...], torch: Any
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
    ):
        raise AssertionError(f"{name} is not finite float32 with shape {shape}")


def _raw_policy(env: Any, torch: Any) -> Any:
    with torch.inference_mode():
        assembly = env.semantic_pipeline.assemble_policy_observation(
            env.wrist_history.storage,
            env.adapter.proprio(),
            env.nominal_action_history.storage,
            env.executed_action_history.storage[:, :, 0],
        )
    return assembly.bundle.flatten()


def _expected_critic(env: Any, *, policy: Any, torch: Any) -> Any:
    signals = env.adapter.progress_signals()
    batch = env.num_envs
    phase = (
        env.adapter.reference_step.float() / env.adapter.reference.length
    ).unsqueeze(1)
    progress_state = torch.cat(
        (phase, signals.progress, signals.progress_delta, signals.success), dim=-1
    )
    contact_state = torch.cat(
        (
            env._clean_wrench.reshape(batch, 12),
            signals.contact_truth,
            signals.expected_contact,
            env.adapter.support_contact_count(),
        ),
        dim=-1,
    )
    support = signals.contact_truth.new_zeros(batch, 2)
    contact_forces = env.contacts.data.net_forces_w.index_select(
        1, env.adapter.contact_support_ids
    )
    support.copy_((torch.linalg.vector_norm(contact_forces, dim=-1) > 1.0).float())
    stability_state = torch.cat(
        (
            env.robot.data.root_link_pos_w[:, 2:3],
            env.robot.data.projected_gravity_b,
            env.robot.data.root_lin_vel_b,
            support,
        ),
        dim=-1,
    )
    fields = env.normalizer.normalize_critic(
        {
            "policy": policy,
            "object_state": env.adapter.build_object_state(),
            "physics_mismatch": env.parameter_store.physics_mismatch,
            "scaffold_mismatch": env.parameter_store.scaffold_mismatch,
            "sensor_mismatch": env.parameter_store.sensor_mismatch,
            "progress_state": progress_state,
            "contact_state": contact_state,
            "stability_state": stability_state,
        },
        task=env.task_spec.task,
    )
    return torch.cat(
        (
            fields["policy"],
            fields["object_state"],
            fields["physics_mismatch"],
            fields["scaffold_mismatch"],
            fields["sensor_mismatch"],
            fields["progress_state"],
            fields["contact_state"],
            fields["stability_state"],
        ),
        dim=-1,
    )


def _validate_observations(env: Any, observations: dict[str, Any], torch: Any) -> None:
    expected = {
        "policy": (env.num_envs, 668),
        "critic": (env.num_envs, 845),
        "semantic_target": (env.num_envs, 31),
    }
    for name, shape in expected.items():
        _require_finite(name, observations[name], shape=shape, torch=torch)
    raw_policy = _raw_policy(env, torch)
    expected_policy = env.normalizer.normalize_policy(raw_policy)
    if not torch.equal(observations["policy"], expected_policy):
        raise AssertionError("policy output did not use FixedFieldNormalizer")
    expected_critic = _expected_critic(env, policy=raw_policy, torch=torch)
    if not torch.equal(observations["critic"], expected_critic):
        raise AssertionError("critic output did not use FixedFieldNormalizer")


def _validate_raw_observations(
    env: Any, observations: dict[str, Any], torch: Any
) -> None:
    expected = {
        "policy": (env.num_envs, 668),
        "critic": (env.num_envs, 845),
        "semantic_target": (env.num_envs, 31),
    }
    for name, shape in expected.items():
        _require_finite(name, observations[name], shape=shape, torch=torch)
    if any(
        value is not None
        for value in (
            env.normalizer,
            env.reward_manager,
            env.episode_termination,
            env.episode_metric_log,
            env.curriculum,
        )
    ):
        raise AssertionError("scaffold_only enabled residual runtime components")


def _recomputed_reward(env: Any, torch: Any) -> Any:
    signals = env.adapter.progress_signals()
    return env.reward_manager.compute(
        progress_delta=signals.progress_delta.squeeze(1),
        expected_contact=signals.expected_contact,
        contact_truth=signals.contact_truth,
        wrench=env._observed_wrench,
        stability_margin=signals.stability_margin.squeeze(1),
        delta_safe=env._delta_safe,
        authority=env._authority,
        previous_delta_safe=env._previous_delta_safe,
        success_latched=env.episode_termination.success,
        failure_latched=env.episode_termination.failure,
        nonfinite=env._last_nonfinite,
        terminated=env._last_terminated,
        time_outs=env._last_time_outs,
    ).total


def _assert_no_reset_advance() -> None:
    source = (REPO_ROOT / "somaforce_cross/envs/residual_env.py").read_text(
        encoding="utf-8"
    )
    reset = source[
        source.index("def _reset_scaffold_only_idx") : source.index(
            "def _owned_sensor_state"
        )
    ]
    if "self.sim.step" in reset or "self.sim.forward" in reset:
        raise AssertionError("selected reset contains a global simulator advance")


def _selected_reset_evidence(env: Any, selected: Any, torch: Any) -> dict[str, Any]:
    if env.episode_metric_log is None:
        raise AssertionError("non-C0 environment has no EpisodeMetricLog")
    mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    mask[selected] = False
    env._episode_active[selected] = True
    env._last_time_outs[selected] = True
    env._episode_reward_steps[selected] = 1
    env._episode_return[selected] = 1.0
    env.episode_termination.success[selected] = True
    for values in (env._episode_raw_sums, env._episode_weighted_sums):
        for value in values.values():
            value[selected] = 1.0
    before = {
        name: value.clone() for name, value in env.environment_owned_state().items()
    }
    records_before = len(env.episode_metric_log._episodes)
    env._reset_idx(selected)
    after = env.environment_owned_state()
    changed = [
        name
        for name, value in before.items()
        if not torch.equal(value[mask], after[name][mask])
    ]
    if changed:
        raise AssertionError(f"unselected rows changed: {changed}")
    if len(env.episode_metric_log._episodes) != records_before + selected.numel():
        raise AssertionError("completed selected episodes were not recorded")
    cleared = (
        not env.episode_termination.success[selected].any()
        and not env.episode_termination.failure[selected].any()
        and not env.episode_termination.episode_invalid[selected].any()
        and torch.count_nonzero(env._episode_reward_steps[selected]) == 0
        and torch.count_nonzero(env._episode_return[selected]) == 0
        and torch.count_nonzero(env._episode_wrench_history[selected]) == 0
        and torch.count_nonzero(env._episode_force_rate_history[selected]) == 0
        and all(
            torch.count_nonzero(value[selected]) == 0
            for values in (env._episode_raw_sums, env._episode_weighted_sums)
            for value in values.values()
        )
    )
    if not cleared:
        raise AssertionError("selected reset did not clear lifecycle state")
    return {
        "selected_env_ids": selected.cpu().tolist(),
        "unselected_envs": int(mask.sum().item()),
        "unselected_bitwise_unchanged": True,
        "completed_records_added": int(selected.numel()),
        "latch_reward_metric_reset": True,
    }


def _scaffold_only_reset_evidence(
    env: Any, selected: Any, torch: Any
) -> dict[str, Any]:
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
    if torch.count_nonzero(env._authority) != 0:
        raise AssertionError("scaffold_only authority is not zero after reset")
    return {
        "selected_env_ids": selected.cpu().tolist(),
        "unselected_envs": int(mask.sum().item()),
        "unselected_bitwise_unchanged": True,
        "residual_components_disabled": True,
    }


def _environment_probe(
    args: argparse.Namespace, holder: dict[str, Any]
) -> dict[str, Any]:
    import torch

    from somaforce_cross.envs.residual_env import SomaForceResidualEnv
    from somaforce_cross.envs.residual_env_cfg import SomaForceResidualEnvCfg
    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
    from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg

    _validate_case_args(args)
    torch.manual_seed(args.seed)
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
    profile.runtime_mode = args.runtime_mode
    profile.scaffold_stage = "C1"
    profile.numeric_contract_path = str(NUMERIC_CONTRACT)
    cfg.__post_init__()

    env = SomaForceResidualEnv(cfg)
    holder["env"] = env
    observations, _ = env.reset()
    if args.runtime_mode == "residual":
        _validate_observations(env, observations, torch)
        probe = torch.zeros(env.num_envs, 23, device=env.device)
    else:
        _validate_raw_observations(env, observations, torch)
        probe = torch.ones(env.num_envs, 23, device=env.device)
    observations, rewards, terminated, time_outs, _ = env.step(probe)
    if args.runtime_mode == "residual":
        _validate_observations(env, observations, torch)
    else:
        _validate_raw_observations(env, observations, torch)
    _require_finite("reward", rewards, shape=(env.num_envs,), torch=torch)
    if terminated.dtype != torch.bool or time_outs.dtype != torch.bool:
        raise AssertionError("done outputs must be bool tensors")
    if args.runtime_mode == "residual":
        if not torch.equal(rewards, _recomputed_reward(env, torch)):
            raise AssertionError(
                "environment reward differs from manager recomputation"
            )
    elif not torch.equal(rewards, torch.zeros_like(rewards)):
        raise AssertionError("scaffold_only reward is not bitwise zero")
    if args.runtime_mode == "scaffold_only" and torch.count_nonzero(env._authority):
        raise AssertionError("scaffold_only authority is not bitwise zero")
    if not torch.equal(env._a_total, env._step_a_nom):
        raise AssertionError("zero-authority runtime changed a_total")
    selected = torch.tensor(args.selected_env_ids, device=env.device, dtype=torch.long)
    selected_reset = (
        _selected_reset_evidence(env, selected, torch)
        if args.runtime_mode == "residual"
        else _scaffold_only_reset_evidence(env, selected, torch)
    )
    _assert_no_reset_advance()
    return {
        "command": sys.argv,
        "task": args.task,
        "stage": "C1",
        "runtime_mode": args.runtime_mode,
        "contract": str(NUMERIC_CONTRACT),
        "num_envs": args.num_envs,
        "policy_shape": list(observations["policy"].shape),
        "critic_shape": list(observations["critic"].shape),
        "semantic_target_shape": list(observations["semantic_target"].shape),
        "finite": {
            "policy": True,
            "critic": True,
            "semantic_target": True,
            "reward": True,
            "terminated": True,
            "time_outs": True,
        },
        "fixed_normalizer": {
            "policy": args.runtime_mode == "residual",
            "critic": args.runtime_mode == "residual",
        },
        "reward_recomputed_exact": args.runtime_mode == "residual",
        "zero_authority_a_total_eq_a_nom": True,
        "selected_reset": selected_reset,
        "reset_has_no_global_advance": True,
    }


def _worker_main(argv: list[str]) -> int:
    parser = _worker_parser()
    args = parser.parse_args(argv)
    from isaaclab.app import AppLauncher

    launcher: Any | None = None
    holder: dict[str, Any] = {}
    status = "error"
    try:
        launcher = AppLauncher(args)
        repo = str(REPO_ROOT)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        result = _environment_probe(args, holder)
        status = "ok"
    except BaseException as exc:
        result = {
            "command": sys.argv,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    _persist_worker_result(args.result_json, {"status": status, **result})
    env = holder.get("env")
    if env is not None:
        env.close()
        print("ENV_CLOSED", flush=True)
    if launcher is not None:
        launcher.app.close()
    return 0 if status == "ok" else 1


def _suite_main(argv: list[str]) -> int:
    args = _suite_parser().parse_args(argv)
    run_dir = args.output_dir / time.strftime("run_%Y%m%dT%H%M%S")
    evidence: dict[str, Any] = {
        "scaffold_only_guard": None,
        "cases": [],
        "status": "error",
    }
    summary_path = run_dir / "summary.json"
    try:
        guard_dir = run_dir / "scaffold_only_guard"
        guard_command = [
            args.python,
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--task",
            "push_door_hand",
            "--num-envs",
            "2",
            "--selected-env-ids",
            "0",
            "--stage",
            "C1",
            "--runtime-mode",
            "scaffold_only",
            "--contract",
            str(NUMERIC_CONTRACT),
            "--result-json",
            str(guard_dir / "result.json"),
            "--headless",
            "--device",
            "cuda:0",
        ]
        guard_wrapper = _run_worker_case(
            command=guard_command,
            result_json=guard_dir / "result.json",
            log=guard_dir / "worker.log",
            wrapper_json=guard_dir / "wrapper.json",
            timeout_s=args.timeout_s,
            label="scaffold-only-guard-push_door_hand-2",
        )
        guard_result = _read_json(guard_dir / "result.json")
        if guard_result is None:
            raise RuntimeError("scaffold_only guard did not persist result.json")
        evidence["scaffold_only_guard"] = {
            "wrapper": guard_wrapper,
            "result": guard_result,
        }
        for task in TASKS:
            for num_envs in RUNTIME_ENV_COUNTS:
                selected = _expected_selected_ids(num_envs)
                case_dir = run_dir / f"{task}_{num_envs}"
                command = [
                    args.python,
                    str(Path(__file__).resolve()),
                    "--mode",
                    "worker",
                    "--task",
                    task,
                    "--num-envs",
                    str(num_envs),
                    "--selected-env-ids",
                    *(str(value) for value in selected),
                    "--stage",
                    "C1",
                    "--runtime-mode",
                    "residual",
                    "--contract",
                    str(NUMERIC_CONTRACT),
                    "--result-json",
                    str(case_dir / "result.json"),
                    "--headless",
                    "--device",
                    "cuda:0",
                ]
                wrapper = _run_worker_case(
                    command=command,
                    result_json=case_dir / "result.json",
                    log=case_dir / "worker.log",
                    wrapper_json=case_dir / "wrapper.json",
                    timeout_s=args.timeout_s,
                    label=f"environment-{task}-{num_envs}",
                )
                result = _read_json(case_dir / "result.json")
                if result is None:
                    raise RuntimeError("worker did not persist result.json")
                evidence["cases"].append(
                    {
                        "task": task,
                        "num_envs": num_envs,
                        "wrapper": wrapper,
                        "result": result,
                    }
                )
        evidence["status"] = "ok"
        evidence["summary"] = {
            "case_count": len(evidence["cases"]),
            "scaffold_only_guard_passed": True,
            "all_exit_zero": True,
            "all_signal_null": True,
            "all_timed_out_false": True,
            "all_parent_verified_exit0": True,
            "all_status_ok": True,
        }
    except BaseException as exc:
        evidence["failure"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "completed_cases": len(evidence["cases"]),
        }
    _atomic_json(summary_path, evidence)
    print(
        json.dumps({"status": evidence["status"], "summary": str(summary_path)}),
        flush=True,
    )
    return 0 if evidence["status"] == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if "--mode" not in values:
        raise SystemExit("--mode {worker,suite} is required")
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
