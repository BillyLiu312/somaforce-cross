#!/usr/bin/env python3
"""Independent golden/residual traces and full-row Phase 4B3 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = Path("/tmp/somaforce_phase4b3")
ARTIFACT_DIR = REPO_ROOT / "artifacts/scaffolds/hdmi_push_door_hand/v1"
TRACE_FIELDS = (
    "a_nom",
    "raw_residual",
    "authority",
    "a_total",
    "applied_action",
    "joint_target",
    "scaffold_history",
    "nominal_history",
    "executed_history",
    "reference_step",
    "reference_phase",
    "robot_root_state",
    "robot_joint_pos",
    "robot_joint_vel",
    "door_root_state",
    "door_joint_pos",
    "door_joint_vel",
    "door_applied_effort",
    "env_origin",
    "physics_mismatch",
    "reset_seed",
    "finite",
    "terminated",
    "time_outs",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("golden", "residual", "compare"), required=True
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--golden-trace", type=Path)
    parser.add_argument("--residual-trace", type=Path)
    parser.add_argument("--selective-reset-env-ids", type=int, nargs="*")
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--door-friction", type=float, default=0.3)
    parser.add_argument("--door-damping", type=float, default=0.55)
    parser.add_argument("--F-scale", dest="F_scale", type=float, default=1.0)
    parser.add_argument("--M-scale", dest="M_scale", type=float, default=1.0)
    parser.add_argument("--tare-num-samples", type=int, default=1)
    parser.add_argument("--contact-on-threshold", type=float, default=0.8)
    parser.add_argument("--contact-off-threshold", type=float, default=0.4)
    parser.add_argument("--contact-temperature", type=float, default=0.2)
    parser.add_argument("--contact-smoothing-alpha", type=float, default=0.5)
    parser.add_argument(
        "--virtual-ft-axis-quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0)
    )
    parser.add_argument("--scale-error", type=float, default=0.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--drift-rate", type=float, default=0.0)
    parser.add_argument("--drift-noise-std", type=float, default=0.0)
    parser.add_argument("--white-noise-std", type=float, default=0.0)
    parser.add_argument("--sensor-delay-steps", type=int, default=0)
    parser.add_argument("--filter-alpha", type=float, default=0.0)
    parser.add_argument("--force-saturation", type=float, default=1.0e6)
    parser.add_argument("--torque-saturation", type=float, default=1.0e6)
    parser.add_argument("--dropout-probability", type=float, default=0.0)
    parser.add_argument("--ramp-attack-step", type=float, default=0.1)
    parser.add_argument("--ramp-release-step", type=float, default=0.1)
    parser.add_argument("--episode-length-steps", type=int, default=8)
    parser.add_argument("--smoke-reward", type=float, default=0.0)
    parser.add_argument("--parity-atol", type=float, default=1.0e-5)
    parser.add_argument("--parity-rtol", type=float, default=1.0e-5)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("num_envs and steps must be positive")
    if args.steps > args.episode_length_steps:
        raise ValueError("bounded smoke steps must not exceed the episode length")
    if args.physics_dt * args.decimation != args.control_dt:
        raise ValueError("physics/control dt and decimation are inconsistent")
    if args.mode == "compare" and (
        args.golden_trace is None or args.residual_trace is None
    ):
        raise ValueError("compare mode requires --golden-trace and --residual-trace")
    if args.selective_reset_env_ids and args.mode != "residual":
        raise ValueError("selective reset evidence is available in residual mode only")
    if args.selective_reset_env_ids:
        ids = args.selective_reset_env_ids
        if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= args.num_envs:
            raise ValueError(
                "selective reset ids must be unique valid environment rows"
            )


def _profile_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": "push_door_hand",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "physics_dt": args.physics_dt,
        "control_dt": args.control_dt,
        "decimation": args.decimation,
        "delay": args.delay,
        "alpha": args.alpha,
        "door_friction": args.door_friction,
        "door_damping": args.door_damping,
        "F_scale": args.F_scale,
        "M_scale": args.M_scale,
        "tare_num_samples": args.tare_num_samples,
        "contact_on_threshold": args.contact_on_threshold,
        "contact_off_threshold": args.contact_off_threshold,
        "contact_temperature": args.contact_temperature,
        "contact_smoothing_alpha": args.contact_smoothing_alpha,
        "virtual_ft_axis_quat": list(args.virtual_ft_axis_quat),
        "scale_error": args.scale_error,
        "bias": args.bias,
        "drift_rate": args.drift_rate,
        "drift_noise_std": args.drift_noise_std,
        "white_noise_std": args.white_noise_std,
        "sensor_delay_steps": args.sensor_delay_steps,
        "filter_alpha": args.filter_alpha,
        "force_saturation": args.force_saturation,
        "torque_saturation": args.torque_saturation,
        "dropout_probability": args.dropout_probability,
        "ramp_attack_step": args.ramp_attack_step,
        "ramp_release_step": args.ramp_release_step,
        "episode_length_steps": args.episode_length_steps,
        "smoke_reward": args.smoke_reward,
        "parity_atol": args.parity_atol,
        "parity_rtol": args.parity_rtol,
    }


def _config(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    payload = _profile_payload(args)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(encoded).hexdigest()


def _trace_path(args: argparse.Namespace, mode: str, config_hash: str) -> Path:
    if args.trace is not None:
        return args.trace.expanduser().resolve()
    return (
        TRACE_ROOT
        / f"{mode}_{args.num_envs}env_{args.steps}step_{config_hash[:12]}.npz"
    )


def _gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            text=True,
        )
        return int(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _cpu(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    import torch

    result: dict[str, np.ndarray] = {}
    for name, value in snapshot.items():
        if isinstance(value, torch.Tensor):
            result[name] = value.detach().cpu().numpy()
    return result


def _stack_trace(
    snapshots: list[dict[str, np.ndarray]],
    *,
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    arrays = {
        name: np.stack([snapshot[name] for snapshot in snapshots], axis=0)
        for name in snapshots[0]
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return arrays


def _save_trace(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    actual_fields = sorted(name for name in arrays if name != "metadata_json")
    metadata["trace_fields"] = actual_fields
    metadata["trace_shapes"] = {
        name: list(arrays[name].shape[1:]) for name in actual_fields
    }
    metadata["trace_dtypes"] = {name: str(arrays[name].dtype) for name in actual_fields}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def _golden_snapshot(
    runtime: Any, a_nom: Any, config_hash: str, base_seed: int
) -> dict[str, Any]:
    import torch

    action_runtime = runtime.action_runtime
    target = (
        action_runtime.default_joint_pos[:, action_runtime.joint_ids]
        + action_runtime.applied_action * action_runtime.scale
    )
    finite = torch.ones(runtime.num_envs, device=runtime.device, dtype=torch.bool)
    for value in (
        a_nom,
        runtime.robot.data.root_state_w,
        runtime.robot.data.joint_pos,
        runtime.door.data.joint_pos,
    ):
        finite &= torch.isfinite(value).reshape(runtime.num_envs, -1).all(dim=1)
    reference_step = torch.full(
        (runtime.num_envs,),
        runtime.reference_step,
        device=runtime.device,
        dtype=torch.long,
    )
    physics_mismatch = torch.zeros(
        runtime.num_envs, 39, device=runtime.device, dtype=torch.float32
    )
    physics_mismatch[:, 3] = runtime.door_friction
    physics_mismatch[:, 4] = runtime.door_damping
    return {
        "a_nom": a_nom,
        "raw_residual": torch.ones_like(a_nom),
        "authority": torch.zeros_like(a_nom),
        "a_total": a_nom,
        "applied_action": action_runtime.applied_action,
        "joint_target": target,
        "scaffold_history": runtime.history.previous_actions,
        "nominal_history": runtime.history.previous_actions,
        "executed_history": runtime.history.previous_actions,
        "reference_step": reference_step,
        "reference_phase": runtime.history.phase,
        "robot_root_state": runtime.robot.data.root_state_w,
        "robot_joint_pos": runtime.robot.data.joint_pos,
        "robot_joint_vel": runtime.robot.data.joint_vel,
        "door_root_state": runtime.door.data.root_state_w,
        "door_joint_pos": runtime.door.data.joint_pos,
        "door_joint_vel": runtime.door.data.joint_vel,
        "door_applied_effort": runtime.door.data.applied_torque,
        "env_origin": runtime.scene.env_origins,
        "physics_mismatch": physics_mismatch,
        "reset_seed": torch.arange(
            base_seed,
            base_seed + runtime.num_envs,
            device=runtime.device,
            dtype=torch.long,
        ),
        "finite": finite,
        "configuration_hash": config_hash,
    }


def _rewind_golden_initial_state(runtime: Any) -> None:
    """Re-establish the frame-zero reset without a legacy reset substep."""
    import torch

    env_ids = torch.arange(runtime.num_envs, device=runtime.device, dtype=torch.long)
    origins = runtime.scene.env_origins
    root_pose = torch.cat(
        (
            runtime.reference.data["body_pos_w"][[0], runtime.root_ref_body_id]
            .float()
            .expand(runtime.num_envs, -1)
            + origins,
            runtime.reference.data["body_quat_w"][[0], runtime.root_ref_body_id]
            .float()
            .expand(runtime.num_envs, -1),
        ),
        dim=-1,
    )
    root_velocity = torch.cat(
        (
            runtime.reference.data["body_lin_vel_w"][[0], runtime.root_ref_body_id]
            .float()
            .expand(runtime.num_envs, -1),
            runtime.reference.data["body_ang_vel_w"][[0], runtime.root_ref_body_id]
            .float()
            .expand(runtime.num_envs, -1),
        ),
        dim=-1,
    )
    runtime.scene.reset()
    runtime.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    runtime.robot.write_root_com_velocity_to_sim(root_velocity, env_ids=env_ids)
    runtime.robot.write_joint_state_to_sim(
        runtime._reference_robot_joint_state("joint_pos"),
        runtime._reference_robot_joint_state("joint_vel"),
        env_ids=env_ids,
    )
    door_pose = torch.cat(
        (
            runtime.reference.data["body_pos_w"][[0], runtime.door_ref_body_id]
            .float()
            .expand(runtime.num_envs, -1)
            + origins,
            runtime.reference.data["body_quat_w"][[0], runtime.door_ref_body_id]
            .float()
            .expand(runtime.num_envs, -1),
        ),
        dim=-1,
    )
    runtime.door.write_root_link_pose_to_sim(door_pose, env_ids=env_ids)
    runtime.door.write_root_com_velocity_to_sim(
        torch.zeros(runtime.num_envs, 6, device=runtime.device), env_ids=env_ids
    )
    runtime.door.write_joint_state_to_sim(
        runtime.reference.data["joint_pos"][[0], runtime.door_ref_joint_id]
        .float()
        .expand(runtime.num_envs, 1),
        runtime.reference.data["joint_vel"][[0], runtime.door_ref_joint_id]
        .float()
        .expand(runtime.num_envs, 1),
        env_ids=env_ids,
    )
    runtime.scene.write_data_to_sim()
    runtime.sim.forward()
    runtime.reference_step = 0
    runtime.action_runtime.reset(env_ids)
    runtime.history.reset(
        env_ids,
        root_ang_vel=runtime.robot.data.root_ang_vel_b,
        projected_gravity=runtime.robot.data.projected_gravity_b,
        joint_pos=runtime.robot.data.joint_pos,
        motion_length=runtime.reference.length,
    )


def _run_golden(args: argparse.Namespace, app: Any) -> int:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    from somaforce_cross.scaffold.pretrained_hdmi_isaac import (
        PretrainedHDMIIsaacRuntime,
    )

    del app
    payload, config_hash = _config(args)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    memory_before = _gpu_memory_mib()
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=args.physics_dt,
            render_interval=args.decimation,
            device=args.device,
        )
    )
    runtime = PretrainedHDMIIsaacRuntime(
        sim,
        ARTIFACT_DIR,
        num_envs=args.num_envs,
        delay=args.delay,
        alpha=args.alpha,
        door_friction=args.door_friction,
        door_damping=args.door_damping,
    )
    _rewind_golden_initial_state(runtime)
    snapshots: list[dict[str, np.ndarray]] = []
    terminated_rows: list[np.ndarray] = []
    timeout_rows: list[np.ndarray] = []
    try:
        for step in range(args.steps):
            a_nom = runtime.step()
            snapshot = _golden_snapshot(runtime, a_nom, config_hash, args.seed)
            snapshots.append(_cpu(snapshot))
            root_failure = runtime.robot.data.root_link_pos_w[:, 2] < 0.45
            nonfinite = ~snapshot["finite"]
            exhausted = runtime.reference_step + 32 >= runtime.reference.length
            terminated_rows.append((nonfinite | root_failure).cpu().numpy())
            timeout_rows.append(
                torch.full(
                    (args.num_envs,),
                    step + 1 >= args.episode_length_steps or exhausted,
                    device=runtime.device,
                    dtype=torch.bool,
                )
                .cpu()
                .numpy()
            )
        path = _trace_path(args, "golden", config_hash)
        metadata = {
            "mode": "golden",
            "configuration": payload,
            "configuration_hash": config_hash,
            "action_joint_names": list(runtime.task_spec.action_joint_names),
            "robot_joint_names": list(runtime.robot.joint_names),
            "robot_body_names": list(runtime.robot.body_names),
            "dtype": str(runtime.robot.data.joint_pos.dtype),
            "device": str(runtime.device),
        }
        arrays = _stack_trace(snapshots, metadata=metadata)
        arrays["terminated"] = np.stack(terminated_rows)
        arrays["time_outs"] = np.stack(timeout_rows)
        _save_trace(path, arrays)
        result = {
            "mode": "golden",
            "trace": str(path),
            "configuration_hash": config_hash,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "unique_origins": int(np.unique(arrays["env_origin"][0], axis=0).shape[0]),
            "finite": bool(arrays["finite"].all()),
            "terminated": int(arrays["terminated"].sum()),
            "time_outs": int(arrays["time_outs"].sum()),
            "gpu_memory_mib_before": memory_before,
            "gpu_memory_mib_after": _gpu_memory_mib(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        del runtime
        sim.clear_all_callbacks()
        sim.clear_instance()


def _apply_profile(args: argparse.Namespace, profile: Any) -> None:
    for name in asdict(profile):
        if hasattr(args, name):
            value = getattr(args, name)
            if name == "virtual_ft_axis_quat":
                value = tuple(value)
            setattr(profile, name, value)
    profile.physics_dt = args.physics_dt
    profile.control_dt = args.control_dt
    profile.decimation = args.decimation
    profile.smoke_control_steps = args.steps


def _selective_reset_evidence(env: Any, ids: list[int]) -> dict[str, Any]:
    import torch

    selected = torch.tensor(ids, device=env.device, dtype=torch.long)
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
        raise AssertionError(f"selective reset changed unselected rows: {changed}")
    max_abs_by_field: dict[str, float] = {}
    for name, value in before.items():
        left = value[mask]
        right = after[name][mask]
        if left.is_floating_point():
            difference = (left - right).abs()
        else:
            difference = (left != right).to(dtype=torch.float32)
        max_abs_by_field[name] = (
            float(difference.max().item()) if difference.numel() else 0.0
        )
    physics_fields = (
        "robot_root_state",
        "robot_joint_pos",
        "robot_joint_vel",
        "door_root_state",
        "door_joint_pos",
        "door_joint_vel",
        "applied_action",
        "joint_target",
    )
    return {
        "selected_env_ids": ids,
        "unchanged_rows": int(mask.sum()),
        "checked_fields": sorted(before),
        "physics_max_abs": max(max_abs_by_field[name] for name in physics_fields),
        "physics_max_abs_by_field": {
            name: max_abs_by_field[name] for name in physics_fields
        },
        "selected_seeds": {
            str(env_id): int(env.random_stream.seeds[env_id].item()) for env_id in ids
        },
        "passed": True,
    }


def _run_residual(args: argparse.Namespace, app: Any) -> int:
    print("[phase4b3] entering residual mode", file=sys.stderr, flush=True)
    import torch

    from somaforce_cross.envs.residual_env import SomaForceDoorResidualEnv
    from somaforce_cross.envs.residual_env_cfg import SomaForceDoorResidualEnvCfg

    del app
    payload, config_hash = _config(args)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    memory_before = _gpu_memory_mib()
    cfg = SomaForceDoorResidualEnvCfg()
    _apply_profile(args, cfg.smoke_profile)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.artifact_dir = str(ARTIFACT_DIR)
    cfg.configuration_hash = config_hash
    cfg.__post_init__()
    try:
        env = SomaForceDoorResidualEnv(cfg)
    except BaseException:
        traceback.print_exc()
        raise
    snapshots: list[dict[str, np.ndarray]] = []
    terminated_rows: list[np.ndarray] = []
    timeout_rows: list[np.ndarray] = []
    selective_result: dict[str, Any] | None = None
    try:
        observations, _ = env.reset()
        expected_shapes = {
            "policy": (args.num_envs, 668),
            "critic": (args.num_envs, 845),
            "semantic_target": (args.num_envs, 31),
        }
        for name, shape in expected_shapes.items():
            if observations[name].shape != shape:
                raise AssertionError(
                    f"{name} shape is {observations[name].shape}, expected {shape}"
                )
        raw_probe = torch.ones(args.num_envs, 23, device=env.device)
        for _ in range(args.steps):
            observations, rewards, terminated, time_outs, _ = env.step(raw_probe)
            if not torch.equal(rewards, torch.zeros_like(rewards)):
                raise AssertionError("door C0 smoke reward must remain zero")
            snapshots.append(_cpu(env.trace_snapshot()))
            terminated_rows.append(terminated.cpu().numpy())
            timeout_rows.append(time_outs.cpu().numpy())
        if args.selective_reset_env_ids:
            selective_result = _selective_reset_evidence(
                env, args.selective_reset_env_ids
            )
        path = _trace_path(args, "residual", config_hash)
        metadata = {
            "mode": "residual",
            "configuration": payload,
            "configuration_hash": config_hash,
            "action_joint_names": list(env.task_spec.action_joint_names),
            "robot_joint_names": list(env.robot.joint_names),
            "robot_body_names": list(env.robot.body_names),
            "dtype": str(env.robot.data.joint_pos.dtype),
            "device": str(env.device),
        }
        arrays = _stack_trace(snapshots, metadata=metadata)
        arrays["terminated"] = np.stack(terminated_rows)
        arrays["time_outs"] = np.stack(timeout_rows)
        _save_trace(path, arrays)
        result = {
            "mode": "residual",
            "trace": str(path),
            "configuration_hash": config_hash,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "shapes": {name: list(shape) for name, shape in expected_shapes.items()},
            "dtype": str(observations["policy"].dtype),
            "device": str(observations["policy"].device),
            "unique_origins": int(np.unique(arrays["env_origin"][0], axis=0).shape[0]),
            "finite": bool(arrays["finite"].all()),
            "c0_a_total_exact": bool(
                np.array_equal(arrays["a_total"], arrays["a_nom"])
            ),
            "terminated": int(arrays["terminated"].sum()),
            "time_outs": int(arrays["time_outs"].sum()),
            "selective_reset": selective_result,
            "gpu_memory_mib_before": memory_before,
            "gpu_memory_mib_after": _gpu_memory_mib(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        env.close()


def _metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return json.loads(str(archive["metadata_json"].item()))


def _component_name(
    field: str, index: tuple[int, ...], metadata: dict[str, Any]
) -> str:
    last = index[-1] if index else 0
    if field in {"a_nom", "a_total", "applied_action", "joint_target"}:
        names = metadata["action_joint_names"]
        return f"joint={names[last]} index={last}"
    if field in {"robot_joint_pos", "robot_joint_vel"}:
        names = metadata["robot_joint_names"]
        return f"joint={names[last]} index={last}"
    return f"component_index={last}"


def _first_failure(
    field: str,
    left: np.ndarray,
    right: np.ndarray,
    equal: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    index = tuple(int(item) for item in np.argwhere(~equal)[0])
    left_value = left[index]
    right_value = right[index]
    absolute = float(abs(float(left_value) - float(right_value)))
    denominator = max(
        abs(float(left_value)), abs(float(right_value)), np.finfo(float).tiny
    )
    return {
        "control_step": index[0],
        "env_id": index[1] if len(index) > 1 else 0,
        "field": field,
        "component": _component_name(field, index[2:], metadata),
        "golden": float(left_value),
        "residual": float(right_value),
        "absolute_error": absolute,
        "relative_error": absolute / denominator,
    }


def _compare(args: argparse.Namespace) -> int:
    exact_fields = (
        "finite",
        "terminated",
        "time_outs",
        "a_nom",
        "raw_residual",
        "authority",
        "a_total",
        "applied_action",
        "joint_target",
        "scaffold_history",
        "nominal_history",
        "executed_history",
        "reference_step",
        "reference_phase",
        "env_origin",
        "physics_mismatch",
        "reset_seed",
    )
    physics_fields = (
        "robot_root_state",
        "robot_joint_pos",
        "robot_joint_vel",
        "door_root_state",
        "door_joint_pos",
        "door_joint_vel",
        "door_applied_effort",
    )
    with (
        np.load(args.golden_trace, allow_pickle=False) as golden,
        np.load(args.residual_trace, allow_pickle=False) as residual,
    ):
        golden_meta = _metadata(golden)
        residual_meta = _metadata(residual)
        expected_keys = set(TRACE_FIELDS) | {"metadata_json"}
        golden_keys = set(golden.files)
        residual_keys = set(residual.files)
        if golden_keys != expected_keys or residual_keys != expected_keys:
            raise ValueError(
                "trace key contract mismatch: "
                f"golden_only={sorted(golden_keys - expected_keys)}, "
                f"residual_only={sorted(residual_keys - expected_keys)}, "
                f"missing_golden={sorted(expected_keys - golden_keys)}, "
                f"missing_residual={sorted(expected_keys - residual_keys)}"
            )
        metadata_contract_keys = {
            "configuration",
            "configuration_hash",
            "trace_fields",
            "trace_shapes",
            "trace_dtypes",
            "action_joint_names",
            "robot_joint_names",
            "robot_body_names",
            "dtype",
            "device",
            "mode",
        }
        if (
            set(golden_meta) != metadata_contract_keys
            or set(residual_meta) != metadata_contract_keys
        ):
            raise ValueError("trace metadata key contract mismatch")
        for name, value in golden_meta.items():
            if name != "mode" and value != residual_meta[name]:
                raise ValueError(f"trace metadata contract mismatch at {name}")
        if golden_meta["trace_fields"] != list(sorted(TRACE_FIELDS)):
            raise ValueError(
                "trace metadata does not declare the frozen field contract"
            )
        if golden_meta["trace_shapes"] != residual_meta["trace_shapes"]:
            raise ValueError("trace field shape contract mismatch")
        for archive, metadata, label in (
            (golden, golden_meta, "golden"),
            (residual, residual_meta, "residual"),
        ):
            for field in TRACE_FIELDS:
                if list(archive[field].shape[1:]) != metadata["trace_shapes"][field]:
                    raise ValueError(
                        f"{label} {field} trace shapes differ from metadata"
                    )
                if str(archive[field].dtype) != metadata["trace_dtypes"][field]:
                    raise ValueError(
                        f"{label} metadata dtype mismatch at {field}: "
                        f"actual {archive[field].dtype}, "
                        f"metadata {metadata['trace_dtypes'][field]}"
                    )
        for field in TRACE_FIELDS:
            if golden[field].shape != residual[field].shape:
                raise ValueError(
                    f"{field} trace shapes differ: {golden[field].shape} vs {residual[field].shape}"
                )
        if golden_meta["configuration_hash"] != residual_meta["configuration_hash"]:
            raise ValueError("trace configuration hashes differ")
        first: dict[str, Any] | None = None
        maxima: dict[str, dict[str, float]] = {}
        c0_checks = {
            "a_total_equals_a_nom": np.array_equal(
                residual["a_total"], residual["a_nom"]
            ),
            "raw_probe_is_one": np.array_equal(
                residual["raw_residual"], np.ones_like(residual["raw_residual"])
            ),
            "authority_is_zero": np.array_equal(
                residual["authority"], np.zeros_like(residual["authority"])
            ),
            "nominal_history_matches_golden": np.array_equal(
                residual["nominal_history"], golden["scaffold_history"]
            ),
            "executed_history_matches_golden": np.array_equal(
                residual["executed_history"], golden["scaffold_history"]
            ),
        }
        c0_pairs = {
            "a_total_equals_a_nom": ("a_nom", "a_total"),
            "raw_probe_is_one": ("raw_residual", "raw_residual"),
            "authority_is_zero": ("authority", "authority"),
            "nominal_history_matches_golden": ("scaffold_history", "nominal_history"),
            "executed_history_matches_golden": ("scaffold_history", "executed_history"),
        }
        for name, passed in c0_checks.items():
            if not passed and first is None:
                left_name, right_name = c0_pairs[name]
                left = golden[left_name]
                if name == "raw_probe_is_one":
                    left = np.ones_like(residual[right_name])
                elif name == "authority_is_zero":
                    left = np.zeros_like(residual[right_name])
                right = residual[right_name]
                first = _first_failure(name, left, right, left == right, golden_meta)
        for field in exact_fields:
            left = golden[field]
            right = residual[field]
            if left.shape != right.shape:
                raise ValueError(
                    f"{field} trace shapes differ: {left.shape} vs {right.shape}"
                )
            equal = left == right
            if not equal.all() and first is None:
                first = _first_failure(field, left, right, equal, golden_meta)
        for field in physics_fields:
            left = golden[field]
            right = residual[field]
            close = np.isclose(
                left,
                right,
                atol=args.parity_atol,
                rtol=args.parity_rtol,
                equal_nan=False,
            )
            absolute = np.abs(left - right)
            relative = absolute / np.maximum(
                np.maximum(np.abs(left), np.abs(right)), np.finfo(np.float32).tiny
            )
            maxima[field] = {
                "max_abs": float(absolute.max(initial=0.0)),
                "max_rel": float(relative.max(initial=0.0)),
            }
            if not close.all() and first is None:
                first = _first_failure(field, left, right, close, golden_meta)
        passed = first is None and all(c0_checks.values())
        result = {
            "mode": "compare",
            "passed": passed,
            "configuration_hash": golden_meta["configuration_hash"],
            "rows_compared": int(golden["a_nom"].shape[1]),
            "steps_compared": int(golden["a_nom"].shape[0]),
            "atol": args.parity_atol,
            "rtol": args.parity_rtol,
            "c0_checks": c0_checks,
            "max_errors": maxima,
            "first_divergence": first,
        }
        print(json.dumps(result, sort_keys=True))
        return 0 if passed else 1


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "compare":
        return _compare(args)

    from isaaclab.app import AppLauncher

    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print(f"[phase4b3] AppLauncher ready for {args.mode}", file=sys.stderr, flush=True)
    try:
        if args.mode == "golden":
            return _run_golden(args, simulation_app)
        return _run_residual(args, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
