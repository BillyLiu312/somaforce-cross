#!/usr/bin/env python3
"""Independent four-task golden/residual C0 traces and strict comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = Path("/tmp/somaforce_phase4b4b")
TRACE_CONTRACT_VERSION = "phase4b4_c0_v1"
TASK_ARTIFACTS = {
    "push_door_hand": REPO_ROOT / "artifacts/scaffolds/hdmi_push_door_hand/v1",
    "push_box": REPO_ROOT / "artifacts/scaffolds/hdmi_push_box/v1",
    "move_suitcase": REPO_ROOT / "artifacts/scaffolds/hdmi_move_suitcase/v1",
    "move_largebox": REPO_ROOT / "artifacts/scaffolds/hdmi_move_largebox/v1",
}
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
    "object_root_state",
    "object_joint_pos",
    "object_joint_vel",
    "object_applied_effort",
    "env_origin",
    "physics_mismatch",
    "reset_seed",
    "finite",
    "terminated",
    "time_outs",
)
LEGACY_TRACE_FIELDS = tuple(
    sorted(
        (
            set(TRACE_FIELDS)
            - {
                "object_root_state",
                "object_joint_pos",
                "object_joint_vel",
                "object_applied_effort",
            }
        )
        | {
            "door_root_state",
            "door_joint_pos",
            "door_joint_vel",
            "door_applied_effort",
        }
    )
)
LEGACY_ALIASES = {
    "door_root_state": "object_root_state",
    "door_joint_pos": "object_joint_pos",
    "door_joint_vel": "object_joint_vel",
    "door_applied_effort": "object_applied_effort",
}
LEGACY_METADATA_KEYS = {
    "action_joint_names",
    "configuration",
    "configuration_hash",
    "device",
    "dtype",
    "mode",
    "robot_body_names",
    "robot_joint_names",
    "trace_dtypes",
    "trace_fields",
    "trace_shapes",
}
NEW_METADATA_KEYS = LEGACY_METADATA_KEYS | {
    "artifact_sha256s_sha256",
    "object_kind",
    "task",
    "trace_contract_version",
}
FLOAT_FIELDS = {
    "a_nom",
    "raw_residual",
    "authority",
    "a_total",
    "applied_action",
    "joint_target",
    "scaffold_history",
    "nominal_history",
    "executed_history",
    "reference_phase",
    "robot_root_state",
    "robot_joint_pos",
    "robot_joint_vel",
    "object_root_state",
    "object_joint_pos",
    "object_joint_vel",
    "object_applied_effort",
    "env_origin",
    "physics_mismatch",
}
INT_FIELDS = {"reference_step", "reset_seed"}
BOOL_FIELDS = {"finite", "terminated", "time_outs"}
FIELD_DTYPES = {
    **{name: "float32" for name in FLOAT_FIELDS},
    **{name: "int64" for name in INT_FIELDS},
    **{name: "bool" for name in BOOL_FIELDS},
}
FIELD_TRAILING_SHAPES = {
    "a_nom": (23,),
    "raw_residual": (23,),
    "authority": (23,),
    "a_total": (23,),
    "applied_action": (23,),
    "joint_target": (23,),
    "scaffold_history": (23, 3),
    "nominal_history": (23, 3),
    "executed_history": (23, 3),
    "reference_step": (),
    "reference_phase": (1,),
    "robot_root_state": (13,),
    "robot_joint_pos": (29,),
    "robot_joint_vel": (29,),
    "object_root_state": (13,),
    "object_joint_pos": (1,),
    "object_joint_vel": (1,),
    "object_applied_effort": (1,),
    "env_origin": (3,),
    "physics_mismatch": (39,),
    "reset_seed": (),
    "finite": (),
    "terminated": (),
    "time_outs": (),
}
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("golden", "residual", "compare"), required=True
    )
    parser.add_argument("--task", choices=tuple(TASK_ARTIFACTS))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260728)
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
    parser.add_argument("--object-friction", type=float, default=0.5)
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
    if args.mode != "compare" and args.task is None:
        raise ValueError("golden/residual mode requires --task")
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("num_envs and steps must be positive")
    if args.steps > args.episode_length_steps:
        raise ValueError("bounded smoke steps must not exceed episode length")
    if args.physics_dt * args.decimation != args.control_dt:
        raise ValueError("physics/control dt and decimation are inconsistent")
    if args.mode == "compare" and (
        args.golden_trace is None or args.residual_trace is None
    ):
        raise ValueError("compare mode requires both trace paths")
    if args.selective_reset_env_ids and args.mode != "residual":
        raise ValueError("selective reset is available only in residual mode")
    if args.selective_reset_env_ids:
        ids = args.selective_reset_env_ids
        if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= args.num_envs:
            raise ValueError("selective reset ids must be unique valid rows")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _profile_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "task": args.task,
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
    if args.task != "push_door_hand":
        payload["object_friction"] = args.object_friction
    return payload


def _config(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    payload = _profile_payload(args)
    return payload, hashlib.sha256(_canonical_json(payload)).hexdigest()


def _artifact_identity(artifact_dir: Path) -> str:
    checksum_path = artifact_dir / "SHA256SUMS"
    raw = checksum_path.read_bytes()
    for raw_line in raw.decode("utf-8").splitlines():
        expected, relative = raw_line.split(maxsplit=1)
        candidate = (artifact_dir / relative).resolve()
        if artifact_dir not in candidate.parents or not candidate.is_file():
            raise ValueError(f"invalid artifact checksum path: {relative}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"artifact checksum mismatch: {relative}")
    return hashlib.sha256(raw).hexdigest()


def _trace_path(args: argparse.Namespace, mode: str, config_hash: str) -> Path:
    if args.trace is not None:
        return args.trace.expanduser().resolve()
    return TRACE_ROOT / f"{args.task}_{args.num_envs}_{mode}_{config_hash[:12]}.npz"


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

    return {
        name: value.detach().cpu().numpy()
        for name, value in snapshot.items()
        if isinstance(value, torch.Tensor)
    }


def _new_metadata(
    *,
    mode: str,
    payload: dict[str, Any],
    config_hash: str,
    task_spec: Any,
    robot: Any,
    device: Any,
    artifact_identity: str,
) -> dict[str, Any]:
    return {
        "action_joint_names": list(task_spec.action_joint_names),
        "artifact_sha256s_sha256": artifact_identity,
        "configuration": payload,
        "configuration_hash": config_hash,
        "device": str(device),
        "dtype": str(robot.data.joint_pos.dtype),
        "mode": mode,
        "object_kind": task_spec.object_kind,
        "robot_body_names": list(robot.body_names),
        "robot_joint_names": list(robot.joint_names),
        "task": task_spec.task,
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "trace_dtypes": dict(FIELD_DTYPES),
        "trace_fields": list(sorted(TRACE_FIELDS)),
        "trace_shapes": {},
    }


def _save_trace(
    path: Path,
    snapshots: list[dict[str, np.ndarray]],
    terminated: list[np.ndarray],
    time_outs: list[np.ndarray],
    metadata: dict[str, Any],
) -> None:
    arrays = {
        name: np.stack([snapshot[name] for snapshot in snapshots])
        for name in TRACE_FIELDS
        if name not in BOOL_FIELDS - {"finite"}
    }
    arrays["terminated"] = np.stack(terminated)
    arrays["time_outs"] = np.stack(time_outs)
    metadata["trace_shapes"] = {
        name: list(arrays[name].shape[1:]) for name in TRACE_FIELDS
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _physics_row(runtime: Any) -> Any:
    import torch
    from somaforce_cross.envs.task_adapter import rigid_nominal_physics_mismatch

    if runtime.task_spec.object_kind == "articulation":
        result = torch.zeros(runtime.num_envs, 39, device=runtime.device)
        result[:, 3] = runtime.door_friction
        result[:, 4] = runtime.door_damping
        return result
    view = runtime.object.root_physx_view
    masses = (
        view.get_masses()
        .to(runtime.device, torch.float32)
        .reshape(runtime.num_envs, -1)
    )
    inertias = (
        view.get_inertias()
        .to(runtime.device, torch.float32)
        .reshape(runtime.num_envs, -1, 9)
    )
    materials = (
        view.get_material_properties()
        .to(runtime.device, torch.float32)
        .reshape(runtime.num_envs, -1, 3)
    )
    return rigid_nominal_physics_mismatch(
        masses[:, 0], inertias[:, 0], materials[:, 0, 0], materials[:, 0, 1]
    )


def _mechanism_state(runtime: Any) -> tuple[Any, Any, Any]:
    import torch

    if runtime.task_spec.object_kind == "articulation":
        return (
            runtime.object.data.joint_pos,
            runtime.object.data.joint_vel,
            runtime.object.data.applied_torque,
        )
    zeros = torch.zeros(runtime.num_envs, 1, device=runtime.device)
    return zeros, zeros.clone(), zeros.clone()


def _golden_snapshot(runtime: Any, a_nom: Any, seed: int) -> dict[str, Any]:
    import torch

    mechanism_pos, mechanism_vel, mechanism_effort = _mechanism_state(runtime)
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
        runtime.object.data.root_state_w,
        mechanism_pos,
    ):
        finite &= torch.isfinite(value).reshape(runtime.num_envs, -1).all(dim=1)
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
        "reference_step": torch.full(
            (runtime.num_envs,),
            runtime.reference_step,
            device=runtime.device,
            dtype=torch.long,
        ),
        "reference_phase": runtime.history.phase,
        "robot_root_state": runtime.robot.data.root_state_w,
        "robot_joint_pos": runtime.robot.data.joint_pos,
        "robot_joint_vel": runtime.robot.data.joint_vel,
        "object_root_state": runtime.object.data.root_state_w,
        "object_joint_pos": mechanism_pos,
        "object_joint_vel": mechanism_vel,
        "object_applied_effort": mechanism_effort,
        "env_origin": runtime.scene.env_origins,
        "physics_mismatch": _physics_row(runtime),
        "reset_seed": torch.arange(
            seed, seed + runtime.num_envs, device=runtime.device, dtype=torch.long
        ),
        "finite": finite,
    }


def _rewind_golden_initial_state(runtime: Any) -> None:
    import torch

    ids = torch.arange(runtime.num_envs, device=runtime.device, dtype=torch.long)
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
    object_pose = torch.cat(
        (
            runtime.reference.data["body_pos_w"][[0], runtime.object_ref_body_id]
            .float()
            .expand(runtime.num_envs, -1)
            + origins,
            runtime.reference.data["body_quat_w"][[0], runtime.object_ref_body_id]
            .float()
            .expand(runtime.num_envs, -1),
        ),
        dim=-1,
    )
    runtime.scene.reset()
    runtime.robot.write_root_link_pose_to_sim(root_pose, env_ids=ids)
    runtime.robot.write_root_com_velocity_to_sim(root_velocity, env_ids=ids)
    runtime.robot.write_joint_state_to_sim(
        runtime._reference_robot_joint_state("joint_pos"),
        runtime._reference_robot_joint_state("joint_vel"),
        env_ids=ids,
    )
    runtime.object.write_root_link_pose_to_sim(object_pose, env_ids=ids)
    runtime.object.write_root_com_velocity_to_sim(
        torch.zeros(runtime.num_envs, 6, device=runtime.device), env_ids=ids
    )
    if runtime.task_spec.object_kind == "articulation":
        runtime.object.write_joint_state_to_sim(
            runtime.reference.data["joint_pos"][[0], runtime.door_ref_joint_id]
            .float()
            .expand(runtime.num_envs, 1),
            runtime.reference.data["joint_vel"][[0], runtime.door_ref_joint_id]
            .float()
            .expand(runtime.num_envs, 1),
            env_ids=ids,
        )
    runtime.scene.write_data_to_sim()
    runtime.sim.forward()
    runtime.reference_step = 0
    runtime.action_runtime.reset(ids)
    runtime.history.reset(
        ids,
        root_ang_vel=runtime.robot.data.root_ang_vel_b,
        projected_gravity=runtime.robot.data.projected_gravity_b,
        joint_pos=runtime.robot.data.joint_pos,
        motion_length=runtime.reference.length,
    )


def _run_golden(args: argparse.Namespace, app: Any) -> int:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
    from somaforce_cross.scaffold.pretrained_hdmi_isaac import (
        PretrainedHDMIIsaacRuntime,
    )

    del app
    artifact = TASK_ARTIFACTS[args.task]
    identity = _artifact_identity(artifact)
    payload, config_hash = _config(args)
    spec = get_hdmi_task_spec(args.task)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    before = _gpu_memory_mib()
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=args.physics_dt, render_interval=args.decimation, device=args.device
        )
    )
    runtime = PretrainedHDMIIsaacRuntime(
        sim,
        artifact,
        num_envs=args.num_envs,
        delay=args.delay,
        alpha=args.alpha,
        door_friction=args.door_friction,
        door_damping=args.door_damping,
        object_mass=spec.nominal_object_mass,
        object_friction=args.object_friction,
    )
    _rewind_golden_initial_state(runtime)
    snapshots: list[dict[str, np.ndarray]] = []
    terminated: list[np.ndarray] = []
    time_outs: list[np.ndarray] = []
    try:
        for step in range(args.steps):
            a_nom = runtime.step()
            snapshot = _golden_snapshot(runtime, a_nom, args.seed)
            snapshots.append(_cpu(snapshot))
            threshold = 0.45 if args.task in ("push_door_hand", "push_box") else 0.25
            failed = runtime.robot.data.root_link_pos_w[:, 2] < threshold
            exhausted = (
                runtime.reference_step + 32 >= runtime.reference.length
                if args.task == "push_door_hand"
                else runtime.reference_step >= runtime.reference.length
            )
            terminated.append((~snapshot["finite"] | failed).cpu().numpy())
            time_outs.append(
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
        metadata = _new_metadata(
            mode="golden",
            payload=payload,
            config_hash=config_hash,
            task_spec=runtime.task_spec,
            robot=runtime.robot,
            device=runtime.device,
            artifact_identity=identity,
        )
        _save_trace(path, snapshots, terminated, time_outs, metadata)
        arrays = snapshots[-1]
        result = {
            "mode": "golden",
            "task": args.task,
            "trace": str(path),
            "configuration_hash": config_hash,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "unique_origins": int(np.unique(arrays["env_origin"], axis=0).shape[0]),
            "finite": bool(all(item["finite"].all() for item in snapshots)),
            "gpu_memory_mib_before": before,
            "gpu_memory_mib_after": _gpu_memory_mib(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        del runtime
        sim.clear_all_callbacks()
        sim.clear_instance()


def _apply_profile(args: argparse.Namespace, profile: Any) -> None:
    for name in asdict(profile):
        if hasattr(args, name):
            value = getattr(args, name)
            setattr(
                profile, name, tuple(value) if name == "virtual_ft_axis_quat" else value
            )
    profile.task = args.task
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
    if torch.count_nonzero(env.wrist_history.storage[selected]) != 0:
        raise AssertionError("selected wrist history is not bitwise zero")
    return {
        "selected_env_ids": ids,
        "unchanged_rows": int(mask.sum()),
        "checked_fields": sorted(before),
        "selected_seeds": {str(i): int(env.random_stream.seeds[i]) for i in ids},
        "wrist_history_zero": True,
        "passed": True,
    }


def _run_residual(args: argparse.Namespace, app: Any) -> int:
    import torch
    from somaforce_cross.envs.residual_env import SomaForceResidualEnv
    from somaforce_cross.envs.residual_env_cfg import SomaForceResidualEnvCfg
    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
    from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg

    del app
    artifact = TASK_ARTIFACTS[args.task]
    identity = _artifact_identity(artifact)
    payload, config_hash = _config(args)
    spec = get_hdmi_task_spec(args.task)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    before = _gpu_memory_mib()
    cfg = SomaForceResidualEnvCfg()
    _apply_profile(args, cfg.smoke_profile)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene = make_scene_cfg(
        artifact,
        args.num_envs,
        spec,
        rigid_object_mass=spec.nominal_object_mass or 8.0,
    )
    cfg.artifact_dir = str(artifact)
    cfg.configuration_hash = config_hash
    cfg.__post_init__()
    env = SomaForceResidualEnv(cfg)
    snapshots: list[dict[str, np.ndarray]] = []
    terminated: list[np.ndarray] = []
    time_outs: list[np.ndarray] = []
    selective: dict[str, Any] | None = None
    try:
        observations, _ = env.reset()
        expected_shapes = {
            "policy": (args.num_envs, 668),
            "critic": (args.num_envs, 845),
            "semantic_target": (args.num_envs, 31),
        }
        for name, shape in expected_shapes.items():
            value = observations[name]
            if (
                value.shape != shape
                or value.dtype != torch.float32
                or str(value.device) != str(env.device)
                or not torch.isfinite(value).all()
            ):
                raise AssertionError(f"invalid {name} observation contract")
        raw_probe = torch.ones(args.num_envs, 23, device=env.device)
        for _ in range(args.steps):
            observations, rewards, done, timeout, _ = env.step(raw_probe)
            if not torch.equal(rewards, torch.zeros_like(rewards)):
                raise AssertionError("C0 smoke reward must remain zero")
            snapshots.append(_cpu(env.trace_snapshot()))
            terminated.append(done.cpu().numpy())
            time_outs.append(timeout.cpu().numpy())
        if args.selective_reset_env_ids:
            selective = _selective_reset_evidence(env, args.selective_reset_env_ids)
        path = _trace_path(args, "residual", config_hash)
        metadata = _new_metadata(
            mode="residual",
            payload=payload,
            config_hash=config_hash,
            task_spec=env.task_spec,
            robot=env.robot,
            device=env.device,
            artifact_identity=identity,
        )
        _save_trace(path, snapshots, terminated, time_outs, metadata)
        stacked_nom = np.stack([item["a_nom"] for item in snapshots])
        stacked_total = np.stack([item["a_total"] for item in snapshots])
        result = {
            "mode": "residual",
            "task": args.task,
            "trace": str(path),
            "configuration_hash": config_hash,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "shapes": {name: list(shape) for name, shape in expected_shapes.items()},
            "dtype": str(observations["policy"].dtype),
            "device": str(observations["policy"].device),
            "unique_origins": int(
                np.unique(snapshots[0]["env_origin"], axis=0).shape[0]
            ),
            "finite": bool(all(item["finite"].all() for item in snapshots)),
            "c0_a_total_exact": bool(np.array_equal(stacked_total, stacked_nom)),
            "selected_reset": selective,
            "gpu_memory_mib_before": before,
            "gpu_memory_mib_after": _gpu_memory_mib(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        env.close()


def _metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in archive.files:
        raise ValueError("trace is missing metadata_json")
    value = json.loads(str(archive["metadata_json"].item()))
    if not isinstance(value, dict):
        raise ValueError("trace metadata must be a JSON object")
    return value


def _validate_common_arrays(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    fields: tuple[str, ...],
    label: str,
) -> None:
    if set(arrays) != set(fields):
        raise ValueError(f"{label} trace field key contract mismatch")
    if metadata.get("trace_fields") != list(sorted(fields)):
        raise ValueError(f"{label} trace_fields contract mismatch")
    for map_name in ("trace_shapes", "trace_dtypes"):
        value = metadata.get(map_name)
        if not isinstance(value, dict) or set(value) != set(fields):
            raise ValueError(f"{label} {map_name} key contract mismatch")
    steps: int | None = None
    batch: int | None = None
    for field in fields:
        array = arrays[field]
        if array.ndim < 2:
            raise ValueError(f"{label} {field} must include step and batch axes")
        if steps is None:
            steps, batch = array.shape[:2]
            if steps <= 0 or batch <= 0:
                raise ValueError(f"{label} trace axes must be nonempty")
        elif array.shape[:2] != (steps, batch):
            raise ValueError(f"{label} {field} step/batch shape mismatch")
        if list(array.shape[1:]) != metadata["trace_shapes"][field]:
            raise ValueError(f"{label} {field} trace shapes differ from metadata")
        if str(array.dtype) != metadata["trace_dtypes"][field]:
            raise ValueError(
                f"{label} metadata dtype mismatch at {field}: actual {array.dtype}, metadata {metadata['trace_dtypes'][field]}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{label} {field} contains nonfinite values")


def _validate_robot_names(metadata: dict[str, Any], label: str) -> None:
    robot_joints = metadata.get("robot_joint_names")
    if (
        not isinstance(robot_joints, list)
        or len(robot_joints) != 29
        or not all(isinstance(name, str) for name in robot_joints)
        or len(set(robot_joints)) != 29
    ):
        raise ValueError(
            f"{label} robot_joint_names must be a length-29 list[str] "
            "with unique entries"
        )
    robot_bodies = metadata.get("robot_body_names")
    if (
        not isinstance(robot_bodies, list)
        or not robot_bodies
        or not all(isinstance(name, str) for name in robot_bodies)
        or len(set(robot_bodies)) != len(robot_bodies)
    ):
        raise ValueError(
            f"{label} robot_body_names must be a nonempty list[str] with unique entries"
        )


def _validate_new_metadata(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    label: str,
    expected_mode: str,
) -> tuple[str, str, str]:
    if set(metadata) != NEW_METADATA_KEYS:
        raise ValueError(f"{label} trace metadata key contract mismatch")
    if metadata["trace_contract_version"] != TRACE_CONTRACT_VERSION:
        raise ValueError(f"{label} trace contract version mismatch")
    if metadata["mode"] != expected_mode:
        raise ValueError(f"{label} trace mode mismatch")
    task = metadata["task"]
    if task not in TASK_ARTIFACTS:
        raise ValueError(f"{label} task is unsupported")
    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec

    spec = get_hdmi_task_spec(task)
    if metadata["object_kind"] != spec.object_kind:
        raise ValueError(f"{label} object_kind mismatch")
    configuration = metadata["configuration"]
    if (
        not isinstance(configuration, dict)
        or configuration.get("task") != task
        or "mode" in configuration
    ):
        raise ValueError(f"{label} configuration task/mode contract mismatch")
    computed_config = hashlib.sha256(_canonical_json(configuration)).hexdigest()
    if metadata["configuration_hash"] != computed_config:
        raise ValueError(f"{label} configuration hash mismatch")
    identity = metadata["artifact_sha256s_sha256"]
    if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise ValueError(f"{label} artifact identity format mismatch")
    live = _artifact_identity(TASK_ARTIFACTS[task])
    if identity != live:
        raise ValueError(f"{label} artifact identity mismatch")
    if metadata["action_joint_names"] != list(spec.action_joint_names):
        raise ValueError(f"{label} action joint names mismatch")
    _validate_robot_names(metadata, label)
    if metadata["dtype"] != "torch.float32" or not isinstance(metadata["device"], str):
        raise ValueError(f"{label} dtype/device contract mismatch")
    if metadata["trace_dtypes"] != FIELD_DTYPES:
        raise ValueError(f"{label} trace dtype map contract mismatch")
    _validate_common_arrays(arrays, metadata, tuple(TRACE_FIELDS), label)
    batch = next(iter(arrays.values())).shape[1]
    expected_shapes = {
        name: [batch, *FIELD_TRAILING_SHAPES[name]] for name in TRACE_FIELDS
    }
    if metadata["trace_shapes"] != expected_shapes:
        raise ValueError(f"{label} trace shape map contract mismatch")
    return task, spec.object_kind, live


def _validate_legacy_metadata(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    label: str,
    expected_mode: str,
) -> tuple[str, str, str]:
    if "trace_contract_version" in metadata or set(metadata) != LEGACY_METADATA_KEYS:
        raise ValueError(f"{label} unknown unversioned trace metadata")
    configuration = metadata.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("task") != "push_door_hand"
    ):
        raise ValueError(f"{label} legacy door configuration mismatch")
    if metadata.get("mode") != expected_mode:
        raise ValueError(f"{label} trace mode mismatch")
    if metadata.get("trace_fields") != list(LEGACY_TRACE_FIELDS):
        raise ValueError(f"{label} legacy trace field contract mismatch")
    if (
        metadata.get("configuration_hash")
        != hashlib.sha256(_canonical_json(configuration)).hexdigest()
    ):
        raise ValueError(f"{label} legacy configuration hash mismatch")
    _validate_robot_names(metadata, label)
    _validate_common_arrays(arrays, metadata, LEGACY_TRACE_FIELDS, label)
    for source, target in LEGACY_ALIASES.items():
        if source not in arrays or target in arrays:
            raise ValueError(f"{label} legacy alias contract mismatch at {source}")
        arrays[target] = arrays.pop(source)
        metadata["trace_shapes"][target] = metadata["trace_shapes"].pop(source)
        metadata["trace_dtypes"][target] = metadata["trace_dtypes"].pop(source)
    metadata["trace_fields"] = list(sorted(TRACE_FIELDS))
    live = _artifact_identity(TASK_ARTIFACTS["push_door_hand"])
    return "push_door_hand", "articulation", live


def _load_canonical(
    path: Path, label: str, expected_mode: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], str, tuple[str, str, str]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = _metadata(archive)
        expected_fields = metadata.get("trace_fields")
        if not isinstance(expected_fields, list):
            raise ValueError(f"{label} trace_fields must be a list")
        expected_keys = set(expected_fields) | {"metadata_json"}
        if set(archive.files) != expected_keys:
            raise ValueError(f"{label} trace key contract mismatch")
        arrays = {name: archive[name] for name in expected_fields}
    if "trace_contract_version" in metadata:
        context = _validate_new_metadata(metadata, arrays, label, expected_mode)
        kind = TRACE_CONTRACT_VERSION
    else:
        context = _validate_legacy_metadata(metadata, arrays, label, expected_mode)
        kind = "legacy_phase4b3_door_unversioned"
    return arrays, metadata, kind, context


def _first_failure(
    field: str,
    left: np.ndarray,
    right: np.ndarray,
    equal: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    index = tuple(int(item) for item in np.argwhere(~equal)[0])
    left_value, right_value = left[index], right[index]
    absolute = abs(float(left_value) - float(right_value))
    names = (
        metadata.get("action_joint_names", [])
        if field in {"a_nom", "a_total", "applied_action", "joint_target"}
        else metadata.get("robot_joint_names", [])
        if field in {"robot_joint_pos", "robot_joint_vel"}
        else []
    )
    component = index[-1] if len(index) > 2 else 0
    return {
        "control_step": index[0],
        "env_id": index[1],
        "field": field,
        "component": f"{names[component] if component < len(names) else 'index'}={component}",
        "golden": float(left_value),
        "residual": float(right_value),
        "absolute_error": absolute,
        "relative_error": absolute
        / max(abs(float(left_value)), abs(float(right_value)), np.finfo(float).tiny),
    }


def _compare(args: argparse.Namespace) -> int:
    golden_path = args.golden_trace.expanduser().resolve()
    residual_path = args.residual_trace.expanduser().resolve()
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (golden_path, residual_path)
    }
    golden, golden_meta, golden_kind, golden_context = _load_canonical(
        golden_path, "golden", "golden"
    )
    residual, residual_meta, residual_kind, residual_context = _load_canonical(
        residual_path, "residual", "residual"
    )
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (golden_path, residual_path)
    }
    if before != after:
        raise ValueError("trace archive bytes changed during comparison")
    if golden_context != residual_context:
        raise ValueError("trace task/object/artifact context mismatch")
    common_metadata = LEGACY_METADATA_KEYS - {
        "mode",
        "trace_fields",
        "trace_shapes",
        "trace_dtypes",
    }
    for name in common_metadata:
        if golden_meta[name] != residual_meta[name]:
            raise ValueError(f"trace metadata contract mismatch at {name}")
    if (
        golden_kind == TRACE_CONTRACT_VERSION
        and residual_kind == TRACE_CONTRACT_VERSION
    ):
        for name in NEW_METADATA_KEYS - {"mode"}:
            if golden_meta[name] != residual_meta[name]:
                raise ValueError(f"trace metadata contract mismatch at {name}")
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
    tolerance_fields = (
        "robot_root_state",
        "robot_joint_pos",
        "robot_joint_vel",
        "object_root_state",
        "object_joint_pos",
        "object_joint_vel",
        "object_applied_effort",
    )
    for field in TRACE_FIELDS:
        if golden[field].shape != residual[field].shape:
            raise ValueError(f"{field} trace shapes differ")
    c0_checks = {
        "finite_all": bool(golden["finite"].all() and residual["finite"].all()),
        "a_total_equals_a_nom": np.array_equal(residual["a_total"], residual["a_nom"]),
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
    first: dict[str, Any] | None = None
    maxima: dict[str, dict[str, float]] = {}
    for field in exact_fields:
        equal = golden[field] == residual[field]
        if not equal.all() and first is None:
            first = _first_failure(
                field, golden[field], residual[field], equal, golden_meta
            )
    for field in tolerance_fields:
        left, right = golden[field], residual[field]
        close = np.isclose(
            left, right, atol=args.parity_atol, rtol=args.parity_rtol, equal_nan=False
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
        "task": golden_context[0],
        "passed": passed,
        "golden_contract": golden_kind,
        "residual_contract": residual_kind,
        "artifact_sha256s_sha256": golden_context[2],
        "configuration_hash": golden_meta["configuration_hash"],
        "rows_compared": int(golden["a_nom"].shape[1]),
        "steps_compared": int(golden["a_nom"].shape[0]),
        "atol": args.parity_atol,
        "rtol": args.parity_rtol,
        "c0_checks": c0_checks,
        "max_errors": maxima,
        "first_divergence": first,
        "archives_byte_immutable": before == after,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
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
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        return (
            _run_golden(args, simulation_app)
            if args.mode == "golden"
            else _run_residual(args, simulation_app)
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
