#!/usr/bin/env python3
"""Bounded headless validation of the V1 Isaac wrist-wrench source adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


TASKS = ("push_door_hand", "push_box", "move_suitcase", "move_largebox")
WRIST_JOINT_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--artifact", type=Path)
parser.add_argument("--sensor-offset-left", nargs=3, type=float, required=True)
parser.add_argument("--sensor-offset-right", nargs=3, type=float, required=True)
parser.add_argument("--warmup-steps", type=int, required=True)
parser.add_argument("--settle-steps", type=int, required=True)
parser.add_argument("--stale-load-steps", type=int, required=True)
parser.add_argument("--load-steps", type=int, required=True)
parser.add_argument("--window-steps", type=int, required=True)
parser.add_argument("--required-consecutive-windows", type=int, required=True)
parser.add_argument("--stability-std", type=float, required=True)
parser.add_argument("--stability-mean-delta", type=float, required=True)
parser.add_argument("--force", type=float, required=True)
parser.add_argument("--torque", type=float, required=True)
parser.add_argument("--torque-sweep-magnitudes", nargs="+", type=float)
parser.add_argument("--full-matrix", action="store_true")
parser.add_argument("--door-followup", action="store_true")
parser.add_argument("--output-jsonl", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

for value, name in (
    (args.warmup_steps, "warmup_steps"),
    (args.settle_steps, "settle_steps"),
    (args.stale_load_steps, "stale_load_steps"),
    (args.load_steps, "load_steps"),
    (args.window_steps, "window_steps"),
    (args.required_consecutive_windows, "required_consecutive_windows"),
    (args.stability_std, "stability_std"),
    (args.stability_mean_delta, "stability_mean_delta"),
    (args.force, "force"),
    (args.torque, "torque"),
):
    if value <= 0:
        parser.error(f"--{name.replace('_', '-')} must be positive")
if args.full_matrix and args.door_followup:
    parser.error("--full-matrix and --door-followup are mutually exclusive")
if args.load_steps % args.window_steps:
    parser.error("--load-steps must be divisible by --window-steps")
minimum_windows = args.required_consecutive_windows + 1
if args.load_steps < minimum_windows * args.window_steps:
    parser.error("--load-steps must contain K final windows plus a preceding window")
if args.settle_steps < max(args.warmup_steps, args.window_steps):
    parser.error("--settle-steps must cover warmup and one averaging window")
if args.door_followup:
    if args.task != "push_door_hand":
        parser.error("--door-followup requires --task push_door_hand")
    if not args.torque_sweep_magnitudes:
        parser.error("--door-followup requires --torque-sweep-magnitudes")
    if any(value <= 0 for value in args.torque_sweep_magnitudes):
        parser.error("--torque-sweep-magnitudes must be positive")
if not args.output_jsonl.parent.is_dir():
    parser.error("--output-jsonl parent directory must already exist")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
output_stream = args.output_jsonl.open("w", encoding="utf-8")


def emit(record_type: str, **values: object) -> None:
    output_stream.write(json.dumps({"record": record_type, **values}, sort_keys=True))
    output_stream.write("\n")
    output_stream.flush()
    print(f"[wrench-diagnostic] wrote {record_type}", flush=True)


def _linear_fit(x: list[float], y: list[float]) -> tuple[float, float]:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y, strict=True)
    ) / max(denominator, 1.0e-12)
    return mean_y - slope * mean_x, slope


def main() -> int:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass

    from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
    from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg
    from somaforce_cross.scaffold.wrist_wrench_isaac import IsaacWristWrenchSource

    artifact = (
        (
            args.artifact
            if args.artifact is not None
            else REPO_ROOT / f"artifacts/scaffolds/hdmi_{args.task}/v1"
        )
        .expanduser()
        .resolve()
    )
    if not artifact.is_dir():
        raise FileNotFoundError(f"artifact does not exist: {artifact}")
    robot_cfg = make_scene_cfg(artifact, 1, get_hdmi_task_spec(args.task)).robot
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 2.0)
    robot_cfg.spawn.articulation_props.fix_root_link = True

    @configclass
    class WrenchAuditSceneCfg(InteractiveSceneCfg):
        robot = robot_cfg

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(WrenchAuditSceneCfg(num_envs=1, env_spacing=5.0))
    sim.reset()
    scene.update(0.005)
    robot = scene["robot"]
    sensor_offset = torch.tensor(
        [args.sensor_offset_left, args.sensor_offset_right],
        device=robot.device,
        dtype=robot.data.body_pos_w.dtype,
    )
    artifact_identity = str(artifact.relative_to(REPO_ROOT))
    source = IsaacWristWrenchSource(
        robot,
        artifact_identity=artifact_identity,
        sensor_offset_J=sensor_offset,
        warmup_steps=args.warmup_steps,
    )
    all_env_ids = torch.arange(source.batch_size, device=source.device)
    zero_force = torch.zeros(
        source.batch_size, robot.num_bodies, 3, device=source.device, dtype=source.dtype
    )
    zero_torque = torch.zeros_like(zero_force)
    wrist_joint_names: list[list[str]] = []
    wrist_joint_ids: list[torch.Tensor] = []
    for side in ("left", "right"):
        names = [f"{side}_{suffix}_joint" for suffix in WRIST_JOINT_SUFFIXES]
        ids = [robot.joint_names.index(name) for name in names]
        wrist_joint_names.append(names)
        wrist_joint_ids.append(
            torch.tensor(ids, device=source.device, dtype=torch.long)
        )

    def set_load(
        wrist_index: int | None = None,
        *,
        force_w: torch.Tensor | None = None,
        torque_w: torch.Tensor | None = None,
    ) -> None:
        forces = zero_force.clone()
        torques = zero_torque.clone()
        if wrist_index is not None:
            body_id = source.wrist_body_ids[wrist_index]
            if force_w is not None:
                forces[:, body_id] = force_w
            if torque_w is not None:
                torques[:, body_id] = torque_w
        robot.set_external_force_and_torque(forces, torques, is_global=True)

    def physics_step():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(0.005)
        return source.update_after_physics_step()

    def reset_pose() -> None:
        set_load()
        robot.reset(all_env_ids)
        robot.write_root_link_pose_to_sim(robot.data.default_root_state[:, :7])
        robot.write_root_com_velocity_to_sim(robot.data.default_root_state[:, 7:13])
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos, robot.data.default_joint_vel
        )
        scene.reset(all_env_ids)
        source.reset(all_env_ids)

    def settle_baseline() -> dict[str, object]:
        clean_frames = []
        raw_frames = []
        joint_pos_frames = []
        joint_vel_frames = []
        first_valid = 0
        for step in range(1, args.settle_steps + 1):
            output = physics_step()
            if first_valid == 0 and bool(output.valid.all()):
                first_valid = step
            clean_frames.append(output.clean_total_wrench_base_yaw[0].clone())
            raw_frames.append(output.raw_joint_wrench[0].clone())
            joint_pos_frames.append(robot.data.joint_pos[0].clone())
            joint_vel_frames.append(robot.data.joint_vel[0].clone())
        if first_valid == 0:
            raise RuntimeError("source did not become valid during baseline settling")
        return {
            "clean": torch.stack(clean_frames[-args.window_steps :]).mean(dim=0),
            "raw": torch.stack(raw_frames[-args.window_steps :]).mean(dim=0),
            "joint_pos": torch.stack(joint_pos_frames[-args.window_steps :]).mean(
                dim=0
            ),
            "joint_vel": torch.stack(joint_vel_frames[-args.window_steps :]).mean(
                dim=0
            ),
            "body_pos": robot.data.body_pos_w[0].clone(),
            "body_quat": robot.data.body_quat_w[0].clone(),
            "first_valid_step": first_valid,
        }

    def world_to_base_yaw(vector_w: torch.Tensor) -> torch.Tensor:
        q = robot.data.body_quat_w[0, source.pelvis_body_id]
        w, x, y, z = q
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        return torch.stack(
            (
                c * vector_w[0] + s * vector_w[1],
                -s * vector_w[0] + c * vector_w[1],
                vector_w[2],
            )
        )

    def quaternion_angle(q1: torch.Tensor, q2: torch.Tensor) -> float:
        dot = torch.dot(q1, q2).abs().clamp(max=1.0)
        return float(2.0 * torch.acos(dot))

    g1_path = artifact / "assets/g1.usd"
    identity_pose = all(
        max(abs(value) for value in item.child_local_pos) <= 1.0e-8
        and max(
            abs(value - expected)
            for value, expected in zip(
                item.child_local_quat_wxyz, (1.0, 0.0, 0.0, 0.0), strict=True
            )
        )
        <= 1.0e-8
        for item in source.metadata
    )
    reset_pose()
    initial_baseline = settle_baseline()
    sample = physics_step()
    emit(
        "artifact",
        task=args.task,
        artifact_identity=artifact_identity,
        g1_sha256=hashlib.sha256(g1_path.read_bytes()).hexdigest(),
        metadata=[
            {
                "body_name": item.body_name,
                "body_id": item.body_id,
                "joint_path": item.joint_path,
                "child_local_pos": list(item.child_local_pos),
                "child_local_quat_wxyz": list(item.child_local_quat_wxyz),
            }
            for item in source.metadata
        ],
        identity_child_transform=identity_pose,
        raw_shape=list(sample.raw_joint_wrench.shape),
        raw_dtype=str(sample.raw_joint_wrench.dtype),
        raw_device=str(sample.raw_joint_wrench.device),
        raw_finite=bool(torch.isfinite(sample.raw_joint_wrench).all()),
        first_valid_step=initial_baseline["first_valid_step"],
        sensor_offset_J=sensor_offset.tolist(),
        warmup_steps=args.warmup_steps,
        simulation_boundary_not_hardware_calibration=True,
        parameters={
            "baseline_settle_steps": args.settle_steps,
            "load_steps": args.load_steps,
            "window_steps": args.window_steps,
            "required_consecutive_windows": args.required_consecutive_windows,
            "stability_std": args.stability_std,
            "stability_mean_delta": args.stability_mean_delta,
        },
    )

    stale_force = torch.tensor(
        [args.force, 0.0, 0.0], device=source.device, dtype=source.dtype
    )
    set_load(0, force_w=stale_force)
    loaded_frames = [
        physics_step().clean_total_wrench_base_yaw[0].clone()
        for _ in range(args.stale_load_steps)
    ]
    loaded = torch.stack(loaded_frames[-args.window_steps :]).mean(dim=0)
    reset_pose()
    fresh = settle_baseline()
    stale_delta = fresh["clean"] - initial_baseline["clean"]
    emit(
        "reset",
        finite=bool(torch.isfinite(fresh["clean"]).all()),
        first_valid_step=fresh["first_valid_step"],
        loaded_to_fresh_norm=float(torch.linalg.vector_norm(loaded - fresh["clean"])),
        stale_residual_norm=float(torch.linalg.vector_norm(stale_delta)),
        stale_residual=stale_delta.tolist(),
    )

    case_specs: list[dict[str, object]] = []
    if args.full_matrix:
        for wrist_index, wrist_name in enumerate(("left", "right")):
            for kind, magnitude in (("force", args.force), ("torque", args.torque)):
                for axis_index, axis_name in enumerate(("X", "Y", "Z")):
                    for sign in (-1.0, 1.0):
                        case_specs.append(
                            {
                                "category": "full_matrix",
                                "wrist_index": wrist_index,
                                "wrist": wrist_name,
                                "kind": kind,
                                "axis_index": axis_index,
                                "axis": f"{'+' if sign > 0 else '-'}{axis_name}",
                                "sign": sign,
                                "magnitude": float(magnitude),
                            }
                        )
    if args.door_followup:
        for wrist_index, wrist_name in enumerate(("left", "right")):
            for magnitude in args.torque_sweep_magnitudes:
                for sign in (-1.0, 1.0):
                    case_specs.append(
                        {
                            "category": "torque_attribution",
                            "wrist_index": wrist_index,
                            "wrist": wrist_name,
                            "kind": "torque",
                            "axis_index": 1,
                            "axis": f"{'+' if sign > 0 else '-'}Y",
                            "sign": sign,
                            "magnitude": float(magnitude),
                        }
                    )
            for axis_index, axis_name, sign in (
                (0, "X", 1.0),
                (1, "Y", -1.0),
                (1, "Y", 1.0),
                (2, "Z", -1.0),
                (2, "Z", 1.0),
            ):
                case_specs.append(
                    {
                        "category": "force_slow_drift",
                        "wrist_index": wrist_index,
                        "wrist": wrist_name,
                        "kind": "force",
                        "axis_index": axis_index,
                        "axis": f"{'+' if sign > 0 else '-'}{axis_name}",
                        "sign": sign,
                        "magnitude": float(args.force),
                    }
                )

    axes = torch.eye(3, device=source.device, dtype=source.dtype)
    case_records: list[dict[str, object]] = []
    for spec in case_specs:
        wrist_index = int(spec["wrist_index"])
        kind = str(spec["kind"])
        component = slice(0, 3) if kind == "force" else slice(3, 6)
        reset_pose()
        baseline = settle_baseline()
        applied_w = (
            float(spec["sign"])
            * float(spec["magnitude"])
            * axes[int(spec["axis_index"])]
        )
        if kind == "force":
            set_load(wrist_index, force_w=applied_w)
        else:
            set_load(wrist_index, torque_w=applied_w)

        clean_history = []
        raw_history = []
        body_pos_history = []
        body_quat_history = []
        joint_pos_history = []
        joint_vel_history = []
        body_id = source.wrist_body_ids[wrist_index]
        joint_ids = wrist_joint_ids[wrist_index]
        for _ in range(args.load_steps):
            output = physics_step()
            clean_history.append(
                output.clean_total_wrench_base_yaw[0, wrist_index].clone()
            )
            raw_history.append(output.raw_joint_wrench[0, wrist_index].clone())
            body_pos_history.append(robot.data.body_pos_w[0, body_id].clone())
            body_quat_history.append(robot.data.body_quat_w[0, body_id].clone())
            joint_pos_history.append(
                robot.data.joint_pos[0].index_select(0, joint_ids).clone()
            )
            joint_vel_history.append(
                robot.data.joint_vel[0].index_select(0, joint_ids).clone()
            )

        clean = torch.stack(clean_history)
        raw = torch.stack(raw_history)
        body_pos = torch.stack(body_pos_history)
        body_quat = torch.stack(body_quat_history)
        joint_pos = torch.stack(joint_pos_history)
        joint_vel = torch.stack(joint_vel_history)
        clean_delta = clean - baseline["clean"][wrist_index]
        trace: list[dict[str, object]] = []
        previous_mean = None
        for start in range(0, args.load_steps, args.window_steps):
            end = start + args.window_steps
            primary_window = clean_delta[start:end, component]
            mean = primary_window.mean(dim=0)
            std = primary_window.std(dim=0, unbiased=False)
            adjacent_mean_delta = (
                None
                if previous_mean is None
                else float(torch.linalg.vector_norm(mean - previous_mean))
            )
            stable = (
                adjacent_mean_delta is not None
                and float(std.max()) <= args.stability_std
                and adjacent_mean_delta <= args.stability_mean_delta
            )
            trace.append(
                {
                    "window_index": len(trace),
                    "start_step": start + 1,
                    "end_step": end,
                    "primary_mean": mean.tolist(),
                    "primary_std": std.tolist(),
                    "std_max": float(std.max()),
                    "adjacent_mean_delta": adjacent_mean_delta,
                    "stable": stable,
                    "clean_wrench_mean": clean[start:end].mean(dim=0).tolist(),
                    "clean_delta_mean": clean_delta[start:end].mean(dim=0).tolist(),
                    "raw_incoming_wrench_mean": raw[start:end].mean(dim=0).tolist(),
                    "wrist_body_pos_mean": body_pos[start:end].mean(dim=0).tolist(),
                    "wrist_body_quat_final_wxyz": body_quat[end - 1].tolist(),
                    "joint_pos_mean": joint_pos[start:end].mean(dim=0).tolist(),
                    "joint_vel_mean": joint_vel[start:end].mean(dim=0).tolist(),
                }
            )
            previous_mean = mean

        final_windows = trace[-args.required_consecutive_windows :]
        final_stable = all(bool(window["stable"]) for window in final_windows)
        convergence_step = None
        if final_stable:
            last_unstable = max(
                (index for index, window in enumerate(trace) if not window["stable"]),
                default=-1,
            )
            convergence_step = (
                last_unstable + 1 + args.required_consecutive_windows
            ) * args.window_steps
        measured = clean_delta[-args.window_steps :, component].mean(dim=0)
        expected = world_to_base_yaw(applied_w)
        measured_norm = torch.linalg.vector_norm(measured)
        expected_norm = torch.linalg.vector_norm(expected)
        direction_cosine = float(
            torch.dot(measured, expected)
            / (measured_norm * expected_norm).clamp_min(1.0e-12)
        )
        magnitude_ratio = float(measured_norm / expected_norm)
        relative_error = abs(magnitude_ratio - 1.0)
        absolute_error = float((measured_norm - expected_norm).abs())
        expected_unit = expected / expected_norm
        signed_axis_error = float(torch.dot(measured - expected, expected_unit))
        baseline_pos = baseline["body_pos"][body_id]
        baseline_quat = baseline["body_quat"][body_id]
        final_pos = body_pos[-args.window_steps :].mean(dim=0)
        final_quat = body_quat[-1]
        baseline_joint_pos = baseline["joint_pos"].index_select(0, joint_ids)
        baseline_joint_vel = baseline["joint_vel"].index_select(0, joint_ids)
        final_joint_pos = joint_pos[-args.window_steps :].mean(dim=0)
        final_joint_vel = joint_vel[-args.window_steps :].mean(dim=0)
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (clean, raw, body_pos, body_quat, joint_pos, joint_vel)
        )
        record: dict[str, object] = {
            **spec,
            "applied_world": applied_w.tolist(),
            "expected_base_yaw": expected.tolist(),
            "measured_delta": measured.tolist(),
            "axis_error": (measured - expected).tolist(),
            "signed_axis_error": signed_axis_error,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "magnitude_ratio": magnitude_ratio,
            "direction_cosine": direction_cosine,
            "finite": finite,
            "final_consecutive_window_stability": final_stable,
            "required_consecutive_windows": args.required_consecutive_windows,
            "convergence_step": convergence_step,
            "no_load_baseline_clean": baseline["clean"][wrist_index].tolist(),
            "no_load_baseline_raw": baseline["raw"][wrist_index].tolist(),
            "final_clean_wrench_mean": clean[-args.window_steps :].mean(dim=0).tolist(),
            "final_raw_incoming_wrench_mean": raw[-args.window_steps :]
            .mean(dim=0)
            .tolist(),
            "wrist_body_pose": {
                "baseline_pos_w": baseline_pos.tolist(),
                "final_pos_w": final_pos.tolist(),
                "position_delta_w": (final_pos - baseline_pos).tolist(),
                "baseline_quat_wxyz": baseline_quat.tolist(),
                "final_quat_wxyz": final_quat.tolist(),
                "rotation_delta_rad": quaternion_angle(baseline_quat, final_quat),
            },
            "wrist_joint_state": {
                "names": wrist_joint_names[wrist_index],
                "baseline_pos": baseline_joint_pos.tolist(),
                "final_pos": final_joint_pos.tolist(),
                "position_delta": (final_joint_pos - baseline_joint_pos).tolist(),
                "baseline_vel": baseline_joint_vel.tolist(),
                "final_vel": final_joint_vel.tolist(),
                "velocity_delta": (final_joint_vel - baseline_joint_vel).tolist(),
            },
            "convergence_trace": trace,
            "accepted": (
                finite
                and final_stable
                and direction_cosine >= 0.99
                and relative_error <= 0.05
            ),
        }
        case_records.append(record)
        emit("case", **record)

    torque_records = [
        record for record in case_records if record["category"] == "torque_attribution"
    ]
    for wrist in ("left", "right"):
        for axis in ("-Y", "+Y"):
            group = sorted(
                (
                    record
                    for record in torque_records
                    if record["wrist"] == wrist and record["axis"] == axis
                ),
                key=lambda record: float(record["magnitude"]),
            )
            if not group:
                continue
            magnitudes = [float(record["magnitude"]) for record in group]
            signed_errors = [float(record["signed_axis_error"]) for record in group]
            absolute_errors = [float(record["absolute_error"]) for record in group]
            relative_errors = [float(record["relative_error"]) for record in group]
            intercept, slope = _linear_fit(magnitudes, signed_errors)
            emit(
                "torque_attribution",
                wrist=wrist,
                axis=axis,
                magnitudes=magnitudes,
                signed_axis_errors=signed_errors,
                absolute_errors=absolute_errors,
                relative_errors=relative_errors,
                signed_error_fit_intercept=intercept,
                signed_error_fit_slope=slope,
                relative_error_decreases=relative_errors[-1] < relative_errors[0],
            )

    emit(
        "summary",
        task=args.task,
        case_count=len(case_records),
        category_counts={
            category: sum(record["category"] == category for record in case_records)
            for category in sorted({str(record["category"]) for record in case_records})
        },
        accepted_count=sum(bool(record["accepted"]) for record in case_records),
        final_stable_count=sum(
            bool(record["final_consecutive_window_stability"])
            for record in case_records
        ),
        all_finite=all(bool(record["finite"]) for record in case_records),
        all_direction_cosine_ge_0_99=all(
            float(record["direction_cosine"]) >= 0.99 for record in case_records
        ),
        all_relative_error_le_0_05=all(
            float(record["relative_error"]) <= 0.05 for record in case_records
        ),
        signal_rescale_applied=False,
        contact_sensor_used=False,
        sensor_offset_J=sensor_offset.tolist(),
        warmup_steps=args.warmup_steps,
        simulation_boundary_not_hardware_calibration=True,
    )
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    output_stream.close()
    sys.stdout.flush()
    sys.stderr.flush()
    shutdown_watchdog = threading.Timer(10.0, lambda: os._exit(exit_code))
    shutdown_watchdog.daemon = True
    shutdown_watchdog.start()
    try:
        simulation_app.close()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        shutdown_watchdog.cancel()
    raise SystemExit(exit_code)
