"""Procedural G1 scaffold trajectories for first door and box diagnostics."""

from __future__ import annotations

from typing import Literal

import torch

from somaforce_cross.scaffold.contracts import G1_FULL_JOINT_NAMES, MotionTrajectory, ScaffoldTask

InteractionMode = Literal["push", "pull"]

_JOINT_INDEX = {name: i for i, name in enumerate(G1_FULL_JOINT_NAMES)}


def make_g1_push_pull_door_trajectory(
    mode: InteractionMode,
    num_frames: int = 80,
    duration_s: float = 2.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> MotionTrajectory:
    """Build a nominal Unitree G1 right-hand door interaction trajectory.

    The trajectory is a scaffold-only task reference. It is deliberately
    kinematic and does not use force, hinge state, contact truth, or object-side
    privileged labels.
    """

    phase = _phase(num_frames, device=device, dtype=dtype)
    joint_pos = _repeat_pose(_neutral_g1_pose(device=device, dtype=dtype), num_frames)

    reach = _smooth_lerp(0.0, 1.0, phase)
    direction = 1.0 if mode == "push" else -1.0

    _set(joint_pos, "waist_yaw_joint", 0.08 * direction * reach)
    _set(joint_pos, "waist_pitch_joint", -0.08 * reach)
    _set(joint_pos, "right_shoulder_pitch_joint", -0.35 - 0.18 * reach)
    _set(joint_pos, "right_shoulder_roll_joint", -0.16 - 0.08 * reach)
    _set(joint_pos, "right_shoulder_yaw_joint", 0.18 * direction * reach)
    _set(joint_pos, "right_elbow_joint", 0.65 - 0.28 * direction * reach)
    _set(joint_pos, "right_wrist_pitch_joint", 0.08 * direction * reach)
    _set(joint_pos, "left_shoulder_pitch_joint", -0.12 - 0.06 * reach)
    _set(joint_pos, "left_elbow_joint", 0.45)

    hand_pos = _hand_reference(
        phase=phase,
        start=(0.42, -0.28, 1.08),
        delta=(0.24 * direction, 0.02 * direction, 0.02),
        left_idle=(0.30, 0.27, 0.96),
    )
    body_ref = _body_reference(phase, x_delta=0.08 * direction, yaw_delta=0.08 * direction)
    cmd_6d = torch.tensor([direction, 0.0, 0.0, 0.0, 0.0, 0.08 * direction], device=device, dtype=dtype)

    return MotionTrajectory(
        joint_pos=joint_pos,
        time_s=_time(num_frames, duration_s, device=device, dtype=dtype),
        hand_pose_w=hand_pos,
        body_pose_w=body_ref,
        cmd_6d=cmd_6d,
    )


def make_g1_push_pull_box_trajectory(
    mode: InteractionMode,
    num_frames: int = 80,
    duration_s: float = 2.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> MotionTrajectory:
    """Build a nominal Unitree G1 two-hand box push/pull trajectory."""

    phase = _phase(num_frames, device=device, dtype=dtype)
    joint_pos = _repeat_pose(_neutral_g1_pose(device=device, dtype=dtype), num_frames)

    reach = _smooth_lerp(0.0, 1.0, phase)
    direction = 1.0 if mode == "push" else -1.0
    elbow_delta = -0.30 if mode == "push" else 0.34

    _set(joint_pos, "waist_pitch_joint", -0.10 * reach)
    _set(joint_pos, "left_shoulder_pitch_joint", -0.32 - 0.16 * reach)
    _set(joint_pos, "right_shoulder_pitch_joint", -0.32 - 0.16 * reach)
    _set(joint_pos, "left_shoulder_roll_joint", 0.18)
    _set(joint_pos, "right_shoulder_roll_joint", -0.18)
    _set(joint_pos, "left_elbow_joint", 0.70 + elbow_delta * reach)
    _set(joint_pos, "right_elbow_joint", 0.70 + elbow_delta * reach)
    _set(joint_pos, "left_wrist_pitch_joint", 0.06 * direction * reach)
    _set(joint_pos, "right_wrist_pitch_joint", 0.06 * direction * reach)

    hand_pos = _two_hand_reference(
        phase=phase,
        left_start=(0.45, 0.20, 0.92),
        right_start=(0.45, -0.20, 0.92),
        delta=(0.28 * direction, 0.0, 0.0),
    )
    body_ref = _body_reference(phase, x_delta=0.10 * direction, yaw_delta=0.0)
    cmd_6d = torch.tensor([direction, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)

    return MotionTrajectory(
        joint_pos=joint_pos,
        time_s=_time(num_frames, duration_s, device=device, dtype=dtype),
        hand_pose_w=hand_pos,
        body_pose_w=body_ref,
        cmd_6d=cmd_6d,
    )


def make_g1_task_trajectory(
    task: ScaffoldTask,
    mode: InteractionMode,
    num_frames: int = 80,
    duration_s: float = 2.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> MotionTrajectory:
    """Dispatch to the selected first-task G1 trajectory template."""

    if task == ScaffoldTask.PUSH_PULL_DOOR:
        return make_g1_push_pull_door_trajectory(mode, num_frames, duration_s, device, dtype)
    if task == ScaffoldTask.PUSH_PULL_BOX:
        return make_g1_push_pull_box_trajectory(mode, num_frames, duration_s, device, dtype)
    raise ValueError(f"Unsupported scaffold task: {task}")


def _neutral_g1_pose(
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    pose = torch.zeros(len(G1_FULL_JOINT_NAMES), device=device, dtype=dtype)
    values = {
        "left_hip_pitch_joint": -0.22,
        "right_hip_pitch_joint": -0.22,
        "left_knee_joint": 0.48,
        "right_knee_joint": 0.48,
        "left_ankle_pitch_joint": -0.26,
        "right_ankle_pitch_joint": -0.26,
        "left_shoulder_pitch_joint": -0.10,
        "right_shoulder_pitch_joint": -0.10,
        "left_elbow_joint": 0.45,
        "right_elbow_joint": 0.45,
    }
    for name, value in values.items():
        pose[_JOINT_INDEX[name]] = value
    return pose


def _repeat_pose(pose: torch.Tensor, num_frames: int) -> torch.Tensor:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    return pose.unsqueeze(0).repeat(num_frames, 1)


def _phase(
    num_frames: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    return torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)


def _time(
    num_frames: int,
    duration_s: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    return torch.linspace(0.0, duration_s, num_frames, device=device, dtype=dtype)


def _smooth_lerp(start: float, end: float, phase: torch.Tensor) -> torch.Tensor:
    smooth = phase * phase * (3.0 - 2.0 * phase)
    return start + (end - start) * smooth


def _set(joint_pos: torch.Tensor, joint_name: str, values: float | torch.Tensor) -> None:
    joint_pos[:, _JOINT_INDEX[joint_name]] = values


def _hand_reference(
    phase: torch.Tensor,
    start: tuple[float, float, float],
    delta: tuple[float, float, float],
    left_idle: tuple[float, float, float],
) -> torch.Tensor:
    right = _path(phase, start, delta)
    left = torch.tensor(left_idle, device=phase.device, dtype=phase.dtype).expand_as(right)
    return torch.stack((left, right), dim=1)


def _two_hand_reference(
    phase: torch.Tensor,
    left_start: tuple[float, float, float],
    right_start: tuple[float, float, float],
    delta: tuple[float, float, float],
) -> torch.Tensor:
    left = _path(phase, left_start, delta)
    right = _path(phase, right_start, delta)
    return torch.stack((left, right), dim=1)


def _path(
    phase: torch.Tensor,
    start: tuple[float, float, float],
    delta: tuple[float, float, float],
) -> torch.Tensor:
    start_t = torch.tensor(start, device=phase.device, dtype=phase.dtype)
    delta_t = torch.tensor(delta, device=phase.device, dtype=phase.dtype)
    progress = _smooth_lerp(0.0, 1.0, phase).unsqueeze(-1)
    return start_t + progress * delta_t


def _body_reference(phase: torch.Tensor, x_delta: float, yaw_delta: float) -> torch.Tensor:
    body = torch.zeros((phase.shape[0], 1, 6), device=phase.device, dtype=phase.dtype)
    progress = _smooth_lerp(0.0, 1.0, phase)
    body[:, 0, 0] = progress * x_delta
    body[:, 0, 2] = 0.76
    body[:, 0, 5] = progress * yaw_delta
    return body
