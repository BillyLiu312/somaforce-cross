"""Export scaffold trajectories to Sonic motion-lib files."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from somaforce_cross.scaffold.contracts import MotionTrajectory, ScaffoldTask
from somaforce_cross.scaffold.task_trajectories import (
    InteractionMode,
    make_g1_push_pull_box_trajectory,
    make_g1_push_pull_door_trajectory,
)

NUM_G1_DOF = 29
NUM_G1_MOTION_BODIES = 30

# G1 MJCF actuator axes used by Sonic's motion-lib converter.
G1_DOF_AXIS = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)


def motion_trajectory_to_sonic_entry(
    trajectory: MotionTrajectory,
    fps: int | None = None,
) -> dict[str, np.ndarray | int]:
    """Convert a single G1 scaffold trajectory to a Sonic motion-lib entry."""

    joint_pos = _to_numpy(trajectory.joint_pos)
    if joint_pos.ndim != 2:
        raise ValueError("Sonic motion export expects unbatched joint_pos with shape [T, 29]")
    if joint_pos.shape[1] != NUM_G1_DOF:
        raise ValueError(f"Sonic G1 motion export expects {NUM_G1_DOF} DOFs")

    num_frames = joint_pos.shape[0]
    root_trans_offset, yaw = _root_from_body_ref(trajectory, num_frames)
    root_rot = _yaw_to_xyzw_quat(yaw)
    object_root_pos, object_root_quat = _default_object_root(num_frames)

    pose_aa = np.zeros((num_frames, NUM_G1_MOTION_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, 2] = yaw
    pose_aa[:, 1:, :] = G1_DOF_AXIS[None, :, :] * joint_pos[:, :, None]

    return {
        "root_trans_offset": root_trans_offset.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": joint_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
        "object_root_pos": object_root_pos,
        "object_root_quat": object_root_quat,
        "fps": int(fps if fps is not None else _infer_fps(trajectory, num_frames)),
    }


def make_sonic_motion_library(
    num_frames: int = 101,
    duration_s: float = 2.0,
    fps: int = 50,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Create the first SomaForce-Cross G1 scaffold motion library."""

    specs: tuple[tuple[str, ScaffoldTask, InteractionMode], ...] = (
        ("somaforce_g1_door_push", ScaffoldTask.PUSH_PULL_DOOR, "push"),
        ("somaforce_g1_door_pull", ScaffoldTask.PUSH_PULL_DOOR, "pull"),
        ("somaforce_g1_box_push", ScaffoldTask.PUSH_PULL_BOX, "push"),
        ("somaforce_g1_box_pull", ScaffoldTask.PUSH_PULL_BOX, "pull"),
    )
    library = {}
    for key, task, mode in specs:
        if task == ScaffoldTask.PUSH_PULL_DOOR:
            trajectory = make_g1_push_pull_door_trajectory(mode, num_frames, duration_s)
        else:
            trajectory = make_g1_push_pull_box_trajectory(mode, num_frames, duration_s)
        library[key] = motion_trajectory_to_sonic_entry(trajectory, fps=fps)
    return library


def write_sonic_motion_file(
    path: Path,
    motions: dict[str, MotionTrajectory] | None = None,
    num_frames: int = 101,
    duration_s: float = 2.0,
    fps: int = 50,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Write a Sonic-compatible joblib motion file and return its contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if motions is None:
        library = make_sonic_motion_library(num_frames=num_frames, duration_s=duration_s, fps=fps)
    else:
        library = {
            key: motion_trajectory_to_sonic_entry(trajectory, fps=fps)
            for key, trajectory in motions.items()
        }
    joblib.dump(library, path)
    return library


def write_single_task_sonic_motion_file(
    path: Path,
    task: ScaffoldTask,
    mode: InteractionMode,
    num_frames: int = 101,
    duration_s: float = 2.0,
    fps: int = 50,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Write one scaffold task motion for fast Sonic manager-env checks."""

    if task == ScaffoldTask.PUSH_PULL_DOOR:
        trajectory = make_g1_push_pull_door_trajectory(mode, num_frames, duration_s)
    elif task == ScaffoldTask.PUSH_PULL_BOX:
        trajectory = make_g1_push_pull_box_trajectory(mode, num_frames, duration_s)
    else:
        raise ValueError(f"Unsupported scaffold task: {task}")
    key = f"somaforce_g1_{_task_slug(task)}_{mode}"
    return write_sonic_motion_file(path, motions={key: trajectory}, fps=fps)


def make_sonic_object_motion_library(
    motion_keys: tuple[str, ...],
    num_frames: int = 101,
    fps: int = 50,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Create Sonic object-motion entries matching scaffold motion keys."""

    root_pos, root_quat = _default_object_root(num_frames)
    return {
        key: {
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy(),
            "fps": int(fps),
        }
        for key in motion_keys
    }


def write_single_task_sonic_object_motion_file(
    path: Path,
    task: ScaffoldTask,
    mode: InteractionMode,
    num_frames: int = 101,
    fps: int = 50,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Write one Sonic-compatible object motion file for object-enabled checks."""

    key = f"somaforce_g1_{_task_slug(task)}_{mode}"
    library = make_sonic_object_motion_library((key,), num_frames=num_frames, fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(library, path)
    return library


def _root_from_body_ref(
    trajectory: MotionTrajectory,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    root = np.zeros((num_frames, 3), dtype=np.float32)
    root[:, 2] = 0.76
    yaw = np.zeros(num_frames, dtype=np.float32)
    if trajectory.body_pose_w is None:
        return root, yaw

    body_ref = _to_numpy(trajectory.body_pose_w)
    if body_ref.ndim != 3 or body_ref.shape[0] != num_frames or body_ref.shape[-1] < 6:
        raise ValueError("body_pose_w must have shape [T, K, >=6] for Sonic export")
    root = body_ref[:, 0, :3].astype(np.float32)
    yaw = body_ref[:, 0, 5].astype(np.float32)
    return root, yaw


def _yaw_to_xyzw_quat(yaw: np.ndarray) -> np.ndarray:
    half = 0.5 * yaw
    quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    quat[:, 2] = np.sin(half)
    quat[:, 3] = np.cos(half)
    return quat


def _default_object_root(num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a nominal one-object pose sequence for Sonic object-enabled envs."""

    root_pos = np.zeros((num_frames, 1, 3), dtype=np.float32)
    root_pos[:, 0, :] = np.array([0.72, 0.0, 0.75], dtype=np.float32)
    root_quat = np.zeros((num_frames, 1, 4), dtype=np.float32)
    root_quat[:, 0, 0] = 1.0
    return root_pos, root_quat


def _infer_fps(trajectory: MotionTrajectory, num_frames: int) -> int:
    if trajectory.time_s is None:
        return 50
    time_s = _to_numpy(trajectory.time_s)
    duration = float(time_s[-1] - time_s[0])
    if duration <= 0:
        return 50
    return max(1, int(round((num_frames - 1) / duration)))


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _task_slug(task: ScaffoldTask) -> str:
    if task == ScaffoldTask.PUSH_PULL_DOOR:
        return "door"
    if task == ScaffoldTask.PUSH_PULL_BOX:
        return "box"
    raise ValueError(f"Unsupported scaffold task: {task}")
