"""Convert released HDMI robot-object motion files to the canonical schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from somaforce_cross.scaffold.contracts import G1_FULL_JOINT_NAMES, ScaffoldTask
from somaforce_cross.scaffold.reference_schema import (
    CanonicalReferenceEpisode,
    ReferenceMetadata,
    ReferenceSource,
    finite_difference,
    reconstruct_contact_targets,
)


def load_hdmi_reference(
    path: Path,
    *,
    episode_id: str,
    source_clip_id: str,
    task: ScaffoldTask,
    hand_body_indices: tuple[int, int],
    object_body_index: int = -1,
    contact_hand_index: int = 1,
    fps: float = 50.0,
    source_revision: str = "unknown",
) -> CanonicalReferenceEpisode:
    """Load an HDMI ``motion.npz`` file and convert it to the shared contract."""

    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return hdmi_mapping_to_reference(
        data,
        episode_id=episode_id,
        source_clip_id=source_clip_id,
        task=task,
        hand_body_indices=hand_body_indices,
        object_body_index=object_body_index,
        contact_hand_index=contact_hand_index,
        fps=fps,
        source_revision=source_revision,
    )


def hdmi_mapping_to_reference(
    data: Mapping[str, Any],
    *,
    episode_id: str,
    source_clip_id: str,
    task: ScaffoldTask,
    hand_body_indices: tuple[int, int],
    object_body_index: int = -1,
    contact_hand_index: int = 1,
    fps: float = 50.0,
    source_revision: str = "unknown",
) -> CanonicalReferenceEpisode:
    """Convert HDMI arrays without importing HDMI or Isaac Lab."""

    required = ("joint_pos", "body_pos_w", "body_quat_w")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"HDMI reference is missing required fields: {missing}")
    if contact_hand_index not in (0, 1):
        raise ValueError("contact_hand_index must be 0 or 1")

    joint_pos_all = _float_tensor(data["joint_pos"])
    body_pos_all = _float_tensor(data["body_pos_w"])
    body_quat_all = _float_tensor(data["body_quat_w"])
    if joint_pos_all.ndim != 2 or joint_pos_all.shape[1] < len(G1_FULL_JOINT_NAMES):
        raise ValueError("HDMI joint_pos must have shape [T, >=29]")
    frames = joint_pos_all.shape[0]
    if tuple(body_pos_all.shape[:1]) != (frames,) or body_pos_all.ndim != 3:
        raise ValueError("HDMI body_pos_w must have shape [T, B, 3]")
    if tuple(body_quat_all.shape) != (*body_pos_all.shape[:2], 4):
        raise ValueError("HDMI body_quat_w must have shape [T, B, 4]")

    object_index = _normalize_index(object_body_index, body_pos_all.shape[1])
    hand_indices = tuple(
        _normalize_index(index, body_pos_all.shape[1]) for index in hand_body_indices
    )
    if object_index in hand_indices:
        raise ValueError("object_body_index cannot also be a hand body index")
    robot_indices = [
        index for index in range(body_pos_all.shape[1]) if index != object_index
    ]

    time_s = torch.arange(frames, dtype=joint_pos_all.dtype) / fps
    joint_pos = joint_pos_all[:, : len(G1_FULL_JOINT_NAMES)]
    joint_vel = _optional_float(data, "joint_vel")
    if joint_vel is None:
        joint_vel = finite_difference(joint_pos, time_s)
    else:
        joint_vel = joint_vel[:, : len(G1_FULL_JOINT_NAMES)]

    body_pos = body_pos_all[:, robot_indices]
    body_quat = body_quat_all[:, robot_indices]
    hand_pose = torch.cat(
        (body_pos_all[:, hand_indices, :], body_quat_all[:, hand_indices, :]), dim=-1
    )
    object_pose = torch.cat(
        (body_pos_all[:, object_index, :], body_quat_all[:, object_index, :]), dim=-1
    )
    object_twist = _object_twist(data, object_index, object_pose, time_s)

    object_joint_pos = joint_pos_all[:, len(G1_FULL_JOINT_NAMES) :]
    raw_joint_vel = _optional_float(data, "joint_vel")
    if object_joint_pos.shape[1] == 0:
        object_joint_vel = torch.empty((frames, 0), dtype=joint_pos.dtype)
    elif raw_joint_vel is not None and raw_joint_vel.shape[1] >= joint_pos_all.shape[1]:
        object_joint_vel = raw_joint_vel[
            :, len(G1_FULL_JOINT_NAMES) : joint_pos_all.shape[1]
        ]
    else:
        object_joint_vel = finite_difference(object_joint_pos, time_s)

    contact_intent, contact_confidence = _contact_fields(
        data.get("object_contact"), frames, contact_hand_index
    )
    contact_target = reconstruct_contact_targets(hand_pose, object_pose)
    reference_valid = _finite_frame_mask(
        joint_pos, body_pos, body_quat, hand_pose, object_pose, object_twist
    )
    phase = torch.linspace(0.0, 1.0, frames, dtype=joint_pos.dtype)
    cmd_6d = _cmd_6d(data.get("cmd_6d"), object_pose, frames)

    return CanonicalReferenceEpisode(
        metadata=ReferenceMetadata(
            episode_id=episode_id,
            source=ReferenceSource.HDMI,
            source_clip_id=source_clip_id,
            task=task,
            fps=fps,
            source_revision=source_revision,
            retarget_version="native",
        ),
        time_s=time_s,
        phase=phase,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        hand_pose_w=hand_pose,
        object_root_pose_w=object_pose,
        object_root_twist_w=object_twist,
        object_joint_pos=object_joint_pos,
        object_joint_vel=object_joint_vel,
        contact_target_obj=contact_target,
        contact_intent=contact_intent,
        contact_confidence=contact_confidence,
        reference_valid=reference_valid,
        cmd_6d=cmd_6d,
    )


def _object_twist(
    data: Mapping[str, Any],
    object_index: int,
    object_pose: torch.Tensor,
    time_s: torch.Tensor,
) -> torch.Tensor:
    linear = _optional_float(data, "body_lin_vel_w")
    angular = _optional_float(data, "body_ang_vel_w")
    if linear is not None and angular is not None:
        return torch.cat((linear[:, object_index], angular[:, object_index]), dim=-1)
    result = torch.zeros((object_pose.shape[0], 6), dtype=object_pose.dtype)
    result[:, :3] = finite_difference(object_pose[:, :3], time_s)
    return result


def _contact_fields(
    raw: Any | None,
    frames: int,
    contact_hand_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    intent = torch.zeros((frames, 2), dtype=torch.bool)
    confidence = torch.zeros((frames, 2), dtype=torch.float32)
    if raw is None:
        return intent, confidence
    contact = torch.as_tensor(raw)
    if contact.ndim == 1:
        contact = contact[:, None]
    if contact.shape[0] != frames or contact.shape[1] not in (1, 2):
        raise ValueError("HDMI object_contact must have shape [T], [T, 1], or [T, 2]")
    active = contact > 0
    if active.shape[1] == 1:
        intent[:, contact_hand_index] = active[:, 0]
        confidence[:, contact_hand_index] = active[:, 0].float()
    else:
        intent = active.bool()
        confidence = active.float()
    return intent, confidence


def _cmd_6d(raw: Any | None, object_pose: torch.Tensor, frames: int) -> torch.Tensor:
    if raw is not None:
        cmd = _float_tensor(raw)
        if tuple(cmd.shape) == (6,):
            return cmd.expand(frames, -1).clone()
        if tuple(cmd.shape) == (frames, 6):
            return cmd
        raise ValueError("cmd_6d must have shape [6] or [T, 6]")
    displacement = object_pose[-1, :3] - object_pose[0, :3]
    norm = torch.linalg.vector_norm(displacement)
    direction = displacement / norm if norm > 1e-6 else torch.zeros_like(displacement)
    cmd = torch.zeros((frames, 6), dtype=object_pose.dtype)
    cmd[:, :3] = direction
    return cmd


def _finite_frame_mask(*tensors: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(tensors[0].shape[0], dtype=torch.bool)
    for tensor in tensors:
        mask &= torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1)
    return mask


def _optional_float(data: Mapping[str, Any], key: str) -> torch.Tensor | None:
    return _float_tensor(data[key]) if key in data else None


def _float_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _normalize_index(index: int, size: int) -> int:
    normalized = index + size if index < 0 else index
    if not 0 <= normalized < size:
        raise ValueError(f"body index {index} is out of range for {size} bodies")
    return normalized
