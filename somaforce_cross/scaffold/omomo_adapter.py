"""Convert jointly retargeted OMOMO clips to the canonical scaffold schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from somaforce_cross.scaffold.contracts import ScaffoldTask
from somaforce_cross.scaffold.reference_schema import (
    CanonicalReferenceEpisode,
    ReferenceMetadata,
    ReferenceSource,
    finite_difference,
    reconstruct_contact_targets,
)


def load_retargeted_omomo_reference(
    path: Path,
    *,
    episode_id: str,
    source_clip_id: str,
    retarget_version: str,
    fps: float = 50.0,
    source_revision: str = "unknown",
    retarget_quality: Mapping[str, float] | None = None,
) -> CanonicalReferenceEpisode:
    """Load a retargeter output NPZ, not a raw SMPL-H OMOMO archive."""

    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return omomo_retarget_mapping_to_reference(
        data,
        episode_id=episode_id,
        source_clip_id=source_clip_id,
        retarget_version=retarget_version,
        fps=fps,
        source_revision=source_revision,
        retarget_quality=retarget_quality,
    )


def omomo_retarget_mapping_to_reference(
    data: Mapping[str, Any],
    *,
    episode_id: str,
    source_clip_id: str,
    retarget_version: str,
    fps: float = 50.0,
    source_revision: str = "unknown",
    retarget_quality: Mapping[str, float] | None = None,
) -> CanonicalReferenceEpisode:
    """Validate a joint G1/object retarget result and reconstruct contact targets."""

    if not retarget_version or retarget_version in ("native", "unknown"):
        raise ValueError(
            "OMOMO conversion requires an explicit non-native retarget_version"
        )

    required = (
        "joint_pos",
        "body_pos_w",
        "body_quat_w",
        "hand_pose_w",
        "object_root_pose_w",
        "contact_intent",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(
            "Retargeted OMOMO reference is missing fields. Raw OMOMO is not accepted: "
            f"{missing}"
        )

    joint_pos = _float(data["joint_pos"])
    frames = joint_pos.shape[0]
    time_s = _time(data.get("time_s"), frames, fps)
    joint_vel = (
        _float(data["joint_vel"])
        if "joint_vel" in data
        else finite_difference(joint_pos, time_s)
    )
    body_pos = _float(data["body_pos_w"])
    body_quat = _float(data["body_quat_w"])
    hand_pose = _float(data["hand_pose_w"])
    object_pose = _float(data["object_root_pose_w"])
    object_twist = _object_twist(data, object_pose, time_s)
    contact_intent = torch.as_tensor(data["contact_intent"]).bool()
    contact_confidence = (
        _float(data["contact_confidence"])
        if "contact_confidence" in data
        else contact_intent.float()
    )
    reference_valid = (
        torch.as_tensor(data["reference_valid"]).bool()
        if "reference_valid" in data
        else torch.ones(frames, dtype=torch.bool)
    )
    phase = (
        _float(data["phase"]) if "phase" in data else torch.linspace(0.0, 1.0, frames)
    )
    cmd_6d = _cmd(data.get("cmd_6d"), object_pose, frames)
    contact_target = reconstruct_contact_targets(hand_pose, object_pose)

    return CanonicalReferenceEpisode(
        metadata=ReferenceMetadata(
            episode_id=episode_id,
            source=ReferenceSource.OMOMO,
            source_clip_id=source_clip_id,
            task=ScaffoldTask.HEAVY_PAYLOAD,
            fps=fps,
            source_revision=source_revision,
            retarget_version=retarget_version,
            retarget_quality=retarget_quality or {},
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
        object_joint_pos=torch.empty((frames, 0), dtype=joint_pos.dtype),
        object_joint_vel=torch.empty((frames, 0), dtype=joint_pos.dtype),
        contact_target_obj=contact_target,
        contact_intent=contact_intent,
        contact_confidence=contact_confidence,
        reference_valid=reference_valid,
        cmd_6d=cmd_6d,
    )


def _time(raw: Any | None, frames: int, fps: float) -> torch.Tensor:
    if raw is None:
        return torch.arange(frames, dtype=torch.float32) / fps
    return _float(raw)


def _object_twist(
    data: Mapping[str, Any], object_pose: torch.Tensor, time_s: torch.Tensor
) -> torch.Tensor:
    if "object_root_twist_w" in data:
        return _float(data["object_root_twist_w"])
    twist = torch.zeros((object_pose.shape[0], 6), dtype=object_pose.dtype)
    twist[:, :3] = finite_difference(object_pose[:, :3], time_s)
    return twist


def _cmd(raw: Any | None, object_pose: torch.Tensor, frames: int) -> torch.Tensor:
    if raw is not None:
        cmd = _float(raw)
        if tuple(cmd.shape) == (6,):
            return cmd.expand(frames, -1).clone()
        return cmd
    displacement = object_pose[-1, :3] - object_pose[0, :3]
    norm = torch.linalg.vector_norm(displacement)
    direction = displacement / norm if norm > 1e-6 else torch.zeros_like(displacement)
    cmd = torch.zeros((frames, 6), dtype=object_pose.dtype)
    cmd[:, :3] = direction
    return cmd


def _float(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)
