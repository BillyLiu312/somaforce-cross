"""Canonical HDMI/OMOMO robot-object reference contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import torch

from somaforce_cross.scaffold.contracts import (
    G1_FULL_JOINT_NAMES,
    ScaffoldTask,
)


class ReferenceSource(str, Enum):
    """Provenance of a canonical scaffold reference."""

    HDMI = "hdmi"
    OMOMO = "omomo"
    INTERNAL = "internal"


@dataclass(frozen=True)
class ReferenceMetadata:
    """Episode-level provenance and conversion metadata."""

    episode_id: str
    source: ReferenceSource
    source_clip_id: str
    task: ScaffoldTask
    fps: float = 50.0
    source_revision: str = "unknown"
    retarget_version: str = "native"
    retarget_quality: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.source_clip_id:
            raise ValueError("source_clip_id must be non-empty")
        if self.fps <= 0:
            raise ValueError("fps must be positive")


@dataclass(frozen=True)
class CanonicalReferenceEpisode:
    """One validated 50 Hz G1/object/contact reference episode.

    Quaternions use scalar-first ``wxyz`` order. Pose tensors use
    ``[x, y, z, qw, qx, qy, qz]``.
    """

    metadata: ReferenceMetadata
    time_s: torch.Tensor
    phase: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    hand_pose_w: torch.Tensor
    object_root_pose_w: torch.Tensor
    object_root_twist_w: torch.Tensor
    object_joint_pos: torch.Tensor
    object_joint_vel: torch.Tensor
    contact_target_obj: torch.Tensor
    contact_intent: torch.Tensor
    contact_confidence: torch.Tensor
    reference_valid: torch.Tensor
    cmd_6d: torch.Tensor

    def __post_init__(self) -> None:
        t = self.joint_pos.shape[0] if self.joint_pos.ndim >= 1 else 0
        if t < 2:
            raise ValueError("CanonicalReferenceEpisode requires at least two frames")
        self._expect("time_s", self.time_s, (t,))
        self._expect("phase", self.phase, (t,))
        self._expect("joint_pos", self.joint_pos, (t, len(G1_FULL_JOINT_NAMES)))
        self._expect("joint_vel", self.joint_vel, (t, len(G1_FULL_JOINT_NAMES)))
        self._expect_suffix("body_pos_w", self.body_pos_w, t, (3,))
        self._expect_suffix("body_quat_w", self.body_quat_w, t, (4,))
        if self.body_pos_w.shape[1] == 0:
            raise ValueError("canonical reference requires at least one robot body")
        if self.body_pos_w.shape[1] != self.body_quat_w.shape[1]:
            raise ValueError("body_pos_w and body_quat_w body counts must match")
        self._expect("hand_pose_w", self.hand_pose_w, (t, 2, 7))
        self._expect("object_root_pose_w", self.object_root_pose_w, (t, 7))
        self._expect("object_root_twist_w", self.object_root_twist_w, (t, 6))
        self._expect_suffix("object_joint_pos", self.object_joint_pos, t, ())
        self._expect_suffix("object_joint_vel", self.object_joint_vel, t, ())
        if self.object_joint_pos.shape != self.object_joint_vel.shape:
            raise ValueError("object_joint_pos and object_joint_vel shapes must match")
        self._expect("contact_target_obj", self.contact_target_obj, (t, 2, 3))
        self._expect("contact_intent", self.contact_intent, (t, 2))
        self._expect("contact_confidence", self.contact_confidence, (t, 2))
        self._expect("reference_valid", self.reference_valid, (t,))
        self._expect("cmd_6d", self.cmd_6d, (t, 6))
        if self.contact_intent.dtype != torch.bool:
            raise ValueError("contact_intent must be bool")
        if self.reference_valid.dtype != torch.bool:
            raise ValueError("reference_valid must be bool")
        if torch.any((self.contact_confidence < 0) | (self.contact_confidence > 1)):
            raise ValueError("contact_confidence must be in [0, 1]")
        if torch.any((self.phase < 0) | (self.phase > 1)):
            raise ValueError("phase must be in [0, 1]")
        if torch.any(self.time_s[1:] <= self.time_s[:-1]):
            raise ValueError("time_s must be strictly increasing")
        if not self._all_finite():
            raise ValueError("canonical reference contains non-finite values")
        self._check_quaternions("body_quat_w", self.body_quat_w)
        self._check_quaternions("hand_pose_w", self.hand_pose_w[..., 3:])
        self._check_quaternions("object_root_pose_w", self.object_root_pose_w[..., 3:])

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]

    @property
    def num_object_joints(self) -> int:
        return self.object_joint_pos.shape[1]

    def as_serializable(self) -> dict[str, Any]:
        """Return a torch-save-compatible dictionary with explicit metadata."""

        return {
            "schema_version": 1,
            "metadata": {
                "episode_id": self.metadata.episode_id,
                "source": self.metadata.source.value,
                "source_clip_id": self.metadata.source_clip_id,
                "task": self.metadata.task.value,
                "fps": self.metadata.fps,
                "source_revision": self.metadata.source_revision,
                "retarget_version": self.metadata.retarget_version,
                "retarget_quality": dict(self.metadata.retarget_quality),
            },
            **{name: getattr(self, name) for name in _TENSOR_FIELDS},
        }

    @classmethod
    def from_serializable(cls, data: Mapping[str, Any]) -> CanonicalReferenceEpisode:
        """Reconstruct and revalidate an episode from serialized data."""

        if data.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported reference schema version: {data.get('schema_version')}"
            )
        raw = data["metadata"]
        metadata = ReferenceMetadata(
            episode_id=str(raw["episode_id"]),
            source=ReferenceSource(str(raw["source"])),
            source_clip_id=str(raw["source_clip_id"]),
            task=ScaffoldTask(str(raw["task"])),
            fps=float(raw["fps"]),
            source_revision=str(raw["source_revision"]),
            retarget_version=str(raw["retarget_version"]),
            retarget_quality={
                str(key): float(value) for key, value in raw["retarget_quality"].items()
            },
        )
        return cls(metadata=metadata, **{name: data[name] for name in _TENSOR_FIELDS})

    @staticmethod
    def _expect(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )

    @staticmethod
    def _expect_suffix(
        name: str, tensor: torch.Tensor, frames: int, suffix: tuple[int, ...]
    ) -> None:
        if (
            tensor.ndim < 2
            or tensor.shape[0] != frames
            or tuple(tensor.shape[2:]) != suffix
        ):
            expected = (
                f"[{frames}, K"
                + (", " + ", ".join(map(str, suffix)) if suffix else "")
                + "]"
            )
            raise ValueError(
                f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
            )

    def _all_finite(self) -> bool:
        return all(
            bool(torch.all(torch.isfinite(getattr(self, name))))
            for name in _TENSOR_FIELDS
            if getattr(self, name).dtype != torch.bool
        )

    @staticmethod
    def _check_quaternions(name: str, quaternion: torch.Tensor) -> None:
        norm = torch.linalg.vector_norm(quaternion, dim=-1)
        if torch.any(torch.abs(norm - 1.0) > 1e-3):
            raise ValueError(f"{name} contains quaternions with norm error > 1e-3")


_TENSOR_FIELDS = (
    "time_s",
    "phase",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "hand_pose_w",
    "object_root_pose_w",
    "object_root_twist_w",
    "object_joint_pos",
    "object_joint_vel",
    "contact_target_obj",
    "contact_intent",
    "contact_confidence",
    "reference_valid",
    "cmd_6d",
)


def finite_difference(values: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
    """Differentiate a frame-major tensor while preserving its shape."""

    if values.shape[0] != time_s.shape[0]:
        raise ValueError("values and time_s frame counts must match")
    dt = time_s[1:] - time_s[:-1]
    view_shape = (dt.shape[0],) + (1,) * (values.ndim - 1)
    velocity = torch.empty_like(values)
    velocity[1:] = (values[1:] - values[:-1]) / dt.reshape(view_shape)
    velocity[0] = velocity[1]
    return velocity


def reconstruct_contact_targets(
    hand_pose_w: torch.Tensor,
    object_root_pose_w: torch.Tensor,
) -> torch.Tensor:
    """Transform world-frame hand positions into the object frame."""

    if tuple(hand_pose_w.shape[1:]) != (2, 7):
        raise ValueError("hand_pose_w must have shape [T, 2, 7]")
    if tuple(object_root_pose_w.shape) != (hand_pose_w.shape[0], 7):
        raise ValueError("object_root_pose_w must have shape [T, 7]")
    relative_w = hand_pose_w[..., :3] - object_root_pose_w[:, None, :3]
    object_quat = object_root_pose_w[:, None, 3:].expand(-1, 2, -1)
    return quat_rotate_wxyz(quat_conjugate_wxyz(object_quat), relative_w)


def quat_conjugate_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate 3D vectors by scalar-first unit quaternions."""

    q_w = quaternion[..., :1]
    q_v = quaternion[..., 1:]
    return vector + 2.0 * (
        q_w * torch.cross(q_v, vector, dim=-1)
        + torch.cross(q_v, torch.cross(q_v, vector, dim=-1), dim=-1)
    )
