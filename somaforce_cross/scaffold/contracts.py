"""Scaffold data contracts.

The scaffold owns nominal task motion only. It must not consume force semantics or
privileged contact labels; SomaForce-Cross adds the bounded force residual later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


G1_BODY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_FULL_JOINT_NAMES: tuple[str, ...] = G1_BODY_JOINT_NAMES


class ScaffoldTask(str, Enum):
    """First diagnostic scaffold tasks."""

    PUSH_PULL_DOOR = "push_pull_door"
    HEAVY_PAYLOAD = "heavy_payload"
    PUSH_PULL_BOX = "push_pull_box"


@dataclass(frozen=True)
class MotionTrajectory:
    """A nominal G1 motion trajectory sampled at fixed or known timestamps.

    Attributes:
        joint_pos: Joint targets in G1 joint order, shaped ``[T, J]`` or
            ``[B, T, J]``.
        time_s: Optional timestamps, shaped ``[T]``. If omitted, frames are
            treated as unit-spaced and selected by integer frame index.
        hand_pose_w: Optional hand/end-effector reference, shaped
            ``[T, H, D]`` or ``[B, T, H, D]``.
        body_pose_w: Optional body reference, shaped ``[T, K, D]`` or
            ``[B, T, K, D]``.
        cmd_6d: Optional task command vector, shaped ``[6]``, ``[T, 6]``,
            ``[B, 6]``, or ``[B, T, 6]``.
    """

    joint_pos: torch.Tensor
    time_s: torch.Tensor | None = None
    hand_pose_w: torch.Tensor | None = None
    body_pose_w: torch.Tensor | None = None
    cmd_6d: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.joint_pos.ndim not in (2, 3):
            raise ValueError(
                "MotionTrajectory.joint_pos must have shape [T, J] or [B, T, J], "
                f"got {tuple(self.joint_pos.shape)}"
            )
        if self.joint_pos.shape[-1] != len(G1_FULL_JOINT_NAMES):
            raise ValueError(
                "MotionTrajectory.joint_pos must use G1 29-DOF order, got "
                f"{self.joint_pos.shape[-1]} joints"
            )
        if self.time_s is not None and self.time_s.ndim != 1:
            raise ValueError("MotionTrajectory.time_s must have shape [T]")
        if self.time_s is not None and self.time_s.shape[0] != self.joint_pos.shape[-2]:
            raise ValueError("MotionTrajectory.time_s length must match joint_pos time dimension")


@dataclass(frozen=True)
class ScaffoldOutput:
    """Nominal scaffold output consumed by the force residual stack."""

    a_nom: torch.Tensor
    nominal_joint_pos: torch.Tensor
    nominal_hand_ref: torch.Tensor | None = None
    nominal_body_ref: torch.Tensor | None = None
    cmd_6d: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    task: ScaffoldTask | None = None
    object_root_ref: torch.Tensor | None = None
    object_joint_ref: torch.Tensor | None = None
    contact_target_obj: torch.Tensor | None = None
    contact_intent: torch.Tensor | None = None
    contact_confidence: torch.Tensor | None = None
    phase: torch.Tensor | None = None
    reference_valid: torch.Tensor | None = None
    reference_source: str | None = None
