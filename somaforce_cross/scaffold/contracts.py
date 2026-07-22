"""Joint naming and task identity shared by canonical reference adapters."""

from __future__ import annotations

from enum import Enum


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
    """Tasks supported by the retained canonical reference boundary."""

    PUSH_PULL_DOOR = "push_pull_door"
    PUSH_BOX = "push_box"
    MOVE_SUITCASE = "move_suitcase"
    HEAVY_PAYLOAD = "heavy_payload"
