"""Frame and origin transform for V1 wrist joint reaction wrenches."""

from __future__ import annotations

import torch
from torch import nn


def _floating_tensor(value: object, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.shape != shape:
        raise ValueError(
            f"{name} must have shape {list(shape)}, got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _compatible(value: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {value.device}")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} must have dtype {reference.dtype}, got {value.dtype}")


def _normalized_quaternion(value: torch.Tensor, name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(norm == 0):
        raise ValueError(f"{name} must contain only non-zero quaternions")
    return value / norm


def _quat_apply(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.linalg.cross(q_xyz, vector, dim=-1)
    return vector + q[..., :1] * t + torch.linalg.cross(q_xyz, t, dim=-1)


class WristWrenchTransform(nn.Module):
    """Convert parent-on-child joint wrenches to V1 distal-load base-yaw wrenches.

    Quaternions use Hamilton ``wxyz`` ordering and actively rotate local vectors
    into world. Positions are in meters, forces in N, and moments in Nm. The
    caller supplies each cached child-joint pose; no body/joint identity transform
    is assumed. Inputs are always ordered left wrist then right wrist.
    """

    load_sign = -1.0

    def forward(
        self,
        raw_joint_wrench: torch.Tensor,
        joint_quat_w: torch.Tensor,
        joint_pos_w: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        pelvis_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(raw_joint_wrench, torch.Tensor):
            raise TypeError("raw_joint_wrench must be a torch.Tensor")
        if (
            raw_joint_wrench.ndim != 3
            or raw_joint_wrench.shape[0] == 0
            or raw_joint_wrench.shape[1:] != (2, 6)
        ):
            raise ValueError(
                f"raw_joint_wrench must have shape [B, 2, 6], got {tuple(raw_joint_wrench.shape)}"
            )
        if not raw_joint_wrench.is_floating_point():
            raise TypeError("raw_joint_wrench must have a floating-point dtype")
        if not torch.isfinite(raw_joint_wrench).all():
            raise ValueError("raw_joint_wrench must contain only finite values")

        batch_size = raw_joint_wrench.shape[0]
        values = (
            (
                _floating_tensor(joint_quat_w, "joint_quat_w", (batch_size, 2, 4)),
                "joint_quat_w",
            ),
            (
                _floating_tensor(joint_pos_w, "joint_pos_w", (batch_size, 2, 3)),
                "joint_pos_w",
            ),
            (
                _floating_tensor(sensor_pos_w, "sensor_pos_w", (batch_size, 2, 3)),
                "sensor_pos_w",
            ),
            (
                _floating_tensor(pelvis_quat_w, "pelvis_quat_w", (batch_size, 4)),
                "pelvis_quat_w",
            ),
        )
        for value, name in values:
            _compatible(value, raw_joint_wrench, name)

        joint_quat_w = _normalized_quaternion(joint_quat_w, "joint_quat_w")
        pelvis_quat_w = _normalized_quaternion(pelvis_quat_w, "pelvis_quat_w")

        force_w = self.load_sign * _quat_apply(joint_quat_w, raw_joint_wrench[..., :3])
        moment_w_joint = self.load_sign * _quat_apply(
            joint_quat_w, raw_joint_wrench[..., 3:]
        )
        moment_w_sensor = moment_w_joint + torch.linalg.cross(
            joint_pos_w - sensor_pos_w, force_w, dim=-1
        )

        w, x, y, z = pelvis_quat_w.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        half_yaw = 0.5 * yaw
        zeros = torch.zeros_like(half_yaw)
        yaw_quat_w = torch.stack(
            (torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1
        )
        yaw_quat_w = yaw_quat_w[:, None, :].expand(-1, 2, -1)
        yaw_quat_inverse = yaw_quat_w.clone()
        yaw_quat_inverse[..., 1:] *= -1.0
        force_yaw = _quat_apply(yaw_quat_inverse, force_w)
        moment_yaw = _quat_apply(yaw_quat_inverse, moment_w_sensor)
        return torch.cat((force_yaw, moment_yaw), dim=-1)
