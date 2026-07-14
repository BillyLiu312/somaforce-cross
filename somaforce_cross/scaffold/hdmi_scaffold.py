"""HDMI-style nominal reference scaffold for SomaForce-Cross Phase 0."""

from __future__ import annotations

import torch

from somaforce_cross.scaffold.contracts import ScaffoldOutput
from somaforce_cross.scaffold.reference_schema import CanonicalReferenceEpisode


class HDMIReferenceScaffold:
    """Expose a canonical robot-object reference as the nominal scaffold policy.

    This class implements the reference/replay boundary used before and around
    HDMI-style co-tracking training. It does not claim to replace HDMI's PPO
    trainer and does not consume force-semantic labels.
    """

    def __init__(
        self,
        reference: CanonicalReferenceEpisode,
        action_offset: torch.Tensor | None = None,
        action_scale: torch.Tensor | None = None,
        action_clip: float | None = 20.0,
    ) -> None:
        self.reference = reference
        action_dim = reference.joint_pos.shape[-1]
        device = reference.joint_pos.device
        dtype = reference.joint_pos.dtype
        self.action_offset = (
            torch.zeros(action_dim, device=device, dtype=dtype)
            if action_offset is None
            else action_offset.to(device=device, dtype=dtype)
        )
        self.action_scale = (
            torch.ones(action_dim, device=device, dtype=dtype)
            if action_scale is None
            else action_scale.to(device=device, dtype=dtype)
        )
        if tuple(self.action_offset.shape) != (action_dim,):
            raise ValueError(f"action_offset must have shape [{action_dim}]")
        if tuple(self.action_scale.shape) != (action_dim,) or torch.any(
            self.action_scale == 0
        ):
            raise ValueError(f"action_scale must have non-zero shape [{action_dim}]")
        self.action_clip = action_clip

    def at_frame(self, frame_index: int | torch.Tensor) -> ScaffoldOutput:
        """Return the canonical nominal scaffold output at one or more frames."""

        joint_pos = self.reference.joint_pos[frame_index]
        a_nom = (joint_pos - self.action_offset) / self.action_scale
        if self.action_clip is not None:
            a_nom = torch.clamp(a_nom, -self.action_clip, self.action_clip)
        valid = self.reference.reference_valid[frame_index]
        confidence = valid.to(dtype=a_nom.dtype)
        if confidence.ndim == 0:
            confidence = confidence.reshape(1)
        body_ref = torch.cat(
            (
                self.reference.body_pos_w[frame_index],
                self.reference.body_quat_w[frame_index],
            ),
            dim=-1,
        )
        return ScaffoldOutput(
            a_nom=a_nom,
            nominal_joint_pos=joint_pos,
            nominal_hand_ref=self.reference.hand_pose_w[frame_index],
            nominal_body_ref=body_ref,
            cmd_6d=self.reference.cmd_6d[frame_index],
            confidence=confidence,
            task=self.reference.metadata.task,
            object_root_ref=self.reference.object_root_pose_w[frame_index],
            object_joint_ref=self.reference.object_joint_pos[frame_index],
            contact_target_obj=self.reference.contact_target_obj[frame_index],
            contact_intent=self.reference.contact_intent[frame_index],
            contact_confidence=self.reference.contact_confidence[frame_index],
            phase=self.reference.phase[frame_index],
            reference_valid=valid,
            reference_source=self.reference.metadata.source.value,
        )

    def at_time(self, time_s: float | torch.Tensor) -> ScaffoldOutput:
        query = torch.as_tensor(time_s, device=self.reference.time_s.device)
        indices = torch.argmin(
            torch.abs(self.reference.time_s[:, None] - query.reshape(1, -1)), dim=0
        )
        if indices.numel() == 1:
            return self.at_frame(int(indices.item()))
        return self.at_frame(indices)
