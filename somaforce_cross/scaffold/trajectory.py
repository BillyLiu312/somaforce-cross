"""Motion-trajectory scaffold for early SomaForce-Cross diagnostics."""

from __future__ import annotations

import torch

from somaforce_cross.scaffold.contracts import MotionTrajectory, ScaffoldOutput, ScaffoldTask


class MotionTrajectoryScaffold:
    """Generate ``a_nom`` from a G1 motion trajectory.

    This class is intentionally independent from Isaac Lab. It supports fast unit
    tests and lets us validate the scaffold contract before the Sonic manager env
    is fully running. In an Isaac Lab rollout, ``SonicScaffoldAdapter`` should be
    used to convert Sonic's ``TrackingCommand`` reference into the same output.
    """

    def __init__(
        self,
        trajectory: MotionTrajectory,
        task: ScaffoldTask,
        action_offset: torch.Tensor | None = None,
        action_scale: torch.Tensor | None = None,
        action_clip: float | None = 20.0,
    ) -> None:
        self.trajectory = trajectory
        self.task = task
        device = trajectory.joint_pos.device
        dtype = trajectory.joint_pos.dtype
        action_dim = trajectory.joint_pos.shape[-1]
        self.action_offset = (
            action_offset.to(device=device, dtype=dtype)
            if action_offset is not None
            else torch.zeros(action_dim, device=device, dtype=dtype)
        )
        self.action_scale = (
            action_scale.to(device=device, dtype=dtype)
            if action_scale is not None
            else torch.ones(action_dim, device=device, dtype=dtype)
        )
        if self.action_offset.shape != (action_dim,):
            raise ValueError(f"action_offset must have shape [{action_dim}]")
        if self.action_scale.shape != (action_dim,):
            raise ValueError(f"action_scale must have shape [{action_dim}]")
        if torch.any(self.action_scale == 0):
            raise ValueError("action_scale must be non-zero for every joint")
        self.action_clip = action_clip

    def at_frame(self, frame_index: int | torch.Tensor) -> ScaffoldOutput:
        """Return nominal scaffold output at an integer frame index."""

        joint_pos = self._select_frame(self.trajectory.joint_pos, frame_index)
        hand_ref = self._select_optional(self.trajectory.hand_pose_w, frame_index)
        body_ref = self._select_optional(self.trajectory.body_pose_w, frame_index)
        a_nom = (joint_pos - self.action_offset) / self.action_scale
        if self.action_clip is not None:
            a_nom = torch.clamp(a_nom, -self.action_clip, self.action_clip)
        batch = a_nom.shape[0] if a_nom.ndim == 2 else 1
        confidence = torch.ones(batch, device=a_nom.device, dtype=a_nom.dtype)
        return ScaffoldOutput(
            a_nom=a_nom,
            nominal_joint_pos=joint_pos,
            nominal_hand_ref=hand_ref,
            nominal_body_ref=body_ref,
            cmd_6d=self.trajectory.cmd_6d,
            confidence=confidence,
            task=self.task,
        )

    def at_time(self, time_s: float | torch.Tensor) -> ScaffoldOutput:
        """Return nominal scaffold output at the nearest trajectory timestamp."""

        if self.trajectory.time_s is None:
            raise ValueError("Cannot query by time because trajectory.time_s is not set")
        query = torch.as_tensor(time_s, device=self.trajectory.time_s.device)
        frame_index = torch.argmin(torch.abs(self.trajectory.time_s[:, None] - query.reshape(1, -1)), dim=0)
        if frame_index.numel() == 1:
            return self.at_frame(int(frame_index.item()))
        return self.at_frame(frame_index)

    @staticmethod
    def _select_frame(tensor: torch.Tensor, frame_index: int | torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor[frame_index]
        if isinstance(frame_index, int):
            return tensor[:, frame_index]
        env_ids = torch.arange(tensor.shape[0], device=tensor.device)
        return tensor[env_ids, frame_index.to(device=tensor.device)]

    @classmethod
    def _select_optional(
        cls, tensor: torch.Tensor | None, frame_index: int | torch.Tensor
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.ndim == 3:
            return tensor[frame_index]
        if isinstance(frame_index, int):
            return tensor[:, frame_index]
        env_ids = torch.arange(tensor.shape[0], device=tensor.device)
        return tensor[env_ids, frame_index.to(device=tensor.device)]
