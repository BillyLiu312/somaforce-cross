"""Legacy baseline adapter from Sonic ``TrackingCommand`` to scaffold outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from somaforce_cross.scaffold.contracts import ScaffoldOutput, ScaffoldTask


@dataclass(frozen=True)
class SonicScaffoldAdapter:
    """Expose Sonic manager-env references as a compatibility ``a_nom`` path.

    The adapter expects an Isaac Lab ``ManagerBasedRLEnv`` with Sonic's
    ``motion`` command term. Imports from ``gear_sonic.envs.manager_env.mdp`` are
    intentionally lazy because those modules require an active Isaac/Omni app.
    The selected Phase 0 route is HDMI + OMOMO; this adapter remains for baseline
    comparisons and existing smoke tests.
    """

    env: Any
    command_name: str = "motion"
    task: ScaffoldTask | None = None
    action_clip: float | None = 20.0

    def get_output(self) -> ScaffoldOutput:
        """Read Sonic references and return the nominal scaffold action."""

        motion_cmd = self.env.command_manager.get_term(self.command_name)
        a_nom = self._residual_joint_pos_action(self.env, self.command_name)
        if self.action_clip is not None:
            a_nom = torch.clamp(a_nom, -self.action_clip, self.action_clip)

        confidence = self._confidence_like(a_nom, motion_cmd)
        return ScaffoldOutput(
            a_nom=a_nom,
            nominal_joint_pos=motion_cmd.joint_pos,
            nominal_hand_ref=self._optional_attr(motion_cmd, "vr_3point_body_pos_w_multi_future"),
            nominal_body_ref=self._optional_attr(motion_cmd, "body_pos_w_multi_future"),
            cmd_6d=self._optional_attr(motion_cmd, "command_multi_future"),
            confidence=confidence,
            task=self.task,
        )

    @staticmethod
    def _residual_joint_pos_action(env: Any, command_name: str) -> torch.Tensor:
        from gear_sonic.envs.manager_env.mdp.observations import residual_joint_pos_action

        return residual_joint_pos_action(env, command_name=command_name)

    @staticmethod
    def _optional_attr(obj: Any, name: str) -> Any | None:
        return getattr(obj, name, None)

    @staticmethod
    def _confidence_like(a_nom: torch.Tensor, motion_cmd: Any) -> torch.Tensor:
        validity = getattr(motion_cmd, "validity", None)
        if validity is not None:
            return validity.to(device=a_nom.device, dtype=a_nom.dtype)
        return torch.ones(a_nom.shape[0], device=a_nom.device, dtype=a_nom.dtype)
