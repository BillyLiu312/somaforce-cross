"""Sonic-style task scaffold.

The scaffold provides nominal task motion ``a_nom``. SomaForce-Cross learns only a
bounded force-conditioned residual over this nominal motion.
"""

from dataclasses import dataclass

import torch


@dataclass
class ScaffoldOutput:
    """Nominal scaffold outputs consumed by the residual policy."""

    a_nom: torch.Tensor
    hand_reference: torch.Tensor | None = None
    body_reference: torch.Tensor | None = None


class SonicScaffold:
    """Placeholder Sonic-style base motion scaffold."""

    def __call__(
        self,
        task_id: torch.Tensor,
        cmd_6d: torch.Tensor,
        robot_state: torch.Tensor,
        object_prior: torch.Tensor | None = None,
    ) -> ScaffoldOutput:
        """Return a nominal action.

        This placeholder returns zeros with a shape inferred from ``cmd_6d``. Replace it
        with a Sonic/base-policy adapter or a parameterized trajectory library.
        """

        del task_id, object_prior
        batch_shape = cmd_6d.shape[:-1]
        action_dim = robot_state.shape[-1] if robot_state.ndim > 0 else 1
        a_nom = torch.zeros(*batch_shape, action_dim, device=cmd_6d.device, dtype=cmd_6d.dtype)
        return ScaffoldOutput(a_nom=a_nom)
