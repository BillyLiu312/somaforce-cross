"""Pure PyTorch residual action composition for SomaForce-Cross V1."""

from somaforce_cross.residual.action_composer import (
    ContactGatedResidual,
    ContactGatedResidualOutput,
    ResidualActionComposer,
    ResidualActionOutput,
)
from somaforce_cross.residual.action_history import ExecutedActionHistory
from somaforce_cross.residual.authority import ContactGainRamp, PerJointAuthority
from somaforce_cross.residual.safety import (
    JointMarginLimiter,
    ResidualInterval,
    VelocityMarginLimiter,
)

__all__ = [
    "ContactGainRamp",
    "ContactGatedResidual",
    "ContactGatedResidualOutput",
    "ExecutedActionHistory",
    "JointMarginLimiter",
    "PerJointAuthority",
    "ResidualActionComposer",
    "ResidualActionOutput",
    "ResidualInterval",
    "VelocityMarginLimiter",
]
