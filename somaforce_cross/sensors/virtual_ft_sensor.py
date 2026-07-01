"""Virtual wrist F/T sensor model.

This module should model the deployable wrist/end-effector F/T observation used by the
student policy. It should not expose privileged object-side contact truth to the actor.
"""

from dataclasses import dataclass

import torch


@dataclass
class VirtualFTSensorConfig:
    """Configuration for a degraded virtual 6-axis wrist F/T sensor."""

    force_limit: float = 300.0
    torque_limit: float = 80.0
    noise_std_force: float = 0.0
    noise_std_torque: float = 0.0
    bias_force: float = 0.0
    bias_torque: float = 0.0


class VirtualFTSensor:
    """Applies deployable F/T sensor degradation to clean wrist wrench signals."""

    def __init__(self, cfg: VirtualFTSensorConfig | None = None):
        self.cfg = cfg or VirtualFTSensorConfig()

    def observe(self, clean_wrench: torch.Tensor) -> torch.Tensor:
        """Return a degraded wrench observation.

        Args:
            clean_wrench: Tensor with shape ``[..., 6]`` in sensor frame.

        Returns:
            Tensor with shape ``[..., 6]`` representing deployable F/T observation.
        """

        if clean_wrench.shape[-1] != 6:
            raise ValueError("clean_wrench must have last dimension 6: [Fx,Fy,Fz,Tx,Ty,Tz]")

        force = clean_wrench[..., :3]
        torque = clean_wrench[..., 3:]

        if self.cfg.noise_std_force > 0:
            force = force + torch.randn_like(force) * self.cfg.noise_std_force
        if self.cfg.noise_std_torque > 0:
            torque = torque + torch.randn_like(torque) * self.cfg.noise_std_torque

        force = force + self.cfg.bias_force
        torque = torque + self.cfg.bias_torque
        force = torch.clamp(force, -self.cfg.force_limit, self.cfg.force_limit)
        torque = torch.clamp(torque, -self.cfg.torque_limit, self.cfg.torque_limit)
        return torch.cat([force, torque], dim=-1)
