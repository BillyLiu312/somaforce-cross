"""Runtime safety gate for bounded residual deployment."""

from dataclasses import dataclass

import torch


@dataclass
class RuntimeSafetyConfig:
    force_limit: float = 250.0
    force_rate_limit: float = 500.0
    min_confidence: float = 0.2


class RuntimeSafetyGate:
    """Scales or blocks residual actions based on safety signals."""

    def __init__(self, cfg: RuntimeSafetyConfig | None = None):
        self.cfg = cfg or RuntimeSafetyConfig()

    def residual_gain(self, force_norm: torch.Tensor, alpha_conf: torch.Tensor) -> torch.Tensor:
        safe_force = (force_norm < self.cfg.force_limit).to(force_norm.dtype)
        confident = (alpha_conf > self.cfg.min_confidence).to(alpha_conf.dtype)
        return safe_force * confident
