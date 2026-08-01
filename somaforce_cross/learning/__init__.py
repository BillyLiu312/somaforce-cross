"""Phase 5 deployable-input learning components.

This package remains importable with only PyTorch.  The training factory owns
the delayed RSL-RL and Isaac imports required by a real rollout.
"""

from somaforce_cross.learning.actor_critic import (
    PrivilegedCritic,
    ResidualActor,
    ResidualActorCritic,
)
from somaforce_cross.learning.config import Phase5Config, load_phase5_config
from somaforce_cross.learning.losses import (
    SemanticLossOutput,
    semantic_losses,
    weighted_soft_target_kl,
)
from somaforce_cross.learning.semantic_ppo import SemanticPPO

__all__ = [
    "Phase5Config",
    "PrivilegedCritic",
    "ResidualActor",
    "ResidualActorCritic",
    "SemanticLossOutput",
    "SemanticPPO",
    "load_phase5_config",
    "semantic_losses",
    "weighted_soft_target_kl",
]
