"""Common task-adapter contract for Phase 4 vectorized environments."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch


def stable_env_seeds(
    base_seed: int,
    env_ids: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map each selected environment to ``base_seed + env_id``.

    The mapping is intentionally independent of the order of ``env_ids`` so a
    selected reset cannot change an environment's random stream identity.
    """
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if not isinstance(env_ids, torch.Tensor):
        raise TypeError("env_ids must be a torch.Tensor")
    if env_ids.ndim != 1 or env_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("env_ids must be a one-dimensional integer tensor")
    target = env_ids.device if device is None else torch.device(device)
    ids = env_ids.to(device=target, dtype=torch.long)
    if torch.any(ids < 0):
        raise ValueError("env_ids must be nonnegative")
    return ids + base_seed


def door_nominal_physics_mismatch(
    batch_size: int,
    *,
    mechanism_friction: float,
    mechanism_damping: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the frozen door 39-D critic row in the audited field order."""
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    values = torch.zeros(batch_size, 39, device=device, dtype=torch.float32)
    values[:, 3] = mechanism_friction
    values[:, 4] = mechanism_damping
    return values


class TaskProgressSignals(NamedTuple):
    """Fixed task-agnostic progress, contact, and stability fields."""

    progress: torch.Tensor
    progress_delta: torch.Tensor
    success: torch.Tensor
    expected_contact: torch.Tensor
    contact_truth: torch.Tensor
    stability_margin: torch.Tensor
    failure: torch.Tensor
    reference_exhausted: torch.Tensor


class TaskAdapter(Protocol):
    """Door/payload data boundary without task-specific residual networks."""

    task_name: str
    reference_step: torch.Tensor

    def validate_reset(self, env_ids: torch.Tensor) -> None:
        """Validate a selected reset without mutating state."""

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset exactly the selected reference and progress rows."""

    def write_scene_reset(self, env_ids: torch.Tensor) -> None:
        """Write selected robot and object state to the simulator."""

    def advance(self) -> None:
        """Advance reference and progress once after a control step."""

    def build_scaffold_observation(self) -> object:
        """Build the artifact-specific frozen scaffold observation."""

    def build_object_state(self) -> torch.Tensor:
        """Return the common critic object state ``[B,16]``."""

    def progress_signals(self) -> TaskProgressSignals:
        """Return the fixed common task signals."""


class SmokeDoneOutput(NamedTuple):
    """Separated terminated and timeout tensors for bounded smoke runs."""

    terminated: torch.Tensor
    time_outs: torch.Tensor


def smoke_done_flags(
    nonfinite_state: torch.Tensor,
    task_failure: torch.Tensor,
    episode_length: torch.Tensor,
    reference_exhausted: torch.Tensor,
    *,
    episode_length_steps: int,
) -> SmokeDoneOutput:
    """Apply the approved Phase 4B3 smoke-only done contract."""
    for value, name in (
        (nonfinite_state, "nonfinite_state"),
        (task_failure, "task_failure"),
        (reference_exhausted, "reference_exhausted"),
    ):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean torch.Tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} must have shape [B]")
    if not isinstance(episode_length, torch.Tensor):
        raise TypeError("episode_length must be a torch.Tensor")
    if episode_length.ndim != 1 or episode_length.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("episode_length must be a one-dimensional integer tensor")
    shape = nonfinite_state.shape
    tensors = (task_failure, reference_exhausted, episode_length)
    if any(value.shape != shape for value in tensors):
        raise ValueError("all done inputs must have the same batch shape")
    if any(value.device != nonfinite_state.device for value in tensors):
        raise ValueError("all done inputs must be on the same device")
    if (
        not isinstance(episode_length_steps, int)
        or isinstance(episode_length_steps, bool)
        or episode_length_steps <= 0
    ):
        raise ValueError("episode_length_steps must be a positive integer")
    return SmokeDoneOutput(
        terminated=nonfinite_state | task_failure,
        time_outs=(episode_length >= episode_length_steps) | reference_exhausted,
    )
