"""Nominal normalized-action history owned by the Phase 4 environment."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from somaforce_cross.envs.mismatch import validate_env_ids


_JOINTS = 23
_HISTORY = 3


class NominalActionHistory:
    """Store ``a_nom`` action-major and newest-to-oldest, separately from execution."""

    def __init__(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point dtype")
        self._storage = torch.zeros(
            batch_size, _JOINTS, _HISTORY, device=device, dtype=dtype
        )

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    @property
    def history(self) -> torch.Tensor:
        return self._storage

    @property
    def batch_size(self) -> int:
        return self._storage.shape[0]

    @property
    def device(self) -> torch.device:
        return self._storage.device

    @property
    def dtype(self) -> torch.dtype:
        return self._storage.dtype

    def _validate_actions(self, value: object, name: str, *, rows: int) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if value.shape != (rows, _JOINTS):
            raise ValueError(
                f"{name} must have shape [{rows}, 23], got {list(value.shape)}"
            )
        if value.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {value.device}")
        if value.dtype != self.dtype:
            raise ValueError(f"{name} must have dtype {self.dtype}, got {value.dtype}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return value

    def push(self, a_nom: torch.Tensor) -> torch.Tensor:
        """Push a full-batch current nominal action into slot zero."""
        actions = self._validate_actions(a_nom, "a_nom", rows=self.batch_size)
        with torch.no_grad():
            self._storage[:, :, 1:].copy_(self._storage[:, :, :-1].clone())
            self._storage[:, :, 0].copy_(actions)
        return self._storage

    def set_current(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        current_a_nom: torch.Tensor,
    ) -> torch.Tensor:
        """Write only selected slot-zero rows without changing older slots."""
        ids = validate_env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        actions = self._validate_actions(
            current_a_nom, "current_a_nom", rows=ids.numel()
        )
        with torch.no_grad():
            self._storage[ids, :, 0] = actions
        return self._storage

    def reset(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        current_a_nom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Clear selected rows, optionally installing their explicit current action."""
        ids = validate_env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        actions = None
        if current_a_nom is not None:
            actions = self._validate_actions(
                current_a_nom, "current_a_nom", rows=ids.numel()
            )
        with torch.no_grad():
            self._storage[ids] = 0
            if actions is not None:
                self._storage[ids, :, 0] = actions
        return self._storage
