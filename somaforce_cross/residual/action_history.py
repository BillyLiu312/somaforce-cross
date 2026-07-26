"""Executed normalized-action history for residual actor observations."""

from __future__ import annotations

from typing import Iterable

import torch

from somaforce_cross.residual.authority import _env_ids, _validate_batch_size


_JOINTS = 23
_HISTORY = 3


class ExecutedActionHistory:
    """Store three executed actions newest-to-oldest for every environment."""

    def __init__(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        _validate_batch_size(batch_size)
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

    def push(self, a_total: torch.Tensor) -> torch.Tensor:
        if not isinstance(a_total, torch.Tensor):
            raise TypeError("a_total must be a torch.Tensor")
        if not a_total.is_floating_point():
            raise TypeError("a_total must have a floating-point dtype")
        if a_total.shape != (self.batch_size, _JOINTS):
            raise ValueError(
                f"a_total must have shape [{self.batch_size}, 23], got {tuple(a_total.shape)}"
            )
        if a_total.device != self.device:
            raise ValueError(f"a_total must be on {self.device}, got {a_total.device}")
        if a_total.dtype != self.dtype:
            raise ValueError(f"a_total must have dtype {self.dtype}, got {a_total.dtype}")
        if not torch.isfinite(a_total).all():
            raise ValueError("a_total must contain only finite values")
        with torch.no_grad():
            self._storage[:, :, 1:].copy_(self._storage[:, :, :-1].clone())
            self._storage[:, :, 0].copy_(a_total)
        return self._storage

    def reset(self, env_ids: int | Iterable[int] | torch.Tensor) -> torch.Tensor:
        ids = _env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        with torch.no_grad():
            self._storage[ids] = 0
        return self._storage
