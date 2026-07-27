"""Explicit Phase 4 episode parameters and independent per-env random streams."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import NamedTuple

import torch

from somaforce_cross.sensing.virtual_ft import VirtualFTSensorParameters


_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def validate_env_ids(
    env_ids: int | Iterable[int] | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return unique selected rows as a one-dimensional device int64 tensor."""
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    target_device = torch.device(device)

    if isinstance(env_ids, bool):
        raise TypeError("env_ids must not be a boolean scalar")
    if isinstance(env_ids, Integral):
        ids = torch.tensor([int(env_ids)], device=target_device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        if env_ids.dtype == torch.bool:
            if env_ids.shape != (batch_size,):
                raise ValueError("boolean env_ids must have shape [batch_size]")
            ids = torch.nonzero(env_ids, as_tuple=False).flatten().to(target_device)
        else:
            if env_ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            if env_ids.dtype not in _INTEGER_DTYPES:
                raise TypeError("env_ids must contain integer indices")
            ids = env_ids.to(device=target_device, dtype=torch.long)
    else:
        if isinstance(env_ids, (str, bytes)):
            raise TypeError("env_ids must be an integer or iterable of integers")
        try:
            values = list(env_ids)
        except TypeError as exc:
            raise TypeError(
                "env_ids must be an integer or iterable of integers"
            ) from exc
        if any(isinstance(value, (list, tuple, torch.Tensor)) for value in values):
            raise ValueError("env_ids must be one-dimensional")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in values
        ):
            raise TypeError("env_ids must contain integer indices")
        ids = torch.tensor(
            [int(value) for value in values],
            device=target_device,
            dtype=torch.long,
        )

    if ids.numel() and torch.any((ids < 0) | (ids >= batch_size)):
        raise IndexError("env_ids contains an index outside the state")
    if ids.numel() and torch.unique(ids).numel() != ids.numel():
        raise ValueError("env_ids must not contain duplicates")
    return ids


def _selected_float32_rows(
    value: object,
    name: str,
    *,
    rows: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.shape != (rows, width):
        raise ValueError(
            f"{name} must have shape [{rows}, {width}], got {list(value.shape)}"
        )
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


class EpisodeParameterStore:
    """Store approved critic mismatch rows without sampling any defaults."""

    physics_width = 39
    scaffold_width = 2
    sensor_width = 90

    def __init__(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        self.physics_mismatch = torch.zeros(
            batch_size, self.physics_width, device=device, dtype=torch.float32
        )
        self.scaffold_mismatch = torch.zeros(
            batch_size, self.scaffold_width, device=device, dtype=torch.float32
        )
        self.sensor_mismatch = torch.zeros(
            batch_size, self.sensor_width, device=device, dtype=torch.float32
        )

    @property
    def batch_size(self) -> int:
        return self.physics_mismatch.shape[0]

    @property
    def device(self) -> torch.device:
        return self.physics_mismatch.device

    def validate_rows(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        physics_mismatch: torch.Tensor,
        scaffold_mismatch: torch.Tensor,
        sensor_mismatch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate every selected parameter group without changing storage."""
        ids = validate_env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        count = ids.numel()
        physics = _selected_float32_rows(
            physics_mismatch,
            "physics_mismatch",
            rows=count,
            width=self.physics_width,
            device=self.device,
        )
        scaffold = _selected_float32_rows(
            scaffold_mismatch,
            "scaffold_mismatch",
            rows=count,
            width=self.scaffold_width,
            device=self.device,
        )
        sensor_raw = _selected_float32_rows(
            sensor_mismatch,
            "sensor_mismatch",
            rows=count,
            width=self.sensor_width,
            device=self.device,
        )
        sensor = VirtualFTSensorParameters.canonicalize(
            VirtualFTSensorParameters.from_flat(sensor_raw),
            device=self.device,
            dtype=torch.float32,
        ).flatten()
        return ids, physics, scaffold, sensor

    def _apply_validated(
        self,
        ids: torch.Tensor,
        physics_mismatch: torch.Tensor,
        scaffold_mismatch: torch.Tensor,
        sensor_mismatch: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.physics_mismatch[ids] = physics_mismatch
            self.scaffold_mismatch[ids] = scaffold_mismatch
            self.sensor_mismatch[ids] = sensor_mismatch

    def apply(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        physics_mismatch: torch.Tensor,
        scaffold_mismatch: torch.Tensor,
        sensor_mismatch: torch.Tensor,
    ) -> None:
        """Atomically replace only the selected explicit episode rows."""
        prepared = self.validate_rows(
            env_ids, physics_mismatch, scaffold_mismatch, sensor_mismatch
        )
        self._apply_validated(*prepared)


class VirtualFTDraws(NamedTuple):
    """Unit variates consumed explicitly by ``VirtualFTSensor``."""

    drift: torch.Tensor
    noise: torch.Tensor
    dropout: torch.Tensor


class PerEnvRandomStream:
    """Independent generators for existing VirtualFT normal/uniform draws.

    This class samples no episode parameters and defines no curriculum. Each
    environment must be explicitly seeded before its first draw.
    """

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
        if dtype != torch.float32:
            raise TypeError("PerEnvRandomStream dtype must be torch.float32")
        self.batch_size = batch_size
        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        self.seeds = torch.full((batch_size,), -1, device=self.device, dtype=torch.long)
        self._seeded = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        self._generators = [
            torch.Generator(device=self.device) for _ in range(batch_size)
        ]

    def validate_seeds(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        seeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = validate_env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        if not isinstance(seeds, torch.Tensor):
            raise TypeError("seeds must be a torch.Tensor")
        if seeds.ndim != 1 or seeds.shape[0] != ids.numel():
            raise ValueError(f"seeds must have shape [{ids.numel()}]")
        if seeds.dtype not in _INTEGER_DTYPES:
            raise TypeError("seeds must have an integer dtype")
        if seeds.device != self.device:
            raise ValueError(f"seeds must be on {self.device}, got {seeds.device}")
        seeds_i64 = seeds.to(dtype=torch.long)
        if torch.any(seeds_i64 < 0):
            raise ValueError("seeds must be nonnegative")
        return ids, seeds_i64

    def _reseed_validated(self, ids: torch.Tensor, seeds: torch.Tensor) -> None:
        for env_id, seed in zip(ids.tolist(), seeds.tolist(), strict=True):
            self._generators[env_id].manual_seed(seed)
        with torch.no_grad():
            self.seeds[ids] = seeds
            self._seeded[ids] = True

    def reseed(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        seeds: torch.Tensor,
    ) -> None:
        """Reset only selected generator streams to caller-provided seeds."""
        ids, prepared_seeds = self.validate_seeds(env_ids, seeds)
        self._reseed_validated(ids, prepared_seeds)

    def reset(
        self,
        env_ids: int | Iterable[int] | torch.Tensor,
        seeds: torch.Tensor,
    ) -> None:
        """Alias the explicit reset/reseed contract."""
        self.reseed(env_ids, seeds)

    def draw_virtual_ft(
        self,
        env_ids: int | Iterable[int] | torch.Tensor | None = None,
    ) -> VirtualFTDraws:
        """Draw per-env drift/noise standard normals and dropout uniforms."""
        if env_ids is None:
            ids = torch.arange(self.batch_size, device=self.device, dtype=torch.long)
        else:
            ids = validate_env_ids(
                env_ids, batch_size=self.batch_size, device=self.device
            )
        if ids.numel() and not self._seeded[ids].all():
            missing = ids[~self._seeded[ids]].tolist()
            raise RuntimeError(
                f"random streams must be seeded before drawing: {missing}"
            )

        count = ids.numel()
        drift = torch.empty(count, 2, 6, device=self.device, dtype=self.dtype)
        noise = torch.empty_like(drift)
        dropout = torch.empty(count, 2, device=self.device, dtype=self.dtype)
        for row, env_id in enumerate(ids.tolist()):
            generator = self._generators[env_id]
            drift[row] = torch.randn(
                (2, 6),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            noise[row] = torch.randn(
                (2, 6),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            dropout[row] = torch.rand(
                (2,),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
        return VirtualFTDraws(drift=drift, noise=noise, dropout=dropout)
