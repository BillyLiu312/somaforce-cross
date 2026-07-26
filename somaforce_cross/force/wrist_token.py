"""Wrist history storage and shared causal temporal token encoding."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


_WRISTS = 2
_HISTORY = 16
_FEATURES = 14


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_floating(value: torch.Tensor, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


class WristHistoryBuffer:
    """Fixed-size oldest-to-newest history for both wrist tokens.

    The buffer is intentionally a small state holder rather than an ``nn.Module``.
    ``storage`` and ``history`` both expose the ``[B, 2, 16, 14]`` tensor.
    """

    def __init__(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point dtype")
        self._storage = torch.zeros(
            batch_size,
            _WRISTS,
            _HISTORY,
            _FEATURES,
            device=device,
            dtype=dtype,
        )

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    @property
    def history(self) -> torch.Tensor:
        return self._storage

    @property
    def buffer(self) -> torch.Tensor:
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

    def _validate_frame(self, frame: object, name: str, expected_batch: int) -> torch.Tensor:
        frame = _require_tensor(frame, name)
        _require_floating(frame, name)
        if frame.shape != (expected_batch, _WRISTS, _FEATURES):
            raise ValueError(
                f"{name} must have shape [{expected_batch}, 2, 14], got {tuple(frame.shape)}"
            )
        if frame.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {frame.device}")
        if frame.dtype != self.dtype:
            raise ValueError(f"{name} must have dtype {self.dtype}, got {frame.dtype}")
        if not torch.isfinite(frame).all():
            raise ValueError(f"{name} must contain only finite values")
        return frame

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        """Append the current frame and return the updated storage."""
        frame = self._validate_frame(frame, "frame", self.batch_size)
        with torch.no_grad():
            self._storage[:, :, :-1].copy_(self._storage[:, :, 1:])
            self._storage[:, :, -1].copy_(frame)
        return self._storage

    def reset(
        self,
        env_ids: Iterable[int] | torch.Tensor,
        initial_frame: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reset selected environments, optionally broadcasting initial frames."""
        if isinstance(env_ids, int) and not isinstance(env_ids, bool):
            ids = torch.tensor([env_ids], device=self.device, dtype=torch.long)
        elif isinstance(env_ids, torch.Tensor):
            if env_ids.dtype == torch.bool:
                if env_ids.ndim != 1 or env_ids.numel() != self.batch_size:
                    raise ValueError("boolean env_ids must have shape [batch_size]")
                ids = torch.nonzero(env_ids, as_tuple=False).flatten().to(self.device)
            else:
                if env_ids.ndim != 1:
                    raise ValueError("env_ids must be one-dimensional")
                if env_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    raise TypeError("env_ids must contain integer indices")
                ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            try:
                ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
            except TypeError as exc:
                raise TypeError("env_ids must be an iterable of integer indices") from exc
            if ids.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
        if ids.numel() and ((ids < 0).any() or (ids >= self.batch_size).any()):
            raise IndexError("env_ids contains an index outside the buffer")
        if ids.numel() and torch.unique(ids).numel() != ids.numel():
            raise ValueError("env_ids must not contain duplicates")

        if initial_frame is None:
            with torch.no_grad():
                self._storage[ids] = 0
            return self._storage

        initial_frame = _require_tensor(initial_frame, "initial_frame")
        _require_floating(initial_frame, "initial_frame")
        if initial_frame.device != self.device:
            raise ValueError(
                f"initial_frame must be on {self.device}, got {initial_frame.device}"
            )
        if initial_frame.dtype != self.dtype:
            raise ValueError(
                f"initial_frame must have dtype {self.dtype}, got {initial_frame.dtype}"
            )
        if not torch.isfinite(initial_frame).all():
            raise ValueError("initial_frame must contain only finite values")
        if initial_frame.shape == (self.batch_size, _WRISTS, _FEATURES):
            frames = initial_frame[ids]
        elif initial_frame.shape == (ids.numel(), _WRISTS, _FEATURES):
            frames = initial_frame
        else:
            raise ValueError(
                "initial_frame must have shape [batch_size, 2, 14] or [len(env_ids), 2, 14]"
            )
        with torch.no_grad():
            self._storage[ids] = frames[:, :, None, :]
        return self._storage


class _CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.dilation * (self.kernel_size - 1), 0))
        return self.activation(self.conv(x))


class WristTokenEncoder(nn.Module):
    """Encode both wrists with one shared three-block causal TCN."""

    def __init__(
        self,
        input_dim: int = _FEATURES,
        channels: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        side_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if input_dim != _FEATURES:
            raise ValueError(f"V1 wrist token input_dim must be {_FEATURES}")
        if channels != 128 or kernel_size != 3 or tuple(dilations) != (1, 2, 4):
            raise ValueError("V1 WristTokenEncoder requires channels=128, kernel=3, dilations=(1, 2, 4)")
        if side_embedding_dim != 8:
            raise ValueError("V1 WristTokenEncoder requires an 8-D side embedding")
        blocks = []
        in_channels = input_dim
        for dilation in dilations:
            blocks.append(_CausalConvBlock(in_channels, channels, kernel_size, dilation))
            in_channels = channels
        self.tcn = nn.ModuleList(blocks)
        self.side_embedding = nn.Parameter(torch.zeros(_WRISTS, side_embedding_dim))
        nn.init.normal_(self.side_embedding, mean=0.0, std=0.02)
        self.output_dim = channels
        self.fused_dim = _WRISTS * (channels + side_embedding_dim)

    @property
    def shared_tcn(self) -> nn.ModuleList:
        return self.tcn

    def forward(self, wrist_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _require_tensor(wrist_history, "wrist_history")
        _require_floating(wrist_history, "wrist_history")
        if wrist_history.ndim != 4 or wrist_history.shape[1:] != (_WRISTS, _HISTORY, _FEATURES):
            raise ValueError(
                "wrist_history must have shape [B, 2, 16, 14], "
                f"got {tuple(wrist_history.shape)}"
            )
        if not torch.isfinite(wrist_history).all():
            raise ValueError("wrist_history must contain only finite values")
        batch = wrist_history.shape[0]
        x = wrist_history.reshape(batch * _WRISTS, _HISTORY, _FEATURES).transpose(1, 2)
        for block in self.tcn:
            x = block(x)
        # The last two positions have receptive fields covering frames 0..14
        # and 1..15 respectively; their union covers the complete history.
        per_wrist = x[:, :, -2:].mean(dim=-1).reshape(batch, _WRISTS, self.output_dim)
        side = self.side_embedding.to(dtype=per_wrist.dtype, device=per_wrist.device)
        fused = torch.cat(
            (per_wrist, side.unsqueeze(0).expand(batch, -1, -1)),
            dim=-1,
        ).reshape(batch, self.fused_dim)
        return per_wrist, fused
