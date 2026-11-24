"""Shared time, index, and physical-vector embedding primitives."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from foveawam.models.common.numeric_checks import _check_numeric

_FOURIER_BASE = 10_000.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class _TimeEmbedding(nn.Module):
    """Fixed Fourier features for finite relative times in seconds."""

    frequencies: Tensor

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        _require(hidden_size > 0, "time embedding size must be positive")
        frequency_count = (hidden_size + 1) // 2
        denominator = max(frequency_count - 1, 1)
        frequencies = torch.exp(
            -math.log(_FOURIER_BASE)
            * torch.arange(frequency_count, dtype=torch.float32)
            / denominator
        )
        self.hidden_size = hidden_size
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, relative_time_s: Tensor) -> Tensor:
        _require(relative_time_s.dtype == torch.float32, "relative time must be float32 seconds")
        _check_numeric(torch.isfinite(relative_time_s).all(), "relative time must be finite")
        angles = relative_time_s[..., None] * self.frequencies
        features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return features[..., : self.hidden_size].contiguous()


class _IndexEmbedding(nn.Module):
    """Learned embedding for a closed, zero-based view or horizon index."""

    def __init__(self, count: int, hidden_size: int) -> None:
        super().__init__()
        _require(count > 0 and hidden_size > 0, "index embedding dimensions must be positive")
        self.count = count
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(count, hidden_size)

    def forward(self, index: Tensor) -> Tensor:
        _require(index.dtype == torch.int64, "embedding indices must be int64")
        _require(
            bool(((index >= 0) & (index < self.count)).all().item()),
            "embedding index is outside its closed range",
        )
        return cast(Tensor, self.embedding(index))


class _VectorEmbedding(nn.Module):
    """Project one finite float32 physical vector into model width."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        _require(input_size > 0 and hidden_size > 0, "vector embedding dimensions must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection = nn.Linear(input_size, hidden_size)

    def _validate(self, value: Tensor) -> None:
        _require(value.dtype == torch.float32, "physical vector embedding input must be float32")
        _require(
            value.ndim > 0 and int(value.shape[-1]) == self.input_size,
            "physical vector embedding input has the wrong final dimension",
        )
        _check_numeric(
            torch.isfinite(value).all(),
            "physical vector embedding input must be finite",
        )

    def forward(self, value: Tensor, *, _validated: bool = False) -> Tensor:
        if not _validated:
            self._validate(value)
        return cast(Tensor, self.projection(value))
