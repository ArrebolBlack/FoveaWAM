"""Shared rotary-position encoding mathematics."""

from __future__ import annotations

import torch
from torch import Tensor

from foveawam.models.common.numeric_checks import _check_numeric

_ROPE_BASE = 10_000.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rotate_axis(value: Tensor, coordinate: Tensor, *, start: int, pairs: int) -> Tensor:
    width = 2 * pairs
    selected = value[..., start : start + width]
    channels = torch.arange(width, device=value.device)
    frequencies = torch.pow(
        torch.full((), _ROPE_BASE, dtype=value.dtype, device=value.device),
        -(channels // 2).to(value.dtype) / pairs,
    )
    angles = coordinate[:, None, :, None].to(value.dtype) * frequencies
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    # Explicit channel partners also preserve indexing when Inductor fuses the
    # strided QKV producer and padding masks with this rotation.
    partner = selected.index_select(-1, channels ^ 1)
    signs = (2 * (channels % 2) - 1).to(value.dtype)
    rotated = selected * cosine + partner * (sine * signs)
    return torch.cat((value[..., :start], rotated, value[..., start + width :]), dim=-1)


def _validate_3d_rope_shapes(query: Tensor, key: Tensor, position_m: Tensor) -> int:
    _require(query.shape == key.shape and query.ndim == 4, "3D-RoPE Q/K shapes must agree")
    batch, _, tokens, head_dimension = (int(axis) for axis in query.shape)
    _require(
        tuple(position_m.shape) == (batch, tokens, 3),
        "3D-RoPE positions must have axes [N,L,3]",
    )
    return head_dimension


def _require_finite_3d_rope_positions(position_m: Tensor) -> None:
    _check_numeric(
        torch.isfinite(position_m).all(),
        "3D-RoPE positions must be finite",
        _compiled_async=not torch.are_deterministic_algorithms_enabled(),
    )


def _apply_3d_rope_math(
    query: Tensor,
    key: Tensor,
    position_m: Tensor,
    *,
    head_dimension: int,
) -> tuple[Tensor, Tensor]:
    pairs = head_dimension // 6
    _require(pairs > 0, "3D-RoPE attention head dimension must be at least 6")

    def rotate(value: Tensor) -> Tensor:
        # Autocast can promote the first concatenation. Keep that cast boundary
        # and its backward rounding before rotating the two remaining bands.
        value = _rotate_axis(value, position_m[..., 0], start=0, pairs=pairs)
        width = 2 * pairs
        channels = torch.arange(width, device=value.device)
        frequencies = torch.pow(
            torch.full((), _ROPE_BASE, dtype=value.dtype, device=value.device),
            -(channels // 2).to(value.dtype) / pairs,
        )
        angles = torch.stack(
            (
                position_m[..., 1, None].to(value.dtype) * frequencies,
                position_m[..., 2, None].to(value.dtype) * frequencies,
            ),
            dim=-2,
        )[:, None]
        selected = value[..., width : 3 * width].reshape(*value.shape[:-1], 2, width)
        partner = selected.index_select(-1, channels ^ 1)
        signs = (2 * (channels % 2) - 1).to(value.dtype)
        rotated = selected * torch.cos(angles) + partner * (torch.sin(angles) * signs)
        return torch.cat(
            (value[..., :width], rotated.flatten(start_dim=-2), value[..., 3 * width :]),
            dim=-1,
        )

    return rotate(query), rotate(key)


def _apply_3d_rope(query: Tensor, key: Tensor, position_m: Tensor) -> tuple[Tensor, Tensor]:
    """Apply equal, disjoint x/y/z rotary bands to spatial-attention Q and K."""

    head_dimension = _validate_3d_rope_shapes(query, key, position_m)
    _require_finite_3d_rope_positions(position_m)
    return _apply_3d_rope_math(
        query,
        key,
        position_m,
        head_dimension=head_dimension,
    )
