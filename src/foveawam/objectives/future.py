"""Per-position regression and visibility losses."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional

from foveawam.models.common.numeric_checks import _check_numeric
from foveawam.objectives import _LossResult


def _position_mean(values: Tensor, valid: Tensor) -> _LossResult:
    if values.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("per-position loss and mask shapes differ")
    _check_numeric(torch.isfinite(values).all(), "per-position loss must be finite")
    count = valid.sum(dtype=torch.int64)
    return _LossResult(values.masked_fill(~valid, 0).sum() / count.clamp_min(1), count)


def evaluate_future_regression(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    channel_axis: int,
) -> _LossResult:
    """Mean squared error: first average channels, then valid positions."""
    if (
        prediction.shape != target.shape
        or prediction.dtype != torch.float32
        or target.dtype != torch.float32
    ):
        raise ValueError("future regression requires matching float32 target/prediction shapes")
    if prediction.device != target.device or prediction.device != valid.device:
        raise ValueError("future objective tensors must share a device")
    # Native MSE reuses the existing inputs for backward instead of retaining a
    # full-size residual. Keep the original per-channel and valid-position means.
    values = functional.mse_loss(prediction, target, reduction="none").mean(dim=channel_axis)
    return _position_mean(values, valid)


def evaluate_future_displacements(
    prediction: Tensor,
    displacement_robot_base_m: Tensor,
    visible: Tensor,
    valid: Tensor,
    *,
    scale_m: float,
    visibility_weight: float = 1.0,
) -> _LossResult:
    """Mean component MSE plus visibility BCE, with explicit meter scale."""
    if not math.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("direct 3D displacement scale must be explicitly positive")
    if not math.isfinite(visibility_weight) or visibility_weight < 0:
        raise ValueError("visibility loss weight must be finite and nonnegative")
    if prediction.shape != (*valid.shape, 4) or displacement_robot_base_m.shape != (
        *valid.shape,
        3,
    ):
        raise ValueError("direct 3D prediction requires xyz and visibility at fixed point IDs")
    if visible.shape != valid.shape or visible.dtype != torch.bool:
        raise ValueError("direct 3D visibility axes differ")
    values = (prediction[..., :3] - displacement_robot_base_m / scale_m).square().mean(-1)
    if visibility_weight:
        values = values + visibility_weight * functional.binary_cross_entropy_with_logits(
            prediction[..., 3], visible.float(), reduction="none"
        )
    return _position_mean(values, valid)
