"""Reconstruct the currently visible object crop with an RGB L1 loss."""

from __future__ import annotations

import torch
from torch import Tensor

from foveawam.models.common.numeric_checks import _check_numeric_many
from foveawam.objectives import _LossResult


def evaluate_object_reconstruction(
    prediction: Tensor, target: Tensor, valid: Tensor, prediction_mask: Tensor
) -> _LossResult:
    if (
        prediction.dtype != torch.float32
        or target.dtype != torch.float32
        or prediction.ndim != 5
        or prediction.shape != target.shape
        or tuple(prediction.shape[2:]) != (3, 32, 32)
    ):
        raise ValueError("object reconstruction RGB must be float32 [B,S,3,32,32]")
    if (
        valid.dtype != torch.bool
        or tuple(valid.shape) != (*prediction.shape[:2], 1, 32, 32)
        or prediction_mask.dtype != torch.bool
        or prediction_mask.shape != prediction.shape[:2]
        or not prediction.device == target.device == valid.device == prediction_mask.device
    ):
        raise ValueError("object reconstruction masks differ in axes/dtype/device")
    checks = torch.stack(
        (
            ~(valid & ~prediction_mask[..., None, None, None]).any(),
            torch.isfinite(prediction).all()
            & torch.isfinite(target).all()
            & ((prediction >= 0) & (prediction <= 1)).all()
            & ((target >= 0) & (target <= 1)).all(),
            ~((~valid) & target.ne(0)).any(),
        )
    )
    messages = (
        "object reconstruction validity extends into prediction padding",
        "object reconstruction RGB must be finite in [0,1]",
        "object reconstruction target must be zero outside its mask",
    )
    _check_numeric_many(tuple(zip(checks.unbind(), messages, strict=True)))
    local_numerator = ((prediction - target).abs() * valid).sum()
    count = valid.sum(dtype=torch.int64) * 3
    return _LossResult(local_numerator / count.clamp_min(1), count)
