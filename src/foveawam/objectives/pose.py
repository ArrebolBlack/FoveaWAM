"""Predict relative target-object translation and rotation in the robot base."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from foveawam.models.common.numeric_checks import _check_numeric
from foveawam.models.io import _rotation_code_quaternion
from foveawam.objectives import _LossResult


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _quaternion_product(left: Tensor, right: Tensor) -> Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:]
    right_xyz, right_w = right[..., :3], right[..., 3:]
    xyz = left_w * right_xyz + right_w * left_xyz + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    scalar = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, scalar), dim=-1)


def _quaternion_conjugate(value: Tensor) -> Tensor:
    return torch.cat((-value[..., :3], value[..., 3:]), dim=-1)


def _normalized_quaternion(value: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    _check_numeric((norm > 0).all(), "valid Anchor quaternions must be non-zero")
    return cast(Tensor, value / norm)


def _quaternion_matrix(value: Tensor) -> Tensor:
    """Convert a unit xyzw quaternion without selecting a quaternion sign."""

    x, y, z, w = value.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*value.shape[:-1], 3, 3)


def evaluate_object_pose(
    predicted_delta: Tensor,
    *,
    current_position_m: Tensor,
    current_quaternion_xyzw: Tensor,
    future_position_m: Tensor,
    future_quaternion_xyzw: Tensor,
    valid: Tensor,
) -> _LossResult:
    """Squared translation distance and squared SO(3) angle, in metres and radians."""
    _require(
        valid.dtype == torch.bool and valid.ndim == 3 and all(axis > 0 for axis in valid.shape),
        "object validity must be bool [B,S,H]",
    )
    batch, predictions, horizon = valid.shape
    values = {
        "prediction": (predicted_delta, (batch, predictions, horizon, 6)),
        "current position": (current_position_m, (batch, predictions, 3)),
        "current quaternion": (current_quaternion_xyzw, (batch, predictions, 4)),
        "future position": (future_position_m, (batch, predictions, horizon, 3)),
        "future quaternion": (future_quaternion_xyzw, (batch, predictions, horizon, 4)),
    }
    for name, (value, shape) in values.items():
        _require(
            value.dtype == torch.float32
            and tuple(value.shape) == shape
            and value.device == valid.device,
            f"object {name} differs from axes/dtype/device",
        )
        _check_numeric(torch.isfinite(value).all(), f"object {name} must be finite")
    for name, value in (("position", future_position_m), ("quaternion", future_quaternion_xyzw)):
        _check_numeric(
            ~((value != 0) & ~valid[..., None]).any(),
            f"invalid future object {name} must be zero",
        )
    current_valid = valid.any(dim=-1)
    for name, quaternion, mask in (
        ("current", current_quaternion_xyzw, current_valid),
        ("future", future_quaternion_xyzw, valid),
    ):
        norms = torch.linalg.vector_norm(quaternion, dim=-1)
        _check_numeric(
            (torch.isclose(norms, torch.ones_like(norms), atol=1.0e-5, rtol=0) | ~mask).all(),
            f"valid {name} object quaternions must be unit norm",
        )
    identity_current = torch.zeros_like(current_quaternion_xyzw)
    identity_current[..., 3] = 1
    current = torch.where(current_valid[..., None], current_quaternion_xyzw, identity_current)
    identity_future = torch.zeros_like(future_quaternion_xyzw)
    identity_future[..., 3] = 1
    future = torch.where(valid[..., None], future_quaternion_xyzw, identity_future)
    target_rotation = _quaternion_matrix(
        _normalized_quaternion(
            _quaternion_product(future, _quaternion_conjugate(current[:, :, None]))
        )
    )
    predicted = predicted_delta[..., 3:].masked_fill(~valid[..., None], 0)
    predicted_rotation = _quaternion_matrix(
        _normalized_quaternion(_rotation_code_quaternion(predicted, "rotvec"))
    )
    # Keep the physical rotation error and its gradient in FP32 under AMP.
    with torch.autocast(device_type=predicted_delta.device.type, enabled=False):
        error = predicted_rotation @ target_rotation.transpose(-1, -2)
    skew = torch.stack(
        (
            error[..., 2, 1] - error[..., 1, 2],
            error[..., 0, 2] - error[..., 2, 0],
            error[..., 1, 0] - error[..., 0, 1],
        ),
        dim=-1,
    )
    # atan2 avoids acos' singular derivative at identity and its loss of
    # precision near pi. The squared angle has a finite zero gradient at zero;
    # at exactly pi the geodesic itself has the unavoidable SO(3) cut locus.
    sine = torch.linalg.vector_norm(skew, dim=-1) * 0.5
    cosine = (error.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) * 0.5
    angle = torch.atan2(sine, cosine)
    translation_error = (
        predicted_delta[..., :3] - (future_position_m - current_position_m[:, :, None])
    ).masked_fill(~valid[..., None], 0)
    translation_sum = translation_error.square().sum(dim=-1).sum()
    rotation_sum = angle.square().masked_fill(~valid, 0).sum()
    count = valid.sum(dtype=torch.int64)
    denominator = count.clamp_min(1)
    return _LossResult(translation_sum / denominator + 0.1 * (rotation_sum / denominator), count)
