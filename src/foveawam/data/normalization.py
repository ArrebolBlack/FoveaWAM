"""ActionSpec-bound physical Action statistics and reversible transforms."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from foveawam.models.common.numeric_checks import _check_numeric, _check_numeric_many
from foveawam.models.io import ActionSpec


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _shape(value: Tensor) -> tuple[int, ...]:
    return tuple(int(axis) for axis in value.shape)


def _validate_action_statistics(
    action_spec: ActionSpec,
    action_mean: Tensor,
    action_std: Tensor,
) -> None:
    dimension = action_spec.dimension
    for name, value in (("action_mean", action_mean), ("action_std", action_std)):
        _require(value.dtype == torch.float32, f"{name} must use torch.float32")
        _require(_shape(value) == (dimension,), f"{name} must have shape [{dimension}]")
        _require(value.is_contiguous(), f"{name} must be contiguous")
        _check_numeric(torch.isfinite(value).all(), f"{name} must be finite")
    _require(
        action_mean.device == action_std.device,
        "Action normalization tensors must share a device",
    )
    _check_numeric((action_std > 0).all(), "action_std must be strictly positive")


def _normalize_action(
    action_spec: ActionSpec,
    physical: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
) -> Tensor:
    _validate_action_statistics(action_spec, action_mean, action_std)
    _require(
        physical.dtype == torch.float32
        and physical.is_contiguous()
        and physical.ndim >= 1
        and int(physical.shape[-1]) == action_spec.dimension,
        "physical Action tensor differs from normalization contract",
    )
    _check_numeric(torch.isfinite(physical).all(), "physical Action must be finite")
    mean = action_mean.to(device=physical.device)
    std = action_std.to(device=physical.device)
    return ((physical - mean) / std).contiguous()


def _denormalize_action(
    action_spec: ActionSpec,
    normalized: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
) -> Tensor:
    _validate_action_statistics(action_spec, action_mean, action_std)
    _require(
        normalized.dtype == torch.float32
        and normalized.is_contiguous()
        and normalized.ndim >= 1
        and int(normalized.shape[-1]) == action_spec.dimension,
        "normalized Action tensor differs from normalization contract",
    )
    _check_numeric(torch.isfinite(normalized).all(), "normalized Action must be finite")
    mean = action_mean.to(device=normalized.device)
    std = action_std.to(device=normalized.device)
    return (normalized * std + mean).contiguous()


def _validate_action_labels(action_spec: ActionSpec, value: Tensor, valid: Tensor) -> None:
    """Validate issued labels independently of controller execution limits."""

    _require(value.dtype == torch.float32, "physical Action must use torch.float32")
    _require(valid.dtype == torch.bool, "Action validity must use torch.bool")
    _require(value.is_contiguous() and valid.is_contiguous(), "Action tensors must be contiguous")
    _require(value.ndim >= 2, "physical Action needs at least one record axis")
    _require(_shape(value)[:-1] == _shape(valid), "Action values and validity axes must agree")
    _require(
        int(value.shape[-1]) == action_spec.dimension,
        "physical Action dimension differs from ActionSpec",
    )
    _require(value.device == valid.device, "Action values and validity must share a device")
    checks = [
        (torch.isfinite(value).all(), "physical Action must be finite"),
        (
            ~((value != 0) & ~valid[..., None]).any(),
            "invalid physical Action values must be zero",
        ),
    ]
    gripper = value[..., action_spec.gripper_index]
    allowed = (gripper == -1) | (gripper == 0) | (gripper == 1)
    checks.append(((allowed | ~valid).all(), "valid gripper commands must use {-1,0,1}"))
    _check_numeric_many(checks)


def validate_physical_action(action_spec: ActionSpec, value: Tensor, valid: Tensor) -> None:
    """Check recorded delta commands against the six controller limits."""
    _validate_action_labels(action_spec, value, valid)
    limits = value.new_tensor(action_spec.motion_limits)
    _check_numeric((value[..., :6].abs() <= limits).all(), "delta action exceeds its motion limits")


@dataclass(slots=True, kw_only=True)
class ActionNormalization:
    """Checkpoint-ready statistics bound to one complete ActionSpec."""

    action_spec: ActionSpec
    action_mean: Tensor
    action_std: Tensor
    action_sample_count: int

    def __post_init__(self) -> None:
        _validate_action_statistics(self.action_spec, self.action_mean, self.action_std)
        _require(
            not isinstance(self.action_sample_count, bool)
            and isinstance(self.action_sample_count, int)
            and self.action_sample_count > 0,
            "action_sample_count must be a positive integer",
        )

    def normalize(self, action_spec: ActionSpec, physical: Tensor) -> Tensor:
        _require(action_spec == self.action_spec, "normalization ActionSpec mismatch")
        return _normalize_action(action_spec, physical, self.action_mean, self.action_std)

    def denormalize(self, action_spec: ActionSpec, normalized: Tensor) -> Tensor:
        _require(action_spec == self.action_spec, "normalization ActionSpec mismatch")
        return _denormalize_action(
            action_spec,
            normalized,
            self.action_mean,
            self.action_std,
        )


def _fit_action_normalization_stream(
    action_spec: ActionSpec,
    records: Iterable[tuple[Tensor, Tensor]],
) -> ActionNormalization:
    """Merge FP64 population moments over valid recorded actions."""
    count = 0
    mean: Tensor | None = None
    sum_squared_deviation: Tensor | None = None
    for action_physical, action_valid in records:
        _validate_action_labels(action_spec, action_physical, action_valid)
        selected = action_physical[action_valid]
        if not len(selected):
            continue
        selected_float64 = selected.to(dtype=torch.float64)
        batch_count = int(selected_float64.shape[0])
        batch_mean = selected_float64.mean(dim=0)
        centered = selected_float64 - batch_mean
        batch_squared_deviation = centered.square().sum(dim=0)
        if mean is None:
            mean = batch_mean
            sum_squared_deviation = batch_squared_deviation
            count = batch_count
            continue
        assert sum_squared_deviation is not None
        _require(
            batch_mean.device == mean.device,
            "Action statistics records must share a device",
        )
        total = count + batch_count
        delta = batch_mean - mean
        sum_squared_deviation = (
            sum_squared_deviation
            + batch_squared_deviation
            + delta.square() * (count * batch_count / total)
        )
        mean = mean + delta * (batch_count / total)
        count = total

    _require(
        mean is not None and sum_squared_deviation is not None,
        "cannot fit Action normalization without valid records",
    )
    assert mean is not None and sum_squared_deviation is not None
    scale = (sum_squared_deviation / count).sqrt().clamp_min(1e-6)
    return ActionNormalization(
        action_spec=action_spec,
        action_mean=mean.to(torch.float32).contiguous(),
        action_std=scale.to(torch.float32).contiguous(),
        action_sample_count=count,
    )
