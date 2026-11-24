"""Flow matching over normalized, eight-step end-effector commands."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from foveawam.data.normalization import ActionNormalization, _validate_action_labels
from foveawam.models.common.numeric_checks import _check_numeric
from foveawam.models.io import ActionSpec
from foveawam.objectives import _LossResult


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ActionFlowSample:
    state_normalized: Tensor
    tau: Tensor
    velocity_target_normalized: Tensor
    valid: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class _ActionFlowTarget:
    data: Tensor
    valid: Tensor
    missing: Tensor
    prediction_mask: Tensor


def _prepare_action_flow_target(
    *,
    action_spec: ActionSpec,
    normalization: ActionNormalization,
    action_physical: Tensor,
    action_valid: Tensor,
    prediction_mask: Tensor,
    future_record_valid: Tensor,
) -> _ActionFlowTarget:
    """Construct the normalized labels shared by independent noise draws."""
    _validate_action_labels(action_spec, action_physical, action_valid)
    _require(normalization.action_spec == action_spec, "Action normalization spec mismatch")
    _require(action_physical.ndim == 4, "Action target must have axes [B,S,K,A]")
    batch, predictions, _, dimension = action_physical.shape
    _require(dimension == action_spec.dimension, "Action target dimension differs from ActionSpec")
    _require(
        tuple(action_valid.shape) == tuple(action_physical.shape[:-1]),
        "Action validity axes differ from Action target",
    )
    _require(
        prediction_mask.dtype == torch.bool
        and tuple(prediction_mask.shape) == (batch, predictions)
        and prediction_mask.device == action_physical.device,
        "prediction_mask differs from Action target axes/device",
    )
    _check_numeric(
        ~(action_valid & ~prediction_mask[:, :, None]).any(),
        "Action validity cannot extend into prediction padding",
    )
    if future_record_valid.dtype != torch.bool or future_record_valid.shape != action_valid.shape:
        raise ValueError("future record mask must match action validity")
    joint_valid = action_valid & prediction_mask.unsqueeze(-1)
    missing = prediction_mask.unsqueeze(-1) & ~future_record_valid
    _check_numeric(~(joint_valid & missing).any(), "valid actions require real future records")
    data = normalization.normalize(action_spec, action_physical)
    return _ActionFlowTarget(
        data=data.masked_fill(~joint_valid.unsqueeze(-1), 0),
        valid=joint_valid,
        missing=missing,
        prediction_mask=prediction_mask,
    )


def _sample_action_flow_target(
    target: _ActionFlowTarget,
    *,
    generator: torch.Generator | None = None,
) -> _ActionFlowSample:
    data, joint_valid, missing = target.data, target.valid, target.missing
    prediction_mask = target.prediction_mask
    batch, predictions = prediction_mask.shape
    expanded_valid = joint_valid.unsqueeze(-1)
    source = torch.randn(
        data.shape,
        dtype=torch.float32,
        device=data.device,
        generator=generator,
    )
    source = source.masked_fill(~(joint_valid | missing).unsqueeze(-1), 0)
    tau = torch.rand(
        (batch, predictions),
        dtype=torch.float32,
        device=data.device,
        generator=generator,
    )
    tau = tau.masked_fill(~prediction_mask, 0).contiguous()
    expanded_tau = tau[:, :, None, None]
    state = ((1 - expanded_tau) * source + expanded_tau * data).masked_fill(~expanded_valid, 0)
    # Missing labels must not expose a zero suffix to the bidirectional Action tokens.
    state = torch.where(missing[..., None], source, state)
    velocity = (data - source).masked_fill(~expanded_valid, 0)
    return _ActionFlowSample(
        state_normalized=state.contiguous(),
        tau=tau,
        velocity_target_normalized=velocity.contiguous(),
        valid=joint_valid,
    )


def _action_flow_terms(
    predicted_velocity_normalized: Tensor, target_velocity_normalized: Tensor, valid: Tensor
) -> tuple[Tensor, Tensor]:
    if (
        predicted_velocity_normalized.shape != target_velocity_normalized.shape
        or predicted_velocity_normalized.shape[:-1] != valid.shape
    ):
        raise ValueError(
            "action velocities and mask must have matching batch, time and horizon axes"
        )
    _check_numeric(
        (
            torch.isfinite(predicted_velocity_normalized)
            & torch.isfinite(target_velocity_normalized)
        ).all(),
        "action velocities must be finite",
    )
    squared = (predicted_velocity_normalized - target_velocity_normalized).square()
    count = valid.sum(dtype=torch.int64)
    numerator = (squared * valid.unsqueeze(-1)).sum()
    return numerator / (count.clamp_min(1) * squared.shape[-1]), count


def _reduce_action_flow_terms(terms: list[tuple[Tensor, Tensor]]) -> _LossResult:
    """Average independent draws over the same valid command slots."""
    return _LossResult(torch.stack([mean for mean, _ in terms]).mean(), terms[0][1])
