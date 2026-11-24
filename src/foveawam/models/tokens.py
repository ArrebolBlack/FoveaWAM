"""Embed the noisy end-effector trajectory and its horizon index."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from foveawam.models.common.embeddings import _IndexEmbedding
from foveawam.models.common.numeric_checks import _check_numeric_many

_ACTION_HORIZON = 8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _prediction_mask(value: Tensor, *, batch: int, predictions: int, device: torch.device) -> None:
    _require(
        value.dtype == torch.bool
        and tuple(value.shape) == (batch, predictions)
        and value.device == device,
        "prediction_mask differs from token B/S axes or device",
    )


class _ActionTokenProjection(nn.Module):
    """Project normalized Action state while preserving the K axis."""

    def __init__(self, action_dimension: int, hidden_size: int, action_horizon: int = 8) -> None:
        super().__init__()
        _require(action_dimension == 7, "FoveaWAM actions have seven components")
        _require(hidden_size > 0, "Action token width must be positive")
        _require(
            type(action_horizon) is int and action_horizon == 8,
            "action horizon must be eight",
        )
        self.action_horizon = action_horizon
        self.action_dimension = action_dimension
        self.hidden_size = hidden_size
        self.state = nn.Linear(action_dimension, hidden_size)
        self.horizon = _IndexEmbedding(_ACTION_HORIZON, hidden_size)

    def forward(
        self,
        state_normalized: Tensor,
        tau: Tensor,
        prediction_mask: Tensor,
    ) -> Tensor:
        _require(state_normalized.ndim == 4, "Action state must have axes [B,S,K,A]")
        batch, predictions, horizon, action = (int(axis) for axis in state_normalized.shape)
        _require(
            (horizon, action) == (self.action_horizon, self.action_dimension),
            "Action state differs from configured K or A",
        )
        _require(state_normalized.dtype == torch.float32, "normalized Action state must be float32")
        finite_state = torch.isfinite(state_normalized).all()
        try:
            _prediction_mask(
                prediction_mask,
                batch=batch,
                predictions=predictions,
                device=state_normalized.device,
            )
            _require(
                tau.dtype == torch.float32
                and tuple(tau.shape) == (batch, predictions)
                and tau.device == state_normalized.device,
                "Action tau differs from state B/S axes, dtype, device, or finiteness",
            )
        except ValueError:
            _require(bool(finite_state.item()), "normalized Action state must be finite")
            raise
        valid_state = prediction_mask[:, :, None, None]
        checks = torch.stack(
            (
                finite_state,
                torch.isfinite(tau).all(),
                (valid_state | (state_normalized == 0)).all(),
                (prediction_mask | (tau == 0)).all(),
                (~prediction_mask | ((tau >= 0) & (tau <= 1))).all(),
            )
        )
        messages = (
            "normalized Action state must be finite",
            "Action tau differs from state B/S axes, dtype, device, or finiteness",
            "padding Action state must be zero",
            "padding Action tau must be zero",
            "Action tau must lie in [0,1]",
        )
        _check_numeric_many(tuple(zip(checks.unbind(), messages, strict=True)))

        return self._project(state_normalized, prediction_mask)

    def _project(self, state_normalized: Tensor, prediction_mask: Tensor) -> Tensor:
        """Numerical projection; callers own state/mask validity and finiteness."""
        horizon_index = torch.arange(self.action_horizon, device=state_normalized.device)
        tokens = cast(Tensor, self.state(state_normalized)) + cast(
            Tensor, self.horizon.embedding(horizon_index)
        )
        return tokens.masked_fill(~prediction_mask[:, :, None, None], 0).contiguous()
