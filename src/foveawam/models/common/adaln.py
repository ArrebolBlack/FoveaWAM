"""Flow per-token and Action scalar adaptive layer normalization."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from foveawam.models.common.numeric_checks import _check_numeric, _check_numeric_many


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_tokens(tokens: Tensor, prediction_mask: Tensor, hidden_size: int) -> tuple[int, int]:
    _require(tokens.ndim == 4, "AdaLN tokens must have axes [B,S,N,D]")
    batch, predictions, _, hidden = (int(axis) for axis in tokens.shape)
    _require(hidden == hidden_size, "AdaLN token width differs from the configured width")
    _require(tokens.is_floating_point(), "AdaLN tokens must be floating point")
    _check_numeric(
        torch.isfinite(tokens).all(),
        "AdaLN tokens must be finite",
        _compiled_async=not torch.are_deterministic_algorithms_enabled(),
    )
    _require(
        prediction_mask.dtype == torch.bool
        and tuple(prediction_mask.shape) == (batch, predictions)
        and prediction_mask.device == tokens.device,
        "AdaLN prediction mask differs from token B/S axes or device",
    )
    return batch, predictions


def _modulate(
    norm: nn.LayerNorm, modulation: nn.Module, tokens: Tensor, condition: Tensor
) -> Tensor:
    scale, shift = cast(Tensor, modulation(condition)).chunk(2, dim=-1)
    # The projection follows autocast; normalization and the affine preserve
    # small scale updates that would disappear in BF16's `1 + scale`.
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        return cast(Tensor, norm(tokens.float())) * (1 + scale.float()) + shift.float()


class _ActionScalarAdaLN(nn.Module):
    """Apply one Action-tau modulation per prediction to all K tokens."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        _require(hidden_size > 0, "Action AdaLN width must be positive")
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(1, 2 * hidden_size))

    def forward(self, tokens: Tensor, tau: Tensor, prediction_mask: Tensor) -> Tensor:
        batch, predictions = _validate_tokens(tokens, prediction_mask, self.hidden_size)
        _require(
            tau.dtype == torch.float32
            and tuple(tau.shape) == (batch, predictions)
            and tau.device == tokens.device,
            "Action AdaLN tau differs from token B/S axes, dtype, or device",
        )
        checks = torch.stack(
            (
                torch.isfinite(tau).all(),
                (~prediction_mask | ((tau >= 0) & (tau <= 1))).all(),
                (prediction_mask | (tau == 0)).all(),
            )
        )
        messages = (
            "Action AdaLN tau must be finite",
            "Action AdaLN tau must lie in [0,1]",
            "padding Action AdaLN tau must be zero",
        )
        _check_numeric_many(
            tuple(zip(checks.unbind(), messages, strict=True)),
            _compiled_async=not torch.are_deterministic_algorithms_enabled(),
        )
        return self._forward_from_validated_inputs(tokens, tau, prediction_mask)

    def _forward_from_validated_inputs(
        self, tokens: Tensor, tau: Tensor, prediction_mask: Tensor
    ) -> Tensor:
        """Internal layer math after the trunk has checked changing FM inputs."""
        batch, predictions = tokens.shape[:2]
        condition = tau[..., None, None]
        result = _modulate(self.norm, self.modulation, tokens, condition)
        invalid = ~prediction_mask.reshape(batch, predictions, 1, 1)
        return result.masked_fill(invalid, 0).contiguous()
