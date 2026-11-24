"""FoveaWAM physical and normalized output heads."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from foveawam.models.common.numeric_checks import _check_numeric

_PATCH_COUNT = 196
_FUTURE_HORIZON = 8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hidden(
    name: str,
    value: Tensor,
    *,
    rank: int,
    suffix: tuple[int, ...],
    hidden_size: int,
) -> tuple[int, int]:
    _require(value.ndim == rank, f"{name} has the wrong rank")
    shape = tuple(int(axis) for axis in value.shape)
    _require(shape[2:-1] == suffix and shape[-1] == hidden_size, f"{name} has the wrong shape")
    _require(value.is_floating_point(), f"{name} must be floating point")
    _check_numeric(torch.isfinite(value).all(), f"{name} must be finite")
    return shape[0], shape[1]


def _mask(value: Tensor, prediction_mask: Tensor, *, batch: int, predictions: int) -> Tensor:
    _require(
        prediction_mask.dtype == torch.bool
        and tuple(prediction_mask.shape) == (batch, predictions)
        and prediction_mask.device == value.device,
        "head prediction mask differs from hidden B/S axes or device",
    )
    invalid = ~prediction_mask
    while invalid.ndim < value.ndim:
        invalid = invalid.unsqueeze(-1)
    return value.masked_fill(invalid, 0).to(torch.float32).contiguous()


class _ObjectPoseDeltaHead(nn.Module):
    """Predict the target pose trajectory from the final summary tokens of the world stream."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dimension = 6
        self.projection = nn.Linear(hidden_size, _FUTURE_HORIZON * self.dimension)
        self.reset_identity()

    @torch.no_grad()
    def reset_identity(self) -> None:
        self.projection.weight.zero_()
        bias = self.projection.bias.reshape(_FUTURE_HORIZON, self.dimension)
        bias.zero_()

    def forward(self, meta_hidden: Tensor, prediction_mask: Tensor) -> Tensor:
        batch, predictions = _hidden(
            "WM meta mean", meta_hidden, rank=3, suffix=(), hidden_size=self.hidden_size
        )
        delta = self.projection(meta_hidden).reshape(
            batch, predictions, _FUTURE_HORIZON, self.dimension
        )
        return _mask(delta, prediction_mask, batch=batch, predictions=predictions)


class _CurrentObjectRegionHead(nn.Module):
    """Reconstruct the current target region in RGB from the same summary tokens (training only)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 3 * 32 * 32),
            nn.Sigmoid(),
        )

    def forward(self, meta_hidden: Tensor, prediction_mask: Tensor) -> Tensor:
        batch, predictions = _hidden(
            "WM meta mean", meta_hidden, rank=3, suffix=(), hidden_size=self.hidden_size
        )
        rgb = self.projection(meta_hidden).reshape(batch, predictions, 3, 32, 32)
        return _mask(rgb, prediction_mask, batch=batch, predictions=predictions)


class _ActionVelocityHead(nn.Module):
    def __init__(self, hidden_size: int, action_dimension: int, action_horizon: int = 8) -> None:
        super().__init__()
        _require(hidden_size > 0, "Action velocity head width must be positive")
        _require(action_dimension == 7, "FoveaWAM actions have seven components")
        _require(
            type(action_horizon) is int and action_horizon == 8,
            "action horizon must be eight",
        )
        self.action_horizon = action_horizon
        self.hidden_size = hidden_size
        self.action_dimension = action_dimension
        self.projection = nn.Linear(hidden_size, action_dimension)

    def forward(self, action_hidden: Tensor, prediction_mask: Tensor) -> Tensor:
        _hidden(
            "Action hidden",
            action_hidden,
            rank=4,
            suffix=(self.action_horizon,),
            hidden_size=self.hidden_size,
        )
        return self._project(action_hidden, prediction_mask)

    def _project(self, action_hidden: Tensor, prediction_mask: Tensor) -> Tensor:
        """Project hidden states to the action velocity. The sampler checks finiteness."""
        with torch.autocast(device_type=action_hidden.device.type, enabled=False):
            velocity = cast(Tensor, self.projection(action_hidden.float()))
        return _mask(
            velocity,
            prediction_mask,
            batch=action_hidden.shape[0],
            predictions=action_hidden.shape[1],
        )


class _FutureReadouts(nn.Module):
    """Explicit predictions from the existing scene and meta endpoints.

    Depth and RGB heads predict a 16x16 tile per scene patch.
    The FoveaWAM action stream reads world hidden states directly. These
    readouts provide supervision during training and can be skipped at inference.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.projections = nn.ModuleDict()
        dimensions = {
            "environment": 8 * 4,
            "depth": 8 * 16 * 16,
            "dino": 8 * 1024,
            "dynamic_rgb": 8 * 3 * 16 * 16,
        }
        for name, dimension in dimensions.items():
            with torch.random.fork_rng(devices=[]):
                self.projections[name] = nn.Linear(hidden_size, dimension)

    def scene_displacements(self, scene: Tensor, prediction_mask: Tensor) -> Tensor:
        b, s = prediction_mask.shape
        output = self.projections["environment"](scene).reshape(b, s, 196, 8, 4)
        return _mask(output, prediction_mask, batch=b, predictions=s)

    def _pixels(self, scene: Tensor, projection: nn.Module, channels: int) -> Tensor:
        b, s = scene.shape[:2]
        # [patch_y, patch_x, future, channel, sub_y, sub_x] -> future image.
        projected: Tensor = projection(scene)
        return (
            projected.reshape(b, s, 14, 14, 8, channels, 16, 16)
            .permute(0, 1, 4, 5, 2, 6, 3, 7)
            .reshape(b, s, 8, channels, 224, 224)
        )

    def forward(
        self,
        scene: Tensor,
        prediction_mask: Tensor,
    ) -> dict[str, Tensor]:
        b, s = _hidden("future scene", scene, rank=4, suffix=(196,), hidden_size=self.hidden_size)
        result = {}
        for name, projection in self.projections.items():
            if name == "environment":
                result[name] = self.scene_displacements(scene, prediction_mask)
                continue
            elif name == "dino":
                value = projection(scene).reshape(b, s, 196, 8, 1024).transpose(2, 3)
            else:
                value = self._pixels(scene, projection, 3 if name == "dynamic_rgb" else 1)
            result[name] = _mask(value, prediction_mask, batch=b, predictions=s)
        return result
