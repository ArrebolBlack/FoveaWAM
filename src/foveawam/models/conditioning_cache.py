"""One observation's per-layer attention-input scene/meta projections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import torch
from torch import Tensor

from foveawam.models.common.numeric_checks import _check_numeric

_BASE_META_TOKENS = 16
_Validation = TypeVar("_Validation", bound=Callable[..., None])
_eager_validation = cast(Callable[[_Validation], _Validation], torch.compiler.disable)
_FiniteCheck = Callable[[tuple[Tensor, ...]], Tensor]


def _conditioning_isfinite(values: tuple[Tensor, ...]) -> Tensor:
    """A pure predicate that inference may fuse without capturing cache storage."""
    return torch.stack([torch.isfinite(value).all() for value in values]).all()


@dataclass(frozen=True, slots=True)
class _ConditioningCache:
    """Per-layer scene and summary K/V, indexed by observation and prediction."""

    owner: object
    layers: tuple[tuple[Tensor, Tensor, Tensor, Tensor], ...]
    scene_mask: Tensor
    meta_mask: Tensor
    prediction_mask: Tensor

    def detached(self) -> _ConditioningCache:
        with torch.inference_mode(False), torch.no_grad():
            return _ConditioningCache(
                self.owner,
                tuple(
                    (
                        scene_key.detach().clone(),
                        scene_value.detach().clone(),
                        meta_key.detach().clone(),
                        meta_value.detach().clone(),
                    )
                    for scene_key, scene_value, meta_key, meta_value in self.layers
                ),
                self.scene_mask.detach().clone(),
                self.meta_mask.detach().clone(),
                self.prediction_mask.detach().clone(),
            )

    @_eager_validation
    def validate(
        self,
        owner: object,
        *,
        depth: int,
        heads: int,
        head_dimension: int,
        expected_meta_token_count: int = _BASE_META_TOKENS,
        finite_check: _FiniteCheck = _conditioning_isfinite,
        allow_bf16_autocast: bool = False,
    ) -> None:
        # K/V stores flattened B*S axes. Keep host-side validation outside capture
        # so a changed B cannot leave an unbound symbol in a reduction-only graph.
        # This boundary neither detaches K/V nor excludes the attention computation.
        if self.owner is not owner or len(self.layers) != depth:
            raise ValueError("conditioning cache belongs to a different trunk or depth")
        if self.prediction_mask.ndim != 2 or self.prediction_mask.dtype != torch.bool:
            raise ValueError("conditioning prediction mask must have axes [B,S] and bool dtype")
        batch, predictions = self.prediction_mask.shape
        if batch < 1 or predictions < 1:
            raise ValueError("conditioning B/S axes must be nonempty")
        for mask in (self.scene_mask, self.meta_mask):
            if (
                mask.ndim != 3
                or mask.shape[:2] != (batch, predictions)
                or mask.dtype != torch.bool
                or mask.device != self.prediction_mask.device
            ):
                raise ValueError("conditioning masks must have axes [B,S,L] on one device")
        if expected_meta_token_count != _BASE_META_TOKENS:
            raise ValueError("FoveaWAM uses 16 summary tokens")
        if self.meta_mask.shape[-1] != expected_meta_token_count:
            raise ValueError(
                f"conditioning requires exactly {expected_meta_token_count} meta-condition tokens"
            )
        dtype = self.layers[0][0].dtype
        values: list[Tensor] = []
        for layer in self.layers:
            for value, mask in zip(
                layer, (self.scene_mask,) * 2 + (self.meta_mask,) * 2, strict=True
            ):
                if (
                    tuple(value.shape)
                    != (batch * predictions, heads, mask.shape[-1], head_dimension)
                    or value.device != self.prediction_mask.device
                    or not value.is_floating_point()
                    or (
                        value.dtype not in (torch.float32, torch.bfloat16)
                        if allow_bf16_autocast
                        else value.dtype != dtype
                    )
                ):
                    raise ValueError("conditioning K/V shape, dtype, device, or finiteness differs")
                values.append(value)
        _check_numeric(
            finite_check(tuple(values)),
            "conditioning K/V shape, dtype, device, or finiteness differs",
        )
