"""Transformer block for observed scene, summary and action tokens."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import torch
from torch import Tensor, nn

from foveawam.models.attention import _WorldActionAttention
from foveawam.models.common.adaln import _ActionScalarAdaLN

_Normalization = TypeVar("_Normalization", bound=Callable[..., Tensor])
_dispatch_norm = cast(
    Callable[..., Callable[[_Normalization], _Normalization]], torch.compiler.disable
)(recursive=False)


class _WorldActionLayer(nn.Module):
    """Scene and summary share an MLP. The action MLP is conditioned on noise time."""

    def __init__(self, dimension: int, heads: int, mlp_ratio: float, dropout_p: float) -> None:
        super().__init__()
        self.dimension = dimension
        self.attention = _WorldActionAttention(dimension, heads)
        self.wm_attention_norm = nn.LayerNorm(dimension)
        self.wm_ffn_norm = nn.LayerNorm(dimension)

        def ffn() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(dimension, max(dimension, round(dimension * mlp_ratio))),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.Linear(max(dimension, round(dimension * mlp_ratio)), dimension),
                nn.Dropout(dropout_p),
            )

        self.wm_ffn = ffn()
        self.action_attention_norm = _ActionScalarAdaLN(dimension)
        self.action_ffn_norm = _ActionScalarAdaLN(dimension)
        self.action_ffn = ffn()

    def _project_scene_kv(
        self,
        tokens: Tensor,
        key_mask: Tensor,
        position_m: Tensor | None,
        geometry_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """The first layer's token-local projection, safe for TemporalMemory reuse."""

        normalized = self._normalize_wm(
            tokens.masked_fill(~key_mask[..., None], 0.0), feed_forward=False
        )
        return self.attention.project_kv(normalized, key_mask, position_m, geometry_mask)

    def _normalize_wm(self, tokens: Tensor, *, feed_forward: bool) -> Tensor:
        module = self.wm_ffn_norm if feed_forward else self.wm_attention_norm
        return cast(Tensor, module(tokens))

    def _project_conditioning_kv(
        self,
        scene: Tensor,
        meta: Tensor,
        scene_mask: Tensor,
        meta_mask: Tensor,
        position_m: Tensor | None,
        geometry_mask: Tensor | None,
        scene_kv: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project the keys and values of the last world layer without running its readouts."""

        batch, predictions = scene.shape[:2]
        entries: list[Tensor] = []
        for index, (tokens, mask) in enumerate(((scene, scene_mask), (meta, meta_mask))):
            if index == 0 and scene_kv is not None:
                entries.extend(scene_kv)
                continue
            normalized = self._normalize_wm(
                tokens.masked_fill(~mask[..., None], 0.0), feed_forward=False
            ).reshape(batch * predictions, tokens.shape[2], self.dimension)
            flat_position = (
                position_m.reshape(batch * predictions, -1, 3)
                if index == 0 and position_m is not None
                else None
            )
            flat_geometry = (
                geometry_mask.reshape(batch * predictions, geometry_mask.shape[2])
                if index == 0 and geometry_mask is not None
                else None
            )
            entries.extend(
                self.attention.project_kv(
                    normalized,
                    mask.reshape(batch * predictions, mask.shape[2]),
                    flat_position,
                    flat_geometry,
                )
            )
        return entries[0], entries[1], entries[2], entries[3]

    def _norm(
        self,
        index: int,
        tokens: Tensor,
        prediction_mask: Tensor,
        action_tau: Tensor | None,
        *,
        feed_forward: bool,
        reuse_action_input_validation: bool = False,
    ) -> Tensor:
        if tokens.shape[2] == 0:
            return tokens
        if index < 2:
            return self._normalize_wm(tokens, feed_forward=feed_forward)
        action_norm = self.action_ffn_norm if feed_forward else self.action_attention_norm
        if action_norm is None or action_tau is None:
            raise ValueError("Action tokens require enabled Action modules and scalar tau")
        # K is a token axis, not the dynamic batch/prediction count. Specialize it
        # before capture so equal S/K values cannot become one symbolic extent.
        torch._dynamo.mark_static(tokens, 2)
        if reuse_action_input_validation:
            return action_norm._forward_from_validated_inputs(tokens, action_tau, prediction_mask)
        return cast(Tensor, action_norm(tokens, action_tau, prediction_mask))

    _dispatched_norm = _dispatch_norm(_norm)

    def forward(
        self,
        groups: tuple[Tensor, Tensor, Tensor],
        masks: tuple[Tensor, Tensor, Tensor],
        *,
        prediction_mask: Tensor,
        action_tau: Tensor | None,
        positions_m: tuple[Tensor | None, Tensor | None, Tensor | None],
        geometry_masks: tuple[Tensor | None, Tensor | None, Tensor | None],
        scene_kv: tuple[Tensor, Tensor] | None = None,
        conditioning_kv: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        conditioning_masks: tuple[Tensor, Tensor] | None = None,
        _reuse_action_input_validation: bool = False,
        _action_noise_repeats: int = 1,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        if conditioning_kv is not None:
            if scene_kv is not None or conditioning_masks is None:
                raise ValueError("conditioned layer requires only its scene/meta K/V and masks")
            if groups[0].shape[2] != 0 or groups[1].shape[2] != 0:
                raise ValueError("conditioned execution must not recompute WM queries")
        elif conditioning_masks is not None:
            raise ValueError("conditioning masks require conditioning K/V")
        batch, predictions = prediction_mask.shape
        normalize = (
            self._dispatched_norm
            if torch.are_deterministic_algorithms_enabled() and not _reuse_action_input_validation
            else self._norm
        )
        queries: list[Tensor] = []
        keys: list[Tensor] = []
        values: list[Tensor] = []
        clean_groups: list[Tensor] = []
        key_masks = list(masks)
        for index, (tokens, mask, position, geometry) in enumerate(
            zip(groups, masks, positions_m, geometry_masks, strict=True)
        ):
            clean = tokens.masked_fill(~mask[..., None], 0.0)
            clean_groups.append(clean)
            normalized = normalize(
                index,
                clean,
                prediction_mask,
                action_tau,
                feed_forward=False,
                reuse_action_input_validation=_reuse_action_input_validation,
            ).reshape(batch * predictions, tokens.shape[2], self.dimension)
            flat_mask = mask.reshape(batch * predictions, mask.shape[2])
            flat_position = (
                None if position is None else position.reshape(batch * predictions, -1, 3)
            )
            flat_geometry = (
                None
                if geometry is None
                else geometry.reshape(batch * predictions, geometry.shape[2])
            )
            if conditioning_kv is not None and index < 2:
                assert conditioning_masks is not None
                key, value = conditioning_kv[2 * index : 2 * index + 2]
                query = key.new_empty(
                    (batch * predictions, self.attention.heads, 0, self.attention.head_dimension)
                )
                key_masks[index] = conditioning_masks[index]
            elif tokens.shape[2] == 0:
                query = keys[0].new_empty(
                    (batch * predictions, self.attention.heads, 0, self.attention.head_dimension)
                )
                key, value = query, query
            elif index == 0 and scene_kv is not None:
                query = self.attention.project_query(
                    normalized, flat_mask, flat_position, flat_geometry
                )
                key, value = scene_kv
            else:
                query, key, value = self.attention.project(
                    normalized,
                    query_mask=flat_mask,
                    key_mask=flat_mask,
                    geometry_mask=flat_geometry,
                    position_m=flat_position,
                )
            queries.append(query)
            keys.append(key)
            values.append(value)
        updates = self.attention.attend(
            tuple(queries),
            tuple(keys),
            tuple(values),
            query_masks=tuple(mask.reshape(batch * predictions, mask.shape[2]) for mask in masks),
            key_masks=tuple(mask.reshape(batch * predictions, mask.shape[2]) for mask in key_masks),
            _skip_empty_queries=_reuse_action_input_validation,
            _prefix_noise_repeats=_action_noise_repeats,
        )
        results = []
        for index, (tokens, update, mask) in enumerate(
            zip(clean_groups, updates, masks, strict=True)
        ):
            residual = tokens.float() + update.reshape_as(tokens).float()
            if tokens.shape[2] != 0:
                normalized = normalize(
                    index,
                    residual,
                    prediction_mask,
                    action_tau,
                    feed_forward=True,
                    reuse_action_input_validation=_reuse_action_input_validation,
                )
                ffn = self.wm_ffn if index < 2 else self.action_ffn
                residual = residual + ffn(normalized).float()
            results.append(residual.masked_fill(~mask[..., None], 0.0).contiguous())
        return (results[0], results[1], results[2]), (keys[0], values[0], keys[1], values[1])
