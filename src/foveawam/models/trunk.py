"""Encode the observed world once and reuse its keys at every action noise level."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from foveawam.models.common.embeddings import _TimeEmbedding
from foveawam.models.common.numeric_checks import _check_numeric, _check_numeric_many
from foveawam.models.conditioning_cache import (
    _conditioning_isfinite,
    _ConditioningCache,
    _FiniteCheck,
)
from foveawam.models.layer import _WorldActionLayer
from foveawam.models.temporal_memory import TemporalMemory, _TemporalWindow


class _WorldActionTrunk(nn.Module):
    def __init__(
        self,
        dimension: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        dropout_p: float,
        *,
        history_length: int,
        action_horizon: int,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.action_horizon = action_horizon
        self.dimension = dimension
        self.conditioning_meta_token_count = 16
        self._prepared_bf16_inference = False
        self.time_embedding = _TimeEmbedding(dimension)
        layer_seed = int(torch.randint(2**63 - 1, (), dtype=torch.int64, device="cpu").item())
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(layer_seed)
            self.layers = nn.ModuleList(
                _WorldActionLayer(dimension, heads, mlp_ratio, dropout_p) for _ in range(depth)
            )

    def prepare_bf16_inference(self) -> None:
        self._prepared_bf16_inference = True
        for layer in self.layers:
            cast(_WorldActionLayer, layer).attention._prepared_bf16_inference = True

    def _new_temporal_memory(self) -> TemporalMemory:
        first = cast(_WorldActionLayer, self.layers[0])
        return TemporalMemory(
            first._project_scene_kv,
            self.time_embedding,
            history_length=self.history_length,
            allow_bf16_autocast=(
                self._prepared_bf16_inference and not self.training and not torch.is_grad_enabled()
            ),
        )

    def _window_inputs(
        self, windows: tuple[_TemporalWindow, ...], meta: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, tuple[Tensor, Tensor]]:
        if not windows or meta.ndim != 4:
            raise ValueError("trunk requires windows and meta tokens [B,S,16,D]")
        batch, predictions = meta.shape[:2]
        if tuple(meta.shape) != (batch, len(windows), 16, self.dimension) or batch < 1:
            raise ValueError("meta tokens differ from B/S/16/D contract")
        if not meta.is_floating_point():
            raise ValueError("meta tokens must be finite floating point")
        _check_numeric(torch.isfinite(meta).all(), "meta tokens must be finite floating point")
        for window in windows:
            frames = window.frames
            if (
                tuple(frames.visual.shape) != (batch, self.history_length, 196, self.dimension)
                or frames.visual.device != meta.device
                or frames.visual.dtype != meta.dtype
                or tuple(frames.frame_mask.shape) != (batch, self.history_length)
                or frames.frame_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "temporal window must contain configured T frame tokens on the meta device/dtype"
                )
            if frames.anchor is None or frames.anchor_valid is None or window.anchor_kv is None:
                raise ValueError("each window needs the target motion history")
            # Historical visual slots still require finite values even when
            # current-only packing omits them from the resulting scene.
            _check_numeric(
                torch.stack(
                    [
                        torch.isfinite(value).all()
                        for value in (frames.visual, frames.anchor, frames.robot)
                    ]
                ).all(),
                "window scene tokens must be finite",
            )
            if frames.scene_position is None or frames.scene_valid is None:
                raise ValueError("scene patches require positions and their validity mask")
        return self._pack_window_inputs(windows, meta)

    def _pack_window_inputs(
        self, windows: tuple[_TemporalWindow, ...], meta: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, tuple[Tensor, Tensor]]:
        """Pack validated windows without reading tensors back to the host."""
        batch, predictions = meta.shape[:2]

        def pack(groups: list[Tensor]) -> Tensor:
            # Past frames contribute target and robot tokens. The current frame also supplies 196 patches.
            states = torch.cat(groups[1:], dim=2)
            return torch.cat((states[:, :-1].flatten(1, 2), groups[0][:, -1], states[:, -1]), dim=1)

        scenes, masks, prediction_masks, positions, geometries = [], [], [], [], []
        all_keys: list[Tensor] = []
        all_values: list[Tensor] = []
        for window in windows:
            frames = window.frames
            prediction = window.prediction_mask
            valid_frames = frames.frame_mask & prediction[:, None]
            tokens = [frames.visual]
            visual_mask = valid_frames[..., None].expand(batch, self.history_length, 196)
            token_masks = [visual_mask]
            assert window.visual_kv is not None
            kv_groups = [window.visual_kv]
            assert frames.anchor is not None and frames.anchor_valid is not None
            assert window.anchor_kv is not None
            tokens.append(frames.anchor)
            object_mask = valid_frames & frames.anchor_valid
            token_masks.append(object_mask[..., None])
            kv_groups.append(window.anchor_kv)
            tokens.append(frames.robot)
            token_masks.append(valid_frames[..., None])
            kv_groups.append(window.robot_kv)
            mask = pack(token_masks)
            scene = pack(tokens)
            scenes.append(scene.masked_fill(~mask[..., None], 0.0))
            masks.append(mask)
            prediction_masks.append(prediction)
            for component, destination in enumerate((all_keys, all_values)):
                # Bring projected groups to [B,T,N,H,d] for identical packing.
                flat = pack([pair[component].transpose(2, 3) for pair in kv_groups]).transpose(1, 2)
                destination.append(flat.masked_fill(~mask[:, None, :, None], 0.0))
            # Only image patches carry camera geometry for 3D RoPE.
            state_position = scene.new_zeros(batch, self.history_length, 1, 3)
            state_geometry = valid_frames.new_zeros(batch, self.history_length, 1)
            positions.append(pack([frames.scene_position, state_position, state_position]))
            geometries.append(pack([frames.scene_valid, state_geometry, state_geometry]) & mask)
        prediction_mask = torch.stack(prediction_masks, dim=1)
        scene_mask = torch.stack(masks, dim=1)
        meta_mask = prediction_mask[..., None].expand(batch, predictions, 16)
        scene_key = torch.stack(all_keys, dim=1).flatten(0, 1)
        scene_value = torch.stack(all_values, dim=1).flatten(0, 1)
        return (
            torch.stack(scenes, dim=1),
            scene_mask,
            meta_mask,
            prediction_mask,
            torch.stack(positions, dim=1),
            torch.stack(geometries, dim=1),
            (scene_key, scene_value),
        )

    def prepare_conditioning(
        self,
        windows: tuple[_TemporalWindow, ...],
        meta: Tensor,
        *,
        detach: bool = False,
    ) -> tuple[Tensor, Tensor, _ConditioningCache]:
        current, meta_hidden, cache = self._prepare_conditioning(
            windows, meta, detach=detach, action_only=False
        )
        assert current is not None and meta_hidden is not None
        return current, meta_hidden, cache

    def _prepare_action_conditioning(
        self,
        windows: tuple[_TemporalWindow, ...],
        meta: Tensor,
    ) -> _ConditioningCache:
        """Cache detached keys and values for the action stream; world readouts are skipped."""
        if self.training:
            raise ValueError("Action-only conditioning snapshots are inference-only")
        with torch.inference_mode(False), torch.no_grad():
            _, _, cache = self._prepare_conditioning(windows, meta, detach=True, action_only=True)
        return cache

    def _prepare_conditioning(
        self,
        windows: tuple[_TemporalWindow, ...],
        meta: Tensor,
        *,
        detach: bool,
        action_only: bool,
    ) -> tuple[Tensor | None, Tensor | None, _ConditioningCache]:
        if detach and self.training:
            raise ValueError("detached conditioning snapshots are inference-only")
        scene, scene_mask, meta_mask, prediction, position, geometry, scene_kv = (
            self._window_inputs(windows, meta)
        )
        empty = meta.new_empty(*meta.shape[:2], 0, self.dimension)
        empty_mask = prediction[..., None].expand(*prediction.shape, 0)
        groups = (scene, meta, empty)
        entries = []
        for index, layer in enumerate(self.layers):
            layer = cast(_WorldActionLayer, layer)
            if action_only and index == len(self.layers) - 1:
                # The action stream needs only this layer's keys and values. Its
                # attention and MLP outputs feed the world readouts alone.
                entries.append(
                    layer._project_conditioning_kv(
                        groups[0],
                        groups[1],
                        scene_mask,
                        meta_mask,
                        position,
                        geometry,
                        scene_kv if index == 0 else None,
                    )
                )
                break
            groups, kv = layer(
                groups,
                (scene_mask, meta_mask, empty_mask),
                prediction_mask=prediction,
                action_tau=None,
                positions_m=(position, None, None),
                geometry_masks=(geometry, None, None),
                scene_kv=scene_kv if index == 0 else None,
            )
            entries.append(kv)
        cache = _ConditioningCache(self, tuple(entries), scene_mask, meta_mask, prediction)
        if action_only:
            return None, None, cache.detached()
        current = groups[0][:, :, -198:].contiguous()
        if detach:
            with torch.inference_mode(False), torch.no_grad():
                return current.detach().clone(), groups[1].detach().clone(), cache.detached()
        return current, groups[1], cache

    def _validate_conditioning_cache(
        self,
        cache: _ConditioningCache,
        *,
        finite_check: _FiniteCheck = _conditioning_isfinite,
        allow_bf16_autocast: bool = False,
    ) -> None:
        first = cast(_WorldActionLayer, self.layers[0]).attention
        cache.validate(
            self,
            depth=len(self.layers),
            heads=first.heads,
            head_dimension=first.head_dimension,
            expected_meta_token_count=self.conditioning_meta_token_count,
            finite_check=finite_check,
            allow_bf16_autocast=allow_bf16_autocast,
        )

    def action_hidden(
        self,
        cache: _ConditioningCache,
        action: Tensor,
        tau: Tensor,
        *,
        repeats: int = 1,
        validate: bool = True,
    ) -> Tensor:
        prediction = cache.prediction_mask
        scene_mask, meta_mask = cache.scene_mask, cache.meta_mask
        if repeats > 1:
            prediction = prediction.repeat(repeats, 1)
            scene_mask = scene_mask.repeat(repeats, 1, 1)
            meta_mask = meta_mask.repeat(repeats, 1, 1)
        if validate:
            if action.shape != (*prediction.shape, self.action_horizon, self.dimension):
                raise ValueError("action tokens must have shape [B,S,H,D]")
            if (
                tau.shape != prediction.shape
                or tau.dtype != torch.float32
                or tau.device != action.device
            ):
                raise ValueError("action time must be float32 [B,S] on the token device")
            _check_numeric_many(
                (
                    (torch.isfinite(action).all(), "action tokens must be finite"),
                    (
                        torch.isfinite(tau).all()
                        & (~prediction | ((tau >= 0) & (tau <= 1))).all()
                        & (prediction | (tau == 0)).all(),
                        "invalid action noise time",
                    ),
                )
            )
        empty = action.new_empty(*prediction.shape, 0, self.dimension)
        empty_mask = prediction[..., None].expand(*prediction.shape, 0)
        action_mask = prediction[..., None].expand(*prediction.shape, self.action_horizon)
        groups = (empty, empty, action)
        for layer, kv in zip(self.layers, cache.layers, strict=True):
            groups, _ = layer(
                groups,
                (empty_mask, empty_mask, action_mask),
                prediction_mask=prediction,
                action_tau=tau,
                positions_m=(None, None, None),
                geometry_masks=(None, None, None),
                conditioning_kv=kv,
                conditioning_masks=(scene_mask, meta_mask),
                _reuse_action_input_validation=True,
                _action_noise_repeats=repeats,
            )
        return groups[2]
