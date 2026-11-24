"""Time-stamped observation windows and their first-layer attention keys."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from foveawam.models.common.numeric_checks import _check_numeric
from foveawam.models.io import _predicate_results, _require_predicate_results

_PATCHES = 196
_ProjectKV = Callable[[Tensor, Tensor, Tensor | None, Tensor | None], tuple[Tensor, Tensor]]
_EmbedTime = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class _Frames:
    visual: Tensor
    robot: Tensor
    frame_mask: Tensor
    timestamps_s: Tensor
    anchor: Tensor | None
    anchor_valid: Tensor | None
    scene_position: Tensor
    scene_valid: Tensor

    def slice(self, start: int, stop: int) -> _Frames:
        return _Frames(
            self.visual[:, start:stop],
            self.robot[:, start:stop],
            self.frame_mask[:, start:stop],
            self.timestamps_s[:, start:stop],
            None,
            None,
            self.scene_position[:, start:stop],
            self.scene_valid[:, start:stop],
        )


@dataclass(frozen=True)
class _TemporalWindow:
    frames: _Frames
    visual_kv: tuple[Tensor, Tensor]
    robot_kv: tuple[Tensor, Tensor]
    anchor_kv: tuple[Tensor, Tensor] | None

    @property
    def prediction_mask(self) -> Tensor:
        """A window can contain real history while its current query is padding."""
        return self.frames.frame_mask[:, -1]


class TemporalMemory:
    """Project the visual and robot observations in each eight-frame window."""

    def __init__(
        self,
        project_kv: _ProjectKV,
        embed_time: _EmbedTime,
        *,
        history_length: int,
        allow_bf16_autocast: bool = False,
    ) -> None:
        self.history_length = history_length
        self._allow_bf16_autocast = allow_bf16_autocast
        self._project_kv = project_kv
        self._embed_time = embed_time

    def _add_relative_time(self, frames: _Frames) -> _Frames:
        current_valid = frames.frame_mask[:, -1:]
        valid = frames.frame_mask & current_valid
        # Keep both operands float64 through the subtraction. In particular, do
        # not lose 20 Hz differences when a source clock has a large origin.
        relative_time_s = (frames.timestamps_s - frames.timestamps_s[:, -1:]).masked_fill(
            ~valid, 0.0
        )
        relative_time_s = relative_time_s.to(dtype=torch.float32)
        time = self._embed_time(relative_time_s)
        width = frames.visual.shape[-1]
        if (
            tuple(time.shape) != (*frames.frame_mask.shape, width)
            or time.dtype != torch.float32
            or time.device != frames.visual.device
        ):
            raise ValueError("time embedding must return finite float32 [B,F,D]")
        _check_numeric(
            torch.isfinite(time).all(), "time embedding must return finite float32 [B,F,D]"
        )
        time = time.masked_fill(~valid[..., None], 0.0)

        def add(tokens: Tensor, token_valid: Tensor) -> Tensor:
            return (
                (tokens + time[:, :, None])
                .masked_fill(~token_valid[:, :, None, None], 0.0)
                .contiguous()
            )

        return replace(frames, visual=add(frames.visual, valid), robot=add(frames.robot, valid))

    def _project(self, frames: _Frames) -> _TemporalWindow:
        frames = self._add_relative_time(frames)

        def stream(
            tokens: Tensor,
            valid: Tensor,
            position: Tensor | None,
            geometry_valid: Tensor | None,
        ) -> tuple[Tensor, Tensor]:
            batch, length, count, width = tokens.shape
            token_mask = valid[:, :, None].expand(batch, length, count)
            key, value = self._project_kv(
                tokens.reshape(batch * length, count, width),
                token_mask.reshape(batch * length, count),
                None if position is None else position.reshape(batch * length, count, 3),
                None if geometry_valid is None else geometry_valid.reshape(batch * length, count),
            )
            # BF16 autocast can leave the geometry-rotated key in FP32 while
            # the value projection is BF16. Keep the cached K/V pair in the
            # lower compute dtype before the existing shape/device contract.
            if (
                self._allow_bf16_autocast
                and not torch.is_grad_enabled()
                and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16
                and key.dtype != value.dtype
            ):
                key = key.to(dtype=value.dtype)
            if (
                key.ndim != 4
                or key.shape[0] != batch * length
                or key.shape[1] < 1
                or key.shape[2] != count
                or key.shape[3] < 1
                or key.shape != value.shape
                or key.device != tokens.device
                or value.device != tokens.device
                or not key.is_floating_point()
                or value.dtype != key.dtype
            ):
                raise ValueError(
                    "project_kv must return matching floating K,V [B*F,heads,N,head_dim]"
                )
            key = key.reshape(batch, length, *key.shape[1:])
            value = value.reshape(batch, length, *value.shape[1:])
            mask = valid[:, :, None, None, None]
            return key.masked_fill(~mask, 0), value.masked_fill(~mask, 0)

        return _TemporalWindow(
            frames,
            stream(frames.visual, frames.frame_mask, frames.scene_position, frames.scene_valid),
            stream(frames.robot, frames.frame_mask, None, None),
            None,
        )

    def forward_chunk(
        self,
        visual: Tensor,
        robot: Tensor,
        frame_mask: Tensor,
        timestamps_s: Tensor,
        prediction_mask: Tensor,
        *,
        scene_position: Tensor,
        scene_valid: Tensor,
    ) -> tuple[_TemporalWindow, ...]:
        """Return S causal windows from F=T-1+S unique observation embeddings."""
        frames = _Frames(
            visual, robot, frame_mask, timestamps_s, None, None, scene_position, scene_valid
        )
        count = visual.shape[1] - self.history_length + 1
        if count < 1 or prediction_mask.shape != (visual.shape[0], count):
            raise ValueError("F must equal T-1+S with prediction_mask [B,S] and S >= 1")
        prediction_error = "prediction_mask must be a nonempty valid prefix matching current frames"
        if prediction_mask.dtype != torch.bool or prediction_mask.device != visual.device:
            raise ValueError(prediction_error)
        # A real episode is contiguous; only the initial history and the suffix
        # after the last prediction may be padding.
        history = frame_mask[:, : self.history_length]
        _require_predicate_results(
            _predicate_results(
                (prediction_mask == frame_mask[:, self.history_length - 1 :]).all()
                & prediction_mask[:, 0].all()
                & ~(prediction_mask[:, 1:] & ~prediction_mask[:, :-1]).any(),
                ~(history[:, :-1] & ~history[:, 1:]).any(),
            ),
            (prediction_error, "history padding must be an episode-start left prefix"),
        )
        return tuple(
            self._project(frames.slice(start, start + self.history_length))
            for start in range(count)
        )
