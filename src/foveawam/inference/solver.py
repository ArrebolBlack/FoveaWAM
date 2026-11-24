"""Five-step flow-matching sampler for a single FoveaWAM action query."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from foveawam.models.common.precision import BF16_OPTIONS, CUDAGRAPH_OPTIONS
from foveawam.models.conditioning_cache import _conditioning_isfinite
from foveawam.models.frame_encoder import _FrameEncoding
from foveawam.models.io import _ObservationWindow
from foveawam.models.model import _FoveaWAMModel, _WorldConditioning


class _ActionSolver:
    def __init__(self, model: _FoveaWAMModel, *, compile_model: bool) -> None:
        self.model = model
        self.compiled = compile_model
        device = next(model.parameters()).device
        self._times = torch.tensor(
            [step / 5 for step in range(5)], dtype=torch.float32, device=device
        ).view(5, 1, 1)
        self._prepare_motion = model._prepare_conditioning_from_frames
        self._prepare_actions = model._prepare_action_conditioning_from_frames
        self._integrate = self._euler
        self._cache_finite = _conditioning_isfinite
        if compile_model:
            model.prepare_compiled_inference()
            self._prepare_motion = torch.compile(self._prepare_motion, options=BF16_OPTIONS)
            self._prepare_actions = torch.compile(self._prepare_actions, options=CUDAGRAPH_OPTIONS)
            self._integrate = torch.compile(
                self._euler,
                fullgraph=True,
                dynamic=False,
                options={**CUDAGRAPH_OPTIONS, "max_autotune": True},
            )
            self._cache_finite = torch.compile(
                _conditioning_isfinite,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

    def condition(
        self,
        frames: _FrameEncoding,
        window: _ObservationWindow,
        *,
        predict_motion: bool,
    ) -> _WorldConditioning:
        prepare = self._prepare_motion if predict_motion else self._prepare_actions
        context = prepare(frames, window)
        if self.compiled and not predict_motion:
            # The next captured graph may reuse the conditioning graph's output buffers.
            context = replace(
                context,
                cache=context.cache.detached(),
                prediction_mask=context.prediction_mask.clone(),
            )
        self.model.trunk._validate_conditioning_cache(
            context.cache,
            finite_check=self._cache_finite,
            allow_bf16_autocast=self.compiled,
        )
        return context

    def _euler(self, context: _WorldConditioning, source: Tensor) -> tuple[Tensor, Tensor]:
        state = source
        valid = torch.isfinite(source).all()
        for step in range(5):
            velocity, finite = self.model._action_numerics(
                context, state[None, None], self._times[step]
            )
            state = state + 0.2 * velocity[0, 0].contiguous()
            valid = valid & finite
        return state, valid & torch.isfinite(state).all()

    @torch.inference_mode()
    def sample(self, context: _WorldConditioning) -> Tensor:
        source = torch.randn((8, 7), dtype=torch.float32, device=self._times.device)
        state, valid = self._integrate(context, source)
        if not bool(valid.item()):
            raise ValueError("FoveaWAM produced a nonfinite action during Euler integration")
        return state.clone() if self.compiled else state
