"""FoveaWAM world conditioning, auxiliary predictions and action flow matching."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from foveawam.models.common.precision import CUDAGRAPH_OPTIONS, cast_bf16_linears
from foveawam.models.conditioning_cache import _ConditioningCache
from foveawam.models.frame_encoder import _FrameEncoder, _FrameEncoding
from foveawam.models.heads import (
    _ActionVelocityHead,
    _CurrentObjectRegionHead,
    _FutureReadouts,
    _ObjectPoseDeltaHead,
)
from foveawam.models.io import (
    ActionSpec,
    ModelInput,
    ModelOutput,
    _ObservationWindow,
    _validate_observation,
)
from foveawam.models.layer import _WorldActionLayer
from foveawam.models.object_tokens import _object_motion_descriptors, _ObjectMotionTokens
from foveawam.models.temporal_memory import _TemporalWindow
from foveawam.models.tokens import _ActionTokenProjection
from foveawam.models.trunk import _WorldActionTrunk
from foveawam.perception.frontend import _PerceptionFrontend
from foveawam.perception.vision import _build_deployment_vision_encoder, _build_vision_encoder


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _WorldConditioning:
    """World-stream keys and values for one observation window, with its world predictions."""

    owner: object
    cache: _ConditioningCache
    prediction_mask: Tensor
    future_object_pose_delta: Tensor | None = None
    future_predictions: dict[str, Tensor] | None = None
    current_object_region_rgb: Tensor | None = None


class _FoveaWAMModel(nn.Module):
    """Predict object and scene motion from observations and denoise the action chunk."""

    future_dino_teacher: nn.Module
    _requires_cached_dino: bool

    def __init__(
        self,
        *,
        action_spec: ActionSpec,
        perception_frontend: _PerceptionFrontend,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        mlp_ratio: float,
        history_length: int,
        action_horizon: int,
        object_input_mean: tuple[float, ...],
        object_input_std: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.view_names = ("agentview",)
        self.action_spec = action_spec
        self.history_length = history_length
        self.hidden_size = hidden_size
        self.perception_frontend = perception_frontend
        self.frame_encoder = _FrameEncoder(
            perception_frontend.vision_encoder.output_dimension, hidden_size
        )
        self.trunk = _WorldActionTrunk(
            hidden_size,
            num_layers,
            num_attention_heads,
            mlp_ratio,
            0.0,
            history_length=history_length,
            action_horizon=action_horizon,
        )
        self.object_token_encoder = _ObjectMotionTokens(
            hidden_size, object_input_mean, object_input_std
        )
        self.meta_tokens = nn.Parameter(torch.empty(16, hidden_size))
        nn.init.normal_(self.meta_tokens, mean=0.0, std=0.02)
        self.action_projection = _ActionTokenProjection(
            action_spec.dimension, hidden_size, action_horizon
        )
        self.action_head = _ActionVelocityHead(hidden_size, action_spec.dimension, action_horizon)
        self.object_pose_head = _ObjectPoseDeltaHead(hidden_size)
        with torch.random.fork_rng(devices=[]):
            self.future_readouts = _FutureReadouts(hidden_size)
        with torch.random.fork_rng(devices=[]):
            self.object_region_head = _CurrentObjectRegionHead(hidden_size)
        self._call_validate_observation = _validate_observation
        self._call_encode_frame_selection = self._encode_frame_selection

    def prepare_compiled_inference(self) -> None:
        # The loader sets the vision encoder's precision. The action head stays in FP32.
        vision = self.perception_frontend.vision_encoder
        cast_bf16_linears(self, exclude=(vision, self.action_head))
        self.trunk.prepare_bf16_inference()
        vision.compile(mode="reduce-overhead")
        self.frame_encoder.prepare_compiled_inference()
        self._call_encode_frame_selection = torch.compile(self._encode_frame_selection)
        self._call_validate_observation = torch.compile(
            _validate_observation, options=CUDAGRAPH_OPTIONS
        )

    def forward(
        self, inputs: ModelInput, *, _action_noise_batch: tuple[Tensor, Tensor] | None = None
    ) -> ModelOutput | tuple[ModelOutput, Tensor]:
        frames = self._encode_frame_selection(inputs)
        world = self._conditioning_from_frame_encoding(frames, inputs, detach=False)
        if _action_noise_batch is not None:
            return self._forward_action_noise_batch(world, *_action_noise_batch)
        tokens = self.action_projection(
            inputs.action_state_normalized, inputs.action_tau, world.prediction_mask
        )
        hidden = self.trunk.action_hidden(world.cache, tokens, inputs.action_tau)
        return self._output(world, self.action_head(hidden, world.prediction_mask))

    def _encode_frame_selection(
        self,
        window: _ObservationWindow,
        *,
        current_visual_only: bool = False,
    ) -> _FrameEncoding:
        perception = self.perception_frontend(
            rgb=window.rgb,
            frame_mask=window.frame_mask,
            camera_intrinsics=window.camera_intrinsics,
            T_robot_base_from_camera=window.T_robot_base_from_camera,
            depth_m=window.depth_m,
            depth_valid=window.depth_valid,
            current_visual_only=current_visual_only,
        )
        frames: _FrameEncoding = self.frame_encoder(
            perception=perception,
            frame_mask=window.frame_mask,
            eef_position_robot_base_m=window.eef_position_robot_base_m,
            eef_quaternion_robot_base_xyzw=window.eef_quaternion_robot_base_xyzw,
            gripper_open_fraction=window.gripper_open_fraction,
            anchor_position_m=window.anchor_position_m,
            anchor_quaternion_xyzw=window.anchor_quaternion_xyzw,
            anchor_valid=window.anchor_valid,
            current_visual_only=current_visual_only,
        )
        return frames

    def _windows_from_frame_encoding(
        self,
        frame: _FrameEncoding,
        observations: _ObservationWindow,
    ) -> tuple[_TemporalWindow, ...]:
        windows = self.trunk._new_temporal_memory().forward_chunk(
            frame.scene_tokens,
            frame.robot_token,
            observations.frame_mask,
            observations.timestamps_s,
            observations.prediction_mask,
            scene_position=frame.scene_position_robot_base_m,
            scene_valid=frame.scene_position_valid,
        )
        updated = []
        first = cast(_WorldActionLayer, self.trunk.layers[0])
        for index, window in enumerate(windows):
            stop = index + self.history_length
            valid = (
                observations.anchor_valid[:, index:stop]
                & observations.frame_mask[:, index:stop]
                & observations.prediction_mask[:, index : index + 1]
            )
            object_position = frame.object_position_m[:, index:stop]
            object_quaternion = frame.object_quaternion_xyzw[:, index:stop]
            values, feature_valid, token_valid, token_time = _object_motion_descriptors(
                object_position,
                object_quaternion,
                observations.timestamps_s[:, index:stop],
                valid,
            )
            tokens = (
                self.object_token_encoder(values, feature_valid)
                + self.trunk.time_embedding(token_time)
            ).masked_fill(~token_valid[..., None], 0)
            batch, length, width = tokens.shape
            key, value = first._project_scene_kv(
                tokens.reshape(batch * length, 1, width),
                token_valid.reshape(batch * length, 1),
                None,
                None,
            )
            updated.append(
                replace(
                    window,
                    frames=replace(
                        window.frames, anchor=tokens[:, :, None], anchor_valid=token_valid
                    ),
                    anchor_kv=(
                        key.reshape(batch, length, *key.shape[1:]),
                        value.reshape(batch, length, *value.shape[1:]),
                    ),
                )
            )
        return tuple(updated)

    def _conditioning_from_frame_encoding(
        self,
        frame: _FrameEncoding,
        observations: _ObservationWindow,
        *,
        detach: bool,
        action_only: bool = False,
    ) -> _WorldConditioning:
        if detach and self.training:
            raise ValueError("detached world conditioning requires evaluation mode")
        windows = self._windows_from_frame_encoding(frame, observations)
        prediction_mask = observations.prediction_mask
        batch, num_predictions = prediction_mask.shape
        meta = self.meta_tokens[None, None].expand(batch, num_predictions, -1, -1)
        meta = meta.masked_fill(~prediction_mask[:, :, None, None], 0.0).contiguous()
        pose = region = predictions = None
        if action_only:
            cache = self.trunk._prepare_action_conditioning(windows, meta)
        else:
            current, summary, cache = self.trunk.prepare_conditioning(windows, meta, detach=detach)
            pooled = summary.mean(dim=2)
            pose = self.object_pose_head(pooled, prediction_mask)
            if detach:
                predictions = {
                    "environment": self.future_readouts.scene_displacements(
                        current[:, :, :196], prediction_mask
                    )
                }
            else:
                region = self.object_region_head(pooled, prediction_mask)
                predictions = self.future_readouts(current[:, :, :196], prediction_mask)
        return _WorldConditioning(self, cache, prediction_mask, pose, predictions, region)

    def _encode_online_frames(self, window: _ObservationWindow) -> _FrameEncoding:
        if (
            self.training
            or window.frame_mask.shape[1] != self.history_length
            or window.prediction_mask.shape[1] != 1
        ):
            raise ValueError(
                "online encoding requires an evaluation model and a single observation window"
            )
        self._call_validate_observation(window)
        with torch.inference_mode(False), torch.no_grad():
            return self._call_encode_frame_selection(window, current_visual_only=True)

    def _prepare_conditioning_from_frames(
        self,
        frames: _FrameEncoding,
        window: _ObservationWindow,
    ) -> _WorldConditioning:
        return self._prepare_online_conditioning(frames, window, action_only=False)

    def _prepare_action_conditioning_from_frames(
        self,
        frames: _FrameEncoding,
        window: _ObservationWindow,
    ) -> _WorldConditioning:
        return self._prepare_online_conditioning(frames, window, action_only=True)

    def _prepare_online_conditioning(
        self,
        frames: _FrameEncoding,
        window: _ObservationWindow,
        *,
        action_only: bool,
    ) -> _WorldConditioning:
        _require(frames.owner is self.frame_encoder, "frame encoding belongs to another model")
        with torch.inference_mode(False), torch.no_grad():
            return self._conditioning_from_frame_encoding(
                frames,
                window,
                action_only=action_only,
                detach=True,
            )

    def _action_numerics(
        self, conditioning: _WorldConditioning, state: Tensor, tau: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the action field and a device-side finiteness check for one Euler step."""
        tokens = self.action_projection._project(state, conditioning.prediction_mask)
        hidden = self.trunk.action_hidden(conditioning.cache, tokens, tau, validate=False)
        velocity = self.action_head._project(hidden, conditioning.prediction_mask)
        checks = torch.stack(
            tuple(torch.isfinite(value).all() for value in (state, tokens, hidden, velocity))
        )
        return velocity, checks.all()

    @staticmethod
    def _output(world: _WorldConditioning, velocity: Tensor) -> ModelOutput:
        return ModelOutput(
            action_velocity_normalized=velocity,
            future_object_pose_delta=world.future_object_pose_delta,
            future_predictions=world.future_predictions,
            current_object_region_rgb=world.current_object_region_rgb,
        )

    def _forward_action_noise_batch(
        self,
        conditioning: _WorldConditioning,
        states: Tensor,
        taus: Tensor,
    ) -> tuple[ModelOutput, Tensor]:
        """Independent noise draws share conditioning; expand only allowed keys."""
        _require(conditioning.owner is self, "Action conditioning belongs to another model")
        _require(
            states.ndim == 5 and taus.shape == states.shape[:3],
            "Action noise batch requires [R,B,S,H,A] states and [R,B,S] tau",
        )
        repeats, batch, predictions = states.shape[:3]
        _require(
            repeats >= 1 and (batch, predictions) == conditioning.prediction_mask.shape,
            "Action noise axes differ from the conditioning",
        )
        self.trunk._validate_conditioning_cache(conditioning.cache)
        prediction = conditioning.prediction_mask.repeat(repeats, 1)
        tau = taus.flatten(0, 1)
        tokens = self.action_projection(states.flatten(0, 1), tau, prediction)
        hidden = self.trunk.action_hidden(conditioning.cache, tokens, tau, repeats=repeats)
        velocity = self.action_head(hidden, prediction).reshape(
            repeats, batch, predictions, *states.shape[3:]
        )
        return self._output(conditioning, velocity[0]), velocity


def _action_spec(model: DictConfig) -> ActionSpec:
    action = model.action_spec
    return ActionSpec(motion_limits=tuple(float(value) for value in action.motion_limits))


def _construct_model(model: DictConfig, frontend: _PerceptionFrontend) -> _FoveaWAMModel:
    return _FoveaWAMModel(
        action_spec=_action_spec(model),
        perception_frontend=frontend,
        hidden_size=int(model.hidden_size),
        num_layers=int(model.num_layers),
        num_attention_heads=int(model.num_attention_heads),
        mlp_ratio=float(model.mlp_ratio),
        object_input_mean=tuple(model.object_input_mean),
        object_input_std=tuple(model.object_input_std),
        history_length=int(model.history_length),
        action_horizon=8,
    )


def _state_mapping(value: object, owner: str) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or not all(isinstance(name, str) for name in value):
        raise TypeError(f"{owner} must be a string-key tensor mapping")
    result: dict[str, Tensor] = {}
    for raw_name, item in value.items():
        name = cast(str, raw_name)
        if not isinstance(item, Tensor):
            raise TypeError(f"{owner}.{name} must be a tensor")
        _require(item.device.type == "cpu", f"{owner}.{name} must be on CPU")
        _require(item.is_contiguous(), f"{owner}.{name} must be contiguous")
        if item.is_floating_point():
            _require(item.dtype == torch.float32, f"{owner}.{name} must use float32")
            _require(bool(torch.isfinite(item).all().item()), f"{owner}.{name} must be finite")
        result[name] = item
    return result


def _core_state(model: _FoveaWAMModel) -> dict[str, Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith(("perception_frontend.", "future_dino_teacher."))
    }


def _validate_state_matches(
    expected: Mapping[str, Tensor],
    stored: Mapping[str, Tensor],
    owner: str,
) -> None:
    _require(set(stored) == set(expected), f"{owner} state keys differ")
    for name, expected_value in expected.items():
        stored_value = stored[name]
        _require(stored_value.shape == expected_value.shape, f"{owner}.{name} shape differs")
        _require(stored_value.dtype == expected_value.dtype, f"{owner}.{name} dtype differs")


def _build_deployment_model(
    config: DictConfig,
    state_dicts: Mapping[str, Mapping[str, Tensor] | None],
) -> _FoveaWAMModel:
    from foveawam.config.validate import _validate_deployment_config

    _validate_deployment_config(config)
    if not isinstance(state_dicts, Mapping) or not all(
        isinstance(name, str) for name in state_dicts
    ):
        raise TypeError("deployment state_dicts must be a string-key mapping")
    _require(
        set(state_dicts) == {"model", "vision", "depth", "flow"},
        "deployment state_dict groups differ",
    )
    model_state = _state_mapping(state_dicts["model"], "state_dicts.model")
    vision_state = _state_mapping(state_dicts["vision"], "state_dicts.vision")
    _require(
        state_dicts["depth"] is None and state_dicts["flow"] is None,
        "OnTimeBench supplies depth directly",
    )
    vision = config.perception.vision
    encoder = _build_deployment_vision_encoder(
        identity=str(vision.identity),
        input_size_px=int(vision.input_size_px),
        trainable=bool(vision.trainable),
        state_dict=vision_state,
    )
    result = _construct_model(config.model, _PerceptionFrontend(encoder)).to(
        device="cpu", dtype=torch.float32
    )
    expected_core = _core_state(result)
    _validate_state_matches(expected_core, model_state, "state_dicts.model")
    full_state = result.state_dict()
    full_state.update(model_state)
    result.load_state_dict(full_state, strict=True)
    return result.eval()


def build_model(config: DictConfig) -> nn.Module:
    """Build the training network with frozen BF16 MAE features."""
    from foveawam.config.validate import validate_config
    from foveawam.perception.future_targets import build_future_dino_teacher
    from foveawam.perception.vision import _apply_frozen_mae_precision

    if not isinstance(config, DictConfig) or not OmegaConf.is_readonly(config):
        raise ValueError("model construction requires a resolved read-only configuration")
    validate_config(config)
    encoder = _build_vision_encoder(weights_path=str(config.perception.vision.weights_path))
    result = _construct_model(config.model, _PerceptionFrontend(encoder))
    result._requires_cached_dino = config.training.get("future_dino_cache_root") is not None
    if not result._requires_cached_dino:
        with torch.random.fork_rng(devices=[]):
            result.future_dino_teacher = build_future_dino_teacher(
                str(config.model.future_dino_teacher_weights_path)
            )
    result = result.to(device="cpu", dtype=torch.float32).train()
    _apply_frozen_mae_precision(result, "bf16")
    return result
