"""One RGB-D observation history and its next FoveaWAM prediction."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch
from torch import Tensor

from foveawam.data.normalization import _denormalize_action, validate_physical_action
from foveawam.inference.solver import _ActionSolver
from foveawam.models.io import ActionSpec, _ObservationWindow
from foveawam.models.model import _FoveaWAMModel, _WorldConditioning


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyObservation:
    """One 20 Hz simulation tick. All tensors are contiguous and reside on the CPU."""

    view_names: tuple[str, ...]
    rgb: Tensor
    timestamp_s: Tensor
    eef_position_robot_base_m: Tensor
    eef_quaternion_robot_base_xyzw: Tensor
    gripper_open_fraction: Tensor
    camera_intrinsics: Tensor
    T_robot_base_from_camera: Tensor
    depth_m: Tensor
    depth_valid: Tensor
    target_position_m: Tensor
    target_quaternion_xyzw: Tensor
    target_valid: Tensor

    def __post_init__(self) -> None:
        if self.view_names != ("agentview",):
            raise ValueError("the simulation policy requires one agentview camera")
        for name, (shape, dtype) in _OBSERVATION_FIELDS.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU tensor")
            if value.shape != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous {dtype} with shape {shape}")
            if value.requires_grad or (
                value.is_floating_point() and not torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be finite and detached")
        if not torch.equal(self.depth_valid, self.depth_m > 0) or (self.depth_m < 0).any():
            raise ValueError(
                "depth_valid must mark positive depths and invalid depths must be zero"
            )
        if abs(self.eef_quaternion_robot_base_xyzw.norm().item() - 1.0) > 1e-5:
            raise ValueError("end-effector quaternion must have unit norm")
        if self.target_valid:
            if abs(self.target_quaternion_xyzw.norm().item() - 1.0) > 1e-5:
                raise ValueError("target quaternion must have unit norm")
        elif self.target_position_m.any() or self.target_quaternion_xyzw.any():
            raise ValueError("an invalid target pose must be zero")


_OBSERVATION_FIELDS = {
    "rgb": ((1, 3, 224, 224), torch.uint8),
    "timestamp_s": ((), torch.float64),
    "eef_position_robot_base_m": ((3,), torch.float32),
    "eef_quaternion_robot_base_xyzw": ((4,), torch.float32),
    "gripper_open_fraction": ((), torch.float32),
    "camera_intrinsics": ((1, 3, 3), torch.float32),
    "T_robot_base_from_camera": ((1, 4, 4), torch.float32),
    "depth_m": ((1, 1, 224, 224), torch.float32),
    "depth_valid": ((1, 1, 224, 224), torch.bool),
    "target_position_m": ((3,), torch.float32),
    "target_quaternion_xyzw": ((4,), torch.float32),
    "target_valid": ((), torch.bool),
}

# Observation fields whose name differs inside the model, which calls the target its anchor.
_MODEL_FIELD_NAMES = {
    "timestamp_s": "timestamps_s",
    "target_position_m": "anchor_position_m",
    "target_quaternion_xyzw": "anchor_quaternion_xyzw",
    "target_valid": "anchor_valid",
}


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """Eight physical E7 commands, each a translation, rotation vector and gripper."""

    actions: Tensor


@dataclass(frozen=True, slots=True)
class MotionPrediction:
    """Physical actions, target motion and patchwise scene motion for one query."""

    actions_e7: Tensor
    target_pose_delta: Tensor
    scene_displacement_m: Tensor
    scene_visibility_probability: Tensor
    query_timestamp_s: Tensor


class FoveaWAMPolicy:
    """A stateful simulation policy. Feed every tick once and reset between episodes."""

    default_gripper_deadband = 0.1
    gripper_deadband_by_task = MappingProxyType({"ball_capture": 0.5})

    def __init__(
        self,
        model: _FoveaWAMModel,
        *,
        action_mean: Tensor,
        action_std: Tensor,
        gripper_deadband: float | None,
        displacement_scale_m: float,
        compile_model: bool,
    ) -> None:
        self._model = model
        self._device = next(model.parameters()).device
        self._mean = action_mean.to(self._device)
        self._std = action_std.to(self._device)
        self._displacement_scale_m = displacement_scale_m
        self._solver = _ActionSolver(model, compile_model=compile_model)
        self._history: deque[PolicyObservation] = deque(maxlen=model.history_length - 1)
        self._gripper_override: float | None = None
        self._gripper_deadband = self.default_gripper_deadband
        if gripper_deadband is not None:
            self.set_gripper_deadband(gripper_deadband)

    @property
    def action_spec(self) -> ActionSpec:
        return self._model.action_spec

    @property
    def history_length(self) -> int:
        return self._model.history_length

    def set_gripper_deadband(self, value: float) -> None:
        """Override the task decoder with a fixed neutral band."""
        if self._history:
            raise ValueError("reset the policy before changing its gripper deadband")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("the gripper deadband must be a finite number in [0, 1]")
        self._gripper_deadband = float(value)
        self._gripper_override = self._gripper_deadband

    def configure_task(self, task_id: str | None = None) -> None:
        """Select the task's gripper decoder before feeding the first observation."""
        if self._history:
            raise ValueError("reset the policy before selecting another task")
        if self._gripper_override is None:
            self._gripper_deadband = self.gripper_deadband_by_task.get(
                task_id or "", self.default_gripper_deadband
            )

    def reset(self, task_id: str | None = None) -> None:
        self._history.clear()
        self.configure_task(task_id)

    def _next(self, observation: PolicyObservation) -> PolicyObservation:
        if self._history:
            previous = self._history[-1]
            dt = (observation.timestamp_s - previous.timestamp_s).item()
            if abs(dt - 0.05) > 1e-6:
                raise ValueError("feed consecutive observations at 20 Hz")
            if (
                torch.dot(
                    previous.eef_quaternion_robot_base_xyzw,
                    observation.eef_quaternion_robot_base_xyzw,
                )
                < 0
            ):
                raise ValueError("end-effector quaternion signs must be continuous")
        # The environment can reuse its observation buffers after this call.
        return replace(
            observation,
            **{name: getattr(observation, name).clone() for name in _OBSERVATION_FIELDS},
        )

    def observe(self, observation: PolicyObservation) -> None:
        """Append a tick without drawing action noise or running the model."""
        self._history.append(self._next(observation))

    def _window(self, current: PolicyObservation) -> _ObservationWindow:
        observations = [*self._history, current]
        count = len(observations)
        padding = self.history_length - count
        window = {}
        for name in _OBSERVATION_FIELDS:
            values = [getattr(observation, name) for observation in observations]
            first = values[0]
            buffer = torch.empty(
                (self.history_length, *first.shape), dtype=first.dtype, pin_memory=True
            )
            buffer[:padding].zero_()
            torch.stack(values, out=buffer[padding:])
            window[_MODEL_FIELD_NAMES.get(name, name)] = buffer.unsqueeze(0).to(
                self._device, non_blocking=True
            )
        window["frame_mask"] = torch.tensor(
            [[False] * padding + [True] * count], device=self._device
        )
        window["prediction_mask"] = torch.ones((1, 1), dtype=torch.bool, device=self._device)
        return _ObservationWindow(view_names=current.view_names, **window)

    def _predict(
        self, observation: PolicyObservation, *, motion: bool
    ) -> tuple[ActionChunk, _WorldConditioning, PolicyObservation]:
        with torch.inference_mode(False), torch.no_grad():
            current = self._next(observation)
            window = self._window(current)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                frames = self._model._encode_online_frames(window)
                context = self._solver.condition(frames, window, predict_motion=motion)
                normalized = self._solver.sample(context)
            actions = _denormalize_action(self.action_spec, normalized, self._mean, self._std).cpu()
            if not torch.isfinite(actions).all():
                raise ValueError("denormalized E7 actions must be finite")
            limits = actions.new_tensor(self.action_spec.motion_limits)
            actions[:, :6].clamp_(min=-limits, max=limits)
            gripper = actions[:, -1]
            actions[:, -1] = torch.where(
                gripper.double().abs() > self._gripper_deadband, gripper.sign(), 0.0
            )
            validate_physical_action(
                self.action_spec, actions, torch.ones(len(actions), dtype=torch.bool)
            )
        return ActionChunk(actions), context, current

    def predict_action_chunk(self, observation: PolicyObservation) -> ActionChunk:
        """Append the query tick and return its eight-step action chunk."""
        chunk, _, current = self._predict(observation, motion=False)
        self._history.append(current)
        return chunk

    def predict_motion(self, observation: PolicyObservation) -> MotionPrediction:
        """Predict actions and world motion from the same eight-tick observation window."""
        chunk, context, current = self._predict(observation, motion=True)
        pose, predictions = context.future_object_pose_delta, context.future_predictions
        if pose is None or predictions is None:
            raise RuntimeError("the simulation model did not produce its world predictions")
        scene = predictions["environment"][0, 0].float()
        result = MotionPrediction(
            actions_e7=chunk.actions,
            target_pose_delta=pose[0, 0].float().cpu(),
            scene_displacement_m=(scene[..., :3] * self._displacement_scale_m).cpu(),
            scene_visibility_probability=scene[..., 3].sigmoid().cpu(),
            query_timestamp_s=current.timestamp_s.clone(),
        )
        self._history.append(current)
        return result
