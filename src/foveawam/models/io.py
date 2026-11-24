"""Tensor inputs, physical action limits and predictions for FoveaWAM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from foveawam.models.common.numeric_checks import (
    _check_numeric,
    _check_numeric_many,
    _collecting_training_checks,
)

_FUTURE_HORIZON = 8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _shape(value: Tensor) -> tuple[int, ...]:
    return tuple(int(axis) for axis in value.shape)


def _rotation_code_quaternion(value: Tensor, codec: str) -> Tensor:
    """Convert a rotation vector to an xyzw quaternion."""
    _require(codec == "rotvec", "object rotations use radians in rotation-vector form")
    angle = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = torch.where(
        angle > 1e-8, torch.sin(angle / 2) / angle.clamp_min(1e-8), 0.5 - angle.square() / 48
    )
    return torch.cat((value * scale, torch.cos(angle / 2)), dim=-1)


def _unit_quaternion_rotation_code(q: Tensor, codec: str) -> Tensor:
    """Convert a unit xyzw quaternion to a rotation vector. The caller checks the norm."""
    _require(codec == "rotvec", "object rotations use radians in rotation-vector form")
    xyz, scalar = q[..., :3], q[..., 3]
    first = torch.where(
        xyz[..., 0] != 0, xyz[..., 0], torch.where(xyz[..., 1] != 0, xyz[..., 1], xyz[..., 2])
    )
    flip = (scalar < 0) | ((scalar == 0) & (first < 0))
    q = torch.where(flip[..., None], -q, q)
    length = torch.linalg.vector_norm(q[..., :3], dim=-1)
    angle = 2 * torch.atan2(length, q[..., 3].clamp_min(0))
    scale = torch.where(length > 1e-8, angle / length.clamp_min(1e-8), torch.full_like(length, 2))
    return q[..., :3] * scale[..., None]


def _zero_predicate(value: Tensor, invalid: Tensor) -> Tensor:
    expanded = invalid
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    return ~((value != 0) & expanded).any()


def _predicate_results(*checks: Tensor) -> Tensor:
    values = torch.stack(checks)
    return torch.cat((values.all().reshape(1), values))


def _require_predicate_results(
    results: Tensor,
    messages: tuple[str, ...],
    numeric_checks: list[tuple[Tensor, str]] | None = None,
) -> None:
    _require(
        results.ndim == 1 and len(results) == len(messages) + 1,
        "internal tensor validation result differs",
    )
    if numeric_checks is not None:
        numeric_checks.extend(zip(results[1:].unbind(), messages, strict=True))
        return
    _check_numeric_many(tuple(zip(results[1:].unbind(), messages, strict=True)))


def _tensor(
    name: str,
    value: Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    finite: bool = True,
    numeric_checks: list[tuple[Tensor, str]] | None = None,
) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}, got {value.dtype}")
    if _shape(value) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {_shape(value)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if finite and (value.is_floating_point() or value.is_complex()):
        check = torch.isfinite(value).all()
        if numeric_checks is None:
            if _collecting_training_checks():
                _check_numeric(check, f"{name} must be finite")
            elif not bool(check.item()):
                raise ValueError(f"{name} must be finite")
        else:
            numeric_checks.append((check, f"{name} must be finite"))


def _zero_where(
    name: str,
    value: Tensor,
    invalid: Tensor,
) -> None:
    _check_numeric(
        _zero_predicate(value, invalid),
        f"invalid {name} must be zero",
    )


def _observation_history_predicates(
    frame_mask: Tensor,
    prediction_mask: Tensor,
    rgb: Tensor,
    timestamps_s: Tensor,
    eef_position_robot_base_m: Tensor,
    eef_quaternion_robot_base_xyzw: Tensor,
    gripper_open_fraction: Tensor,
) -> Tensor:
    invalid_frames = ~frame_mask
    history_length = frame_mask.shape[1] - prediction_mask.shape[1] + 1
    histories = frame_mask.unfold(1, history_length, 1)
    holes = (histories[..., :-1] & ~histories[..., 1:]).any(dim=-1)
    # Keep the previous *real* frame even for a directly validated mask with
    # holes outside enabled predictions. No dynamic nonzero/gather allocation.
    positions = torch.arange(frame_mask.shape[1], device=frame_mask.device).expand_as(frame_mask)
    last_real = torch.where(frame_mask, positions, -1).cummax(dim=1).values
    previous = torch.cat(
        (last_real.new_full((frame_mask.shape[0], 1), -1), last_real[:, :-1]), dim=1
    )
    paired = frame_mask & (previous >= 0)
    indices = previous.clamp_min(0)
    previous_time = timestamps_s.gather(1, indices)
    previous_quaternion = eef_quaternion_robot_base_xyzw.gather(
        1, indices[..., None].expand(-1, -1, 4)
    )
    norms = torch.linalg.vector_norm(eef_quaternion_robot_base_xyzw, dim=-1)
    real_checks = torch.stack(
        (
            ((timestamps_s > previous_time) | ~paired).all(dim=1),
            (torch.isclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0) | invalid_frames).all(
                dim=1
            ),
            (
                ~((eef_quaternion_robot_base_xyzw * previous_quaternion).sum(dim=-1) < 0) | ~paired
            ).all(dim=1),
        ),
        dim=1,
    )
    return _predicate_results(
        (histories[..., -1] | ~prediction_mask).all(),
        ~(holes & prediction_mask).any(),
        _zero_predicate(rgb, invalid_frames),
        _zero_predicate(timestamps_s, invalid_frames),
        _zero_predicate(eef_position_robot_base_m, invalid_frames),
        _zero_predicate(eef_quaternion_robot_base_xyzw, invalid_frames),
        _zero_predicate(gripper_open_fraction, invalid_frames),
        *real_checks.flatten().unbind(),
    )


def _camera_intrinsics_predicate(intrinsics: Tensor, invalid_cameras: Tensor) -> Tensor:
    return (
        ((intrinsics[..., 0, 0] > 0) | invalid_cameras).all()
        & ((intrinsics[..., 1, 1] > 0) | invalid_cameras).all()
        & (
            (
                (intrinsics[..., 2, :] - torch.tensor((0.0, 0.0, 1.0), device=intrinsics.device))
                .abs()
                .le(1e-6)
                .all(dim=-1)
                | invalid_cameras
            ).all()
        )
    )


def _camera_transform_predicate(transform: Tensor, invalid_cameras: Tensor) -> Tensor:
    rotation = transform[..., :3, :3].to(torch.float64)
    return (
        (
            torch.isclose(
                transform[..., 3, :],
                torch.tensor((0.0, 0.0, 0.0, 1.0), device=transform.device).expand_as(
                    transform[..., 3, :]
                ),
                atol=1e-5,
                rtol=0,
            ).all(dim=-1)
            | invalid_cameras
        ).all()
        & (
            torch.isclose(
                rotation.transpose(-1, -2) @ rotation,
                torch.eye(3, dtype=torch.float64, device=rotation.device).expand_as(rotation),
                atol=1e-4,
                rtol=0,
            ).all(dim=(-1, -2))
            | invalid_cameras
        ).all()
        & (
            torch.isclose(
                torch.linalg.det(rotation),
                torch.ones(rotation.shape[:-2], dtype=torch.float64, device=rotation.device),
                atol=1e-4,
                rtol=0,
            )
            | invalid_cameras
        ).all()
    )


def _camera_geometry_predicates(
    frame_mask: Tensor,
    intrinsics: Tensor,
    transform: Tensor,
) -> Tensor:
    invalid_frames = ~frame_mask
    invalid_cameras = invalid_frames.unsqueeze(-1)
    return _predicate_results(
        _zero_predicate(intrinsics, invalid_frames),
        _zero_predicate(transform, invalid_frames),
        _camera_intrinsics_predicate(intrinsics, invalid_cameras),
        _camera_transform_predicate(transform, invalid_cameras),
    )


def _depth_predicates(
    frame_mask: Tensor,
    depth_m: Tensor,
    depth_valid: Tensor,
) -> Tensor:
    invalid_frames = ~frame_mask
    return _predicate_results(
        (depth_valid == (depth_m > 0)).all(),
        _zero_predicate(depth_m, ~depth_valid),
        _zero_predicate(depth_m, invalid_frames),
        _zero_predicate(depth_valid, invalid_frames),
    )


def _anchor_padding_predicates(
    frame_mask: Tensor,
    anchor_position_m: Tensor,
    anchor_quaternion_xyzw: Tensor,
    anchor_valid: Tensor,
) -> Tensor:
    invalid_frames = ~frame_mask
    return _predicate_results(
        _zero_predicate(anchor_position_m, ~anchor_valid),
        _zero_predicate(anchor_quaternion_xyzw, ~anchor_valid),
        _zero_predicate(anchor_position_m, invalid_frames),
        _zero_predicate(anchor_quaternion_xyzw, invalid_frames),
        _zero_predicate(anchor_valid, invalid_frames),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionSpec:
    """End-effector translation and rotation increments followed by a discrete gripper."""

    mode: Literal["E7"] = "E7"
    representation: Literal["delta"] = "delta"
    gripper_mode: Literal["discrete"] = "discrete"
    motion_limits: tuple[float, ...]

    def __post_init__(self) -> None:
        _require(
            (self.mode, self.representation, self.gripper_mode) == ("E7", "delta", "discrete"),
            "FoveaWAM actions are E7 increments with a discrete gripper",
        )
        _require(len(self.motion_limits) == 6, "the E7 action needs six motion limits")
        for value in self.motion_limits:
            _require(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and value > 0,
                "motion_limits must contain finite positive numbers",
            )

    @property
    def dimension(self) -> int:
        return 7

    @property
    def component_names(self) -> tuple[str, ...]:
        return ("delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper")

    @property
    def units(self) -> tuple[str, ...]:
        return ("m", "m", "m", "rad", "rad", "rad", "unitless")

    @property
    def reference_frame(self) -> str:
        return "robot_base"

    @property
    def gripper_index(self) -> int:
        return 6

    @property
    def lower_bounds(self) -> tuple[float, ...]:
        return tuple(-float(value) for value in self.motion_limits) + (-1.0,)

    @property
    def upper_bounds(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.motion_limits) + (1.0,)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ObservationWindow:
    """A batch of aligned RGB-D and robot observations, including left padding."""

    view_names: tuple[str, ...]
    rgb: Tensor
    timestamps_s: Tensor
    frame_mask: Tensor
    prediction_mask: Tensor
    eef_position_robot_base_m: Tensor
    eef_quaternion_robot_base_xyzw: Tensor
    gripper_open_fraction: Tensor
    camera_intrinsics: Tensor
    T_robot_base_from_camera: Tensor  # noqa: N815
    depth_m: Tensor
    depth_valid: Tensor
    anchor_position_m: Tensor
    anchor_quaternion_xyzw: Tensor
    anchor_valid: Tensor


def _validate_observation(window: _ObservationWindow) -> tuple[int, int]:
    """Validate frame alignment, episode padding and the physical calibration in one window."""
    _require(window.view_names == ("agentview",), "FoveaWAM expects one agentview RGB-D camera")
    rgb, frame_mask, prediction_mask = window.rgb, window.frame_mask, window.prediction_mask
    _require(rgb.ndim == 6, "rgb must have axes [B,F,1,3,H,W]")
    batch, frames, views, channels, height, width = rgb.shape
    _require(
        (views, channels) == (1, 3) and min(batch, height, width) > 0, "invalid RGB image shape"
    )
    predictions = frames - 7
    _require(predictions > 0, "FoveaWAM needs eight frames per prediction")
    bf = (batch, frames)
    image = (*bf, 1, 1, height, width)
    fields = (
        ("rgb", rgb, torch.uint8, tuple(rgb.shape)),
        ("timestamps_s", window.timestamps_s, torch.float64, bf),
        ("frame_mask", frame_mask, torch.bool, bf),
        ("prediction_mask", prediction_mask, torch.bool, (batch, predictions)),
        ("eef_position_robot_base_m", window.eef_position_robot_base_m, torch.float32, (*bf, 3)),
        (
            "eef_quaternion_robot_base_xyzw",
            window.eef_quaternion_robot_base_xyzw,
            torch.float32,
            (*bf, 4),
        ),
        ("gripper_open_fraction", window.gripper_open_fraction, torch.float32, bf),
        ("camera_intrinsics", window.camera_intrinsics, torch.float32, (*bf, 1, 3, 3)),
        (
            "T_robot_base_from_camera",
            window.T_robot_base_from_camera,
            torch.float32,
            (*bf, 1, 4, 4),
        ),
        ("depth_m", window.depth_m, torch.float32, image),
        ("depth_valid", window.depth_valid, torch.bool, image),
        ("anchor_position_m", window.anchor_position_m, torch.float32, (*bf, 3)),
        ("anchor_quaternion_xyzw", window.anchor_quaternion_xyzw, torch.float32, (*bf, 4)),
        ("anchor_valid", window.anchor_valid, torch.bool, bf),
    )
    checks: list[tuple[Tensor, str]] = []
    for name, tensor, dtype, shape in fields:
        _tensor(name, tensor, dtype=dtype, shape=shape, numeric_checks=checks)
        _require(tensor.device == rgb.device, f"{name} and RGB must share a device")
    checks.extend(
        (
            (prediction_mask.any(dim=1).all(), "each sample needs a valid prediction"),
            (
                (frame_mask[:, 7:] == prediction_mask).all(),
                "prediction positions must match real frames",
            ),
            (
                ~(prediction_mask[:, 1:] & ~prediction_mask[:, :-1]).any(),
                "prediction positions must form a prefix",
            ),
        )
    )
    _require_predicate_results(
        _observation_history_predicates(
            frame_mask,
            prediction_mask,
            rgb,
            window.timestamps_s,
            window.eef_position_robot_base_m,
            window.eef_quaternion_robot_base_xyzw,
            window.gripper_open_fraction,
        ),
        (
            "a prediction needs its current frame",
            "invalid history padding",
            "padded RGB must be zero",
            "padded timestamps must be zero",
            "padded positions must be zero",
            "padded quaternions must be zero",
            "padded gripper values must be zero",
        )
        + (
            "timestamps must increase",
            "EEF quaternions must be normalized",
            "EEF quaternion signs must be continuous",
        )
        * batch,
        checks,
    )
    _require_predicate_results(
        _camera_geometry_predicates(
            frame_mask, window.camera_intrinsics, window.T_robot_base_from_camera
        ),
        (
            "padded intrinsics must be zero",
            "padded transforms must be zero",
            "invalid focal length or intrinsics",
            "camera-to-base transforms must be rigid",
        ),
        checks,
    )
    _require_predicate_results(
        _depth_predicates(frame_mask, window.depth_m, window.depth_valid),
        (
            "depth validity must match positive depth",
            "invalid depth must be zero",
            "padded depth must be zero",
            "padded depth must be invalid",
        ),
        checks,
    )
    _require_predicate_results(
        _anchor_padding_predicates(
            frame_mask, window.anchor_position_m, window.anchor_quaternion_xyzw, window.anchor_valid
        ),
        (
            "invalid object positions must be zero",
            "invalid object rotations must be zero",
            "padded object positions must be zero",
            "padded object rotations must be zero",
            "padded object poses must be invalid",
        ),
        checks,
    )
    norms = torch.linalg.vector_norm(window.anchor_quaternion_xyzw, dim=-1)
    checks.append(
        (
            (
                torch.isclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0)
                | ~window.anchor_valid
            ).all(),
            "object quaternions must be normalized",
        )
    )
    _check_numeric_many(checks)
    return batch, predictions


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelInput(_ObservationWindow):
    """The observation window and a noisy action chunk at flow time tau."""

    action_state_normalized: Tensor
    action_tau: Tensor

    def __post_init__(self) -> None:
        batch, predictions = _validate_observation(self)
        _tensor(
            "action_state_normalized",
            self.action_state_normalized,
            dtype=torch.float32,
            shape=(batch, predictions, 8, 7),
        )
        _tensor("action_tau", self.action_tau, dtype=torch.float32, shape=(batch, predictions))
        _require(
            self.action_state_normalized.device == self.action_tau.device == self.rgb.device,
            "the noisy action, flow time and observations must share a device",
        )
        _zero_where("action state", self.action_state_normalized, ~self.prediction_mask)
        _zero_where("action time", self.action_tau, ~self.prediction_mask)
        _check_numeric(
            ((self.action_tau >= 0) & (self.action_tau <= 1)).all(), "action_tau must lie in [0,1]"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelOutput:
    """Action velocity and the visual or geometric predictions used for supervision."""

    action_velocity_normalized: Tensor
    future_object_pose_delta: Tensor | None = None
    future_predictions: dict[str, Tensor] | None = None
    current_object_region_rgb: Tensor | None = None

    def __post_init__(self) -> None:
        action = self.action_velocity_normalized
        _require(
            action.ndim == 4 and action.shape[-2:] == (8, 7), "actions must have shape [B,S,8,7]"
        )
        prefix = tuple(action.shape[:2])
        fields = {"action_velocity_normalized": action}
        if self.future_object_pose_delta is not None:
            _require(
                self.future_object_pose_delta.shape == (*prefix, 8, 6),
                "object motion must have shape [B,S,8,6]",
            )
            fields["object pose"] = self.future_object_pose_delta
        if self.current_object_region_rgb is not None:
            _require(
                self.current_object_region_rgb.shape == (*prefix, 3, 32, 32),
                "object crop must have shape [B,S,3,32,32]",
            )
            fields["object crop"] = self.current_object_region_rgb
        if self.future_predictions is not None:
            shapes = {
                "environment": (196, 8, 4),
                "depth": (8, 1, 224, 224),
                "dino": (8, 196, 1024),
                "dynamic_rgb": (8, 3, 224, 224),
            }
            for name, value in self.future_predictions.items():
                _require(
                    name in shapes and value.shape == (*prefix, *shapes[name]),
                    f"invalid future prediction {name!r}",
                )
            fields.update(self.future_predictions)
        for name, value in fields.items():
            _require(
                value.dtype == torch.float32 and tuple(value.shape[:2]) == prefix,
                f"{name} must use float32 and share the action batch and prediction axes",
            )
        _check_numeric_many(
            [
                (torch.isfinite(value).all(), f"{name} must be finite")
                for name, value in fields.items()
            ]
        )


__all__ = ["ActionSpec", "ModelInput", "ModelOutput"]
