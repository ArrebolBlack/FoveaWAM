"""Read public OnTimeBench RGB-D demonstrations and their supervision labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.nn import functional

from foveawam.data.normalization import validate_physical_action
from foveawam.models.io import ActionSpec

_HORIZON = 8
_SOURCE_LIMITS = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
_METADATA_KEYS = {
    "episode_id",
    "task_id",
    "episode_seed",
    "view_names",
    "control_frequency_hz",
    "action_spec",
    "geom_table",
    "event_names",
}
_TARGET_BODY = {
    "moving_receptacle": "goal_receptacle",
    "rotating_receptacle": "goal_receptacle",
    "wire_passage": "buzz_frame",
    "full_wire_passage": "buzz_frame",
}

_EpisodeInfoValue = str | int | tuple[str, ...]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    value = group[name]
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"expected HDF5 dataset: {name}")
    return value


def _group(group: h5py.Group, name: str) -> h5py.Group:
    value = group[name]
    if not isinstance(value, h5py.Group):
        raise ValueError(f"expected HDF5 group: {name}")
    return value


def _metadata(handle: h5py.File) -> tuple[dict[str, object], int]:
    if set(handle) != {"metadata", "policy", "action", "supervision"}:
        raise ValueError("Expected only the OnTimeBench training episode groups")
    if dict(handle.attrs) != {"format": "OnTimeBench", "schema_version": 1}:
        raise ValueError("Expected an OnTimeBench training episode with schema version 1")
    encoded = _dataset(handle, "metadata/episode_json").asstr()[()]
    if not isinstance(encoded, str):
        raise ValueError("OnTimeBench episode metadata must be a UTF-8 string")
    metadata = _mapping(json.loads(encoded), "episode metadata")
    if set(metadata) != _METADATA_KEYS:
        raise ValueError("Unexpected OnTimeBench episode metadata fields")
    for key in ("episode_id", "task_id"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"Expected a nonempty {key}")
    seed = metadata["episode_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Expected a nonnegative episode seed")
    if metadata["view_names"] != ["agentview"] or metadata["control_frequency_hz"] != 20:
        raise ValueError("Expected one agentview camera at 20 Hz")
    body = _TARGET_BODY.get(cast(str, metadata["task_id"]))
    if body is not None:
        table = metadata["geom_table"]
        if not isinstance(table, list):
            raise ValueError("Expected a geometry table for the moving target")
        target_ids, body_ids = [], []
        for index, raw in enumerate(table):
            row = _mapping(raw, "geometry row")
            roles = row.get("semantic_identity")
            if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
                raise ValueError("Expected explicit geometry roles")
            if "target_object" in roles:
                target_ids.append(index)
            if row.get("body_name") == body:
                body_ids.append(index)
        if not body_ids or body_ids != target_ids:
            raise ValueError(f"Target geometry must cover exactly the {body} body")
    expected_action = {
        "mode": "E7",
        "representation": "delta",
        "motion_limits": list(_SOURCE_LIMITS),
    }
    if metadata["action_spec"] != expected_action:
        raise ValueError("Expected the OnTimeBench E7 delta action specification")
    frames = len(_dataset(handle, "policy/tick"))
    if frames < 2 or len(_dataset(handle, "action/value")) != frames - 1:
        raise ValueError("Expected one more observation than actions")
    return metadata, frames


def _episode_identity(metadata: dict[str, object], frames: int) -> dict[str, _EpisodeInfoValue]:
    return {
        "episode_id": cast(str, metadata["episode_id"]),
        "task_id": cast(str, metadata["task_id"]),
        "episode_seed": cast(int, metadata["episode_seed"]),
        "frames": frames,
        "view_names": tuple(cast(list[str], metadata["view_names"])),
    }


def _validate_action_clock(handle: h5py.File, frames: int) -> None:
    timestamps = np.asarray(_dataset(handle, "policy/timestamp_s")[:], dtype=np.float64)
    ticks = np.asarray(_dataset(handle, "policy/tick")[:], dtype=np.int64)
    issued = np.asarray(_dataset(handle, "action/issued_tick")[:], dtype=np.int64)
    applied = np.asarray(_dataset(handle, "action/applied_tick")[:], dtype=np.int64)
    if (
        not np.isfinite(timestamps).all()
        or not np.all(np.diff(timestamps) > 0)
        or not np.allclose(timestamps - timestamps[0], np.arange(frames) / 20, atol=1e-8, rtol=0)
        or not np.array_equal(ticks, np.arange(frames))
        or not np.array_equal(issued, ticks[:-1])
        or not np.all((applied == -1) | ((applied >= 0) & (applied < frames)))
    ):
        raise ValueError("Invalid OnTimeBench native clock or issued/applied Action alignment")


def _index_action_handle(handle: h5py.File) -> dict[str, _EpisodeInfoValue]:
    metadata, frames = _metadata(handle)
    count = frames - 1
    layouts = {
        "value": ((count, 7), "float32"),
        "valid": ((count,), "bool"),
        "issued_tick": ((count,), "int64"),
        "applied_tick": ((count,), "int64"),
    }
    action = _group(handle, "action")
    for name, (shape, dtype) in layouts.items():
        value = _dataset(action, name)
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"Invalid OnTimeBench Action shape or dtype: action/{name}")
    _validate_action_clock(handle, frames)
    return _episode_identity(metadata, frames)


def _take[Scalar: np.generic](
    dataset: h5py.Dataset, indices: NDArray[np.int64], dtype: np.dtype[Scalar]
) -> NDArray[Scalar]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(dataset)):
        raise ValueError(
            "OnTimeBench indices must be real in-episode records; windowing owns padding"
        )
    unique, inverse = np.unique(indices, return_inverse=True)
    selected: NDArray[Scalar] = np.asarray(dataset[unique], dtype=dtype)
    return selected[inverse]


def _tensor(value: NDArray[np.generic], dtype: np.dtype[np.generic] | None = None) -> Tensor:
    array = np.ascontiguousarray(value, dtype=dtype)
    return torch.from_numpy(array)


def _quaternion_sequence(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    quaternion = np.asarray(Rotation.from_matrix(rotation).as_quat(), dtype=np.float64)
    if quaternion[0, 3] < 0:
        quaternion[0] *= -1
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index], quaternion[index - 1]) < 0:
            quaternion[index] *= -1
    return quaternion


def _rigid(value: NDArray[np.generic]) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    rotation = matrix[..., :3, :3]
    if (
        matrix.shape[-2:] != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[..., 3, :], (0, 0, 0, 1), atol=1e-8, rtol=0)
        or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid OnTimeBench source rigid transform")
    return matrix


def index_episode(path: str | Path) -> dict[str, _EpisodeInfoValue]:
    with h5py.File(str(path), "r") as handle:
        metadata, frames = _metadata(handle)
        count = frames - 1
        view_count = len(cast(list[str], metadata["view_names"]))
        geom_table = metadata["geom_table"]
        if not isinstance(geom_table, list):
            raise ValueError("OnTimeBench geom_table must be a JSON array")
        rows = [_mapping(row, "geom_table row") for row in geom_table]
        if [row.get("geom_id") for row in rows] != list(range(len(rows))):
            raise ValueError("OnTimeBench geom_table must cover each stored geometry exactly once")
        layouts: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {
            "policy": {
                "tick": ((frames,), "int64"),
                "timestamp_s": ((frames,), "float64"),
                "rgb": ((frames, view_count, 224, 224, 3), "uint8"),
                "depth_m": ((frames, view_count, 224, 224), "float32"),
                "camera_intrinsics": ((frames, view_count, 3, 3), "float64"),
                "world_from_camera_cv": ((frames, view_count, 4, 4), "float64"),
                "T_robot_base_from_world": ((4, 4), "float64"),
                "joint_position": ((frames, 7), "float32"),
                "joint_velocity": ((frames, 7), "float32"),
                "eef_position": ((frames, 3), "float32"),
                "eef_quaternion_xyzw": ((frames, 4), "float32"),
                "gripper_position": ((frames, 2), "float32"),
                "gripper_velocity": ((frames, 2), "float32"),
            },
            "action": {
                "value": ((count, 7), "float32"),
                "valid": ((count,), "bool"),
                "issued_tick": ((count,), "int64"),
                "applied_tick": ((count,), "int64"),
                "normalized_value": ((count, 7), "float64"),
            },
            "supervision": {
                "geom_id": ((frames, 1, 224, 224), "int32"),
                "world_from_geom": ((frames, len(rows), 4, 4), "float64"),
                "target_object_pose_world": ((frames, 7), "float64"),
                "flow_2d_px": ((count, 1, 224, 224, 2), "float32"),
                "flow_valid": ((count, 1, 224, 224), "bool"),
                "flow_future_visible": ((count, 1, 224, 224), "bool"),
                "reward": ((count,), "float32"),
                "done": ((count,), "bool"),
                "success": ((frames,), "bool"),
                "event_flags": ((frames, len(cast(list[object], metadata["event_names"]))), "bool"),
            },
        }
        for group_name, fields in layouts.items():
            group = _group(handle, group_name)
            if set(group) != set(fields):
                raise ValueError(f"Unexpected OnTimeBench fields in {group_name}")
            for name, (shape, dtype) in fields.items():
                dataset = cast(h5py.Dataset, group[name])
                if dataset.shape != shape or dataset.dtype != np.dtype(dtype):
                    raise ValueError(f"Invalid OnTimeBench shape or dtype: {group_name}/{name}")
        _validate_action_clock(handle, frames)
        return _episode_identity(metadata, frames)


def read_frames(
    path: str | Path, indices: NDArray[np.int64]
) -> dict[str, Tensor | tuple[str, ...] | str]:
    """Read measured RGB-D, robot state and the target object's causal pose history."""
    with h5py.File(str(path), "r") as handle:
        metadata, _ = _metadata(handle)
        return {
            "task_id": cast(str, metadata["task_id"]),
            "view_names": ("agentview",),
            **_frame_tensors(handle, indices),
        }


def _frame_tensors(handle: h5py.File, indices: NDArray[np.int64]) -> dict[str, Tensor]:
    gripper = _take(_dataset(handle, "policy/gripper_position"), indices, np.dtype(np.float32))
    fraction = (gripper[:, 0] - gripper[:, 1]) / np.float32(0.08)
    if not np.isfinite(fraction).all():
        raise ValueError("Nonfinite measured OnTimeBench gripper observation")
    rgb = _take(_dataset(handle, "policy/rgb"), indices, np.dtype(np.uint8))
    fields: dict[str, Tensor] = {
        "rgb": _tensor(np.asarray(rgb.transpose(0, 1, 4, 2, 3))),
        "timestamps_s": _tensor(
            _take(_dataset(handle, "policy/timestamp_s"), indices, np.dtype(np.float64))
        ),
        "eef_position_robot_base_m": _tensor(
            _take(_dataset(handle, "policy/eef_position"), indices, np.dtype(np.float32))
        ),
        "eef_quaternion_robot_base_xyzw": _tensor(
            _take(
                _dataset(handle, "policy/eef_quaternion_xyzw"),
                indices,
                np.dtype(np.float32),
            )
        ),
        "gripper_open_fraction": _tensor(fraction, np.dtype(np.float32)),
    }
    base = _rigid(np.asarray(_dataset(handle, "policy/T_robot_base_from_world")[:]))
    intrinsic = _take(_dataset(handle, "policy/camera_intrinsics"), indices, np.dtype(np.float64))
    if not np.isfinite(intrinsic).all() or np.any(intrinsic[..., (0, 1), (0, 1)] <= 0):
        raise ValueError("Invalid OnTimeBench camera intrinsics")
    camera = _rigid(
        _take(
            _dataset(handle, "policy/world_from_camera_cv"),
            indices,
            np.dtype(np.float64),
        )
    )
    fields["camera_intrinsics"] = _tensor(intrinsic, np.dtype(np.float32))
    fields["T_robot_base_from_camera"] = _tensor(_rigid(base @ camera), np.dtype(np.float32))
    metric = _take(_dataset(handle, "policy/depth_m"), indices, np.dtype(np.float32))[:, :, None]
    valid = np.isfinite(metric) & (metric > 0)
    fields["depth_m"] = _tensor(np.asarray(np.where(valid, metric, 0)), np.dtype(np.float32))
    fields["depth_valid"] = _tensor(np.asarray(valid))
    pose = _take(
        _dataset(handle, "supervision/target_object_pose_world"),
        indices,
        np.dtype(np.float64),
    )
    fields["anchor_position_m"] = _tensor(
        np.asarray(pose[:, :3] @ base[:3, :3].T + base[:3, 3]), np.dtype(np.float32)
    )
    # Quaternion sign continuity is local to this requested observation
    # sequence; neither older rows nor future targets may be read here.
    rotations = np.asarray(Rotation.from_quat(pose[:, 3:]).as_matrix(), dtype=np.float64)
    quaternion = _quaternion_sequence(np.asarray(base[:3, :3] @ rotations))
    fields["anchor_quaternion_xyzw"] = _tensor(quaternion, np.dtype(np.float32))
    fields["anchor_valid"] = _tensor(np.ones(len(indices), dtype=np.bool_))
    return fields


def _current_object_regions(
    handle: h5py.File, metadata: dict[str, object], prediction_indices: NDArray[np.int64]
) -> dict[str, Tensor]:
    """Crop and resize the visible target at each current frame."""
    table = metadata["geom_table"]
    if not isinstance(table, list) or not table:
        raise ValueError("object region requires a nonempty OnTimeBench geom_table")
    target_ids = []
    for identity, raw_row in enumerate(table):
        row = _mapping(raw_row, "geom_table row")
        if type(row.get("geom_id")) is not int or row["geom_id"] != identity:
            raise ValueError("object region geom_table must cover each geometry exactly once")
        roles = row.get("semantic_identity")
        if (
            not isinstance(roles, list)
            or not roles
            or not all(isinstance(role, str) and role for role in roles)
        ):
            raise ValueError("object region requires explicit geometry semantic_identity")
        if "target_object" in roles:
            target_ids.append(identity)
    if not target_ids:
        raise ValueError("object region requires target_object geometry semantics")
    rgb = _take(_dataset(handle, "policy/rgb"), prediction_indices, np.dtype(np.uint8))[:, 0]
    geom = _take(_dataset(handle, "supervision/geom_id"), prediction_indices, np.dtype(np.int32))[
        :, 0
    ]
    if np.any(geom < -1) or np.any(geom >= len(table)):
        raise ValueError("object region raster contains an unknown geometry identity")
    region_rgb = torch.zeros((len(rgb), 3, 32, 32), dtype=torch.float32)
    region_valid = torch.zeros((len(rgb), 1, 32, 32), dtype=torch.bool)
    for index, (frame, identities) in enumerate(zip(rgb, geom, strict=True)):
        visible = np.isin(identities, target_ids)
        y, x = np.nonzero(visible)
        if len(y) == 0:
            # Fully occluded/out-of-view targets do not change sample exposure.
            continue
        top, bottom = int(y.min()), int(y.max()) + 1
        left, right = int(x.min()), int(x.max()) + 1
        crop = _tensor(frame[top:bottom, left:right].transpose(2, 0, 1)).to(torch.float32)
        mask = _tensor(visible[top:bottom, left:right]).to(torch.float32)
        resized = functional.interpolate(
            crop[None] / 255.0, size=(32, 32), mode="bilinear", align_corners=False
        )[0]
        valid = functional.interpolate(mask[None, None], size=(32, 32), mode="nearest")[0].bool()
        region_rgb[index] = resized.masked_fill(~valid, 0)
        region_valid[index] = valid
    return {
        "current_object_region_rgb": region_rgb,
        "current_object_region_valid": region_valid,
    }


def _validate_action_spec(action_spec: ActionSpec) -> None:
    if (
        action_spec.mode,
        action_spec.representation,
        action_spec.gripper_mode,
        action_spec.motion_limits,
    ) != (
        "E7",
        "delta",
        "discrete",
        _SOURCE_LIMITS,
    ):
        raise ValueError("Expected the OnTimeBench E7 delta action specification")


def _read_actions(
    handle: h5py.File, indices: NDArray[np.int64], action_spec: ActionSpec
) -> tuple[Tensor, Tensor]:
    values = _tensor(_take(_dataset(handle, "action/value"), indices, np.dtype(np.float32)))
    valid = _tensor(_take(_dataset(handle, "action/valid"), indices, np.dtype(np.bool_)))
    validate_physical_action(action_spec, values, valid)
    return values, valid


def _read_action_episode(
    path: Path, action_spec: ActionSpec
) -> tuple[dict[str, _EpisodeInfoValue], Tensor, Tensor]:
    with h5py.File(str(path), "r") as handle:
        info = _index_action_handle(handle)
        indices = np.arange(len(_dataset(handle, "action/value")), dtype=np.int64)
        values, valid = _read_actions(handle, indices, action_spec)
    return info, values, valid


def _object_poses(handle: h5py.File, indices: NDArray[np.int64]) -> dict[str, Tensor]:
    """Express current and future poses in the robot base, with continuous xyzw signs."""
    base = _rigid(np.asarray(_dataset(handle, "policy/T_robot_base_from_world")[:]))
    whole = np.asarray(
        _dataset(handle, "supervision/target_object_pose_world")[:], dtype=np.float64
    )
    base_positions = whole[:, :3] @ base[:3, :3].T + base[:3, 3]
    rotations = np.asarray(Rotation.from_quat(whole[:, 3:]).as_matrix(), dtype=np.float64)
    base_quaternions = _quaternion_sequence(np.asarray(base[:3, :3] @ rotations))
    future_indices = indices[:, None] + np.arange(1, _HORIZON + 1, dtype=np.int64)
    future_valid = future_indices < len(whole)
    future_position = np.zeros((*future_indices.shape, 3), dtype=np.float64)
    future_quaternion = np.zeros((*future_indices.shape, 4), dtype=np.float64)
    future_position[future_valid] = base_positions[future_indices[future_valid]]
    future_quaternion[future_valid] = base_quaternions[future_indices[future_valid]]
    flip = np.sum(future_quaternion * base_quaternions[indices, None], axis=-1) < 0
    future_quaternion[flip] *= -1
    return {
        "future_anchor_position_robot_base_m": _tensor(future_position, np.dtype(np.float32)),
        "future_anchor_quaternion_robot_base_xyzw": _tensor(
            future_quaternion, np.dtype(np.float32)
        ),
        "future_anchor_valid": _tensor(future_valid),
        "current_object_position_robot_base_m": _tensor(
            base_positions[indices], np.dtype(np.float32)
        ),
        "current_object_quaternion_robot_base_xyzw": _tensor(
            base_quaternions[indices], np.dtype(np.float32)
        ),
    }
