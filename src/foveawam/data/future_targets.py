"""Future RGB-D and scene displacement labels on the 14 by 14 patch grid."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from foveawam.data.ontimebench import _HORIZON, _dataset, _rigid, _take

DINO_REPRESENTATION = "raw_dinov3_vitl16_fp32_196x1024"


def dino_cache_path(root: str | Path, episode_id: str) -> Path:
    relative = Path(episode_id)
    if relative.anchor or not relative.parts or ".." in relative.parts:
        raise ValueError("DINO cache episode_id must be a relative episode name")
    return Path(root) / (str(relative) + ".h5")


def _validate_cached_features(
    handle: h5py.File, episode_id: str, source: Path, frames: int
) -> h5py.Dataset:
    stat = source.stat()
    identity = {
        "complete": 1,
        "episode_id": episode_id,
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "representation": DINO_REPRESENTATION,
    }
    if any(handle.attrs.get(name) != value for name, value in identity.items()):
        raise ValueError(f"DINO cache does not match {episode_id} in {source}")
    features = _dataset(handle, "features")
    if features.shape != (frames, 196, 1024) or features.dtype != np.float32:
        raise ValueError(f"Expected FP32 DINO features [{frames},196,1024] for {episode_id}")
    return features


def read_dino_episode_cache(
    root: str | Path,
    episode_id: str,
    source: Path,
    indices: NDArray[np.int64],
    *,
    expected_frames: int,
) -> Tensor:
    with h5py.File(str(dino_cache_path(root, episode_id)), "r") as handle:
        cached = _validate_cached_features(handle, episode_id, source, expected_frames)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= expected_frames - 1):
            raise ValueError("DINO targets require a real observation and at least one successor")
        future = indices[:, None] + np.arange(1, _HORIZON + 1, dtype=np.int64)
        valid = future < expected_frames
        unique, inverse = np.unique(future[valid], return_inverse=True)
        values = np.asarray(cached[unique], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite DINO teacher cache")
        output = np.zeros((*future.shape, 196, 1024), dtype=np.float32)
        output[valid] = values[inverse]
    return torch.from_numpy(output)


def _scene_displacements(
    depth: NDArray[np.float32],
    geom: NDArray[np.int32],
    transforms: NDArray[np.float64],
    cameras: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    base_rotation: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_]]:
    """Track each current patch centre in its geometry's local frame."""
    height, width = depth.shape[1:]
    y, x = np.indices((14, 14)).reshape(2, -1)
    y = ((2 * y + 1) * height - 1) // 28
    x = ((2 * x + 1) * width - 1) // 28
    displacement = np.zeros((len(x), _HORIZON, 3), dtype=np.float32)
    valid = np.zeros((len(x), _HORIZON), dtype=np.bool_)
    visible = np.zeros_like(valid)
    identity = geom[0, y, x]
    good = (
        np.isfinite(depth[0, y, x])
        & (depth[0, y, x] > 0)
        & (identity >= 0)
        & (identity < transforms.shape[1])
    )
    rows = np.flatnonzero(good)
    if not len(rows):
        return displacement, valid, visible
    pixels = np.column_stack((x[rows], y[rows], np.ones(len(rows))))
    camera_points = np.asarray(
        (pixels @ np.linalg.inv(intrinsics[0]).T) * depth[0, y[rows], x[rows], None],
        dtype=np.float64,
    )
    world = camera_points @ cameras[0, :3, :3].T + cameras[0, :3, 3]
    ids = identity[rows]
    local = np.einsum("nji,nj->ni", transforms[0, ids, :3, :3], world - transforms[0, ids, :3, 3])
    for h in range(len(transforms) - 1):
        future = (
            np.einsum("nij,nj->ni", transforms[h + 1, ids, :3, :3], local)
            + transforms[h + 1, ids, :3, 3]
        )
        cam = (future - cameras[h + 1, :3, 3]) @ cameras[h + 1, :3, :3]
        projected = cam @ intrinsics[h + 1].T
        finite = np.isfinite(projected).all(-1)
        projectable = finite & (cam[:, 2] > 0)
        uv = np.zeros((len(rows), 2), dtype=np.float64)
        uv[projectable] = projected[projectable, :2] / projected[projectable, 2, None]
        valid[rows, h] = projectable
        physical = (future - world) @ base_rotation.T
        if not np.isfinite(physical[projectable]).all():
            raise ValueError("Nonfinite displacement for a projectable scene point")
        displacement[rows[projectable], h] = physical[projectable]
        inside = (
            projectable
            & (uv[:, 0] >= 0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= height - 1)
        )
        selected = np.flatnonzero(inside)
        u, v = np.floor(uv[selected] + 0.5).astype(np.int64).T
        visible[rows[selected], h] = (geom[h + 1, v, u] == ids[selected]) & (
            np.abs(depth[h + 1, v, u] - cam[selected, 2])
            <= np.maximum(0.001, 0.001 * cam[selected, 2])
        )
    return displacement, valid, visible


def _future_dynamic_mask(
    depth: NDArray[np.float32],
    geom: NDArray[np.int32],
    transforms: NDArray[np.float64],
    cameras: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Future visible surfaces moving >1 mm in world space from current pose.

    The test includes every geometry role, including robot and moving distractors.
    Future depth reconstructs each visible surface point, then the exact same
    local point is transformed at current time. Camera motion is excluded.
    """
    result = np.zeros((_HORIZON, 1, 224, 224), dtype=np.bool_)
    for h in range(len(transforms) - 1):
        # An unchanged world transform cannot move any point of that geometry.
        # Exclude those surfaces before materializing a rotation per image pixel.
        moving_ids = np.flatnonzero(np.any(transforms[h + 1] != transforms[0], axis=(-2, -1)))
        valid = (
            np.isfinite(depth[h + 1])
            & (depth[h + 1] > 0)
            & (geom[h + 1] >= 0)
            & (geom[h + 1] < transforms.shape[1])
            & np.isin(geom[h + 1], moving_ids)
        )
        y, x = np.nonzero(valid)
        ids = geom[h + 1, y, x]
        pixels = cast(NDArray[np.float64], np.column_stack((x, y, np.ones(len(y)))))
        cam = (pixels @ np.linalg.inv(intrinsics[h + 1]).T) * depth[h + 1, y, x, None]
        world = cam @ cameras[h + 1, :3, :3].T + cameras[h + 1, :3, 3]
        local = np.einsum(
            "nji,nj->ni", transforms[h + 1, ids, :3, :3], world - transforms[h + 1, ids, :3, 3]
        )
        current = (
            np.einsum("nij,nj->ni", transforms[0, ids, :3, :3], local) + transforms[0, ids, :3, 3]
        )
        result[h, 0, y, x] = np.linalg.norm(world - current, axis=-1) > 0.001
    return result


def _read_future_targets(
    handle: h5py.File, indices: NDArray[np.int64], frames: int
) -> dict[str, Tensor]:
    """Read the overlapping horizons once and pad after the last observation."""
    fields: dict[str, list[NDArray[np.generic]]] = {}

    def append(name: str, value: NDArray[np.generic]) -> None:
        fields.setdefault(name, []).append(value)

    def padded[Scalar: np.generic](value: NDArray[Scalar]) -> NDArray[Scalar]:
        result = np.zeros((_HORIZON, *value.shape[1:]), dtype=value.dtype)
        result[: len(value)] = value
        return result

    requested = [np.arange(i, min(i + _HORIZON + 1, frames), dtype=np.int64) for i in indices]
    rows = np.unique(np.concatenate(requested))
    rgb = _take(_dataset(handle, "policy/rgb"), rows, np.dtype(np.uint8))[:, 0]
    depth = _take(_dataset(handle, "policy/depth_m"), rows, np.dtype(np.float32))[:, 0]
    geom = _take(_dataset(handle, "supervision/geom_id"), rows, np.dtype(np.int32))[:, 0]
    transforms = _rigid(
        _take(_dataset(handle, "supervision/world_from_geom"), rows, np.dtype(np.float64))
    )
    cameras = _rigid(
        _take(_dataset(handle, "policy/world_from_camera_cv"), rows, np.dtype(np.float64))[:, 0]
    )
    intrinsics = _take(_dataset(handle, "policy/camera_intrinsics"), rows, np.dtype(np.float64))[
        :, 0
    ]
    base_rotation = _rigid(np.asarray(_dataset(handle, "policy/T_robot_base_from_world")[:]))[:3, :3]
    for future in requested:
        select = np.searchsorted(rows, future)
        geometry = (
            depth[select],
            geom[select],
            transforms[select],
            cameras[select],
            intrinsics[select],
        )
        motion, valid, visible = _scene_displacements(*geometry, base_rotation)
        append("future_scene_displacements_robot_base_m", motion)
        append("future_scene_tracks_valid", valid)
        append("future_scene_tracks_visible", visible)
        append("future_dynamic_mask", _future_dynamic_mask(*geometry))
        metric = depth[select[1:]]
        depth_valid = np.isfinite(metric) & (metric > 0)
        append("future_depth_m", padded(np.where(depth_valid, metric, 0)[:, None]))
        append("future_depth_valid", padded(depth_valid[:, None]))
        append("future_rgb", padded(rgb[select[1:]].transpose(0, 3, 1, 2)))
    return {
        name: torch.from_numpy(np.ascontiguousarray(np.stack(values)))
        for name, values in fields.items()
    }
