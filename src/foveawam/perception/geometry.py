"""Back-project depth to the robot base frame and align it with image patches."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as functional

_PATCHES_PER_AXIS = 14
_PATCH_COUNT = _PATCHES_PER_AXIS**2
_VISION_SIZE = 224


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tensor(
    name: str,
    value: Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    _require(value.dtype == dtype, f"{name} has the wrong dtype")
    _require(tuple(value.shape) == shape, f"{name} has the wrong shape")
    _require(value.device == device, f"{name} is on the wrong device")


def _align_image_values(value: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    _require(value.ndim == 6, "patch-aligned values must have axes [B,F,V,C,H,W]")
    _require(valid.ndim == 6 and int(valid.shape[3]) == 1, "patch validity has wrong axes")
    _require(
        tuple(valid.shape[:3]) == tuple(value.shape[:3])
        and tuple(valid.shape[-2:]) == tuple(value.shape[-2:]),
        "patch value and validity axes differ",
    )
    _require(value.is_floating_point(), "patch-aligned values must be floating point")
    _require(valid.dtype == torch.bool, "patch validity must be boolean")
    batch, frames, views, channels, height, width = (int(axis) for axis in value.shape)
    flattened_value = value.reshape(batch * frames * views, channels, height, width)
    flattened_valid = valid.reshape(batch * frames * views, 1, height, width)
    weights = flattened_valid.to(value.dtype)
    aligned_weights = functional.interpolate(
        weights,
        size=(_VISION_SIZE, _VISION_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    aligned_numerator = functional.interpolate(
        flattened_value * weights,
        size=(_VISION_SIZE, _VISION_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    aligned_valid = aligned_weights > 0
    aligned = (
        aligned_numerator / aligned_weights.clamp_min(torch.finfo(value.dtype).tiny)
    ).masked_fill(
        ~aligned_valid,
        0,
    )
    return (
        aligned.reshape(batch, frames, views, channels, _VISION_SIZE, _VISION_SIZE).contiguous(),
        aligned_valid.reshape(batch, frames, views, 1, _VISION_SIZE, _VISION_SIZE).contiguous(),
    )


def _pool_patch_grid(value: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    batch, frames, views, channels, height, width = (int(axis) for axis in value.shape)
    flattened_value = value.reshape(batch * frames * views, channels, height, width)
    flattened_valid = valid.reshape(batch * frames * views, 1, height, width)
    weights = flattened_valid.to(value.dtype)
    numerator = functional.adaptive_avg_pool2d(
        flattened_value * weights,
        (_PATCHES_PER_AXIS, _PATCHES_PER_AXIS),
    )
    denominator = functional.adaptive_avg_pool2d(
        weights,
        (_PATCHES_PER_AXIS, _PATCHES_PER_AXIS),
    )
    patch_valid = denominator > 0
    averaged = (numerator / denominator.clamp_min(torch.finfo(value.dtype).tiny)).masked_fill(
        ~patch_valid,
        0,
    )
    averaged = averaged.permute(0, 2, 3, 1).reshape(
        batch,
        frames,
        views,
        _PATCH_COUNT,
        channels,
    )
    patch_valid = patch_valid.flatten(start_dim=2).reshape(
        batch,
        frames,
        views,
        _PATCH_COUNT,
    )
    return averaged.contiguous(), patch_valid.contiguous()


def _patch_average(value: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Align and average valid image values without invalid-pixel leakage."""

    aligned, aligned_valid = _align_image_values(value, valid)
    return _pool_patch_grid(aligned, aligned_valid)


def _robot_base_patch_source_points_checked_values(
    depth_m: Tensor,
    depth_valid: Tensor,
    camera_intrinsics: Tensor,
    T_robot_base_from_camera: Tensor,  # noqa: N803
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the original per-image numerics and its required validity predicate."""

    _require(depth_m.ndim == 6 and int(depth_m.shape[3]) == 1, "Depth has wrong axes")
    _require(tuple(depth_valid.shape) == tuple(depth_m.shape), "Depth validity shape differs")
    _require(depth_m.is_floating_point(), "Depth must be floating point")
    _require(depth_valid.dtype == torch.bool, "Depth validity must be boolean")
    batch, frames, views, _, height, width = (int(axis) for axis in depth_m.shape)
    _require(
        tuple(camera_intrinsics.shape) == (batch, frames, views, 3, 3),
        "camera intrinsics shape differs from Depth",
    )
    _require(
        tuple(T_robot_base_from_camera.shape) == (batch, frames, views, 4, 4),
        "camera transform shape differs from Depth",
    )

    u, v, z, source_valid, inverse_intrinsics, inverse_valid = _patch_source_intrinsics(
        depth_m, depth_valid, camera_intrinsics
    )
    homogeneous_pixel = torch.stack((u, v, torch.ones_like(u)), dim=-1).to(depth_m.dtype)
    # Metric observations must not inherit the neural network's autocast.
    # Returning float32 after a low-precision transform cannot recover lost XYZ.
    with torch.autocast(device_type=depth_m.device.type, enabled=False):
        camera_ray = torch.einsum("bfvij,pj->bfvpi", inverse_intrinsics, homogeneous_pixel)
        camera_xyz = camera_ray * z[..., None]
        rotation = T_robot_base_from_camera[..., :3, :3]
        translation = T_robot_base_from_camera[..., :3, 3]
        robot_base_xyz = torch.einsum("bfvij,bfvpj->bfvpi", rotation, camera_xyz)
        robot_base_xyz = robot_base_xyz + translation[..., None, :]
    return (
        robot_base_xyz.masked_fill(~source_valid[..., None], 0).contiguous(),
        source_valid.contiguous(),
        inverse_valid,
    )


def _patch_source_intrinsics(
    depth_m: Tensor, depth_valid: Tensor, camera_intrinsics: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Sample the same fixed source pixels and check depth-conditioned inverses."""
    height, width = depth_m.shape[-2:]
    patch = torch.arange(_PATCHES_PER_AXIS, dtype=torch.int64, device=depth_m.device)
    # Half-pixel resize centers, nearest integer with exact half ties rounded down.
    rows = ((2 * patch + 1) * height - 1) // (2 * _PATCHES_PER_AXIS)
    columns = ((2 * patch + 1) * width - 1) // (2 * _PATCHES_PER_AXIS)
    v, u = torch.meshgrid(rows, columns, indexing="ij")
    v, u = v.flatten(), u.flatten()
    z = depth_m[..., 0, v, u]
    source_valid = depth_valid[..., 0, v, u] & torch.isfinite(z) & (z > 0)
    z = z.masked_fill(~source_valid, 0)
    has_depth = source_valid.any(dim=-1)
    identity = torch.eye(3, dtype=camera_intrinsics.dtype, device=camera_intrinsics.device)
    safe_intrinsics = torch.where(
        has_depth[..., None, None],
        camera_intrinsics,
        identity.expand_as(camera_intrinsics),
    )
    inverse_intrinsics, info = torch.linalg.inv_ex(safe_intrinsics)
    inverse_valid = (info == 0).all() & torch.isfinite(inverse_intrinsics).all()
    return u, v, z, source_valid, inverse_intrinsics, inverse_valid
