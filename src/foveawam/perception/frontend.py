"""RGB-D patch features in the robot base frame."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch._dynamo.decorators import skip as _skip_frame

from foveawam.models.common.numeric_checks import _check_numeric, _check_numeric_many
from foveawam.perception.geometry import (
    _patch_average,
    _patch_source_intrinsics,
    _robot_base_patch_source_points_checked_values,
)
from foveawam.perception.vision import _MAEVisionEncoder

_PATCH_COUNT = 196
_VISION_SIZE = 224


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_VisionPacking = Callable[["_PerceptionFrontend", Tensor, Tensor], Tensor]
_disable_vision_packing = cast(
    Callable[[_VisionPacking], _VisionPacking],
    _skip_frame,
)


@dataclass(frozen=True, slots=True)
class _PerceptionFeatures:
    vision_tokens: Tensor
    depth_patch_m: Tensor
    depth_patch_valid: Tensor
    robot_base_patch_source_m: Tensor
    geometry_patch_valid: Tensor


_GeometryPacking = Callable[..., tuple[Tensor, Tensor]]
_disable_geometry_packing = cast(Callable[[_GeometryPacking], _GeometryPacking], _skip_frame)


@_disable_geometry_packing
def _per_image_robot_base_patch_source_points(
    depth_m: Tensor,
    depth_valid: Tensor,
    camera_intrinsics: Tensor,
    T_robot_base_from_camera: Tensor,  # noqa: N803
    frame_mask: Tensor,
    *,
    batch_images: bool = False,
) -> tuple[Tensor, Tensor]:
    """Backproject real images together, or singly for fixed evaluation numerics."""

    _require(depth_m.ndim == 6 and int(depth_m.shape[3]) == 1, "Depth has wrong axes")
    batch, frames, views = (int(axis) for axis in depth_m.shape[:3])
    _require(tuple(depth_valid.shape) == tuple(depth_m.shape), "Depth validity shape differs")
    _require(
        tuple(camera_intrinsics.shape) == (batch, frames, views, 3, 3),
        "camera intrinsics have wrong shape",
    )
    _require(
        tuple(T_robot_base_from_camera.shape) == (batch, frames, views, 4, 4),
        "camera transform has wrong shape",
    )
    _require(tuple(frame_mask.shape) == (batch, frames), "frame mask has wrong shape")
    valid_images = frame_mask[:, :, None].expand(batch, frames, views).reshape(-1)
    valid_indices = torch.nonzero(valid_images, as_tuple=False).flatten()
    _require(valid_indices.numel() > 0, "Camera Geometry requires at least one real image")
    groups = (valid_indices,) if batch_images else torch.split(valid_indices, 1)
    fields = tuple(
        value.flatten(0, 2)
        for value in (depth_m, depth_valid, camera_intrinsics, T_robot_base_from_camera)
    )
    xyz_rows: list[Tensor] = []
    valid_rows: list[Tensor] = []
    inverse_checks: list[Tensor] = []
    for indices in groups:
        xyz, valid, inverse_valid = _robot_base_patch_source_points_checked_values(
            *(value.index_select(0, indices)[:, None, None].contiguous() for value in fields),
        )
        xyz_rows.append(xyz.reshape(-1, _PATCH_COUNT, 3))
        valid_rows.append(valid.reshape(-1, _PATCH_COUNT))
        inverse_checks.append(inverse_valid)
    _check_numeric(
        torch.stack(inverse_checks).all(),
        "camera intrinsics must be finite and invertible for backprojection",
    )
    packed_xyz = torch.cat(xyz_rows)
    packed_valid = torch.cat(valid_rows)
    flat_xyz = packed_xyz.new_zeros((batch * frames * views, _PATCH_COUNT, 3)).index_copy(
        0,
        valid_indices,
        packed_xyz,
    )
    flat_valid = packed_valid.new_zeros((batch * frames * views, _PATCH_COUNT)).index_copy(
        0, valid_indices, packed_valid
    )
    return (
        flat_xyz.reshape(batch, frames, views, _PATCH_COUNT, 3).contiguous(),
        flat_valid.reshape(batch, frames, views, _PATCH_COUNT).contiguous(),
    )


class _PerceptionFrontend(nn.Module):
    """Encode RGB with MAE and backproject the aligned depth patches."""

    def __init__(self, vision_encoder: _MAEVisionEncoder) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder

    @_disable_vision_packing
    def _vision_tokens(self, rgb: Tensor, frame_mask: Tensor) -> Tensor:
        """Pack real images outside capture and retain their frame/view order."""

        batch, frames, views = (int(axis) for axis in rgb.shape[:3])
        image_shape = tuple(int(axis) for axis in rgb.shape[3:])
        flat_rgb = rgb.reshape(batch * frames * views, *image_shape)
        valid = frame_mask[:, :, None].expand(batch, frames, views).reshape(-1)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        _require(len(valid_indices) > 0, "perception requires at least one valid frame/view")
        packed_rgb = flat_rgb.index_select(0, valid_indices).contiguous()
        batch_vision = self.training or not torch.are_deterministic_algorithms_enabled()
        if not batch_vision:
            encoded_rows: list[Tensor] = []
            for row in range(len(valid_indices)):
                # Preserve graph output before the next call may reuse its storage.
                encoded_rows.append(self.vision_encoder(packed_rgb[row : row + 1]).clone())
            encoded = torch.cat(encoded_rows)
        else:
            encoded = self.vision_encoder(packed_rgb)
        _require(
            tuple(encoded.shape)
            == (len(valid_indices), _PATCH_COUNT, self.vision_encoder.output_dimension),
            "vision encoder output differs from its fixed patch and channel dimensions",
        )
        channels = self.vision_encoder.output_dimension
        flat_tokens = encoded.new_zeros((batch * frames * views, _PATCH_COUNT, channels))
        flat_tokens.index_copy_(0, valid_indices, encoded)
        return flat_tokens.reshape(
            batch,
            frames,
            views,
            _PATCH_COUNT,
            channels,
        ).contiguous()

    def _features(
        self,
        *,
        vision_tokens: Tensor,
        rgb: Tensor,
        frame_mask: Tensor,
        camera_intrinsics: Tensor,
        T_robot_base_from_camera: Tensor,  # noqa: N803
        depth_m: Tensor,
        depth_valid: Tensor,
        current_visual_only: bool = False,
    ) -> _PerceptionFeatures:
        batch, frames, views = (int(axis) for axis in rgb.shape[:3])
        height, width = (int(axis) for axis in rgb.shape[-2:])
        expected_vision = (batch, frames, views, _PATCH_COUNT, self.vision_encoder.output_dimension)
        _require(tuple(vision_tokens.shape) == expected_vision, "vision patches have wrong axes")
        _require(vision_tokens.dtype == torch.float32, "vision patches must be float32")
        _require(vision_tokens.device == rgb.device, "vision patches and RGB must share a device")
        _require(vision_tokens.is_contiguous(), "vision patches must be contiguous")
        invalid_images = ~frame_mask[:, :, None, None, None]
        finite_vision = torch.isfinite(vision_tokens).all()
        zero_padding = ~((vision_tokens != 0) & invalid_images).any()
        _check_numeric_many(
            (
                (finite_vision, "vision patches must be finite"),
                (zero_padding, "invalid-frame vision patches must be zero"),
            )
        )

        expected_depth = (batch, frames, views, 1, height, width)
        _require(tuple(depth_m.shape) == expected_depth, "Depth has wrong axes")
        _require(tuple(depth_valid.shape) == expected_depth, "Depth validity has wrong axes")
        _require(depth_m.dtype == torch.float32, "Depth must be float32 meters")
        _require(depth_valid.dtype == torch.bool, "Depth validity must be bool")
        _require(
            depth_m.device == rgb.device == depth_valid.device,
            "Depth fields and RGB must share a device",
        )
        _require(
            depth_m.is_contiguous() and depth_valid.is_contiguous(), "Depth must be contiguous"
        )
        finite_depth = torch.isfinite(depth_m).all()
        zero_depth = ~((depth_m != 0) & ~depth_valid).any()
        positive_depth = ((depth_m > 0) | ~depth_valid).all()
        _check_numeric_many(
            (
                (finite_depth, "Depth must be finite"),
                (zero_depth, "invalid Depth must be zero"),
                (positive_depth, "valid Depth must be positive"),
            )
        )
        depth_patch_m, depth_patch_valid = _patch_average(depth_m, depth_valid)
        geometry_mask = frame_mask
        if current_visual_only:
            geometry_mask = torch.zeros_like(frame_mask)
            geometry_mask[:, -1] = frame_mask[:, -1]
            if frame_mask.shape[1] > 1:
                # Historical XYZ is unused, but bad source intrinsics
                # must still fail when sampled historical depth is valid.
                *_, history_inverse_valid = _patch_source_intrinsics(
                    depth_m[:, :-1], depth_valid[:, :-1], camera_intrinsics[:, :-1]
                )
                _check_numeric(
                    history_inverse_valid,
                    "camera intrinsics must be finite and invertible for backprojection",
                )
        geometry_xyz, geometry_valid = _per_image_robot_base_patch_source_points(
            depth_m,
            depth_valid,
            camera_intrinsics,
            T_robot_base_from_camera,
            geometry_mask,
            batch_images=self.training,
        )
        return _PerceptionFeatures(
            vision_tokens=vision_tokens,
            depth_patch_m=depth_patch_m,
            depth_patch_valid=depth_patch_valid,
            robot_base_patch_source_m=geometry_xyz,
            geometry_patch_valid=geometry_valid,
        )

    def forward(
        self,
        *,
        rgb: Tensor,
        frame_mask: Tensor,
        camera_intrinsics: Tensor,
        T_robot_base_from_camera: Tensor,  # noqa: N803
        depth_m: Tensor,
        depth_valid: Tensor,
        current_visual_only: bool = False,
    ) -> _PerceptionFeatures:
        vision_frame_mask = frame_mask
        if current_visual_only:
            # Online current-only conditioning consumes the newest visual tokens.
            # Keep the full frame mask for geometry and all proprio/Anchor history.
            vision_frame_mask = torch.zeros_like(frame_mask)
            vision_frame_mask[:, -1] = frame_mask[:, -1]
        vision_tokens = self._vision_tokens(rgb, vision_frame_mask)
        return self._features(
            vision_tokens=vision_tokens,
            rgb=rgb,
            frame_mask=frame_mask,
            camera_intrinsics=camera_intrinsics,
            T_robot_base_from_camera=T_robot_base_from_camera,
            depth_m=depth_m,
            depth_valid=depth_valid,
            current_visual_only=current_visual_only,
        )
