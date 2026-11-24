"""Encode current RGB-D patches and end-effector history."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from foveawam.models.common.embeddings import _IndexEmbedding, _VectorEmbedding
from foveawam.models.common.numeric_checks import _check_numeric
from foveawam.models.common.precision import BF16_OPTIONS, CUDAGRAPH_OPTIONS
from foveawam.perception.frontend import _PerceptionFeatures

_PATCH_COUNT = 196


class _PreparedLinear(nn.Linear):
    """Preserve module-call semantics while reusing one prepared parameter cast."""

    def forward(
        self,
        input: Tensor,
        *,
        prepared_weight: Tensor | None = None,
        prepared_bias: Tensor | None = None,
    ) -> Tensor:
        assert prepared_weight is not None
        return functional.linear(input, prepared_weight, prepared_bias)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_FramePacking = Callable[..., tuple[Tensor, Tensor]]
_disable_frame_packing = cast(
    Callable[..., Callable[[_FramePacking], _FramePacking]],
    torch.compiler.disable,
)(recursive=False)


@dataclass(frozen=True, slots=True)
class _FrameEncoding:
    owner: object
    scene_tokens: Tensor
    robot_token: Tensor
    scene_position_robot_base_m: Tensor
    scene_position_valid: Tensor
    object_position_m: Tensor
    object_quaternion_xyzw: Tensor


class _FrameEncoder(nn.Module):
    """Encode unique observation frames without assigning attention groups."""

    def __init__(self, vision_dimension: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vision_projection = _PreparedLinear(vision_dimension, hidden_size)
        self.view_embedding = _IndexEmbedding(1, hidden_size)
        self.depth_projection = _PreparedLinear(1, hidden_size, bias=False)
        self.geometry_projection = _PreparedLinear(3, hidden_size, bias=False)
        self.robot_embedding = _VectorEmbedding(7, hidden_size)
        self._call_visual_projection_parameters = self._visual_projection_parameters
        self._call_visual_frame = self._visual_frame
        self._call_robot_frame = self._robot_frame
        self._call_robot_history: Callable[[Tensor, Tensor], Tensor] | None = None

    def prepare_compiled_inference(self) -> None:
        self._call_visual_projection_parameters = torch.compile(
            self._visual_projection_parameters, options=BF16_OPTIONS
        )
        self._call_visual_frame = torch.compile(self._visual_frame, options=BF16_OPTIONS)
        self._call_robot_frame = torch.compile(self._robot_frame, options=BF16_OPTIONS)
        self._call_robot_history = torch.compile(
            self._robot_history, fullgraph=True, dynamic=False, options=CUDAGRAPH_OPTIONS
        )

    def _robot_state(
        self,
        *,
        frame_mask: Tensor,
        eef_position_robot_base_m: Tensor,
        eef_quaternion_robot_base_xyzw: Tensor,
        gripper_open_fraction: Tensor,
    ) -> Tensor:
        # Padding quaternions are zero by contract; replace only masked frames.
        identity = torch.zeros_like(eef_quaternion_robot_base_xyzw)
        identity[..., 3] = 1
        q = torch.where(frame_mask[..., None], eef_quaternion_robot_base_xyzw, identity)
        norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        _check_numeric((norm > 0).all(), "real EEF quaternions must be nonzero")
        q = q / norm
        xyz = q[..., :3]
        first = torch.where(
            xyz[..., 0] != 0,
            xyz[..., 0],
            torch.where(xyz[..., 1] != 0, xyz[..., 1], xyz[..., 2]),
        )
        flip = (q[..., 3] < 0) | ((q[..., 3] == 0) & (first < 0))
        q = torch.where(flip[..., None], -q, q)
        length = torch.linalg.vector_norm(q[..., :3], dim=-1)
        angle = 2 * torch.atan2(length, q[..., 3].clamp_min(0))
        scale = torch.where(
            length > 1e-8, angle / length.clamp_min(1e-8), torch.full_like(length, 2)
        )
        rotvec = (q[..., :3] * scale[..., None]).masked_fill(~frame_mask[..., None], 0)
        return torch.cat(
            (eef_position_robot_base_m, rotvec, gripper_open_fraction[..., None]), dim=-1
        )

    def _visual_frame(
        self,
        *,
        vision: Tensor,
        depth: Tensor,
        depth_valid: Tensor,
        position: Tensor,
        position_valid: Tensor,
        view_bias: Tensor,
        vision_weight: Tensor,
        vision_bias: Tensor,
        depth_weight: Tensor,
        geometry_weight: Tensor,
    ) -> Tensor:
        tokens: Tensor = self.vision_projection(
            vision, prepared_weight=vision_weight, prepared_bias=vision_bias
        )
        tokens = tokens + view_bias
        tokens = (
            tokens
            + self.depth_projection(depth, prepared_weight=depth_weight) * depth_valid[..., None]
        )
        tokens = (
            tokens
            + self.geometry_projection(position, prepared_weight=geometry_weight)
            * position_valid[..., None]
        )
        return tokens[:, :, 0].contiguous().reshape(-1, _PATCH_COUNT, self.hidden_size).squeeze(0)

    def _visual_projection_parameters(
        self, vision: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bias = self.vision_projection.bias
        assert bias is not None
        weight = self.vision_projection.weight
        depth_weight = self.depth_projection.weight
        geometry_weight = self.geometry_projection.weight
        if torch.is_autocast_enabled(vision.device.type):
            dtype = torch.get_autocast_dtype(vision.device.type)
            return (
                weight.to(dtype),
                bias.to(dtype),
                depth_weight.to(dtype),
                geometry_weight.to(dtype),
            )
        return weight, bias, depth_weight, geometry_weight

    def _robot_frame(self, state: Tensor) -> Tensor:
        return (
            cast(
                Tensor,
                self.robot_embedding(
                    state, _validated=self._aggregate_validation(self.robot_embedding)
                ),
            )
            .reshape(-1, self.hidden_size)
            .squeeze(0)
        )

    def _aggregate_validation(self, module: nn.Module) -> bool:
        # Hooks may replace inputs before forward or repair outputs afterward.
        # Their original checked-module ordering must remain observable.
        return not (
            self.training
            or module._forward_pre_hooks
            or module._forward_hooks
            or torch.nn.modules.module._global_forward_pre_hooks
            or torch.nn.modules.module._global_forward_hooks
        )

    def _robot_history(self, state: Tensor, valid: Tensor) -> Tensor:
        # Project each frame separately, as at evaluation, but trace the eight
        # projections into one fixed-shape graph.
        rows = [
            self._robot_frame(state[index : index + 1].unsqueeze(1))
            for index in range(state.shape[0])
        ]
        return torch.stack(rows).masked_fill(~valid[:, None], 0).float()

    @_disable_frame_packing
    def _encode_real_frames(
        self,
        frame_mask: Tensor,
        vision: Tensor,
        view_bias: Tensor,
        depth: Tensor,
        depth_valid: Tensor,
        position: Tensor,
        position_valid: Tensor,
        robot_state: Tensor,
        current_visual_only: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Batch independent training frames, using singleton projections at evaluation."""
        batch, frames = (int(axis) for axis in frame_mask.shape)
        indices = torch.nonzero(frame_mask.reshape(-1), as_tuple=False).flatten()
        _require(indices.numel() > 0, "frame encoder requires at least one real frame")
        batch_frames = self.training
        selected_indices = (indices,) if batch_frames else torch.split(indices, 1)
        visual_indices = indices
        if current_visual_only:
            visual_indices = indices[(indices + 1) % frames == 0]
        selected_visual_indices = (
            (visual_indices,) if batch_frames else torch.split(visual_indices, 1)
        )
        flat_frame_mask = frame_mask.reshape(-1)
        flat_vision = vision.flatten(0, 1)
        flat_depth, flat_depth_valid = depth.flatten(0, 1), depth_valid.flatten(0, 1)
        flat_position, flat_position_valid = position.flatten(0, 1), position_valid.flatten(0, 1)
        flat_robot_state = robot_state.flatten(0, 1)
        vision_weight, vision_bias, depth_weight, geometry_weight = (
            self._call_visual_projection_parameters(vision)
        )
        scene_rows = []
        for index in selected_visual_indices:
            scene_rows.append(
                self._call_visual_frame(
                    vision=flat_vision.index_select(0, index).unsqueeze(1),
                    depth=flat_depth.index_select(0, index).unsqueeze(1),
                    depth_valid=flat_depth_valid.index_select(0, index).unsqueeze(1),
                    position=flat_position.index_select(0, index).unsqueeze(1),
                    position_valid=flat_position_valid.index_select(0, index).unsqueeze(1),
                    vision_weight=vision_weight,
                    vision_bias=vision_bias,
                    depth_weight=depth_weight,
                    geometry_weight=geometry_weight,
                    view_bias=view_bias,
                )
            )
        scene_packed = torch.cat(
            [row.reshape(-1, _PATCH_COUNT, self.hidden_size) for row in scene_rows]
        ).to(dtype=torch.float32)
        _check_numeric(
            torch.isfinite(scene_packed).all(), "projected patch features must be finite"
        )
        scene_flat = scene_packed.new_zeros(
            (batch * frames, _PATCH_COUNT, self.hidden_size)
        ).index_copy(0, visual_indices, scene_packed)
        scene_tokens = scene_flat.reshape(
            batch, frames, _PATCH_COUNT, self.hidden_size
        ).contiguous()

        if self._aggregate_validation(self.robot_embedding):
            # Validate exactly the real states in a fixed-shape tensor. Selecting
            # them first specializes the validator on the startup's real count.
            self.robot_embedding._validate(
                flat_robot_state.masked_fill(~flat_frame_mask[:, None], 0)
            )
        packed_history = (
            self._call_robot_history is not None
            and not batch_frames
            and all(self._aggregate_validation(module) for module in self.robot_embedding.modules())
        )
        if packed_history:
            assert self._call_robot_history is not None
            robot_flat = self._call_robot_history(
                flat_robot_state.masked_fill(~flat_frame_mask[:, None], 0), flat_frame_mask
            ).unsqueeze(1)
        else:
            robot_rows = [
                self._call_robot_frame(flat_robot_state.index_select(0, index).unsqueeze(1))
                for index in selected_indices
            ]
            robot_packed = torch.cat(
                [row.reshape(-1, self.hidden_size) for row in robot_rows]
            ).float()
            robot_flat = robot_packed.new_zeros((batch * frames, 1, self.hidden_size)).index_copy(
                0, indices, robot_packed.unsqueeze(1)
            )
        robot_token = robot_flat.reshape(batch, frames, 1, self.hidden_size).contiguous()
        return scene_tokens, robot_token

    def forward(
        self,
        *,
        perception: _PerceptionFeatures,
        frame_mask: Tensor,
        eef_position_robot_base_m: Tensor,
        eef_quaternion_robot_base_xyzw: Tensor,
        gripper_open_fraction: Tensor,
        anchor_position_m: Tensor,
        anchor_quaternion_xyzw: Tensor,
        anchor_valid: Tensor,
        current_visual_only: bool = False,
    ) -> _FrameEncoding:
        vision = perception.vision_tokens
        position = perception.robot_base_patch_source_m
        position_valid = perception.geometry_patch_valid
        view_bias = self.view_embedding.embedding(torch.arange(1, device=vision.device))[
            None, None, :, None, :
        ]
        robot_state = self._robot_state(
            frame_mask=frame_mask,
            eef_position_robot_base_m=eef_position_robot_base_m,
            eef_quaternion_robot_base_xyzw=eef_quaternion_robot_base_xyzw,
            gripper_open_fraction=gripper_open_fraction,
        )
        scene, robot = self._encode_real_frames(
            frame_mask,
            vision,
            view_bias,
            perception.depth_patch_m,
            perception.depth_patch_valid,
            position,
            position_valid,
            robot_state,
            current_visual_only,
        )
        return _FrameEncoding(
            owner=self,
            scene_tokens=scene,
            robot_token=robot,
            object_position_m=anchor_position_m,
            object_quaternion_xyzw=anchor_quaternion_xyzw,
            scene_position_robot_base_m=position[:, :, 0]
            .masked_fill(~frame_mask[:, :, None, None], 0)
            .contiguous(),
            scene_position_valid=(position_valid[:, :, 0] & frame_mask[:, :, None]).contiguous(),
        )
