"""A batch of observations and the labels used by the seven training objectives."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, fields
from typing import cast

import torch
from torch import Tensor


@dataclass(slots=True, kw_only=True)
class TrainingTargets:
    action_physical: Tensor
    current_object_position_robot_base_m: Tensor
    current_object_quaternion_robot_base_xyzw: Tensor
    future_anchor_position_robot_base_m: Tensor
    future_anchor_quaternion_robot_base_xyzw: Tensor
    current_object_region_rgb: Tensor
    future_scene_displacements_robot_base_m: Tensor
    future_depth_m: Tensor
    future_rgb: Tensor
    future_dino: Tensor | None = None


@dataclass(slots=True, kw_only=True)
class TargetMasks:
    action_valid: Tensor
    future_record_valid: Tensor
    future_anchor_valid: Tensor
    current_object_region_valid: Tensor
    future_scene_tracks_valid: Tensor
    future_scene_tracks_visible: Tensor
    future_depth_valid: Tensor
    future_dynamic_mask: Tensor


@dataclass(slots=True, kw_only=True)
class TrainingBatch:
    view_names: tuple[str, ...]
    rgb: Tensor
    timestamps_s: Tensor
    frame_mask: Tensor
    prediction_mask: Tensor
    eef_position_robot_base_m: Tensor
    eef_quaternion_robot_base_xyzw: Tensor
    gripper_open_fraction: Tensor
    camera_intrinsics: Tensor
    T_robot_base_from_camera: Tensor
    depth_m: Tensor
    depth_valid: Tensor
    anchor_position_m: Tensor
    anchor_quaternion_xyzw: Tensor
    anchor_valid: Tensor
    targets: TrainingTargets
    masks: TargetMasks
    episode_id: tuple[str, ...]
    task_id: tuple[str, ...]
    episode_seed: tuple[int, ...]
    prediction_frame_index: Tensor

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> TrainingBatch:
        return self._map_tensors(lambda value: value.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> TrainingBatch:
        return self._map_tensors(Tensor.pin_memory)

    def _map_tensors(self, fn: Callable[[Tensor], Tensor]) -> TrainingBatch:
        def mapped[Group: (TrainingBatch, TrainingTargets, TargetMasks)](group: Group) -> Group:
            result = copy(group)
            for field in fields(group):
                value = getattr(group, field.name)
                if isinstance(value, Tensor):
                    setattr(result, field.name, fn(value))
            return result

        result = mapped(self)
        result.targets = mapped(self.targets)
        result.masks = mapped(self.masks)
        return result


def collate_episode_chunks(samples: list[dict[str, object]]) -> TrainingBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")

    def stack(name: str) -> Tensor:
        return torch.stack([cast(Tensor, sample[name]) for sample in samples])

    has_dino = ["future_dino" in sample for sample in samples]
    if any(has_dino) != all(has_dino):
        raise ValueError("a batch must use one DINO target source")
    targets = TrainingTargets(
        **{f.name: stack(f.name) for f in fields(TrainingTargets) if f.name != "future_dino"},
        future_dino=stack("future_dino") if all(has_dino) else None,
    )
    masks = TargetMasks(**{f.name: stack(f.name) for f in fields(TargetMasks)})
    metadata = ("view_names", "episode_id", "task_id", "episode_seed", "targets", "masks")
    return TrainingBatch(
        **{f.name: stack(f.name) for f in fields(TrainingBatch) if f.name not in metadata},
        view_names=("agentview",),
        episode_id=tuple(cast(str, sample["episode_id"]) for sample in samples),
        task_id=tuple(cast(str, sample["task_id"]) for sample in samples),
        episode_seed=tuple(cast(int, sample["episode_seed"]) for sample in samples),
        targets=targets,
        masks=masks,
    )
