"""Action flow matching and six auxiliary losses on one OnTimeBench chunk."""

from typing import cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel

from foveawam.data.batch import TrainingBatch
from foveawam.data.normalization import ActionNormalization
from foveawam.models.io import ModelInput
from foveawam.models.model import _FoveaWAMModel
from foveawam.objectives.action import (
    _action_flow_terms,
    _prepare_action_flow_target,
    _reduce_action_flow_terms,
    _sample_action_flow_target,
)
from foveawam.objectives.future import evaluate_future_displacements, evaluate_future_regression
from foveawam.objectives.object_reconstruction import evaluate_object_reconstruction
from foveawam.objectives.pose import evaluate_object_pose
from foveawam.perception.future_targets import encode_future_dino_targets


def training_objective_counts(batch: TrainingBatch) -> dict[str, Tensor]:
    """Count labels before transfer so each update can normalize over every rank and chunk."""

    def count(mask: Tensor) -> Tensor:
        return torch.tensor(np.count_nonzero(mask.numpy()), dtype=torch.int64)

    masks = batch.masks
    slots = batch.prediction_mask[..., None]
    return {
        "action": count(slots & masks.action_valid),
        "pose": count(slots & masks.future_anchor_valid),
        "reconstruction": 3 * count(masks.current_object_region_valid),
        "future_environment": count(masks.future_scene_tracks_valid),
        "future_depth": count(masks.future_depth_valid),
        "future_dino": 196 * count(slots & masks.future_record_valid),
        "future_dynamic_rgb": count(masks.future_dynamic_mask),
    }


def compute_training_step(
    model: nn.Module,
    batch: TrainingBatch,
    normalization: ActionNormalization,
    config: DictConfig,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Return local means and their counts; all eight action draws share one world encoding."""
    core = cast(
        _FoveaWAMModel, model.module if isinstance(model, DistributedDataParallel) else model
    )
    objective = config.objective
    targets, masks = batch.targets, batch.masks
    target = _prepare_action_flow_target(
        action_spec=core.action_spec,
        normalization=normalization,
        action_physical=targets.action_physical,
        action_valid=masks.action_valid,
        prediction_mask=batch.prediction_mask,
        future_record_valid=masks.future_record_valid,
    )
    draws = [_sample_action_flow_target(target) for _ in range(objective.action.noise_repeats)]
    inputs = ModelInput(
        view_names=batch.view_names,
        rgb=batch.rgb,
        timestamps_s=batch.timestamps_s,
        frame_mask=batch.frame_mask,
        prediction_mask=batch.prediction_mask,
        eef_position_robot_base_m=batch.eef_position_robot_base_m,
        eef_quaternion_robot_base_xyzw=batch.eef_quaternion_robot_base_xyzw,
        gripper_open_fraction=batch.gripper_open_fraction,
        camera_intrinsics=batch.camera_intrinsics,
        T_robot_base_from_camera=batch.T_robot_base_from_camera,
        depth_m=batch.depth_m,
        depth_valid=batch.depth_valid,
        anchor_position_m=batch.anchor_position_m,
        anchor_quaternion_xyzw=batch.anchor_quaternion_xyzw,
        anchor_valid=batch.anchor_valid,
        action_state_normalized=draws[0].state_normalized,
        action_tau=draws[0].tau,
    )
    output, velocities = model(
        inputs,
        _action_noise_batch=(
            torch.stack([draw.state_normalized for draw in draws]),
            torch.stack([draw.tau for draw in draws]),
        ),
    )
    action = _reduce_action_flow_terms(
        [
            _action_flow_terms(velocity, draw.velocity_target_normalized, draw.valid)
            for velocity, draw in zip(velocities, draws, strict=True)
        ],
    )
    pose = evaluate_object_pose(
        output.future_object_pose_delta,
        current_position_m=targets.current_object_position_robot_base_m,
        current_quaternion_xyzw=targets.current_object_quaternion_robot_base_xyzw,
        future_position_m=targets.future_anchor_position_robot_base_m,
        future_quaternion_xyzw=targets.future_anchor_quaternion_robot_base_xyzw,
        valid=masks.future_anchor_valid & batch.prediction_mask[..., None],
    )
    region = evaluate_object_reconstruction(
        output.current_object_region_rgb.float(),
        targets.current_object_region_rgb,
        masks.current_object_region_valid,
        batch.prediction_mask,
    )
    future = output.future_predictions
    scene = evaluate_future_displacements(
        future["environment"],
        targets.future_scene_displacements_robot_base_m,
        masks.future_scene_tracks_visible,
        masks.future_scene_tracks_valid,
        scale_m=config.model.future_displacement_scale_m,
        visibility_weight=objective.future_environment.visibility_weight,
    )
    teacher = targets.future_dino
    if teacher is None:
        teacher = encode_future_dino_targets(
            core.future_dino_teacher,
            targets.future_rgb,
            batch.prediction_mask,
            future_record_valid=masks.future_record_valid,
        )["future_dino"]
    teacher = functional.layer_norm(teacher, (1024,)).float()
    record_valid = batch.prediction_mask[..., None] & masks.future_record_valid
    return {
        "action": action,
        "pose": pose,
        "reconstruction": region,
        "future_environment": scene,
        "future_depth": evaluate_future_regression(
            future["depth"],
            targets.future_depth_m,
            masks.future_depth_valid.squeeze(3),
            channel_axis=3,
        ),
        "future_dino": evaluate_future_regression(
            future["dino"],
            teacher,
            record_valid[..., None].expand(*teacher.shape[:-1]),
            channel_axis=-1,
        ),
        "future_dynamic_rgb": evaluate_future_regression(
            future["dynamic_rgb"],
            targets.future_rgb.float() / 255.0,
            masks.future_dynamic_mask.squeeze(3),
            channel_axis=3,
        ),
    }
