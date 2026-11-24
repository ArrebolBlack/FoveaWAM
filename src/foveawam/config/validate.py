"""Validate a training recipe or an exported OnTimeBench policy."""

import math

from omegaconf import DictConfig, OmegaConf


def _check_dimensions(model: DictConfig) -> None:
    if (
        model.history_length != 8
        or model.num_layers < 2
        or model.num_attention_heads < 1
        or model.hidden_size < 1
        or model.hidden_size % model.num_attention_heads
        or not model.mlp_ratio > 0
    ):
        raise ValueError("invalid FoveaWAM history, depth, width or head count")
    if len(model.object_input_mean) != 25 or len(model.object_input_std) != 25:
        raise ValueError("object kinematics need 25 normalization statistics")
    if not all(math.isfinite(v) for v in model.object_input_mean) or not all(
        math.isfinite(v) and v > 0 for v in model.object_input_std
    ):
        raise ValueError("object kinematics normalization is invalid")
    if (
        not math.isfinite(model.future_displacement_scale_m)
        or model.future_displacement_scale_m <= 0
    ):
        raise ValueError("scene displacement scale must be finite and positive")


def _validate_deployment_config(config: DictConfig) -> None:
    model = config.model
    _check_dimensions(model)
    if model.architecture_id != "FoveaWAM" or model.action_spec != {
        "mode": "E7",
        "representation": "delta",
        "gripper_mode": "discrete",
        "motion_limits": [0.10] * 3 + [0.50] * 3,
    }:
        raise ValueError("invalid FoveaWAM policy or delta E7 execution convention")
    # The original public checkpoint describes its architecture in these additional fields.
    recorded = {
        "action_horizon": 8,
        "future_horizon": 8,
        "views": ["agentview"],
        "anchor": {"enabled": True},
        "camera_geometry": {"enabled": True},
        "streams": {"action": True, "flow": True},
        "parameterization": "split_ffn",
        "attention_implementation": "factorized",
        "dropout_p": 0.0,
        "visual_history_encoder": "current",
        "history_query_tokens": 0,
        "proprio_kind": "eef7",
        "proprio_history": "history",
        "se3_rotation": "rotvec",
        "action_rotation_codec": "rotvec",
        "object_input_rotation": "rotvec",
        "object_input_encoding": "kinematic_tokens",
        "object_input_mode": "current_kinematics",
        "future_environment": "sparse_scene_displacements",
        "future_object": "se3",
        "future_depth": True,
        "future_dino": True,
        "future_dynamic_rgb": True,
        "object_region_reconstruction": True,
        "action_attention_scene_keys": True,
        "action_attention_separate_projections": False,
        "object_pose_late_injection": False,
        "future_track_fm_scale": None,
        "visual_history_length": None,
    }
    dimensions = {
        "architecture_id",
        "action_spec",
        "history_length",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "mlp_ratio",
        "object_input_mean",
        "object_input_std",
        "future_displacement_scale_m",
    }
    extra = set(model) - dimensions
    if not extra <= recorded.keys() or any(model[key] != recorded[key] for key in extra):
        raise ValueError("checkpoint architecture differs from the OnTimeBench model")
    if (
        config.perception.vision
        != {
            "identity": "mae_vitb16",
            "input_size_px": 224,
            "trainable": False,
            "frozen_precision": "bf16",
        }
        or config.perception.depth != {"source": "simulator_exact"}
        or config.perception.flow != {"source": "disabled"}
    ):
        raise ValueError("checkpoint must use the BF16 MAE and simulator depth")
    inference = config.inference
    if (
        inference.solver.method != "euler"
        or inference.solver.nfe != 5
        or inference.solver.get("velocity_cache", False)
        or inference.get("temporal_cache", False)
        or not inference.get("conditioning_cache", True)
        or inference.get(
            "execution", {"profile": "synchronous_receding_horizon", "steps_per_replan": 1}
        )
        != {"profile": "synchronous_receding_horizon", "steps_per_replan": 1}
    ):
        raise ValueError("the exported policy uses five Euler steps and replans after each action")
    if (
        config.compute.device != "cuda"
        or config.compute.precision != "bf16"
        or any(config.compute.get(key, False) for key in ("compile", "distributed", "allow_tf32"))
    ):
        raise ValueError("FoveaWAM deployment requires single-GPU BF16 inference")


def validate_config(config: DictConfig) -> None:
    for key in ("data.root", "perception.vision.weights_path"):
        value = OmegaConf.select(config, key, throw_on_missing=True)
        if not isinstance(value, str) or not value:
            raise ValueError(f"provide a path for {key}")
    training = config.training
    if not training.future_dino_cache_root and not config.model.future_dino_teacher_weights_path:
        raise ValueError(
            "provide model.future_dino_teacher_weights_path or training.future_dino_cache_root"
        )
    _check_dimensions(config.model)
    if not 0 < training.max_epochs <= training.scheduler.total_epochs:
        raise ValueError("training epochs must fit inside the LR schedule")
    if (
        training.scheduler.warmup_optimizer_steps < 0
        or not 0 <= training.scheduler.min_learning_rate_ratio <= 1
    ):
        raise ValueError("invalid warmup or minimum learning rate")
    if (
        min(training.gradient_accumulation_steps, training.log_every_optimizer_steps) < 1
        or training.gradient_clip_norm <= 0
    ):
        raise ValueError("gradient accumulation, clipping and log intervals must be positive")
    if training.checkpoint_every_optimizer_steps < 0 or (
        training.stop_after_optimizer_steps is not None and training.stop_after_optimizer_steps < 1
    ):
        raise ValueError("invalid optimizer-step checkpoint or stopping interval")
    losses = {
        "action",
        "pose",
        "future_environment",
        "future_depth",
        "future_dino",
        "future_dynamic_rgb",
        "reconstruction",
    }
    if set(config.objective) != losses or any(
        not math.isfinite(group.weight) or group.weight <= 0 for group in config.objective.values()
    ):
        raise ValueError("FoveaWAM requires all seven training losses")
    repeats = config.objective.action.noise_repeats
    if (
        type(repeats) is not int
        or repeats < 1
        or config.objective.future_environment.visibility_weight <= 0
    ):
        raise ValueError(
            "use a positive integer for action draws and a positive scene visibility weight"
        )
