"""Export a trained FoveaWAM policy for OnTimeBench."""

import argparse
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from foveawam import __version__
from foveawam.config.validate import _validate_deployment_config
from foveawam.inference.bundle import _write_bundle
from foveawam.models.model import _build_deployment_model
from foveawam.training.checkpoint import _read_training_checkpoint


def _deployment_config(saved: dict[str, Any]) -> DictConfig:
    model = dict(saved["model"])
    model.pop("future_dino_teacher_weights_path")
    model["architecture_id"] = "FoveaWAM"
    model["action_spec"] = {
        "mode": "E7",
        "representation": "delta",
        "gripper_mode": "discrete",
        "motion_limits": [0.10] * 3 + [0.50] * 3,
    }
    config = OmegaConf.create(
        {
            "model": model,
            "perception": {
                "vision": {
                    "identity": "mae_vitb16",
                    "input_size_px": 224,
                    "trainable": False,
                    "frozen_precision": "bf16",
                },
                "depth": {"source": "simulator_exact"},
                "flow": {"source": "disabled"},
            },
            "inference": {"solver": {"method": "euler", "nfe": 5}},
            "compute": {"device": "cuda", "precision": "bf16"},
            "objective": {},
        }
    )
    OmegaConf.set_readonly(config, True)
    _validate_deployment_config(config)
    return config


def _export_checkpoint(checkpoint_path: Path, bundle_path: Path) -> None:
    if checkpoint_path.resolve() == bundle_path.resolve():
        raise ValueError("choose an output path distinct from the training checkpoint")
    checkpoint = _read_training_checkpoint(checkpoint_path)
    config = _deployment_config(checkpoint["config"])
    model: dict[str, Tensor] = {}
    vision: dict[str, Tensor] = {}
    for name, tensor in checkpoint["model_state"].items():
        if name.startswith("perception_frontend.vision_encoder."):
            # The public bundle stores the frozen BF16 backbone in FP32 containers.
            vision[name.removeprefix("perception_frontend.vision_encoder.")] = (
                tensor.float() if tensor.is_floating_point() else tensor
            )
        elif not name.startswith("future_dino_teacher."):
            model[name] = tensor
    states: dict[str, dict[str, Tensor] | None] = {
        "model": model,
        "vision": vision,
        "depth": None,
        "flow": None,
    }
    normalization = checkpoint["normalization_state"]
    _build_deployment_model(config, states)
    _write_bundle(
        bundle_path,
        {
            "artifact_type": "deployment_bundle",
            "schema_version": 1,
            "package_version": __version__,
            "deployment_config": OmegaConf.to_container(config),
            "normalization_state": {
                "action_mean": normalization["action_mean"],
                "action_std": normalization["action_std"],
            },
            "state_dicts": states,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
