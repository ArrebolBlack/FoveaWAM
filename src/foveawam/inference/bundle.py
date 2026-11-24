"""Read and restore a self-contained policy bundle."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf
from torch import Tensor

from foveawam.inference.policy import FoveaWAMPolicy
from foveawam.models.model import _build_deployment_model, _FoveaWAMModel
from foveawam.perception.vision import _apply_frozen_mae_precision

_BUNDLE_KEYS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "package_version",
        "deployment_config",
        "normalization_state",
        "state_dicts",
    }
)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-key dictionary")
    return dict(value)


def _tensor_state(value: object, name: str) -> dict[str, Tensor]:
    mapping = _mapping(value, name)
    result: dict[str, Tensor] = {}
    for key, tensor in mapping.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name}.{key} must be a tensor")
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"{name}.{key} must be contiguous CPU")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"{name}.{key} floating state must be float32")
        if tensor.is_complex() or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name}.{key} must be finite real state")
        result[key] = tensor
    return result


def _validate_bundle_container(value: object) -> dict[str, object]:
    """Check the published bundle schema and the CPU tensors it contains."""
    payload = _mapping(value, "deployment bundle")
    if set(payload) != _BUNDLE_KEYS:
        raise ValueError("deployment bundle top-level keys differ from schema")
    if payload["artifact_type"] != "deployment_bundle":
        raise ValueError("artifact_type must be deployment_bundle")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported deployment bundle schema_version")
    if not isinstance(payload["package_version"], str) or not payload["package_version"]:
        raise ValueError("deployment bundle must record its producer package version")
    config = _mapping(payload["deployment_config"], "deployment_config")
    if set(config) != {"model", "perception", "objective", "inference", "compute"}:
        raise ValueError("deployment_config must contain exactly the five deployment groups")
    for group, fields in config.items():
        _mapping(fields, f"deployment_config.{group}")
    model = _mapping(config["model"], "model")
    if model.get("architecture_id") != "FoveaWAM":
        raise ValueError("Expected a FoveaWAM deployment checkpoint")
    action = _mapping(model.get("action_spec"), "model.action_spec")
    if (action.get("mode"), action.get("representation"), action.get("gripper_mode")) != (
        "E7",
        "delta",
        "discrete",
    ):
        raise ValueError("the simulation bundle requires discrete E7 delta actions")
    normalization = _mapping(payload["normalization_state"], "normalization_state")
    if set(normalization) != {"action_mean", "action_std"}:
        raise ValueError("deployment normalization contains only action_mean and action_std")
    for name, tensor in _tensor_state(normalization, "normalization_state").items():
        if tensor.dtype != torch.float32 or tensor.shape != (7,):
            raise ValueError(f"{name} must be float32[7]")
        if name == "action_std" and not bool((tensor > 0).all().item()):
            raise ValueError("action_std must be strictly positive")
    states = _mapping(payload["state_dicts"], "state_dicts")
    if set(states) != {"model", "vision", "depth", "flow"}:
        raise ValueError("deployment state_dicts keys differ from schema")
    for name in ("model", "vision"):
        if not _tensor_state(states[name], f"state_dicts.{name}"):
            raise ValueError(f"state_dicts.{name} is empty")
    if states["depth"] is not None or states["flow"] is not None:
        raise ValueError("the simulation bundle stores only model and MAE weights")
    return payload


def _read_bundle(bundle_path: str | os.PathLike[str]) -> dict[str, object]:
    path = Path(bundle_path).expanduser()
    if path.suffix != ".pt" or not path.is_file():
        raise ValueError(f"expected an existing FoveaWAM .pt checkpoint: {path}")
    value: object = torch.load(path, map_location="cpu", weights_only=True)
    return _validate_bundle_container(value)


def _write_bundle(bundle_path: str | os.PathLike[str], payload: dict[str, object]) -> None:
    """Atomically serialize a payload after the exporter has strictly checked the model."""
    _validate_bundle_container(payload)
    path = Path(bundle_path).expanduser()
    if path.suffix != ".pt":
        raise ValueError("deployment bundle must use the .pt suffix")
    if path.exists() and (not path.is_file() or path.is_symlink()):
        raise ValueError("bundle output must be an ordinary file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_policy(
    bundle_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cuda",
    compile_model: bool = False,
    gripper_deadband: float | None = None,
) -> FoveaWAMPolicy:
    """Restore a BF16 simulation policy on one CUDA device."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FoveaWAM simulation inference requires a CUDA device")
    if type(compile_model) is not bool:
        raise TypeError("compile_model must be a boolean")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{device} does not support BF16 inference")

        payload = _read_bundle(bundle_path)
        config = OmegaConf.create(cast(dict[str, object], payload["deployment_config"]))
        OmegaConf.set_readonly(config, True)
        states = cast(dict[str, dict[str, Tensor] | None], payload["state_dicts"])
        # The builder checks the architecture and loads both state dictionaries strictly.
        model = _build_deployment_model(config, states)
        if not isinstance(model, _FoveaWAMModel):
            raise TypeError("the bundle must contain a FoveaWAM simulation model")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        _apply_frozen_mae_precision(model, str(config.perception.vision.frozen_precision))
        stats = cast(dict[str, Tensor], payload["normalization_state"])
        return FoveaWAMPolicy(
            model,
            action_mean=stats["action_mean"],
            action_std=stats["action_std"],
            gripper_deadband=gripper_deadband,
            displacement_scale_m=float(config.model.future_displacement_scale_m),
            compile_model=compile_model,
        )


__all__ = ["load_policy"]
