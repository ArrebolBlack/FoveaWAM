"""Save an optimizer boundary and resume the same shuffled epoch on every rank."""

import os
import random
import tempfile
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from foveawam.data.normalization import ActionNormalization
from foveawam.models.model import _FoveaWAMModel
from foveawam.training.distributed import rank_zero_operation


class _RankRNG(TypedDict):
    python: tuple[object, ...]
    cpu: Tensor
    cuda: Tensor


class _Statistics(TypedDict):
    action_mean: Tensor
    action_std: Tensor
    action_sample_count: int


class _TrainingCheckpoint(TypedDict):
    config: dict[str, Any]
    epoch: int
    epoch_offset: int
    epoch_size: int
    optimizer_step: int
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    normalization_state: _Statistics
    rank_rng: list[_RankRNG]


def _resume_settings(config: DictConfig | dict[str, Any]) -> DictConfig:
    """Training inputs that determine sample order, gradients or the LR curve."""
    cfg = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    )
    assert isinstance(cfg, DictConfig)
    for name in ("root", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor"):
        del cfg.data[name]
    del cfg.perception.vision.weights_path
    del cfg.model.future_dino_teacher_weights_path
    cfg.training.future_dino_cache_root = cfg.training.future_dino_cache_root is not None
    for name in (
        "output_dir",
        "resume_from",
        "stop_after_optimizer_steps",
        "checkpoint_every_optimizer_steps",
        "log_every_optimizer_steps",
    ):
        del cfg.training[name]
    return cfg


def _rng_state(device: torch.device) -> _RankRNG:
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def save_checkpoint(
    path: Path,
    *,
    config: DictConfig,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    normalization: ActionNormalization,
    epoch: int,
    epoch_offset: int,
    optimizer_step: int,
    epoch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    local_rng = _rng_state(device)
    if dist.is_initialized():
        rngs: list[object] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_rng, rngs, dst=0)
    else:
        rngs = [local_rng]
    with rank_zero_operation(rank, device):
        if rank != 0:
            return
        core = model.module if isinstance(model, DistributedDataParallel) else model
        payload = {
            "artifact_type": "foveawam_training",
            "config": OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            "epoch": epoch,
            "epoch_offset": epoch_offset,
            "epoch_size": epoch_size,
            "optimizer_step": optimizer_step,
            "model_state": core.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "normalization_state": {
                "action_mean": normalization.action_mean,
                "action_std": normalization.action_std,
                "action_sample_count": normalization.action_sample_count,
            },
            "rank_rng": rngs,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stage:
            temporary = Path(stage.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_training_checkpoint(path: str | Path) -> _TrainingCheckpoint:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("artifact_type") != "foveawam_training":
        raise ValueError("expected a FoveaWAM training checkpoint")
    return cast(_TrainingCheckpoint, checkpoint)


def resume_checkpoint(
    path: str | Path,
    *,
    config: DictConfig,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[int, int, int, ActionNormalization]:
    checkpoint = _read_training_checkpoint(path)
    if _resume_settings(checkpoint["config"]) != _resume_settings(config):
        raise ValueError("training settings differ from the checkpoint")
    if epoch_size != checkpoint["epoch_size"] or len(checkpoint["rank_rng"]) != world_size:
        raise ValueError("resuming requires the same dataset size and number of ranks")
    core = cast(
        _FoveaWAMModel, model.module if isinstance(model, DistributedDataParallel) else model
    )
    core.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    stats = checkpoint["normalization_state"]
    normalization = ActionNormalization(
        action_spec=core.action_spec,
        action_mean=stats["action_mean"].clone(),
        action_std=stats["action_std"].clone(),
        action_sample_count=stats["action_sample_count"],
    )
    rng = checkpoint["rank_rng"][rank]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"], device)
    return (
        checkpoint["epoch"],
        checkpoint["epoch_offset"],
        checkpoint["optimizer_step"],
        normalization,
    )
