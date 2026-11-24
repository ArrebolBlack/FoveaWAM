"""Build a shuffled OnTimeBench loader and fit action statistics on the training split."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from foveawam.data.batch import TrainingBatch
from foveawam.data.dataset import (
    OnTimeBenchDataset,
    _iter_unique_issued_actions,
    build_episode_chunk_loader,
)
from foveawam.data.normalization import ActionNormalization, _fit_action_normalization_stream
from foveawam.models.io import ActionSpec
from foveawam.training.distributed import rank_zero_operation


def build_training_loader(
    config: DictConfig, *, rank: int, world_size: int
) -> tuple[DataLoader[TrainingBatch], ActionSpec]:
    data = config.data
    action_spec = ActionSpec(
        mode="E7",
        motion_limits=tuple(config.model.action_spec.motion_limits),
    )
    cache = config.training.future_dino_cache_root
    if cache is not None:
        info = json.loads((Path(cache) / "index.json").read_text(encoding="utf-8"))
        teacher = config.model.future_dino_teacher_weights_path
        if teacher is not None:
            source = Path(teacher).stat()
            if (source.st_size, source.st_mtime_ns) != (
                info["teacher_weights_bytes"],
                info["teacher_weights_mtime_ns"],
            ):
                raise ValueError("DINO cache was produced with different teacher weights")
        expected_compute = {
            "precision": "fp32",
            "matmul_tf32": config.compute.allow_tf32,
            "cudnn_tf32": config.compute.allow_tf32,
            "deterministic": config.compute.deterministic,
        }
        if (
            info["complete"] is not True
            or info["representation"] != "raw_dinov3_vitl16_fp32_196x1024"
            or info["teacher_compute"] != expected_compute
        ):
            raise ValueError("DINO cache and training teacher or compute settings differ")
    dataset = OnTimeBenchDataset(
        Path(data.root) / data.manifest,
        action_spec=action_spec,
        history_length=config.model.history_length,
        chunk_steps=data.chunk_steps,
        window_stride=data.window_stride,
        episode_ids=tuple(data.episode_ids),
        future_dino_cache_root=cache,
    )
    if len(dataset) < world_size or (data.episode_batch_size == 1 and len(dataset) % world_size):
        raise ValueError("dataset chunks cannot form a non-empty final batch on every rank")
    loader = build_episode_chunk_loader(
        dataset,
        batch_size=data.episode_batch_size,
        shuffle=True,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
        num_workers=data.num_workers,
        pin_memory=data.pin_memory,
        persistent_workers=data.persistent_workers and data.num_workers > 0,
        prefetch_factor=data.prefetch_factor,
        drop_last=False,
        length_bucket_size=data.episode_batch_size * world_size,
    )
    return loader, action_spec


def fit_normalization(
    dataset: OnTimeBenchDataset, action_spec: ActionSpec, rank: int, device: torch.device
) -> ActionNormalization:
    with rank_zero_operation(rank, device):
        if rank == 0:
            fit = _fit_action_normalization_stream(
                action_spec, _iter_unique_issued_actions(dataset)
            )
            stats = torch.stack((fit.action_mean, fit.action_std)).to(device)
            count = torch.tensor(fit.action_sample_count, dtype=torch.int64, device=device)
        else:
            stats = torch.empty((2, action_spec.dimension), dtype=torch.float32, device=device)
            count = torch.empty((), dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.broadcast(stats, src=0)
        dist.broadcast(count, src=0)
    return ActionNormalization(
        action_spec=action_spec,
        action_mean=stats[0].cpu(),
        action_std=stats[1].cpu(),
        action_sample_count=int(count.item()),
    )
