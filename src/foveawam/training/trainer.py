"""Optimize the seven FoveaWAM losses across episode chunks and ranks."""

import json
import math
from contextlib import AbstractContextManager, nullcontext
from itertools import islice
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from foveawam.data.batch import TrainingBatch
from foveawam.data.normalization import ActionNormalization
from foveawam.data.sampler import EpisodeChunkBatchSampler
from foveawam.models.common.numeric_checks import _TrainingNumericChecks
from foveawam.training.checkpoint import resume_checkpoint, save_checkpoint
from foveawam.training.distributed import sum_across_ranks
from foveawam.training.optimization import build_optimization
from foveawam.training.step import compute_training_step, training_objective_counts


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        loader: DataLoader[TrainingBatch],
        normalization: ActionNormalization | None,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.loader = loader
        self.sampler = cast(EpisodeChunkBatchSampler, loader.batch_sampler)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.accumulation = config.training.gradient_accumulation_steps
        self.optimizer, self.scheduler = build_optimization(
            config, model, updates_per_epoch=math.ceil(len(loader) / self.accumulation)
        )
        self.epoch = 0
        self.offset = 0
        self.step = 0
        self.stop_requested = False
        if config.training.resume_from is not None:
            self.epoch, self.offset, self.step, normalization = resume_checkpoint(
                config.training.resume_from,
                config=config,
                model=model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch_size=self.sampler.epoch_size,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        if normalization is None:
            raise ValueError("training needs fitted action statistics or a resumable checkpoint")
        self.normalization = normalization

    def _update(self, chunks: list[TrainingBatch]) -> tuple[tuple[str, ...], Tensor]:
        # The heads count different things: actions, poses, pixels or DINO patches.
        # Sum their denominators over the whole optimizer batch before any backward pass.
        local = [training_objective_counts(chunk) for chunk in chunks]
        names = tuple(local[0])
        counts = torch.stack([torch.stack([row[name] for name in names]) for row in local])
        counts = sum_across_ranks(counts.to(self.device))
        if not (counts[:, :3].sum(1) > 0).all():
            raise ValueError("an optimizer microbatch has no action, pose or region targets")
        denominator = counts.sum(0).clamp_min(1).float()
        local_losses = torch.zeros_like(denominator)
        self.optimizer.zero_grad(set_to_none=True)
        checks = _TrainingNumericChecks()
        for i, chunk in enumerate(chunks):
            sync: AbstractContextManager[None] = nullcontext()
            if isinstance(self.model, DistributedDataParallel) and i + 1 != len(chunks):
                sync = self.model.no_sync()
            with sync, checks.collect(enabled=self.world_size > 1):
                with torch.autocast(self.device.type, dtype=torch.bfloat16):
                    losses = compute_training_step(
                        self.model,
                        chunk.to(self.device, non_blocking=True),
                        self.normalization,
                        self.config,
                    )
                    total = losses["action"][0] * 0
                    for index, name in enumerate(names):
                        loss, count = losses[name]
                        fraction = count.float() / denominator[index]
                        local_losses[index] += loss.detach() * fraction
                        total = total + self.config.objective[name].weight * loss * fraction
                    # DDP averages gradients across ranks.
                    total = total * self.world_size
                torch.autograd.backward(total)

        checks.check()
        parameters = list(self.model.parameters())
        # DDP bucket views can differ in stride from single-GPU gradients.
        grads = [
            p.grad.detach().clone(memory_format=torch.contiguous_format)
            for p in parameters
            if p.grad is not None
        ]
        norm = torch.nn.utils.get_total_norm(grads, error_if_nonfinite=True)
        del grads
        torch.nn.utils.clip_grads_with_norm_(
            parameters, self.config.training.gradient_clip_norm, norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return names, local_losses

    def _save(self) -> None:
        save_checkpoint(
            Path(self.config.training.output_dir) / "latest.pt",
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            normalization=self.normalization,
            epoch=self.epoch,
            epoch_offset=self.offset,
            optimizer_step=self.step,
            epoch_size=self.sampler.epoch_size,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
        )

    def fit(self) -> None:
        options = self.config.training
        self.model.train()
        if (
            options.stop_after_optimizer_steps is not None
            and self.step >= options.stop_after_optimizer_steps
        ):
            return
        while self.epoch < options.max_epochs:
            self.sampler.set_epoch(self.epoch)
            if self.offset:
                self.sampler.resume_epoch(
                    start_index=self.offset,
                    epoch_size=self.sampler.epoch_size,
                    length_bucket_size=self.sampler.length_bucket_size,
                )
            batches = iter(self.loader)
            while chunks := list(islice(batches, self.accumulation)):
                lr = self.optimizer.param_groups[0]["lr"]
                names, losses = self._update(chunks)
                stopped = self.stop_requested or self.step == options.stop_after_optimizer_steps
                local_size = sum(batch.rgb.shape[0] for batch in chunks)
                size, stopped = sum_across_ranks(
                    torch.tensor([local_size, int(stopped)], device=self.device)
                ).tolist()
                self.offset += size
                epoch_done = self.offset == self.sampler.epoch_size
                if self.step % options.log_every_optimizer_steps == 0 or stopped or epoch_done:
                    values = sum_across_ranks(losses).tolist()
                    if self.rank == 0:
                        print(
                            json.dumps(
                                {
                                    "epoch": self.epoch + 1,
                                    "step": self.step,
                                    "lr": lr,
                                    **dict(zip(names, values, strict=True)),
                                }
                            ),
                            flush=True,
                        )
                if epoch_done:
                    self.epoch += 1
                    self.offset = 0
                cadence = options.checkpoint_every_optimizer_steps
                if stopped or epoch_done or (cadence and self.step % cadence == 0):
                    self._save()
                if stopped:
                    return
                if epoch_done:
                    break
            else:
                raise RuntimeError("loader ended before consuming every sampled chunk")
