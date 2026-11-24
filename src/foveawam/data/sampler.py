"""Episode-local chunk batching with deterministic global DDP ordering."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import torch
from torch.utils.data import Sampler


class _ChunkDataset(Protocol):
    def __len__(self) -> int: ...

    def chunk_valid_prediction_count(self, index: int) -> int: ...


class EpisodeChunkBatchSampler(Sampler[list[int]]):
    """Order fixed episode chunks, then shard one deterministic order by rank."""

    def __init__(
        self,
        dataset: _ChunkDataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
        length_bucket_size: int = 0,
        accumulation_steps: int = 1,
    ) -> None:
        size = len(dataset)
        if size < 1:
            raise ValueError("EpisodeChunkBatchSampler requires a non-empty dataset")
        values = (batch_size, seed, rank, world_size, length_bucket_size, accumulation_steps)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("sampler integer settings must be integers, not bool")
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid batch size or DDP rank/world size")
        if seed < 0 or length_bucket_size < 0:
            raise ValueError("seed and length_bucket_size must be non-negative")
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self._update_batch_size = batch_size * world_size * accumulation_steps
        self.length_bucket_size = length_bucket_size
        self._configured_length_bucket_size = length_bucket_size
        self.start_index = 0
        self._epoch_size: int | None = None
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch
        self.start_index = 0
        self._epoch_size = None
        self.length_bucket_size = self._configured_length_bucket_size

    @property
    def epoch_size(self) -> int:
        if self._epoch_size is not None:
            return self._epoch_size
        size = len(self.dataset)
        return size - size % self._update_batch_size if self.drop_last else size

    def resume_epoch(self, *, start_index: int, epoch_size: int, length_bucket_size: int) -> None:
        """Consume the saved global suffix, preserving this epoch's original ordering."""
        if any(type(value) is not int for value in (start_index, epoch_size, length_bucket_size)):
            raise TypeError("sampler resume positions must be integers")
        if not 0 <= start_index <= epoch_size <= len(self.dataset) or length_bucket_size < 0:
            raise ValueError("invalid sampler resume position")
        remaining = epoch_size - start_index
        if remaining and (
            remaining < self.world_size or self.batch_size == 1 and remaining % self.world_size
        ):
            raise ValueError(
                "remaining training chunks cannot form nonempty rank batches; use fewer ranks or a larger per-rank batch"
            )
        self.start_index = start_index
        self._epoch_size = epoch_size
        self.length_bucket_size = length_bucket_size

    def _global_order(self) -> list[int]:
        size = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(size, generator=generator).tolist()
        else:
            order = list(range(size))
        if self.length_bucket_size:
            bucket = self.length_bucket_size
            order = [
                index
                for start in range(0, size, bucket)
                for index in sorted(
                    order[start : start + bucket],
                    key=self.dataset.chunk_valid_prediction_count,
                    reverse=True,
                )
            ]
        return order

    def _rank_order(self) -> list[int]:
        order = self._global_order()[: self.epoch_size][self.start_index :]
        if self.drop_last and self._epoch_size is None:
            global_batch = self.batch_size * self.world_size
            order = order[: len(order) - len(order) % global_batch]
        return order[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        if self.world_size > 1 and (
            self.start_index
            or not self.drop_last
            and self.batch_size > 1
            and self.epoch_size >= self.world_size
        ):
            # Rebalance only the final two global batches when sharding leaves
            # fewer than world_size chunks in the tail. Never pad or repeat data.
            order = self._global_order()[self.start_index : self.epoch_size]
            start = 0
            while start < len(order):
                remaining = len(order) - start
                count = min(self.batch_size * self.world_size, remaining)
                tail = remaining - count
                if 0 < tail < self.world_size:
                    count -= self.world_size - tail
                yield order[start : start + count][self.rank :: self.world_size]
                start += count
            return
        order = self._rank_order()
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last or self._epoch_size is not None:
                yield batch

    def __len__(self) -> int:
        size = self.epoch_size - self.start_index
        if self.start_index or (
            self.world_size > 1
            and not self.drop_last
            and self.batch_size > 1
            and size >= self.world_size
        ):
            return (size + self.batch_size * self.world_size - 1) // (
                self.batch_size * self.world_size
            )
        if self.drop_last:
            per_rank = (size // (self.batch_size * self.world_size)) * self.batch_size
            return per_rank // self.batch_size
        per_rank = len(range(self.rank, size, self.world_size))
        return (per_rank + self.batch_size - 1) // self.batch_size
