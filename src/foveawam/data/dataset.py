"""Sliding observation histories and eight-step targets from a training manifest."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from foveawam.data import ontimebench as episode_reader
from foveawam.data.batch import TrainingBatch, collate_episode_chunks
from foveawam.data.future_targets import _read_future_targets, read_dino_episode_cache
from foveawam.data.sampler import EpisodeChunkBatchSampler
from foveawam.data.windowing import build_episode_chunk_window, episode_chunk_count
from foveawam.models.io import ActionSpec


class Episode(TypedDict):
    episode_id: str
    task_id: str
    path: Path
    frames: int
    valid_actions: int


def read_manifest(path: str | Path) -> list[Episode]:
    """Resolve episode paths relative to the public JSONL manifest."""
    manifest = Path(path).expanduser().resolve(strict=True)
    episodes = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != Episode.__required_keys__:
                    raise ValueError("expected episode_id, task_id, path, frames and valid_actions")
                if any(
                    not isinstance(row[key], str) or not row[key]
                    for key in ("episode_id", "task_id", "path")
                ):
                    raise ValueError("episode_id, task_id and path must be nonempty strings")
                frames, valid = row["frames"], row["valid_actions"]
                if (
                    type(frames) is not int
                    or frames < 2
                    or type(valid) is not int
                    or frames - 1 - valid not in (0, 1)
                ):
                    raise ValueError(
                        "expected a valid action prefix with at most one masked final action"
                    )
                if row["episode_id"] in seen:
                    raise ValueError(f"duplicate episode {row['episode_id']}")
                seen.add(row["episode_id"])
                episodes.append(
                    Episode(
                        episode_id=row["episode_id"],
                        task_id=row["task_id"],
                        path=(manifest.parent / row["path"]).resolve(),
                        frames=frames,
                        valid_actions=valid,
                    )
                )
            except ValueError as error:
                raise ValueError(f"{manifest}:{number}: {error}") from error
    if not episodes:
        raise ValueError(f"empty training manifest: {manifest}")
    return episodes


def _check_identity(row: Episode, info: Mapping[str, object], valid: Tensor) -> None:
    if (row["episode_id"], row["task_id"], row["frames"]) != (
        info["episode_id"],
        info["task_id"],
        info["frames"],
    ):
        raise ValueError(f"recording and manifest disagree for {row['episode_id']}")
    if not torch.equal(valid, torch.arange(row["frames"] - 1) < row["valid_actions"]):
        raise ValueError(
            f"recorded action validity differs from the manifest for {row['episode_id']}"
        )


def _scatter_rows(value: Tensor, mask: NDArray[np.bool_]) -> Tensor:
    if mask.all():
        return value.reshape(*mask.shape, *value.shape[1:])
    result = value.new_zeros((*mask.shape, *value.shape[1:]))
    result[torch.from_numpy(mask)] = value
    return result


class OnTimeBenchDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        action_spec: ActionSpec,
        history_length: int = 8,
        chunk_steps: int = 8,
        window_stride: int = 1,
        episode_ids: Sequence[str] = (),
        future_dino_cache_root: str | Path | None = None,
    ) -> None:
        episode_reader._validate_action_spec(action_spec)
        if (
            not 1 <= history_length <= 16
            or chunk_steps < 1
            or not 1 <= window_stride <= chunk_steps
        ):
            raise ValueError("expected a 1–16 frame history and a stride between 1 and chunk_steps")
        rows = read_manifest(manifest_path)
        if episode_ids:
            requested = set(episode_ids)
            if len(requested) != len(episode_ids) or requested.difference(
                row["episode_id"] for row in rows
            ):
                raise ValueError("episode_ids must enumerate unique episodes in the manifest")
            rows = [row for row in rows if row["episode_id"] in requested]
        self.entries = rows
        self.action_spec = action_spec
        self.history_length = history_length
        self.chunk_steps = chunk_steps
        self.window_stride = window_stride
        self.future_dino_cache_root = future_dino_cache_root
        self._chunk_ends: list[int] = []
        total = 0
        for row in rows:
            total += episode_chunk_count(row["valid_actions"], chunk_steps, window_stride)
            self._chunk_ends.append(total)
        self._checked_episodes: set[int] = set()

    @property
    def coverage(self) -> dict[str, int]:
        return {"manifest_episodes": len(self.entries), "episode_chunks": len(self)}

    def __len__(self) -> int:
        return self._chunk_ends[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self._chunk_ends, index)
        return episode, index - (self._chunk_ends[episode - 1] if episode else 0)

    def chunk_valid_prediction_count(self, index: int) -> int:
        episode, chunk = self._locate(index)
        return min(
            self.chunk_steps, self.entries[episode]["valid_actions"] - chunk * self.window_stride
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        episode, chunk = self._locate(index)
        row = self.entries[episode]
        window = build_episode_chunk_window(
            frames=row["frames"],
            prediction_count=row["valid_actions"],
            chunk_index=chunk,
            chunk_steps=self.chunk_steps,
            history_length=self.history_length,
            window_stride=self.window_stride,
        )
        current = window.prediction_indices[window.prediction_mask]
        with h5py.File(str(row["path"]), "r") as handle:
            metadata, frames = episode_reader._metadata(handle)
            if episode not in self._checked_episodes:
                info = episode_reader.index_episode(row["path"])
                _check_identity(
                    row, info, torch.from_numpy(episode_reader._dataset(handle, "action/valid")[:])
                )
                self._checked_episodes.add(episode)
            observation = episode_reader._frame_tensors(
                handle, window.frame_indices[window.frame_mask]
            )
            targets = episode_reader._object_poses(handle, current)
            targets.update(episode_reader._current_object_regions(handle, metadata, current))
            targets.update(_read_future_targets(handle, current, frames))
            actions, action_valid = episode_reader._read_actions(
                handle, window.action_indices[window.action_mask], self.action_spec
            )
        if self.future_dino_cache_root is not None:
            targets["future_dino"] = read_dino_episode_cache(
                self.future_dino_cache_root,
                row["episode_id"],
                row["path"],
                current,
                expected_frames=frames,
            )
        return {
            **{
                name: _scatter_rows(value, window.frame_mask) for name, value in observation.items()
            },
            **{
                name: _scatter_rows(value, window.prediction_mask)
                for name, value in targets.items()
            },
            "action_physical": _scatter_rows(actions, window.action_mask),
            "action_valid": _scatter_rows(action_valid, window.action_mask),
            "frame_mask": torch.from_numpy(window.frame_mask),
            "prediction_mask": torch.from_numpy(window.prediction_mask),
            "future_record_valid": torch.from_numpy(window.future_mask),
            "prediction_frame_index": torch.from_numpy(window.prediction_indices),
            "view_names": ("agentview",),
            "episode_id": row["episode_id"],
            "task_id": row["task_id"],
            "episode_seed": metadata["episode_seed"],
        }


def _iter_unique_issued_actions(dataset: OnTimeBenchDataset) -> Iterator[tuple[Tensor, Tensor]]:
    """Give each recorded command equal weight despite overlapping training windows."""
    for row in dataset.entries:
        info, action, valid = episode_reader._read_action_episode(row["path"], dataset.action_spec)
        _check_identity(row, info, valid)
        yield action, valid


def build_episode_chunk_loader(
    dataset: OnTimeBenchDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = True,
    length_bucket_size: int = 0,
    accumulation_steps: int = 1,
) -> DataLoader[TrainingBatch]:
    """Partition chunks deterministically across ranks and accumulation steps."""
    if batch_size < 1 or num_workers < 0 or prefetch_factor < 1:
        raise ValueError("invalid DataLoader batch/worker/prefetch setting")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    sampler = EpisodeChunkBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
        drop_last=drop_last,
        length_bucket_size=length_bucket_size,
        accumulation_steps=accumulation_steps,
    )
    return cast(
        DataLoader[TrainingBatch],
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers else None,
            multiprocessing_context="spawn" if num_workers else None,
            collate_fn=collate_episode_chunks,
            generator=torch.Generator().manual_seed(seed),
        ),
    )
