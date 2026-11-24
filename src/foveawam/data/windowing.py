"""Frame and transition indices for one dense OnTimeBench training window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from foveawam.data.ontimebench import _HORIZON


@dataclass(frozen=True, slots=True)
class _EpisodeChunkWindow:
    frame_indices: NDArray[np.int64]
    frame_mask: NDArray[np.bool_]
    prediction_indices: NDArray[np.int64]
    prediction_mask: NDArray[np.bool_]
    future_mask: NDArray[np.bool_]
    action_indices: NDArray[np.int64]
    action_mask: NDArray[np.bool_]


def episode_chunk_count(predictions: int, chunk_steps: int, window_stride: int) -> int:
    if predictions < 0 or not 1 <= window_stride <= chunk_steps:
        raise ValueError("invalid episode length, chunk size or stride")
    return (predictions + window_stride - 1) // window_stride


def build_episode_chunk_window(
    *,
    frames: int,
    prediction_count: int,
    chunk_index: int,
    history_length: int,
    chunk_steps: int,
    window_stride: int,
) -> _EpisodeChunkWindow:
    """Left-pad observation history and mask targets beyond the recorded episode."""
    if history_length < 1 or not 0 <= prediction_count < frames:
        raise ValueError("invalid history length or action prefix")
    if not 0 <= chunk_index < episode_chunk_count(prediction_count, chunk_steps, window_stride):
        raise IndexError(chunk_index)
    start = chunk_index * window_stride
    current = start + np.arange(chunk_steps, dtype=np.int64)
    prediction_mask = current < prediction_count
    history = np.arange(start - history_length + 1, start + chunk_steps, dtype=np.int64)
    frame_mask = (history >= 0) & (history < prediction_count)
    actions = current[:, None] + np.arange(_HORIZON, dtype=np.int64)
    # Command h is issued at t+h and predicts the transition to frame t+h+1.
    future_mask = prediction_mask[:, None] & (actions + 1 < frames)
    return _EpisodeChunkWindow(
        frame_indices=np.where(frame_mask, history, 0),
        frame_mask=frame_mask,
        prediction_indices=np.where(prediction_mask, current, 0),
        prediction_mask=prediction_mask,
        future_mask=future_mask,
        action_indices=np.where(future_mask, actions, 0),
        action_mask=future_mask,
    )
