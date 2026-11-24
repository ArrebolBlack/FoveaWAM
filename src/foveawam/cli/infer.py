"""Predict an eight-step action chunk from an OnTimeBench episode."""

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from foveawam.api import PolicyObservation, load_policy
from foveawam.data.ontimebench import read_frames


def _read_window(episode: Path, start: int, length: int) -> tuple[str, dict[str, Tensor]]:
    if start < 0:
        raise ValueError("the start tick must be nonnegative")
    frames = read_frames(episode, np.arange(start, start + length, dtype=np.int64))
    names = {
        "timestamps_s": "timestamp_s",
        "anchor_position_m": "target_position_m",
        "anchor_quaternion_xyzw": "target_quaternion_xyzw",
        "anchor_valid": "target_valid",
    }
    tensors = {
        names.get(name, name): value for name, value in frames.items() if isinstance(value, Tensor)
    }
    return cast(str, frames["task_id"]), tensors


def _observation(frames: dict[str, Tensor], tick: int) -> PolicyObservation:
    return PolicyObservation(
        view_names=("agentview",),
        **{name: value[tick].clone().contiguous() for name, value in frames.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="simulation checkpoint (.pt)")
    parser.add_argument(
        "--episode", type=Path, required=True, help="OnTimeBench training episode (.hdf5)"
    )
    parser.add_argument("--output", type=Path, required=True, help="action chunk (.npz)")
    parser.add_argument(
        "--start", type=int, default=0, help="first observation tick of the eight-tick window"
    )
    parser.add_argument(
        "--seed", type=int, default=20261023, help="seed for the flow-matching action noise"
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the BF16 CUDA policy (default: enabled)",
    )
    args = parser.parse_args()

    policy = load_policy(args.bundle, compile_model=args.compile_model)
    task_id, frames = _read_window(args.episode, args.start, policy.history_length)
    policy.reset(task_id)
    torch.manual_seed(args.seed)
    for tick in range(policy.history_length - 1):
        policy.observe(_observation(frames, tick))
    actions = policy.predict_action_chunk(_observation(frames, -1)).actions.numpy()
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, actions_e7=actions)
    print(f"Saved actions_e7 {actions.shape} to {args.output}")


if __name__ == "__main__":
    main()
