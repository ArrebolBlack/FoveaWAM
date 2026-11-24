"""Cache FP32 DINOv3 patch targets for the OnTimeBench training demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import h5py
import numpy as np
import torch
from torch import nn

from foveawam.data.dataset import Episode, read_manifest
from foveawam.data.future_targets import (
    DINO_REPRESENTATION,
    _validate_cached_features,
    dino_cache_path,
)
from foveawam.data.ontimebench import _dataset, _metadata
from foveawam.perception.future_targets import build_future_dino_teacher


@contextmanager
def _open_index(output: Path) -> Iterator[BinaryIO]:
    """Keep one writer per cache. Closing the index releases the OS lock after a crash."""
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / "index.json", os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0))
    with os.fdopen(descriptor, "r+b") as stream:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError) as error:
            raise RuntimeError(f"another writer is building this cache: {output}") from error
        yield stream


def _write_index(stream: BinaryIO, index: dict[str, object]) -> None:
    payload = (json.dumps(index, indent=2) + "\n").encode()
    length = stream.seek(0, os.SEEK_END)
    stream.seek(0)
    stream.write(payload.ljust(length))
    stream.flush()
    os.fsync(stream.fileno())


@torch.no_grad()
def _cache_episode(
    encoder: nn.Module,
    row: Episode,
    target: Path,
    device: torch.device,
    batch_size: int,
    compute: dict[str, object],
) -> None:
    source = row["path"]
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".h5.partial")
    with h5py.File(str(source), "r") as native:
        metadata, frames = _metadata(native)
        if (
            metadata["episode_id"] != row["episode_id"]
            or metadata["task_id"] != row["task_id"]
            or frames != row["frames"]
        ):
            raise ValueError(f"manifest and HDF5 identity disagree for {source}")
        rgb = _dataset(native, "policy/rgb")
        if rgb.shape != (frames, 1, 224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"expected uint8 agentview RGB in {source}")
        with h5py.File(str(pending), "w") as cache:
            attributes = {
                "episode_id": row["episode_id"],
                "source_bytes": before.st_size,
                "source_mtime_ns": before.st_mtime_ns,
                "representation": DINO_REPRESENTATION,
                "batch_size": batch_size,
                **{"teacher_" + key: value for key, value in compute.items()},
            }
            for key, attribute in attributes.items():
                cache.attrs[key] = attribute
            features = cache.create_dataset(
                "features", (frames, 196, 1024), dtype="f4", chunks=(1, 196, 1024)
            )
            for start in range(0, frames, batch_size):
                images = (
                    torch.from_numpy(np.asarray(rgb[start : start + batch_size, 0]))
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .to(device)
                )
                with torch.autocast(device_type=device.type, enabled=False):
                    value = encoder(images)
                if (
                    value.dtype != torch.float32
                    or value.shape != (len(images), 196, 1024)
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        "the DINOv3 teacher must return finite FP32 [frames,196,1024] patches"
                    )
                features[start : start + len(images)] = value.cpu().numpy()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"source changed while caching {source}")
            cache.attrs["complete"] = 1
    pending.replace(target)


def cache_manifests(
    manifests: list[Path], weights: Path, output: Path, *, device: torch.device, batch_size: int
) -> None:
    if not manifests or batch_size < 1:
        raise ValueError("provide at least one manifest and a positive batch size")
    weights = weights.resolve(strict=True)
    episodes: dict[Path, Episode] = {}
    for manifest in manifests:
        for row in read_manifest(manifest):
            target = dino_cache_path(output, row["episode_id"])
            if target in episodes:
                raise ValueError(f"duplicate episode in manifests: {row['episode_id']}")
            episodes[target] = row

    compute = dict(
        precision="fp32",
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )
    stat = weights.stat()
    identity = {
        "representation": DINO_REPRESENTATION,
        "teacher_weights_bytes": stat.st_size,
        "teacher_weights_mtime_ns": stat.st_mtime_ns,
        "teacher_compute": compute,
        "episode_count": len(episodes),
    }
    index = {**identity, "teacher_weights": weights.name}
    with _open_index(output) as stream:
        previous = stream.read()
        complete = False
        if previous:
            saved = json.loads(previous)
            complete = saved["complete"]
            if type(complete) is not bool or any(
                saved.get(key) != value for key, value in identity.items()
            ):
                raise ValueError(
                    "cache belongs to a different teacher, compute policy, or episode count"
                )
        elif any(output.rglob("*.h5")):
            raise ValueError("cached episodes lack their teacher index")
        else:
            _write_index(stream, {**index, "complete": False})
        if any(path not in episodes for path in output.rglob("*.h5")):
            raise ValueError("cache contains episodes outside the requested manifests")
        encoder = None
        for number, (target, row) in enumerate(episodes.items(), 1):
            if target.is_file():
                with h5py.File(str(target), "r") as cached:
                    _validate_cached_features(cached, row["episode_id"], row["path"], row["frames"])
                    if any(
                        cached.attrs.get("teacher_" + key) != value
                        for key, value in compute.items()
                    ):
                        raise ValueError(f"teacher compute policy differs for {target}")
                continue
            if complete:
                raise ValueError(f"completed cache is missing {target}")
            if encoder is None:
                encoder = build_future_dino_teacher(str(weights)).to(device)
                if encoder.training or any(
                    p.requires_grad or (p.is_floating_point() and p.dtype != torch.float32)
                    for p in encoder.parameters()
                ):
                    raise ValueError("DINOv3 caching requires a frozen FP32 teacher in eval mode")
            _cache_episode(encoder, row, target, device, batch_size, compute)
            print(f"[{number}/{len(episodes)}] {row['episode_id']}", flush=True)
        if not complete:
            after = weights.stat()
            if (after.st_size, after.st_mtime_ns) != (
                stat.st_size,
                stat.st_mtime_ns,
            ):
                raise ValueError("teacher weights changed during precomputation")
            _write_index(stream, {**index, "complete": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True, help="DINOv3 ViT-L/16 safetensors")
    parser.add_argument(
        "--output", type=Path, required=True, help="cache directory to create or resume"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="frames per teacher call")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cache_manifests(
        args.manifests,
        args.weights,
        args.output,
        device=torch.device(args.device),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
