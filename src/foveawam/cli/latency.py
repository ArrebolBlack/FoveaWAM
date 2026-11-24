"""Time a local batch-one query from a CPU observation to eight simulator commands."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from foveawam.api import load_policy
from foveawam.cli.infer import _observation, _read_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True, help="OnTimeBench episode (.hdf5)")
    parser.add_argument(
        "--output", type=Path, required=True, help="OnTimeBench latency profile (.json)"
    )
    parser.add_argument("--start", type=int, default=0, help="first observation tick of the window")
    parser.add_argument("--warmup", type=int, default=30, help="untimed queries")
    parser.add_argument("--calls", type=int, default=300, help="measured queries")
    parser.add_argument("--seed", type=int, default=20261103, help="seed for action noise")
    parser.add_argument("--eager", action="store_true", help="measure eager BF16 execution")
    args = parser.parse_args()
    if args.warmup < 0 or args.calls <= 0:
        parser.error("--warmup must be nonnegative and --calls must be positive")

    from foveawam.evaluation.ontimebench import benchmark_action, benchmark_action_spec

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    policy = load_policy(args.bundle, compile_model=not args.eager)
    task_id, frames = _read_window(args.episode, args.start, policy.history_length)
    spec = benchmark_action_spec(policy.action_spec)
    torch.manual_seed(args.seed)
    samples = []
    for call in range(args.warmup + args.calls):
        policy.reset(task_id)
        for tick in range(policy.history_length - 1):
            policy.observe(_observation(frames, tick))
        torch.cuda.synchronize()
        start = perf_counter()
        chunk = policy.predict_action_chunk(_observation(frames, -1))
        for action in chunk.actions:
            benchmark_action(action.contiguous(), spec)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        if call >= args.warmup:
            samples.append(elapsed)

    args.output.write_text(
        json.dumps({"latency_samples_s": samples, "seed": 0}, indent=2) + "\n", encoding="utf-8"
    )
    p50, p95 = 1000 * np.quantile(samples, [0.5, 0.95])
    mode = "eager" if args.eager else "compiled"
    print(f"{torch.cuda.get_device_name()} | BF16 {mode} | {len(samples)} queries")
    print(f"Local query latency in milliseconds: p50 {p50:.2f}, p95 {p95:.2f}")


if __name__ == "__main__":
    main()
