"""Evaluate the released FoveaWAM BF16 model on OnTimeBench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ontimebench.api import PROTOCOLS, check_output_path, evaluate, write_report

_INFERENCE_SEED = 20261023


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m foveawam.cli.evaluate", description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="native")
    parser.add_argument(
        "--latency-profile", type=Path, help="batch-one latency JSON (default: RTX 5090 reference)"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--no-compile",
        dest="compile_model",
        action="store_false",
        help="use eager BF16 inference",
    )
    args = parser.parse_args(argv)
    check_output_path(args.output)

    import torch

    from foveawam.api import load_policy
    from foveawam.evaluation.ontimebench import bind_foveawam

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    model = load_policy(args.bundle, compile_model=args.compile_model)
    profile = args.latency_profile or (
        Path(__file__).resolve().parents[1] / "configs" / "latency_rtx5090.json"
    )

    def seed_model(index: int) -> None:
        # The published policy uses the scene's index before sharding.
        torch.manual_seed(_INFERENCE_SEED + index)

    summary = evaluate(
        bind_foveawam(model),
        protocol=args.protocol,
        target_pose_source="simulator",
        latency_profile=profile,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        policy_settings={
            "model": "FoveaWAM",
            "bundle": args.bundle.name,
            "compile_model": args.compile_model,
            "latency_profile": profile.name if args.latency_profile else profile.stem,
            "latency_profile_source": "user" if args.latency_profile else "bundled",
            "gripper_deadband": model.default_gripper_deadband,
            "gripper_deadband_by_task": dict(model.gripper_deadband_by_task),
            "inference_seed": {"base": _INFERENCE_SEED, "offset": "test_id_manifest_index"},
        },
        before_episode=seed_model,
    )
    write_report(summary, args.output)


if __name__ == "__main__":
    main()
