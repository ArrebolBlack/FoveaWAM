"""Train FoveaWAM on the OnTimeBench demonstrations."""

import os
import random
import signal
from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from foveawam.cli._training_setup import build_training_loader, fit_normalization
from foveawam.config.validate import validate_config
from foveawam.data.dataset import OnTimeBenchDataset
from foveawam.models.model import build_model
from foveawam.training.distributed import _DistributedRuntime
from foveawam.training.trainer import Trainer


def _run_training(config: DictConfig) -> None:
    if config.compute.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("FoveaWAM training requires a CUDA GPU with BF16 support")
    if not config.compute.distributed and int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("multi-rank torchrun requires compute.distributed=true")
    runtime = _DistributedRuntime() if config.compute.distributed else None
    try:
        rank, world_size, device = (
            (0, 1, torch.device("cuda", 0))
            if runtime is None
            else (runtime.rank, runtime.world_size, runtime.device)
        )
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = config.compute.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.compute.allow_tf32
        torch.backends.cudnn.benchmark = not config.compute.deterministic
        torch.use_deterministic_algorithms(config.compute.deterministic)
        setattr(torch.utils.deterministic, "fill_uninitialized_memory", True)
        random.seed(config.seed + rank)
        torch.manual_seed(config.seed + rank)
        loader, spec = build_training_loader(config, rank=rank, world_size=world_size)
        normalization = None
        if config.training.resume_from is None:
            normalization = fit_normalization(
                cast(OnTimeBenchDataset, loader.dataset), spec, rank, device
            )
        model = build_model(config).to(device)
        if runtime is not None:
            model = runtime.wrap_model(model)
        trainer = Trainer(
            config, model, loader, normalization, rank=rank, world_size=world_size, device=device
        )
        pause = getattr(signal, "SIGUSR1", None)
        if pause is not None:
            previous = signal.signal(pause, lambda *_: setattr(trainer, "stop_requested", True))
        try:
            setattr(torch.utils.deterministic, "fill_uninitialized_memory", False)
            trainer.fit()
        finally:
            if pause is not None:
                signal.signal(pause, previous)
    finally:
        if runtime is not None:
            runtime.close()


@hydra.main(version_base=None, config_path="pkg://foveawam.configs", config_name="foveawam_train10")
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    validate_config(config)
    OmegaConf.set_readonly(config, True)
    _run_training(config)


if __name__ == "__main__":
    main()
