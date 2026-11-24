"""AdamW with linear warmup followed by cosine decay."""

import math

from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimization(
    config: DictConfig, model: nn.Module, *, updates_per_epoch: int
) -> tuple[AdamW, LambdaLR]:
    options = config.training.optimizer
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=options.learning_rate,
        betas=tuple(options.betas),
        eps=options.eps,
        weight_decay=options.weight_decay,
        fused=options.fused,
    )
    schedule = config.training.scheduler
    warmup = int(schedule.warmup_optimizer_steps)
    start = int(schedule.warmup_start_step)
    total = updates_per_epoch * int(schedule.total_epochs)
    minimum = float(schedule.min_learning_rate_ratio)

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + start) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return minimum + 0.5 * (1 - minimum) * (1 + math.cos(math.pi * progress))

    return optimizer, LambdaLR(optimizer, lr_lambda=multiplier)
