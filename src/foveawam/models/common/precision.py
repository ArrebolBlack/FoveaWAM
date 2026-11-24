"""Precision settings for compiled simulation inference."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

type _CompileOptions = dict[str, str | int | bool | Callable[..., Any]]
BF16_OPTIONS: _CompileOptions = {"emulate_precision_casts": True}
CUDAGRAPH_OPTIONS: _CompileOptions = {**BF16_OPTIONS, "triton.cudagraphs": True}


def cast_bf16_linears(model: nn.Module, *, exclude: tuple[nn.Module, ...]) -> None:
    keep = {id(p) for module in exclude for p in module.parameters()}
    for module in model.modules():
        if isinstance(module, nn.Linear) and id(module.weight) not in keep:
            module.to(dtype=torch.bfloat16)
