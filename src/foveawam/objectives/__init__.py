"""Masked objectives for action generation and scene prediction."""

from typing import NamedTuple

from torch import Tensor


class _LossResult(NamedTuple):
    loss: Tensor
    valid_count: Tensor
