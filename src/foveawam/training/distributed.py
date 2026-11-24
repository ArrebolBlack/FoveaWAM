"""CUDA process group used by torchrun training."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


class _DistributedRuntime:
    def __init__(self) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend="nccl", device_id=self.device)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def wrap_model(self, model: nn.Module) -> DistributedDataParallel:
        return DistributedDataParallel(
            model, device_ids=[self.local_rank], gradient_as_bucket_view=True
        )

    def close(self) -> None:
        dist.destroy_process_group()


def sum_across_ranks(value: Tensor) -> Tensor:
    """Sum a detached tensor with the same shape and dtype on each rank."""
    result = value.detach().contiguous().clone()
    if dist.is_initialized():
        dist.all_reduce(result)
    return result


@contextmanager
def rank_zero_operation(rank: int, device: torch.device) -> Iterator[None]:
    """Propagate a preparation or write failure before the next collective."""
    failure = None
    try:
        yield
    except Exception as error:
        failure = error
    if dist.is_initialized():
        status = torch.tensor(int(failure is not None), device=device)
        dist.broadcast(status, src=0)
        if rank != 0 and status.item():
            raise RuntimeError("rank zero failed; see its traceback")
    if failure is not None:
        raise failure
