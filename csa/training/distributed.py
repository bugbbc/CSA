"""Distributed data parallel utilities."""

import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def is_dist_avail_and_initialized():
    """Check if distributed training is available."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get number of processes in distributed training."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get rank of current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    """Get local rank."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """Check if current process is main (rank 0)."""
    return get_rank() == 0


def setup_ddp(backend: str = "nccl"):
    """Initialize distributed process group."""
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return local_rank


def cleanup_ddp():
    """Clean up distributed process group."""
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = True,
):
    """Wrap model in DDP if distributed training is initialized."""
    if is_dist_avail_and_initialized():
        local_rank = get_local_rank()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average tensor across all processes."""
    if not is_dist_avail_and_initialized():
        return tensor
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor
