"""Optimizer and learning rate scheduler."""

from typing import Optional

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1
):
    """Linear warmup followed by linear decay."""

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_steps: int = 100000,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
):
    """Build AdamW optimizer with cosine schedule."""
    # Separate weight decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "layer_norm" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps=warmup_steps, total_steps=max_steps
    )

    return optimizer, scheduler
