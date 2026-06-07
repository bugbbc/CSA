"""Logging utilities."""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

import wandb


def setup_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class WandbLogger:
    """Wrapper around wandb for experiment tracking."""

    def __init__(self, enabled: bool = True, **kwargs):
        self.enabled = enabled
        if enabled:
            wandb.init(**kwargs)

    def log(self, metrics: dict, step: Optional[int] = None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()
