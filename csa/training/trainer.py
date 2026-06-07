"""Training loop for CSA models."""

from __future__ import annotations
import os
import time
from typing import Dict, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .optimizer import build_optimizer
from .distributed import is_main_process, wrap_model_ddp, get_local_rank
from ..utils.logging import WandbLogger


class Trainer:
    """Trainer for CSA experiments."""

    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device,
        logger: Optional[WandbLogger] = None,
        log_dir: str = "logs",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Optimizer
        self.optimizer, self.scheduler = build_optimizer(
            model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.training.use_amp)

        # DDP
        self.model = wrap_model_ddp(
            model,
            find_unused_parameters=(config.model.attention.type == "csa"),
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask").to(self.device) if "attention_mask" in batch else None
        labels = batch.get("labels").to(self.device) if "labels" in batch else input_ids.clone()

        with torch.cuda.amp.autocast(enabled=self.config.training.use_amp):
            output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output["loss"]

        # Scale loss for gradient accumulation
        loss = loss / self.config.training.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        metrics = {"loss": loss.item() * self.config.training.gradient_accumulation_steps}

        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process()):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask").to(self.device) if "attention_mask" in batch else None
            labels = batch.get("labels").to(self.device) if "labels" in batch else input_ids.clone()

            with torch.cuda.amp.autocast(enabled=self.config.training.use_amp):
                output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output["loss"]

            total_loss += loss.item()
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)

        if self.logger:
            self.logger.log({"eval_loss": avg_loss}, step=self.global_step)

        return {"loss": avg_loss}

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 1,
    ):
        """Full training loop."""
        self.model.train()
        total_steps = 0

        for epoch in range(num_epochs):
            self.epoch = epoch
            progress = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                disable=not is_main_process(),
            )

            epoch_start = time.time()
            for batch in progress:
                metrics = self.train_step(batch)
                total_steps += 1

                # Logging
                if total_steps % 10 == 0 and is_main_process():
                    lr = self.scheduler.get_last_lr()[0]
                    progress.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "lr": f"{lr:.2e}",
                    })

                    if self.logger:
                        self.logger.log({
                            "train_loss": metrics["loss"],
                            "lr": lr,
                            "epoch": epoch,
                        }, step=total_steps)

                # Evaluation
                if eval_dataloader and total_steps % self.config.training.eval_every == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    if is_main_process():
                        print(f"Step {total_steps}: eval_loss = {eval_metrics['loss']:.4f}")

                    if eval_metrics["loss"] < self.best_loss:
                        self.best_loss = eval_metrics["loss"]
                        self.save_checkpoint("best")

                # Max steps reached
                if self.config.training.max_steps > 0 and total_steps >= self.config.training.max_steps:
                    break

            epoch_time = time.time() - epoch_start
            if is_main_process():
                print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")

            # Save checkpoint at epoch end
            self.save_checkpoint(f"epoch_{epoch + 1}")

    def save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        if not is_main_process():
            return
        ckpt_dir = os.path.join(self.log_dir, "checkpoints", tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

        torch.save({
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }, os.path.join(ckpt_dir, "checkpoint.pt"))

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        model_state = checkpoint["model_state_dict"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
