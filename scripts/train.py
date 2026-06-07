#!/usr/bin/env python3
"""
Training entry point for CSA models.

Usage:
    python scripts/train.py --model csa --epochs 10 --batch-size 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, TensorDataset

from csa.utils.seed import set_seed
from csa.models.encoder import CSAEncoder
from csa.training.trainer import Trainer
from csa.utils.logging import WandbLogger
from configs.base import ExperimentConfig, ModelConfig, TrainingConfig, AttentionConfig


def make_dummy_data(
    vocab_size: int, seq_length: int, num_examples: int, batch_size: int
):
    """Create dummy data for smoke testing."""
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    input_ids = torch.randint(0, vocab_size, (num_examples, seq_length))
    attention_mask = torch.ones(num_examples, seq_length)
    labels = input_ids.clone()

    dataset = DictDataset(input_ids, attention_mask, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def main():
    parser = argparse.ArgumentParser(description="Train CSA model")
    parser.add_argument("--model", type=str, default="csa",
                       choices=["dense", "local_window", "random_topk",
                                "similarity_topk", "gated", "csa"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/train")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--baseline", type=str, default="zero",
                       choices=["zero", "mask", "mean"])

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Config
    config = ExperimentConfig(
        experiment_name=f"train_{args.model}",
        output_dir=os.path.join(args.log_dir, "results"),
        model=ModelConfig(
            d_model=512,
            d_ff=2048,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            max_len=args.seq_length + 64,
            task="lm",
            attention=AttentionConfig(
                type=args.model,
                window=args.window,
                k=args.k,
                refresh_interval=4,
                baseline_type=args.baseline,
            ),
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            warmup_steps=100,
            seed=args.seed,
            use_amp=torch.cuda.is_available(),
        ),
        log_wandb=False,
    )

    # Model
    model = CSAEncoder(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout,
        max_len=config.model.max_len,
        pad_token_id=config.model.pad_token_id,
        task=config.model.task,
        attn_type=config.model.attention.type,
        window=config.model.attention.window,
        k=config.model.attention.k,
        refresh_interval=config.model.attention.refresh_interval,
        baseline_type=config.model.attention.baseline_type,
    ).to(device)

    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_loader = make_dummy_data(
        config.model.vocab_size,
        args.seq_length,
        num_examples=100,
        batch_size=args.batch_size,
    )
    eval_loader = make_dummy_data(
        config.model.vocab_size,
        args.seq_length,
        num_examples=20,
        batch_size=args.batch_size,
    )

    # Trainer
    logger = WandbLogger(enabled=False)
    trainer = Trainer(
        model, config, device, logger=logger, log_dir=args.log_dir
    )

    # Train
    trainer.train(
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        num_epochs=args.epochs,
    )

    print(f"Training complete. Final loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()
