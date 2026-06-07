"""Experiment 6: Ablation Study.

Systematic analysis of hyperparameters:
- Different k values (8, 16, 32, 64, 128)
- Different w values (32, 64, 128, 256)
- Different refresh intervals r (1, 2, 4, 8)
- Different baselines (zero, mask, mean)
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

from ..models.encoder import CSAEncoder
from ..data.causal_robustness import CausalRobustnessDataset, collate_causal
from ..evaluation.metrics import accuracy


BASE_CONFIG = {
    "seq_length": 64,
    "num_train": 100,
    "num_test": 50,
    "batch_size": 16,
    "num_epochs": 1,
    "window": 128,
    "k": 32,
}


def evaluate_ablation(
    model: CSAEncoder,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
            logits = output["logits"]
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def run_exp6_ablation(
    method: str,
    seed: int,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, float]:
    """Run ablation experiments for CSA."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    results = {}

    # ── Vary k ──────────────────────────────────────────────────────
    k_values = [8, 16, 32, 64, 128]
    for k in k_values:
        model = CSAEncoder(
            attn_type="csa",
            task="classification",
            num_classes=2,
            max_len=128,
            window=BASE_CONFIG["window"],
            k=k,
            refresh_interval=2,
            baseline_type="zero",
        ).to(device)

        train_ds = CausalRobustnessDataset(
            num_examples=BASE_CONFIG["num_train"],
            seq_length=BASE_CONFIG["seq_length"],
            split="train",
            seed=seed,
        )
        test_ds = CausalRobustnessDataset(
            num_examples=BASE_CONFIG["num_test"],
            seq_length=BASE_CONFIG["seq_length"],
            split="test",
            seed=seed + 1,
        )
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_causal)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_causal)

        # Quick training
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for epoch in range(BASE_CONFIG["num_epochs"]):
            for batch in train_loader:
                model.train()
                output = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = output["loss"]
                loss.backward()
                optim.step()
                optim.zero_grad()

        acc = evaluate_ablation(model, test_loader, device)
        results[f"k_{k}"] = acc

    # ── Vary w ──────────────────────────────────────────────────────
    w_values = [32, 64, 128, 256]
    for w in w_values:
        model = CSAEncoder(
            attn_type="csa",
            task="classification",
            num_classes=2,
            max_len=512,
            window=w,
            k=BASE_CONFIG["k"],
            refresh_interval=2,
            baseline_type="zero",
        ).to(device)

        train_ds = CausalRobustnessDataset(
            num_examples=BASE_CONFIG["num_train"],
            seq_length=min(256, w * 2),
            split="train",
            seed=seed,
        )
        test_ds = CausalRobustnessDataset(
            num_examples=BASE_CONFIG["num_test"],
            seq_length=min(256, w * 2),
            split="test",
            seed=seed + 1,
        )
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_causal)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_causal)

        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for epoch in range(BASE_CONFIG["num_epochs"]):
            for batch in train_loader:
                model.train()
                output = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = output["loss"]
                loss.backward()
                optim.step()
                optim.zero_grad()

        acc = evaluate_ablation(model, test_loader, device)
        results[f"w_{w}"] = acc

    # ── Vary r (refresh interval) ────────────────────────────────────
    r_values = [1, 2, 4, 8]
    for r in r_values:
        model = CSAEncoder(
            attn_type="csa",
            task="classification",
            num_classes=2,
            max_len=128,
            window=BASE_CONFIG["window"],
            k=BASE_CONFIG["k"],
            refresh_interval=r,
            baseline_type="zero",
        ).to(device)

        train_ds = CausalRobustnessDataset(
            num_examples=BASE_CONFIG["num_train"],
            seq_length=BASE_CONFIG["seq_length"],
            split="train",
            seed=seed,
        )
        test_ds = CausalRobustnessDataset(
            num_examples=BASE_CONFIG["num_test"],
            seq_length=BASE_CONFIG["seq_length"],
            split="test",
            seed=seed + 1,
        )
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_causal)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_causal)

        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for epoch in range(BASE_CONFIG["num_epochs"]):
            for batch in train_loader:
                model.train()
                output = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = output["loss"]
                loss.backward()
                optim.step()
                optim.zero_grad()

        acc = evaluate_ablation(model, test_loader, device)
        results[f"r_{r}"] = acc

    # ── Vary baseline type ───────────────────────────────────────────
    baselines = ["zero", "mask", "mean"]
    for baseline in baselines:
        model = CSAEncoder(
            attn_type="csa",
            task="classification",
            num_classes=2,
            max_len=128,
            window=BASE_CONFIG["window"],
            k=BASE_CONFIG["k"],
            refresh_interval=2,
            baseline_type=baseline,
        ).to(device)

        train_ds = CausalRobustnessDataset(
            num_examples=BASE_CONFIG["num_train"],
            seq_length=BASE_CONFIG["seq_length"],
            split="train",
            seed=seed,
        )
        test_ds = CausalRobustnessDataset(
            num_examples=BASE_CONFIG["num_test"],
            seq_length=BASE_CONFIG["seq_length"],
            split="test",
            seed=seed + 1,
        )
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_causal)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_causal)

        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for epoch in range(BASE_CONFIG["num_epochs"]):
            for batch in train_loader:
                model.train()
                output = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = output["loss"]
                loss.backward()
                optim.step()
                optim.zero_grad()

        acc = evaluate_ablation(model, test_loader, device)
        results[f"baseline_{baseline}"] = acc

    return results
