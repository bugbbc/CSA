"""Experiment 4: Causal Robustness.

Tests whether CSA preserves task-relevant evidence and resists spurious correlations.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from ..models.encoder import CSAEncoder
from ..data.causal_robustness import CausalRobustnessDataset, collate_causal
from ..evaluation.metrics import accuracy
from ..utils.metrics import evidence_recall_at_k, spurious_inclusion_at_k
from ..training.trainer import Trainer
from ..utils.logging import WandbLogger


METHODS = ["dense", "local_window", "similarity_topk", "csa"]


def run_exp4_causal(
    method: str,
    seed: int,
    num_train: int = 200,
    num_test: int = 100,
    seq_length: int = 128,
    batch_size: int = 16,
    num_epochs: int = 2,
    device: str = "cuda",
    window: int = 128,
    k: int = 32,
    **kwargs,
) -> Dict[str, float]:
    """Run causal robustness experiment."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build model with classification head
    model = CSAEncoder(
        attn_type=method,
        window=window,
        k=k,
        task="classification",
        num_classes=2,
        max_len=seq_length + 64,
        refresh_interval=2,
        baseline_type="zero",
    ).to(device)

    # Training data (correlated)
    train_dataset = CausalRobustnessDataset(
        num_examples=num_train,
        seq_length=seq_length,
        num_evidence=3,
        spurious_correlated=True,
        split="train",
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_causal,
    )

    # Standard test (still correlated)
    test_dataset_corr = CausalRobustnessDataset(
        num_examples=num_test,
        seq_length=seq_length,
        num_evidence=3,
        spurious_correlated=True,
        split="test",
        seed=seed + 1,
    )
    test_loader_corr = DataLoader(
        test_dataset_corr,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_causal,
    )

    # Robustness test (reversed correlation)
    test_dataset_rev = CausalRobustnessDataset(
        num_examples=num_test,
        seq_length=seq_length,
        num_evidence=3,
        spurious_correlated=False,
        split="test",
        seed=seed + 2,
    )
    test_loader_rev = DataLoader(
        test_dataset_rev,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_causal,
    )

    # Quick training
    config = type('Config', (), {
        'training': type('TrainingConfig', (), {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 10,
            'max_steps': 50,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'use_amp': False,
            'eval_every': 50,
        })(),
        'model': type('ModelConfig', (), {
            'attention': type('AttentionConfig', (), {
                'type': method,
            })(),
        })(),
    })

    trainer = Trainer(
        model,
        config,
        device,
        logger=None,
    )

    # Train
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.train()
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.optimizer.zero_grad()

    # Evaluate
    def get_accuracy(loader):
        model.eval()
        correct = 0
        total = 0
        for batch in loader:
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

    # Compute evidence/spurious recall/inclusion
    def compute_er_si(loader, model_k=8):
        """Compute ER@k and SI@k for CSA using contribution scores."""
        if method != "csa":
            return {"er_at_k": 0.0, "si_at_k": 0.0}

        model.eval()
        er_scores = []
        si_scores = []

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Get contribution scores from CSA layers (need return_aux)
            with torch.no_grad():
                output = model(input_ids, attention_mask=attention_mask, labels=labels, return_aux=True)

            # Try to get contribution scores from aux
            contrib_scores_list = []
            if "aux" in output:
                for layer_aux in output["aux"]:
                    if isinstance(layer_aux, dict) and "contrib_scores" in layer_aux:
                        cs = layer_aux["contrib_scores"]
                        if isinstance(cs, torch.Tensor) and cs.ndim >= 2:
                            contrib_scores_list.append(cs)

            # Use mean contribution scores across layers that have them
            if contrib_scores_list:
                # Average across layers, take first batch item
                avg_contrib = torch.stack(contrib_scores_list).mean(dim=0)  # [B, L]
                for b_idx in range(input_ids.size(0)):
                    scores = avg_contrib[b_idx].cpu().numpy()
                    selected_indices = np.argsort(-scores)[:model_k]  # top-k by contribution

                    # Evidence positions
                    ev_ids = batch["evidence_token_ids"][b_idx] if b_idx < len(batch["evidence_token_ids"]) else batch["evidence_token_ids"][0]
                    evidence_set = set(ev_ids) if isinstance(ev_ids, list) else set(ev_ids.tolist() if hasattr(ev_ids, 'tolist') else [ev_ids])
                    seq = input_ids[b_idx].cpu().numpy()
                    evidence_positions = np.where(np.isin(seq, list(evidence_set)))[0]

                    er = evidence_recall_at_k(selected_indices, evidence_positions, model_k)
                    er_scores.append(er)

                    # Spurious positions
                    for sp_idx in range(len(batch["spurious_token_ids"])):
                        sp_ids = batch["spurious_token_ids"][sp_idx] if sp_idx < len(batch["spurious_token_ids"]) else batch["spurious_token_ids"][0]
                        spurious_set = set(sp_ids) if isinstance(sp_ids, list) else set(sp_ids.tolist() if hasattr(sp_ids, 'tolist') else [sp_ids])
                        spurious_positions = np.where(np.isin(seq, list(spurious_set)))[0]
                        if len(spurious_positions) > 0:
                            si = spurious_inclusion_at_k(selected_indices, spurious_positions, model_k)
                            si_scores.append(si)
            else:
                # Fallback: use routing mask (less precise)
                if "aux" in output:
                    for aux in output["aux"]:
                        if isinstance(aux, dict) and "routing_mask" in aux:
                            mask = aux["routing_mask"][0, 0]
                            selected = mask.any(dim=0).cpu().numpy()
                            selected_indices = np.where(selected)[0]

                            for ev_idx in range(len(batch["evidence_token_ids"])):
                                ev_ids = batch["evidence_token_ids"][ev_idx]
                                evidence_set = set(ev_ids) if isinstance(ev_ids, list) else set(ev_ids.tolist() if hasattr(ev_ids, 'tolist') else [ev_ids])
                                seq = input_ids[min(ev_idx % input_ids.size(0), input_ids.size(0) - 1)].cpu().numpy()
                                evidence_positions = np.where(np.isin(seq, list(evidence_set)))[0]
                                er = evidence_recall_at_k(selected_indices, evidence_positions, min(model_k, len(selected_indices)))
                                er_scores.append(er)

                            for sp_idx in range(len(batch["spurious_token_ids"])):
                                sp_ids = batch["spurious_token_ids"][sp_idx]
                                spurious_set = set(sp_ids) if isinstance(sp_ids, list) else set(sp_ids.tolist() if hasattr(sp_ids, 'tolist') else [sp_ids])
                                seq = input_ids[min(sp_idx % input_ids.size(0), input_ids.size(0) - 1)].cpu().numpy()
                                spurious_positions = np.where(np.isin(seq, list(spurious_set)))[0]
                                if len(spurious_positions) > 0:
                                    si = spurious_inclusion_at_k(selected_indices, spurious_positions, min(model_k, len(selected_indices)))
                                    si_scores.append(si)
                            break

        return {
            "er_at_k": float(np.mean(er_scores)) if er_scores else 0.0,
            "si_at_k": float(np.mean(si_scores)) if si_scores else 0.0,
        }

    std_accuracy = get_accuracy(test_loader_corr)
    robust_accuracy = get_accuracy(test_loader_rev)
    er_si = compute_er_si(test_loader_corr)

    return {
        "accuracy": std_accuracy,
        "robust_accuracy": robust_accuracy,
        "er_at_k": er_si["er_at_k"],
        "si_at_k": er_si["si_at_k"],
        "robustness_gap": std_accuracy - robust_accuracy,
    }
