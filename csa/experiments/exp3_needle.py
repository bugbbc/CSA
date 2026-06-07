"""Experiment 3: Needle-in-a-Haystack."""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

from ..models.encoder import CSAEncoder
from ..data.needle_haystack import NeedleHaystackDataset, collate_needle
from ..evaluation.metrics import exact_match
from ..attention.csa import CSAAttention


METHODS = ["dense", "local_window", "similarity_topk", "csa"]
CONTEXT_LENGTHS = [8192, 16384, 32768, 65536]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]


def evaluate_needle(
    model: CSAEncoder,
    dataloader: DataLoader,
    device: torch.device,
    return_overlap: bool = False,
) -> Dict:
    """Evaluate needle retrieval accuracy and needle recall."""
    model.eval()
    em_scores = []
    needle_recalls = []
    depth_info = []
    context_len_info = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, return_aux=return_overlap)
            logits = output.get("logits")

        # Get predictions
        if isinstance(logits, tuple):
            start_logits, end_logits = logits
            pred_answers = []
            for s, e in zip(start_logits.argmax(-1), end_logits.argmax(-1)):
                if s <= e:
                    pred_answers.append(f"[{s},{e}]")
                else:
                    pred_answers.append("")
        else:
            pred_ids = logits.argmax(dim=-1)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
            pred_answers = []
            for tokens in pred_ids:
                gen = tokens[input_ids.shape[1] // 2:].tolist()
                pred_answers.append(tokenizer.decode(gen, skip_special_tokens=True).strip())

        # Compute exact match
        for pred, true_ans in zip(pred_answers, batch["answers"]):
            em = exact_match(pred, true_ans)
            em_scores.append(em)

        depth_info.extend(batch["depths"])
        context_len_info.extend(batch["context_lengths"])

        # Needle recall: check if CSA selected tokens overlap with needle position
        if return_overlap and "aux" in output:
            aux_list = output["aux"]
            for aux in aux_list:
                if "routing_mask" in aux:
                    # Check if needle positions are in the routing mask
                    for i, needle_pos in enumerate(batch["needle_start_positions"]):
                        if isinstance(needle_pos, torch.Tensor):
                            needle_pos = needle_pos.item()
                        # Simple overlap: was the needle position selected?
                        mask = aux["routing_mask"][i]  # [1, L, L]
                        mask_queries = mask.any(dim=0).squeeze(0)  # [L]
                        needle_recall = 1.0 if needle_pos < mask_queries.shape[0] and mask_queries[needle_pos].item() else 0.0
                        needle_recalls.append(needle_recall)
        else:
            needle_recalls = [0.0] * len(em_scores)

    results = {
        "accuracy": float(np.mean(em_scores)) if em_scores else 0.0,
        "needle_recall": float(np.mean(needle_recalls)) if needle_recalls else 0.0,
    }
    return results


def run_exp3_needle(
    method: str,
    seed: int,
    context_lengths: List[int] = None,
    depths: List[float] = None,
    batch_size: int = 2,
    max_batches: int = 3,
    device: str = "cuda",
    window: int = 128,
    k: int = 32,
    **kwargs,
) -> Dict[str, float]:
    """Run Needle-in-a-Haystack evaluation."""
    if context_lengths is None:
        context_lengths = CONTEXT_LENGTHS
    if depths is None:
        depths = DEPTHS

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Single model per method (smaller for testing)
    model = CSAEncoder(
        attn_type=method,
        window=window,
        k=k,
        max_len=max(context_lengths),
        refresh_interval=1,
        baseline_type="zero",
    ).to(device)
    model.eval()

    dataset = NeedleHaystackDataset(
        context_lengths=context_lengths,
        depths=depths,
        num_examples_per_config=2,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, 4),
        shuffle=False,
        collate_fn=collate_needle,
    )

    results = evaluate_needle(
        model, dataloader, device,
        return_overlap=(method == "csa"),
    )

    return results
