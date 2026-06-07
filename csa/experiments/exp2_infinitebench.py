"""Experiment 2: Very Long Context (InfiniteBench)."""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

from ..models.encoder import CSAEncoder
from ..data.infinitebench import InfiniteBenchDataset, collate_infinitebench
from ..evaluation.metrics import exact_match, retrieval_success


METHODS = ["dense", "local_window", "similarity_topk", "csa"]
CONTEXT_LENGTHS = [8192, 16384, 32768, 65536]  # 8K, 16K, 32K, 64K


def evaluate_infinitebench(
    model: CSAEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on InfiniteBench."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)

        # Decode predictions
        logits = output["logits"]
        pred_ids = logits.argmax(dim=-1)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
        for tokens, inp in zip(pred_ids, input_ids):
            # Only decode the generated part (after input)
            gen_tokens = tokens[inp.shape[0] // 2:].tolist()
            pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            all_preds.append(pred_text)

        all_targets.extend(batch["answers"])

    # Metrics
    em_scores = [exact_match(p, t) for p, t in zip(all_preds, all_targets)]
    retrieval = retrieval_success(all_preds, [[t] for t in all_targets])

    return {
        "accuracy": float(np.mean(em_scores)),
        "retrieval_success_rate": retrieval,
    }


def run_exp2_infinitebench(
    method: str,
    seed: int,
    context_lengths: List[int] = None,
    batch_size: int = 2,
    max_batches: int = 5,
    device: str = "cuda",
    window: int = 128,
    k: int = 32,
    **kwargs,
) -> Dict[str, float]:
    """Run InfiniteBench evaluation."""
    if context_lengths is None:
        context_lengths = CONTEXT_LENGTHS

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    results = {}

    for ctx_len in context_lengths:
        # Build model with sufficient max_len
        model = CSAEncoder(
            attn_type=method,
            window=window,
            k=k,
            max_len=max(ctx_len + 1024, 65536),
            refresh_interval=4,
            baseline_type="zero",
        ).to(device)
        model.eval()

        dataset = InfiniteBenchDataset(
            context_length=ctx_len,
            num_examples=10,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, 2),
            shuffle=False,
            collate_fn=collate_infinitebench,
        )

        ctx_results = evaluate_infinitebench(model, dataloader, device)
        results[f"{ctx_len}_accuracy"] = ctx_results["accuracy"]
        results[f"{ctx_len}_retrieval"] = ctx_results["retrieval_success_rate"]

    return results
