"""Experiment 1: LongBench Main Benchmarks."""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

from ..attention import build_attention
from ..models.encoder import CSAEncoder
from ..data.longbench import LongBenchDataset, collate_longbench, LONGBENCH_TASKS
from ..evaluation.metrics import compute_metric, exact_match_set


METHODS = [
    "dense",
    "local_window",
    "random_topk",
    "similarity_topk",
    "gated",
    "gated_sparse",
    "csa",
]

METRICS = {
    "narrativeqa": "rouge-l",
    "qasper": "rouge-l",
    "multifieldqa_en": "f1",
    "hotpotqa": "f1",
    "2wikimultihopqa": "f1",
    "musique": "f1",
    "govreport": "rouge-l",
    "qmsum": "rouge-l",
}


def evaluate_longbench(
    model: CSAEncoder,
    dataloader: DataLoader,
    metric: str,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model on a LongBench task."""
    model.eval()
    predictions = []
    references = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)

        # Decode predictions
        logits = output["logits"]
        if isinstance(logits, tuple):
            # QA head
            start_logits, end_logits = logits
            pred_start = start_logits.argmax(dim=-1)
            pred_end = end_logits.argmax(dim=-1)
            for s, e in zip(pred_start, pred_end):
                s, e = s.item(), e.item()
                if s <= e:
                    predictions.append(f"[{s},{e}]")
                else:
                    predictions.append("")
        else:
            # LM/classification head
            pred_tokens = logits.argmax(dim=-1)

            # Convert tokens to text
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
            for tokens in pred_tokens:
                pred_text = tokenizer.decode(tokens[input_ids.shape[1] // 2:].tolist(), skip_special_tokens=True)
                predictions.append(pred_text.strip())

        references.extend(batch["answers"])

    # Compute metrics
    scores = []
    for pred, refs in zip(predictions, references):
        if isinstance(refs, list):
            best = max(compute_metric(pred, r, metric) for r in refs)
        else:
            best = compute_metric(pred, refs, metric)
        scores.append(best)

    return {f"{metric}": float(np.mean(scores))}


def run_exp1_longbench(
    method: str,
    seed: int,
    tasks: List[str] = list(LONGBENCH_TASKS.keys()),
    batch_size: int = 4,
    max_length: int = 4096,
    max_batches: int = 10,
    device: str = "cuda",
    window: int = 128,
    k: int = 32,
    **kwargs,
) -> Dict[str, float]:
    """Run LongBench evaluation for a single method and seed."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build model
    model = CSAEncoder(
        attn_type=method,
        window=window,
        k=k,
        refresh_interval=4,
        baseline_type="zero",
    ).to(device)
    model.eval()

    results = {}
    for task in tasks:
        metric = METRICS.get(task, "f1")

        dataset = LongBenchDataset(
            task_name=task,
            split="test",
            max_length=max_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_longbench,
        )

        task_results = evaluate_longbench(
            model, dataloader, metric, device, max_batches=max_batches
        )
        results.update({f"{task}_{k}": v for k, v in task_results.items()})

    # Compute average
    all_scores = [v for k, v in results.items()]
    if all_scores:
        results["average"] = float(np.mean(all_scores))

    return results
