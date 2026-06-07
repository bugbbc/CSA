"""Experiment 5: Proxy Validation.

Compare different contribution estimators against exact intervention.
Metrics: Spearman, Kendall Tau, Top-k Overlap, NDCG.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from ..models.encoder import CSAEncoder
from ..data.proxy_validation import ProxyValidationDataset, collate_proxy
from ..attention.contrib import GradientProxyEstimator, ExactInterventionEstimator
from ..utils.metrics import (
    spearman_correlation,
    kendall_tau,
    topk_overlap,
    ndcg_at_k,
)
from ..utils.seed import set_seed


METHODS = ["csa"]  # Compare estimators via CSA
ESTIMATORS = ["gradient_norm", "input_x_gradient", "integrated_gradients", "csa_proxy"]
SEQ_LENGTHS = [64, 128, 256]


def compute_gradient_norm(
    model: CSAEncoder,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    """Compute gradient norm per token position."""
    model.zero_grad()
    input_ids = input_ids.clone().detach().requires_grad_(True)

    # We need to compute grad w.r.t. token IDs — use embedding grad norm
    embed = model.embed_tokens
    x = embed(input_ids).detach().requires_grad_(True)

    output = model.forward_with_embeddings(x, None, labels)
    loss = output["loss"]
    loss.backward()

    if x.grad is not None:
        return x.grad.norm(dim=-1).cpu().numpy()
    return np.zeros(input_ids.shape)


def compute_input_x_gradient(
    model: CSAEncoder,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    """Compute input * gradient (simple integrated gradient variant)."""
    model.zero_grad()
    embed = model.embed_tokens
    x = embed(input_ids).detach().requires_grad_(True)

    output = model.forward_with_embeddings(x, None, labels)
    loss = output["loss"]
    loss.backward()

    if x.grad is not None:
        return (x.grad * x).abs().sum(dim=-1).cpu().numpy()
    return np.zeros(input_ids.shape)


def compute_integrated_gradients(
    model: CSAEncoder,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    steps: int = 20,
) -> np.ndarray:
    """Integrated gradients approximation."""
    embed = model.embed_tokens
    baseline = torch.zeros_like(input_ids)
    x_baseline = embed(baseline).detach()
    x_input = embed(input_ids).detach()

    scaled_inputs = [
        x_baseline + (float(i) / steps) * (x_input - x_baseline)
        for i in range(steps + 1)
    ]

    grad_sum = None
    for x_scaled in scaled_inputs:
        x_scaled = x_scaled.detach().requires_grad_(True)
        model.zero_grad()
        output = model.forward_with_embeddings(x_scaled, None, labels)
        loss = output["loss"]
        loss.backward()

        if x_scaled.grad is not None:
            if grad_sum is None:
                grad_sum = x_scaled.grad.clone()
            else:
                grad_sum += x_scaled.grad

    if grad_sum is not None:
        ig = (x_input - x_baseline) * grad_sum / steps
        return ig.abs().sum(dim=-1).cpu().numpy()
    return np.zeros(input_ids.shape)


def compute_csa_proxy(
    model: CSAEncoder,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    """Compute CSA gradient proxy scores."""
    estimator = GradientProxyEstimator(baseline_type="zero")
    scores = estimator.compute(
        model=model,
        input_ids=input_ids,
        attention_mask=None,
        labels=labels,
    )
    return scores.cpu().numpy()


def compute_exact_intervention(
    model: CSAEncoder,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
) -> np.ndarray:
    """Compute exact intervention scores."""
    estimator = ExactInterventionEstimator(baseline_type="zero")
    scores = estimator.compute(
        model=model,
        input_ids=input_ids,
        attention_mask=None,
        labels=labels,
    )
    return scores.cpu().numpy()


def run_exp5_proxy(
    method: str,
    seed: int,
    seq_lengths: List[int] = None,
    num_examples: int = 20,
    batch_size: int = 4,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, float]:
    """Run proxy validation experiment."""
    if seq_lengths is None:
        seq_lengths = SEQ_LENGTHS

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    results = {}

    for seq_len in seq_lengths:
        dataset = ProxyValidationDataset(
            seq_length=seq_len,
            num_examples=num_examples,
            seed=seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_proxy,
        )

        model = CSAEncoder(
            attn_type="csa",
            task="classification",
            num_classes=2,
            max_len=seq_len + 64,
            window=min(64, seq_len),
            k=min(16, seq_len),
        ).to(device)
        model.eval()

        # Run estimator comparison
        estimator_scores = {est: [] for est in ESTIMATORS}
        exact_scores_list = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Exact intervention (baseline truth)
            exact = compute_exact_intervention(model, input_ids, labels)
            exact_scores_list.append(exact)

            # Each approximation
            with torch.no_grad():
                if seq_len <= 128:
                    gn = compute_gradient_norm(model, input_ids, labels)
                    estimator_scores["gradient_norm"].append(gn)

                    ixg = compute_input_x_gradient(model, input_ids, labels)
                    estimator_scores["input_x_gradient"].append(ixg)

                    ig = compute_integrated_gradients(model, input_ids, labels)
                    estimator_scores["integrated_gradients"].append(ig)

                csa = compute_csa_proxy(model, input_ids, labels)
                estimator_scores["csa_proxy"].append(csa)

        # Aggregate and compute correlation metrics
        for est_name in estimator_scores:
            all_est = np.concatenate(estimator_scores[est_name])
            all_exact = np.concatenate(exact_scores_list)

            if len(all_est) == 0 or len(all_exact) == 0:
                continue

            spearman = spearman_correlation(all_est, all_exact)
            kendall = kendall_tau(all_est, all_exact)

            k = min(len(all_est), 10)
            pred_indices = np.argsort(-all_est)
            true_indices = np.argsort(-all_exact)
            overlap = topk_overlap(pred_indices, true_indices, k)
            ndcg = ndcg_at_k(all_est, all_exact, k)

            results[f"{seq_len}_{est_name}_spearman"] = spearman
            results[f"{seq_len}_{est_name}_kendall"] = kendall
            results[f"{seq_len}_{est_name}_overlap"] = overlap
            results[f"{seq_len}_{est_name}_ndcg"] = ndcg

    return results
