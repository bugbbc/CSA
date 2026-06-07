"""Evaluation metrics for CSA experiments."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau


# ─── Ranking Metrics ──────────────────────────────────────────────────────────

def spearman_correlation(
    scores_pred: np.ndarray, scores_true: np.ndarray
) -> float:
    """Spearman rank correlation between predicted and true scores."""
    corr, _ = spearmanr(scores_pred, scores_true)
    return float(corr) if not np.isnan(corr) else 0.0


def kendall_tau(
    scores_pred: np.ndarray, scores_true: np.ndarray
) -> float:
    """Kendall Tau rank correlation."""
    tau, _ = kendalltau(scores_pred, scores_true)
    return float(tau) if not np.isnan(tau) else 0.0


def topk_overlap(
    pred_indices: np.ndarray, true_indices: np.ndarray, k: int
) -> float:
    """Overlap between top-k predicted and true indices."""
    pred_topk = set(pred_indices[:k].tolist())
    true_topk = set(true_indices[:k].tolist())
    if len(true_topk) == 0:
        return 0.0
    return len(pred_topk & true_topk) / min(k, len(true_topk))


def ndcg_at_k(
    scores_pred: np.ndarray, scores_true: np.ndarray, k: int
) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    # Rank by predicted scores, evaluate using true scores
    order = np.argsort(-scores_pred)[:k]
    dcg = sum((2 ** scores_true[i] - 1) / math.log2(idx + 2) for idx, i in enumerate(order))
    # Ideal ordering
    ideal_order = np.argsort(-scores_true)[:k]
    idcg = sum((2 ** scores_true[i] - 1) / math.log2(idx + 2) for idx, i in enumerate(ideal_order))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Sparse Attention Metrics ────────────────────────────────────────────────

def evidence_recall_at_k(
    selected_indices: np.ndarray, evidence_indices: np.ndarray, k: int
) -> float:
    """ER@k: fraction of evidence tokens retrieved in top-k selected."""
    selected = set(selected_indices[:k].tolist())
    evidence = set(evidence_indices.tolist())
    if len(evidence) == 0:
        return 0.0
    return len(selected & evidence) / len(evidence)


def spurious_inclusion_at_k(
    selected_indices: np.ndarray, spurious_indices: np.ndarray, k: int
) -> float:
    """SI@k: fraction of spurious tokens in top-k selected."""
    selected = set(selected_indices[:k].tolist())
    spurious = set(spurious_indices.tolist())
    return len(selected & spurious) / k if k > 0 else 0.0


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate_results(
    results: List[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """Aggregate results across trials: mean ± std."""
    aggregated = {}
    for key in results[0]:
        values = np.array([r[key] for r in results])
        aggregated[key] = (float(np.mean(values)), float(np.std(values)))
    return aggregated


def results_to_table(results: Dict[str, Tuple[float, float]]) -> str:
    """Pretty-print results as a table row with mean±std."""
    lines = []
    for key, (mean, std) in results.items():
        lines.append(f"{key}: {mean:.4f} ± {std:.4f}")
    return "\n".join(lines)
