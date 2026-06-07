"""Evaluation metrics for NLP tasks."""

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np


# ─── Text Normalization ──────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ─── Exact Match ─────────────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def exact_match_set(prediction: str, ground_truths: List[str]) -> float:
    """Exact match against any of the ground truths."""
    for gt in ground_truths:
        if normalize_answer(prediction) == normalize_answer(gt):
            return 1.0
    return 0.0


# ─── F1 Score ────────────────────────────────────────────────────────────────

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ─── ROUGE-L ─────────────────────────────────────────────────────────────────

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L score (F1 variant)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    lcs = _lcs_length(pred_tokens, gt_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0
    recall = lcs / len(gt_tokens) if gt_tokens else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── Accuracy ────────────────────────────────────────────────────────────────

def accuracy(prediction: str, ground_truth: str) -> float:
    """Simple accuracy."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ─── Retrieval Success Rate ──────────────────────────────────────────────────

def retrieval_success(predictions: List[str], ground_truths: List[str]) -> float:
    """Fraction of predictions that match any ground truth."""
    if len(predictions) == 0:
        return 0.0
    matches = sum(
        any(normalize_answer(p) == normalize_answer(gt) for gt in ground_truths)
        for p in predictions
    )
    return matches / len(predictions)


# ─── Metric Router ───────────────────────────────────────────────────────────

def compute_metric(prediction: str, ground_truth: str, metric: str = "f1") -> float:
    """Compute specified metric."""
    if metric == "em" or metric == "exact_match":
        return exact_match(prediction, ground_truth)
    elif metric == "f1":
        return f1_score(prediction, ground_truth)
    elif metric == "rouge-l":
        return rouge_l(prediction, ground_truth)
    elif metric == "accuracy":
        return accuracy(prediction, ground_truth)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_metrics_batch(
    predictions: List[str],
    ground_truths: List[str],
    metrics: List[str] = ["f1"],
) -> Dict[str, float]:
    """Compute multiple metrics for batch of predictions."""
    results = {}
    for metric in metrics:
        scores = [
            compute_metric(p, gt, metric)
            for p, gt in zip(predictions, ground_truths)
        ]
        results[metric] = float(np.mean(scores))
    return results
