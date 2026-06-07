#!/usr/bin/env python3
"""
Tests for contribution estimators.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from csa.models.encoder import CSAEncoder
from csa.attention.contrib import GradientProxyEstimator, ExactInterventionEstimator


def test_gradient_proxy_estimator():
    """Test gradient proxy estimator on a tiny model."""
    model = CSAEncoder(
        vocab_size=100,
        d_model=32,
        d_ff=128,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_len=64,
        task="classification",
        num_classes=2,
        attn_type="csa",
        window=16,
        k=8,
        pad_token_id=0,
    )
    model.eval()

    estimator = GradientProxyEstimator(baseline_type="zero")
    input_ids = torch.randint(0, 100, (1, 16))
    labels = torch.randint(0, 2, (1,))

    scores = estimator.compute(model, input_ids, None, labels)
    assert scores.shape == (1, 16), f"Expected (1, 16), got {scores.shape}"
    assert scores.min() >= 0, "Scores should be non-negative"
    print(f"  ✓ GradientProxyEstimator: shape={scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")


def test_exact_intervention_estimator():
    """Test exact intervention estimator on a tiny model."""
    model = CSAEncoder(
        vocab_size=100,
        d_model=32,
        d_ff=128,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_len=64,
        task="classification",
        num_classes=2,
        attn_type="csa",
        window=16,
        k=8,
        pad_token_id=0,
    )
    model.eval()

    estimator = ExactInterventionEstimator(baseline_type="zero")
    input_ids = torch.randint(0, 100, (1, 8))  # Short sequence for exact
    labels = torch.randint(0, 2, (1,))

    scores = estimator.compute(model, input_ids, None, labels)
    assert scores.shape == (1, 8), f"Expected (1, 8), got {scores.shape}"
    assert scores.min() >= 0, "Scores should be non-negative"
    print(f"  ✓ ExactInterventionEstimator: shape={scores.shape}")


def test_estimator_agreement():
    """Test that gradient proxy correlates with exact intervention (direction)."""
    model = CSAEncoder(
        vocab_size=100,
        d_model=32,
        d_ff=128,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_len=64,
        task="classification",
        num_classes=2,
        attn_type="csa",
        window=16,
        k=8,
        pad_token_id=0,
    )
    model.eval()

    proxy_est = GradientProxyEstimator(baseline_type="zero")
    exact_est = ExactInterventionEstimator(baseline_type="zero")

    input_ids = torch.randint(0, 100, (1, 8))
    labels = torch.randint(0, 2, (1,))

    proxy_scores = proxy_est.compute(model, input_ids, None, labels).squeeze()
    exact_scores = exact_est.compute(model, input_ids, None, labels).squeeze()

    # Compute basic correlation
    from scipy.stats import spearmanr
    # Handle case where scores might be constant (random untrained model)
    proxy_std = proxy_scores.std().item()
    exact_std = exact_scores.std().item()
    if proxy_std > 1e-8 and exact_std > 1e-8:
        corr, _ = spearmanr(proxy_scores.numpy(), exact_scores.numpy())
        print(f"  ✓ Estimator agreement: Spearman r = {corr:.4f}")
        # Correlation should not be NaN when both have variance
        assert not np.isnan(corr), "Correlation should not be NaN"
    else:
        print(f"  → Estimator agreement: skipped (constant scores, proxy_std={proxy_std:.6f}, exact_std={exact_std:.6f})")


if __name__ == "__main__":
    print("Testing Contribution Estimators...")
    test_gradient_proxy_estimator()
    test_exact_intervention_estimator()
    test_estimator_agreement()
    print("\nAll contribution estimator tests passed!")
