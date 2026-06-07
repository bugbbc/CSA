#!/usr/bin/env python3
"""
Tests for attention modules.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from csa.attention import (
    DenseAttention,
    LocalWindowSparseAttention,
    RandomTopKAttention,
    SimilarityTopKAttention,
    CSAAttention,
)


def test_dense_attention():
    """Test dense attention forward pass."""
    attn = DenseAttention(d_model=128, n_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 128)
    out, aux = attn(x)
    assert out.shape == (2, 16, 128), f"Expected (2, 16, 128), got {out.shape}"
    print("  ✓ DenseAttention")


def test_local_window_attention():
    """Test local window attention."""
    attn = LocalWindowSparseAttention(d_model=128, n_heads=4, window=8, dropout=0.0)
    x = torch.randn(2, 16, 128)
    out, aux = attn(x)
    assert out.shape == (2, 16, 128)
    assert aux is not None
    print("  ✓ LocalWindowSparseAttention")


def test_random_topk_attention():
    """Test random top-k attention."""
    attn = RandomTopKAttention(d_model=128, n_heads=4, k=8, dropout=0.0)
    x = torch.randn(2, 16, 128)
    out, aux = attn(x)
    assert out.shape == (2, 16, 128)
    print("  ✓ RandomTopKAttention")


def test_similarity_topk_attention():
    """Test similarity top-k attention."""
    attn = SimilarityTopKAttention(d_model=128, n_heads=4, window=8, k=8, dropout=0.0)
    x = torch.randn(2, 16, 128)
    out, aux = attn(x)
    assert out.shape == (2, 16, 128)
    print("  ✓ SimilarityTopKAttention")


def test_csa_attention():
    """Test CSA attention forward pass."""
    attn = CSAAttention(
        d_model=128, n_heads=4, window=8, k=8,
        refresh_interval=2, dropout=0.0,
    )
    x = torch.randn(2, 16, 128)
    out, aux = attn(x)
    assert out.shape == (2, 16, 128)
    assert aux is not None
    assert "routing_mask" in aux
    print("  ✓ CSAAttention")


if __name__ == "__main__":
    print("Testing Attention Modules...")
    test_dense_attention()
    test_local_window_attention()
    test_random_topk_attention()
    test_similarity_topk_attention()
    test_csa_attention()
    print("\nAll attention tests passed!")
