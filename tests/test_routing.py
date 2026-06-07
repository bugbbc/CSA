#!/usr/bin/env python3
"""
Tests for attention routing strategies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from csa.attention.routing import (
    DenseRouting,
    LocalWindowRouting,
    TopKRouting,
    RandomTopKRouting,
    LocalWindowPlusTopKRouting,
)


def test_dense_routing():
    """Test dense (full) routing mask."""
    routing = DenseRouting()
    scores = torch.randn(2, 4, 8, 8)
    mask = routing.compute_mask(scores)
    assert mask.shape == (2, 1, 8, 8), f"Expected (2, 1, 8, 8), got {mask.shape}"
    assert mask.all(), "Dense mask should be all True"
    print("  ✓ DenseRouting")


def test_local_window_routing():
    """Test local window routing mask shape and pattern."""
    routing = LocalWindowRouting(window=4)
    scores = torch.randn(1, 1, 10, 10)
    mask = routing.compute_mask(scores)
    assert mask.shape == (1, 1, 10, 10)
    # Check that far-away positions are masked out
    assert not mask[0, 0, 0, 9].item(), "Position |0-9| >= 2 should be False"
    assert mask[0, 0, 0, 1].item(), "Position |0-1| < 2 should be True"
    print("  ✓ LocalWindowRouting")


def test_topk_routing():
    """Test top-k routing selects correct positions."""
    routing = TopKRouting(k=3)
    # Create scores where only first 3 positions have high values
    scores = torch.zeros(1, 1, 5, 10)
    scores[0, 0, :, :3] = 10.0
    mask = routing.compute_mask(scores)
    assert mask.shape == (1, 1, 5, 10)
    # Top-k should select the 3 highest scoring positions
    selected = mask[0, 0, 0].sum().item()
    assert selected == 3, f"Expected 3 selected, got {selected}"
    print("  ✓ TopKRouting")


def test_random_topk_routing():
    """Test random top-k routing."""
    routing = RandomTopKRouting(k=4)
    scores = torch.randn(1, 1, 8, 20)
    mask = routing.compute_mask(scores)
    assert mask.shape == (1, 1, 8, 20)
    # Should have exactly k selected positions per query
    selected = mask[0, 0, 0].sum().item()
    assert selected == 4, f"Expected 4 selected, got {selected}"
    print("  ✓ RandomTopKRouting")


def test_window_plus_topk():
    """Test combined local window + top-k routing."""
    routing = LocalWindowPlusTopKRouting(window=4, k=3)
    scores = torch.randn(1, 1, 10, 10)
    mask = routing.compute_mask(scores)
    assert mask.shape == (1, 1, 10, 10)
    # Should have at least window-mask positions selected
    window_only = (torch.arange(10).view(-1, 1) - torch.arange(10).view(1, -1)).abs() < 2
    assert (mask[0, 0] >= window_only).all(), "Window positions must be selected"
    print("  ✓ LocalWindowPlusTopKRouting")


if __name__ == "__main__":
    print("Testing Routing Strategies...")
    test_dense_routing()
    test_local_window_routing()
    test_topk_routing()
    test_random_topk_routing()
    test_window_plus_topk()
    print("\nAll routing tests passed!")
