#!/usr/bin/env python3
"""
Tests for the CSA encoder model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from csa.models.encoder import CSAEncoder


def test_model_creation():
    """Test model instantiation."""
    model = CSAEncoder(
        vocab_size=1000,
        d_model=64,
        d_ff=256,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_len=256,
        task="lm",
        attn_type="dense",
        pad_token_id=0,
    )
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"
    print(f"  ✓ Model created ({total_params:,} params)")


def test_lm_forward():
    """Test LM forward pass."""
    model = CSAEncoder(
        vocab_size=1000,
        d_model=64,
        d_ff=256,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_len=256,
        task="lm",
        attn_type="dense",
        pad_token_id=0,
    )
    input_ids = torch.randint(0, 1000, (2, 32))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    assert "logits" in output
    assert output["logits"].shape == (2, 32, 1000)
    assert "loss" in output
    assert output["loss"].item() > 0, "Loss should be positive"
    print("  ✓ LM forward + loss")


def test_classification_forward():
    """Test classification forward pass."""
    model = CSAEncoder(
        vocab_size=1000,
        d_model=64,
        d_ff=256,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_len=256,
        task="classification",
        num_classes=3,
        attn_type="csa",
        window=16,
        k=8,
        pad_token_id=0,
    )
    input_ids = torch.randint(0, 1000, (2, 32))
    labels = torch.randint(0, 3, (2,))

    output = model(input_ids, labels=labels)
    assert "logits" in output
    assert output["logits"].shape == (2, 3)
    assert "loss" in output
    print("  ✓ Classification forward + loss")


def test_forward_with_embeddings():
    """Test forward_with_embeddings (used by CSA contrib estimator)."""
    model = CSAEncoder(
        vocab_size=1000,
        d_model=64,
        d_ff=256,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_len=256,
        task="lm",
        attn_type="csa",
        pad_token_id=0,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    embeddings = model.embed_tokens(input_ids)
    labels = input_ids.clone()

    output = model.forward_with_embeddings(embeddings, labels=labels)
    assert "logits" in output
    assert "loss" in output
    print("  ✓ forward_with_embeddings")


def test_gradient_flow():
    """Test gradients flow through all parameters."""
    model = CSAEncoder(
        vocab_size=1000,
        d_model=64,
        d_ff=256,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_len=256,
        task="lm",
        attn_type="csa",
        window=16,
        k=8,
        pad_token_id=0,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    loss = output["loss"]
    loss.backward()

    grad_exists = False
    zero_grad = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_exists = True
            if param.grad.abs().sum().item() > 0:
                zero_grad = False
                break

    assert grad_exists, "No gradients found"
    assert not zero_grad, "All gradients are zero"
    print("  ✓ Gradient flow")


if __name__ == "__main__":
    print("Testing CSAEncoder Model...")
    test_model_creation()
    test_lm_forward()
    test_classification_forward()
    test_forward_with_embeddings()
    test_gradient_flow()
    print("\nAll model tests passed!")
