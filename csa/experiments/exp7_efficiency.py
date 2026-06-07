"""Experiment 7: Efficiency Analysis.

Measure: inference latency, throughput, GPU memory, FLOPs.
Compare: dense, local_window, similarity_topk, csa, csa_exact.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional

import torch
import numpy as np

from ..models.encoder import CSAEncoder
from ..experiments.runner import measure_time

METHODS = ["dense", "local_window", "similarity_topk", "csa"]
SEQ_LENGTHS = [2048, 4096, 8192, 16384, 32768]


def count_flops(
    model: CSAEncoder,
    seq_length: int,
    batch_size: int = 1,
) -> float:
    """Estimate FLOPs for one forward pass."""

    def _flops_per_attention(d_model, n_heads, seq_len):
        """FLOPs for one attention layer."""
        head_dim = d_model // n_heads
        # QKV projections: 3 * (d_model * d_model)
        proj_flops = 3 * 2 * d_model * d_model
        # QK^T: 2 * seq_len * seq_len * d_model
        attn_flops = 2 * seq_len * seq_len * d_model
        # Softmax (exp + sum + div): ~3 * seq_len * seq_len * n_heads
        softmax_flops = 3 * seq_len * seq_len * n_heads
        # Attention over V: 2 * seq_len * seq_len * d_model
        attnv_flops = 2 * seq_len * seq_len * d_model
        # Output projection: 2 * d_model * d_model
        out_flops = 2 * d_model * d_model
        return proj_flops + attn_flops + softmax_flops + attnv_flops + out_flops

    def _flops_per_ffn(d_model, d_ff):
        """FLOPs for one FFN layer (two linear + GELU)."""
        return 2 * (2 * d_model * d_ff) + 2 * d_ff  # linear1 + linear2

    total_flops = 0
    # Embedding: lookup is negligible
    for _ in range(model.n_layers):
        total_flops += _flops_per_attention(model.d_model, model.n_heads, seq_length)
        total_flops += _flops_per_ffn(model.d_model, model.d_ff)

    return total_flops * batch_size


def measure_latency_memory(
    model: CSAEncoder,
    seq_length: int,
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
    n_warmup: int = 3,
    n_iters: int = 10,
) -> Dict[str, float]:
    """Measure latency and GPU memory usage."""
    model.eval()

    dummy_input = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    dummy_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.long)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy_input, attention_mask=dummy_mask)

    # Measure latency
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(dummy_input, attention_mask=dummy_mask)
    torch.cuda.synchronize()
    avg_latency = (time.time() - start) / n_iters

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input, attention_mask=dummy_mask)
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    torch.cuda.reset_peak_memory_stats(device)

    # Throughput
    throughput = batch_size / avg_latency

    return {
        "latency_ms": avg_latency * 1000,
        "throughput": throughput,
        "peak_memory_gb": peak_memory,
    }


def run_exp7_efficiency(
    method: str,
    seed: int,
    seq_lengths: List[int] = None,
    batch_size: int = 1,
    device: str = "cuda",
    window: int = 128,
    k: int = 32,
    **kwargs,
) -> Dict[str, float]:
    """Run efficiency analysis."""
    if seq_lengths is None:
        seq_lengths = SEQ_LENGTHS

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    results = {}

    for seq_len in seq_lengths:
        if seq_len > 32768 and method in ["dense", "csa_exact"]:
            # Skip too long sequences for expensive methods
            continue

        model = CSAEncoder(
            attn_type=method,
            window=window,
            k=k,
            max_len=seq_len + 1024,
            refresh_interval=4,
            baseline_type="zero",
        ).to(device)
        model.eval()

        try:
            eff = measure_latency_memory(
                model, seq_len, batch_size=batch_size, device=device
            )
            flops = count_flops(model, seq_len, batch_size)

            results[f"{seq_len}_latency_ms"] = eff["latency_ms"]
            results[f"{seq_len}_throughput"] = eff["throughput"]
            results[f"{seq_len}_memory_gb"] = eff["peak_memory_gb"]
            results[f"{seq_len}_flops_g"] = flops / 1e9

        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                torch.cuda.empty_cache()
                results[f"{seq_len}_latency_ms"] = -1.0
                results[f"{seq_len}_throughput"] = -1.0
                results[f"{seq_len}_memory_gb"] = -1.0
                results[f"{seq_len}_flops_g"] = -1.0
            else:
                raise

    return results
