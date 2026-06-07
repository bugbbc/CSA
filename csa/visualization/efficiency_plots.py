"""Efficiency plots: latency, memory, FLOPs vs context length."""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .style import set_publication_style, save_figure, METHOD_COLORS, METHOD_MARKERS, METHOD_LABELS


def plot_latency_vs_length(
    results: Dict[str, Dict[int, float]],
    save_dir: str = "figures/efficiency",
    fmt: str = "png",
):
    """
    Plot inference latency vs sequence length for each method.

    Args:
        results: {method: {seq_length: latency_ms}}
    """
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        lengths = sorted(data.keys())
        latencies = [data[l] for l in lengths]
        if all(l > 0 for l in latencies):
            ax.plot(
                lengths, latencies,
                color=METHOD_COLORS.get(method, 'black'),
                marker=METHOD_MARKERS.get(method, 'o'),
                label=METHOD_LABELS.get(method, method),
                linewidth=2,
            )

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency vs Context Length')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks([2**i for i in range(11, 18)])  # 2K to 128K
    ax.set_xticklabels([f"{2**i // 1024}K" for i in range(11, 18)])

    plt.tight_layout()
    save_figure(fig, f"{save_dir}/latency_vs_length", [fmt])
    plt.close()


def plot_memory_vs_length(
    results: Dict[str, Dict[int, float]],
    save_dir: str = "figures/efficiency",
    fmt: str = "png",
):
    """Plot GPU memory vs sequence length."""
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        lengths = sorted(data.keys())
        memories = [data[l] for l in lengths]
        if all(m > 0 for m in memories):
            ax.plot(
                lengths, memories,
                color=METHOD_COLORS.get(method, 'black'),
                marker=METHOD_MARKERS.get(method, 'o'),
                label=METHOD_LABELS.get(method, method),
                linewidth=2,
            )

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('GPU Memory (GB)')
    ax.set_title('GPU Memory Usage vs Context Length')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks([2**i for i in range(11, 18)])
    ax.set_xticklabels([f"{2**i // 1024}K" for i in range(11, 18)])

    plt.tight_layout()
    save_figure(fig, f"{save_dir}/memory_vs_length", [fmt])
    plt.close()


def plot_throughput_vs_length(
    results: Dict[str, Dict[int, float]],
    save_dir: str = "figures/efficiency",
    fmt: str = "png",
):
    """Plot throughput vs sequence length."""
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        lengths = sorted(data.keys())
        throughputs = [data[l] for l in lengths]
        if all(t > 0 for t in throughputs):
            ax.plot(
                lengths, throughputs,
                color=METHOD_COLORS.get(method, 'black'),
                marker=METHOD_MARKERS.get(method, 'o'),
                label=METHOD_LABELS.get(method, method),
                linewidth=2,
            )

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Throughput (seq/s)')
    ax.set_title('Throughput vs Context Length')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks([2**i for i in range(11, 18)])
    ax.set_xticklabels([f"{2**i // 1024}K" for i in range(11, 18)])

    plt.tight_layout()
    save_figure(fig, f"{save_dir}/throughput_vs_length", [fmt])
    plt.close()


def plot_flops_vs_length(
    results: Dict[str, Dict[int, float]],
    save_dir: str = "figures/efficiency",
    fmt: str = "png",
):
    """Plot FLOPs vs sequence length."""
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        lengths = sorted(data.keys())
        flops = [data[l] for l in lengths]
        if all(f > 0 for f in flops):
            ax.plot(
                lengths, flops,
                color=METHOD_COLORS.get(method, 'black'),
                marker=METHOD_MARKERS.get(method, 'o'),
                label=METHOD_LABELS.get(method, method),
                linewidth=2,
            )

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('FLOPs (G)')
    ax.set_title('Computational Cost vs Context Length')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks([2**i for i in range(11, 18)])
    ax.set_xticklabels([f"{2**i // 1024}K" for i in range(11, 18)])

    plt.tight_layout()
    save_figure(fig, f"{save_dir}/flops_vs_length", [fmt])
    plt.close()
