"""Metrics plots: ER@k, SI@k, accuracy vs context length."""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import set_publication_style, save_figure, METHOD_COLORS, METHOD_MARKERS, METHOD_LABELS


def plot_erk_curves(
    results: Dict[str, Dict[int, float]],
    save_dir: str = "figures/metrics",
    fmt: str = "png",
):
    """
    Plot Evidence Recall @ k curves.

    Args:
        results: {method: {k: er_at_k}}
    """
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        k_values = sorted(data.keys())
        er_values = [data[k] for k in k_values]
        ax.plot(
            k_values, er_values,
            color=METHOD_COLORS.get(method, 'black'),
            marker=METHOD_MARKERS.get(method, 'o'),
            label=METHOD_LABELS.get(method, method),
            linewidth=2,
        )

    ax.set_xlabel('Budget k')
    ax.set_ylabel('Evidence Recall ER@k')
    ax.set_title('Evidence Recall vs Attention Budget')
    ax.legend()
    ax.set_xticks(sorted(set().union(*[data.keys() for data in results.values()])))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, f"{save_dir}/erk_curves", [fmt])
    plt.close()


def plot_sik_curves(
    results: Dict[str, Dict[int, float]],
    save_dir: str = "figures/metrics",
    fmt: str = "png",
):
    """
    Plot Spurious Inclusion @ k curves.

    Args:
        results: {method: {k: si_at_k}}
    """
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        k_values = sorted(data.keys())
        si_values = [data[k] for k in k_values]
        ax.plot(
            k_values, si_values,
            color=METHOD_COLORS.get(method, 'black'),
            marker=METHOD_MARKERS.get(method, 'o'),
            label=METHOD_LABELS.get(method, method),
            linewidth=2,
        )

    ax.set_xlabel('Budget k')
    ax.set_ylabel('Spurious Inclusion SI@k')
    ax.set_title('Spurious Inclusion vs Attention Budget')
    ax.legend()
    ax.set_xticks(sorted(set().union(*[data.keys() for data in results.values()])))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, f"{save_dir}/sik_curves", [fmt])
    plt.close()


def plot_accuracy_vs_context_length(
    results: Dict[str, Dict[int, float]],
    ylabel: str = "Accuracy",
    title: str = "Accuracy vs Context Length",
    save_dir: str = "figures/metrics",
    fmt: str = "png",
):
    """
    Plot accuracy vs context length.

    Args:
        results: {method: {context_length: accuracy}}
    """
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        lengths = sorted(data.keys())
        accuracies = [data[l] for l in lengths]
        ax.plot(
            lengths, accuracies,
            color=METHOD_COLORS.get(method, 'black'),
            marker=METHOD_MARKERS.get(method, 'o'),
            label=METHOD_LABELS.get(method, method),
            linewidth=2,
        )

    ax.set_xlabel('Context Length')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks(list(lengths))  # Use actual tick positions
    ax.set_xticklabels([f"{l // 1024}K" if l >= 1024 else str(l) for l in lengths])

    plt.tight_layout()
    save_figure(fig, f"{save_dir}/{title.lower().replace(' ', '_')}", [fmt])
    plt.close()
