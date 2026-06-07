"""Attention map visualization."""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .style import set_publication_style, save_figure, METHOD_COLORS, METHOD_LABELS


def plot_attention_maps(
    attention_weights: Dict[str, np.ndarray],
    save_dir: str = "figures/attention",
    fmt: str = "png",
):
    """
    Plot attention maps for different methods.

    Args:
        attention_weights: {method: np.ndarray [H, L, L]}
    """
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    for method_name, attn in attention_weights.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for h in range(min(attn.shape[0], 8)):
            im = axes[h].imshow(
                attn[h], cmap='Blues', aspect='auto',
                vmin=0, vmax=attn[h].max() if attn[h].max() > 0 else 1,
            )
            axes[h].set_title(f'Head {h}')
            axes[h].set_xlabel('Key')
            axes[h].set_ylabel('Query')

        plt.suptitle(f'Attention Maps: {METHOD_LABELS.get(method_name, method_name)}')
        plt.tight_layout()
        save_figure(fig, f"{save_dir}/attention_{method_name}", [fmt])
        plt.close()


def plot_needle_heatmaps(
    depth_results: Dict[str, np.ndarray],
    context_lengths: List[int],
    depths: List[float],
    save_dir: str = "figures/needle",
    fmt: str = "png",
):
    """
    Plot needle-in-a-haystack heatmaps showing accuracy across depths and lengths.

    Args:
        depth_results: {method: np.ndarray [len(lengths), len(depths)]}
    """
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(depth_results), figsize=(5 * len(depth_results), 4))
    if len(depth_results) == 1:
        axes = [axes]

    for ax, (method, data) in zip(axes, depth_results.items()):
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{int(d * 100)}%" for d in depths])
        ax.set_yticks(range(len(context_lengths)))
        ax.set_yticklabels([f"{cl // 1024}K" for cl in context_lengths])
        ax.set_xlabel('Needle Depth')
        ax.set_ylabel('Context Length')
        ax.set_title(METHOD_LABELS.get(method, method))

        # Add text annotations
        for i in range(len(context_lengths)):
            for j in range(len(depths)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center',
                       fontsize=8, color='black' if data[i, j] > 0.5 else 'white')

    plt.suptitle('Needle-in-a-Haystack: Retrieval Accuracy')
    plt.tight_layout()
    save_figure(fig, f"{save_dir}/needle_heatmap", [fmt])
    plt.close()


def plot_evidence_selection_heatmap(
    contrib_scores: np.ndarray,
    token_labels: Optional[List[str]] = None,
    save_path: str = "figures/evidence_selection.png",
    fmt: str = "png",
):
    """
    Plot heatmap showing which tokens are selected by CSA.

    Args:
        contrib_scores: [L] contribution scores per token position
    """
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 3))

    L = len(contrib_scores)
    im = ax.imshow(contrib_scores.reshape(1, -1), cmap='Reds', aspect='auto')

    ax.set_yticks([])
    ax.set_xlabel('Token Position')
    ax.set_title('CSA Token Contribution Scores')

    if token_labels:
        ax.set_xticks(range(L))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=6)

    plt.colorbar(im, ax=ax, label='Contribution Score')
    plt.tight_layout()
    save_figure(fig, save_path.replace(f".{fmt}", ""), [fmt])
    plt.close()
