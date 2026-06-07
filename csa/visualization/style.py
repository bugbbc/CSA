"""Matplotlib style configuration for publication-quality figures."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use Type 1 fonts (required for many conferences)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = False  # Set to True if LaTeX is available

# Font sizes
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14

# Figure
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

# Lines
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 6

# Grid
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'

# Color scheme (colorblind-friendly)
COLORS = {
    'blue': '#0072B2',
    'red': '#D55E00',
    'green': '#009E73',
    'orange': '#E69F00',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'pink': '#F0E442',
    'grey': '#999999',
}

METHOD_COLORS = {
    'dense': COLORS['blue'],
    'local_window': COLORS['green'],
    'similarity_topk': COLORS['orange'],
    'random_topk': COLORS['grey'],
    'gated': COLORS['purple'],
    'gated_sparse': COLORS['cyan'],
    'csa': COLORS['red'],
    'csa_exact': COLORS['pink'],
}

METHOD_MARKERS = {
    'dense': 'o',
    'local_window': 's',
    'similarity_topk': '^',
    'random_topk': 'v',
    'gated': 'D',
    'gated_sparse': 'P',
    'csa': '*',
    'csa_exact': 'X',
}

METHOD_LABELS = {
    'dense': 'Dense',
    'local_window': 'Local Window',
    'similarity_topk': 'Similarity Top-k',
    'random_topk': 'Random Top-k',
    'gated': 'Gated',
    'gated_sparse': 'Gated Sparse',
    'csa': 'CSA (Ours)',
    'csa_exact': 'CSA Exact',
}


def set_publication_style():
    """Apply publication-quality style to all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update(mpl.rcParams)  # Apply our custom settings


def save_figure(fig, path: str, formats: list = ['pdf', 'png']):
    """Save figure in multiple formats."""
    import os
    for fmt in formats:
        save_path = f"{path}.{fmt}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
