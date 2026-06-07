"""Attention module factory — updated with causal gating variants."""

from .dense import DenseAttention
from .local_window import LocalWindowSparseAttention
from .random_topk import RandomTopKAttention
from .similarity_topk import SimilarityTopKAttention
from .gated import GatedAttention, GatedSparseAttention
from .csa import CSAAttention, CSAExactAttention
from .causal_gated import CausalGatedAttention, CausalGatedSparseAttention


def build_attention(attn_type: str, d_model: int = 512, n_heads: int = 8,
                    window: int = 128, k: int = 32, refresh_interval: int = 4,
                    baseline_type: str = "zero", dropout: float = 0.1,
                    vocab_size: int = 97, pad_token_id: int = 0):
    if attn_type == "dense":
        return DenseAttention(d_model, n_heads, dropout)
    elif attn_type == "local_window":
        return LocalWindowSparseAttention(d_model, n_heads, window, dropout)
    elif attn_type == "random_topk":
        return RandomTopKAttention(d_model, n_heads, k, dropout)
    elif attn_type == "similarity_topk":
        return SimilarityTopKAttention(d_model, n_heads, window, k, dropout)
    elif attn_type == "gated":
        return GatedAttention(d_model, n_heads, k, window, dropout)
    elif attn_type == "gated_sparse":
        return GatedSparseAttention(d_model, n_heads, window, k, dropout)
    elif attn_type == "csa":
        return CSAAttention(d_model, n_heads, window, k, refresh_interval,
                            baseline_type, dropout, vocab_size, pad_token_id)
    elif attn_type == "csa_exact":
        return CSAExactAttention(d_model, n_heads, window, k, refresh_interval,
                                 baseline_type, dropout, vocab_size, pad_token_id)
    elif attn_type == "causal_gated":
        return CausalGatedAttention(d_model, n_heads, temperature=1.0,
                                    refresh_interval=refresh_interval,
                                    baseline_type=baseline_type, dropout=dropout,
                                    vocab_size=vocab_size, pad_token_id=pad_token_id)
    elif attn_type == "causal_gated_sparse":
        return CausalGatedSparseAttention(d_model, n_heads, window=window, k=k,
                                           temperature=1.0,
                                           refresh_interval=refresh_interval,
                                           baseline_type=baseline_type, dropout=dropout,
                                           vocab_size=vocab_size, pad_token_id=pad_token_id)
    raise ValueError(f"Unknown attention type: {attn_type}")


__all__ = [
    "DenseAttention", "LocalWindowSparseAttention", "RandomTopKAttention",
    "SimilarityTopKAttention", "GatedAttention", "GatedSparseAttention",
    "CSAAttention", "CSAExactAttention",
    "CausalGatedAttention", "CausalGatedSparseAttention",
    "build_attention",
]
