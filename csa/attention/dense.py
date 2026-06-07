"""Dense (full) attention with manual implementation for maximum compatibility."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAttention(nn.Module):
    """Standard dense softmax attention with combined causal + padding masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, 1, 1, L] or None (padding mask)
        Returns:
            output: [B, L, D]
            aux: dict with attention weights
        """
        B, L, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]

        # Build combined mask: causal + padding
        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=scores.device))
        combined = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        if attention_mask is not None:
            # attention_mask is [B, 1, 1, L] — broadcast to [B, 1, L, L]
            combined = combined & attention_mask

        scores = scores.masked_fill(~combined, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)

        return output, {"attn_type": "dense"}
