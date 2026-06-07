"""Random top-k sparse attention (control baseline)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing import RandomTopKRouting


class RandomTopKAttention(nn.Module):
    """Attention with random top-k routing (control baseline)."""

    def __init__(self, d_model: int, n_heads: int, k: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.k = k
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.routing = RandomTopKRouting(k)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        B, L, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        routing_mask = self.routing.compute_mask(scores)

        causal_mask = torch.tril(torch.ones(L, L, device=scores.device, dtype=torch.bool))
        combined_mask = routing_mask & causal_mask.unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(~combined_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)

        return output, {"routing_mask": routing_mask, "attn_type": "random_topk"}
