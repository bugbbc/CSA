"""Gated attention variants: learned gating for global token selection."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """
    Gated attention with learned gating for global token selection.

    Uses a learned gating network to score each (query, key) pair and selects
    top-k keys per query. Combined with local window attention.
    """

    def __init__(self, d_model: int, n_heads: int, k: int, window: int = 0, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.k = k
        self.window = window
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learned gating: projects hidden states to gate logits per head
        self.gate_q = nn.Linear(d_model, n_heads * k, bias=False)
        self.gate_k = nn.Linear(d_model, n_heads * k, bias=False)

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

        # Full similarity scores (needed for attention weights after routing)
        sim_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Learned gating scores: compute per-head gate logits for queries and keys
        gq = self.gate_q(hidden_states).view(B, L, self.n_heads, self.k)  # [B, L, H, k]
        gk = self.gate_k(hidden_states).view(B, L, self.n_heads, self.k)  # [B, L, H, k]

        # Compute gating affinity: how much each query-head wants each key
        # gq: [B, L_q, H, k], gk: [B, L_k, H, k]
        # The k dimension must match: transpose gk to [B, H, k, L_k] and multiply
        gk_t = gk.permute(0, 2, 3, 1)  # [B, H, k, L_k]
        gq_t = gq.permute(0, 2, 1, 3)  # [B, H, L_q, k]
        gate_scores = torch.matmul(gq_t, gk_t) / (self.k ** 0.5)  # [B, H, L_q, L_k]

        # Build routing mask: local window + top-k by gate scores
        routing_mask = torch.zeros(B, 1, L, L, dtype=torch.bool, device=hidden_states.device)

        if self.window > 0:
            q_idx = torch.arange(L, device=hidden_states.device).view(-1, 1)
            k_idx = torch.arange(L, device=hidden_states.device).view(1, -1)
            local_mask = (q_idx - k_idx).abs() < (self.window // 2)
            routing_mask = routing_mask | local_mask.unsqueeze(0).unsqueeze(0)

        # Top-k by learned gate scores
        k_actual = min(self.k, L)
        # Average gate scores across heads for routing
        gate_avg = gate_scores.mean(dim=1, keepdim=True)  # [B, 1, L_q, L_k]
        _, gate_indices = torch.topk(gate_avg, k_actual, dim=-1)
        routing_mask.scatter_(-1, gate_indices.expand(-1, 1, L, -1), True)

        causal_mask = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool))
        combined_mask = routing_mask & causal_mask.unsqueeze(0).unsqueeze(0)

        scores = sim_scores.masked_fill(~combined_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)

        aux = {
            "routing_mask": routing_mask,
            "gate_scores": gate_scores,
            "attn_type": "gated",
        }
        return output, aux


class GatedSparseAttention(nn.Module):
    """Gated attention with local window + learned sparse gating."""

    def __init__(self, d_model: int, n_heads: int, window: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.gated = GatedAttention(d_model, n_heads, k, window, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        output, aux = self.gated(hidden_states, attention_mask, use_cache)
        if aux:
            aux["attn_type"] = "gated_sparse"
        return output, aux
