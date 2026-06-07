"""
CSA (Causal Sparse Attention) - Contribution-Guided Sparse Routing.

Support(q_i) = LocalWindow(q_i, w) ∪ TopK(C_hat, k)
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing import LocalWindowPlusTopKRouting
from .contrib import GradientProxyEstimator


class CSAAttention(nn.Module):
    """
    Causal Sparse Attention with gradient-proxy contribution estimation.
    Support(q_i) = LocalWindow(q_i, w) ∪ TopK(C_hat, k)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window: int = 128,
        k: int = 64,
        refresh_interval: int = 4,
        baseline_type: str = "zero",
        dropout: float = 0.1,
        vocab_size: int = 50257,
        pad_token_id: int = 50256,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.k = k
        self.refresh_interval = refresh_interval
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.routing = LocalWindowPlusTopKRouting(window, k)
        self.contrib_estimator = GradientProxyEstimator(
            baseline_type=baseline_type, vocab_size=vocab_size, pad_token_id=pad_token_id)

        self.register_buffer("_step_counter", torch.zeros(1, dtype=torch.long))
        self._contrib_valid = False
        self._cached_contrib: Optional[torch.Tensor] = None
        object.__setattr__(self, "_full_model_ref", None)

    def set_full_model(self, model):
        object.__setattr__(self, "_full_model_ref", model)

    def _should_refresh(self) -> bool:
        if not self._contrib_valid:
            return True
        # During eval, don't refresh; use cached scores
        if not self.training:
            return False
        return self._step_counter.item() % self.refresh_interval == 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        B, L, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        sim_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Handle contribution scores with recursion guard
        in_contrib_pass = False
        if self._full_model_ref is not None:
            in_contrib_pass = getattr(self._full_model_ref, '_in_contrib_pass', False)

        # Invalidate cache on any shape mismatch
        if self._cached_contrib is not None:
            if self._cached_contrib.shape[-1] != L or self._cached_contrib.shape[0] != B:
                self._cached_contrib = None
                self._contrib_valid = False

        if in_contrib_pass:
            contrib_scores = (self._cached_contrib if self._cached_contrib is not None
                              else torch.ones(B, L, device=hidden_states.device) / L)
        elif (self._should_refresh() and self._full_model_ref is not None
              and labels is not None and input_ids is not None):
            contrib_scores = self.contrib_estimator.compute(
                model=self._full_model_ref, input_ids=input_ids,
                attention_mask=attention_mask, labels=labels)
            self._cached_contrib = contrib_scores.detach()
            self._contrib_valid = True
        elif self._cached_contrib is None:
            contrib_scores = torch.ones(B, L, device=hidden_states.device) / L
        else:
            contrib_scores = self._cached_contrib

        # Expand for routing
        if contrib_scores.dim() == 2:
            contrib_for_routing = contrib_scores.unsqueeze(1).unsqueeze(2).expand(-1, 1, L, -1)
        elif contrib_scores.dim() == 3 and contrib_scores.shape[1] == 1:
            contrib_for_routing = contrib_scores.unsqueeze(2).expand(-1, -1, L, -1)
        else:
            contrib_for_routing = contrib_scores

        routing_mask = self.routing.compute_mask(contrib_for_routing)
        causal_mask = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool))
        combined_mask = routing_mask & causal_mask.unsqueeze(0).unsqueeze(0)

        masked_scores = sim_scores.masked_fill(~combined_mask, float('-inf'))
        attn_weights = F.softmax(masked_scores, dim=-1, dtype=torch.float32).to(sim_scores.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)

        self._step_counter += 1

        aux = {
            "routing_mask": routing_mask,
            "contrib_scores": contrib_scores.detach() if isinstance(contrib_scores, torch.Tensor) else contrib_scores,
            "attn_type": "csa",
            "n_selected": routing_mask.float().sum(dim=-1).mean().item(),
        }
        return output, aux

    def get_contrib_scores(self) -> Optional[torch.Tensor]:
        if self._contrib_valid:
            return self._cached_contrib
        return None


class CSAExactAttention(CSAAttention):
    """CSA with exact intervention-based contribution estimation (short sequences only)."""

    def __init__(self, d_model: int, n_heads: int, window: int = 128, k: int = 32,
                 refresh_interval: int = 4, baseline_type: str = "zero",
                 dropout: float = 0.1, vocab_size: int = 50257, pad_token_id: int = 50256):
        from .contrib import ExactInterventionEstimator
        super().__init__(d_model, n_heads, window, k, refresh_interval,
                         baseline_type, dropout, vocab_size, pad_token_id)
        self.contrib_estimator = ExactInterventionEstimator(
            baseline_type=baseline_type, vocab_size=vocab_size, pad_token_id=pad_token_id)
