"""
Causal Gating Attention variants for the routing-vs-weighting experiment.

Variant A: CausalGatedAttention — CSA scores as soft post-hoc weight multipliers over dense attention.
Variant B: CausalGatedSparseAttention — Standard sparse support + causal gates over surviving edges.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing import LocalWindowPlusTopKRouting
from .contrib import GradientProxyEstimator


class CausalGatedAttention(nn.Module):
    """
    Variant A: Use CSA contribution scores as soft post-hoc gates (weights)
    applied over dense attention. This tests whether causal info helps when
    used as a weighting signal rather than routing signal.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        temperature: float = 1.0,
        refresh_interval: int = 4,
        baseline_type: str = "zero",
        dropout: float = 0.1,
        vocab_size: int = 97,
        pad_token_id: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.temperature = temperature
        self.refresh_interval = refresh_interval
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.contrib_estimator = GradientProxyEstimator(
            baseline_type=baseline_type, vocab_size=vocab_size, pad_token_id=pad_token_id)

        self.register_buffer("_step_counter", torch.zeros(1, dtype=torch.long))
        self._cached_contrib: Optional[torch.Tensor] = None
        object.__setattr__(self, "_full_model_ref", None)

    def set_full_model(self, model):
        object.__setattr__(self, "_full_model_ref", model)

    def _should_refresh(self) -> bool:
        if self._cached_contrib is None:
            return True
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

        # Full dense attention (unmasked)
        sim_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        causal_mask = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool))
        scores = sim_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(sim_scores.dtype)

        # --- Get causal contribution scores ---
        in_contrib_pass = False
        if self._full_model_ref is not None:
            in_contrib_pass = getattr(self._full_model_ref, '_in_contrib_pass', False)

        if self._cached_contrib is not None and self._cached_contrib.shape[-1] != L:
            self._cached_contrib = None

        if in_contrib_pass:
            contrib = (self._cached_contrib if self._cached_contrib is not None
                       else torch.ones(B, L, device=hidden_states.device) / L)
        elif (self._should_refresh() and self._full_model_ref is not None
              and labels is not None and input_ids is not None):
            contrib = self.contrib_estimator.compute(
                model=self._full_model_ref, input_ids=input_ids,
                attention_mask=attention_mask, labels=labels)
            self._cached_contrib = contrib.detach()
        elif self._cached_contrib is None:
            contrib = torch.ones(B, L, device=hidden_states.device) / L
        else:
            contrib = self._cached_contrib

        # --- Apply as soft gates over dense attention ---
        # Normalize contribution scores: sigmoid(temperature * z-score)
        contrib_std = contrib.std(dim=-1, keepdim=True) + 1e-8
        contrib_mean = contrib.mean(dim=-1, keepdim=True)
        contrib_norm = (contrib - contrib_mean) / contrib_std
        gates = torch.sigmoid(self.temperature * contrib_norm)  # [B, L]

        # Gates applied per key position: [B, L] -> [B, 1, 1, L]
        weight_mask = gates.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        # Apply soft weighting to attention weights
        weighted_attn = attn_weights * weight_mask
        # Renormalize
        weighted_attn = weighted_attn / (weighted_attn.sum(dim=-1, keepdim=True) + 1e-8)

        weighted_attn = F.dropout(weighted_attn, p=self.dropout, training=self.training)

        output = torch.matmul(weighted_attn, v)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)

        self._step_counter += 1

        aux = {
            "contrib_scores": contrib.detach() if isinstance(contrib, torch.Tensor) else contrib,
            "gates": gates.detach(),
            "attn_type": "causal_gated",
        }
        return output, aux


class CausalGatedSparseAttention(nn.Module):
    """
    Variant B: Standard sparse support (local window + similarity top-k),
    then apply causal gates over the surviving edges.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window: int = 128,
        k: int = 32,
        temperature: float = 1.0,
        refresh_interval: int = 4,
        baseline_type: str = "zero",
        dropout: float = 0.1,
        vocab_size: int = 97,
        pad_token_id: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.k = k
        self.temperature = temperature
        self.refresh_interval = refresh_interval
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Standard sparse routing (similarity-based, same as SimilarityTopK)
        self.routing = LocalWindowPlusTopKRouting(window, k)
        self.contrib_estimator = GradientProxyEstimator(
            baseline_type=baseline_type, vocab_size=vocab_size, pad_token_id=pad_token_id)

        self.register_buffer("_step_counter", torch.zeros(1, dtype=torch.long))
        self._cached_contrib: Optional[torch.Tensor] = None
        object.__setattr__(self, "_full_model_ref", None)

    def set_full_model(self, model):
        object.__setattr__(self, "_full_model_ref", model)

    def _should_refresh(self) -> bool:
        if self._cached_contrib is None:
            return True
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

        # Standard similarity-based sparse routing
        routing_mask = self.routing.compute_mask(sim_scores)

        # --- Get causal contribution scores ---
        in_contrib_pass = False
        if self._full_model_ref is not None:
            in_contrib_pass = getattr(self._full_model_ref, '_in_contrib_pass', False)

        if self._cached_contrib is not None and self._cached_contrib.shape[-1] != L:
            self._cached_contrib = None

        if in_contrib_pass:
            contrib = (self._cached_contrib if self._cached_contrib is not None
                       else torch.ones(B, L, device=hidden_states.device) / L)
        elif (self._should_refresh() and self._full_model_ref is not None
              and labels is not None and input_ids is not None):
            contrib = self.contrib_estimator.compute(
                model=self._full_model_ref, input_ids=input_ids,
                attention_mask=attention_mask, labels=labels)
            self._cached_contrib = contrib.detach()
        elif self._cached_contrib is None:
            contrib = torch.ones(B, L, device=hidden_states.device) / L
        else:
            contrib = self._cached_contrib

        # --- Create causal gates ---
        contrib_std = contrib.std(dim=-1, keepdim=True) + 1e-8
        contrib_mean = contrib.mean(dim=-1, keepdim=True)
        contrib_norm = (contrib - contrib_mean) / contrib_std
        gates = torch.sigmoid(self.temperature * contrib_norm)  # [B, L]

        # Apply sparse routing + causal mask
        causal_mask = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool))
        combined_mask = routing_mask & causal_mask.unsqueeze(0).unsqueeze(0)

        masked_scores = sim_scores.masked_fill(~combined_mask, float('-inf'))
        sparse_attn = F.softmax(masked_scores, dim=-1, dtype=torch.float32).to(sim_scores.dtype)

        # Apply causal gates as post-hoc weighting over surviving sparse edges
        weight_mask = gates.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        weighted_attn = sparse_attn * weight_mask
        weighted_attn = weighted_attn / (weighted_attn.sum(dim=-1, keepdim=True) + 1e-8)
        weighted_attn = F.dropout(weighted_attn, p=self.dropout, training=self.training)

        output = torch.matmul(weighted_attn, v)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)

        self._step_counter += 1

        aux = {
            "routing_mask": routing_mask,
            "contrib_scores": contrib.detach() if isinstance(contrib, torch.Tensor) else contrib,
            "gates": gates.detach(),
            "attn_type": "causal_gated_sparse",
        }
        return output, aux
