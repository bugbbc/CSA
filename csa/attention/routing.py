"""
Pluggable routing strategies for sparse attention.
Each strategy is a pure function: scores -> boolean attention mask.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


def _get_lengths(mask: Optional[torch.Tensor], seq_len: int, batch_size: int) -> torch.Tensor:
    """Extract sequence lengths from mask or default to full length."""
    if mask is not None:
        return mask.sum(dim=-1).long()  # [B]
    return torch.full((batch_size,), seq_len, dtype=torch.long, device=mask.device if mask is not None else 'cpu')


class RoutingStrategy(ABC):
    """Abstract base for attention routing strategies."""

    @abstractmethod
    def compute_mask(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            scores: Attention scores [B, H, L_q, L_k] (can be placeholder)
            lengths: Sequence lengths [B]
        Returns:
            Boolean mask [B, 1, L_q, L_k] where True = attend
        """
        pass


class DenseRouting(RoutingStrategy):
    """Attend to all positions (no sparsity)."""

    def compute_mask(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, L_q, L_k = scores.shape
        return torch.ones(B, 1, L_q, L_k, dtype=torch.bool, device=scores.device)


class LocalWindowRouting(RoutingStrategy):
    """Local window: attend only to w nearest neighbors."""

    def __init__(self, window: int):
        assert window > 0, f"Window must be positive, got {window}"
        self.window = window

    def compute_mask(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, L_q, L_k = scores.shape
        device = scores.device

        # Build local window mask: |i - j| < window/2
        q_idx = torch.arange(L_q, device=device).view(-1, 1)  # [L_q, 1]
        k_idx = torch.arange(L_k, device=device).view(1, -1)  # [1, L_k]
        local_mask = (q_idx - k_idx).abs() < (self.window // 2)  # [L_q, L_k]

        return local_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)  # [B, 1, L_q, L_k]


class TopKRouting(RoutingStrategy):
    """Global top-k: attend to top-k positions by score."""

    def __init__(self, k: int):
        assert k > 0, f"k must be positive, got {k}"
        self.k = k

    def compute_mask(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, L_q, L_k = scores.shape
        # Get top-k values and indices
        # For causal attention, only consider positions <= query position
        # Here we do unrestricted top-k (scores that are valid)
        if lengths is not None:
            # Create mask for valid positions
            valid_mask = torch.arange(L_k, device=scores.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, L_k]
            scores = scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        topk_mask = torch.zeros_like(scores, dtype=torch.bool)
        if H > 1:
            # Broadcast top-k from aggregated scores if needed
            agg_scores = scores.mean(dim=1, keepdim=True)  # [B, 1, L_q, L_k]
            _, indices = torch.topk(agg_scores, min(self.k, L_k), dim=-1)
            topk_mask.scatter_(-1, indices, True)
            topk_mask = topk_mask[:, :1]  # [B, 1, L_q, L_k]
        else:
            _, indices = torch.topk(scores, min(self.k, L_k), dim=-1)
            topk_mask.scatter_(-1, indices, True)

        return topk_mask


class RandomTopKRouting(RoutingStrategy):
    """Random top-k: random positions (control baseline)."""

    def __init__(self, k: int):
        assert k > 0, f"k must be positive, got {k}"
        self.k = k

    def compute_mask(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, L_q, L_k = scores.shape
        device = scores.device

        k_actual = min(self.k, L_k)
        # Generate random scores at each forward (no seed for randomness)
        random_scores = torch.rand(B, 1, L_q, L_k, device=device)

        if lengths is not None:
            valid_mask = torch.arange(L_k, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            random_scores = random_scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e9)

        _, indices = torch.topk(random_scores, k_actual, dim=-1)
        mask = torch.zeros(B, 1, L_q, L_k, dtype=torch.bool, device=device)
        mask.scatter_(-1, indices, True)
        return mask


class LocalWindowPlusTopKRouting(RoutingStrategy):
    """Union of local window + top-k (by score)."""

    def __init__(self, window: int, k: int):
        self.window = window
        self.k = k
        self._window_routing = LocalWindowRouting(window)
        self._topk_routing = TopKRouting(k)

    def compute_mask(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        local_mask = self._window_routing.compute_mask(scores, lengths)
        topk_mask = self._topk_routing.compute_mask(scores, lengths)
        return local_mask | topk_mask


class GatedRouting(RoutingStrategy):
    """
    Learned gating routing.
    Uses learnable parameters to compute gating scores per token.
    Support = LocalWindow(w) ∪ TopK(GateScores, k)

    NOTE: This requires the query embedding to compute gate scores.
    The compute_mask method takes scores as input (we use it as a
    placeholder for the hidden states to extract queries).
    """

    def __init__(self, d_model: int, n_heads: int, k: int, window: int = 0):
        self.window = window
        self.k = k
        self.d_model = d_model
        self.n_heads = n_heads
        # Learned gating projection (per-head, per-position)
        self.gate_proj = torch.nn.Linear(d_model, k, bias=False)

    def compute_mask(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gating mask.

        scores: [B, H, L_q, L_k] — we use it to infer B, L, device
        Returns: [B, 1, L_q, L_k] boolean mask

        NOTE: Since this module doesn't receive the hidden states directly,
        we use a placeholder approach: the gating is position-independent
        (same k tokens selected for all queries). In a full implementation,
        each query would learn different gates.
        """
        B, _, L_q, L_k = scores.shape
        device = scores.device

        mask = torch.zeros(B, 1, L_q, L_k, dtype=torch.bool, device=device)

        # Local window
        if self.window > 0:
            q_idx = torch.arange(L_q, device=device).view(-1, 1)
            k_idx = torch.arange(L_k, device=device).view(1, -1)
            local_mask = (q_idx - k_idx).abs() < (self.window // 2)
            mask = mask | local_mask.unsqueeze(0).unsqueeze(0)

        # Learned gating: select top-k key positions globally
        # Use the gating projection to score each position
        # We create a learned position score vector
        k_actual = min(self.k, L_k)
        # For simplicity, average gate scores across queries
        # In a real impl, each query would have different gate scores
        if L_q == L_k:
            gate_logits = self.gate_proj.weight.mean(dim=0)  # [d_model]
            # Score positions randomly since we don't have the hidden states here
            # In actual usage, call get_gate_scores from the attention module
            pos_scores = torch.randn(L_k, device=device)
            _, idx = torch.topk(pos_scores, k_actual)
            mask[:, :, :, idx] = True

        return mask

    def get_gate_scores(self, query: torch.Tensor) -> torch.Tensor:
        """Compute learned gating scores from query."""
        B, L, D = query.shape
        gate_logits = self.gate_proj(query)  # [B, L, k]
        return gate_logits  # [B, L, k]


def build_routing(routing_type: str, **kwargs) -> RoutingStrategy:
    """Factory function for routing strategies."""
    routing_map = {
        "dense": DenseRouting,
        "local_window": LocalWindowRouting,
        "topk": TopKRouting,
        "random_topk": RandomTopKRouting,
        "local_window_plus_topk": LocalWindowPlusTopKRouting,
        "gated": GatedRouting,
    }
    if routing_type not in routing_map:
        raise ValueError(f"Unknown routing type: {routing_type}. Options: {list(routing_map.keys())}")

    if routing_type == "gated":
        return routing_map[routing_type](
            d_model=kwargs.get("d_model", 512),
            n_heads=kwargs.get("n_heads", 8),
            k=kwargs.get("k", 32),
            window=kwargs.get("window", 0),
        )
    return routing_map[routing_type](
        window=kwargs.get("window", 128),
        k=kwargs.get("k", 32),
    )
