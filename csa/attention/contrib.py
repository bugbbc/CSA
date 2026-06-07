"""
Contribution estimators for Causal Sparse Attention.

GradientProxyEstimator: C_hat_j = |⟨∇_{x_j}L, x_j - x̃_j⟩|
ExactInterventionEstimator: C_exact_j = KL(f(X), f(X^{do(j)}))
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContributionEstimator(ABC):
    """Abstract base for contribution score estimators."""

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns:
            Contribution scores [B, L] (token-level scores).
        """
        pass


class GradientProxyEstimator(ContributionEstimator):
    """
    First-order Taylor approximation of token contribution.

    C_hat_j = |⟨∇_{x_j}L, x_j - x̃_j⟩|

    Uses backward hooks on the embedding layer to capture input gradients.
    Designed to be called periodically every r steps.
    """

    def __init__(
        self,
        baseline_type: str = "zero",
        vocab_size: int = 50257,
        pad_token_id: int = 50256,
    ):
        self.baseline_type = baseline_type  # "zero", "mask", "mean"
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self._cached_grad: Optional[torch.Tensor] = None
        self._cached_input: Optional[torch.Tensor] = None
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _get_baseline(
        self, x: torch.Tensor, embed_layer: nn.Embedding
    ) -> torch.Tensor:
        """Get baseline embedding for each position."""
        B, L, D = x.shape
        device = x.device

        if self.baseline_type == "zero":
            return torch.zeros_like(x)

        elif self.baseline_type == "mask":
            # Use pad_token embedding as baseline
            return embed_layer(torch.full((B, L), self.pad_token_id, device=device))

        elif self.baseline_type == "mean":
            return x.mean(dim=1, keepdim=True).expand_as(x)

        else:
            raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

    def _hook_fn(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward hook that captures input gradients."""
        self._cached_grad = grad.detach().clone()
        return grad  # pass through

    def compute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute gradient-based contribution proxy.

        Note: Does NOT change model's training mode to avoid
        recursion issues with circular module references.

        Args:
            model: The full model (must have get_input_embeddings or embedding accessor)
            input_ids: [B, L]
            attention_mask: [B, L] or None
            labels: [B, L] for LM loss
        Returns:
            C_hat: [B, L] contribution scores
        """
        was_training = model.training
        # Try to set eval mode; if circular ref causes recursion, skip
        if not hasattr(model, '_module_recursion_guard'):
            try:
                model.eval()
            except RecursionError:
                pass
        B, L = input_ids.shape
        device = input_ids.device

        # Get embedding layer
        embed_layer = model.get_input_embeddings()

        # Use raw token embeddings only.
        # forward_with_embeddings applies embed_pos internally.
        with torch.no_grad():
            x = embed_layer(input_ids)  # [B, L, D]
        x = x.detach().clone().requires_grad_(True)

        # Register hook
        self._cached_grad = None
        self._cached_input = x.detach().clone()
        handle = x.register_hook(self._hook_fn)

        # Forward pass with the hooked embeddings
        # We need to call model with pre-computed embeddings
        output = model.forward_with_embeddings(x, attention_mask, labels)
        loss = output["loss"] if isinstance(output, dict) else output.loss

        # Backward to capture gradients
        loss.backward(retain_graph=False)
        handle.remove()

        if self._cached_grad is None:
            raise RuntimeError("Hook did not capture gradient")

        grad_x = self._cached_grad  # [B, L, D]
        x_tilde = self._get_baseline(x.detach(), embed_layer)

        # First-order contribution: |⟨grad, x - x_tilde⟩|
        diff = x.detach() - x_tilde  # [B, L, D]
        inner_product = (grad_x * diff).sum(dim=-1)  # [B, L]
        C_hat = inner_product.abs()  # [B, L]

        model.zero_grad()
        return C_hat


class ExactInterventionEstimator(ContributionEstimator):
    """
    Exact intervention-based contribution.

    For each position j, replaces x_j with baseline and measures
    KL(f(X) || f(X_{do(j)})) where do(j) means replacing token at position j.

    Complexity: O(L) forward passes. Only for short sequences.
    """

    def __init__(
        self,
        baseline_type: str = "zero",
        vocab_size: int = 50257,
        pad_token_id: int = 50256,
    ):
        self.baseline_type = baseline_type
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def _get_baseline_embedding(
        self, embed_layer: nn.Embedding, device: torch.device
    ) -> torch.Tensor:
        """Get a single baseline embedding vector [D]."""
        if self.baseline_type == "zero":
            return torch.zeros(embed_layer.embedding_dim, device=device)
        elif self.baseline_type == "mask":
            return embed_layer(torch.tensor(self.pad_token_id, device=device))
        else:
            # mean of all token embeddings
            return embed_layer.weight.mean(dim=0)

    @torch.no_grad()
    def compute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute exact intervention importance for each position.

        Args:
            model: Model with forward_with_embeddings method
            input_ids: [B, L]
            attention_mask: [B, L] or None
            labels: [B, L] or None
        Returns:
            C_exact: [B, L] intervention scores
        """
        model.eval()
        B, L = input_ids.shape
        device = input_ids.device

        embed_layer = model.get_input_embeddings()
        baseline_emb = self._get_baseline_embedding(embed_layer, device)

        # Get original output logits
        x_orig = model.embed(input_ids).detach()
        orig_output = model.forward_with_embeddings(x_orig, attention_mask, labels)
        orig_logits = orig_output["logits"]  # [B, L, V]

        if labels is not None:
            orig_logits = orig_logits[:, :-1].contiguous()
            orig_probs = F.softmax(orig_logits, dim=-1)  # [B, L-1, V]

        # For each position, intervene and measure KL
        scores = torch.zeros(B, L, device=device)

        for j in range(L):
            # Clone and intervene at position j
            x_interv = x_orig.clone()  # [B, L, D]
            x_interv[:, j, :] = baseline_emb

            interv_output = model.forward_with_embeddings(x_interv, attention_mask, labels)
            interv_logits = interv_output["logits"]  # [B, L, V]

            if labels is not None:
                interv_logits = interv_logits[:, :-1].contiguous()
                interv_probs = F.softmax(interv_logits, dim=-1)

                # KL divergence between original and intervened distributions
                kl = (orig_probs * (orig_probs.log() - interv_probs.log())).sum(dim=-1)  # [B, L-1]
                # Sum KL over all positions (total distribution change)
                scores[:, j] = kl.sum(dim=-1)
            else:
                # Without labels, compare logit changes
                diff = (orig_logits - interv_logits).norm(dim=-1)
                scores[:, j] = diff.sum(dim=-1)

        return scores
