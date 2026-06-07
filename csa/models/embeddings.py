"""Positional embeddings."""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding (non-learnable), supports dynamic max_len."""

    def __init__(self, d_model: int, max_len: int = 131072):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._pe = None  # lazy compute

    def _compute_pe(self, L: int, device: torch.device) -> torch.Tensor:
        """Compute sinusoidal PE for length L on given device."""
        position = torch.arange(L, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, L, self.d_model, device=device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding to input [B, L, D]."""
        L = x.size(1)
        return x + self._compute_pe(L, x.device)


class LearnedPositionalEmbedding(nn.Module):
    """Learnable positional embedding."""

    def __init__(self, d_model: int, max_len: int = 131072):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input [B, L, D]."""
        L = x.size(1)
        positions = torch.arange(L, device=x.device)
        return x + self.pe(positions).unsqueeze(0)
