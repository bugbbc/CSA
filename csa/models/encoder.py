"""
CSA Transformer Encoder.

Standard transformer encoder where self-attention is replaced
with the configurable CSA (Causal Sparse Attention) mechanism.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention import build_attention, CSAAttention, CSAExactAttention
from .embeddings import SinusoidalPositionalEmbedding
from .head import build_head


class CSAEncoderLayer(nn.Module):
    """
    A single transformer encoder layer with pluggable attention.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        attn_module: nn.Module,
    ):
        super().__init__()
        self.self_attn = attn_module
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Pre-norm architecture
        residual = x
        x_norm = self.norm1(x)

        attn_kwargs = {"hidden_states": x_norm}
        # Convert [B, L] padding mask to [B, 1, 1, L] boolean for attention
        if attention_mask is not None and attention_mask.dim() == 2:
            attn_kwargs["attention_mask"] = attention_mask.unsqueeze(1).unsqueeze(2).bool()
        else:
            attn_kwargs["attention_mask"] = attention_mask
        # Only pass CSA-specific kwargs if the attention module accepts them
        if hasattr(self.self_attn, 'set_full_model'):
            attn_kwargs["labels"] = labels
            attn_kwargs["input_ids"] = input_ids

        attn_out, aux = self.self_attn(**attn_kwargs)
        x = residual + self.dropout(attn_out)

        # FFN with pre-norm
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.ffn(x_norm)

        return x, aux


class CSAEncoder(nn.Module):
    """
    CSA Transformer Encoder.

    Configurable with different attention variants and task heads.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        d_ff: int = 2048,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 131072,
        pad_token_id: int = 50256,
        task: str = "lm",
        num_classes: int = 2,
        attn_type: str = "dense",
        window: int = 128,
        k: int = 32,
        refresh_interval: int = 4,
        baseline_type: str = "zero",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_token_id = pad_token_id
        self.attn_type = attn_type

        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_pos = SinusoidalPositionalEmbedding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)

        # Encoder layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            attn_module = build_attention(
                attn_type=attn_type,
                d_model=d_model,
                n_heads=n_heads,
                window=min(window, d_model * 2),  # prevent absurdly large windows
                k=min(k, d_model),  # prevent k > seq_len
                refresh_interval=refresh_interval,
                baseline_type=baseline_type,
                dropout=dropout,
                vocab_size=vocab_size,
                pad_token_id=pad_token_id,
            )
            layer = CSAEncoderLayer(d_model, d_ff, n_heads, dropout, attn_module)
            self.layers.append(layer)

            # If CSA, link to full model reference
            if isinstance(attn_module, (CSAAttention, CSAExactAttention)):
                attn_module.set_full_model(self)

        # Final layer norm and head
        self.final_norm = nn.LayerNorm(d_model)
        self.head = build_head(task, d_model, vocab_size, num_classes, dropout=dropout)

        # Tie embedding weights with LM head if configured
        if task == "lm" and hasattr(self.head, "lm_head"):
            self.head.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute embedding from token IDs."""
        x = self.embed_tokens(input_ids)
        x = self.embed_pos(x)
        x = self.embed_dropout(x)
        return x

    def forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with pre-computed embeddings.
        Used by contribution estimators.

        NOTE: embeddings should be raw token embeddings (without positional
        encoding), as positional encoding is applied internally.
        """
        # Set flag to prevent CSA from attempting contribution estimation
        # during its own contribution estimation backward pass (avoid recursion)
        object.__setattr__(self, '_in_contrib_pass', True)
        x = self.embed_pos(embeddings)
        x = self.embed_dropout(x)

        all_aux = []
        for layer in self.layers:
            x, aux = layer(x, attention_mask=attention_mask, labels=labels,
                          input_ids=input_ids)
            if aux:
                all_aux.append(aux)

        object.__setattr__(self, '_in_contrib_pass', False)

        x = self.final_norm(x)
        output = self.head(x, labels=labels, input_ids=input_ids)
        output["aux"] = all_aux
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        """
        Main forward pass.

        Args:
            input_ids: [B, L]
            attention_mask: [B, L] (1 = valid, 0 = pad)
            labels: [B, L] for LM, else task-dependent
            return_aux: whether to return auxiliary info
        """
        x = self.embed(input_ids)

        all_aux = []
        for layer in self.layers:
            x, aux = layer(x, attention_mask=attention_mask, labels=labels, input_ids=input_ids)
            if aux:
                all_aux.append(aux)

        x = self.final_norm(x)
        output = self.head(x, labels=labels, input_ids=input_ids)

        if return_aux:
            output["aux"] = all_aux

        return output
