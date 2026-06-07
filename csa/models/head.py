"""Task-specific heads for the CSA encoder."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskHead(nn.Module, ABC):
    """Abstract base for task-specific heads."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Returns dict with 'logits' and optionally 'loss'."""
        pass


class LMHead(TaskHead):
    """Language modeling head (tied embeddings optional)."""

    def __init__(self, d_model: int, vocab_size: int, pad_token_id: int = 50256):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        hidden_states = self.ln(hidden_states)
        logits = self.lm_head(hidden_states)

        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id,
            )
            output["loss"] = loss

        return output


class ClassificationHead(TaskHead):
    """Classification head with pooling."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Use [CLS] token (first position)
        pooled = self.pooler(hidden_states[:, 0])
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output


class QAHead(TaskHead):
    """Extractive QA head (predict start/end positions)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.qa_outputs = nn.Linear(d_model, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        logits = self.qa_outputs(hidden_states)  # [B, L, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        output = {"logits": (start_logits, end_logits)}
        if labels is not None:
            start_positions, end_positions = labels
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            output["loss"] = (start_loss + end_loss) / 2

        return output


def build_head(task: str, d_model: int, vocab_size: int = 50257, num_classes: int = 2, **kwargs) -> TaskHead:
    """Factory for task heads."""
    if task == "lm":
        return LMHead(d_model, vocab_size, kwargs.get("pad_token_id", 50256))
    elif task == "classification":
        return ClassificationHead(d_model, num_classes, kwargs.get("dropout", 0.1))
    elif task == "qa":
        return QAHead(d_model)
    else:
        raise ValueError(f"Unknown task: {task}")
