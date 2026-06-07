"""
Simple built-in tokenizer that does not require HuggingFace downloads.

Maps string tokens to integer IDs using a simple vocabulary.
Designed for synthetic experiments where pretrained tokenizers are unavailable.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Union

import torch


class SimpleTokenizer:
    """Minimal tokenizer with built-in vocabulary for synthetic experiments."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.vocab = dict(vocab)
        else:
            self.vocab = {
                # Special tokens
                "<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3,
                # Common English words
                "the": 4, "a": 5, "an": 6, "is": 7, "was": 8, "are": 9,
                "were": 10, "this": 11, "that": 12, "it": 13, "with": 14,
                "from": 15, "for": 16, "on": 17, "at": 18, "by": 19,
                "as": 20, "to": 21, "in": 22, "of": 23, "and": 24,
                "or": 25, "not": 26, "but": 27, "be": 28, "has": 29,
                "have": 30, "do": 31, "does": 32, "will": 33, "can": 34,
                "would": 35, "could": 36, "should": 37, "may": 38,
                "hello": 39, "world": 40, "foo": 41, "bar": 42,
                # Evidence words
                "network": 43, "layer": 44, "gradient": 45, "backprop": 46,
                "activation": 47, "database": 48, "query": 49, "index": 50,
                "schema": 51, "transaction": 52,
                # Spurious words
                "sunny": 53, "rainy": 54, "cloudy": 98, "windy": 99,
                # Important words
                "critical": 55, "essential": 56, "vital": 57, "key": 58,
                "crucial": 59,
                # Needle
                "programming": 60, "language": 61, "research": 62,
                "Python": 63, "best": 64, "Answer": 65, "Question": 66,
                "Context": 67, "Summary": 68,
                "secret": 69, "code": 70, "filler": 71, "text": 72,
                "target": 73, "fact": 74,
                # Additional tokens for long texts
                "machine": 75, "learning": 76, "model": 77, "data": 78,
                "processing": 79, "neural": 80, "deep": 81,
                "natural": 82, "self": 83, "attention": 84,
                "mechanisms": 85, "sequence": 86, "modeling": 87,
                "transformers": 88, "use": 89,
                "fox": 90, "jumps": 91, "over": 92, "lazy": 93, "dog": 94,
                "quick": 95, "brown": 96,
                # Batch-efficient padding
                "</w>": 97,  # word boundary marker for tokenization
            }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.unk_token_id = self.vocab.get("<unk>", 1)
        self.eos_token_id = self.vocab.get("</s>", 3)
        self.vocab_size = len(self.vocab)
        self.model_max_length = 131072

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.vocab.get("<s>", 2))
        for word in text.strip().split():
            token_id = self.vocab.get(word.lower(), self.unk_token_id)
            tokens.append(token_id)
        if add_special_tokens:
            tokens.append(self.vocab.get("</s>", 3))
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            word = self.id_to_token.get(tid, "<unk>")
            if skip_special_tokens and word.startswith("<") and word.endswith(">"):
                continue
            tokens.append(word)
        return " ".join(tokens)

    def __call__(self, text: str, truncation: bool = False, max_length: Optional[int] = None,
                 return_tensors: Optional[str] = None, **kwargs) -> dict:
        """Tokenizer-compatible interface similar to HuggingFace."""
        tokens = self.encode(text)
        attention_mask = [1] * len(tokens)

        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            attention_mask = attention_mask[:max_length]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }

        return {"input_ids": tokens, "attention_mask": attention_mask}
