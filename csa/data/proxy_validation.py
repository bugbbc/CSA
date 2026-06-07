"""
Proxy validation dataset.

Short sequences for comparing gradient proxy contribution scores
against exact intervention. Sequence lengths: 64, 128, 256.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
import numpy as np

from .tokenizer import SimpleTokenizer


class ProxyValidationDataset(Dataset):
    """
    Short-sequence dataset for proxy validation experiments.

    Compares ranking from different estimators:
    1. Gradient norm
    2. Input x Gradient
    3. Integrated Gradients
    4. CSA Proxy (ours)
    Against: Exact Intervention
    """

    def __init__(
        self,
        seq_length: int = 128,
        num_examples: int = 100,
        vocab_size: int = 500,
        tokenizer_name: str = "gpt2",
        seed: int = 42,
    ):
        self.seq_length = seq_length
        self.num_examples = num_examples
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.tokenizer = SimpleTokenizer()

        self.examples = self._generate()

    def _generate(self) -> List[Dict]:
        """Generate short sequences with known important tokens."""
        # Vocabulary of words with different "importance" levels
        important_words = ["critical", "essential", "vital", "key", "crucial"]
        neutral_words = ["hello", "world", "this", "that", "and", "the", "is"]

        examples = []
        for i in range(self.num_examples):
            # Pick a random important token to be the "evidence"
            important_word = self.rng.choice(important_words)
            num_important = self.rng.randint(1, 3)

            # Build sequence
            words = self.rng.choice(neutral_words, self.seq_length - num_important, replace=True).tolist()
            # Insert important words at known positions
            important_positions = self.rng.choice(
                range(self.seq_length), num_important, replace=False
            )
            for pos in important_positions:
                words[pos] = important_word

            sequence = " ".join(words)
            encoding = self.tokenizer(
                sequence, truncation=True, max_length=self.seq_length, return_tensors="pt"
            )

            examples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "important_word": important_word,
                "important_positions": important_positions,
                "label": self.rng.randint(0, 2),
            })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate_proxy(batch: List[Dict]) -> Dict:
    """Collate for proxy validation."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=50256,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([item["label"] for item in batch]),
        "important_words": [item["important_word"] for item in batch],
        "important_positions": [item["important_positions"] for item in batch],
    }
