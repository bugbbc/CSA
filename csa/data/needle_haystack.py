"""
Needle-in-a-Haystack (NIAH) evaluation.

Constructs contexts containing one target fact at varying depths
with large amounts of distractor text.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

from .tokenizer import SimpleTokenizer


class NeedleHaystackDataset(Dataset):
    """
    Needle-in-a-Haystack dataset.
    Places a target fact at configurable depth positions within long
    distractor contexts at various context lengths.
    """

    NEEDLE = "The best programming language for AI research is Python."
    DISTRACTOR_SENTENCE = ("The quick brown fox jumps over the lazy dog. "
                           "Machine learning models process large amounts of data. "
                           "Neural networks consist of multiple layers of perceptrons. "
                           "Deep learning has revolutionized natural language processing. "
                           "Transformers use self-attention mechanisms for sequence modeling. ")

    def __init__(
        self,
        context_lengths: List[int] = [8192, 16384, 32768, 65536],
        depths: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        num_examples_per_config: int = 5,
        seed: int = 42,
    ):
        self.context_lengths = context_lengths
        self.depths = depths
        self.num_examples_per_config = num_examples_per_config
        self.seed = seed
        self.tokenizer = SimpleTokenizer()

        self.examples = self._build_examples()

    def _build_examples(self) -> List[Dict]:
        """Build all examples covering all (length, depth) combinations."""
        rng = np.random.RandomState(self.seed)
        examples = []

        for ctx_len in self.context_lengths:
            for depth in self.depths:
                for i in range(self.num_examples_per_config):
                    needle_tokens = self.tokenizer.encode(self.NEEDLE)
                    distractor = self.DISTRACTOR_SENTENCE * (ctx_len // len(self.DISTRACTOR_SENTENCE) + 1)
                    total_tokens = self.tokenizer.encode(distractor)

                    if len(total_tokens) == 0:
                        continue

                    target_pos = int(len(total_tokens) * depth)
                    target_pos = min(max(target_pos, 10), len(total_tokens) - len(needle_tokens) - 10)
                    target_pos = max(target_pos, 0)

                    context_tokens = (total_tokens[:target_pos] + needle_tokens +
                                      total_tokens[target_pos:])
                    context_tokens = context_tokens[:ctx_len]

                    query = "What is the best programming language for AI research?"
                    answer = "Python"

                    examples.append({
                        "context": context_tokens,
                        "context_length": ctx_len,
                        "depth": depth,
                        "needle": self.NEEDLE,
                        "question": query,
                        "answer": answer,
                        "needle_start_pos": target_pos,
                    })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        context_tokens = example["context"]
        question = example["question"]
        q_tokens = self.tokenizer.encode(f"\nQuestion: {question}\nAnswer:")
        text_ids = context_tokens + q_tokens

        input_ids = torch.tensor(text_ids[:self.tokenizer.model_max_length], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer": example["answer"],
            "needle": example["needle"],
            "context_length": example["context_length"],
            "depth": example["depth"],
            "needle_start_pos": example["needle_start_pos"],
        }


def collate_needle(batch: List[Dict]) -> Dict:
    """Collate for NIAH."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answers": [item["answer"] for item in batch],
        "needles": [item["needle"] for item in batch],
        "context_lengths": [item["context_length"] for item in batch],
        "depths": [item["depth"] for item in batch],
        "needle_start_positions": [item["needle_start_pos"] for item in batch],
    }
