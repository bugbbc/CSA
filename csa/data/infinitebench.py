"""
InfiniteBench dataset loader.

Evaluates models at varying context lengths: 8K, 16K, 32K, 64K, 128K.

Synthetic data generation for testing retrieval at scale.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import SimpleTokenizer


class InfiniteBenchDataset(Dataset):
    """InfiniteBench: retrieval accuracy at varying context lengths."""

    def __init__(
        self,
        context_length: int = 8192,
        num_examples: int = 50,
        split: str = "test",
        data_dir: str = "data/infinitebench",
        tokenizer_name: str = "gpt2",
    ):
        self.context_length = context_length
        self.num_examples = num_examples
        self.split = split
        self.data_dir = data_dir
        self.tokenizer = SimpleTokenizer()

        self.examples = self._load_or_generate()

    def _load_or_generate(self) -> List[Dict]:
        """Load from file or generate synthetic."""
        json_path = os.path.join(self.data_dir, f"infinitebench_{self.context_length}_{self.split}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                return json.load(f)

        # Generate synthetic data
        examples = []
        for i in range(self.num_examples):
            target_fact = f"The secret code is {i * 7 + 42}."
            filler = "This is filler text. " * (self.context_length // 20 - 5)
            # Insert target fact at a random position
            context = filler + target_fact + filler
            examples.append({
                "context": context[:self.context_length * 4],  # rough char limit
                "question": f"What is the secret code?",
                "answer": f"{i * 7 + 42}",
                "target_fact": target_fact,
            })
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        text = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "answer": example["answer"],
            "target_fact": example.get("target_fact", ""),
        }


def collate_infinitebench(batch: List[Dict]) -> Dict:
    """Collate function for InfiniteBench."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=50256)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answers": [item["answer"] for item in batch],
        "target_facts": [item.get("target_fact", "") for item in batch],
    }
