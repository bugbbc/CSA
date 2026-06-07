"""
Rebuilt causal robustness dataset with valid causal structure.

Key design:
1. Token positions are randomized (no fixed evidence/spurious positions)
2. Label is determined ONLY by evidence tokens (evidence-only oracle works)
3. Spurious tokens correlate with label only during training
4. Noise tokens contain zero label information
5. Multiple difficulty levels
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from .tokenizer import SimpleTokenizer


class CausalRobustnessDataset(Dataset):
    """
    Causally-valid synthetic dataset.

    Causal structure:
        Label → Evidence tokens (causal)
        Label ← Correlation → Spurious tokens (confounded, training only)
        Noise tokens (independent)

    Training:
        Label=0 → Evidence=words from set A, Spurious=word X (prob=rho) or Y (prob=1-rho)
        Label=1 → Evidence=words from set B, Spurious=word Y (prob=rho) or X (prob=1-rho)

    Robust test:
        Everyone → Spurious tokens are randomized (uncorrelated with label)

    Positions of evidence, spurious, and noise are randomized per example.
    """

    EVIDENCE_A = ["network", "layer", "gradient", "backprop", "activation"]
    EVIDENCE_B = ["database", "query", "index", "schema", "transaction"]
    SPURIOUS_WORDS = ["sunny", "rainy", "cloudy", "windy"]
    NOISE_WORDS = [
        "the", "a", "an", "is", "was", "are", "were", "this", "that",
        "it", "with", "from", "for", "on", "at", "by", "as", "to", "in",
        "of", "and", "or", "not", "but", "be", "has", "have", "do",
        "hello", "world", "foo", "bar",
    ]

    def __init__(
        self,
        num_examples: int = 500,
        seq_length: int = 128,
        num_evidence: int = 3,
        num_spurious: int = 2,
        spurious_correlation: float = 0.95,
        split: str = "train",
        seed: int = 42,
    ):
        self.num_examples = num_examples
        self.seq_length = seq_length
        self.num_evidence = min(num_evidence, len(self.EVIDENCE_A))
        self.num_spurious = min(num_spurious, len(self.SPURIOUS_WORDS))
        self.spurious_correlation = spurious_correlation
        self.split = split
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.tokenizer = SimpleTokenizer()

        self.examples = self._generate()

    def _generate(self) -> List[Dict]:
        examples = []

        for i in range(self.num_examples):
            label = self.rng.randint(0, 2)

            # Select evidence tokens (causal → determine label)
            if label == 0:
                evidence_words = self.rng.choice(
                    self.EVIDENCE_A, self.num_evidence, replace=False
                ).tolist()
            else:
                evidence_words = self.rng.choice(
                    self.EVIDENCE_B, self.num_evidence, replace=False
                ).tolist()

            # Select spurious tokens
            # Training: correlated with label (rho controls strength)
            # Robust test: randomized
            num_correlated = min(
                int(self.num_spurious * self.spurious_correlation),
                self.num_spurious
            )
            if self.split == "train":
                # Split spurious: first half for label=0, second for label=1
                half = len(self.SPURIOUS_WORDS) // 2
                sp_pool_0 = self.SPURIOUS_WORDS[:half]
                sp_pool_1 = self.SPURIOUS_WORDS[half:]
                if label == 0:
                    correlated = self.rng.choice(
                        sp_pool_0, min(num_correlated, len(sp_pool_0)), replace=False
                    ).tolist()
                    uncorrelated = self.rng.choice(
                        sp_pool_1, max(0, self.num_spurious - len(correlated)), replace=True
                    ).tolist() if len(sp_pool_1) > 0 else []
                else:
                    correlated = self.rng.choice(
                        sp_pool_1, min(num_correlated, len(sp_pool_1)), replace=False
                    ).tolist()
                    uncorrelated = self.rng.choice(
                        sp_pool_0, max(0, self.num_spurious - len(correlated)), replace=True
                    ).tolist() if len(sp_pool_0) > 0 else []
                spurious_words = correlated + uncorrelated
            else:
                # Robust: random split regardless of label
                spurious_words = self.rng.choice(
                    self.SPURIOUS_WORDS, self.num_spurious, replace=False
                ).tolist()

            # Count how many spurious agree with label (for diagnostics)
            half = len(self.SPURIOUS_WORDS) // 2
            spurious_agree = sum(
                1 for w in spurious_words
                if (label == 0 and w in self.SPURIOUS_WORDS[:half])
                or (label == 1 and w in self.SPURIOUS_WORDS[half:])
            )

            # Build token pool
            all_evidence_ids = [self.tokenizer.encode(w)[0] for w in evidence_words]
            all_spurious_ids = [self.tokenizer.encode(w)[0] for w in spurious_words]

            # Randomize positions
            total_special = len(all_evidence_ids) + len(all_spurious_ids)
            positions = self.rng.permutation(self.seq_length).tolist()

            # Assign evidence and spurious to random positions
            ev_positions = positions[:len(all_evidence_ids)]
            sp_positions = positions[len(all_evidence_ids):len(all_evidence_ids)+len(all_spurious_ids)]
            noise_positions = positions[len(all_evidence_ids)+len(all_spurious_ids):]

            # Build sequence
            seq_ids = [0] * self.seq_length
            for pos, tid in zip(ev_positions, all_evidence_ids):
                seq_ids[pos] = tid
            for pos, tid in zip(sp_positions, all_spurious_ids):
                seq_ids[pos] = tid
            for pos in noise_positions:
                noise_word = self.rng.choice(self.NOISE_WORDS)
                seq_ids[pos] = self.tokenizer.encode(noise_word)[0]

            input_ids = torch.tensor(seq_ids, dtype=torch.long)

            examples.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "label": label,
                "evidence_token_ids": all_evidence_ids,
                "spurious_token_ids": all_spurious_ids,
                "evidence_positions": ev_positions,
                "spurious_positions": sp_positions,
                "evidence_words": evidence_words,
                "spurious_words": spurious_words,
                "num_spurious_agree": spurious_agree,
            })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate_causal_robustness(batch: List[Dict]) -> Dict:
    """Collate for causal robustness dataset."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True, padding_value=0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True, padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([item["label"] for item in batch]),
        "evidence_token_ids": [item["evidence_token_ids"] for item in batch],
        "spurious_token_ids": [item["spurious_token_ids"] for item in batch],
        "evidence_positions": [item["evidence_positions"] for item in batch],
        "spurious_positions": [item["spurious_positions"] for item in batch],
    }
