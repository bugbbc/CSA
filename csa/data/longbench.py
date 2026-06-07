"""
LongBench dataset loader.

LongBench datasets: NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA,
2WikiMultihopQA, Musique, GovReport, QMSum.

If datasets are not available via HuggingFace, falls back to local JSON files
in the data/longbench/ directory.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Callable

import torch
from torch.utils.data import Dataset

from .tokenizer import SimpleTokenizer

LONGBENCH_TASKS = {
    "narrativeqa": {"type": "qa", "metric": "rouge-l"},
    "qasper": {"type": "qa", "metric": "rouge-l"},
    "multifieldqa_en": {"type": "qa", "metric": "f1"},
    "hotpotqa": {"type": "qa", "metric": "f1"},
    "2wikimultihopqa": {"type": "qa", "metric": "f1"},
    "musique": {"type": "qa", "metric": "f1"},
    "govreport": {"type": "summarization", "metric": "rouge-l"},
    "qmsum": {"type": "summarization", "metric": "rouge-l"},
}

LONGBENCH_DATASET_NAMES = {
    "narrativeqa": "narrativeqa",
    "qasper": "qasper",
    "multifieldqa_en": "multifield_qa_en",
    "hotpotqa": "hotpot_qa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
    "govreport": "gov_report",
    "qmsum": "qmsum",
}


class LongBenchDataset(Dataset):
    """LongBench dataset for a specific task."""

    def __init__(
        self,
        task_name: str,
        split: str = "test",
        max_length: int = 4096,
        data_dir: str = "data/longbench",
        tokenizer_name: str = "gpt2",
    ):
        self.task_name = task_name
        self.split = split
        self.max_length = max_length
        self.data_dir = data_dir
        self.tokenizer = SimpleTokenizer()

        self.examples = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Load data from local JSON or HF datasets."""
        # Try local JSON first
        json_path = os.path.join(
            self.data_dir, LONGBENCH_DATASET_NAMES.get(self.task_name, self.task_name),
            f"{self.split}.json"
        )
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            examples = []
            for item in data:
                examples.append({
                    "context": item.get("context", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "answers": item.get("answers", [item.get("answer", "")]),
                })
            return examples

        # Fall back to HF datasets (may fail without internet)
        try:
            from datasets import load_dataset
            hf_name = f"LongBench/{LONGBENCH_DATASET_NAMES.get(self.task_name, self.task_name)}"
            dataset = load_dataset(hf_name, split=self.split, trust_remote_code=True)
            examples = []
            for item in dataset:
                examples.append({
                    "context": item.get("context", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "answers": item.get("answers", [item.get("answer", "")]),
                })
            return examples
        except Exception as e:
            print(f"Warning: Could not load LongBench dataset '{self.task_name}': {e}")
            print(f"Place data at: {json_path}")
            # Return synthetic examples for smoke testing
            return self._synthetic_data()

    def _synthetic_data(self) -> List[Dict]:
        """Generate synthetic examples for smoke testing."""
        examples = []
        for i in range(10):
            examples.append({
                "context": f"This is a long context passage number {i}. " * 50,
                "question": f"What is the answer for example {i}?",
                "answer": f"The answer for example {i} is 42.",
                "answers": [f"The answer for example {i} is 42."],
            })
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        context = example["context"]
        question = example.get("question", "")
        answer = example.get("answer", "")

        # Format as prompt + completion
        if self.task_name in ["govreport", "qmsum"]:
            # Summarization
            text = f"Summarize: {context}\nSummary:"
        else:
            # QA
            text = f"Context: {context}\nQuestion: {question}\nAnswer:"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "answer": answer,
            "answers": example["answers"],
        }


def collate_longbench(batch: List[Dict]) -> Dict:
    """Collate function for LongBench."""
    pad_id = 0  # SimpleTokenizer pad ID
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    answers = [item["answer"] for item in batch]
    answers_list = [item["answers"] for item in batch]

    # Pad
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer": answers,
        "answers": answers_list,
    }
