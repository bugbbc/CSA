"""
Unified configuration system for CSA experiments.
All configs are serializable dataclasses for full reproducibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class AttentionConfig:
    """Configuration for attention variant."""
    type: str = "csa"  # dense, local_window, random_topk, similarity_topk, gated, gated_sparse, csa, csa_exact
    window: int = 128   # local window size w
    k: int = 32         # global budget k
    refresh_interval: int = 4  # inference refresh interval r
    baseline_type: str = "zero"  # zero, mask, mean
    n_heads: int = 8
    head_dim: Optional[int] = None  # if None, computed as d_model // n_heads


@dataclass
class ModelConfig:
    """Base transformer configuration."""
    d_model: int = 512
    d_ff: int = 2048
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    max_len: int = 131072
    vocab_size: int = 97  # SimpleTokenizer vocabulary size
    pad_token_id: int = 0
    task: str = "lm"  # lm, classification, qa
    num_classes: int = 2  # for classification
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    tie_embeddings: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    trials: int = 3
    seeds: Tuple[int, ...] = (42, 123, 3407)
    use_amp: bool = True
    use_compile: bool = False
    use_ddp: bool = True
    eval_every: int = 500
    save_every: int = 5000
    max_seq_length: int = 4096
    label_smoothing: float = 0.0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    experiment_name: str = ""
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    log_wandb: bool = True
    wandb_project: str = "csa-experiments"
    wandb_entity: Optional[str] = None
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        if self.experiment_name and not self.output_dir.endswith(self.experiment_name):
            self.output_dir = f"{self.output_dir}/{self.experiment_name}"
