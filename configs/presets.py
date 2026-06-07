"""Config presets for all experiments."""

from .base import AttentionConfig, ModelConfig, TrainingConfig, ExperimentConfig


def longbench_config() -> ExperimentConfig:
    """Default config for LongBench experiments (Exp1)."""
    return ExperimentConfig(
        experiment_name="exp1_longbench",
        model=ModelConfig(
            d_model=512,
            d_ff=2048,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            max_len=8192,
            task="lm",
            attention=AttentionConfig(
                type="csa",
                window=128,
                k=32,
                refresh_interval=4,
                baseline_type="zero",
            ),
        ),
        training=TrainingConfig(
            batch_size=4,
            max_seq_length=4096,
            seed=42,
            seeds=(42, 123, 3407),
            use_amp=True,
        ),
    )


def infinitebench_config() -> ExperimentConfig:
    """Default config for InfiniteBench experiments (Exp2)."""
    return ExperimentConfig(
        experiment_name="exp2_infinitebench",
        model=ModelConfig(
            d_model=512,
            d_ff=2048,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            max_len=131072,
            task="lm",
            attention=AttentionConfig(
                type="csa",
                window=128,
                k=32,
                refresh_interval=4,
                baseline_type="zero",
            ),
        ),
        training=TrainingConfig(
            batch_size=2,
            max_seq_length=65536,
            seed=42,
            seeds=(42, 123, 3407),
        ),
    )


def needle_config() -> ExperimentConfig:
    """Default config for Needle-in-Haystack experiments (Exp3)."""
    return ExperimentConfig(
        experiment_name="exp3_needle",
        model=ModelConfig(
            d_model=512,
            d_ff=2048,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            max_len=65536,
            task="lm",
            attention=AttentionConfig(
                type="csa",
                window=128,
                k=32,
                refresh_interval=1,
                baseline_type="zero",
            ),
        ),
        training=TrainingConfig(
            batch_size=2,
            max_seq_length=65536,
            seed=42,
            seeds=(42, 123, 3407),
        ),
    )


def causal_robustness_config() -> ExperimentConfig:
    """Default config for Causal Robustness experiments (Exp4)."""
    return ExperimentConfig(
        experiment_name="exp4_causal",
        model=ModelConfig(
            d_model=128,  # Smaller for faster synthetic experiments
            d_ff=512,
            n_layers=3,
            n_heads=4,
            dropout=0.1,
            max_len=512,
            task="classification",
            num_classes=2,
            attention=AttentionConfig(
                type="csa",
                window=128,
                k=16,
                refresh_interval=2,
                baseline_type="zero",
            ),
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=1e-4,
            max_steps=100,
            seed=42,
            seeds=(42, 123, 3407),
            use_amp=False,
        ),
    )


def proxy_validation_config() -> ExperimentConfig:
    """Default config for Proxy Validation experiments (Exp5)."""
    return ExperimentConfig(
        experiment_name="exp5_proxy",
        model=ModelConfig(
            d_model=64,  # Tiny model for exact intervention feasibility
            d_ff=256,
            n_layers=2,
            n_heads=4,
            dropout=0.1,
            max_len=512,
            task="classification",
            num_classes=2,
            attention=AttentionConfig(
                type="csa",
                window=64,
                k=16,
                refresh_interval=1,
                baseline_type="zero",
            ),
        ),
        training=TrainingConfig(
            batch_size=4,
            max_seq_length=256,
            seed=42,
            seeds=(42,),
            use_amp=False,
        ),
    )


def ablation_config() -> ExperimentConfig:
    """Default config for Ablation studies (Exp6)."""
    return ExperimentConfig(
        experiment_name="exp6_ablation",
        model=ModelConfig(
            d_model=128,
            d_ff=512,
            n_layers=3,
            n_heads=4,
            dropout=0.1,
            max_len=512,
            task="classification",
            num_classes=2,
            attention=AttentionConfig(
                type="csa",
                window=128,
                k=32,
                refresh_interval=2,
                baseline_type="zero",
            ),
        ),
        training=TrainingConfig(
            batch_size=8,
            learning_rate=1e-4,
            max_steps=50,
            seed=42,
            seeds=(42, 123, 3407),
            use_amp=False,
        ),
    )


def efficiency_config() -> ExperimentConfig:
    """Default config for Efficiency analysis (Exp7)."""
    return ExperimentConfig(
        experiment_name="exp7_efficiency",
        model=ModelConfig(
            d_model=512,
            d_ff=2048,
            n_layers=6,
            n_heads=8,
            dropout=0.0,
            max_len=131072,
            task="lm",
            attention=AttentionConfig(
                type="csa",
                window=128,
                k=32,
                refresh_interval=4,
                baseline_type="zero",
            ),
        ),
        training=TrainingConfig(
            batch_size=1,
            max_seq_length=65536,
            seed=42,
            seeds=(42,),
            use_amp=False,
        ),
    )
