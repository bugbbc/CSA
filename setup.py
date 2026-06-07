from setuptools import setup, find_packages

setup(
    name="csa",
    version="0.1.0",
    description="Causal Sparse Attention: Contribution-Guided Sparse Routing via Causal Evidence Estimation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "einops>=0.6.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
)
