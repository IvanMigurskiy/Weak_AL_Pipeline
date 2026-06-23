"""WeakAL Pipeline — Hybrid Active Learning + Weak Supervision."""

from .config import PipelineConfig, ExperimentConfig
from .data import Dataset, load_dataset, get_stratified_seed_indices
from .encoders import (
    TextEncoder,
    TfidfEncoder,
    BM25Encoder,
    FastTextSparseEncoder,
    FastTextDenseEncoder,
    SpladeEncoder,
    DenseEncoder,
    HybridEncoder,
    create_encoder,
)
from .active_learning import ActiveLearner, QueryStrategy
from .weak_supervision import WeakSupervisor, LabelAggregator, WeakCertainty
from .pipeline import HybridPipeline, ALOnlyPipeline, WSOnlyPipeline, PipelineResult
from .experiments import run_experiment, run_comparison

__all__ = [
    "PipelineConfig",
    "ExperimentConfig",
    "Dataset",
    "load_dataset",
    "get_stratified_seed_indices",
    # Encoders
    "TextEncoder",
    "TfidfEncoder",
    "BM25Encoder",
    "FastTextSparseEncoder",
    "FastTextDenseEncoder",
    "SpladeEncoder",
    "DenseEncoder",
    "HybridEncoder",
    "create_encoder",
    # AL & WS
    "ActiveLearner",
    "QueryStrategy",
    "WeakSupervisor",
    "LabelAggregator",
    "WeakCertainty",
    # Pipelines
    "HybridPipeline",
    "ALOnlyPipeline",
    "WSOnlyPipeline",
    "PipelineResult",
    # Experiments
    "run_experiment",
    "run_comparison",
]
