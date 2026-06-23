"""
Configuration for WeakAL+AutoWS hybrid pipeline experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PipelineConfig:
    """Master configuration for all experiments."""

    # ── Dataset ──────────────────────────────────────────────
    dataset_name: str = "customer_tickets"
    max_samples: int | None = 5000        # cap dataset size for speed
    test_size: float = 0.2
    random_seed: int = 42

    # ── TF-IDF ──────────────────────────────────────────────
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)

    # ── Active Learning ─────────────────────────────────────
    query_strategy: Literal[
        "random",
        "uncertainty_least_confident",
        "uncertainty_margin",
        "uncertainty_entropy",
        "badge",
        "cost_sensitive",
    ] = "uncertainty_entropy"
    batch_size: int = 10
    initial_per_class: int = 2            # seed labels per class
    max_human_labels: int = 300           # human labeling budget

    # ── Weak Supervision (AutoWS-style) ─────────────────────
    # Labeling functions
    use_tfidf_lf: bool = True
    use_nb_lf: bool = True
    use_svm_lf: bool = True
    use_rf_lf: bool = True
    use_knn_lf: bool = False       # Disabled: bistable coverage with bm25, corrupts Dawid-Skene
    use_lr_lf: bool = True
    use_keyword_lf: bool = True
    use_topic_lf: bool = False
    topic_n_topics: int = 20
    topic_model: Literal["nmf", "lda"] = "nmf"

    # Label aggregation
    label_model: Literal["majority_vote", "dawid_skene"] = "dawid_skene"

    # LF confidence threshold (abstain below)
    lf_confidence_threshold: float = 0.7

    # ── Hybrid (WeakAL-style) ───────────────────────────────
    # Minimum classifier accuracy on L before WS kicks in
    ws_accuracy_threshold: float = 0.6
    # WeakCert: minimum prediction certainty to auto-label
    weak_cert_alpha: float = 0.9
    # WS confidence filter: minimum WS confidence to accept weak label
    ws_confidence_filter: float = 0.8

    # ── Encoder ────────────────────────────────────────────
    encoder_type: Literal[
        "tfidf", "bm25", "fasttext_sparse", "fasttext_dense",
        "splade", "dense", "hybrid",
    ] = "fasttext_dense"
    hybrid_alpha: float = 0.5              # sparse/dense balance for hybrid encoder

    # ── Auto-labeling toggles ───────────────────────────────
    use_weak_cert: bool = True             # WeakCert auto-labeling
    use_weak_clust: bool = False           # WeakClust auto-labeling

    # ── Classifier ──────────────────────────────────────────
    classifier_type: Literal["rf", "lr", "svm"] = "lr"

    # ── Technique-specific parameters (T3–T14) ──────────────
    # T3: Platt calibration cross-validation folds
    t3_calibration_cv: int = 3
    # T5: Unanimous voting parameters
    t5_agreement_ratio: float = 1.0
    t5_min_voters: int = 2
    # T6: Per-class WS thresholds
    t6_base_threshold: float = 0.7
    # T7: Isotonic calibration cross-validation folds
    t7_calibration_cv: int = 3
    # T10: Label propagation parameters
    t10_n_neighbors: int = 10
    t10_max_iter: int = 30
    t10_confidence_threshold: float = 0.8
    # T11: Cost-sensitive AL weight power
    t11_weight_power: float = 1.0
    # T12: Adaptive budget minimum batch fraction
    t12_min_batch_fraction: float = 0.3
    # T14: Calibrated pseudo-labeling parameters
    t14_use_energy_scoring: bool = True
    t14_temperature: float = 1.0
    t14_class_distribution_aware: bool = True

    def with_overrides(self, **kwargs) -> PipelineConfig:
        """Create a new PipelineConfig with some fields overridden."""
        from dataclasses import asdict
        d = asdict(self)
        d.update(kwargs)
        return PipelineConfig(**d)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment comparison."""
    name: str = ""
    description: str = ""
    pipeline_mode: Literal[
        "baseline",      # full supervision (all labels)
        "random_labels",  # random subset of labels (same count as hybrid)
        "al_only",       # AL without WS
        "ws_only",       # WS without AL
        "hybrid",        # AL + WS (WeakAL)
    ] = "hybrid"
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    n_repeats: int = 3                     # statistical robustness
