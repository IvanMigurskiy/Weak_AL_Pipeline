"""
Configuration for WeakAL+AutoWS hybrid pipeline experiments.

ALL hyperparameters are centralized here — no magic numbers in pipeline code.
Every technique (T1-T6, T7-T12, T14) draws its parameters from this single config.
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
    # Labeling functions (enable/disable)
    use_nb_lf: bool = True
    use_svm_lf: bool = True
    use_rf_lf: bool = True
    use_knn_lf: bool = True
    use_lr_lf: bool = True
    use_keyword_lf: bool = True

    # Label aggregation
    label_model: Literal["majority_vote", "dawid_skene"] = "dawid_skene"

    # LF confidence threshold (abstain below this)
    lf_confidence_threshold: float = 0.7

    # ── Hybrid Pipeline (WeakAL-style) ──────────────────────
    # WS activation: minimum classifier train accuracy before WS kicks in
    ws_accuracy_threshold: float = 0.6
    # WS activation: minimum human labels before WS kicks in
    ws_min_human_labels: int | None = None  # None = auto (initial_per_class * n_classes + batch_size)
    # WeakCert: minimum prediction certainty to auto-label
    weak_cert_alpha: float = 0.9
    # WS: minimum confidence for AutoWS aggregated labels
    ws_confidence_filter: float = 0.8
    # WS: maximum WS labels added per step (cap)
    ws_batch_limit: int | None = None  # None = same as batch_size

    # ── Classifier ──────────────────────────────────────────
    classifier_type: Literal["rf", "lr", "svm"] = "rf"

    # ── T1: Weighted Training ──────────────────────────────
    # WS labels get this weight vs human labels (1.0)
    t1_ws_weight: float = 0.5

    # ── T2: WS Label Verification ──────────────────────────
    # Verify WS labels every N steps (0 = disabled)
    t2_verify_every: int = 3
    # Verification mode: "remove" bad labels, or "downgrade" their weight
    t2_mode: Literal["remove", "downgrade"] = "remove"

    # ── T3: LF Calibration ─────────────────────────────────
    # Number of CV folds for Platt scaling (0 = disabled)
    t3_calibration_cv: int = 3

    # ── T4: Self-Training Pseudo-Labels ────────────────────
    # Starting confidence threshold for pseudo-labeling
    t4_initial_threshold: float = 0.9
    # Minimum threshold (stops decaying below this)
    t4_min_threshold: float = 0.7
    # Threshold decay per iteration
    t4_decay: float = 0.95
    # Max self-training iterations
    t4_max_iterations: int = 10
    # Pseudo-labels added per iteration
    t4_batch_size: int = 20
    # Total pseudo-labels allowed
    t4_max_pseudo_labels: int = 200

    # ── T5: Unanimous Voting ───────────────────────────────
    # Fraction of voting LFs that must agree (1.0 = unanimous)
    t5_agreement_ratio: float = 1.0
    # Minimum non-abstaining LFs needed to produce a label
    t5_min_voters: int = 2

    # ── T6: Per-Class Thresholds ───────────────────────────
    # Base threshold for all classes (adjusted per-class around this)
    t6_base_threshold: float = 0.7
    # How much to raise threshold for hard classes
    t6_difficulty_multiplier: float = 1.5
    # Minimum per-class threshold
    t6_min_threshold: float = 0.6
    # Maximum per-class threshold
    t6_max_threshold: float = 0.98

    # ── T7: Isotonic Calibration ───────────────────────────
    # Calibration method: "sigmoid" (Platt/T3) or "isotonic" (T7)
    t7_calibration_method: Literal["sigmoid", "isotonic"] = "isotonic"
    # Number of CV folds for isotonic calibration
    t7_calibration_cv: int = 5

    # ── T8: BERT-based LF ─────────────────────────────────
    # Enable BERT sentence-transformer as 7th LF
    use_bert_lf: bool = False
    # Sentence transformer model name
    bert_model_name: str = "all-MiniLM-L6-v2"

    # ── T9: BADGE Query Strategy ──────────────────────────
    # BADGE: diversity-aware batch active learning
    # (parameters are handled in query strategy)

    # ── T10: Label Propagation ────────────────────────────
    # Enable LP as additional WS source
    t10_use_label_propagation: bool = False
    # Number of neighbors for LP graph
    t10_n_neighbors: int = 10
    # Maximum propagation iterations
    t10_max_iter: int = 30
    # Confidence threshold for propagated labels
    t10_confidence_threshold: float = 0.8

    # ── T11: Cost-Sensitive AL ────────────────────────────
    # Weight AL queries by inverse class frequency
    t11_cost_sensitive: bool = False
    # Power of class weight (1.0 = linear, 0.5 = sqrt)
    t11_weight_power: float = 1.0

    # ── T12: Adaptive Budget ──────────────────────────────
    # Dynamically adjust batch_size based on WS activity
    t12_adaptive_budget: bool = False
    # Minimum batch size (fraction of base batch_size)
    t12_min_batch_fraction: float = 0.3

    # ── T13: FlyingSquid Aggregation ──────────────────────
    # Use FlyingSquid triplet-based aggregation instead of Dawid-Skene
    t13_use_flyingsquid: bool = True
    # Max refinement iterations for accuracy estimation
    t13_max_iter: int = 5
    # Convergence tolerance
    t13_tol: float = 1e-3

    # ── T14: Calibrated Pseudo-Labeling ───────────────────
    # Use energy-based scoring instead of softmax confidence
    t14_use_energy_scoring: bool = False
    # Class-distribution-aware thresholding
    t14_class_distribution_aware: bool = True
    # Temperature for energy scoring
    t14_temperature: float = 1.0

    # ── Helper properties ───────────────────────────────────

    def get_ws_min_human(self, n_classes: int) -> int:
        """Compute minimum human labels before WS activates."""
        if self.ws_min_human_labels is not None:
            return self.ws_min_human_labels
        return self.initial_per_class * n_classes + self.batch_size

    def get_ws_batch_limit(self) -> int:
        """Compute maximum WS labels per step."""
        if self.ws_batch_limit is not None:
            return self.ws_batch_limit
        return self.batch_size

    def with_overrides(self, **kwargs) -> PipelineConfig:
        """Create a copy of this config with some parameters overridden."""
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
