"""
Active Learning module.

Implements:
- Query strategies: random, uncertainty (least confident, margin, entropy)
- Active learning loop with budget tracking
- Support for cluster-based query strategies (WeakAL-style)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
import warnings

# Silence Liblinear convergence warnings globally — we handle convergence
# by scaling features (MaxAbsScaler) and increasing max_iter to 10000.
# Any remaining "failed to converge" warnings are benign for our use case
# (the model still produces usable predictions).
warnings.filterwarnings("ignore", message="Liblinear failed to converge")

# Numba JIT acceleration — graceful fallback if numba not installed
try:
    from numba import njit as _njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def _njit(func=None, **kwargs):
        """No-op fallback when numba is not installed."""
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper


@_njit(cache=True)
def _kmeans_pp_init_kernel(
    X: np.ndarray,
    k: int,
    first_idx: int,
) -> np.ndarray:
    """Numba-accelerated k-means++ initialization.

    Args:
        X: (n, d) float64 data matrix
        k: number of centers to select
        first_idx: index of the first selected point

    Returns:
        Array of selected indices (k,)
    """
    n = X.shape[0]
    selected = np.empty(k, dtype=np.int64)
    selected[0] = first_idx
    n_selected = 1

    # Compute initial min distances to the first point
    min_dist_sq = np.full(n, np.inf)
    for j in range(X.shape[1]):
        diff = X[first_idx, j] - X[0, j]
    # We'll compute distances in the loop below

    # Re-compute properly
    for i in range(n):
        d = 0.0
        for j in range(X.shape[1]):
            diff = X[i, j] - X[first_idx, j]
            d += diff * diff
        min_dist_sq[i] = d

    for step in range(1, k):
        # Update distances based on last selected point
        last_idx = selected[n_selected - 1]
        for i in range(n):
            d = 0.0
            for j in range(X.shape[1]):
                diff = X[i, j] - X[last_idx, j]
                d += diff * diff
            if d < min_dist_sq[i]:
                min_dist_sq[i] = d

        # Sample proportional to distance (diversity)
        total = 0.0
        for i in range(n):
            total += min_dist_sq[i]

        if total == 0.0:
            # All points coincide — pick next sequentially
            selected[n_selected] = n_selected
        else:
            # Weighted random selection using cumulative distribution
            r = np.random.random() * total
            cumsum = 0.0
            idx = n - 1
            for i in range(n):
                cumsum += min_dist_sq[i]
                if cumsum >= r:
                    idx = i
                    break
            selected[n_selected] = idx

        n_selected += 1

    return selected


# =========================================================================
# QUERY STRATEGIES
# =========================================================================

class QueryStrategy(str, Enum):
    RANDOM = "random"
    UNCERTAINTY_LEAST_CONFIDENT = "uncertainty_least_confident"
    UNCERTAINTY_MARGIN = "uncertainty_margin"
    UNCERTAINTY_ENTROPY = "uncertainty_entropy"
    BADGE = "badge"
    COST_SENSITIVE = "cost_sensitive"


def _predict_proba(classifier, X) -> np.ndarray:
    """Get probability predictions from any classifier type."""
    if hasattr(classifier, 'predict_proba'):
        return classifier.predict_proba(X)
    elif hasattr(classifier, 'decision_function'):
        # LinearSVC: convert decision function to probabilities
        decision = classifier.decision_function(X)
        if decision.ndim == 1:
            # Binary case
            prob = np.vstack([1 - decision, decision]).T
        else:
            prob = decision
        # Softmax normalization
        exp_prob = np.exp(prob - prob.max(axis=1, keepdims=True))
        return exp_prob / exp_prob.sum(axis=1, keepdims=True)
    else:
        raise ValueError("Classifier has neither predict_proba nor decision_function")


def query_random(
    unlabeled_indices: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random sampling baseline."""
    n = min(n, len(unlabeled_indices))
    return rng.choice(unlabeled_indices, size=n, replace=False)


def query_uncertainty_least_confident(
    classifier,
    X_pool: np.ndarray | object,
    unlabeled_indices: np.ndarray,
    n: int,
) -> np.ndarray:
    """Select samples the classifier is least confident about."""
    X_unlabeled = X_pool[unlabeled_indices]
    proba = _predict_proba(classifier, X_unlabeled)
    # Least confident = 1 - max(P(y|x))
    uncertainty = 1 - np.max(proba, axis=1)
    top_k = np.argsort(uncertainty)[-n:]
    return unlabeled_indices[top_k]


def query_uncertainty_margin(
    classifier,
    X_pool: np.ndarray | object,
    unlabeled_indices: np.ndarray,
    n: int,
) -> np.ndarray:
    """Select samples with smallest margin between top-2 predictions."""
    X_unlabeled = X_pool[unlabeled_indices]
    proba = _predict_proba(classifier, X_unlabeled)
    # Sort probabilities descending
    sorted_proba = np.sort(proba, axis=1)
    # Margin = difference between top-1 and top-2
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    # Smallest margin = most uncertain
    top_k = np.argsort(margin)[:n]
    return unlabeled_indices[top_k]


def query_uncertainty_entropy(
    classifier,
    X_pool: np.ndarray | object,
    unlabeled_indices: np.ndarray,
    n: int,
) -> np.ndarray:
    """Select samples with highest prediction entropy."""
    X_unlabeled = X_pool[unlabeled_indices]
    proba = _predict_proba(classifier, X_unlabeled)
    # Entropy
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    top_k = np.argsort(entropy)[-n:]
    return unlabeled_indices[top_k]


def query_badge(
    classifier,
    X_pool: np.ndarray | object,
    unlabeled_indices: np.ndarray,
    n: int,
    rng: np.random.Generator,
    n_classes: int = 2,
) -> np.ndarray:
    """
    T9: BADGE — Batch Active learning by Diverse Gradient Embeddings.
    
    Combines uncertainty with diversity via k-means++ on gradient proxy
    embeddings. Selects a batch of points that are both uncertain AND
    representative of different regions of the feature space.
    
    Reference: Ash et al. (ICML 2020) — BADGE.
    """
    X_unlabeled = X_pool[unlabeled_indices]
    proba = _predict_proba(classifier, X_unlabeled)
    
    # Compute uncertainty (entropy)
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    
    # Compute gradient proxy embeddings
    # ── #4 OPTIMIZATION: Avoid .toarray() — use sparse-aware projection ──
    # Compute gradient proxy embeddings (uncertainty-weighted features)
    X_dense = X_unlabeled
    if hasattr(X_unlabeled, 'toarray'):
        # Sparse matrix: use random projection that works on sparse input
        if X_unlabeled.shape[1] > 200:
            try:
                from sklearn.random_projection import GaussianRandomProjection
                # GaussianRandomProjection supports sparse input
                rp = GaussianRandomProjection(n_components=200, random_state=42)
                X_dense = rp.fit_transform(X_unlabeled)
            except Exception:
                # Fallback: slice first 200 columns (still sparse)
                X_dense = X_unlabeled[:, :200].toarray()
        else:
            X_dense = X_unlabeled.toarray()
    else:
        X_dense = np.asarray(X_unlabeled)
        if X_dense.shape[1] > 200:
            try:
                from sklearn.random_projection import GaussianRandomProjection
                rp = GaussianRandomProjection(n_components=200, random_state=42)
                X_dense = rp.fit_transform(X_dense)
            except Exception:
                X_dense = X_dense[:, :200]
    
    # Scale features by uncertainty (gradient proxy: uncertain points get larger magnitude)
    uncertainty_weights = entropy / max(entropy.max(), 1e-10)
    grad_proxy = X_dense * uncertainty_weights[:, np.newaxis]
    
    # Select diverse points using k-means++ initialization
    selected = _kmeans_pp_init(grad_proxy, n, rng)
    return unlabeled_indices[selected]


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialization: select k diverse points from X (numba-accelerated)."""
    n = X.shape[0]
    if k >= n:
        return np.arange(n)

    X_dense = np.ascontiguousarray(X, dtype=np.float64)
    first_idx = int(rng.integers(n))

    selected = _kmeans_pp_init_kernel(X_dense, k, first_idx)
    return selected


def query_cost_sensitive(
    classifier,
    X_pool: np.ndarray | object,
    unlabeled_indices: np.ndarray,
    n: int,
    class_weights: dict[int, float] | None = None,
) -> np.ndarray:
    """
    T11: Cost-Sensitive Active Learning.
    
    Weight AL queries by inverse class frequency: examples from rare
    classes get priority. This improves F1-macro on imbalanced datasets.
    """
    X_unlabeled = X_pool[unlabeled_indices]
    proba = _predict_proba(classifier, X_unlabeled)
    
    # Uncertainty (entropy)
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    
    # Predicted class for each sample
    predicted_classes = np.argmax(proba, axis=1)
    
    # Weight by inverse class frequency
    if class_weights:
        sample_weights = np.array([class_weights.get(int(c), 1.0) for c in predicted_classes])
    else:
        sample_weights = np.ones(len(unlabeled_indices))
    
    # Combined score: uncertainty × class weight
    score = entropy * sample_weights
    top_k = np.argsort(score)[-n:]
    return unlabeled_indices[top_k]


def select_queries(
    strategy: QueryStrategy,
    classifier,
    X_pool,
    unlabeled_indices: np.ndarray,
    n: int,
    rng: np.random.Generator,
    n_classes: int | None = None,
    class_weights: dict[int, float] | None = None,
) -> np.ndarray:
    """Dispatch to the appropriate query strategy."""
    if strategy == QueryStrategy.RANDOM:
        return query_random(unlabeled_indices, n, rng)
    elif strategy == QueryStrategy.UNCERTAINTY_LEAST_CONFIDENT:
        return query_uncertainty_least_confident(classifier, X_pool, unlabeled_indices, n)
    elif strategy == QueryStrategy.UNCERTAINTY_MARGIN:
        return query_uncertainty_margin(classifier, X_pool, unlabeled_indices, n)
    elif strategy == QueryStrategy.UNCERTAINTY_ENTROPY:
        return query_uncertainty_entropy(classifier, X_pool, unlabeled_indices, n)
    elif strategy == QueryStrategy.BADGE:
        return query_badge(classifier, X_pool, unlabeled_indices, n, rng,
                           n_classes=n_classes or 2)
    elif strategy == QueryStrategy.COST_SENSITIVE:
        return query_cost_sensitive(classifier, X_pool, unlabeled_indices, n,
                                     class_weights=class_weights or {})
    else:
        raise ValueError(f"Unknown query strategy: {strategy}")


# =========================================================================
# CLASSIFIER FACTORY
# =========================================================================

def create_classifier(
    classifier_type: Literal["rf", "lr", "svm"] = "lr",
    random_state: int = 42,
):
    """Create a classifier instance.

    For SVM, returns a sklearn Pipeline (MaxAbsScaler → LinearSVC) so that
    features are automatically scaled before training/prediction.  This
    prevents the "Liblinear failed to converge" warning.

    Note: n_jobs=1 for LR and RF because n_jobs=-1 has massive
    multiprocessing overhead on the small datasets typical in AL
    (50-500 labeled samples), making it 3-4× slower than single-threaded.
    """
    if classifier_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        )
    elif classifier_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=1,
            solver="liblinear",
        )
    elif classifier_type == "svm":
        svm = LinearSVC(
            max_iter=10000,
            random_state=random_state,
            dual="auto",
        )
        # Wrap in a Pipeline with feature scaling for convergence
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning,
                                   message="Liblinear failed to converge")
            return make_pipeline(MaxAbsScaler(), svm)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


# =========================================================================
# ACTIVE LEARNER
# =========================================================================

class ActiveLearner:
    """
    Active Learning loop manager.

    Tracks labeled/unlabeled sets, applies query strategies,
    and records per-iteration metrics.
    """

    def __init__(
        self,
        X_pool,
        y_pool: np.ndarray,
        texts_pool: tuple[str, ...],
        query_strategy: QueryStrategy = QueryStrategy.UNCERTAINTY_ENTROPY,
        classifier_type: Literal["rf", "lr", "svm"] = "lr",
        batch_size: int = 10,
        random_seed: int = 42,
    ):
        self.X_pool = X_pool
        self.y_pool = y_pool.copy()
        self.texts_pool = texts_pool
        self.query_strategy = query_strategy
        self.classifier_type = classifier_type
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed

        # State tracking
        self.labeled_mask = np.zeros(len(y_pool), dtype=bool)
        self.classifier = None
        self.human_labels_used = 0

        # Per-iteration history
        self.history: list[dict] = []

    @property
    def labeled_indices(self) -> np.ndarray:
        return np.where(self.labeled_mask)[0]

    @property
    def unlabeled_indices(self) -> np.ndarray:
        return np.where(~self.labeled_mask)[0]

    @property
    def n_labeled(self) -> int:
        return int(self.labeled_mask.sum())

    @property
    def n_unlabeled(self) -> int:
        return int((~self.labeled_mask).sum())

    def seed_labels(self, seed_indices: np.ndarray) -> None:
        """Initialize with seed labeled data."""
        self.labeled_mask[seed_indices] = True
        self.human_labels_used += len(seed_indices)
        self._retrain()

    def add_labels(self, indices: np.ndarray, source: str = "human") -> None:
        """Add labels for given indices (from human or weak supervision)."""
        # Ensure indices are valid and not already labeled
        new_indices = indices[~self.labeled_mask[indices]]
        self.labeled_mask[new_indices] = True
        if source == "human":
            self.human_labels_used += len(new_indices)

    def _retrain(self) -> None:
        """Retrain classifier on current labeled set."""
        labeled_idx = self.labeled_indices
        if len(labeled_idx) < 2:
            return
        X_train = self.X_pool[labeled_idx]
        y_train = self.y_pool[labeled_idx]
        self.classifier = create_classifier(
            self.classifier_type, self.random_seed
        )
        # SVM is returned as a Pipeline(MaxAbsScaler, LinearSVC) —
        # scaling and convergence warnings are handled internally.
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate current classifier on test set."""
        if self.classifier is None:
            return {"accuracy": 0.0, "f1_macro": 0.0}

        from sklearn.metrics import accuracy_score, f1_score

        y_pred = self.classifier.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        }

    def query(self, n: int | None = None) -> np.ndarray:
        """Select next batch of samples to label."""
        if n is None:
            n = self.batch_size
        unlabeled = self.unlabeled_indices
        if len(unlabeled) == 0:
            return np.array([], dtype=int)
        n = min(n, len(unlabeled))
        return select_queries(
            self.query_strategy,
            self.classifier,
            self.X_pool,
            unlabeled,
            n,
            self.rng,
        )

    def step(self, X_test=None, y_test=None) -> dict:
        """
        Execute one AL iteration: query → label → retrain → evaluate.

        Returns metrics dict for this iteration.
        """
        # Query
        query_indices = self.query()

        if len(query_indices) == 0:
            return {"step": len(self.history), "accuracy": 0.0, "f1_macro": 0.0}

        # "Ask human" → use ground truth (in real scenario, human provides this)
        self.add_labels(query_indices, source="human")
        self._retrain()

        # Evaluate
        metrics = {
            "step": len(self.history),
            "n_labeled": self.n_labeled,
            "human_labels_used": self.human_labels_used,
            "n_unlabeled": self.n_unlabeled,
        }

        if X_test is not None and y_test is not None:
            eval_metrics = self.evaluate(X_test, y_test)
            metrics.update(eval_metrics)

        self.history.append(metrics)
        return metrics

    def run(
        self,
        budget: int,
        X_test=None,
        y_test=None,
    ) -> list[dict]:
        """
        Run AL loop until budget exhausted.

        Args:
            budget: maximum number of human labels to request
            X_test, y_test: test set for evaluation each step

        Returns:
            List of per-iteration metric dicts
        """
        while self.human_labels_used < budget and self.n_unlabeled > 0:
            self.step(X_test, y_test)

        return self.history
