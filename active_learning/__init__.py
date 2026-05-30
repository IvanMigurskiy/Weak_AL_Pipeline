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
    X_dense = X_unlabeled.toarray() if hasattr(X_unlabeled, 'toarray') else np.asarray(X_unlabeled)
    
    # Reduce dimensionality if needed (for speed)
    if X_dense.shape[1] > 200:
        try:
            from sklearn.random_projection import GaussianRandomProjection
            rp = GaussianRandomProjection(n_components=200, random_state=42)
            X_dense = rp.fit_transform(X_dense)
        except Exception:
            # Fallback: just use first 200 features
            X_dense = X_dense[:, :200]
    
    # Scale features by uncertainty (gradient proxy: uncertain points get larger magnitude)
    uncertainty_weights = entropy / max(entropy.max(), 1e-10)
    grad_proxy = X_dense * uncertainty_weights[:, np.newaxis]
    
    # Select diverse points using k-means++ initialization
    selected = _kmeans_pp_init(grad_proxy, n, rng)
    return unlabeled_indices[selected]


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialization: select k diverse points from X."""
    n = X.shape[0]
    if k >= n:
        return np.arange(n)
    
    selected = []
    # Start with a random point
    idx = int(rng.integers(n))
    selected.append(idx)
    
    # Compute squared distances to nearest selected point
    min_dist_sq = np.full(n, np.inf)
    
    for _ in range(1, k):
        # Update distances based on last selected point
        last = X[selected[-1]:selected[-1]+1]
        dists = np.sum((X - last) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dists)
        
        # Sample proportional to distance (diversity)
        total = min_dist_sq.sum()
        if total == 0:
            idx = int(rng.integers(n))
        else:
            probs = min_dist_sq / total
            idx = int(rng.choice(n, p=probs))
        selected.append(idx)
    
    return np.array(selected)


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
    classifier_type: Literal["rf", "lr", "svm"] = "rf",
    random_state: int = 42,
):
    """Create a classifier instance."""
    if classifier_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
    elif classifier_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
        )
    elif classifier_type == "svm":
        return LinearSVC(
            max_iter=2000,
            random_state=random_state,
        )
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
        classifier_type: Literal["rf", "lr", "svm"] = "rf",
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
