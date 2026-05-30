"""
Weak Supervision module (AutoWS-style).

Implements:
- Labeling functions: TF-IDF based classifiers (NB, SVM, RF, KNN, LR), keyword-based
- Label aggregation: majority voting, Dawid-Skene
- WeakClust (WeakAL-style cluster label propagation)
- WeakCert (WeakAL-style classifier certainty auto-labeling)
- LF scoring and greedy selection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans


# =========================================================================
# LABELING FUNCTIONS
# =========================================================================

class LabelingFunction(ABC):
    """Base class for labeling functions."""

    def __init__(self, name: str, abstain_threshold: float = 0.6):
        self.name = name
        self.abstain_threshold = abstain_threshold
        self._model = None
        self._score: float | None = None

    @property
    def score(self) -> float:
        return self._score or 0.0

    @abstractmethod
    def fit(self, X, y: np.ndarray) -> None:
        """Train the labeling function on labeled data."""
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Predict labels for unlabeled data.
        Returns -1 for abstentions (low confidence).
        """
        ...

    def _predict_with_abstain(self, proba: np.ndarray) -> np.ndarray:
        """Convert probabilities to labels with abstention."""
        max_proba = np.max(proba, axis=1)
        labels = np.argmax(proba, axis=1)
        # Abstain where confidence is below threshold
        labels[max_proba < self.abstain_threshold] = -1
        return labels


class NaiveBayesLF(LabelingFunction):
    """Multinomial Naive Bayes labeling function."""

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("naive_bayes", abstain_threshold)

    def fit(self, X, y: np.ndarray) -> None:
        self._model = MultinomialNB(alpha=0.1)
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        proba = self._model.predict_proba(X)
        return self._predict_with_abstain(proba)


class SVMLF(LabelingFunction):
    """Linear SVM labeling function."""

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("svm", abstain_threshold)

    def fit(self, X, y: np.ndarray) -> None:
        self._model = LinearSVC(max_iter=2000, random_state=42)
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        decision = self._model.decision_function(X)
        if decision.ndim == 1:
            prob = np.vstack([1 - decision, decision]).T
        else:
            prob = decision
        # Softmax
        exp_prob = np.exp(prob - prob.max(axis=1, keepdims=True))
        proba = exp_prob / exp_prob.sum(axis=1, keepdims=True)
        return self._predict_with_abstain(proba)


class RandomForestLF(LabelingFunction):
    """Random Forest labeling function."""

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("random_forest", abstain_threshold)

    def fit(self, X, y: np.ndarray) -> None:
        self._model = RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=-1
        )
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        proba = self._model.predict_proba(X)
        return self._predict_with_abstain(proba)


class KNNLF(LabelingFunction):
    """K-Nearest Neighbors labeling function."""

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("knn", abstain_threshold)

    def fit(self, X, y: np.ndarray) -> None:
        n_neighbors = min(5, X.shape[0])
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        proba = self._model.predict_proba(X)
        return self._predict_with_abstain(proba)


class LogisticRegressionLF(LabelingFunction):
    """Logistic Regression labeling function."""

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("logistic_regression", abstain_threshold)

    def fit(self, X, y: np.ndarray) -> None:
        self._model = LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        )
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        proba = self._model.predict_proba(X)
        return self._predict_with_abstain(proba)


class KeywordLF(LabelingFunction):
    """
    Keyword-based labeling function.
    Assigns labels based on class-specific keyword matching.
    """

    def __init__(self, abstain_threshold: float = 0.6, top_keywords: int = 10):
        super().__init__("keyword", abstain_threshold)
        self.top_keywords = top_keywords
        self._class_keywords: dict[int, list[str]] = {}
        self._vectorizer = None
        self._feature_names = None

    def fit(self, X, y: np.ndarray) -> None:
        """Extract top TF-IDF keywords per class."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # If X is sparse (already TF-IDF), use feature names from parent vectorizer
        # For keyword matching, we'll use a simpler approach: per-class mean TF-IDF
        n_classes = len(np.unique(y))
        self._n_classes = n_classes

        if sparse.issparse(X):
            for c in range(n_classes):
                mask = y == c
                if mask.sum() > 0:
                    class_mean = np.asarray(X[mask].mean(axis=0)).flatten()
                    top_idx = np.argsort(class_mean)[-self.top_keywords:]
                    self._class_keywords[c] = top_idx.tolist()

    def predict(self, X) -> np.ndarray:
        """Predict based on keyword overlap scores."""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self._n_classes))

        if sparse.issparse(X):
            X_dense = np.asarray(X.tocsr()[:n_samples].toarray())
        else:
            X_dense = X

        for c, keyword_idx in self._class_keywords.items():
            scores[:, c] = X_dense[:, keyword_idx].sum(axis=1)

        # Normalize to pseudo-probabilities
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        proba = scores / row_sums

        return self._predict_with_abstain(proba)


# =========================================================================
# LABEL AGGREGATION
# =========================================================================

class LabelAggregator:
    """Aggregate noisy labels from multiple labeling functions."""

    @staticmethod
    def majority_vote(
        lf_predictions: list[np.ndarray],
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Majority voting over LF predictions.

        Args:
            lf_predictions: list of prediction arrays (-1 = abstain)
            n_classes: number of classes

        Returns:
            (labels, confidences): aggregated labels and confidence scores
        """
        n_samples = len(lf_predictions[0])
        vote_counts = np.zeros((n_samples, n_classes), dtype=int)

        for preds in lf_predictions:
            for i, p in enumerate(preds):
                if p >= 0:  # not abstaining
                    vote_counts[i, p] += 1

        labels = np.argmax(vote_counts, axis=1)
        total_votes = vote_counts.sum(axis=1)
        with np.errstate(invalid='ignore'):
            confidences = np.where(
                total_votes > 0,
                vote_counts[np.arange(n_samples), labels] / np.maximum(total_votes, 1),
                0.0,
            )

        # Mark samples with no votes
        no_votes = total_votes == 0
        labels[no_votes] = -1
        confidences[no_votes] = 0.0

        return labels, confidences

    @staticmethod
    def dawid_skene(
        lf_predictions: list[np.ndarray],
        n_classes: int,
        max_iter: int = 20,
        tol: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Dawid-Skene label aggregation (EM algorithm).

        More sophisticated than majority vote — models per-LF error rates.
        """
        n_samples = len(lf_predictions[0])
        n_lfs = len(lf_predictions)
        K = n_classes

        # Initialize with majority vote
        labels, confidences = LabelAggregator.majority_vote(
            lf_predictions, n_classes
        )

        # Build observation matrix: (n_samples, n_lfs) → label or -1
        obs = np.column_stack(lf_predictions)  # (n_samples, n_lfs)

        # Initialize class priors
        class_probs = np.zeros((n_samples, K))
        for i in range(n_samples):
            if labels[i] >= 0:
                class_probs[i, labels[i]] = 1.0
            else:
                class_probs[i, :] = 1.0 / K

        # EM iterations
        for iteration in range(max_iter):
            # E-step: estimate per-LF confusion matrices
            confusion = np.zeros((n_lfs, K, K))
            for lf_idx in range(n_lfs):
                for i in range(n_samples):
                    if obs[i, lf_idx] >= 0:
                        for k in range(K):
                            confusion[lf_idx, k, obs[i, lf_idx]] += class_probs[i, k]

                # Normalize rows
                row_sums = confusion[lf_idx].sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                confusion[lf_idx] /= row_sums

            # M-step: update class probabilities
            new_class_probs = np.ones((n_samples, K)) / K
            for lf_idx in range(n_lfs):
                for i in range(n_samples):
                    if obs[i, lf_idx] >= 0:
                        for k in range(K):
                            new_class_probs[i, k] *= confusion[lf_idx, k, obs[i, lf_idx]]

            # Normalize
            row_sums = new_class_probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            new_class_probs /= row_sums

            # Check convergence
            diff = np.abs(new_class_probs - class_probs).max()
            class_probs = new_class_probs

            if diff < tol:
                break

        labels = np.argmax(class_probs, axis=1)
        confidences = np.max(class_probs, axis=1)

        # Mark abstentions (no LF voted)
        no_votes = np.all(obs == -1, axis=1)
        labels[no_votes] = -1
        confidences[no_votes] = 0.0

        return labels, confidences


# =========================================================================
# WEAK SUPERVISOR
# =========================================================================

class WeakSupervisor:
    """
    AutoWS-style weak supervision pipeline.

    1. Train multiple labeling functions on labeled data
    2. Generate noisy labels for unlabeled data
    3. Score and optionally select top LFs (greedy search)
    4. Aggregate noisy labels into probabilistic labels
    5. Return weak labels with confidence scores
    """

    def __init__(
        self,
        n_classes: int,
        lf_confidence_threshold: float = 0.6,
        label_model: Literal["majority_vote", "dawid_skene"] = "majority_vote",
        use_tfidf_lf: bool = True,
        use_nb_lf: bool = True,
        use_svm_lf: bool = True,
        use_rf_lf: bool = True,
        use_knn_lf: bool = True,
        use_lr_lf: bool = True,
        use_keyword_lf: bool = True,
    ):
        self.n_classes = n_classes
        self.lf_confidence_threshold = lf_confidence_threshold
        self.label_model = label_model

        # Build LF pool
        self.lfs: list[LabelingFunction] = []
        if use_nb_lf:
            self.lfs.append(NaiveBayesLF(abstain_threshold=lf_confidence_threshold))
        if use_svm_lf:
            self.lfs.append(SVMLF(abstain_threshold=lf_confidence_threshold))
        if use_rf_lf:
            self.lfs.append(RandomForestLF(abstain_threshold=lf_confidence_threshold))
        if use_knn_lf:
            self.lfs.append(KNNLF(abstain_threshold=lf_confidence_threshold))
        if use_lr_lf:
            self.lfs.append(LogisticRegressionLF(abstain_threshold=lf_confidence_threshold))
        if use_keyword_lf:
            self.lfs.append(KeywordLF(abstain_threshold=lf_confidence_threshold))

        self._trained = False

    def fit(self, X_labeled, y_labeled: np.ndarray) -> None:
        """Train all labeling functions on labeled data."""
        for lf in self.lfs:
            try:
                lf.fit(X_labeled, y_labeled)
            except Exception as e:
                print(f"  Warning: LF '{lf.name}' failed to fit: {e}")

        # Score LFs on training data
        for lf in self.lfs:
            if lf._model is not None or lf.name == "keyword":
                try:
                    preds = lf.predict(X_labeled)
                    valid = preds >= 0
                    if valid.sum() > 0:
                        lf._score = float(accuracy_score(y_labeled[valid], preds[valid]))
                    else:
                        lf._score = 0.0
                except Exception:
                    lf._score = 0.0

        self._trained = True

    def predict(self, X_unlabeled) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate weak labels for unlabeled data.

        Returns:
            (weak_labels, confidences): labels (-1 = abstain) and confidence scores
        """
        if not self._trained:
            raise RuntimeError("Must call fit() before predict()")

        # Collect LF predictions
        lf_preds = []
        for lf in self.lfs:
            try:
                preds = lf.predict(X_unlabeled)
                lf_preds.append(preds)
            except Exception:
                # All abstain
                lf_preds.append(np.full(X_unlabeled.shape[0], -1, dtype=int))

        # Aggregate
        if self.label_model == "majority_vote":
            labels, confidences = LabelAggregator.majority_vote(
                lf_preds, self.n_classes
            )
        elif self.label_model == "dawid_skene":
            labels, confidences = LabelAggregator.dawid_skene(
                lf_preds, self.n_classes
            )
        else:
            raise ValueError(f"Unknown label model: {self.label_model}")

        return labels, confidences

    def select_top_lfs(self, X_dev, y_dev: np.ndarray, top_k: int | None = None) -> None:
        """
        Greedy selection of top labeling functions (AutoWS optimization).

        Evaluates LFs on dev set and keeps only the best subset.
        """
        if top_k is None:
            # Auto-select: keep LFs above median score
            scores = [lf.score for lf in self.lfs]
            median_score = np.median(scores)
            self.lfs = [lf for lf in self.lfs if lf.score >= median_score]
        else:
            sorted_lfs = sorted(self.lfs, key=lambda lf: lf.score, reverse=True)
            self.lfs = sorted_lfs[:top_k]


# =========================================================================
# WEAK CLUSTER (WeakAL-style)
# =========================================================================

class WeakCluster:
    """
    WeakClust: propagate majority label in cluster to unlabeled members.

    From WeakAL paper: if a cluster has enough labeled data (γ threshold)
    and the labels are homogeneous (β threshold), propagate the majority
    label to all unlabeled members.
    """

    def __init__(
        self,
        beta: float = 0.8,     # minimum cluster homogeneity
        gamma: float = 0.3,    # minimum labeled ratio in cluster
        n_clusters: int | None = None,
    ):
        self.beta = beta
        self.gamma = gamma
        self.n_clusters = n_clusters
        self._cluster_model = None
        self._cluster_labels = None

    def fit(self, X, labeled_mask: np.ndarray, y: np.ndarray) -> None:
        """Cluster data and assign cluster IDs."""
        if self.n_clusters is None:
            self.n_clusters = max(X.shape[0] // 8, 2)

        self._cluster_model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=min(100, X.shape[0]),
        )
        self._cluster_labels = self._cluster_model.fit_predict(X)

    def predict(
        self,
        labeled_mask: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate labels in homogeneous clusters.

        Returns:
            (weak_indices, weak_labels): indices and labels for auto-labeled samples
        """
        if self._cluster_labels is None:
            raise RuntimeError("Must call fit() first")

        weak_indices = []
        weak_labels = []

        for cluster_id in range(self.n_clusters):
            cluster_mask = self._cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Labeled members in this cluster
            labeled_in_cluster = cluster_mask & labeled_mask
            n_labeled_in_cluster = labeled_in_cluster.sum()
            n_cluster = len(cluster_indices)

            if n_cluster == 0 or n_labeled_in_cluster == 0:
                continue

            # Check gamma: enough labeled ratio?
            labeled_ratio = n_labeled_in_cluster / n_cluster
            if labeled_ratio < self.gamma:
                continue

            # Check beta: cluster homogeneity
            labels_in_cluster = y[labeled_in_cluster]
            unique, counts = np.unique(labels_in_cluster, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            homogeneity = counts.max() / counts.sum()

            if homogeneity >= self.beta:
                # Propagate to unlabeled members
                unlabeled_in_cluster = cluster_mask & ~labeled_mask
                unlabeled_indices = np.where(unlabeled_in_cluster)[0]
                if len(unlabeled_indices) > 0:
                    weak_indices.extend(unlabeled_indices.tolist())
                    weak_labels.extend([majority_label] * len(unlabeled_indices))

        return np.array(weak_indices, dtype=int), np.array(weak_labels, dtype=int)


# =========================================================================
# WEAK CERTAINT (WeakAL-style)
# =========================================================================

class WeakCertainty:
    """
    WeakCert: use classifier's most-certain predictions as weak labels.

    From WeakAL paper: if the classifier's prediction confidence exceeds
    threshold α, auto-label that sample.
    """

    def __init__(self, alpha: float = 0.85):
        self.alpha = alpha

    def predict(
        self,
        classifier,
        X_pool,
        unlabeled_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Auto-label high-certainty predictions.

        Returns:
            (weak_indices, weak_labels): indices and labels for auto-labeled samples
        """
        if len(unlabeled_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        X_unlabeled = X_pool[unlabeled_indices]

        # Get probabilities
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(X_unlabeled)
        elif hasattr(classifier, 'decision_function'):
            decision = classifier.decision_function(X_unlabeled)
            if decision.ndim == 1:
                prob = np.vstack([1 - decision, decision]).T
            else:
                prob = decision
            exp_prob = np.exp(prob - prob.max(axis=1, keepdims=True))
            proba = exp_prob / exp_prob.sum(axis=1, keepdims=True)
        else:
            return np.array([], dtype=int), np.array([], dtype=int)

        # Find high-certainty predictions
        max_proba = np.max(proba, axis=1)
        predicted_labels = np.argmax(proba, axis=1)

        # Select samples above certainty threshold
        certain_mask = max_proba >= self.alpha
        certain_indices = unlabeled_indices[certain_mask]
        certain_labels = predicted_labels[certain_mask]

        return certain_indices, certain_labels
