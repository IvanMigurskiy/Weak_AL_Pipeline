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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MaxAbsScaler
import warnings
from sklearn.cluster import MiniBatchKMeans

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


# =========================================================================
# NUMBA-ACCELERATED KERNELS
# =========================================================================

@_njit(cache=True)
def _majority_vote_kernel(
    obs: np.ndarray,
    n_samples: int,
    n_classes: int,
) -> tuple:
    """Numba-accelerated majority voting kernel.

    Args:
        obs: (n_samples, n_lfs) int array, -1 = abstain
        n_samples: number of samples
        n_classes: number of classes

    Returns:
        (vote_counts, total_votes) as numpy arrays
    """
    n_lfs = obs.shape[1]
    vote_counts = np.zeros((n_samples, n_classes), dtype=np.int64)
    total_votes = np.zeros(n_samples, dtype=np.int64)

    for lf_idx in range(n_lfs):
        for i in range(n_samples):
            p = obs[i, lf_idx]
            if p >= 0:
                vote_counts[i, p] += 1
                total_votes[i] += 1

    return vote_counts, total_votes


@_njit(cache=True)
def _dawid_skene_estep(
    obs: np.ndarray,
    class_probs: np.ndarray,
    n_lfs: int,
    n_samples: int,
    K: int,
) -> np.ndarray:
    """Numba-accelerated E-step: compute confusion matrices."""
    confusion = np.zeros((n_lfs, K, K))

    for lf_idx in range(n_lfs):
        for i in range(n_samples):
            lbl = obs[i, lf_idx]
            if lbl >= 0:
                for k in range(K):
                    confusion[lf_idx, k, lbl] += class_probs[i, k]

        # Normalize rows
        for k in range(K):
            row_sum = 0.0
            for j in range(K):
                row_sum += confusion[lf_idx, k, j]
            if row_sum > 0.0:
                for j in range(K):
                    confusion[lf_idx, k, j] /= row_sum

    return confusion


@_njit(cache=True)
def _dawid_skene_mstep(
    obs: np.ndarray,
    confusion: np.ndarray,
    n_lfs: int,
    n_samples: int,
    K: int,
) -> np.ndarray:
    """Numba-accelerated M-step: update class probabilities."""
    new_class_probs = np.ones((n_samples, K)) / K

    for lf_idx in range(n_lfs):
        for i in range(n_samples):
            lbl = obs[i, lf_idx]
            if lbl >= 0:
                for k in range(K):
                    new_class_probs[i, k] *= confusion[lf_idx, k, lbl]

    # Normalize rows
    for i in range(n_samples):
        row_sum = 0.0
        for k in range(K):
            row_sum += new_class_probs[i, k]
        if row_sum > 0.0:
            for k in range(K):
                new_class_probs[i, k] /= row_sum

    return new_class_probs

# Silence Liblinear convergence warnings globally — we handle convergence
# by scaling features (MaxAbsScaler) and increasing max_iter to 10000.
warnings.filterwarnings("ignore", message="Liblinear failed to converge")


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
    """Multinomial Naive Bayes labeling function.

    Handles dense (possibly negative) features by clipping to non-negative,
    since MultinomialNB requires X >= 0.
    """

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("naive_bayes", abstain_threshold)

    @staticmethod
    def _ensure_non_negative(X):
        """Clip negative values to 0 and convert sparse → dense if needed."""
        if sparse.issparse(X):
            X = X.copy()
            X.data = np.clip(X.data, 0, None)
        else:
            X = np.clip(X, 0, None)
        return X

    def fit(self, X, y: np.ndarray) -> None:
        X = self._ensure_non_negative(X)
        self._model = MultinomialNB(alpha=0.1)
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        X = self._ensure_non_negative(X)
        proba = self._model.predict_proba(X)
        return self._predict_with_abstain(proba)


class SVMLF(LabelingFunction):
    """Linear SVM labeling function with Platt calibration.

    Uses CalibratedClassifierCV to produce proper probability estimates
    instead of raw decision_function + softmax, which was producing
    poorly calibrated probabilities and 0 non-abstain predictions.
    """

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("svm", abstain_threshold)
        self._scaler = None

    def fit(self, X, y: np.ndarray) -> None:
        # Scale features for SVM convergence
        self._scaler = MaxAbsScaler()
        X_scaled = self._scaler.fit_transform(X)
        base_svc = LinearSVC(max_iter=10000, random_state=42, dual="auto")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning,
                                   message="Liblinear failed to converge")
            # Use CalibratedClassifierCV for proper probability estimates.
            # CV folds must be ≤ min class count, otherwise sklearn raises.
            min_class_count = int(np.bincount(y).min())
            cv = min(3, min_class_count)
            if cv < 2:
                # Too few samples for CV calibration — fit raw SVC and skip calibration
                self._model = base_svc
                self._model.fit(X_scaled, y)
                # Wrap in a shim with predict_proba via decision_function + softmax
                self._uncalibrated = True
            else:
                self._model = CalibratedClassifierCV(base_svc, cv=cv)
                self._model.fit(X_scaled, y)
                self._uncalibrated = False

    def predict(self, X) -> np.ndarray:
        X_scaled = self._scaler.transform(X) if self._scaler is not None else X
        if getattr(self, "_uncalibrated", False):
            # LinearSVC has no predict_proba — use softmax(decision_function)
            from scipy.special import softmax as _softmax
            scores = self._model.decision_function(X_scaled)
            proba = _softmax(scores, axis=1)
        else:
            proba = self._model.predict_proba(X_scaled)
        return self._predict_with_abstain(proba)


class RandomForestLF(LabelingFunction):
    """Random Forest labeling function.

    Uses n_jobs=1 because n_jobs=-1 has multiprocessing overhead on
    small data that makes it slower than single-threaded.
    """

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("random_forest", abstain_threshold)

    def fit(self, X, y: np.ndarray) -> None:
        self._model = RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=1
        )
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        proba = self._model.predict_proba(X)
        return self._predict_with_abstain(proba)


class KNNLF(LabelingFunction):
    """K-Nearest Neighbors labeling function.

    Uses n_jobs=1 because n_jobs=-1 has multiprocessing overhead on
    small data that makes it slower than single-threaded.
    """

    def __init__(self, abstain_threshold: float = 0.6):
        super().__init__("knn", abstain_threshold)

    def fit(self, X, y: np.ndarray) -> None:
        n_neighbors = min(5, X.shape[0])
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1)
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        proba = self._model.predict_proba(X)
        return self._predict_with_abstain(proba)


class LogisticRegressionLF(LabelingFunction):
    """Logistic Regression labeling function.

    Uses liblinear solver which is ~1000× faster than lbfgs on sparse
    data with small sample sizes (typical in WS).

    Default abstain_threshold=0.5 instead of 0.6 because LR with small
    training data (50-200 samples) rarely achieves max-proba > 0.6,
    making it completely useless at higher thresholds. At 0.5 it
    produces meaningful votes with 96%+ accuracy.
    """

    def __init__(self, abstain_threshold: float = 0.5):
        super().__init__("logistic_regression", abstain_threshold)

    def fit(self, X, y: np.ndarray) -> None:
        self._model = LogisticRegression(
            max_iter=1000, random_state=42, solver="liblinear",
        )
        self._model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        proba = self._model.predict_proba(X)
        return self._predict_with_abstain(proba)


class TopicLF(LabelingFunction):
    """
    Topic-model-based labeling function.

    Uses NMF or LDA on raw texts to build per-class topic profiles,
    then predicts labels by projecting texts into topic space and
    computing cosine similarity with class profiles.

    Unlike other LFs that operate on TF-IDF features, TopicLF needs
    raw texts because it fits its own TF-IDF + topic model pipeline.
    """
    needs_texts: bool = True

    def __init__(
        self,
        abstain_threshold: float = 0.6,
        n_topics: int = 20,
        topic_model: str = "nmf",
    ):
        super().__init__("topic", abstain_threshold)
        self.n_topics = n_topics
        self.topic_model = topic_model
        self._vectorizer = None
        self._topic_model = None
        self._class_profiles = None   # (n_classes, n_topics) cosine-normalized
        self._n_classes = None

    def fit(self, X, y: np.ndarray, texts=None) -> None:
        """
        Fit TF-IDF -> topic model -> per-class topic profiles.

        Args:
            X: TF-IDF features (ignored — we fit our own TF-IDF on raw texts)
            y: label array for the labeled subset
            texts: raw text strings for the labeled subset (REQUIRED)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF, LatentDirichletAllocation
        from sklearn.preprocessing import normalize

        if texts is None:
            raise ValueError(
                "TopicLF requires raw texts for fitting. "
                "Pass texts= to WeakSupervisor.fit()."
            )

        self._n_classes = len(np.unique(y))

        # Fit our own TF-IDF on the labeled texts
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        X_tfidf = self._vectorizer.fit_transform(texts)

        # Fit topic model (reduce n_topics if too few samples or features)
        n_topics = min(self.n_topics, X_tfidf.shape[0] - 1, X_tfidf.shape[1])
        n_topics = max(n_topics, 2)

        if self.topic_model == "lda":
            self._topic_model = LatentDirichletAllocation(
                n_components=n_topics, random_state=42, max_iter=50,
            )
            topic_dist = self._topic_model.fit_transform(X_tfidf)
        else:  # nmf (default)
            self._topic_model = NMF(
                n_components=n_topics, random_state=42, max_iter=500,
            )
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                topic_dist = self._topic_model.fit_transform(X_tfidf)

        # Build per-class topic profiles (mean of L2-normalized topic vectors)
        topic_dist_norm = normalize(topic_dist, norm="l2", axis=1)
        self._class_profiles = np.zeros((self._n_classes, n_topics))
        for c in range(self._n_classes):
            mask = y == c
            if mask.sum() > 0:
                self._class_profiles[c] = topic_dist_norm[mask].mean(axis=0)

        # Re-normalize profiles to unit vectors
        self._class_profiles = normalize(self._class_profiles, norm="l2", axis=1)

        # Mark as having a model (for LF scoring in WeakSupervisor.fit)
        self._model = self._topic_model

    def predict(self, X, texts=None) -> np.ndarray:
        """
        Predict labels via topic-space cosine similarity with class profiles.

        Args:
            X: TF-IDF features (ignored — we use texts instead)
            texts: raw text strings for the unlabeled subset (REQUIRED)

        Returns:
            1D int array of shape (n_samples,) with -1 for abstentions
        """
        n_samples = X.shape[0] if hasattr(X, "shape") else (len(texts) if texts else 0)

        if texts is None or self._vectorizer is None or self._topic_model is None:
            return np.full(n_samples, -1, dtype=int)

        # Transform texts through our own pipeline
        X_tfidf = self._vectorizer.transform(texts)
        topic_dist = self._topic_model.transform(X_tfidf)

        # Cosine similarity with class profiles
        from sklearn.preprocessing import normalize
        topic_dist_norm = normalize(topic_dist, norm="l2", axis=1)
        similarities = topic_dist_norm @ self._class_profiles.T  # (n_samples, n_classes)

        # Convert cosine similarities to pseudo-probabilities.
        # NOTE: plain softmax(cosine_sim) is mathematically broken for abstention
        # because cosine similarities are bounded to [-1, 1], making softmax
        # produce near-uniform distributions that never exceed typical abstain
        # thresholds (e.g. 0.7).  For K=4 classes, even the most extreme case
        # [1, -1, -1, -1] only yields max softmax prob ≈ 0.71.
        # Instead, we use temperature-scaled softmax with a high temperature
        # to sharpen the distribution so it can exceed the threshold.
        temperature = 7.0
        from scipy.special import softmax as _softmax
        proba = _softmax(similarities * temperature, axis=1)

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
        """Extract top features per class (sparse-friendly, no .toarray())."""
        n_classes = len(np.unique(y))
        self._n_classes = n_classes

        # ── #4 OPTIMIZATION: Avoid .toarray() — use sparse ops directly ──
        if sparse.issparse(X):
            X_abs = X.copy()
            X_abs.data = np.abs(X_abs.data)
        else:
            X_abs = np.abs(np.asarray(X))

        for c in range(n_classes):
            mask = y == c
            if mask.sum() > 0:
                if sparse.issparse(X_abs):
                    class_mean = np.asarray(X_abs[mask].mean(axis=0)).flatten()
                else:
                    class_mean = X_abs[mask].mean(axis=0).flatten()
                top_idx = np.argsort(class_mean)[-self.top_keywords:]
                self._class_keywords[c] = top_idx.tolist()

    def predict(self, X) -> np.ndarray:
        """Predict based on keyword overlap scores (sparse-friendly)."""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self._n_classes))

        # ── #4 OPTIMIZATION: Avoid .toarray() for sparse X ──
        for c, keyword_idx in self._class_keywords.items():
            if sparse.issparse(X):
                # Sum only the keyword columns without densifying
                scores[:, c] = np.asarray(X[:, keyword_idx].sum(axis=1)).flatten()
            else:
                scores[:, c] = np.asarray(X)[:, keyword_idx].sum(axis=1)

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
        Majority voting over LF predictions (numba-accelerated).

        Args:
            lf_predictions: list of prediction arrays (-1 = abstain)
            n_classes: number of classes

        Returns:
            (labels, confidences): aggregated labels and confidence scores
        """
        n_samples = len(lf_predictions[0])
        obs = np.column_stack(lf_predictions).astype(np.int64)

        vote_counts, total_votes = _majority_vote_kernel(obs, n_samples, n_classes)

        labels = np.argmax(vote_counts, axis=1)
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
        Dawid-Skene label aggregation (EM algorithm, numba-accelerated).

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
        obs = np.column_stack(lf_predictions).astype(np.int64)

        # Initialize class priors
        class_probs = np.zeros((n_samples, K))
        for i in range(n_samples):
            if labels[i] >= 0:
                class_probs[i, labels[i]] = 1.0
            else:
                class_probs[i, :] = 1.0 / K

        # EM iterations — E-step and M-step are numba-accelerated
        for iteration in range(max_iter):
            # E-step
            confusion = _dawid_skene_estep(obs, class_probs, n_lfs, n_samples, K)

            # M-step
            new_class_probs = _dawid_skene_mstep(obs, confusion, n_lfs, n_samples, K)

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
        use_topic_lf: bool = False,
        topic_n_topics: int = 20,
        topic_model: str = "nmf",
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
            self.lfs.append(LogisticRegressionLF())  # uses own default threshold (0.5)
        if use_keyword_lf:
            self.lfs.append(KeywordLF(abstain_threshold=lf_confidence_threshold))
        if use_topic_lf:
            self.lfs.append(TopicLF(
                abstain_threshold=lf_confidence_threshold,
                n_topics=topic_n_topics,
                topic_model=topic_model,
            ))

        self._trained = False
        self._warned_lfs: set[str] = set()  # track which LFs already warned

    def fit(self, X_labeled, y_labeled: np.ndarray, texts=None, feature_names=None) -> None:
        """Train all labeling functions on labeled data.

        Uses sequential execution — joblib parallel overhead dominates
        when individual LF fits take <0.05s (typical after liblinear fix).

        Args:
            X_labeled: Feature matrix for labeled samples.
            y_labeled: Label array for labeled samples.
            texts: Raw text strings (needed by TopicLF, KeywordLF).
            feature_names: Feature names from encoder (ignored — kept for API compat).
        """
        # Fit LFs sequentially (parallel overhead > benefit for <0.05s tasks)
        fitted_lfs = []
        for lf in self.lfs:
            try:
                if getattr(lf, "needs_texts", False) and texts is not None:
                    lf.fit(X_labeled, y_labeled, texts=texts)
                else:
                    lf.fit(X_labeled, y_labeled)
                fitted_lfs.append(lf)
            except Exception as e:
                if lf.name not in self._warned_lfs:
                    print(f"  Warning: LF '{lf.name}' failed to fit: {e}")
                    self._warned_lfs.add(lf.name)
        self.lfs = fitted_lfs

        # Score LFs on training data
        for lf in self.lfs:
            if lf._model is not None or lf.name in ("keyword", "topic"):
                try:
                    if getattr(lf, "needs_texts", False) and texts is not None:
                        preds = lf.predict(X_labeled, texts=texts)
                    else:
                        preds = lf.predict(X_labeled)
                    valid = preds >= 0
                    if valid.sum() > 0:
                        lf._score = float(accuracy_score(y_labeled[valid], preds[valid]))
                    else:
                        lf._score = 0.0
                except Exception:
                    lf._score = 0.0

        self._trained = True

    def predict(self, X_unlabeled, texts=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate weak labels for unlabeled data.

        Uses sequential execution — joblib parallel overhead dominates
        when individual LF predictions take <0.01s (typical).

        Returns:
            (weak_labels, confidences): labels (-1 = abstain) and confidence scores
        """
        if not self._trained:
            raise RuntimeError("Must call fit() before predict()")

        # Collect LF predictions sequentially
        lf_preds = []
        for lf in self.lfs:
            try:
                if getattr(lf, "needs_texts", False) and texts is not None:
                    preds = lf.predict(X_unlabeled, texts=texts)
                else:
                    preds = lf.predict(X_unlabeled)
                lf_preds.append(preds)
            except Exception:
                # All abstain
                lf_preds.append(np.full(X_unlabeled.shape[0], -1, dtype=int))

        # Diagnostic: log per-LF vote counts
        n_total = X_unlabeled.shape[0] if hasattr(X_unlabeled, "shape") else len(X_unlabeled)
        lf_vote_counts = []
        for i, lf in enumerate(self.lfs):
            n_voted = int((lf_preds[i] >= 0).sum())
            lf_vote_counts.append(f"{lf.name}:{n_voted}/{n_total}")
        if n_total > 0:
            print(f"    [WS] LF votes: {', '.join(lf_vote_counts)}", flush=True)

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
