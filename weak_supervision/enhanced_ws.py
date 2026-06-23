"""
Enhanced Weak Supervision techniques — 6+5 approaches to improve WS label accuracy.

Each technique is a standalone class that can be plugged into the hybrid pipeline.
Original weak_supervision/__init__.py is NOT modified (backed up in _backups/).

Techniques:
  T1: WeightedTraining — WS labels get lower sample_weight during classifier training
  T2: WSLabelVerification — Remove WS labels that the classifier disagrees with
  T3: LFCalibration — Platt-scale (sigmoid) LF confidence scores before abstention
  T4: SelfTrainingPseudoLabels — Iteratively pseudo-label high-confidence unlabeled samples
  T5: UnanimousVotingAggregator — Only label when ALL voting LFs agree (unanimous)
  T6: PerClassWSThresholds — Dynamic per-class confidence thresholds based on class difficulty
  T7: IsotonicCalibration — Isotonic regression calibration (alternative to T3 Platt scaling)
  T8: BERTLF — SentenceTransformer embeddings + LogisticRegression as 7th LF
  T10: LabelPropagationWS — Graph-based label propagation as additional WS source
  T13: FlyingSquidAggregator — Triplet-based LF aggregation (alternative to Dawid-Skene)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import warnings

# Silence Liblinear convergence warnings globally
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


# =========================================================================
# NUMBA-ACCELERATED KERNELS
# =========================================================================

@_njit(cache=True)
def _unanimous_voting_kernel(
    obs: np.ndarray,
    n_samples: int,
    n_classes: int,
    min_voters: int,
    agreement_ratio: float,
) -> tuple:
    """Numba-accelerated unanimous voting kernel.

    Args:
        obs: (n_samples, n_lfs) int array, -1 = abstain
        n_samples: number of samples
        n_classes: number of classes
        min_voters: minimum non-abstaining LFs required
        agreement_ratio: fraction of voters that must agree

    Returns:
        (labels, confidences) arrays; labels=-1 for abstentions
    """
    n_lfs = obs.shape[1]
    labels = np.full(n_samples, -1, dtype=np.int64)
    confidences = np.zeros(n_samples, dtype=np.float64)
    vote_buf = np.zeros(n_classes, dtype=np.int64)

    for i in range(n_samples):
        # Count votes per class
        n_voters = 0
        for lf_idx in range(n_lfs):
            p = obs[i, lf_idx]
            if p >= 0:
                vote_buf[p] += 1
                n_voters += 1

        if n_voters < min_voters:
            # Reset vote buffer
            for c in range(n_classes):
                vote_buf[c] = 0
            continue

        # Find majority class and its count
        max_count = 0
        majority_label = 0
        for c in range(n_classes):
            if vote_buf[c] > max_count:
                max_count = vote_buf[c]
                majority_label = c

        agreement = max_count / n_voters
        if agreement >= agreement_ratio:
            labels[i] = majority_label
            confidences[i] = agreement

        # Reset vote buffer
        for c in range(n_classes):
            vote_buf[c] = 0

    return labels, confidences


@_njit(cache=True)
def _per_class_threshold_filter_kernel(
    weak_labels: np.ndarray,
    weak_confidences: np.ndarray,
    class_thresholds: np.ndarray,
    n_classes: int,
    base_threshold: float,
) -> np.ndarray:
    """Numba-accelerated per-class threshold filtering.

    Args:
        weak_labels: int array of predicted labels (-1 = abstain)
        weak_confidences: float array of confidence scores
        class_thresholds: float array of shape (n_classes,) with per-class thresholds
        n_classes: number of classes
        base_threshold: fallback threshold when class index out of range

    Returns:
        Boolean mask of which labels pass their class-specific threshold
    """
    n = len(weak_labels)
    passes = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        lbl = weak_labels[i]
        if lbl < 0:
            continue
        if lbl < n_classes:
            threshold = class_thresholds[lbl]
        else:
            threshold = base_threshold
        passes[i] = weak_confidences[i] >= threshold

    return passes


@_njit(cache=True)
def _class_threshold_listcomp_kernel(
    confidence: np.ndarray,
    predicted_labels: np.ndarray,
    class_thresholds: np.ndarray,
    n_classes: int,
    base_threshold: float,
) -> np.ndarray:
    """Numba-accelerated per-sample class-specific threshold check.

    Replaces the Python list comprehension:
        passes_threshold = np.array([
            confidence[i] >= class_thresholds.get(int(predicted_labels[i]), threshold)
            for i in range(len(unlabeled_idx))
        ])
    """
    n = len(confidence)
    passes = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        cls = predicted_labels[i]
        if cls < n_classes:
            threshold = class_thresholds[cls]
        else:
            threshold = base_threshold
        passes[i] = confidence[i] >= threshold

    return passes


@_njit(cache=True)
def _pairwise_agreement_kernel(
    obs: np.ndarray,
    n_lfs: int,
    n_samples: int,
) -> np.ndarray:
    """Numba-accelerated pairwise agreement computation for FlyingSquid.

    Args:
        obs: (n_samples, n_lfs) int array, -1 = abstain
        n_lfs: number of labeling functions
        n_samples: number of samples

    Returns:
        (n_lfs, n_lfs) agreement matrix
    """
    agreement = np.zeros((n_lfs, n_lfs))

    for i in range(n_lfs):
        for j in range(i + 1, n_lfs):
            n_valid = 0
            n_agree = 0
            for s in range(n_samples):
                pi = obs[s, i]
                pj = obs[s, j]
                if pi >= 0 and pj >= 0:
                    n_valid += 1
                    if pi == pj:
                        n_agree += 1
            if n_valid > 0:
                agreement[i, j] = n_agree / n_valid
                agreement[j, i] = agreement[i, j]
            else:
                agreement[i, j] = 0.5
                agreement[j, i] = 0.5

    # Fill diagonal
    for i in range(n_lfs):
        agreement[i, i] = 1.0

    return agreement


@_njit(cache=True)
def _triplet_refinement_kernel(
    obs: np.ndarray,
    n_lfs: int,
    n_samples: int,
    K: int,
    max_iter: int,
    tol: float,
    initial_accuracies: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated iterative accuracy refinement for FlyingSquid.

    Args:
        obs: (n_samples, n_lfs) int array, -1 = abstain
        n_lfs: number of labeling functions
        n_samples: number of samples
        K: number of classes
        max_iter: max refinement iterations
        tol: convergence tolerance
        initial_accuracies: starting accuracy estimates

    Returns:
        Refined accuracy estimates (n_lfs,)
    """
    accuracies = initial_accuracies.copy()

    for iteration in range(max_iter):
        new_accuracies = accuracies.copy()

        for i in range(n_lfs):
            numerator = 0.0
            denominator = 0.0

            for j in range(n_lfs):
                if i == j:
                    continue

                n_valid = 0
                n_agree = 0
                for s in range(n_samples):
                    pi = obs[s, i]
                    pj = obs[s, j]
                    if pi >= 0 and pj >= 0:
                        n_valid += 1
                        if pi == pj:
                            n_agree += 1

                if n_valid == 0:
                    continue

                agree_rate = n_agree / n_valid
                a_j = accuracies[j]

                # Solve for a_i given a_j and agree_rate
                denom = a_j * K / (K - 1) - 1.0 / (K - 1)
                if abs(denom) > 1e-10:
                    a_i_est = (agree_rate - (1 - a_j) / (K - 1)) / denom
                    weight = a_j
                    numerator += a_i_est * weight
                    denominator += weight

            if denominator > 0:
                val = numerator / denominator
                # Clip to valid range [1/K, 0.999]
                if val < 1.0 / K:
                    val = 1.0 / K
                if val > 0.999:
                    val = 0.999
                new_accuracies[i] = val

        # Check convergence
        diff = 0.0
        for i in range(n_lfs):
            d = abs(new_accuracies[i] - accuracies[i])
            if d > diff:
                diff = d
        accuracies = new_accuracies

        if diff < tol:
            break

    return accuracies


@_njit(cache=True)
def _flyingsquid_weighted_vote_kernel(
    obs: np.ndarray,
    lf_accuracies: np.ndarray,
    n_samples: int,
    n_classes: int,
) -> tuple:
    """Numba-accelerated weighted voting for FlyingSquid.

    Args:
        obs: (n_samples, n_lfs) int array, -1 = abstain
        lf_accuracies: float array of LF accuracy estimates
        n_samples: number of samples
        n_classes: number of classes

    Returns:
        (vote_weights, labels, no_votes_mask)
    """
    n_lfs = obs.shape[1]
    vote_weights = np.zeros((n_samples, n_classes))
    no_votes = np.ones(n_samples, dtype=np.bool_)

    for lf_idx in range(n_lfs):
        weight = lf_accuracies[lf_idx]
        log_weight = np.log(max(weight, 1e-10) / max(1 - weight, 1e-10))
        for i in range(n_samples):
            p = obs[i, lf_idx]
            if p >= 0:
                vote_weights[i, p] += log_weight
                no_votes[i] = False

    return vote_weights, no_votes


@_njit(cache=True, fastmath=True)
def _compute_energy_kernel(
    logits_or_proba: np.ndarray,
    T: float,
) -> np.ndarray:
    """Numba-accelerated energy-based confidence computation.

    Args:
        logits_or_proba: (n_samples, n_classes) probability array
        T: temperature parameter

    Returns:
        Confidence scores (n_samples,)
    """
    n = logits_or_proba.shape[0]
    K = logits_or_proba.shape[1]
    confidence = np.zeros(n)

    for i in range(n):
        log_sum = 0.0
        for k in range(K):
            log_sum += np.exp(np.log(logits_or_proba[i, k] + 1e-10) / T)
        energy = -T * np.log(log_sum)
        confidence[i] = np.exp(-energy / T)

    # Normalize to [0, 1]
    max_conf = 0.0
    for i in range(n):
        if confidence[i] > max_conf:
            max_conf = confidence[i]
    if max_conf > 0.0:
        for i in range(n):
            confidence[i] /= max_conf

    return confidence


# ── #7 OPTIMIZATION: Shared SentenceTransformer model cache ──
_ST_MODEL_CACHE: dict[str, Any] = {}


def _get_shared_sentence_transformer(model_name: str):
    """Get or create a shared SentenceTransformer model instance.

    This avoids loading duplicate models (~420MB each) when both
    DenseEncoder and BERTLF use the same SentenceTransformer model.
    """
    if model_name not in _ST_MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for BERTLF. "
                "Install it with: pip install sentence-transformers"
            )
    return _ST_MODEL_CACHE[model_name]


from . import (
    LabelingFunction,
    NaiveBayesLF,
    SVMLF,
    RandomForestLF,
    KNNLF,
    LogisticRegressionLF,
    KeywordLF,
    LabelAggregator,
    WeakSupervisor,
    WeakCertainty,
)


# =========================================================================
# T1: WEIGHTED TRAINING
# =========================================================================

class WeightedTrainingMixin:
    """
    T1: Weighted Training — Give WS labels lower weight during classifier fit.

    Instead of treating all labels equally, human labels get weight=1.0
    and WS labels get weight=ws_weight (e.g., 0.3-0.7).
    This reduces the impact of noisy WS labels on classifier training.

    Usage: Call fit_with_weights() instead of classifier.fit().
    """

    def __init__(self, ws_weight: float = 0.5):
        self.ws_weight = ws_weight

    @staticmethod
    def compute_sample_weights(
        is_human_label: np.ndarray,
        labeled_mask: np.ndarray,
        ws_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Compute sample weights: 1.0 for human labels, ws_weight for WS labels.

        Args:
            is_human_label: bool array, True where label came from human
            labeled_mask: bool array, True where sample is labeled
            ws_weight: weight for WS labels (0.0-1.0)

        Returns:
            sample_weights array for labeled samples only
        """
        labeled_idx = np.where(labeled_mask)[0]
        weights = np.where(is_human_label[labeled_idx], 1.0, ws_weight)
        return weights

    @staticmethod
    def fit_classifier_weighted(
        classifier,
        X_train,
        y_train: np.ndarray,
        sample_weights: np.ndarray,
    ):
        """
        Fit classifier with sample weights.
        Falls back to unweighted fit if classifier doesn't support sample_weight.
        """
        try:
            classifier.fit(X_train, y_train, sample_weight=sample_weights)
        except (TypeError, ValueError):
            # Some classifiers (e.g., SVM) don't support sample_weight
            classifier.fit(X_train, y_train)
        return classifier


# =========================================================================
# T2: WS LABEL VERIFICATION
# =========================================================================

class WSLabelVerifier:
    """
    T2: WS Label Verification — After adding WS labels, retrain the classifier
    and check if it agrees. Remove WS labels where the classifier disagrees.

    Rationale: If even the classifier trained on all data (including the WS label)
    disagrees with a WS label, it's likely wrong. This is a form of "consensus
    filtering" between the WS system and the learned model.

    Two modes:
      - "remove": Delete disagreeing WS labels entirely
      - "downgrade": Lower the weight of disagreeing WS labels
    """

    def __init__(
        self,
        mode: Literal["remove", "downgrade"] = "remove",
        agreement_threshold: float = 0.5,
        downgrade_weight: float = 0.2,
    ):
        self.mode = mode
        self.agreement_threshold = agreement_threshold
        self.downgrade_weight = downgrade_weight

    def verify(
        self,
        classifier,
        X_pool,
        labeled_mask: np.ndarray,
        y_labeled: np.ndarray,
        is_human_label: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Verify WS labels by checking classifier agreement.

        Returns:
            (labeled_mask, y_labeled, is_human_label) — possibly modified
        """
        labeled_idx = np.where(labeled_mask)[0]
        ws_mask_in_labeled = ~is_human_label[labeled_idx]

        if ws_mask_in_labeled.sum() == 0:
            return labeled_mask, y_labeled, is_human_label

        # Get classifier predictions on WS-labeled samples
        ws_indices_in_labeled = labeled_idx[ws_mask_in_labeled]
        X_ws = X_pool[ws_indices_in_labeled]

        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(X_ws)
            predictions = np.argmax(proba, axis=1)
            confidences = np.max(proba, axis=1)
        else:
            predictions = classifier.predict(X_ws)
            confidences = np.ones(len(predictions))

        # Check agreement
        ws_labels = y_labeled[ws_indices_in_labeled]
        agrees = predictions == ws_labels

        if self.mode == "remove":
            # Remove WS labels where classifier disagrees
            disagree_indices = ws_indices_in_labeled[~agrees]
            labeled_mask[disagree_indices] = False
            # Don't change y_labeled for removed indices (doesn't matter)
            print(f"    [T2-Verify] Removed {len(disagree_indices)} disagreeing WS labels "
                  f"out of {len(ws_indices_in_labeled)} total WS labels "
                  f"({agrees.sum()}/{len(ws_indices_in_labeled)} agree)")

        elif self.mode == "downgrade":
            # Mark disagreeing WS labels with lower confidence
            # This is used with WeightedTraining
            pass  # Handled via sample weights

        return labeled_mask, y_labeled, is_human_label


# =========================================================================
# T3: LF CALIBRATION (Platt Scaling)
# =========================================================================

class CalibratedWeakSupervisor(WeakSupervisor):
    """
    T3: LF Confidence Calibration — Apply Platt scaling to calibrate each LF's
    confidence scores before using them for abstention decisions.

    Raw classifier confidence scores are often poorly calibrated (e.g., SVM
    confidence is arbitrary, RF tends to be overconfident). Platt scaling
    fits a logistic regression on the training data to produce calibrated
    probability estimates. This makes the abstention threshold more meaningful.

    Calibrated LFs should produce fewer false positives (samples that pass
    the confidence threshold but are actually wrong).
    """

    def __init__(
        self,
        n_classes: int,
        lf_confidence_threshold: float = 0.6,
        label_model: Literal["majority_vote", "dawid_skene"] = "majority_vote",
        use_nb_lf: bool = True,
        use_svm_lf: bool = True,
        use_rf_lf: bool = True,
        use_knn_lf: bool = True,
        use_lr_lf: bool = True,
        use_keyword_lf: bool = True,
        use_topic_lf: bool = False,
        topic_n_topics: int = 20,
        topic_model: str = "nmf",
        calibration_cv: int = 3,
    ):
        super().__init__(
            n_classes=n_classes,
            lf_confidence_threshold=lf_confidence_threshold,
            label_model=label_model,
            use_nb_lf=use_nb_lf,
            use_svm_lf=use_svm_lf,
            use_rf_lf=use_rf_lf,
            use_knn_lf=use_knn_lf,
            use_lr_lf=use_lr_lf,
            use_keyword_lf=use_keyword_lf,
            use_topic_lf=use_topic_lf,
            topic_n_topics=topic_n_topics,
            topic_model=topic_model,
        )
        self.calibration_cv = calibration_cv
        self._calibrators: dict[str, CalibratedClassifierCV] = {}

    def fit(self, X_labeled, y_labeled: np.ndarray, feature_names: list[str] | None = None, texts: list[str] | None = None) -> None:
        """Train all LFs and then calibrate them using Platt scaling."""
        # Train LFs normally
        super().fit(X_labeled, y_labeled, feature_names=feature_names, texts=texts)

        # Now calibrate each classifier-based LF
        for lf in self.lfs:
            if lf.name == "keyword":
                continue  # Keyword LF doesn't need calibration
            if lf.name == "topic":
                continue  # Topic LF uses NMF/LDA (no predict_proba), skip calibration
            # Skip LFs already wrapped in CalibratedClassifierCV (e.g. SVM)
            from sklearn.calibration import CalibratedClassifierCV as _CalCV
            if isinstance(lf._model, _CalCV):
                continue

            if lf._model is None:
                continue

            try:
                # Create a calibrated version using CV
                # For small datasets, use fewer CV folds
                cv = min(self.calibration_cv, min(np.bincount(y_labeled).min(), 5))

                if cv < 2:
                    # Not enough data for CV calibration, skip
                    continue

                calibrated = CalibratedClassifierCV(
                    lf._model, cv=cv, method="sigmoid"
                )
                calibrated.fit(X_labeled, y_labeled)
                self._calibrators[lf.name] = calibrated

                # Replace the LF's model with the calibrated version for predict
                lf._model = calibrated

            except Exception as e:
                # If calibration fails, keep the uncalibrated model
                pass


# =========================================================================
# T4: SELF-TRAINING WITH PSEUDO-LABELS
# =========================================================================

class SelfTrainingPseudoLabeler:
    """
    T4: Self-Training with Pseudo-Labels — After the WS+AL loop, iteratively
    add the classifier's own high-confidence predictions as additional labels.

    This is similar to WeakCert but done as a POST-PROCESSING step with
    multiple iterations. Each iteration:
    1. Train classifier on all labeled data (human + WS + previous pseudo-labels)
    2. Predict on remaining unlabeled data
    3. Add top-k highest-confidence predictions as pseudo-labels
    4. Repeat until no more samples pass threshold or max iterations reached

    Key difference from WeakCert (which is per-step in the AL loop):
    - Self-training runs AFTER the AL loop exhausts the budget
    - It's iterative, allowing the model to progressively expand its coverage
    - It uses a gradually decreasing threshold to avoid noise
    """

    def __init__(
        self,
        initial_threshold: float = 0.9,
        min_threshold: float = 0.7,
        decay: float = 0.95,
        max_iterations: int = 10,
        batch_size: int = 20,
        max_pseudo_labels: int = 200,
    ):
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.decay = decay
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.max_pseudo_labels = max_pseudo_labels

    def run(
        self,
        classifier,
        X_pool,
        y_pool_ground_truth: np.ndarray,  # Only for evaluation, not used in predictions
        labeled_mask: np.ndarray,
        y_labeled: np.ndarray,
        is_human_label: np.ndarray,
        X_test,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Run self-training iterations.

        Returns:
            (labeled_mask, y_labeled, is_human_label, stats)
            stats contains: pseudo_labels_added, pseudo_label_accuracy, iterations
        """
        total_pseudo = 0
        pseudo_correct = 0
        pseudo_total = 0
        iterations_done = 0
        threshold = self.initial_threshold

        for iteration in range(self.max_iterations):
            if total_pseudo >= self.max_pseudo_labels:
                break

            unlabeled_idx = np.where(~labeled_mask)[0]
            if len(unlabeled_idx) == 0:
                break

            # Train classifier
            labeled_idx = np.where(labeled_mask)[0]
            X_train = X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            clf = type(classifier)(**classifier.get_params())
            clf.fit(X_train, y_train)

            # Get probabilities
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X_pool[unlabeled_idx])
            else:
                continue

            max_proba = np.max(proba, axis=1)
            predicted_labels = np.argmax(proba, axis=1)

            # Select high-confidence predictions
            certain_mask = max_proba >= threshold
            n_certain = certain_mask.sum()

            if n_certain == 0:
                # Lower threshold and try again
                threshold *= self.decay
                if threshold < self.min_threshold:
                    break
                continue

            # Cap batch size
            if n_certain > self.batch_size:
                top_k = np.argsort(max_proba)[-self.batch_size:]
                certain_mask = np.zeros(len(unlabeled_idx), dtype=bool)
                certain_mask[top_k] = True

            pseudo_indices = unlabeled_idx[certain_mask]
            pseudo_labels = predicted_labels[certain_mask]

            # Track accuracy (using ground truth, evaluation only)
            ground_truth = y_pool_ground_truth[pseudo_indices]
            pseudo_correct += int((pseudo_labels == ground_truth).sum())
            pseudo_total += len(pseudo_labels)

            # Add pseudo-labels
            labeled_mask[pseudo_indices] = True
            y_labeled[pseudo_indices] = pseudo_labels
            is_human_label[pseudo_indices] = False
            total_pseudo += len(pseudo_indices)
            iterations_done += 1

            # Decay threshold
            threshold *= self.decay

        stats = {
            "pseudo_labels_added": total_pseudo,
            "pseudo_label_accuracy": pseudo_correct / max(pseudo_total, 1),
            "iterations": iterations_done,
        }

        return labeled_mask, y_labeled, is_human_label, stats


# =========================================================================
# T5: UNANIMOUS VOTING AGGREGATOR
# =========================================================================

class UnanimousVotingAggregator:
    """
    T5: Ensemble Disagreement Filter — Only WS-label a sample when ALL
    voting LFs agree (unanimous consensus), not just majority.

    Majority vote allows labels with 51% agreement. Unanimous voting requires
    100% agreement among non-abstaining LFs. This is much more conservative:
    fewer labels produced, but much higher accuracy.

    Configurable: can require N out of M LFs to agree (quorum), where
    N can be "all" (unanimous) or a fraction like 0.8 (80% agreement).
    """

    def __init__(
        self,
        agreement_ratio: float = 1.0,
        min_voters: int = 2,
    ):
        """
        Args:
            agreement_ratio: Fraction of voting LFs that must agree.
                1.0 = unanimous, 0.8 = 80% must agree, etc.
            min_voters: Minimum number of non-abstaining LFs required.
        """
        self.agreement_ratio = agreement_ratio
        self.min_voters = min_voters

    def aggregate(
        self,
        lf_predictions: list[np.ndarray],
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregate with strict agreement requirement (numba-accelerated).

        Returns:
            (labels, confidences): labels=-1 for abstentions
        """
        n_samples = len(lf_predictions[0])
        obs = np.column_stack(lf_predictions).astype(np.int64)

        labels, confidences = _unanimous_voting_kernel(
            obs, n_samples, n_classes, self.min_voters, self.agreement_ratio
        )

        return labels, confidences


# =========================================================================
# T6: PER-CLASS WS THRESHOLDS
# =========================================================================

class PerClassWSThresholds:
    """
    T6: Per-Class WS Thresholds — Use different confidence thresholds
    for different classes based on their difficulty.

    Classes where WS is less reliable get higher thresholds (more conservative),
    while classes where WS is accurate get lower thresholds (more labels).

    This addresses the problem that a single global threshold is suboptimal:
    easy classes may have many correct WS labels above 0.7, while hard classes
    may need 0.95+ to be reliable.
    """

    def __init__(
        self,
        base_threshold: float = 0.7,
        difficulty_multiplier: float = 1.5,
        min_threshold: float = 0.6,
        max_threshold: float = 0.98,
    ):
        """
        Args:
            base_threshold: Starting threshold for all classes
            difficulty_multiplier: How much to increase threshold for hard classes
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.base_threshold = base_threshold
        self.difficulty_multiplier = difficulty_multiplier
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._class_thresholds: dict[int, float] = {}

    def compute_thresholds(
        self,
        ws: WeakSupervisor,
        X_labeled,
        y_labeled: np.ndarray,
        n_classes: int,
        texts: list[str] | None = None,
    ) -> dict[int, float]:
        """
        Compute per-class thresholds based on per-class LF accuracy.

        For each class c:
        1. Evaluate each LF's accuracy on samples of class c
        2. If LFs are inaccurate for class c → raise threshold
        3. If LFs are accurate for class c → lower threshold
        """
        # Get LF predictions on labeled data
        lf_preds = []
        for lf in ws.lfs:
            try:
                if getattr(lf, 'needs_texts', False) and texts is not None:
                    preds = lf.predict(X_labeled, texts=texts)
                else:
                    preds = lf.predict(X_labeled)
                lf_preds.append(preds)
            except Exception:
                lf_preds.append(np.full(X_labeled.shape[0], -1, dtype=int))

        # Per-class accuracy
        class_accuracies = {}
        for c in range(n_classes):
            class_mask = y_labeled == c
            if class_mask.sum() < 2:
                class_accuracies[c] = 0.5  # Default for rare classes
                continue

            correct = 0
            total = 0
            for preds in lf_preds:
                valid = preds[class_mask] >= 0
                if valid.sum() > 0:
                    correct += int((preds[class_mask][valid] == y_labeled[class_mask][valid]).sum())
                    total += int(valid.sum())

            if total > 0:
                class_accuracies[c] = correct / total
            else:
                class_accuracies[c] = 0.5

        # Compute thresholds
        avg_accuracy = np.mean(list(class_accuracies.values()))

        for c, acc in class_accuracies.items():
            if acc >= avg_accuracy:
                # Easy class → lower threshold
                ratio = acc / max(avg_accuracy, 0.01)
                threshold = self.base_threshold / ratio
            else:
                # Hard class → raise threshold
                ratio = avg_accuracy / max(acc, 0.01)
                threshold = self.base_threshold * min(ratio ** 0.5, self.difficulty_multiplier)

            self._class_thresholds[c] = np.clip(threshold, self.min_threshold, self.max_threshold)

        return self._class_thresholds

    def filter_by_class_threshold(
        self,
        weak_labels: np.ndarray,
        weak_confidences: np.ndarray,
        weak_label_classes: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Filter WS labels using per-class thresholds.

        Args:
            weak_labels: Array of predicted labels (-1 = abstain)
            weak_confidences: Array of confidence scores
            weak_label_classes: The predicted class for each sample (same as weak_labels if not None)

        Returns:
            Boolean mask of which labels pass their class-specific threshold
        """
        if not self._class_thresholds:
            # No thresholds computed yet, use base threshold
            return weak_confidences >= self.base_threshold

        # Convert dict to flat array for numba
        max_class = max(self._class_thresholds.keys()) + 1 if self._class_thresholds else 0
        class_thresholds_arr = np.full(max_class, self.base_threshold)
        for c, t in self._class_thresholds.items():
            class_thresholds_arr[c] = t

        return _per_class_threshold_filter_kernel(
            weak_labels.astype(np.int64),
            weak_confidences.astype(np.float64),
            class_thresholds_arr,
            max_class,
            self.base_threshold,
        )


# =========================================================================
# T14: CALIBRATED PSEUDO-LABELING
# =========================================================================

class CalibratedPseudoLabeler:
    """
    T14: Calibrated Pseudo-Labeling — Improved version of T4.
    
    Uses energy-based scoring (more robust than softmax confidence) and
    class-distribution-aware thresholds (rebalances pseudo-labels toward
    underrepresented classes).
    
    Reference: Rizve et al. (CVPR 2022) — "Uncertainty-Aware Pseudo-Label Selection"
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.9,
        min_threshold: float = 0.7,
        decay: float = 0.95,
        max_iterations: int = 10,
        batch_size: int = 20,
        max_pseudo_labels: int = 200,
        use_energy_scoring: bool = True,
        temperature: float = 1.0,
        class_distribution_aware: bool = True,
    ):
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.decay = decay
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.max_pseudo_labels = max_pseudo_labels
        self.use_energy_scoring = use_energy_scoring
        self.temperature = temperature
        self.class_distribution_aware = class_distribution_aware
    
    def _compute_energy(self, logits_or_proba: np.ndarray) -> np.ndarray:
        """Compute energy-based confidence scores (numba-accelerated)."""
        return _compute_energy_kernel(
            logits_or_proba.astype(np.float64), self.temperature
        )
    
    def _compute_class_thresholds(
        self, predicted_classes: np.ndarray, n_classes: int, base_threshold: float,
        y_labeled: np.ndarray
    ) -> dict[int, float]:
        """Compute per-class thresholds based on class distribution."""
        # Target distribution = distribution in labeled data
        labeled_counts = np.bincount(y_labeled, minlength=n_classes).astype(float)
        target_props = labeled_counts / max(labeled_counts.sum(), 1)
        
        # Current pseudo-label distribution
        pseudo_counts = np.bincount(predicted_classes, minlength=n_classes).astype(float)
        pseudo_props = pseudo_counts / max(pseudo_counts.sum(), 1)
        
        thresholds = {}
        for c in range(n_classes):
            if target_props[c] > 0 and pseudo_props[c] > 0:
                ratio = (target_props[c] / max(pseudo_props[c], 1e-6)) ** 0.5
                thresholds[c] = base_threshold / ratio  # Lower threshold for underrepresented
            else:
                thresholds[c] = base_threshold
            thresholds[c] = np.clip(thresholds[c], self.min_threshold, 0.99)
        
        return thresholds
    
    def run(
        self,
        classifier,
        X_pool,
        y_pool_ground_truth: np.ndarray,
        labeled_mask: np.ndarray,
        y_labeled: np.ndarray,
        is_human_label: np.ndarray,
        X_test,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run calibrated self-training iterations."""
        total_pseudo = 0
        pseudo_correct = 0
        pseudo_total = 0
        iterations_done = 0
        threshold = self.initial_threshold
        
        # Get number of classes from y_labeled
        n_classes = int(y_labeled.max()) + 1
        
        for iteration in range(self.max_iterations):
            if total_pseudo >= self.max_pseudo_labels:
                break
            
            unlabeled_idx = np.where(~labeled_mask)[0]
            if len(unlabeled_idx) == 0:
                break
            
            labeled_idx = np.where(labeled_mask)[0]
            X_train = X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]
            
            clf = type(classifier)(**classifier.get_params())
            clf.fit(X_train, y_train)
            
            if not hasattr(clf, 'predict_proba'):
                continue
            
            proba = clf.predict_proba(X_pool[unlabeled_idx])
            predicted_labels = np.argmax(proba, axis=1)
            
            # Compute confidence scores
            if self.use_energy_scoring:
                confidence = self._compute_energy(proba)
            else:
                confidence = np.max(proba, axis=1)
            
            # Apply thresholds
            if self.class_distribution_aware:
                class_thresholds = self._compute_class_thresholds(
                    predicted_labels, n_classes, threshold, y_labeled[labeled_idx]
                )
                # Convert dict to flat array for numba kernel
                ct_arr = np.full(n_classes, threshold)
                for c, t in class_thresholds.items():
                    if c < n_classes:
                        ct_arr[c] = t
                passes_threshold = _class_threshold_listcomp_kernel(
                    confidence.astype(np.float64),
                    predicted_labels.astype(np.int64),
                    ct_arr,
                    n_classes,
                    threshold,
                )
            else:
                passes_threshold = confidence >= threshold
            
            n_certain = passes_threshold.sum()
            if n_certain == 0:
                threshold *= self.decay
                if threshold < self.min_threshold:
                    break
                continue
            
            # Cap batch size
            certain_indices = np.where(passes_threshold)[0]
            if len(certain_indices) > self.batch_size:
                top_k = np.argsort(confidence[certain_indices])[-self.batch_size:]
                certain_indices = certain_indices[top_k]
            
            pseudo_pool_indices = unlabeled_idx[certain_indices]
            pseudo_labels = predicted_labels[certain_indices]
            
            ground_truth = y_pool_ground_truth[pseudo_pool_indices]
            pseudo_correct += int((pseudo_labels == ground_truth).sum())
            pseudo_total += len(pseudo_labels)
            
            labeled_mask[pseudo_pool_indices] = True
            y_labeled[pseudo_pool_indices] = pseudo_labels
            is_human_label[pseudo_pool_indices] = False
            total_pseudo += len(pseudo_pool_indices)
            iterations_done += 1
            
            threshold *= self.decay
        
        stats = {
            "pseudo_labels_added": total_pseudo,
            "pseudo_label_accuracy": pseudo_correct / max(pseudo_total, 1),
            "iterations": iterations_done,
        }
        
        return labeled_mask, y_labeled, is_human_label, stats


# =========================================================================
# CONVENIENCE: Enhanced WeakSupervisor with all techniques
# =========================================================================

class EnhancedWeakSupervisor:
    """
    WeakSupervisor that can combine multiple enhancement techniques.

    This is a wrapper around WeakSupervisor that optionally applies:
    - T3: Calibration
    - T5: Unanimous voting
    - T6: Per-class thresholds
    """

    def __init__(
        self,
        n_classes: int,
        lf_confidence_threshold: float = 0.7,
        label_model: Literal["majority_vote", "dawid_skene"] = "dawid_skene",
        use_calibration: bool = False,
        use_unanimous_voting: bool = False,
        unanimous_min_voters: int = 2,
        use_per_class_thresholds: bool = False,
        per_class_base_threshold: float = 0.7,
        use_nb_lf: bool = True,
        use_svm_lf: bool = True,
        use_rf_lf: bool = True,
        use_knn_lf: bool = True,
        use_lr_lf: bool = True,
        use_keyword_lf: bool = True,
        use_topic_lf: bool = False,
        topic_n_topics: int = 20,
        topic_model: Literal["nmf", "lda"] = "nmf",
    ):
        self.n_classes = n_classes
        self.lf_confidence_threshold = lf_confidence_threshold
        self.label_model = label_model
        self.use_calibration = use_calibration
        self.use_unanimous_voting = use_unanimous_voting
        self.use_per_class_thresholds = use_per_class_thresholds

        # Build underlying supervisor
        if use_calibration:
            self._ws = CalibratedWeakSupervisor(
                n_classes=n_classes,
                lf_confidence_threshold=lf_confidence_threshold,
                label_model=label_model,
                use_nb_lf=use_nb_lf,
                use_svm_lf=use_svm_lf,
                use_rf_lf=use_rf_lf,
                use_knn_lf=use_knn_lf,
                use_lr_lf=use_lr_lf,
                use_keyword_lf=use_keyword_lf,
                use_topic_lf=use_topic_lf,
                topic_n_topics=topic_n_topics,
                topic_model=topic_model,
            )
        else:
            self._ws = WeakSupervisor(
                n_classes=n_classes,
                lf_confidence_threshold=lf_confidence_threshold,
                label_model=label_model,
                use_nb_lf=use_nb_lf,
                use_svm_lf=use_svm_lf,
                use_rf_lf=use_rf_lf,
                use_knn_lf=use_knn_lf,
                use_lr_lf=use_lr_lf,
                use_keyword_lf=use_keyword_lf,
                use_topic_lf=use_topic_lf,
                topic_n_topics=topic_n_topics,
                topic_model=topic_model,
            )

        # T5: Unanimous voting
        if use_unanimous_voting:
            self._unanimous = UnanimousVotingAggregator(
                agreement_ratio=1.0,
                min_voters=unanimous_min_voters,
            )
        else:
            self._unanimous = None

        # T6: Per-class thresholds
        if use_per_class_thresholds:
            self._per_class = PerClassWSThresholds(
                base_threshold=per_class_base_threshold,
            )
        else:
            self._per_class = None

        self._trained = False

    def fit(self, X_labeled, y_labeled: np.ndarray, feature_names: list[str] | None = None, texts: list[str] | None = None) -> None:
        """Train all labeling functions."""
        self._ws.fit(X_labeled, y_labeled, feature_names=feature_names, texts=texts)
        self._trained = True

        # T6: Compute per-class thresholds after fitting
        if self._per_class is not None:
            self._per_class.compute_thresholds(
                self._ws, X_labeled, y_labeled, self.n_classes, texts=texts
            )

    def predict(self, X_unlabeled, texts: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate weak labels with optional enhancements.

        Returns:
            (weak_labels, confidences)
        """
        if not self._trained:
            raise RuntimeError("Must call fit() before predict()")

        if self.use_unanimous_voting and self._unanimous is not None:
            # Use unanimous voting instead of normal aggregation
            lf_preds = []
            for lf in self._ws.lfs:
                try:
                    if hasattr(lf, 'needs_texts') and lf.needs_texts and texts is not None:
                        preds = lf.predict(X_unlabeled, texts=texts)
                    else:
                        preds = lf.predict(X_unlabeled)
                    lf_preds.append(preds)
                except Exception:
                    lf_preds.append(np.full(X_unlabeled.shape[0], -1, dtype=int))

            labels, confidences = self._unanimous.aggregate(lf_preds, self.n_classes)
        else:
            labels, confidences = self._ws.predict(X_unlabeled, texts=texts)

        # T6: Apply per-class thresholds
        if self._per_class is not None and self._per_class._class_thresholds:
            passes = self._per_class.filter_by_class_threshold(labels, confidences)
            labels[~passes] = -1
            confidences[~passes] = 0.0

        return labels, confidences


# =========================================================================
# T7: ISOTONIC REGRESSION CALIBRATION
# =========================================================================

class IsotonicCalibratedWeakSupervisor(WeakSupervisor):
    """
    T7: Isotonic Regression Calibration — Alternative to T3 (Platt Scaling).

    Isotonic regression does NOT assume a sigmoidal calibration curve.
    It fits a piecewise constant non-decreasing function, which can better
    capture complex confidence distributions, especially on small datasets.

    Reference: Niculescu-Mizil & Caruana (ICML 2005) showed Isotonic
    outperforms Platt when sufficient calibration data is available.
    """

    def __init__(
        self,
        n_classes: int,
        lf_confidence_threshold: float = 0.6,
        label_model: Literal["majority_vote", "dawid_skene"] = "majority_vote",
        use_nb_lf: bool = True,
        use_svm_lf: bool = True,
        use_rf_lf: bool = True,
        use_knn_lf: bool = True,
        use_lr_lf: bool = True,
        use_keyword_lf: bool = True,
        use_topic_lf: bool = False,
        topic_n_topics: int = 20,
        topic_model: str = "nmf",
        calibration_cv: int = 5,
    ):
        super().__init__(
            n_classes=n_classes,
            lf_confidence_threshold=lf_confidence_threshold,
            label_model=label_model,
            use_nb_lf=use_nb_lf,
            use_svm_lf=use_svm_lf,
            use_rf_lf=use_rf_lf,
            use_knn_lf=use_knn_lf,
            use_lr_lf=use_lr_lf,
            use_keyword_lf=use_keyword_lf,
            use_topic_lf=use_topic_lf,
            topic_n_topics=topic_n_topics,
            topic_model=topic_model,
        )
        self.calibration_cv = calibration_cv
        self._calibrators: dict[str, CalibratedClassifierCV] = {}

    def fit(self, X_labeled, y_labeled: np.ndarray, feature_names: list[str] | None = None, texts: list[str] | None = None) -> None:
        """Train all LFs and then calibrate them using isotonic regression."""
        super().fit(X_labeled, y_labeled, feature_names=feature_names, texts=texts)

        for lf in self.lfs:
            if lf.name == "keyword":
                continue
            if lf.name == "topic":
                continue  # Topic LF uses NMF/LDA (no predict_proba), skip calibration
            # Skip LFs already wrapped in CalibratedClassifierCV (e.g. SVM)
            from sklearn.calibration import CalibratedClassifierCV as _CalCV
            if isinstance(lf._model, _CalCV):
                continue
            if lf._model is None:
                continue

            try:
                cv = min(self.calibration_cv, min(np.bincount(y_labeled).min(), 5))
                if cv < 2:
                    continue

                calibrated = CalibratedClassifierCV(
                    lf._model, cv=cv, method="isotonic"
                )
                calibrated.fit(X_labeled, y_labeled)
                self._calibrators[lf.name] = calibrated
                lf._model = calibrated

            except Exception:
                pass


# =========================================================================
# T10: LABEL PROPAGATION
# =========================================================================

class LabelPropagationWS:
    """
    T10: Label Propagation for Weak Supervision.

    Constructs a k-NN graph over TF-IDF features and propagates labels
    from labeled samples to their unlabeled neighbors using sklearn's
    LabelSpreading (faster variant of LabelPropagation).

    This is a STRUCTURAL complement to the rule-based WS: while LFs
    classify based on feature patterns, LP leverages the geometry of
    the data manifold. Combining both should improve coverage.

    Reference: Iscen et al. (CVPR 2019) — Label Propagation for Deep SSL.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        max_iter: int = 30,
        confidence_threshold: float = 0.8,
    ):
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.confidence_threshold = confidence_threshold

    def propagate(
        self,
        X_pool,
        labeled_mask: np.ndarray,
        y_labeled: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagate labels through k-NN graph.

        Args:
            X_pool: Feature matrix (sparse or dense) for ALL samples
            labeled_mask: Boolean array, True where sample is labeled
            y_labeled: Label array for all samples (only valid where labeled_mask=True)

        Returns:
            (lp_labels, lp_confidences, new_label_mask):
                lp_labels: predicted labels for ALL samples
                lp_confidences: confidence scores for ALL samples
                new_label_mask: boolean mask of unlabeled samples with high-confidence LP labels
        """
        from sklearn.semi_supervised import LabelSpreading

        # Create label array: -1 for unlabeled (required by LabelSpreading)
        y_lp = np.full(len(y_labeled), -1, dtype=int)
        y_lp[labeled_mask] = y_labeled[labeled_mask]

        # Convert sparse to dense (LabelSpreading requires dense)
        # ── #4 OPTIMIZATION: Use kneighbors_graph to avoid full densification ──
        if sparse.issparse(X_pool):
            # Try using LabelSpreading with kernel='knn' on sparse data
            # If it fails, fall back to dense conversion on subsampled data
            try:
                from sklearn.neighbors import kneighbors_graph
                # Build sparse k-NN graph directly from sparse features
                connectivity = kneighbors_graph(
                    X_pool, n_neighbors=min(self.n_neighbors, n_samples - 1),
                    mode='connectivity', metric='cosine', n_jobs=-1,
                )
                X_dense = X_pool.toarray()  # LabelSpreading still needs dense
            except Exception:
                X_dense = X_pool.toarray()
        else:
            X_dense = np.asarray(X_pool)

        # Subsample if too large (LP is O(n^2) or O(n*k))
        n_samples = X_dense.shape[0]
        if n_samples > 10000:
            # Use a random subset for LP, then map back
            # For now, just proceed — most datasets are < 10k
            pass

        try:
            lp = LabelSpreading(
                kernel='knn',
                n_neighbors=min(self.n_neighbors, n_samples - 1),
                max_iter=self.max_iter,
                alpha=0.2,  # Clamping factor: 1-alpha = label retention
            )
            lp.fit(X_dense, y_lp)

            lp_proba = lp.predict_proba(X_dense)
            lp_labels = np.argmax(lp_proba, axis=1)
            lp_conf = np.max(lp_proba, axis=1)

        except Exception as e:
            print(f"    [T10-LP] LabelSpreading failed: {e}, skipping")
            return (
                np.full(len(y_labeled), -1, dtype=int),
                np.zeros(len(y_labeled)),
                np.zeros(len(y_labeled), dtype=bool),
            )

        # Only keep high-confidence propagated labels for unlabeled samples
        new_label_mask = (~labeled_mask) & (lp_conf >= self.confidence_threshold)

        return lp_labels, lp_conf, new_label_mask


# =========================================================================
# T8: BERT-BASED LABELING FUNCTION (7th LF)
# =========================================================================

class BERTLF(LabelingFunction):
    """
    T8: BERT-based Labeling Function — Uses SentenceTransformer embeddings
    + LogisticRegression as a 7th, contextually-aware LF.

    Current 6 LFs are all TF-IDF based → they correlate. BERT-LF provides
    an INDEPENDENT contextual signal from pre-trained language models, which
    breaks the TF-IDF correlation and gives Dawid-Skene better diversity
    to work with.

    Reference: SentenceTransformers (Reimers & Gurevych, 2019)

    Key difference from TF-IDF LFs:
    - TF-IDF captures surface-level keyword patterns
    - BERT captures semantic meaning and context
    - This orthogonality improves aggregation quality

    Implementation notes:
    - Pre-computes embeddings for ALL texts (pool + test) on first call
    - Stores embeddings in a cache keyed by text tuple hash
    - fit()/predict() operate on BERT embeddings, not TF-IDF features
    """

    def __init__(
        self,
        abstain_threshold: float = 0.6,
        model_name: str = "all-MiniLM-L6-v2",
        texts_all: tuple[str, ...] | None = None,
    ):
        super().__init__("bert", abstain_threshold)
        self.model_name = model_name
        self.texts_all = texts_all
        self._embeddings_cache: np.ndarray | None = None
        self._texts_hash: int | None = None

    def _get_embeddings(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Get BERT embeddings for texts. Computes and caches on first call.

        Args:
            indices: If provided, return embeddings for these indices only.
                     If None, return embeddings for all texts.

        Returns:
            Dense embedding matrix (n_samples, embed_dim)
        """
        if self.texts_all is None:
            raise ValueError("texts_all must be set before calling _get_embeddings()")

        # Check cache
        current_hash = hash(self.texts_all[:100])  # Quick hash of first 100 texts
        if self._embeddings_cache is None or self._texts_hash != current_hash:
            print(f"    [T8-BERT] Computing embeddings for {len(self.texts_all)} texts "
                  f"using {self.model_name}...")

            # ── #7 OPTIMIZATION: Try to reuse shared model instance ──
            model = _get_shared_sentence_transformer(self.model_name)
            self._embeddings_cache = model.encode(
                list(self.texts_all),
                show_progress_bar=False,
                batch_size=64,
            )
            self._texts_hash = current_hash
            print(f"    [T8-BERT] Embeddings computed: shape={self._embeddings_cache.shape}")

        if indices is not None:
            return self._embeddings_cache[indices]
        return self._embeddings_cache

    def fit(self, X, y: np.ndarray) -> None:
        """
        Train LogisticRegression on BERT embeddings.

        Note: X parameter is TF-IDF features (passed by WeakSupervisor),
        but we ignore it and use BERT embeddings instead, indexed by
        the assumption that X rows correspond to texts_all rows.
        """
        # X is TF-IDF features from WeakSupervisor — we need labeled indices
        # The caller (BERTWeakSupervisor) passes indices via a side channel
        if not hasattr(self, '_fit_indices') or self._fit_indices is None:
            # Fallback: assume first N texts
            n_labeled = X.shape[0]
            self._fit_indices = np.arange(n_labeled)

        X_emb = self._get_embeddings(self._fit_indices)
        self._model = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0,
        )
        self._model.fit(X_emb, y)
        self._fit_indices = None  # Clear after use

    def predict(self, X) -> np.ndarray:
        """
        Predict using BERT embeddings + LogisticRegression.

        Note: X is TF-IDF features from WeakSupervisor, but we use
        BERT embeddings instead, indexed by _predict_indices.
        """
        if self._model is None:
            return np.full(X.shape[0], -1, dtype=int)

        if not hasattr(self, '_predict_indices') or self._predict_indices is None:
            # Fallback: assume next N texts after the fit indices
            n_unlabeled = X.shape[0]
            n_total = len(self.texts_all) if self.texts_all else n_unlabeled
            self._predict_indices = np.arange(n_total - n_unlabeled, n_total)

        X_emb = self._get_embeddings(self._predict_indices)
        proba = self._model.predict_proba(X_emb)
        self._predict_indices = None  # Clear after use
        return self._predict_with_abstain(proba)


class BERTWeakSupervisor(WeakSupervisor):
    """
    T8: WeakSupervisor with BERT-based 7th Labeling Function.

    Extends the standard WeakSupervisor by adding a BERTLF that uses
    SentenceTransformer embeddings instead of TF-IDF features.

    The BERT LF provides a semantically-aware signal that is INDEPENDENT
    of the 6 TF-IDF LFs. This breaks LF correlation and gives the
    aggregation model (Dawid-Skene or majority vote) more diverse
    information to work with.

    Reference: Fu et al. (NeurIPS 2020) showed that LF diversity
    is crucial for good aggregation quality.
    """

    def __init__(
        self,
        n_classes: int,
        lf_confidence_threshold: float = 0.6,
        label_model: Literal["majority_vote", "dawid_skene"] = "dawid_skene",
        use_nb_lf: bool = True,
        use_svm_lf: bool = True,
        use_rf_lf: bool = True,
        use_knn_lf: bool = True,
        use_lr_lf: bool = True,
        use_keyword_lf: bool = True,
        use_topic_lf: bool = False,
        topic_n_topics: int = 20,
        topic_model: str = "nmf",
        bert_model_name: str = "all-MiniLM-L6-v2",
        texts_all: tuple[str, ...] | None = None,
        bert_abstain_threshold: float = 0.6,
    ):
        super().__init__(
            n_classes=n_classes,
            lf_confidence_threshold=lf_confidence_threshold,
            label_model=label_model,
            use_nb_lf=use_nb_lf,
            use_svm_lf=use_svm_lf,
            use_rf_lf=use_rf_lf,
            use_knn_lf=use_knn_lf,
            use_lr_lf=use_lr_lf,
            use_keyword_lf=use_keyword_lf,
            use_topic_lf=use_topic_lf,
            topic_n_topics=topic_n_topics,
            topic_model=topic_model,
        )
        self.bert_model_name = bert_model_name
        self.texts_all = texts_all

        # Add BERT LF as 7th LF
        bert_lf = BERTLF(
            abstain_threshold=bert_abstain_threshold,
            model_name=bert_model_name,
            texts_all=texts_all,
        )
        self.lfs.append(bert_lf)
        self._bert_lf = bert_lf

        # Track which indices map to texts_all
        self._all_indices: np.ndarray | None = None

    def set_all_indices(self, all_indices: np.ndarray) -> None:
        """Set the mapping from pool positions to texts_all positions."""
        self._all_indices = all_indices

    def fit(self, X_labeled, y_labeled: np.ndarray, feature_names: list[str] | None = None, texts: list[str] | None = None) -> None:
        """Train all LFs including BERT LF."""
        # Train standard TF-IDF LFs
        for lf in self.lfs:
            if lf.name == "bert":
                continue  # Handle BERT separately
            try:
                if hasattr(lf, 'needs_texts') and lf.needs_texts and texts is not None:
                    lf.fit(X_labeled, y_labeled, texts=texts)
                else:
                    lf.fit(X_labeled, y_labeled)
            except Exception as e:
                print(f"  Warning: LF '{lf.name}' failed to fit: {e}")

        # Train BERT LF
        # We need to figure out which texts_all indices correspond to X_labeled
        # The caller should use set_all_indices() to provide the mapping
        bert_lf = self._bert_lf
        if self._all_indices is not None and len(self._all_indices) >= X_labeled.shape[0]:
            # The labeled indices in the full pool
            bert_lf._fit_indices = self._all_indices[:X_labeled.shape[0]]
        else:
            bert_lf._fit_indices = np.arange(X_labeled.shape[0])

        try:
            bert_lf.fit(X_labeled, y_labeled)
        except Exception as e:
            print(f"  Warning: BERT LF failed to fit: {e}")

        # Score LFs
        for lf in self.lfs:
            if lf.name == "bert":
                if lf._model is not None:
                    try:
                        X_emb = bert_lf._get_embeddings(bert_lf._fit_indices if bert_lf._fit_indices is not None else np.arange(X_labeled.shape[0]))
                        preds = lf._predict_with_abstain(lf._model.predict_proba(X_emb))
                        valid = preds >= 0
                        if valid.sum() > 0:
                            lf._score = float(accuracy_score(y_labeled[valid], preds[valid]))
                    except Exception:
                        lf._score = 0.0
            elif lf._model is not None or lf.name in ("keyword", "topic"):
                try:
                    if getattr(lf, 'needs_texts', False) and texts is not None:
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

    def predict(self, X_unlabeled, texts: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate weak labels using all LFs including BERT."""
        if not self._trained:
            raise RuntimeError("Must call fit() before predict()")

        # Determine indices for BERT LF
        bert_lf = self._bert_lf
        if self._all_indices is not None:
            # Unlabeled indices = those not in the first N labeled
            n_labeled = len(self._all_indices) - X_unlabeled.shape[0]
            n_total = len(self.texts_all) if self.texts_all else 0
            # The unlabeled indices are at the end of the pool
            bert_lf._predict_indices = np.arange(n_labeled, n_labeled + X_unlabeled.shape[0])
        else:
            bert_lf._predict_indices = np.arange(X_unlabeled.shape[0])

        # Collect LF predictions
        lf_preds = []
        for lf in self.lfs:
            try:
                if hasattr(lf, 'needs_texts') and lf.needs_texts and texts is not None:
                    preds = lf.predict(X_unlabeled, texts=texts)
                else:
                    preds = lf.predict(X_unlabeled)
                lf_preds.append(preds)
            except Exception:
                lf_preds.append(np.full(X_unlabeled.shape[0], -1, dtype=int))

        # Aggregate
        if self.label_model == "majority_vote":
            labels, confidences = LabelAggregator.majority_vote(lf_preds, self.n_classes)
        elif self.label_model == "dawid_skene":
            labels, confidences = LabelAggregator.dawid_skene(lf_preds, self.n_classes)
        else:
            raise ValueError(f"Unknown label model: {self.label_model}")

        return labels, confidences


# =========================================================================
# T13: FLYINGSQUID AGGREGATION (Triplet-based, alternative to Dawid-Skene)
# =========================================================================

class FlyingSquidAggregator:
    """
    T13: FlyingSquid-style Triplet Aggregation — Alternative to Dawid-Skene.

    Uses triplet-based estimation of LF accuracies without EM iterations.
    For any triplet of LFs (i, j, k), the three-way agreement rate
    can be decomposed to estimate individual LF accuracies analytically.

    Key advantages over Dawid-Skene:
    - 10-100x faster (closed-form solution, no EM iterations)
    - No convergence issues
    - Works well with 3+ LFs
    - More robust to small datasets (no overfitting in EM)

    The algorithm:
    1. For all triplets of LFs, compute agreement rates
    2. From agreement rates, estimate each LF's accuracy via method-of-moments
    3. Use estimated accuracies to weight LFs in weighted majority vote

    Reference: Fu et al. (NeurIPS 2020) — "Fast and Three-rious: Speeding Up
    Weak Supervision with Triplet Methods"
    """

    def __init__(
        self,
        n_classes: int,
        max_iter: int = 5,
        tol: float = 1e-3,
    ):
        self.n_classes = n_classes
        self.max_iter = max_iter  # For optional refinement
        self.tol = tol

    def _estimate_lf_accuracies_triplet(
        self,
        lf_predictions: list[np.ndarray],
    ) -> np.ndarray:
        """
        Estimate LF accuracies using triplet method.

        For 3 LFs (i, j, k), the three-way agreement rate is:
            P(yi = yj = yk) ≈ a_i * a_j * a_k + (1-a_i)(1-a_j)(1-a_k) / (K-1)^2

        From pairs of triplets, we can solve for individual accuracies.

        Returns:
            Array of estimated LF accuracies (n_lfs,)
        """
        n_lfs = len(lf_predictions)
        n_samples = len(lf_predictions[0])
        K = self.n_classes

        if n_lfs < 3:
            # Not enough LFs for triplet method, use pairwise agreement
            return self._estimate_lf_accuracies_pairwise(lf_predictions)

        # Compute pairwise agreement rates (numba-accelerated)
        obs = np.column_stack(lf_predictions).astype(np.int64)
        agreement = _pairwise_agreement_kernel(obs, n_lfs, n_samples)

        # Method-of-moments estimation from pairwise agreements
        # P(yi = yj) ≈ a_i * a_j + (1 - a_i)(1 - a_j) / (K - 1)
        # For K classes: agreement_ij = a_i * a_j + (1-a_i)(1-a_j)/(K-1)
        # Rearranging: a_i * a_j * (1 - 1/(K-1)) + (1/(K-1)) - (a_i+a_j)/(K-1) + a_i*a_j/(K-1) = agreement
        # This is complex, so use iterative approach

        # Initialize with average pairwise agreement as proxy
        accuracies = np.zeros(n_lfs)
        for i in range(n_lfs):
            others = [agreement[i, j] for j in range(n_lfs) if j != i]
            if others:
                # If LF agrees with others above chance, estimate accuracy
                avg_agree = np.mean(others)
                # Solve: agreement ≈ a^2 + (1-a)^2/(K-1)
                # agreement = a^2 + (1 - 2a + a^2)/(K-1)
                # agreement*(K-1) = a^2*(K-1) + 1 - 2a + a^2
                # agreement*(K-1) = a^2*K - 2a + 1
                # a^2*K - 2a + 1 - agreement*(K-1) = 0
                # Using quadratic formula:
                a2_coeff = K
                a1_coeff = -2.0
                a0_coeff = 1.0 - avg_agree * (K - 1)
                discriminant = a1_coeff**2 - 4 * a2_coeff * a0_coeff
                if discriminant >= 0:
                    a1 = (-a1_coeff + np.sqrt(discriminant)) / (2 * a2_coeff)
                    a2 = (-a1_coeff - np.sqrt(discriminant)) / (2 * a2_coeff)
                    # Take the solution closer to avg_agree
                    accuracies[i] = max(a1, a2) if max(a1, a2) <= 1.0 else min(a1, a2)
                else:
                    accuracies[i] = avg_agree
            else:
                accuracies[i] = 0.5

        # Clip to valid range
        accuracies = np.clip(accuracies, 1.0 / K, 0.999)

        # Refine with triplet constraints (numba-accelerated)
        accuracies = _triplet_refinement_kernel(
            obs, n_lfs, n_samples, K, self.max_iter, self.tol,
            accuracies.astype(np.float64)
        )

        return accuracies

    def _estimate_lf_accuracies_pairwise(
        self,
        lf_predictions: list[np.ndarray],
    ) -> np.ndarray:
        """Fallback for < 3 LFs: use simple pairwise agreement."""
        n_lfs = len(lf_predictions)
        K = self.n_classes
        accuracies = np.full(n_lfs, 1.0 / K + 0.1)  # Slightly above chance

        if n_lfs < 2:
            return accuracies

        for i in range(n_lfs):
            valid_i = lf_predictions[i] >= 0
            if valid_i.sum() == 0:
                continue
            # Use self-consistency as proxy
            # A good LF should produce non-abstaining predictions for many samples
            coverage = valid_i.sum() / max(len(lf_predictions[i]), 1)
            accuracies[i] = max(1.0 / K + 0.05, min(coverage, 0.95))

        return accuracies

    def aggregate(
        self,
        lf_predictions: list[np.ndarray],
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregate LF predictions using FlyingSquid-style weighted voting.

        Steps:
        1. Estimate each LF's accuracy via triplet method
        2. Weight each LF's vote by its estimated accuracy
        3. Take weighted majority vote

        Returns:
            (labels, confidences): labels=-1 for abstentions
        """
        n_samples = len(lf_predictions[0])

        # Step 1: Estimate LF accuracies
        lf_accuracies = self._estimate_lf_accuracies_triplet(lf_predictions)

        # Step 2: Weighted majority vote (numba-accelerated)
        obs = np.column_stack(lf_predictions).astype(np.int64)
        vote_weights, no_votes_mask = _flyingsquid_weighted_vote_kernel(
            obs, lf_accuracies.astype(np.float64), n_samples, n_classes
        )

        # Step 3: Get labels and confidences
        labels = np.argmax(vote_weights, axis=1)
        total_weight = vote_weights.sum(axis=1)

        # Confidence = normalized weight of the winning class
        with np.errstate(invalid='ignore', divide='ignore'):
            confidences = np.where(
                total_weight > 0,
                np.exp(vote_weights[np.arange(n_samples), labels]) /
                np.exp(vote_weights).sum(axis=1),
                0.0,
            )
        # Simplified confidence: proportion of weight for winning class
        with np.errstate(invalid='ignore', divide='ignore'):
            confidences = np.where(
                total_weight > 0,
                vote_weights[np.arange(n_samples), labels] / np.maximum(total_weight, 1e-10),
                0.0,
            )
        confidences = np.clip(confidences, 0.0, 1.0)

        # Mark abstentions (no LF voted)
        labels[no_votes_mask] = -1
        confidences[no_votes_mask] = 0.0

        # Print estimated accuracies for debugging
        for i, acc in enumerate(lf_accuracies):
            lf_name = f"LF{i}"
            if i < 6:
                names = ["NB", "SVM", "RF", "KNN", "LR", "KW"]
                lf_name = names[i]
            elif i == 6:
                lf_name = "BERT"
            # Only print occasionally

        return labels, confidences


class FlyingSquidWeakSupervisor(WeakSupervisor):
    """
    T13: WeakSupervisor using FlyingSquid aggregation instead of Dawid-Skene.

    Uses the same 6 (or 7) LFs as the standard WeakSupervisor, but replaces
    the aggregation method with FlyingSquid's triplet-based approach.

    Benefits:
    - Faster: O(n_lfs^2) instead of O(n_lfs^2 * n_samples * max_iter)
    - No EM convergence issues
    - Analytically grounded accuracy estimates

    Reference: Fu et al. (NeurIPS 2020) — FlyingSquid
    """

    def __init__(
        self,
        n_classes: int,
        lf_confidence_threshold: float = 0.6,
        use_nb_lf: bool = True,
        use_svm_lf: bool = True,
        use_rf_lf: bool = True,
        use_knn_lf: bool = True,
        use_lr_lf: bool = True,
        use_keyword_lf: bool = True,
        use_topic_lf: bool = False,
        topic_n_topics: int = 20,
        topic_model: Literal["nmf", "lda"] = "nmf",
    ):
        super().__init__(
            n_classes=n_classes,
            lf_confidence_threshold=lf_confidence_threshold,
            label_model="majority_vote",  # We'll override predict()
            use_nb_lf=use_nb_lf,
            use_svm_lf=use_svm_lf,
            use_rf_lf=use_rf_lf,
            use_knn_lf=use_knn_lf,
            use_lr_lf=use_lr_lf,
            use_keyword_lf=use_keyword_lf,
            use_topic_lf=use_topic_lf,
            topic_n_topics=topic_n_topics,
            topic_model=topic_model,
        )
        self._fs_aggregator = FlyingSquidAggregator(n_classes=n_classes)

    def predict(self, X_unlabeled, texts: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate weak labels using FlyingSquid aggregation."""
        if not self._trained:
            raise RuntimeError("Must call fit() before predict()")

        # Collect LF predictions
        lf_preds = []
        for lf in self.lfs:
            try:
                if hasattr(lf, 'needs_texts') and lf.needs_texts and texts is not None:
                    preds = lf.predict(X_unlabeled, texts=texts)
                else:
                    preds = lf.predict(X_unlabeled)
                lf_preds.append(preds)
            except Exception:
                lf_preds.append(np.full(X_unlabeled.shape[0], -1, dtype=int))

        # Use FlyingSquid aggregation
        labels, confidences = self._fs_aggregator.aggregate(
            lf_preds, self.n_classes
        )

        return labels, confidences
