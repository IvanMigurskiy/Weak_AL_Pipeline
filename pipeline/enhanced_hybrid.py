"""
Enhanced Hybrid Pipeline variants — one per pipeline modification technique.

Each variant is a COPY of the original HybridPipeline with ONE technique added.
Original pipeline/__init__.py is NOT modified (backed up in _backups/).

Pipeline Modification Techniques (14 + Combo):
════════════════════════════════════════════════

WS Label Quality Improvements:
  T1  WeightedPipeline           — Weighted training (WS labels weighted lower)
  T2  VerificationPipeline       — WS label verification (remove classifier-disagreed labels)
  T3  CalibratedPipeline         — Calibrated LFs (Platt scaling)
  T5  UnanimousPipeline          — Unanimous voting (all LFs must agree)
  T6  PerClassThresholdPipeline  — Per-class WS thresholds (class-specific confidence gates)

Post-Loop Extensions:
  T4  SelfTrainingPipeline       — Self-training pseudo-labels (post-loop iterative)
  T14 CalibratedPseudoPipeline   — Calibrated pseudo-labeling (energy-based + class-distribution-aware)

Calibration Alternatives:
  T7  IsotonicPipeline           — Isotonic regression calibration (non-parametric alternative to T3)

LF Diversity & Aggregation:
  T8  BERTLFPipeline             — BERT-based 7th LF (SentenceTransformer, breaks TF-IDF correlation)
  T13 FlyingSquidPipeline        — FlyingSquid triplet aggregation (analytical, no EM)

AL Strategy Enhancements:
  T9  BADGEPipeline              — BADGE batch AL (diverse gradient embeddings, k-means++ init)
  T11 CostSensitivePipeline      — Cost-sensitive AL (inverse class frequency weights)

Pipeline Architecture:
  T10 LabelPropagationPipeline   — Label propagation through k-NN graph (structural WS complement)
  T12 AdaptiveBudgetPipeline     — Adaptive AL budget (shrinks batch when WS is active)

Combined:
  ComboPipeline                  — T1+T2+T3+T6 (all conservative techniques)

References:
  T3  — Platt (1999) "Probabilistic Outputs for Support Vector Machines"
  T4  — Scudder (1965); Yarowsky (1995); Karamanolakis et al. (2021) STWS
  T7  — Niculescu-Mizil & Caruana (ICML 2005) "Predicting Good Probabilities"
  T8  — Reimers & Gurevych (2019) SentenceTransformers
  T9  — Ash et al. (ICML 2020) "BADGE"
  T10 — Zhu & Ghahramani (2002) "Learning from Labeled and Unlabeled Data"
  T12 — Mazzetto et al. (2021) WeakAL adaptive budget
  T13 — Fu et al. (NeurIPS 2020) "Fast and Three-rious: Speeding Up Weak Supervision"
  T14 — Rizve et al. (CVPR 2022) "Uncertainty-Aware Pseudo-Label Selection"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import warnings

# Silence Liblinear convergence warnings — handled via scaling + max_iter=10000
warnings.filterwarnings("ignore", message="Liblinear failed to converge")

from ..config import PipelineConfig
from ..data import Dataset, get_stratified_seed_indices
from ..active_learning import (
    ActiveLearner,
    QueryStrategy,
    create_classifier,
    _predict_proba,
)
from ..weak_supervision import WeakSupervisor, WeakCertainty
from ..weak_supervision.enhanced_ws import (
    WeightedTrainingMixin,
    WSLabelVerifier,
    CalibratedWeakSupervisor,
    SelfTrainingPseudoLabeler,
    UnanimousVotingAggregator,
    PerClassWSThresholds,
    EnhancedWeakSupervisor,
    IsotonicCalibratedWeakSupervisor,
    LabelPropagationWS,
    CalibratedPseudoLabeler,
    BERTLF,
    BERTWeakSupervisor,
    FlyingSquidAggregator,
    FlyingSquidWeakSupervisor,
)
from . import HybridPipeline, PipelineResult
from . import _get_effective_use_nb_lf, _get_effective_use_keyword_lf


# =========================================================================
# SHARED HELPERS
# =========================================================================

def _select_queries(strategy, classifier, X_pool, unlabeled_indices, n, rng):
    from ..active_learning import select_queries
    return select_queries(strategy, classifier, X_pool, unlabeled_indices, n, rng)


def _run_baseline(dataset, cfg):
    """Compute baseline accuracy."""
    baseline_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
    baseline_clf.fit(dataset.X_pool, dataset.y_pool)
    y_pred_base = baseline_clf.predict(dataset.X_test)
    return (
        float(accuracy_score(dataset.y_test, y_pred_base)),
        float(f1_score(dataset.y_test, y_pred_base, average="macro")),
    )


def _make_result(name, cfg, dataset, baseline_acc, baseline_f1):
    """Create a PipelineResult with baseline filled in."""
    result = PipelineResult(
        name=name,
        config=cfg,
        n_pool=len(dataset.y_pool),
        n_test=len(dataset.y_test),
        n_classes=dataset.n_classes,
        class_names=dataset.class_names,
    )
    result.baseline_accuracy = baseline_acc
    result.baseline_f1_macro = baseline_f1
    return result


def _do_al_query(cfg, dataset, labeled_mask, y_labeled, is_human_label,
                  human_labels_used, query_strategy, rng):
    """Execute one AL query step. Returns updated human_labels_used."""
    remaining_unlabeled = np.where(~labeled_mask)[0]
    if len(remaining_unlabeled) == 0 or human_labels_used >= cfg.max_human_labels:
        return human_labels_used

    labeled_idx = np.where(labeled_mask)[0]
    classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
    classifier.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])

    n_query = min(
        cfg.batch_size,
        cfg.max_human_labels - human_labels_used,
        len(remaining_unlabeled),
    )
    query_indices = _select_queries(
        query_strategy, classifier, dataset.X_pool,
        remaining_unlabeled, n_query, rng
    )

    if len(query_indices) > 0:
        labeled_mask[query_indices] = True
        y_labeled[query_indices] = dataset.y_pool[query_indices]  # ground truth
        is_human_label[query_indices] = True
        human_labels_used += len(query_indices)

    return human_labels_used


# =========================================================================
# T1: WEIGHTED TRAINING PIPELINE
# =========================================================================

class T1WeightedPipeline:
    """
    Hybrid pipeline with T1: Weighted Training.

    WS labels get lower weight (ws_weight=0.5) during classifier fit.
    Human labels always get weight=1.0. This reduces the impact of
    noisy WS labels on the learned decision boundary.
    """

    def __init__(self, config: PipelineConfig, ws_weight: float = 0.5):
        self.config = config
        self.ws_weight = ws_weight
        self.mixin = WeightedTrainingMixin(ws_weight=ws_weight)

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T1_weighted", cfg, dataset, baseline_acc, baseline_f1)

        # Initialize
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            # Train classifier WITH WEIGHTS
            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]
            sample_weights = self.mixin.compute_sample_weights(
                is_human_label, labeled_mask, self.ws_weight
            )

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            self.mixin.fit_classifier_weighted(classifier, X_train, y_train, sample_weights)

            # Evaluate
            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            # WS auto-label (same logic as original)
            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                # WeakCert
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # AutoWS LFs
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            # AL query
            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            # Record
            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T1] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # Final (weighted)
        labeled_idx = np.where(labeled_mask)[0]
        sample_weights = self.mixin.compute_sample_weights(
            is_human_label, labeled_mask, self.ws_weight
        )
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        self.mixin.fit_classifier_weighted(
            final_clf, dataset.X_pool[labeled_idx], y_labeled[labeled_idx], sample_weights
        )
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T2: WS LABEL VERIFICATION PIPELINE
# =========================================================================

class T2VerificationPipeline:
    """
    Hybrid pipeline with T2: WS Label Verification.

    After each WS labeling step, retrain the classifier and check if it
    agrees with the WS labels. Remove WS labels where the classifier
    disagrees. This is a form of "self-cleaning" the training set.
    """

    def __init__(self, config: PipelineConfig, verify_every: int = 3):
        self.config = config
        self.verify_every = verify_every  # Verify every N steps
        self.verifier = WSLabelVerifier(mode="remove")

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T2_verification", cfg, dataset, baseline_acc, baseline_f1)

        # Initialize
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0
        ws_labels_removed = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            # Train classifier
            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            # Evaluate
            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            # WS auto-label
            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            # T2: Verify WS labels periodically
            if step > 0 and step % self.verify_every == 0 and ws_labels_used > 0:
                # Retrain on HUMAN-ONLY labels to get independent verification
                # (using all labels including WS would be circular — the classifier
                # would agree with its own training data)
                human_idx_v = np.where(is_human_label & labeled_mask)[0]
                if len(human_idx_v) >= dataset.n_classes:
                    clf_v = create_classifier(cfg.classifier_type, cfg.random_seed)
                    clf_v.fit(dataset.X_pool[human_idx_v], y_labeled[human_idx_v])

                    prev_ws = ws_labels_used
                    labeled_mask, y_labeled, is_human_label = self.verifier.verify(
                        clf_v, dataset.X_pool, labeled_mask, y_labeled, is_human_label
                    )
                    ws_labels_removed += prev_ws - int((~is_human_label & labeled_mask).sum())

            # AL query
            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T2] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f} | Removed: {ws_labels_removed}")

        # Final
        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = int((labeled_mask & ~is_human_label).sum())
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = result.total_ws_labels / max(result.total_labels, 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T3: CALIBRATED LFs PIPELINE
# =========================================================================

class T3CalibratedPipeline:
    """
    Hybrid pipeline with T3: Calibrated Labeling Functions.

    Uses Platt scaling to calibrate each LF's confidence scores.
    This makes abstention thresholds more meaningful and should reduce
    false positive WS labels (high confidence but wrong).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T3_calibrated", cfg, dataset, baseline_acc, baseline_f1)

        # Initialize
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            # Train classifier
            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            # WS auto-label
            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                # WeakCert
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # T3: Use CALIBRATED WeakSupervisor
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = CalibratedWeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            calibration_cv=cfg.t3_calibration_cv,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            # AL query
            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T3] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # Final
        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T4: SELF-TRAINING PSEUDO-LABELS PIPELINE
# =========================================================================

class T4SelfTrainingPipeline:
    """
    Hybrid pipeline with T4: Self-Training Pseudo-Labels.

    First runs the normal hybrid AL+WS loop. After the budget is exhausted,
    runs iterative self-training: the classifier pseudo-labels its own
    high-confidence predictions and adds them to the training set.
    This is a POST-PROCESSING step that further expands the training set.
    """

    def __init__(
        self,
        config: PipelineConfig,
        initial_threshold: float = 0.9,
        min_threshold: float = 0.7,
        max_pseudo_labels: int = 200,
    ):
        self.config = config
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_pseudo_labels = max_pseudo_labels

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T4_self_training", cfg, dataset, baseline_acc, baseline_f1)

        # Step 1: Run the normal hybrid pipeline
        base_pipeline = HybridPipeline(cfg)
        base_result = base_pipeline.run(dataset)

        # Copy state from base result
        # Re-extract the final state by re-running the loop
        # (We can't directly access the internal state, so we'll do it differently)

        # Actually, let's just run the full loop here with the self-training at the end
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T4] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # Step 2: Self-training post-processing
        print(f"  [T4] Starting self-training post-processing...")
        final_classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
        labeled_idx = np.where(labeled_mask)[0]
        final_classifier.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])

        self_trainer = SelfTrainingPseudoLabeler(
            initial_threshold=self.initial_threshold,
            min_threshold=self.min_threshold,
            max_pseudo_labels=self.max_pseudo_labels,
            batch_size=20,
            max_iterations=10,
        )

        labeled_mask, y_labeled, is_human_label, st_stats = self_trainer.run(
            final_classifier, dataset.X_pool, dataset.y_pool,
            labeled_mask, y_labeled, is_human_label,
            dataset.X_test, dataset.y_test,
        )

        pseudo_added = st_stats["pseudo_labels_added"]
        pseudo_acc = st_stats["pseudo_label_accuracy"]

        # Add pseudo-labels to WS count
        ws_labels_used += pseudo_added
        ws_label_correct += int(pseudo_acc * pseudo_added)
        ws_label_total += pseudo_added

        print(f"  [T4] Self-training added {pseudo_added} pseudo-labels "
              f"(accuracy: {pseudo_acc:.4f})")

        # Final
        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T5: UNANIMOUS VOTING PIPELINE
# =========================================================================

class T5UnanimousPipeline:
    """
    Hybrid pipeline with T5: Unanimous Voting.

    Only auto-label a sample when ALL non-abstaining LFs agree.
    Much more conservative than majority vote, but higher precision.
    Should produce fewer but more accurate WS labels.
    """

    def __init__(self, config: PipelineConfig, min_voters: int | None = None):
        self.config = config
        self.min_voters = min_voters  # None = read from config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T5_unanimous", cfg, dataset, baseline_acc, baseline_f1)

        # Initialize
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        min_v = self.min_voters if self.min_voters is not None else cfg.t5_min_voters
        unanimous_aggregator = UnanimousVotingAggregator(
            agreement_ratio=cfg.t5_agreement_ratio,
            min_voters=min_v,
        )
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                # WeakCert
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # T5: Unanimous voting for AutoWS LFs
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model="majority_vote",  # Use basic MV first, then filter with unanimous
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)

                        # Collect individual LF predictions
                        lf_preds = []
                        for lf in ws.lfs:
                            try:
                                preds = lf.predict(dataset.X_pool[remaining_unlabeled])
                                lf_preds.append(preds)
                            except Exception:
                                lf_preds.append(np.full(len(remaining_unlabeled), -1, dtype=int))

                        # Use unanimous aggregator instead of normal aggregation
                        weak_labels, weak_confidences = unanimous_aggregator.aggregate(
                            lf_preds, dataset.n_classes
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T5] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # Final
        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T6: PER-CLASS THRESHOLDS PIPELINE
# =========================================================================

class T6PerClassThresholdPipeline:
    """
    Hybrid pipeline with T6: Per-Class WS Thresholds.

    Computes per-class WS accuracy and sets higher confidence thresholds
    for classes where WS is unreliable. Easy classes get lower thresholds
    (more WS labels), hard classes get higher thresholds (fewer but safer).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T6_per_class", cfg, dataset, baseline_acc, baseline_f1)

        # Initialize
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        per_class = PerClassWSThresholds(base_threshold=cfg.lf_confidence_threshold)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                # WeakCert
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # T6: AutoWS with per-class thresholds
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)

                        # Compute per-class thresholds
                        per_class.compute_thresholds(
                            ws, dataset.X_pool[current_labeled_idx],
                            y_labeled[current_labeled_idx], dataset.n_classes,
                            texts=[_texts[i] for i in current_labeled_idx] if _texts else None
                        )

                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        # Apply per-class thresholds instead of global 0.8
                        passes = per_class.filter_by_class_threshold(
                            weak_labels, weak_confidences
                        )
                        confident_mask = (weak_labels >= 0) & passes

                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T6] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # Final
        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# COMBO: ALL CONSERVATIVE TECHNIQUES COMBINED
# =========================================================================

class ComboPipeline:
    """
    Hybrid pipeline combining T1 (Weighted) + T2 (Verification) + T3 (Calibration) + T6 (Per-class).

    This combines all the "conservative" techniques that don't reduce WS coverage:
    - T1: WS labels weighted lower during training
    - T2: Disagreed WS labels are removed
    - T3: LFs are calibrated for better confidence
    - T6: Per-class thresholds adapt to class difficulty

    Should give the highest WS accuracy while maintaining reasonable coverage.
    """

    def __init__(self, config: PipelineConfig, ws_weight: float = 0.5, verify_every: int = 3):
        self.config = config
        self.ws_weight = ws_weight
        self.verify_every = verify_every

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("combo_T1T2T3T6", cfg, dataset, baseline_acc, baseline_f1)

        # Initialize
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        # T1: Weighted training
        wt_mixin = WeightedTrainingMixin(ws_weight=self.ws_weight)
        # T2: Verification
        verifier = WSLabelVerifier(mode="remove")
        # T6: Per-class thresholds
        per_class = PerClassWSThresholds(base_threshold=cfg.t6_base_threshold)

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            # T1: Train with weighted samples
            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]
            sample_weights = wt_mixin.compute_sample_weights(
                is_human_label, labeled_mask, self.ws_weight
            )

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            wt_mixin.fit_classifier_weighted(classifier, X_train, y_train, sample_weights)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                # WeakCert
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # T3+T6: Calibrated LFs with per-class thresholds
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = CalibratedWeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            calibration_cv=cfg.t3_calibration_cv,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)

                        # T6: Compute per-class thresholds
                        per_class.compute_thresholds(
                            ws, dataset.X_pool[current_labeled_idx],
                            y_labeled[current_labeled_idx], dataset.n_classes,
                            texts=[_texts[i] for i in current_labeled_idx] if _texts else None
                        )

                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        # T6: Apply per-class thresholds
                        passes = per_class.filter_by_class_threshold(weak_labels, weak_confidences)
                        confident_mask = (weak_labels >= 0) & passes

                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            # T2: Verify WS labels periodically
            if step > 0 and step % self.verify_every == 0 and ws_labels_used > 0:
                # Retrain on HUMAN-ONLY labels to get independent verification
                human_idx_v = np.where(is_human_label & labeled_mask)[0]
                if len(human_idx_v) >= dataset.n_classes:
                    clf_v = create_classifier(cfg.classifier_type, cfg.random_seed)
                    clf_v.fit(dataset.X_pool[human_idx_v], y_labeled[human_idx_v])

                    labeled_mask, y_labeled, is_human_label = verifier.verify(
                        clf_v, dataset.X_pool, labeled_mask, y_labeled, is_human_label
                    )

            # AL query
            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [Combo] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # Final (weighted)
        labeled_idx = np.where(labeled_mask)[0]
        sw = wt_mixin.compute_sample_weights(is_human_label, labeled_mask, self.ws_weight)
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        wt_mixin.fit_classifier_weighted(
            final_clf, dataset.X_pool[labeled_idx], y_labeled[labeled_idx], sw
        )
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = int((labeled_mask & ~is_human_label).sum())
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = result.total_ws_labels / max(result.total_labels, 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T7: ISOTONIC CALIBRATION PIPELINE
# =========================================================================

class T7IsotonicPipeline:
    """
    Hybrid pipeline with T7: Isotonic Regression Calibration.

    Like T3 (Platt Scaling) but uses isotonic regression for calibration.
    Isotonic regression fits a piecewise constant non-decreasing function,
    which can better capture complex confidence distributions.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T7_isotonic", cfg, dataset, baseline_acc, baseline_f1)

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # T7: Use ISOTONIC CALIBRATED WeakSupervisor
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = IsotonicCalibratedWeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            calibration_cv=cfg.t7_calibration_cv,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T7] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T9: BADGE QUERY STRATEGY PIPELINE
# =========================================================================

class T9BADGEPipeline:
    """
    Hybrid pipeline with T9: BADGE (Batch Active learning by Diverse
    Gradient Embeddings).

    Uses BADGE instead of standard uncertainty sampling. BADGE combines
    uncertainty with diversity via k-means++ on gradient proxy embeddings,
    selecting batches that are both uncertain AND representative.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        # Override query strategy to BADGE
        cfg = cfg.with_overrides(query_strategy="badge")
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T9_badge", cfg, dataset, baseline_acc, baseline_f1)

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy.BADGE
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            # AL query with BADGE strategy
            remaining_unlabeled = np.where(~labeled_mask)[0]
            if len(remaining_unlabeled) > 0 and human_labels_used < cfg.max_human_labels:
                n_query = min(cfg.batch_size, cfg.max_human_labels - human_labels_used, len(remaining_unlabeled))
                from ..active_learning import select_queries
                query_indices = select_queries(
                    query_strategy, classifier, dataset.X_pool,
                    remaining_unlabeled, n_query, rng,
                    n_classes=dataset.n_classes,
                )
                if len(query_indices) > 0:
                    labeled_mask[query_indices] = True
                    y_labeled[query_indices] = dataset.y_pool[query_indices]
                    is_human_label[query_indices] = True
                    human_labels_used += len(query_indices)

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T9] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T10: LABEL PROPAGATION PIPELINE
# =========================================================================

class T10LabelPropagationPipeline:
    """
    Hybrid pipeline with T10: Label Propagation.

    In addition to standard WS (WeakCert + AutoWS), also propagates labels
    through a k-NN graph using sklearn's LabelSpreading. This provides a
    STRUCTURAL complement to the rule-based WS approach.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T10_label_propagation", cfg, dataset, baseline_acc, baseline_f1)

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0
        lp_labels_used = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)
        lp = LabelPropagationWS(
            n_neighbors=cfg.t10_n_neighbors,
            max_iter=cfg.t10_max_iter,
            confidence_threshold=cfg.t10_confidence_threshold,
        )

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                # Standard WS: WeakCert
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # Standard WS: AutoWS
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

                # T10: Label Propagation (every 3 steps to reduce overhead)
                if step > 0 and step % 3 == 0:
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        try:
                            lp_labels, lp_conf, new_label_mask = lp.propagate(
                                dataset.X_pool, labeled_mask, y_labeled
                            )
                            lp_new_indices = np.where(new_label_mask)[0]
                            if len(lp_new_indices) > 0:
                                # Cap LP labels per step
                                max_lp = max(cfg.batch_size - ws_step_labels, cfg.batch_size // 2)
                                if len(lp_new_indices) > max_lp:
                                    lp_conf_new = lp_conf[lp_new_indices]
                                    top_lp = np.argsort(lp_conf_new)[-max_lp:]
                                    lp_new_indices = lp_new_indices[top_lp]

                                lp_new_labels = lp_labels[lp_new_indices]
                                lp_ground_truth = dataset.y_pool[lp_new_indices]
                                ws_label_correct += int((lp_new_labels == lp_ground_truth).sum())
                                ws_label_total += len(lp_new_labels)

                                labeled_mask[lp_new_indices] = True
                                y_labeled[lp_new_indices] = lp_new_labels
                                is_human_label[lp_new_indices] = False
                                ws_labels_used += len(lp_new_indices)
                                lp_labels_used += len(lp_new_indices)
                                ws_step_labels += len(lp_new_indices)
                        except Exception:
                            pass  # LP can fail on small/homogeneous data

            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T10] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} (LP: {lp_labels_used}) | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T11: COST-SENSITIVE AL PIPELINE
# =========================================================================

class T11CostSensitivePipeline:
    """
    Hybrid pipeline with T11: Cost-Sensitive Active Learning.

    Uses inverse class frequency weights from the labeled data to bias
    AL queries toward underrepresented classes. This improves F1-macro
    on imbalanced datasets by ensuring rare classes get more queries.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T11_cost_sensitive", cfg, dataset, baseline_acc, baseline_f1)

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy.COST_SENSITIVE
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            # T11: Cost-sensitive AL query with class weights
            remaining_unlabeled = np.where(~labeled_mask)[0]
            if len(remaining_unlabeled) > 0 and human_labels_used < cfg.max_human_labels:
                # Compute inverse class frequency weights from labeled data
                labeled_idx_cw = np.where(labeled_mask)[0]
                y_lab = y_labeled[labeled_idx_cw]
                class_counts = dict(zip(*np.unique(y_lab, return_counts=True)))
                power = cfg.t11_weight_power
                class_weights = {
                    int(c): (1.0 / count) ** power for c, count in class_counts.items()
                }

                n_query = min(cfg.batch_size, cfg.max_human_labels - human_labels_used, len(remaining_unlabeled))
                from ..active_learning import select_queries
                query_indices = select_queries(
                    query_strategy, classifier, dataset.X_pool,
                    remaining_unlabeled, n_query, rng,
                    class_weights=class_weights,
                )
                if len(query_indices) > 0:
                    labeled_mask[query_indices] = True
                    y_labeled[query_indices] = dataset.y_pool[query_indices]
                    is_human_label[query_indices] = True
                    human_labels_used += len(query_indices)

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T11] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T12: ADAPTIVE BUDGET PIPELINE
# =========================================================================

class T12AdaptiveBudgetPipeline:
    """
    Hybrid pipeline with T12: Adaptive Budget.

    Dynamically adjusts the AL batch_size based on WS activity.
    Early iterations: full batch_size. Later: reduced when WS is active,
    so WS labels "save" human labeling budget more aggressively.

    Formula: adaptive_batch = int(base_batch * max(min_batch_fraction,
                                                     1.0 - ws_labels_used / max(max_human_labels, 1)))
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T12_adaptive_budget", cfg, dataset, baseline_acc, baseline_f1)

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        base_batch = cfg.batch_size
        min_batch_fraction = cfg.t12_min_batch_fraction

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = base_batch
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                if ws_step_labels < base_batch:
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(base_batch - ws_step_labels, 0)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            # T12: Adaptive AL query with dynamic batch_size
            adaptive_batch = int(base_batch * max(min_batch_fraction, 1.0 - ws_labels_used / max(cfg.max_human_labels, 1)))
            adaptive_batch = max(adaptive_batch, 1)

            remaining_unlabeled = np.where(~labeled_mask)[0]
            if len(remaining_unlabeled) > 0 and human_labels_used < cfg.max_human_labels:
                n_query = min(adaptive_batch, cfg.max_human_labels - human_labels_used, len(remaining_unlabeled))
                query_indices = _select_queries(
                    query_strategy, classifier, dataset.X_pool,
                    remaining_unlabeled, n_query, rng
                )
                if len(query_indices) > 0:
                    labeled_mask[query_indices] = True
                    y_labeled[query_indices] = dataset.y_pool[query_indices]
                    is_human_label[query_indices] = True
                    human_labels_used += len(query_indices)

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
                "adaptive_batch": adaptive_batch,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T12] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f} | Batch: {adaptive_batch}")

        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T14: CALIBRATED PSEUDO-LABELING PIPELINE
# =========================================================================

class T14CalibratedPseudoPipeline:
    """
    Hybrid pipeline with T14: Calibrated Pseudo-Labeling.

    Like T4 (Self-Training Pseudo-Labels) but uses CalibratedPseudoLabeler
    instead of SelfTrainingPseudoLabeler. Key improvements:
    - Energy-based scoring (more robust than max softmax confidence)
    - Class-distribution-aware thresholds (rebalances pseudo-labels toward
      underrepresented classes)
    """

    def __init__(
        self,
        config: PipelineConfig,
        initial_threshold: float = 0.9,
        min_threshold: float = 0.7,
        max_pseudo_labels: int = 200,
    ):
        self.config = config
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_pseudo_labels = max_pseudo_labels

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T14_calibrated_pseudo", cfg, dataset, baseline_acc, baseline_f1)

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T14] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # T14: Calibrated pseudo-labeling post-processing
        print(f"  [T14] Starting calibrated pseudo-labeling post-processing...")
        final_classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
        labeled_idx = np.where(labeled_mask)[0]
        final_classifier.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])

        self_trainer = CalibratedPseudoLabeler(
            initial_threshold=self.initial_threshold,
            min_threshold=self.min_threshold,
            max_pseudo_labels=self.max_pseudo_labels,
            batch_size=20,
            max_iterations=10,
            use_energy_scoring=cfg.t14_use_energy_scoring,
            temperature=cfg.t14_temperature,
            class_distribution_aware=cfg.t14_class_distribution_aware,
        )

        labeled_mask, y_labeled, is_human_label, st_stats = self_trainer.run(
            final_classifier, dataset.X_pool, dataset.y_pool,
            labeled_mask, y_labeled, is_human_label,
            dataset.X_test, dataset.y_test,
        )

        pseudo_added = st_stats["pseudo_labels_added"]
        pseudo_acc = st_stats["pseudo_label_accuracy"]

        # Add pseudo-labels to WS count
        ws_labels_used += pseudo_added
        ws_label_correct += int(pseudo_acc * pseudo_added)
        ws_label_total += pseudo_added

        print(f"  [T14] Calibrated pseudo-labeling added {pseudo_added} pseudo-labels "
              f"(accuracy: {pseudo_acc:.4f})")

        # Final
        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T8: BERT-BASED LF PIPELINE (7th LF)
# =========================================================================

class T8BERTLFPipeline:
    """
    Hybrid pipeline with T8: BERT-based Labeling Function as 7th LF.

    Adds a SentenceTransformer embedding + LogisticRegression LF to the
    standard 6 TF-IDF LFs. The BERT LF provides an INDEPENDENT contextual
    signal that breaks the correlation among TF-IDF LFs, giving Dawid-Skene
    better diversity to work with.

    Key insight: All 6 current LFs operate on TF-IDF features → they tend
    to make correlated errors. BERT-LF captures semantic meaning that is
    orthogonal to surface keyword patterns, so when TF-IDF LFs agree but
    are wrong, BERT-LF can disagree and help Dawid-Skene recover the
    correct label.

    Reference: Reimers & Gurevych (2019) — SentenceTransformers
    """

    def __init__(self, config: PipelineConfig, bert_model_name: str = "all-MiniLM-L6-v2"):
        self.config = config
        self.bert_model_name = bert_model_name

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T8_bert_lf", cfg, dataset, baseline_acc, baseline_f1)

        # Pre-compute BERT embeddings for ALL pool texts ONCE
        texts_all = dataset.texts_pool
        print(f"  [T8] Pre-computing BERT embeddings for {len(texts_all)} texts...")
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(self.bert_model_name)
        all_embeddings = st_model.encode(
            list(texts_all), show_progress_bar=False, batch_size=64,
        )
        print(f"  [T8] Embeddings computed: shape={all_embeddings.shape}")

        # Initialize
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                # WeakCert
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # T8: Use BERT-enhanced WS (7 LFs including BERT)
                # Instead of creating BERTWeakSupervisor each step (expensive),
                # we train the BERT LF inline and use standard WS + BERT LF predictions
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        # Train BERT LF on labeled data
                        current_labeled_idx = np.where(labeled_mask)[0]
                        X_bert_train = all_embeddings[current_labeled_idx]
                        bert_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                        bert_clf.fit(X_bert_train, y_labeled[current_labeled_idx])

                        # Standard WS (6 TF-IDF LFs)
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        # Get BERT LF predictions
                        X_bert_unlabeled = all_embeddings[remaining_unlabeled]
                        bert_proba = bert_clf.predict_proba(X_bert_unlabeled)
                        bert_max_proba = np.max(bert_proba, axis=1)
                        bert_preds = np.argmax(bert_proba, axis=1)
                        # Apply abstention threshold
                        bert_abstain_threshold = cfg.lf_confidence_threshold
                        bert_lf_preds = bert_preds.copy()
                        bert_lf_preds[bert_max_proba < bert_abstain_threshold] = -1

                        # Combine: if BERT disagrees with WS and BERT is confident,
                        # use BERT prediction (or average confidences)
                        # Simple approach: if BERT is confident, override or supplement WS
                        bert_confident = bert_max_proba >= bert_abstain_threshold

                        # If WS abstained but BERT is confident → use BERT
                        ws_abstained = weak_labels < 0
                        use_bert_only = ws_abstained & bert_confident

                        # If both agree → boost confidence
                        both_active = (weak_labels >= 0) & bert_confident
                        agree_mask = weak_labels == bert_lf_preds

                        # Update: where WS abstained, use BERT
                        weak_labels[use_bert_only] = bert_lf_preds[use_bert_only]
                        weak_confidences[use_bert_only] = bert_max_proba[use_bert_only]

                        # Where both agree, boost confidence
                        if both_active.sum() > 0 and agree_mask.sum() > 0:
                            boost_mask = both_active & agree_mask
                            weak_confidences[boost_mask] = np.minimum(
                                weak_confidences[boost_mask] * 1.2, 1.0
                            )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T8] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # Final
        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# T13: FLYINGSQUID AGGREGATION PIPELINE
# =========================================================================

class T13FlyingSquidPipeline:
    """
    Hybrid pipeline with T13: FlyingSquid Aggregation.

    Replaces Dawid-Skene or majority vote with FlyingSquid's triplet-based
    LF aggregation. Instead of using EM iterations, FlyingSquid estimates
    LF accuracies analytically from pairwise agreement rates and then
    performs weighted majority voting.

    Benefits:
    - 10-100x faster than Dawid-Skene (no EM)
    - No convergence issues
    - Analytically grounded accuracy estimates
    - Better for small datasets where EM may overfit

    Reference: Fu et al. (NeurIPS 2020) — "Fast and Three-rious: Speeding
    Up Weak Supervision with Triplet Methods"
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("T13_flyingsquid", cfg, dataset, baseline_acc, baseline_f1)

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.batch_size
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # T13: Use FlyingSquidWeakSupervisor instead of standard WS
                # AutoWS LFs — always run (alongside WeakCert, like base HybridPipeline)
                # Total WS budget per step = 2× batch_size (WeakCert + AutoWS each get quota)
                if True:  # Always run AutoWS branch (was: ws_step_labels < cfg.batch_size)
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = FlyingSquidWeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            autows_quota = max(cfg.batch_size, 0)  # AutoWS gets its own quota (like base HybridPipeline)
                            if len(ws_indices) > autows_quota:
                                top_conf = np.argsort(ws_confs)[-autows_quota:]
                                ws_indices = ws_indices[top_conf]
                                ws_preds = ws_preds[top_conf]

                            if len(ws_indices) > 0:
                                ws_ground_truth = dataset.y_pool[ws_indices]
                                ws_label_correct += int((ws_preds == ws_ground_truth).sum())
                                ws_label_total += len(ws_preds)

                                labeled_mask[ws_indices] = True
                                y_labeled[ws_indices] = ws_preds
                                is_human_label[ws_indices] = False
                                ws_labels_used += len(ws_indices)
                                ws_step_labels += len(ws_indices)

            human_labels_used = _do_al_query(
                cfg, dataset, labeled_mask, y_labeled, is_human_label,
                human_labels_used, query_strategy, rng
            )

            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [T13] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f}")

        # Final
        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100

        return result


# =========================================================================
# WS-ADAPTIVE-FLOOD: Wait for high WS accuracy, then flood with WS labels
# =========================================================================

class WSAdaptiveFloodPipeline:
    """
    WS-Adaptive Flood Pipeline: combines T3+T5 (high-precision WS) with
    adaptive WS throughput.

    Strategy:
    1. Phase 1 (Conservative): Use T5 Unanimous Voting + T3 Platt Calibration
       to produce high-precision WS labels. Standard batch_size=10.
    2. Phase 2 (Flood): Once running WS accuracy exceeds a threshold
       (e.g., 85%), INCREASE WS throughput (3-5x more WS labels per step)
       and DECREASE human label queries — trust the WS system.

    Key insight: If WS-accuracy is 90%+, WS labels are almost as good as
    human labels. We should add them aggressively once we trust them,
    saving expensive human labeling budget.
    """

    def __init__(
        self,
        config: PipelineConfig,
        ws_trust_threshold: float = 0.85,
        ws_flood_multiplier: int = 5,
        human_reduction_factor: float = 0.3,
        use_calibration: bool = True,
        use_unanimous: bool = True,
    ):
        self.config = config
        self.ws_trust_threshold = ws_trust_threshold
        self.ws_flood_multiplier = ws_flood_multiplier
        self.human_reduction_factor = human_reduction_factor
        self.use_calibration = use_calibration
        self.use_unanimous = use_unanimous

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        baseline_acc, baseline_f1 = _run_baseline(dataset, cfg)
        result = _make_result("WS_adaptive_flood", cfg, dataset, baseline_acc, baseline_f1)

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        unanimous_aggregator = UnanimousVotingAggregator(
            agreement_ratio=cfg.t5_agreement_ratio,
            min_voters=cfg.t5_min_voters,
        )
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        ws_ema_accuracy = 0.0
        ws_ema_alpha = 0.3
        flood_mode = False
        flood_step = -1

        _feature_names = None
        if dataset.encoder is not None:
            _feature_names = dataset.encoder.get_feature_names()
        _texts = None
        if hasattr(dataset, 'texts_pool') and dataset.texts_pool is not None:
            _texts = dataset.texts_pool
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            ws_step_labels = 0
            ws_step_correct = 0
            ws_step_total = 0
            min_human_for_ws = cfg.initial_per_class * dataset.n_classes + cfg.batch_size

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                ws_batch_limit = cfg.batch_size * self.ws_flood_multiplier if flood_mode else cfg.batch_size

                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_wc = ws_batch_limit
                        if len(cert_indices) > max_wc:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]), axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_wc:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        step_correct = int((cert_labels == cert_ground_truth).sum())
                        ws_step_correct += step_correct
                        ws_step_total += len(cert_labels)
                        ws_label_correct += step_correct
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0 and ws_step_labels < ws_batch_limit:
                    if self.use_calibration:
                        ws = CalibratedWeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            calibration_cv=cfg.t3_calibration_cv,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )
                    else:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=_get_effective_use_nb_lf(cfg, dataset),
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=_get_effective_use_keyword_lf(cfg, dataset),
                            use_topic_lf=cfg.use_topic_lf,
                            topic_n_topics=cfg.topic_n_topics,
                            topic_model=cfg.topic_model,
                        )

                    current_labeled_idx = np.where(labeled_mask)[0]
                    ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx], feature_names=_feature_names, texts=[_texts[i] for i in current_labeled_idx] if _texts else None)

                    if self.use_unanimous:
                        lf_preds = []
                        for lf in ws.lfs:
                            try:
                                preds = lf.predict(dataset.X_pool[remaining_unlabeled])
                                lf_preds.append(preds)
                            except Exception:
                                lf_preds.append(np.full(len(remaining_unlabeled), -1, dtype=int))
                        weak_labels, weak_confidences = unanimous_aggregator.aggregate(
                            lf_preds, dataset.n_classes
                        )
                    else:
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled], texts=[_texts[i] for i in remaining_unlabeled] if _texts else None,
                        )

                    conf_filter = cfg.ws_confidence_filter
                    if flood_mode:
                        conf_filter = max(conf_filter - 0.1, 0.6)

                    confident_mask = (weak_labels >= 0) & (weak_confidences >= conf_filter)
                    if confident_mask.sum() > 0:
                        ws_indices = remaining_unlabeled[confident_mask]
                        ws_preds = weak_labels[confident_mask]
                        ws_confs = weak_confidences[confident_mask]

                        autows_quota = max(ws_batch_limit - ws_step_labels, 0)
                        if len(ws_indices) > autows_quota:
                            top_conf = np.argsort(ws_confs)[-autows_quota:]
                            ws_indices = ws_indices[top_conf]
                            ws_preds = ws_preds[top_conf]

                        if len(ws_indices) > 0:
                            ws_ground_truth = dataset.y_pool[ws_indices]
                            step_correct = int((ws_preds == ws_ground_truth).sum())
                            ws_step_correct += step_correct
                            ws_step_total += len(ws_preds)
                            ws_label_correct += step_correct
                            ws_label_total += len(ws_preds)

                            labeled_mask[ws_indices] = True
                            y_labeled[ws_indices] = ws_preds
                            is_human_label[ws_indices] = False
                            ws_labels_used += len(ws_indices)
                            ws_step_labels += len(ws_indices)

                if ws_step_total > 0:
                    step_acc = ws_step_correct / ws_step_total
                    if ws_ema_accuracy == 0:
                        ws_ema_accuracy = step_acc
                    else:
                        ws_ema_accuracy = (1 - ws_ema_alpha) * ws_ema_accuracy + ws_ema_alpha * step_acc

                if not flood_mode and ws_ema_accuracy >= self.ws_trust_threshold and ws_label_total >= 20:
                    flood_mode = True
                    flood_step = step
                    print(f"  [WS-FLOOD] FLOOD MODE ACTIVATED at step {step}! "
                          f"WS-EMA-acc={ws_ema_accuracy:.4f} >= {self.ws_trust_threshold:.2f}")

                if flood_mode and ws_ema_accuracy < (self.ws_trust_threshold - 0.1) and ws_step_total > 10:
                    flood_mode = False
                    print(f"  [WS-FLOOD] FLOOD MODE DEACTIVATED at step {step}. "
                          f"WS-EMA-acc={ws_ema_accuracy:.4f} dropped")

            if flood_mode:
                effective_batch = max(1, int(cfg.batch_size * self.human_reduction_factor))
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0 and human_labels_used < cfg.max_human_labels:
                    n_query = min(effective_batch, cfg.max_human_labels - human_labels_used, len(remaining_unlabeled))
                    query_indices = _select_queries(query_strategy, classifier, dataset.X_pool, remaining_unlabeled, n_query, rng)
                    if len(query_indices) > 0:
                        labeled_mask[query_indices] = True
                        y_labeled[query_indices] = dataset.y_pool[query_indices]
                        is_human_label[query_indices] = True
                        human_labels_used += len(query_indices)
            else:
                human_labels_used = _do_al_query(cfg, dataset, labeled_mask, y_labeled, is_human_label, human_labels_used, query_strategy, rng)

            mode_str = "FLOOD" if flood_mode else "conservative"
            result.history.append({
                "step": step,
                "n_labeled": int(labeled_mask.sum()),
                "n_human_in_labeled": int(is_human_label.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "ws_step_labels": ws_step_labels,
                "n_unlabeled": int((~labeled_mask).sum()),
                "accuracy": test_accuracy,
                "f1_macro": test_f1,
                "train_accuracy": train_accuracy,
                "ws_ema_accuracy": ws_ema_accuracy,
                "flood_mode": flood_mode,
            })
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(f"  [WS-FLOOD] Step {step:3d} | Human: {human_labels_used:4d} | "
                      f"WS: {ws_labels_used:4d} | Total: {labeled_mask.sum():4d} | "
                      f"Acc: {test_accuracy:.4f} | WS-acc: {ws_acc:.4f} | "
                      f"EMA: {ws_ema_accuracy:.4f} | Mode: {mode_str}")

        labeled_idx = np.where(labeled_mask)[0]
        final_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        final_clf.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
        y_pred_final = final_clf.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred_final))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred_final, average="macro"))
        result.total_human_labels = human_labels_used
        result.total_ws_labels = ws_labels_used
        result.total_labels = int(labeled_mask.sum())

        if ws_label_total > 0:
            result.ws_label_accuracy = ws_label_correct / ws_label_total
        result.ws_contribution_pct = ws_labels_used / max(labeled_mask.sum(), 1) * 100
        result.human_savings_pct = (1 - human_labels_used / max(cfg.max_human_labels, 1)) * 100
        result.flood_activated_at_step = flood_step
        result.ws_ema_final = ws_ema_accuracy

        return result
