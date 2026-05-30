"""
Hybrid Pipeline module — AL + AutoWS + WeakCert.

Core insight: WS should AUGMENT the labeled set, not REPLACE human labels.
The hybrid strategy:

1. Train classifier on L (human + WS labels)
2. WS auto-labels high-confidence samples → adds them to training set
3. AL queries only target samples where WS is UNCERTAIN
   (avoiding wasted human effort on "easy" samples WS already handles)
4. Human labels go into L with ground truth (always correct)
5. WS labels go into L with "weak" status (some noise)

This way, with the same human budget, the hybrid classifier trains on
MUCH MORE data (human + WS) than AL-only (human only), while AL
focuses human effort on the hardest, most informative samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..config import PipelineConfig
from ..data import Dataset, get_stratified_seed_indices
from ..active_learning import (
    ActiveLearner,
    QueryStrategy,
    create_classifier,
    _predict_proba,
)
from ..weak_supervision import WeakSupervisor, WeakCertainty


@dataclass
class PipelineResult:
    """Complete results from a pipeline run."""
    name: str
    config: PipelineConfig

    history: list[dict] = field(default_factory=list)

    final_accuracy: float = 0.0
    final_f1_macro: float = 0.0
    total_human_labels: int = 0
    total_ws_labels: int = 0
    total_labels: int = 0
    ws_label_accuracy: float = 0.0
    ws_contribution_pct: float = 0.0
    human_savings_pct: float = 0.0

    baseline_accuracy: float = 0.0
    baseline_f1_macro: float = 0.0

    n_pool: int = 0
    n_test: int = 0
    n_classes: int = 0
    class_names: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"Dataset: {self.config.dataset_name} | "
            f"Pool: {self.n_pool} | Test: {self.n_test} | Classes: {self.n_classes}",
            f"Final Accuracy:  {self.final_accuracy:.4f}",
            f"Final F1 Macro:  {self.final_f1_macro:.4f}",
            f"Baseline Acc:    {self.baseline_accuracy:.4f} | F1: {self.baseline_f1_macro:.4f}",
            f"Accuracy Drop:   {self.baseline_accuracy - self.final_accuracy:.4f} "
            f"({(self.baseline_accuracy - self.final_accuracy)/max(self.baseline_accuracy, 0.001)*100:.1f}%)",
            f"Human Labels:    {self.total_human_labels}",
            f"WS Labels:       {self.total_ws_labels}  "
            f"(accuracy: {self.ws_label_accuracy:.4f})",
            f"WS Contribution: {self.ws_contribution_pct:.1f}%",
            f"Total Trained On: {self.total_labels} (human + WS)",
        ]
        return "\n".join(lines)


class HybridPipeline:
    """
    Hybrid AL + WS pipeline.

    Strategy: WS augments training data. AL focuses human queries
    on samples where WS is uncertain. This gives the classifier
    more training data than AL-only, with the same human budget.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        result = PipelineResult(
            name="hybrid",
            config=cfg,
            n_pool=len(dataset.y_pool),
            n_test=len(dataset.y_test),
            n_classes=dataset.n_classes,
            class_names=dataset.class_names,
        )

        # ── Baseline ────────────────────────────────────────
        baseline_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        baseline_clf.fit(dataset.X_pool, dataset.y_pool)
        y_pred_base = baseline_clf.predict(dataset.X_test)
        result.baseline_accuracy = float(accuracy_score(dataset.y_test, y_pred_base))
        result.baseline_f1_macro = float(f1_score(dataset.y_test, y_pred_base, average="macro"))

        # ── Initialize ──────────────────────────────────────
        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )

        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()

        # Track which labels came from humans (clean) vs WS (noisy)
        is_human_label = np.zeros(len(dataset.y_pool), dtype=bool)
        is_human_label[seed_indices] = True

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        # ── WS components ───────────────────────────────────
        weak_cert = WeakCertainty(alpha=cfg.weak_cert_alpha)
        query_strategy = QueryStrategy(cfg.query_strategy)
        rng = np.random.default_rng(cfg.random_seed)

        # ── Main loop ───────────────────────────────────────
        step = 0
        while human_labels_used < cfg.max_human_labels:
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            # 1. Train classifier on ALL labeled data (human + WS)
            labeled_idx = np.where(labeled_mask)[0]
            X_train = dataset.X_pool[labeled_idx]
            y_train = y_labeled[labeled_idx]

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            classifier.fit(X_train, y_train)

            # 2. Evaluate
            y_pred = classifier.predict(dataset.X_test)
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            train_accuracy = float(accuracy_score(y_train, classifier.predict(X_train)))

            # 3. WS: auto-label high-confidence samples
            ws_step_labels = 0
            min_human_for_ws = cfg.get_ws_min_human(dataset.n_classes)

            if train_accuracy >= cfg.ws_accuracy_threshold and human_labels_used >= min_human_for_ws:
                # 3a. WeakCert — classifier's own high-confidence predictions
                remaining_unlabeled = np.where(~labeled_mask)[0]
                if len(remaining_unlabeled) > 0:
                    cert_indices, cert_labels = weak_cert.predict(
                        classifier, dataset.X_pool, remaining_unlabeled
                    )
                    if len(cert_indices) > 0:
                        max_ws = cfg.get_ws_batch_limit()
                        if len(cert_indices) > max_ws:
                            cert_proba = np.max(
                                _predict_proba(classifier, dataset.X_pool[cert_indices]),
                                axis=1,
                            )
                            top_k = np.argsort(cert_proba)[-max_ws:]
                            cert_indices = cert_indices[top_k]
                            cert_labels = cert_labels[top_k]

                        cert_ground_truth = dataset.y_pool[cert_indices]
                        ws_label_correct += int((cert_labels == cert_ground_truth).sum())
                        ws_label_total += len(cert_labels)

                        labeled_mask[cert_indices] = True
                        y_labeled[cert_indices] = cert_labels
                        is_human_label[cert_indices] = False  # WS label
                        ws_labels_used += len(cert_indices)
                        ws_step_labels += len(cert_indices)

                # 3b. AutoWS LFs — only if WeakCert didn't cover enough
                if ws_step_labels < cfg.get_ws_batch_limit():
                    remaining_unlabeled = np.where(~labeled_mask)[0]
                    if len(remaining_unlabeled) > 0:
                        ws = WeakSupervisor(
                            n_classes=dataset.n_classes,
                            lf_confidence_threshold=cfg.lf_confidence_threshold,
                            label_model=cfg.label_model,
                            use_nb_lf=cfg.use_nb_lf,
                            use_svm_lf=cfg.use_svm_lf,
                            use_rf_lf=cfg.use_rf_lf,
                            use_knn_lf=cfg.use_knn_lf,
                            use_lr_lf=cfg.use_lr_lf,
                            use_keyword_lf=cfg.use_keyword_lf,
                        )
                        current_labeled_idx = np.where(labeled_mask)[0]
                        ws.fit(dataset.X_pool[current_labeled_idx], y_labeled[current_labeled_idx])
                        weak_labels, weak_confidences = ws.predict(
                            dataset.X_pool[remaining_unlabeled]
                        )

                        confident_mask = (weak_labels >= 0) & (weak_confidences >= cfg.ws_confidence_filter)
                        if confident_mask.sum() > 0:
                            ws_indices = remaining_unlabeled[confident_mask]
                            ws_preds = weak_labels[confident_mask]
                            ws_confs = weak_confidences[confident_mask]

                            remaining_quota = max(cfg.batch_size - ws_step_labels, 0)
                            if len(ws_indices) > remaining_quota:
                                top_conf = np.argsort(ws_confs)[-remaining_quota:]
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

            # 4. AL queries human — ALWAYS, even if WS labeled some
            remaining_unlabeled = np.where(~labeled_mask)[0]
            if len(remaining_unlabeled) > 0 and human_labels_used < cfg.max_human_labels:
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

            # 5. Record
            step_metrics = {
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
            }
            result.history.append(step_metrics)
            step += 1

            if step % 5 == 0 or step == 1:
                ws_acc = ws_label_correct / max(ws_label_total, 1)
                print(
                    f"  Step {step:3d} | Human: {human_labels_used:4d} | "
                    f"WS: {ws_labels_used:4d} | Total labeled: {labeled_mask.sum():4d} | "
                    f"Acc: {test_accuracy:.4f} | F1: {test_f1:.4f} | WS-acc: {ws_acc:.4f}"
                )

        # ── Final ───────────────────────────────────────────
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
        result.ws_contribution_pct = (
            ws_labels_used / max(labeled_mask.sum(), 1) * 100
        )
        result.human_savings_pct = (
            (1 - human_labels_used / cfg.max_human_labels) * 100
        )

        return result


class ALOnlyPipeline:
    """Pure Active Learning — no weak supervision."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        result = PipelineResult(
            name="al_only",
            config=cfg,
            n_pool=len(dataset.y_pool),
            n_test=len(dataset.y_test),
            n_classes=dataset.n_classes,
            class_names=dataset.class_names,
        )

        baseline_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        baseline_clf.fit(dataset.X_pool, dataset.y_pool)
        y_pred_base = baseline_clf.predict(dataset.X_test)
        result.baseline_accuracy = float(accuracy_score(dataset.y_test, y_pred_base))
        result.baseline_f1_macro = float(f1_score(dataset.y_test, y_pred_base, average="macro"))

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        learner = ActiveLearner(
            X_pool=dataset.X_pool,
            y_pool=dataset.y_pool,
            texts_pool=dataset.texts_pool,
            query_strategy=QueryStrategy(cfg.query_strategy),
            classifier_type=cfg.classifier_type,
            batch_size=cfg.batch_size,
            random_seed=cfg.random_seed,
        )
        learner.seed_labels(seed_indices)
        history = learner.run(
            budget=cfg.max_human_labels,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
        )
        result.history = history

        if history:
            last = history[-1]
            result.final_accuracy = last.get("accuracy", 0.0)
            result.final_f1_macro = last.get("f1_macro", 0.0)
            result.total_human_labels = learner.human_labels_used
            result.total_ws_labels = 0
            result.total_labels = learner.n_labeled
            result.ws_contribution_pct = 0.0
            result.human_savings_pct = 0.0

        return result


class WSOnlyPipeline:
    """Pure Weak Supervision — no active learning loop."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        result = PipelineResult(
            name="ws_only",
            config=cfg,
            n_pool=len(dataset.y_pool),
            n_test=len(dataset.y_test),
            n_classes=dataset.n_classes,
            class_names=dataset.class_names,
        )

        baseline_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        baseline_clf.fit(dataset.X_pool, dataset.y_pool)
        y_pred_base = baseline_clf.predict(dataset.X_test)
        result.baseline_accuracy = float(accuracy_score(dataset.y_test, y_pred_base))
        result.baseline_f1_macro = float(f1_score(dataset.y_test, y_pred_base, average="macro"))

        seed_indices = get_stratified_seed_indices(
            dataset.y_pool, cfg.initial_per_class, cfg.random_seed
        )
        labeled_mask = np.zeros(len(dataset.y_pool), dtype=bool)
        labeled_mask[seed_indices] = True
        y_labeled = dataset.y_pool.copy()

        human_labels_used = len(seed_indices)
        ws_labels_used = 0
        ws_label_correct = 0
        ws_label_total = 0

        for iteration in range(50):
            labeled_idx = np.where(labeled_mask)[0]
            unlabeled_idx = np.where(~labeled_mask)[0]
            if len(unlabeled_idx) == 0:
                break

            ws = WeakSupervisor(
                n_classes=dataset.n_classes,
                lf_confidence_threshold=cfg.lf_confidence_threshold,
                label_model=cfg.label_model,
                use_nb_lf=cfg.use_nb_lf,
                use_svm_lf=cfg.use_svm_lf,
                use_rf_lf=cfg.use_rf_lf,
                use_knn_lf=cfg.use_knn_lf,
                use_lr_lf=cfg.use_lr_lf,
                use_keyword_lf=cfg.use_keyword_lf,
            )
            ws.fit(dataset.X_pool[labeled_idx], y_labeled[labeled_idx])
            weak_labels, weak_confidences = ws.predict(dataset.X_pool[unlabeled_idx])

            confident_mask = (weak_labels >= 0) & (weak_confidences >= 0.75)
            if confident_mask.sum() == 0:
                break

            ws_indices = unlabeled_idx[confident_mask]
            ws_preds = weak_labels[confident_mask]
            ws_ground_truth = dataset.y_pool[ws_indices]
            ws_label_correct += int((ws_preds == ws_ground_truth).sum())
            ws_label_total += len(ws_preds)

            labeled_mask[ws_indices] = True
            y_labeled[ws_indices] = ws_preds
            ws_labels_used += len(ws_indices)

            classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
            new_labeled_idx = np.where(labeled_mask)[0]
            classifier.fit(dataset.X_pool[new_labeled_idx], y_labeled[new_labeled_idx])
            y_pred = classifier.predict(dataset.X_test)

            result.history.append({
                "step": iteration,
                "n_labeled": int(labeled_mask.sum()),
                "human_labels_used": human_labels_used,
                "ws_labels_used": ws_labels_used,
                "accuracy": float(accuracy_score(dataset.y_test, y_pred)),
                "f1_macro": float(f1_score(dataset.y_test, y_pred, average="macro")),
            })

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

        return result


class RandomLabelsPipeline:
    """Random sampling — same # of human labels, no strategy."""

    def __init__(self, config: PipelineConfig, n_human_labels: int):
        self.config = config
        self.n_human_labels = n_human_labels

    def run(self, dataset: Dataset) -> PipelineResult:
        cfg = self.config
        result = PipelineResult(
            name="random_labels",
            config=cfg,
            n_pool=len(dataset.y_pool),
            n_test=len(dataset.y_test),
            n_classes=dataset.n_classes,
            class_names=dataset.class_names,
        )

        baseline_clf = create_classifier(cfg.classifier_type, cfg.random_seed)
        baseline_clf.fit(dataset.X_pool, dataset.y_pool)
        y_pred_base = baseline_clf.predict(dataset.X_test)
        result.baseline_accuracy = float(accuracy_score(dataset.y_test, y_pred_base))
        result.baseline_f1_macro = float(f1_score(dataset.y_test, y_pred_base, average="macro"))

        rng = np.random.default_rng(cfg.random_seed)
        all_indices = np.arange(len(dataset.y_pool))
        n = min(self.n_human_labels, len(all_indices))
        selected = rng.choice(all_indices, size=n, replace=False)

        classifier = create_classifier(cfg.classifier_type, cfg.random_seed)
        classifier.fit(dataset.X_pool[selected], dataset.y_pool[selected])
        y_pred = classifier.predict(dataset.X_test)

        result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred))
        result.final_f1_macro = float(f1_score(dataset.y_test, y_pred, average="macro"))
        result.total_human_labels = self.n_human_labels
        result.total_ws_labels = 0
        result.total_labels = self.n_human_labels
        result.ws_contribution_pct = 0.0

        result.history.append({
            "step": 0,
            "n_labeled": self.n_human_labels,
            "human_labels_used": self.n_human_labels,
            "ws_labels_used": 0,
            "accuracy": result.final_accuracy,
            "f1_macro": result.final_f1_macro,
        })

        return result


def _select_queries(
    strategy: QueryStrategy,
    classifier,
    X_pool,
    unlabeled_indices: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    from ..active_learning import select_queries
    return select_queries(strategy, classifier, X_pool, unlabeled_indices, n, rng)
