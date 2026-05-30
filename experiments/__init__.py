"""
Experiment runner — compare all pipeline variants.

5 comparison modes:
1. baseline      — full supervision (all labels)
2. random_labels — random subset, same # human labels as hybrid
3. al_only       — AL without WS
4. ws_only       — WS without AL
5. hybrid        — AL + AutoWS + WeakCert (our method)
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from ..config import PipelineConfig, ExperimentConfig
from ..data import Dataset, load_dataset
from ..pipeline import (
    HybridPipeline,
    ALOnlyPipeline,
    WSOnlyPipeline,
    RandomLabelsPipeline,
    PipelineResult,
)
from ..active_learning import create_classifier
from sklearn.metrics import accuracy_score, f1_score


def run_experiment(config: ExperimentConfig) -> PipelineResult:
    """Run a single experiment configuration."""
    cfg = config.pipeline_config
    dataset = load_dataset(cfg)

    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"Mode: {config.pipeline_mode}")
    print(f"Dataset: {cfg.dataset_name} | Pool: {len(dataset.y_pool)} | "
          f"Test: {len(dataset.y_test)} | Classes: {dataset.n_classes}")
    print(f"{'='*60}")

    start_time = time.time()

    if config.pipeline_mode == "hybrid":
        pipeline = HybridPipeline(cfg)
        result = pipeline.run(dataset)
    elif config.pipeline_mode == "al_only":
        pipeline = ALOnlyPipeline(cfg)
        result = pipeline.run(dataset)
    elif config.pipeline_mode == "ws_only":
        pipeline = WSOnlyPipeline(cfg)
        result = pipeline.run(dataset)
    elif config.pipeline_mode == "random_labels":
        pipeline = RandomLabelsPipeline(cfg, n_human_labels=cfg.max_human_labels)
        result = pipeline.run(dataset)
    elif config.pipeline_mode == "baseline":
        result = _run_baseline(dataset, cfg)
    else:
        raise ValueError(f"Unknown pipeline mode: {config.pipeline_mode}")

    result.name = config.name
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(result.summary())

    return result


def _run_baseline(dataset: Dataset, cfg: PipelineConfig) -> PipelineResult:
    """Full supervision baseline — all labels available."""
    result = PipelineResult(
        name="baseline",
        config=cfg,
        n_pool=len(dataset.y_pool),
        n_test=len(dataset.y_test),
        n_classes=dataset.n_classes,
        class_names=dataset.class_names,
    )

    clf = create_classifier(cfg.classifier_type, cfg.random_seed)
    clf.fit(dataset.X_pool, dataset.y_pool)
    y_pred = clf.predict(dataset.X_test)

    result.final_accuracy = float(accuracy_score(dataset.y_test, y_pred))
    result.final_f1_macro = float(f1_score(dataset.y_test, y_pred, average="macro"))
    result.baseline_accuracy = result.final_accuracy
    result.baseline_f1_macro = result.final_f1_macro
    result.total_human_labels = len(dataset.y_pool)
    result.total_ws_labels = 0
    result.total_labels = len(dataset.y_pool)
    result.ws_contribution_pct = 0.0

    result.history.append({
        "step": 0,
        "n_labeled": len(dataset.y_pool),
        "human_labels_used": len(dataset.y_pool),
        "ws_labels_used": 0,
        "accuracy": result.final_accuracy,
        "f1_macro": result.final_f1_macro,
    })

    return result


def run_comparison(
    base_config: PipelineConfig | None = None,
    n_repeats: int = 3,
) -> dict[str, list[PipelineResult]]:
    """
    Run full comparison of all pipeline variants.

    Returns dict mapping mode name → list of results (one per repeat).
    """
    if base_config is None:
        base_config = PipelineConfig()

    all_results: dict[str, list[PipelineResult]] = {}

    modes = ["baseline", "al_only", "ws_only", "hybrid"]

    for repeat in range(n_repeats):
        print(f"\n\n{'#'*70}")
        print(f"# REPEAT {repeat + 1}/{n_repeats}")
        print(f"{'#'*70}")

        # Vary seed per repeat for robustness
        cfg = PipelineConfig(
            random_seed=42 + repeat * 100,
            dataset_name=base_config.dataset_name,
            max_samples=base_config.max_samples,
            max_human_labels=base_config.max_human_labels,
            batch_size=base_config.batch_size,
            query_strategy=base_config.query_strategy,
            classifier_type=base_config.classifier_type,
            ws_accuracy_threshold=base_config.ws_accuracy_threshold,
            weak_cert_alpha=base_config.weak_cert_alpha,
            lf_confidence_threshold=base_config.lf_confidence_threshold,
            label_model=base_config.label_model,
        )

        for mode in modes:
            exp = ExperimentConfig(
                name=f"{mode}_r{repeat}",
                pipeline_mode=mode,
                pipeline_config=cfg,
                n_repeats=1,
            )
            result = run_experiment(exp)

            if mode not in all_results:
                all_results[mode] = []
            all_results[mode].append(result)

    # Also run random_labels with same human count as hybrid
    hybrid_results = all_results.get("hybrid", [])
    if hybrid_results:
        avg_human = int(np.mean([r.total_human_labels for r in hybrid_results]))
        random_results = []
        for repeat in range(n_repeats):
            cfg = PipelineConfig(
                random_seed=42 + repeat * 100,
                dataset_name=base_config.dataset_name,
                max_samples=base_config.max_samples,
                max_human_labels=avg_human,
                classifier_type=base_config.classifier_type,
            )
            exp = ExperimentConfig(
                name=f"random_labels_r{repeat}",
                pipeline_mode="random_labels",
                pipeline_config=cfg,
            )
            result = run_experiment(exp)
            random_results.append(result)
        all_results["random_labels"] = random_results

    return all_results
