"""
Step-Wise Pipeline Experiments — Thesis Research Points.

Each pipeline step is a research point with two questions:
  R1 (Presence): Does including this step improve the pipeline?
  R2 (Technique): Which specific technique is most effective?

Steps:
  Step 2: AL Query Strategy (Random / Least Confidence / Margin / Entropy)
  Step 3: WS Labeling Functions (with/without TopicLF, LF ablation)
  Step 4: WS Label Aggregation (Majority Vote vs Dawid-Skene)
  Step 5: Auto-labeling (none / WeakCert / WeakClust / both)
  Step 6: Pipeline Integration (AL-only vs WS-only vs Hybrid, adaptive budget)

Usage on Colab:
    from Weak_AL_Pipeline.experiments.step_wise_experiments import (
        run_step2_al_strategy,
        run_step3_topiclf,
        run_step4_aggregation,
        run_step5_auto_labeling,
        run_step6_integration,
    )
    results = run_step2_al_strategy(datasets=["banking77"], quick=True)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..data import load_dataset, DATASET_INFO
from ..pipeline import HybridPipeline, ALOnlyPipeline, WSOnlyPipeline, PipelineResult


# =========================================================================
# CONFIGURATION
# =========================================================================

KEY_DATASETS = [
    "banking77",
    "customer_tickets",
    "bitext_ecommerce",
]

ALL_AL_STRATEGIES = [
    "random",
    "uncertainty_least_confident",
    "uncertainty_margin",
    "uncertainty_entropy",
]

ALL_AGGREGATION_METHODS = [
    "majority_vote",
    "dawid_skene",
]


# =========================================================================
# BUDGET HELPERS
# =========================================================================

def _budget_for_dataset(dataset_name: str, n_classes: int) -> int:
    if n_classes >= 50:
        return min(500, n_classes * 5)
    elif n_classes >= 20:
        return min(300, n_classes * 10)
    elif n_classes >= 10:
        return 200
    else:
        return 150


def _initial_per_class(n_classes: int) -> int:
    if n_classes >= 50:
        return 1
    elif n_classes >= 20:
        return 2
    else:
        return 3


# =========================================================================
# RUNNER HELPER
# =========================================================================

def _run_pipeline(
    mode: str,
    cfg: PipelineConfig,
    dataset,
) -> PipelineResult | None:
    try:
        if mode == "al_only":
            pipe = ALOnlyPipeline(cfg)
        elif mode == "ws_only":
            pipe = WSOnlyPipeline(cfg)
        elif mode == "hybrid":
            pipe = HybridPipeline(cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        start = time.time()
        result = pipe.run(dataset)
        elapsed = time.time() - start
        print(f"    [{mode}] Acc={result.final_accuracy:.4f}, F1={result.final_f1_macro:.4f}, "
              f"WS={result.total_ws_labels}, time={elapsed:.1f}s")
        return result

    except Exception as e:
        print(f"    [{mode}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# =========================================================================
# STEP 2: AL QUERY STRATEGY
# =========================================================================

def run_step2_al_strategy(
    datasets: list[str] | None = None,
    strategies: list[str] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/step_wise",
) -> dict[str, Any]:
    """
    Step 2: AL Query Strategy Comparison.

    R1 (Presence): Does AL (intelligent selection) outperform random sampling?
    R2 (Technique): Which strategy is best — LC, Margin, or Entropy?
    """
    if datasets is None:
        datasets = KEY_DATASETS
    if strategies is None:
        strategies = ALL_AL_STRATEGIES

    print(f"\n{'#'*80}")
    print(f"# STEP 2: AL QUERY STRATEGY")
    print(f"# R1: AL vs Random | R2: LC vs Margin vs Entropy")
    print(f"# Datasets: {datasets}")
    print(f"# Strategies: {strategies}")
    print(f"{'#'*80}")

    all_results: dict[str, dict[str, list[PipelineResult]]] = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")

        cfg_probe = PipelineConfig(dataset_name=dataset_name, max_samples=500 if quick else None)
        dataset = load_dataset(cfg_probe)
        n_classes = dataset.n_classes
        budget = _budget_for_dataset(dataset_name, n_classes)
        per_class = _initial_per_class(n_classes)

        print(f"  Classes: {n_classes}, Pool: {len(dataset.y_pool)}, Budget: {budget}")

        all_results[dataset_name] = {}

        for strategy in strategies:
            label = strategy.replace("uncertainty_", "")
            all_results[dataset_name][label] = []

            for repeat in range(n_repeats):
                seed = 42 + repeat * 100
                cfg = PipelineConfig(
                    dataset_name=dataset_name,
                    max_samples=500 if quick else None,
                    query_strategy=strategy,
                    max_human_labels=budget,
                    batch_size=10 if n_classes < 50 else 20,
                    initial_per_class=per_class,
                    random_seed=seed,
                )
                dataset = load_dataset(cfg)
                result = _run_pipeline("hybrid", cfg, dataset)
                if result is not None:
                    result.name = f"{label}_r{repeat}"
                    all_results[dataset_name][label].append(result)

    _print_step2_table(all_results)
    _save_results(all_results, output_dir, "step2_al_strategy")
    return all_results


# =========================================================================
# STEP 3: WS LABELING FUNCTIONS (TopicLF)
# =========================================================================

def run_step3_topiclf(
    datasets: list[str] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/step_wise",
) -> dict[str, Any]:
    """
    Step 3: WS Labeling Functions — TopicLF Impact.

    R1 (Presence): Does WS improve over AL-only? (conditional on WS-accuracy)
    R2 (Technique): Does TopicLF provide additional value?
    """
    if datasets is None:
        datasets = KEY_DATASETS

    print(f"\n{'#'*80}")
    print(f"# STEP 3: WS LABELING FUNCTIONS (TopicLF)")
    print(f"# R1: WS helps vs AL-only? | R2: TopicLF adds value?")
    print(f"# Datasets: {datasets}")
    print(f"{'#'*80}")

    all_results: dict[str, dict[str, list[PipelineResult]]] = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")

        cfg_probe = PipelineConfig(dataset_name=dataset_name, max_samples=500 if quick else None)
        dataset = load_dataset(cfg_probe)
        n_classes = dataset.n_classes
        budget = _budget_for_dataset(dataset_name, n_classes)
        per_class = _initial_per_class(n_classes)

        all_results[dataset_name] = {}

        # 3a: AL-only baseline
        all_results[dataset_name]["al_only"] = []
        for repeat in range(n_repeats):
            seed = 42 + repeat * 100
            cfg = PipelineConfig(
                dataset_name=dataset_name,
                max_samples=500 if quick else None,
                max_human_labels=budget,
                batch_size=10 if n_classes < 50 else 20,
                initial_per_class=per_class,
                random_seed=seed,
            )
            dataset = load_dataset(cfg)
            result = _run_pipeline("al_only", cfg, dataset)
            if result is not None:
                result.name = f"al_only_r{repeat}"
                all_results[dataset_name]["al_only"].append(result)

        # 3b: Hybrid without TopicLF
        all_results[dataset_name]["hybrid_no_topic"] = []
        for repeat in range(n_repeats):
            seed = 42 + repeat * 100
            cfg = PipelineConfig(
                dataset_name=dataset_name,
                max_samples=500 if quick else None,
                use_topic_lf=False,
                max_human_labels=budget,
                batch_size=10 if n_classes < 50 else 20,
                initial_per_class=per_class,
                random_seed=seed,
            )
            dataset = load_dataset(cfg)
            result = _run_pipeline("hybrid", cfg, dataset)
            if result is not None:
                result.name = f"hybrid_no_topic_r{repeat}"
                all_results[dataset_name]["hybrid_no_topic"].append(result)

        # 3c: Hybrid WITH TopicLF (NMF)
        all_results[dataset_name]["hybrid_topic_nmf"] = []
        for repeat in range(n_repeats):
            seed = 42 + repeat * 100
            cfg = PipelineConfig(
                dataset_name=dataset_name,
                max_samples=500 if quick else None,
                use_topic_lf=True,
                topic_n_topics=min(n_classes, 20),
                topic_model="nmf",
                max_human_labels=budget,
                batch_size=10 if n_classes < 50 else 20,
                initial_per_class=per_class,
                random_seed=seed,
            )
            dataset = load_dataset(cfg)
            result = _run_pipeline("hybrid", cfg, dataset)
            if result is not None:
                result.name = f"hybrid_topic_nmf_r{repeat}"
                all_results[dataset_name]["hybrid_topic_nmf"].append(result)

        # 3d: Hybrid WITH TopicLF (LDA)
        all_results[dataset_name]["hybrid_topic_lda"] = []
        for repeat in range(n_repeats):
            seed = 42 + repeat * 100
            cfg = PipelineConfig(
                dataset_name=dataset_name,
                max_samples=500 if quick else None,
                use_topic_lf=True,
                topic_n_topics=min(n_classes, 20),
                topic_model="lda",
                max_human_labels=budget,
                batch_size=10 if n_classes < 50 else 20,
                initial_per_class=per_class,
                random_seed=seed,
            )
            dataset = load_dataset(cfg)
            result = _run_pipeline("hybrid", cfg, dataset)
            if result is not None:
                result.name = f"hybrid_topic_lda_r{repeat}"
                all_results[dataset_name]["hybrid_topic_lda"].append(result)

    _print_step3_table(all_results)
    _save_results(all_results, output_dir, "step3_topiclf")
    return all_results


# =========================================================================
# STEP 4: WS LABEL AGGREGATION
# =========================================================================

def run_step4_aggregation(
    datasets: list[str] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/step_wise",
) -> dict[str, Any]:
    """
    Step 4: WS Label Aggregation Comparison.

    R1 (Presence): Does Dawid-Skene outperform simple Majority Vote?
    R2 (Technique): Which method produces better aggregated labels?
    """
    if datasets is None:
        datasets = KEY_DATASETS

    print(f"\n{'#'*80}")
    print(f"# STEP 4: WS LABEL AGGREGATION")
    print(f"# R1: Dawid-Skene > Majority Vote? | R2: Best method per dataset?")
    print(f"# Datasets: {datasets}")
    print(f"{'#'*80}")

    all_results: dict[str, dict[str, list[PipelineResult]]] = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")

        cfg_probe = PipelineConfig(dataset_name=dataset_name, max_samples=500 if quick else None)
        dataset = load_dataset(cfg_probe)
        n_classes = dataset.n_classes
        budget = _budget_for_dataset(dataset_name, n_classes)
        per_class = _initial_per_class(n_classes)

        all_results[dataset_name] = {}

        for agg_method in ALL_AGGREGATION_METHODS:
            label = agg_method
            all_results[dataset_name][label] = []

            for repeat in range(n_repeats):
                seed = 42 + repeat * 100
                cfg = PipelineConfig(
                    dataset_name=dataset_name,
                    max_samples=500 if quick else None,
                    label_model=agg_method,
                    max_human_labels=budget,
                    batch_size=10 if n_classes < 50 else 20,
                    initial_per_class=per_class,
                    random_seed=seed,
                )
                dataset = load_dataset(cfg)
                result = _run_pipeline("hybrid", cfg, dataset)
                if result is not None:
                    result.name = f"{label}_r{repeat}"
                    all_results[dataset_name][label].append(result)

    _print_step4_table(all_results)
    _save_results(all_results, output_dir, "step4_aggregation")
    return all_results


# =========================================================================
# STEP 5: AUTO-LABELING
# =========================================================================

def run_step5_auto_labeling(
    datasets: list[str] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/step_wise",
) -> dict[str, Any]:
    """
    Step 5: Auto-labeling Ablation.

    R1 (Presence): Do auto-labeling components help?
    R2 (Technique): WeakCert vs WeakClust vs both vs none?

    Configurations:
      - no_auto:     hybrid without WeakCert or WeakClust
      - weak_cert:   hybrid with WeakCert only
      - weak_clust:  hybrid with WeakClust only (and no WeakCert)
      - both:        hybrid with both WeakCert and WeakClust
    """
    if datasets is None:
        datasets = KEY_DATASETS

    print(f"\n{'#'*80}")
    print(f"# STEP 5: AUTO-LABELING ABLATION")
    print(f"# R1: Auto-labeling helps? | R2: WeakCert vs WeakClust vs both?")
    print(f"# Datasets: {datasets}")
    print(f"{'#'*80}")

    auto_configs = [
        ("no_auto",    {"use_weak_cert": False, "use_weak_clust": False}),
        ("weak_cert",  {"use_weak_cert": True,  "use_weak_clust": False}),
        ("weak_clust", {"use_weak_cert": False, "use_weak_clust": True}),
        ("both",       {"use_weak_cert": True,  "use_weak_clust": True}),
    ]

    all_results: dict[str, dict[str, list[PipelineResult]]] = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")

        cfg_probe = PipelineConfig(dataset_name=dataset_name, max_samples=500 if quick else None)
        dataset = load_dataset(cfg_probe)
        n_classes = dataset.n_classes
        budget = _budget_for_dataset(dataset_name, n_classes)
        per_class = _initial_per_class(n_classes)

        all_results[dataset_name] = {}

        for label, auto_cfg in auto_configs:
            all_results[dataset_name][label] = []

            for repeat in range(n_repeats):
                seed = 42 + repeat * 100
                cfg = PipelineConfig(
                    dataset_name=dataset_name,
                    max_samples=500 if quick else None,
                    max_human_labels=budget,
                    batch_size=10 if n_classes < 50 else 20,
                    initial_per_class=per_class,
                    random_seed=seed,
                    **auto_cfg,
                )
                dataset = load_dataset(cfg)
                result = _run_pipeline("hybrid", cfg, dataset)
                if result is not None:
                    result.name = f"{label}_r{repeat}"
                    all_results[dataset_name][label].append(result)

    _print_step5_table(all_results)
    _save_results(all_results, output_dir, "step5_auto_labeling")
    return all_results


# =========================================================================
# STEP 6: PIPELINE INTEGRATION
# =========================================================================

def run_step6_integration(
    datasets: list[str] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/step_wise",
) -> dict[str, Any]:
    """
    Step 6: Pipeline Integration Comparison.

    R1 (Presence): Does Hybrid outperform AL-only and WS-only?
    R2 (Technique): What integration strategy works best?

    Runs AL-only, WS-only, and Hybrid on each dataset.
    """
    if datasets is None:
        datasets = KEY_DATASETS

    print(f"\n{'#'*80}")
    print(f"# STEP 6: PIPELINE INTEGRATION")
    print(f"# R1: Hybrid > AL-only + WS-only? | R2: Best integration strategy?")
    print(f"# Datasets: {datasets}")
    print(f"{'#'*80}")

    all_results: dict[str, dict[str, list[PipelineResult]]] = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")

        cfg_probe = PipelineConfig(dataset_name=dataset_name, max_samples=500 if quick else None)
        dataset = load_dataset(cfg_probe)
        n_classes = dataset.n_classes
        budget = _budget_for_dataset(dataset_name, n_classes)
        per_class = _initial_per_class(n_classes)

        all_results[dataset_name] = {}

        for mode in ["al_only", "ws_only", "hybrid"]:
            all_results[dataset_name][mode] = []

            for repeat in range(n_repeats):
                seed = 42 + repeat * 100
                cfg = PipelineConfig(
                    dataset_name=dataset_name,
                    max_samples=500 if quick else None,
                    max_human_labels=budget,
                    batch_size=10 if n_classes < 50 else 20,
                    initial_per_class=per_class,
                    random_seed=seed,
                )
                dataset = load_dataset(cfg)
                result = _run_pipeline(mode, cfg, dataset)
                if result is not None:
                    result.name = f"{mode}_r{repeat}"
                    all_results[dataset_name][mode].append(result)

    _print_step6_table(all_results)
    _save_results(all_results, output_dir, "step6_integration")
    return all_results


# =========================================================================
# PRINTING HELPERS
# =========================================================================

def _mean_field(results: list[PipelineResult], field_name: str) -> float:
    vals = [getattr(r, field_name) for r in results]
    return float(np.mean(vals)) if vals else 0.0


def _print_step2_table(all_results: dict) -> None:
    print(f"\n\n{'='*100}")
    print("  STEP 2: AL QUERY STRATEGY (Hybrid Pipeline)")
    print(f"{'='*100}")

    for ds, strat_results in all_results.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Strategy':25s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | {'Human':>6s} | {'WS':>6s} | {'WS Acc':>7s} | {'Δ vs Random':>12s}")
        print(f"  {'-'*95}")

        random_acc = _mean_field(strat_results.get("random", []), "final_accuracy")

        for strategy in ALL_AL_STRATEGIES:
            label = strategy.replace("uncertainty_", "")
            results = strat_results.get(label, [])
            if not results:
                continue
            acc = _mean_field(results, "final_accuracy")
            f1 = _mean_field(results, "final_f1_macro")
            human = int(_mean_field(results, "total_human_labels"))
            ws = int(_mean_field(results, "total_ws_labels"))
            ws_acc = _mean_field(results, "ws_label_accuracy")
            delta = acc - random_acc

            print(f"  {label:25s} | {acc:>8.4f} | {f1:>8.4f} | {human:>6d} | {ws:>6d} | {ws_acc:>7.4f} | {delta:>+12.4f}")

    print(f"\n  Δ vs Random = accuracy improvement over random sampling baseline")


def _print_step3_table(all_results: dict) -> None:
    print(f"\n\n{'='*110}")
    print("  STEP 3: WS LABELING FUNCTIONS — TopicLF Impact")
    print(f"{'='*110}")

    for ds, config_results in all_results.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Config':20s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | {'Human':>6s} | {'WS':>6s} | {'WS Acc':>7s} | {'WS%':>5s} | {'Δ vs AL':>8s} | {'Δ vs No-Topic':>14s}")
        print(f"  {'-'*110}")

        al_acc = _mean_field(config_results.get("al_only", []), "final_accuracy")
        no_topic_acc = _mean_field(config_results.get("hybrid_no_topic", []), "final_accuracy")

        for label in ["al_only", "hybrid_no_topic", "hybrid_topic_nmf", "hybrid_topic_lda"]:
            results = config_results.get(label, [])
            if not results:
                continue
            acc = _mean_field(results, "final_accuracy")
            f1 = _mean_field(results, "final_f1_macro")
            human = int(_mean_field(results, "total_human_labels"))
            ws = int(_mean_field(results, "total_ws_labels"))
            ws_acc = _mean_field(results, "ws_label_accuracy")
            ws_pct = _mean_field(results, "ws_contribution_pct")
            delta_al = acc - al_acc
            delta_topic = acc - no_topic_acc

            print(f"  {label:20s} | {acc:>8.4f} | {f1:>8.4f} | {human:>6d} | {ws:>6d} | {ws_acc:>7.4f} | {ws_pct:>5.1f}% | {delta_al:>+8.4f} | {delta_topic:>+14.4f}")


def _print_step4_table(all_results: dict) -> None:
    print(f"\n\n{'='*100}")
    print("  STEP 4: WS LABEL AGGREGATION (Hybrid Pipeline)")
    print(f"{'='*100}")

    for ds, agg_results in all_results.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Aggregation':20s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | {'WS Acc':>7s} | {'WS%':>5s} | {'Δ vs MV':>8s}")
        print(f"  {'-'*75}")

        mv_acc = _mean_field(agg_results.get("majority_vote", []), "final_accuracy")

        for method in ALL_AGGREGATION_METHODS:
            results = agg_results.get(method, [])
            if not results:
                continue
            acc = _mean_field(results, "final_accuracy")
            f1 = _mean_field(results, "final_f1_macro")
            ws_acc = _mean_field(results, "ws_label_accuracy")
            ws_pct = _mean_field(results, "ws_contribution_pct")
            delta = acc - mv_acc

            print(f"  {method:20s} | {acc:>8.4f} | {f1:>8.4f} | {ws_acc:>7.4f} | {ws_pct:>5.1f}% | {delta:>+8.4f}")

    print(f"\n  Δ vs MV = accuracy improvement over Majority Vote baseline")


def _print_step5_table(all_results: dict) -> None:
    print(f"\n\n{'='*110}")
    print("  STEP 5: AUTO-LABELING ABLATION (Hybrid Pipeline)")
    print(f"{'='*110}")

    for ds, auto_results in all_results.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Config':15s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | {'Human':>6s} | {'WS':>6s} | {'WS Acc':>7s} | {'WS%':>5s} | {'Δ vs None':>10s}")
        print(f"  {'-'*95}")

        none_acc = _mean_field(auto_results.get("no_auto", []), "final_accuracy")

        for label in ["no_auto", "weak_cert", "weak_clust", "both"]:
            results = auto_results.get(label, [])
            if not results:
                continue
            acc = _mean_field(results, "final_accuracy")
            f1 = _mean_field(results, "final_f1_macro")
            human = int(_mean_field(results, "total_human_labels"))
            ws = int(_mean_field(results, "total_ws_labels"))
            ws_acc = _mean_field(results, "ws_label_accuracy")
            ws_pct = _mean_field(results, "ws_contribution_pct")
            delta = acc - none_acc

            print(f"  {label:15s} | {acc:>8.4f} | {f1:>8.4f} | {human:>6d} | {ws:>6d} | {ws_acc:>7.4f} | {ws_pct:>5.1f}% | {delta:>+10.4f}")


def _print_step6_table(all_results: dict) -> None:
    print(f"\n\n{'='*110}")
    print("  STEP 6: PIPELINE INTEGRATION")
    print(f"{'='*110}")

    for ds, mode_results in all_results.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Mode':15s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | {'Human':>6s} | {'WS':>6s} | {'Total':>6s} | {'WS Acc':>7s} | {'WS%':>5s} | {'Δ vs AL':>8s}")
        print(f"  {'-'*100}")

        al_acc = _mean_field(mode_results.get("al_only", []), "final_accuracy")

        for mode in ["al_only", "ws_only", "hybrid"]:
            results = mode_results.get(mode, [])
            if not results:
                continue
            acc = _mean_field(results, "final_accuracy")
            f1 = _mean_field(results, "final_f1_macro")
            human = int(_mean_field(results, "total_human_labels"))
            ws = int(_mean_field(results, "total_ws_labels"))
            total = int(_mean_field(results, "total_labels"))
            ws_acc = _mean_field(results, "ws_label_accuracy")
            ws_pct = _mean_field(results, "ws_contribution_pct")
            delta = acc - al_acc

            print(f"  {mode:15s} | {acc:>8.4f} | {f1:>8.4f} | {human:>6d} | {ws:>6d} | {total:>6d} | {ws_acc:>7.4f} | {ws_pct:>5.1f}% | {delta:>+8.4f}")


# =========================================================================
# SAVE HELPER
# =========================================================================

def _save_results(results: dict, output_dir: str, filename: str) -> None:
    def _serialize(obj):
        if isinstance(obj, PipelineResult):
            return {
                "name": obj.name,
                "final_accuracy": obj.final_accuracy,
                "final_f1_macro": obj.final_f1_macro,
                "total_human_labels": obj.total_human_labels,
                "total_ws_labels": obj.total_ws_labels,
                "total_labels": obj.total_labels,
                "ws_label_accuracy": obj.ws_label_accuracy,
                "ws_contribution_pct": obj.ws_contribution_pct,
                "human_savings_pct": obj.human_savings_pct,
                "baseline_accuracy": obj.baseline_accuracy,
                "baseline_f1_macro": obj.baseline_f1_macro,
                "n_classes": obj.n_classes,
                "history": obj.history,
            }
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / f"{filename}.json"
    json_path.write_text(json.dumps(results, indent=2, default=_serialize))
    print(f"\n  Results saved to {json_path}")
