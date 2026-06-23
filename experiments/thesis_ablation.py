"""
Thesis Ablation Study — Full encoder × pipeline × dataset comparison.

Produces the main comparison tables for the thesis:
  Table 1: Encoder comparison (7 encoders × 3 datasets)
  Table 2: Pipeline mode comparison (AL-only vs WS-only vs Hybrid)
  Table 3: TopicLF impact (with vs without)
  Table 4: hybrid_alpha sensitivity analysis

Usage:
    python -m Weak_AL_Pipeline.experiments.thesis_ablation --quick
    python -m Weak_AL_Pipeline.experiments.thesis_ablation --dataset banking77
    python -m Weak_AL_Pipeline.experiments.thesis_ablation --full
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

# Key datasets for thesis (varied domain, class count, text length)
KEY_DATASETS = [
    "banking77",        # 77 classes, short queries (main thesis dataset)
    "customer_tickets", # 4 classes, long emails
    "bitext_ecommerce", # 13 classes, short queries
]

# All 7 encoders
ALL_ENCODERS = [
    "tfidf",
    "bm25",
    "fasttext_sparse",
    "fasttext_dense",
    "dense",
    "splade",
    "hybrid",
]

# Pipeline modes for comparison
PIPELINE_MODES = ["al_only", "ws_only", "hybrid"]


# =========================================================================
# BUDGET HELPERS
# =========================================================================

def _budget_for_dataset(dataset_name: str, n_classes: int) -> int:
    """Auto-compute human labeling budget based on dataset size."""
    if n_classes >= 50:
        return min(500, n_classes * 5)
    elif n_classes >= 20:
        return min(300, n_classes * 10)
    elif n_classes >= 10:
        return 200
    else:
        return 150


def _initial_per_class(n_classes: int) -> int:
    """Auto-compute initial labels per class."""
    if n_classes >= 50:
        return 1
    elif n_classes >= 20:
        return 2
    else:
        return 3


# =========================================================================
# RUNNER FUNCTIONS
# =========================================================================

def _run_pipeline(
    mode: str,
    cfg: PipelineConfig,
    dataset,
) -> PipelineResult | None:
    """Run a single pipeline mode and return result."""
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


def _check_encoder_available(encoder_type: str) -> tuple[bool, str]:
    """Check if encoder dependencies are installed."""
    if encoder_type in ("tfidf", "bm25"):
        return True, ""

    if encoder_type in ("fasttext_sparse", "fasttext_dense"):
        try:
            import gensim  # noqa: F401
            return True, ""
        except ImportError:
            return False, "gensim not installed"

    if encoder_type == "splade":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True, ""
        except ImportError:
            return False, "torch/transformers not installed"

    if encoder_type == "dense":
        try:
            import sentence_transformers  # noqa: F401
            return True, ""
        except ImportError:
            return False, "sentence-transformers not installed"

    if encoder_type == "hybrid":
        _, msg = _check_encoder_available("dense")
        if msg:
            return False, msg
        return True, ""

    return False, f"Unknown: {encoder_type}"


# =========================================================================
# TABLE 1: ENCODER COMPARISON
# =========================================================================

def run_encoder_comparison(
    datasets: list[str] | None = None,
    encoders: list[str] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/thesis",
) -> dict[str, Any]:
    """Table 1: Compare all encoders on each dataset using hybrid pipeline.

    For each dataset × encoder, runs the hybrid pipeline and records
    accuracy, F1, WS contribution, and delta vs TF-IDF baseline.
    """
    if datasets is None:
        datasets = KEY_DATASETS
    if encoders is None:
        encoders = [e for e in ALL_ENCODERS if _check_encoder_available(e)[0]]

    print(f"\n{'#'*80}")
    print(f"# TABLE 1: ENCODER COMPARISON")
    print(f"# Datasets: {datasets}")
    print(f"# Encoders: {encoders}")
    print(f"# Repeats: {n_repeats}")
    print(f"{'#'*80}")

    all_results: dict[str, dict[str, list[PipelineResult]]] = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")

        # Load dataset once (shared across encoders)
        cfg_probe = PipelineConfig(
            dataset_name=dataset_name,
            max_samples=500 if quick else None,
            encoder_type="fasttext_dense",  # placeholder
        )
        dataset = load_dataset(cfg_probe)
        n_classes = dataset.n_classes
        budget = _budget_for_dataset(dataset_name, n_classes)
        per_class = _initial_per_class(n_classes)

        print(f"  Classes: {n_classes}, Pool: {len(dataset.y_pool)}, "
              f"Test: {len(dataset.y_test)}, Budget: {budget}")

        all_results[dataset_name] = {}

        for encoder_type in encoders:
            all_results[dataset_name][encoder_type] = []

            for repeat in range(n_repeats):
                seed = 42 + repeat * 100
                cfg = PipelineConfig(
                    dataset_name=dataset_name,
                    max_samples=500 if quick else None,
                    encoder_type=encoder_type,
                    max_human_labels=budget,
                    batch_size=10 if n_classes < 50 else 20,
                    initial_per_class=per_class,
                    random_seed=seed,
                    classifier_type="lr",  # LR for fair+fast comparison across all encoders
                )
                # Need to reload dataset for each encoder (different features)
                dataset = load_dataset(cfg)

                result = _run_pipeline("hybrid", cfg, dataset)
                if result is not None:
                    result.name = f"{encoder_type}_r{repeat}"
                    all_results[dataset_name][encoder_type].append(result)

    # Print and save
    _print_encoder_table(all_results)
    _save_results(all_results, output_dir, "table1_encoder_comparison")

    return all_results


# =========================================================================
# TABLE 2: PIPELINE MODE COMPARISON
# =========================================================================

def run_pipeline_comparison(
    datasets: list[str] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/thesis",
) -> dict[str, Any]:
    """Table 2: Compare AL-only vs WS-only vs Hybrid pipelines.

    Uses TF-IDF encoder (baseline). Key question: does WS + AL
    outperform AL-only and WS-only individually?
    """
    if datasets is None:
        datasets = KEY_DATASETS

    print(f"\n{'#'*80}")
    print(f"# TABLE 2: PIPELINE MODE COMPARISON")
    print(f"# Datasets: {datasets}")
    print(f"# Encoder: tfidf (baseline)")
    print(f"# Repeats: {n_repeats}")
    print(f"{'#'*80}")

    all_results: dict[str, dict[str, list[PipelineResult]]] = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")

        cfg_probe = PipelineConfig(
            dataset_name=dataset_name,
            max_samples=500 if quick else None,
            encoder_type="fasttext_dense",
        )
        dataset = load_dataset(cfg_probe)
        n_classes = dataset.n_classes
        budget = _budget_for_dataset(dataset_name, n_classes)
        per_class = _initial_per_class(n_classes)

        all_results[dataset_name] = {}

        for mode in PIPELINE_MODES:
            all_results[dataset_name][mode] = []

            for repeat in range(n_repeats):
                seed = 42 + repeat * 100
                cfg = PipelineConfig(
                    dataset_name=dataset_name,
                    max_samples=500 if quick else None,
                    encoder_type="fasttext_dense",
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

    _print_pipeline_table(all_results)
    _save_results(all_results, output_dir, "table2_pipeline_comparison")

    return all_results


# =========================================================================
# TABLE 3: TOPICLF IMPACT
# =========================================================================

def run_topiclf_comparison(
    datasets: list[str] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/thesis",
) -> dict[str, Any]:
    """Table 3: Compare hybrid pipeline with and without TopicLF.

    Tests whether adding NMF/LDA-based TopicLF improves WS quality
    and final accuracy.
    """
    if datasets is None:
        datasets = KEY_DATASETS

    print(f"\n{'#'*80}")
    print(f"# TABLE 3: TOPICLF IMPACT")
    print(f"# Datasets: {datasets}")
    print(f"# Repeats: {n_repeats}")
    print(f"{'#'*80}")

    all_results: dict[str, dict[str, list[PipelineResult]]] = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*70}")

        cfg_probe = PipelineConfig(
            dataset_name=dataset_name,
            max_samples=500 if quick else None,
            encoder_type="fasttext_dense",
        )
        dataset = load_dataset(cfg_probe)
        n_classes = dataset.n_classes
        budget = _budget_for_dataset(dataset_name, n_classes)
        per_class = _initial_per_class(n_classes)

        all_results[dataset_name] = {}

        for use_topic in [False, True]:
            label = "topic_on" if use_topic else "topic_off"
            all_results[dataset_name][label] = []

            for repeat in range(n_repeats):
                seed = 42 + repeat * 100
                cfg = PipelineConfig(
                    dataset_name=dataset_name,
                    max_samples=500 if quick else None,
                    encoder_type="fasttext_dense",
                    max_human_labels=budget,
                    batch_size=10 if n_classes < 50 else 20,
                    initial_per_class=per_class,
                    random_seed=seed,
                    use_topic_lf=use_topic,
                    topic_n_topics=min(n_classes, 10),
                )
                dataset = load_dataset(cfg)
                result = _run_pipeline("hybrid", cfg, dataset)
                if result is not None:
                    result.name = f"{label}_r{repeat}"
                    all_results[dataset_name][label].append(result)

    _print_topiclf_table(all_results)
    _save_results(all_results, output_dir, "table3_topiclf_impact")

    return all_results


# =========================================================================
# TABLE 4: HYBRID_ALPHA SENSITIVITY
# =========================================================================

def run_alpha_sensitivity(
    dataset_name: str = "banking77",
    alphas: list[float] | None = None,
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/thesis",
) -> dict[str, Any]:
    """Table 4: HybridEncoder alpha sensitivity analysis.

    Tests different sparse/dense weight ratios to find optimal alpha.
    """
    if alphas is None:
        alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    print(f"\n{'#'*80}")
    print(f"# TABLE 4: HYBRID_ALPHA SENSITIVITY")
    print(f"# Dataset: {dataset_name}")
    print(f"# Alphas: {alphas}")
    print(f"{'#'*80}")

    # Probe dataset to get class count for budget
    cfg_probe = PipelineConfig(
        dataset_name=dataset_name,
        max_samples=500 if quick else None,
        encoder_type="fasttext_dense",
    )
    dataset_probe = load_dataset(cfg_probe)
    n_classes = dataset_probe.n_classes
    budget = _budget_for_dataset(dataset_name, n_classes)

    all_results: dict[str, list[PipelineResult]] = {}

    for alpha in alphas:
        label = f"alpha_{alpha:.1f}"
        all_results[label] = []

        for repeat in range(n_repeats):
            seed = 42 + repeat * 100
            cfg = PipelineConfig(
                dataset_name=dataset_name,
                max_samples=500 if quick else None,
                encoder_type="hybrid",
                hybrid_alpha=alpha,
                max_human_labels=budget,
                batch_size=10,
                initial_per_class=2,
                random_seed=seed,
            )
            dataset = load_dataset(cfg)
            result = _run_pipeline("hybrid", cfg, dataset)
            if result is not None:
                result.name = f"{label}_r{repeat}"
                all_results[label].append(result)

    _print_alpha_table(all_results, dataset_name)
    _save_results(all_results, output_dir, "table4_alpha_sensitivity")

    return all_results


# =========================================================================
# FULL ABLATION (ALL TABLES)
# =========================================================================

def run_full_ablation(
    quick: bool = False,
    n_repeats: int = 1,
    output_dir: str = "results/thesis",
) -> dict[str, Any]:
    """Run the complete thesis ablation study (all 4 tables)."""
    print(f"\n{'#'*80}")
    print(f"# FULL THESIS ABLATION STUDY")
    print(f"# Quick: {quick}, Repeats: {n_repeats}")
    print(f"{'#'*80}")

    results = {}

    # Table 1: Encoder comparison
    results["table1"] = run_encoder_comparison(
        quick=quick, n_repeats=n_repeats, output_dir=output_dir
    )

    # Table 2: Pipeline mode comparison
    results["table2"] = run_pipeline_comparison(
        quick=quick, n_repeats=n_repeats, output_dir=output_dir
    )

    # Table 3: TopicLF impact
    results["table3"] = run_topiclf_comparison(
        quick=quick, n_repeats=n_repeats, output_dir=output_dir
    )

    # Table 4: Alpha sensitivity
    results["table4"] = run_alpha_sensitivity(
        quick=quick, n_repeats=n_repeats, output_dir=output_dir
    )

    print(f"\n\n{'='*80}")
    print("FULL ABLATION STUDY COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*80}")

    return results


# =========================================================================
# PRINTING HELPERS
# =========================================================================

def _print_encoder_table(all_results: dict) -> None:
    """Print Table 1: Encoder comparison across datasets."""
    print(f"\n\n{'='*100}")
    print("  TABLE 1: ENCODER COMPARISON (Hybrid Pipeline)")
    print(f"{'='*100}")

    for dataset_name, encoder_results in all_results.items():
        print(f"\n  Dataset: {dataset_name}")
        print(f"  {'Encoder':20s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | "
              f"{'Baseline':>8s} | {'WS%':>6s} | {'WS Acc':>7s} | {'Δ Acc':>8s}")
        print(f"  {'-'*85}")

        # TF-IDF baseline for delta computation
        tfidf_acc = 0.0
        if "tfidf" in encoder_results and encoder_results["tfidf"]:
            tfidf_acc = np.mean([r.final_accuracy for r in encoder_results["tfidf"]])

        # Sort: tfidf first, then by accuracy descending
        sorted_encs = sorted(
            encoder_results.keys(),
            key=lambda k: (0 if k == "tfidf" else 1,
                           -np.mean([r.final_accuracy for r in encoder_results[k]]) if encoder_results[k] else 0),
        )

        for enc in sorted_encs:
            results = encoder_results[enc]
            if not results:
                continue
            acc = np.mean([r.final_accuracy for r in results])
            f1 = np.mean([r.final_f1_macro for r in results])
            baseline = np.mean([r.baseline_accuracy for r in results])
            ws_pct = np.mean([r.ws_contribution_pct for r in results])
            ws_acc_vals = [r.ws_label_accuracy for r in results if r.ws_label_accuracy > 0]
            ws_acc = np.mean(ws_acc_vals) if ws_acc_vals else 0.0
            delta = acc - tfidf_acc

            marker = " *" if enc == "tfidf" else ""
            print(f"  {enc:20s} | {acc:>8.4f} | {f1:>8.4f} | "
                  f"{baseline:>8.4f} | {ws_pct:>5.1f}% | {ws_acc:>7.4f} | "
                  f"{delta:>+8.4f}{marker}")

    print(f"\n  * = TF-IDF baseline  |  Δ = difference from TF-IDF  |  WS% = WS label share")


def _print_pipeline_table(all_results: dict) -> None:
    """Print Table 2: Pipeline mode comparison."""
    print(f"\n\n{'='*100}")
    print("  TABLE 2: PIPELINE MODE COMPARISON (TF-IDF Encoder)")
    print(f"{'='*100}")

    for dataset_name, mode_results in all_results.items():
        print(f"\n  Dataset: {dataset_name}")
        print(f"  {'Mode':15s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | "
              f"{'Human':>6s} | {'WS':>6s} | {'Total':>6s} | {'WS Acc':>7s} | "
              f"{'Δ vs AL':>8s} | {'Δ vs WS':>8s}")
        print(f"  {'-'*105}")

        al_acc = np.mean([r.final_accuracy for r in mode_results.get("al_only", [])]) if mode_results.get("al_only") else 0.0
        ws_acc_base = np.mean([r.final_accuracy for r in mode_results.get("ws_only", [])]) if mode_results.get("ws_only") else 0.0

        for mode in ["al_only", "ws_only", "hybrid"]:
            results = mode_results.get(mode, [])
            if not results:
                continue
            acc = np.mean([r.final_accuracy for r in results])
            f1 = np.mean([r.final_f1_macro for r in results])
            human = int(np.mean([r.total_human_labels for r in results]))
            ws = int(np.mean([r.total_ws_labels for r in results]))
            total = int(np.mean([r.total_labels for r in results]))
            ws_acc_vals = [r.ws_label_accuracy for r in results if r.ws_label_accuracy > 0]
            ws_acc = np.mean(ws_acc_vals) if ws_acc_vals else 0.0

            delta_al = acc - al_acc
            delta_ws = acc - ws_acc_base

            print(f"  {mode:15s} | {acc:>8.4f} | {f1:>8.4f} | "
                  f"{human:>6d} | {ws:>6d} | {total:>6d} | {ws_acc:>7.4f} | "
                  f"{delta_al:>+8.4f} | {delta_ws:>+8.4f}")

    print(f"\n  Δ vs AL  = improvement over AL-only")
    print(f"  Δ vs WS  = improvement over WS-only")


def _print_topiclf_table(all_results: dict) -> None:
    """Print Table 3: TopicLF impact."""
    print(f"\n\n{'='*90}")
    print("  TABLE 3: TOPICLF IMPACT (Hybrid Pipeline, TF-IDF)")
    print(f"{'='*90}")

    for dataset_name, topic_results in all_results.items():
        print(f"\n  Dataset: {dataset_name}")
        print(f"  {'Config':15s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | "
              f"{'WS%':>6s} | {'WS Acc':>7s} | {'Δ Acc':>8s}")
        print(f"  {'-'*70}")

        off_results = topic_results.get("topic_off", [])
        off_acc = np.mean([r.final_accuracy for r in off_results]) if off_results else 0.0

        for label in ["topic_off", "topic_on"]:
            results = topic_results.get(label, [])
            if not results:
                continue
            acc = np.mean([r.final_accuracy for r in results])
            f1 = np.mean([r.final_f1_macro for r in results])
            ws_pct = np.mean([r.ws_contribution_pct for r in results])
            ws_acc_vals = [r.ws_label_accuracy for r in results if r.ws_label_accuracy > 0]
            ws_acc = np.mean(ws_acc_vals) if ws_acc_vals else 0.0
            delta = acc - off_acc

            print(f"  {label:15s} | {acc:>8.4f} | {f1:>8.4f} | "
                  f"{ws_pct:>5.1f}% | {ws_acc:>7.4f} | {delta:>+8.4f}")

    print(f"\n  Δ Acc = accuracy change from adding TopicLF")


def _print_alpha_table(all_results: dict, dataset_name: str) -> None:
    """Print Table 4: Alpha sensitivity."""
    print(f"\n\n{'='*80}")
    print(f"  TABLE 4: HYBRID_ALPHA SENSITIVITY ({dataset_name})")
    print(f"{'='*80}")

    print(f"  {'Alpha':8s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | "
          f"{'WS%':>6s} | {'WS Acc':>7s} | {'Note':20s}")
    print(f"  {'-'*70}")

    for label, results in sorted(all_results.items()):
        if not results:
            continue
        alpha = float(label.split("_")[1])
        acc = np.mean([r.final_accuracy for r in results])
        f1 = np.mean([r.final_f1_macro for r in results])
        ws_pct = np.mean([r.ws_contribution_pct for r in results])
        ws_acc_vals = [r.ws_label_accuracy for r in results if r.ws_label_accuracy > 0]
        ws_acc = np.mean(ws_acc_vals) if ws_acc_vals else 0.0

        note = ""
        if alpha == 0.0:
            note = "dense only"
        elif alpha == 1.0:
            note = "sparse only"
        elif alpha == 0.5:
            note = "equal weight (default)"

        print(f"  {alpha:>7.1f} | {acc:>8.4f} | {f1:>8.4f} | "
              f"{ws_pct:>5.1f}% | {ws_acc:>7.4f} | {note}")

    print(f"\n  alpha=1.0 → sparse only, alpha=0.0 → dense only")


# =========================================================================
# SAVE HELPER
# =========================================================================

def _save_results(
    results: dict,
    output_dir: str,
    filename: str,
) -> None:
    """Save results dict to JSON."""

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
                "n_pool": obj.n_pool,
                "n_test": obj.n_test,
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


# =========================================================================
# MAIN
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Thesis Ablation Study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with small data (500 samples)")
    parser.add_argument("--full", action="store_true",
                        help="Run all 4 tables (default: just Table 1)")
    parser.add_argument("--table", type=int, choices=[1, 2, 3, 4],
                        help="Run a specific table only")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run on a specific dataset only")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeats for variance estimation")
    parser.add_argument("--output", type=str, default="results/thesis",
                        help="Output directory for results")

    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else None

    if args.full:
        run_full_ablation(quick=args.quick, n_repeats=args.repeats, output_dir=args.output)
    elif args.table == 1:
        run_encoder_comparison(datasets=datasets, quick=args.quick,
                               n_repeats=args.repeats, output_dir=args.output)
    elif args.table == 2:
        run_pipeline_comparison(datasets=datasets, quick=args.quick,
                                n_repeats=args.repeats, output_dir=args.output)
    elif args.table == 3:
        run_topiclf_comparison(datasets=datasets, quick=args.quick,
                               n_repeats=args.repeats, output_dir=args.output)
    elif args.table == 4:
        run_alpha_sensitivity(dataset_name=args.dataset or "banking77",
                              quick=args.quick, n_repeats=args.repeats,
                              output_dir=args.output)
    else:
        # Default: Table 1 only
        run_encoder_comparison(datasets=datasets, quick=args.quick,
                               n_repeats=args.repeats, output_dir=args.output)


if __name__ == "__main__":
    main()
