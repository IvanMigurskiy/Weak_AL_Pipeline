#!/usr/bin/env python3
"""
WeakAL+AutoWS Hybrid Pipeline — Main Entry Point.

Research goal: Investigate and develop a hybrid text classification method
combining Active Learning and Weak Supervision strategies to improve
classification quality while significantly reducing manual data labeling costs.

Usage:
    python -m weakal_pipeline --dataset banking77       # Run on banking77
    python -m weakal_pipeline --quick                    # Quick test run
    python -m weakal_pipeline --mode hybrid              # Run only hybrid mode
    python -m weakal_pipeline --all-datasets             # Run on ALL datasets
    python -m weakal_pipeline --key-datasets             # Run on key datasets only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig, ExperimentConfig
from .data import load_dataset, DATASET_INFO
from .experiments import run_experiment, run_comparison
from .visualization import generate_all_plots


# =========================================================================
# KEY DATASETS — most important for thesis
# =========================================================================

KEY_DATASETS = [
    # Many classes: where hybrid should dominate
    "banking77",      # 77 classes — short banking queries
    "clinc150",       # 150 classes — multi-domain intents
    "bitext_insurance",  # 17 classes — insurance intents
    "bitext_ecommerce",  # 13 classes — e-commerce intents
    # Few classes: baseline comparison
    "customer_tickets",  # 4 classes — long noisy emails
]

ALL_DATASETS = list(DATASET_INFO.keys())


def _budget_for_dataset(dataset_name: str, n_classes: int) -> int:
    """Auto-compute reasonable human labeling budget based on dataset size."""
    # More classes = need more human labels, but still much less than full supervision
    if n_classes >= 50:
        return min(500, n_classes * 5)
    elif n_classes >= 20:
        return min(300, n_classes * 10)
    elif n_classes >= 10:
        return 200
    else:
        return 150


def _initial_per_class(n_classes: int) -> int:
    """How many seed labels per class."""
    if n_classes >= 50:
        return 1  # Just 1 per class for 50+ classes (expensive already)
    elif n_classes >= 20:
        return 2
    else:
        return 3


def run_dataset_experiment(
    dataset_name: str,
    mode: str = "all",
    n_repeats: int = 2,
    output_dir: str = "results",
    quick: bool = False,
    budget: int | None = None,
) -> dict:
    """Run experiments on a single dataset."""
    from .data import load_dataset

    # Load dataset to get class count for auto-budget
    cfg_probe = PipelineConfig(
        dataset_name=dataset_name,
        max_samples=500 if quick else None,
        random_seed=42,
    )

    print(f"\n{'#'*70}")
    print(f"# Dataset: {dataset_name}")
    info = DATASET_INFO.get(dataset_name, {})
    print(f"# Domain: {info.get('domain', 'N/A')} | "
          f"Text type: {info.get('text_type', 'N/A')} | "
          f"Expected classes: {info.get('expected_classes', '?')}")
    print(f"{'#'*70}")

    dataset = load_dataset(cfg_probe)
    n_classes = dataset.n_classes

    # Auto-compute budget
    if budget is None:
        budget = _budget_for_dataset(dataset_name, n_classes)
    per_class = _initial_per_class(n_classes)

    print(f"Actual classes: {n_classes}, Pool: {len(dataset.y_pool)}, "
          f"Test: {len(dataset.y_test)}")
    print(f"Budget: {budget} human labels, Seed: {per_class}/class")

    cfg = PipelineConfig(
        dataset_name=dataset_name,
        max_samples=500 if quick else None,
        max_human_labels=budget,
        batch_size=10 if n_classes < 50 else 20,
        initial_per_class=per_class,
        random_seed=42,
    )

    if mode == "all":
        results = run_comparison(base_config=cfg, n_repeats=n_repeats)
        ds_output = Path(output_dir) / dataset_name
        generate_all_plots(results, output_dir=ds_output)
        return results
    else:
        all_results = {}
        for repeat in range(n_repeats):
            repeat_cfg = PipelineConfig(
                random_seed=42 + repeat * 100,
                dataset_name=cfg.dataset_name,
                max_samples=cfg.max_samples,
                max_human_labels=cfg.max_human_labels,
                batch_size=cfg.batch_size,
                initial_per_class=cfg.initial_per_class,
            )
            exp = ExperimentConfig(
                name=f"{mode}_r{repeat}",
                pipeline_mode=mode,
                pipeline_config=repeat_cfg,
            )
            result = run_experiment(exp)
            if mode not in all_results:
                all_results[mode] = []
            all_results[mode].append(result)

        ds_output = Path(output_dir) / dataset_name
        generate_all_plots(all_results, output_dir=ds_output)
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="WeakAL+AutoWS Hybrid Pipeline for Text Classification"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test with small data")
    parser.add_argument(
        "--mode",
        choices=["baseline", "random_labels", "al_only", "ws_only", "hybrid", "all"],
        default="all",
        help="Pipeline mode (default: all)",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_INFO.keys()),
        default=None,
        help="Single dataset to use",
    )
    parser.add_argument(
        "--all-datasets", action="store_true",
        help="Run on ALL datasets",
    )
    parser.add_argument(
        "--key-datasets", action="store_true",
        help="Run on key datasets for thesis (banking77, clinc150, insurance, ecommerce, tickets)",
    )
    parser.add_argument(
        "--classifier", choices=["rf", "lr", "svm"], default="rf",
    )
    parser.add_argument("--budget", type=int, default=None, help="Human labeling budget")
    parser.add_argument("--repeats", type=int, default=2, help="Number of repeats")
    parser.add_argument("--output", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    print("=" * 70)
    print("WeakAL + AutoWS Hybrid Pipeline")
    print("Goal: Minimize human labeling while maintaining accuracy")
    print("=" * 70)
    print(f"Available datasets: {list(DATASET_INFO.keys())}")

    if args.all_datasets:
        datasets_to_run = ALL_DATASETS
    elif args.key_datasets:
        datasets_to_run = KEY_DATASETS
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        print("\nNo dataset specified. Use --dataset, --key-datasets, or --all-datasets")
        print("Quick start: python -m weakal_pipeline --key-datasets --quick")
        return

    # Run experiments
    all_dataset_results = {}
    for ds_name in datasets_to_run:
        print(f"\n\n{'='*70}")
        print(f"RUNNING: {ds_name}")
        print(f"{'='*70}")
        try:
            results = run_dataset_experiment(
                dataset_name=ds_name,
                mode=args.mode,
                n_repeats=args.repeats,
                output_dir=args.output,
                quick=args.quick,
                budget=args.budget,
            )
            all_dataset_results[ds_name] = results
        except Exception as e:
            print(f"ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate cross-dataset summary
    _print_cross_dataset_summary(all_dataset_results)

    print("\nDone!")


import numpy as np


def _print_cross_dataset_summary(all_results: dict) -> None:
    """Print summary table across all datasets."""
    print(f"\n\n{'='*100}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*100}")
    header = f"{'Dataset':30s} | {'Classes':>7s} | {'Baseline':>8s} | {'AL Only':>8s} | {'Hybrid':>8s} | {'Random':>8s} | {'Hybrid Δ':>8s} | {'WS%':>5s}"
    print(header)
    print("-" * 100)

    for ds_name, results in all_results.items():
        info = DATASET_INFO.get(ds_name, {})
        n_classes = info.get("expected_classes", "?")

        baseline_acc = np.mean([r.final_accuracy for r in results.get("baseline", [])]) if results.get("baseline") else 0
        al_acc = np.mean([r.final_accuracy for r in results.get("al_only", [])]) if results.get("al_only") else 0
        hybrid_acc = np.mean([r.final_accuracy for r in results.get("hybrid", [])]) if results.get("hybrid") else 0
        random_acc = np.mean([r.final_accuracy for r in results.get("random_labels", [])]) if results.get("random_labels") else 0
        ws_pct = np.mean([r.ws_contribution_pct for r in results.get("hybrid", [])]) if results.get("hybrid") else 0

        # Hybrid advantage over AL-only
        delta = hybrid_acc - al_acc

        print(f"{ds_name:30s} | {n_classes:>7} | {baseline_acc:>8.4f} | {al_acc:>8.4f} | {hybrid_acc:>8.4f} | {random_acc:>8.4f} | {delta:>+8.4f} | {ws_pct:>5.1f}%")

    print(f"{'='*100}")
    print("Δ = Hybrid accuracy minus AL-only accuracy (positive = hybrid wins)")


if __name__ == "__main__":
    main()
