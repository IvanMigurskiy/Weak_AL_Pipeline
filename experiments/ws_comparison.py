#!/usr/bin/env python3
"""
WS Accuracy Improvement Comparison Experiment.

Runs the original hybrid pipeline and all 6 enhanced variants,
then compares their WS label accuracy and final classification accuracy.

Usage:
    python -m weakal_pipeline.run_ws_comparison --dataset customer_tickets
    python -m weakal_pipeline.run_ws_comparison --dataset banking77
    python -m weakal_pipeline.run_ws_comparison --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .data import load_dataset, DATASET_INFO
from .pipeline import HybridPipeline, ALOnlyPipeline, PipelineResult
from .pipeline.enhanced_hybrid import (
    T1WeightedPipeline,
    T2VerificationPipeline,
    T3CalibratedPipeline,
    T4SelfTrainingPipeline,
    T5UnanimousPipeline,
    T6PerClassThresholdPipeline,
    ComboPipeline,
)


# =========================================================================
# TECHNIQUE REGISTRY
# =========================================================================

TECHNIQUE_INFO = {
    "original":  {"name": "Original Hybrid",     "desc": "Base AL+WS pipeline (current implementation)"},
    "T1":        {"name": "T1: Weighted Training", "desc": "WS labels get lower sample_weight (0.5)"},
    "T2":        {"name": "T2: WS Verification",   "desc": "Remove WS labels classifier disagrees with"},
    "T3":        {"name": "T3: Calibrated LFs",    "desc": "Platt-scale LF confidence scores"},
    "T4":        {"name": "T4: Self-Training",     "desc": "Post-loop iterative pseudo-labeling"},
    "T5":        {"name": "T5: Unanimous Voting",  "desc": "All voting LFs must agree"},
    "T6":        {"name": "T6: Per-Class Thresh",  "desc": "Class-specific confidence thresholds"},
    "combo":     {"name": "Combo: T1+T2+T3+T6",   "desc": "All conservative techniques combined"},
}


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


def run_ws_comparison(
    dataset_name: str = "customer_tickets",
    quick: bool = False,
    n_repeats: int = 2,
    budget: int | None = None,
    output_dir: str = "results/ws_comparison",
) -> dict:
    """
    Run all WS accuracy techniques on a dataset and compare results.
    """
    # Load dataset
    cfg_probe = PipelineConfig(
        dataset_name=dataset_name,
        max_samples=500 if quick else None,
    )

    print(f"\n{'#'*70}")
    print(f"# WS Accuracy Comparison: {dataset_name}")
    info = DATASET_INFO.get(dataset_name, {})
    print(f"# Domain: {info.get('domain', 'N/A')} | "
          f"Text: {info.get('text_type', 'N/A')} | "
          f"Expected classes: {info.get('expected_classes', '?')}")
    print(f"{'#'*70}")

    dataset = load_dataset(cfg_probe)
    n_classes = dataset.n_classes

    if budget is None:
        budget = _budget_for_dataset(dataset_name, n_classes)
    per_class = _initial_per_class(n_classes)

    print(f"Actual classes: {n_classes}, Pool: {len(dataset.y_pool)}, "
          f"Test: {len(dataset.y_test)}")
    print(f"Budget: {budget} human labels, Seed: {per_class}/class")

    # Also run AL-only as baseline
    results: dict[str, list[PipelineResult]] = {}

    for repeat in range(n_repeats):
        seed = 42 + repeat * 100
        print(f"\n{'='*70}")
        print(f"REPEAT {repeat + 1}/{n_repeats} (seed={seed})")
        print(f"{'='*70}")

        cfg = PipelineConfig(
            dataset_name=dataset_name,
            max_samples=500 if quick else None,
            max_human_labels=budget,
            batch_size=10 if n_classes < 50 else 20,
            initial_per_class=per_class,
            random_seed=seed,
        )

        # --- AL-only baseline ---
        print(f"\n--- AL Only ---")
        al_pipe = ALOnlyPipeline(cfg)
        al_result = al_pipe.run(dataset)
        al_result.name = f"al_only_r{repeat}"
        results.setdefault("al_only", []).append(al_result)

        # --- Original hybrid ---
        print(f"\n--- Original Hybrid ---")
        orig_pipe = HybridPipeline(cfg)
        orig_result = orig_pipe.run(dataset)
        orig_result.name = f"original_r{repeat}"
        results.setdefault("original", []).append(orig_result)

        # --- T1: Weighted Training ---
        print(f"\n--- T1: Weighted Training ---")
        t1_pipe = T1WeightedPipeline(cfg, ws_weight=0.5)
        t1_result = t1_pipe.run(dataset)
        results.setdefault("T1", []).append(t1_result)

        # --- T2: WS Verification ---
        print(f"\n--- T2: WS Verification ---")
        t2_pipe = T2VerificationPipeline(cfg, verify_every=3)
        t2_result = t2_pipe.run(dataset)
        results.setdefault("T2", []).append(t2_result)

        # --- T3: Calibrated LFs ---
        print(f"\n--- T3: Calibrated LFs ---")
        t3_pipe = T3CalibratedPipeline(cfg)
        t3_result = t3_pipe.run(dataset)
        results.setdefault("T3", []).append(t3_result)

        # --- T4: Self-Training ---
        print(f"\n--- T4: Self-Training ---")
        t4_pipe = T4SelfTrainingPipeline(cfg, max_pseudo_labels=200)
        t4_result = t4_pipe.run(dataset)
        results.setdefault("T4", []).append(t4_result)

        # --- T5: Unanimous Voting ---
        print(f"\n--- T5: Unanimous Voting ---")
        t5_pipe = T5UnanimousPipeline(cfg, min_voters=2)
        t5_result = t5_pipe.run(dataset)
        results.setdefault("T5", []).append(t5_result)

        # --- T6: Per-Class Thresholds ---
        print(f"\n--- T6: Per-Class Thresholds ---")
        t6_pipe = T6PerClassThresholdPipeline(cfg)
        t6_result = t6_pipe.run(dataset)
        results.setdefault("T6", []).append(t6_result)

        # --- Combo: T1+T2+T3+T6 ---
        print(f"\n--- Combo: T1+T2+T3+T6 ---")
        combo_pipe = ComboPipeline(cfg, ws_weight=0.5, verify_every=3)
        combo_result = combo_pipe.run(dataset)
        results.setdefault("combo", []).append(combo_result)

    # Print summary
    _print_comparison_table(results, dataset_name, n_classes)

    # Save results
    _save_results(results, dataset_name, output_dir)

    return results


def _print_comparison_table(
    results: dict[str, list[PipelineResult]],
    dataset_name: str,
    n_classes: int,
) -> None:
    """Print a comparison table of all techniques."""
    print(f"\n\n{'='*120}")
    print(f"WS ACCURACY COMPARISON: {dataset_name} ({n_classes} classes)")
    print(f"{'='*120}")

    header = (
        f"{'Technique':30s} | {'Accuracy':>8s} | {'F1 Macro':>8s} | "
        f"{'Human':>6s} | {'WS':>6s} | {'Total':>6s} | "
        f"{'WS Acc':>7s} | {'WS%':>5s} | {'vs AL':>7s} | {'vs Orig':>7s}"
    )
    print(header)
    print("-" * 120)

    # Compute AL-only baseline
    al_accs = [r.final_accuracy for r in results.get("al_only", [])]
    al_mean = np.mean(al_accs) if al_accs else 0.0

    # Compute original hybrid baseline
    orig_accs = [r.final_accuracy for r in results.get("original", [])]
    orig_mean = np.mean(orig_accs) if orig_accs else 0.0

    technique_order = ["al_only", "original", "T1", "T2", "T3", "T4", "T5", "T6", "combo"]

    for tech_key in technique_order:
        if tech_key not in results or not results[tech_key]:
            continue

        tech_results = results[tech_key]
        info = TECHNIQUE_INFO.get(tech_key, {"name": tech_key})
        name = info["name"]

        acc = np.mean([r.final_accuracy for r in tech_results])
        f1 = np.mean([r.final_f1_macro for r in tech_results])
        human = int(np.mean([r.total_human_labels for r in tech_results]))
        ws = int(np.mean([r.total_ws_labels for r in tech_results]))
        total = int(np.mean([r.total_labels for r in tech_results]))
        ws_acc_vals = [r.ws_label_accuracy for r in tech_results if r.ws_label_accuracy > 0]
        ws_acc = np.mean(ws_acc_vals) if ws_acc_vals else 0.0
        ws_pct = np.mean([r.ws_contribution_pct for r in tech_results])

        delta_al = acc - al_mean
        delta_orig = acc - orig_mean

        print(
            f"{name:30s} | {acc:>8.4f} | {f1:>8.4f} | "
            f"{human:>6d} | {ws:>6d} | {total:>6d} | "
            f"{ws_acc:>7.4f} | {ws_pct:>5.1f}% | "
            f"{delta_al:>+7.4f} | {delta_orig:>+7.4f}"
        )

    print(f"{'='*120}")
    print("vs AL    = accuracy improvement over AL-only")
    print("vs Orig  = accuracy improvement over Original Hybrid")


def _save_results(
    results: dict[str, list[PipelineResult]],
    dataset_name: str,
    output_dir: str,
) -> None:
    """Save results to JSON."""
    out_path = Path(output_dir) / dataset_name
    out_path.mkdir(parents=True, exist_ok=True)

    json_results = {}
    for tech_key, tech_results in results.items():
        json_results[tech_key] = []
        for r in tech_results:
            json_results[tech_key].append({
                "name": r.name,
                "final_accuracy": r.final_accuracy,
                "final_f1_macro": r.final_f1_macro,
                "total_human_labels": r.total_human_labels,
                "total_ws_labels": r.total_ws_labels,
                "total_labels": r.total_labels,
                "ws_label_accuracy": r.ws_label_accuracy,
                "ws_contribution_pct": r.ws_contribution_pct,
                "human_savings_pct": r.human_savings_pct,
                "baseline_accuracy": r.baseline_accuracy,
                "n_classes": r.n_classes,
                "history": r.history,
            })

    json_path = out_path / "ws_comparison_results.json"
    json_path.write_text(json.dumps(json_results, indent=2))
    print(f"\nResults saved to {json_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WS Accuracy Improvement Comparison"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_INFO.keys()),
        default="customer_tickets",
        help="Dataset to test on",
    )
    parser.add_argument("--quick", action="store_true", help="Quick test with small data")
    parser.add_argument("--repeats", type=int, default=2, help="Number of repeats")
    parser.add_argument("--budget", type=int, default=None, help="Human labeling budget")
    parser.add_argument("--output", type=str, default="results/ws_comparison")

    args = parser.parse_args()

    run_ws_comparison(
        dataset_name=args.dataset,
        quick=args.quick,
        n_repeats=args.repeats,
        budget=args.budget,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
