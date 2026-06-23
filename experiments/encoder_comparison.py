"""
Encoder Comparison Experiment — Thesis Ablation Study.

Runs the hybrid pipeline with each available text encoder and produces
a comparison table showing accuracy, F1, WS contribution, and delta
vs the TF-IDF baseline.

Usage:
    python -m weakal_pipeline --encoder-comparison --dataset banking77
    python -m weakal_pipeline --encoder-comparison --key-datasets --quick
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..data import load_dataset
from ..pipeline import HybridPipeline, PipelineResult


# Encoders that can run without external model downloads
_AVAILABLE_ENCODERS = [
    "tfidf",
    "bm25",
]

# Encoders that require optional dependencies (gensim, torch, etc.)
_OPTIONAL_ENCODERS = [
    "fasttext_sparse",
    "fasttext_dense",
    "splade",
    "dense",
    "hybrid",
]


def _check_encoder_available(encoder_type: str) -> tuple[bool, str]:
    """Check if an encoder's dependencies are installed."""
    if encoder_type in ("tfidf", "bm25"):
        return True, ""

    if encoder_type in ("fasttext_sparse", "fasttext_dense"):
        try:
            import gensim  # noqa: F401
            return True, ""
        except ImportError:
            return False, "gensim not installed (pip install gensim)"

    if encoder_type == "splade":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True, ""
        except ImportError:
            return False, "torch/transformers not installed (pip install torch transformers)"

    if encoder_type == "dense":
        try:
            import sentence_transformers  # noqa: F401
            return True, ""
        except ImportError:
            return False, "sentence-transformers not installed"

    if encoder_type == "hybrid":
        # Hybrid needs both sparse and dense deps
        _, sparse_msg = _check_encoder_available("tfidf")
        _, dense_msg = _check_encoder_available("dense")
        if dense_msg:
            return False, dense_msg
        return True, ""

    return False, f"Unknown encoder type: {encoder_type}"


def run_single_encoder(
    dataset_name: str,
    encoder_type: str,
    budget: int | None = None,
    quick: bool = False,
    random_seed: int = 42,
) -> PipelineResult | None:
    """Run the hybrid pipeline with a single encoder.

    Returns:
        PipelineResult if successful, None if encoder unavailable or failed.
    """
    available, msg = _check_encoder_available(encoder_type)
    if not available:
        print(f"  [{encoder_type}] SKIP — {msg}")
        return None

    print(f"\n  [{encoder_type}] Running hybrid pipeline on {dataset_name}...")

    try:
        cfg = PipelineConfig(
            dataset_name=dataset_name,
            max_samples=500 if quick else None,
            encoder_type=encoder_type,
            random_seed=random_seed,
            max_human_labels=budget or 300,
            batch_size=10,
            initial_per_class=2,
        )

        dataset = load_dataset(cfg)

        # Auto-adjust budget based on dataset
        if budget is None:
            n_classes = dataset.n_classes
            if n_classes >= 50:
                cfg = PipelineConfig(
                    **{k: v for k, v in cfg.__dict__.items() if k != "max_human_labels"},
                    max_human_labels=min(500, n_classes * 5),
                )

        pipeline = HybridPipeline(cfg)
        start_time = time.time()
        result = pipeline.run(dataset)
        elapsed = time.time() - start_time

        print(f"  [{encoder_type}] Done in {elapsed:.1f}s — "
              f"Acc: {result.final_accuracy:.4f}, F1: {result.final_f1_macro:.4f}, "
              f"WS: {result.ws_contribution_pct:.1f}%")

        return result

    except Exception as e:
        print(f"  [{encoder_type}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_encoder_comparison(
    dataset_name: str,
    output_dir: str = "results",
    quick: bool = False,
    budget: int | None = None,
    encoders: list[str] | None = None,
    n_repeats: int = 1,
) -> dict[str, Any]:
    """Run encoder ablation study on a single dataset.

    Runs the hybrid pipeline with each encoder, then produces a comparison
    table with delta vs TF-IDF baseline. Results are saved to JSON.

    Args:
        dataset_name: Dataset to run on.
        output_dir: Where to save results.
        quick: Use small data subset.
        budget: Human labeling budget override.
        encoders: List of encoder types to compare. If None, auto-detects
            available encoders.
        n_repeats: Number of repeats per encoder (for variance estimation).

    Returns:
        Dict mapping encoder_type → summary stats.
    """
    print(f"\n{'='*70}")
    print(f"ENCODER COMPARISON — {dataset_name}")
    print(f"{'='*70}")

    # Determine which encoders to test
    if encoders is None:
        encoders = list(_AVAILABLE_ENCODERS)
        for enc in _OPTIONAL_ENCODERS:
            available, _ = _check_encoder_available(enc)
            if available:
                encoders.append(enc)

    print(f"Encoders to test: {encoders}")

    all_results: dict[str, list[PipelineResult]] = {}

    for encoder_type in encoders:
        all_results[encoder_type] = []
        for repeat in range(n_repeats):
            result = run_single_encoder(
                dataset_name=dataset_name,
                encoder_type=encoder_type,
                budget=budget,
                quick=quick,
                random_seed=42 + repeat * 100,
            )
            if result is not None:
                all_results[encoder_type].append(result)

    # Compute summary stats
    summary = _compute_comparison_table(all_results)

    # Print comparison table
    _print_comparison_table(summary, dataset_name)

    # Save results
    output_path = Path(output_dir) / dataset_name / "encoder_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    return summary


def _compute_comparison_table(
    all_results: dict[str, list[PipelineResult]],
) -> dict[str, Any]:
    """Compute mean stats and delta vs TF-IDF baseline for each encoder."""
    baseline_key = "tfidf"
    baseline_acc = 0.0
    baseline_f1 = 0.0

    # Compute per-encoder means
    encoder_stats: dict[str, dict[str, float]] = {}

    for encoder_type, results in all_results.items():
        if not results:
            continue

        accs = [r.final_accuracy for r in results]
        f1s = [r.final_f1_macro for r in results]
        ws_pcts = [r.ws_contribution_pct for r in results]
        ws_accs = [r.ws_label_accuracy for r in results]
        human_labels = [r.total_human_labels for r in results]
        ws_labels = [r.total_ws_labels for r in results]

        stats = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)) if len(accs) > 1 else 0.0,
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s)) if len(f1s) > 1 else 0.0,
            "ws_contribution_pct_mean": float(np.mean(ws_pcts)),
            "ws_label_accuracy_mean": float(np.mean(ws_accs)),
            "human_labels_mean": float(np.mean(human_labels)),
            "ws_labels_mean": float(np.mean(ws_labels)),
            "n_repeats": len(results),
        }

        if encoder_type == baseline_key:
            baseline_acc = stats["accuracy_mean"]
            baseline_f1 = stats["f1_macro_mean"]

        encoder_stats[encoder_type] = stats

    # Compute deltas vs baseline
    for encoder_type, stats in encoder_stats.items():
        stats["delta_accuracy"] = stats["accuracy_mean"] - baseline_acc
        stats["delta_f1_macro"] = stats["f1_macro_mean"] - baseline_f1
        stats["is_baseline"] = encoder_type == baseline_key

    return encoder_stats


def _print_comparison_table(summary: dict[str, dict], dataset_name: str) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"  ENCODER COMPARISON TABLE — {dataset_name}")
    print(f"{'='*90}")
    header = (
        f"{'Encoder':20s} | {'Accuracy':>10s} | {'F1 Macro':>10s} | "
        f"{'Δ Acc':>10s} | {'Δ F1':>10s} | {'WS%':>6s} | {'WS-Acc':>8s}"
    )
    print(header)
    print("-" * 90)

    # Sort: baseline first, then by accuracy descending
    sorted_encoders = sorted(
        summary.keys(),
        key=lambda k: (0 if summary[k].get("is_baseline") else 1, -summary[k]["accuracy_mean"]),
    )

    for enc in sorted_encoders:
        s = summary[enc]
        delta_acc = s["delta_accuracy"]
        delta_f1 = s["delta_f1_macro"]

        marker = " ← baseline" if s.get("is_baseline") else ""

        print(
            f"{enc:20s} | {s['accuracy_mean']:>10.4f} | {s['f1_macro_mean']:>10.4f} | "
            f"{delta_acc:>+10.4f} | {delta_f1:>+10.4f} | "
            f"{s['ws_contribution_pct_mean']:>5.1f}% | "
            f"{s['ws_label_accuracy_mean']:>8.4f}"
            f"{marker}"
        )

    print(f"{'='*90}")
    print("Δ = difference from TF-IDF baseline (positive = encoder wins)")
