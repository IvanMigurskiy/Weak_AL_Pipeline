"""
Full Experiment Runner for Thesis — multi-seed, multi-dataset, step-wise ablation.

Produces publication-ready results for every pipeline step (R1 + R2) and
all 14 pipeline-modifying techniques + Combo, with statistical significance
across multiple random seeds.

Experiment Structure:
  Phase A — Step-wise ablation (sequential step-locking, full scale)
    Step 1: Text Encoding          → R1: Does encoding matter?  R2: Which encoder?
    Step 1b: Hybrid Alpha          → (only if hybrid encoder wins)
    Step 2: AL Query Strategy      → R1: AL vs Random?  R2: Which strategy?
    Step 4: WS Label Aggregation   → R1: DS vs MV?  R2: Which aggregator?
    Step 5: Auto-labeling          → R1: Does auto-label help?  R2: Which method?
    Step 3: WS Labeling Functions  → R1: Does TopicLF help?  R2: With or without?
    Step 6: Pipeline Integration   → R1: Hybrid vs AL-only vs WS-only
  Phase B — Technique evaluation (against Reference Config)
    T1–T14 + Combo

Each experiment is run with N_SEEDS random seeds and results are aggregated
with mean ± std and paired t-test significance.

Outputs:
  - CSV results table (all raw runs)
  - Summary CSV (mean ± std per technique per dataset)
  - JSON with full PipelineResult history for plotting

Run:
    python -m Weak_AL_Pipeline.experiments.full_experiments
    python -m Weak_AL_Pipeline.experiments.full_experiments --phase A
    python -m Weak_AL_Pipeline.experiments.full_experiments --phase B
    python -m Weak_AL_Pipeline.experiments.full_experiments --step 1
    python -m Weak_AL_Pipeline.experiments.full_experiments --dataset customer_tickets
    python -m Weak_AL_Pipeline.experiments.full_experiments --seeds 5

Colab:
    import sys; sys.path.insert(0, "/content/repo")
    from Weak_AL_Pipeline.experiments.full_experiments import run_full_experiments
    results = run_full_experiments()
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import warnings

# Silence Liblinear convergence warnings — handled via scaling + max_iter=10000
warnings.filterwarnings("ignore", message="Liblinear failed to converge")

# ── #5 OPTIMIZATION: joblib for parallel seed runs ──
from joblib import Parallel, delayed

from ..config import PipelineConfig
from ..data import load_dataset
from ..pipeline import HybridPipeline, ALOnlyPipeline, WSOnlyPipeline, PipelineResult


# =========================================================================
# CONFIGURATION
# =========================================================================

# Full-scale experiment defaults (override via CLI / function args)
FULL_DATASETS = ["customer_tickets", "banking77", "clinc150", "bitext_ecommerce", "cfpb_complaints"]
FULL_MAX_SAMPLES = 5000                # realistic dataset size
FULL_MAX_HUMAN = 300                   # full labeling budget
FULL_BATCH_SIZE = 10
FULL_N_SEEDS = 5                       # recommended for publication
FULL_SEEDS = [42, 142, 242, 342, 442]  # well-separated seeds

# Reference Config — accumulated by Phase A step-locking
# These are the smoke-test results; Phase A will re-confirm at full scale
DEFAULT_REFERENCE = {
    "classifier_type": "lr",
    "ws_accuracy_threshold": 0.6,
    "weak_cert_alpha": 0.9,
    "lf_confidence_threshold": 0.7,
    "ws_confidence_filter": 0.8,
    "encoder_type": "bm25",
    "use_knn_lf": False,                  # Removed: bistable coverage with bm25
    "query_strategy": "uncertainty_entropy",
    "label_model": "dawid_skene",
    "use_weak_cert": True,
    "use_weak_clust": False,
    "use_topic_lf": True,
}

# Output directory
OUTPUT_DIR = Path("experiment_results")


# =========================================================================
# HELPERS
# =========================================================================

def _save_incremental(all_rows: list[dict], output_dir: Path, label: str = "",
                      phase_a_rows: list[dict] | None = None) -> None:
    """Save results accumulated so far to CSV (called after each step).

    If phase_a_rows is provided, they are prepended to all_rows before saving
    so that Phase A data is never lost when Phase B incremental saves happen.
    """
    if not all_rows and not phase_a_rows:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = (phase_a_rows or []) + all_rows
    raw_path = output_dir / "raw_results.csv"
    pd.DataFrame(combined).to_csv(raw_path, index=False)
    suffix = f" ({label})" if label else ""
    print(f"  💾 Incremental save{suffix}: {len(combined)} rows → {raw_path}", flush=True)

def make_config(
    dataset: str = "customer_tickets",
    max_samples: int = FULL_MAX_SAMPLES,
    max_human: int = FULL_MAX_HUMAN,
    batch_size: int = FULL_BATCH_SIZE,
    seed: int = 42,
    **overrides,
) -> PipelineConfig:
    """Create a PipelineConfig with sensible full-experiment defaults."""
    defaults = dict(
        dataset_name=dataset,
        max_samples=max_samples,
        max_human_labels=max_human,
        batch_size=batch_size,
        initial_per_class=2,
        random_seed=seed,
        # Start from reference config
        **DEFAULT_REFERENCE,
    )
    # Override reference with explicit overrides
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _result_to_dict(result: PipelineResult, label: str, seed: int,
                     dataset: str, step: str) -> dict:
    """Flatten a PipelineResult into a flat dict for CSV export."""
    d = {
        "label": label,
        "seed": seed,
        "dataset": dataset,
        "step": step,
        "final_accuracy": result.final_accuracy,
        "final_f1_macro": result.final_f1_macro,
        "baseline_accuracy": result.baseline_accuracy,
        "baseline_f1_macro": result.baseline_f1_macro,
        "total_human_labels": result.total_human_labels,
        "total_ws_labels": result.total_ws_labels,
        "total_labels": result.total_labels,
        "ws_label_accuracy": result.ws_label_accuracy,
        "ws_contribution_pct": result.ws_contribution_pct,
        "human_savings_pct": result.human_savings_pct,
        "n_pool": result.n_pool,
        "n_test": result.n_test,
        "n_classes": result.n_classes,
        # Per-step learning curve as JSON string (for plotting)
        "history_json": json.dumps(result.history) if result.history else None,
    }
    # Add config details
    for k, v in asdict(result.config).items():
        if k not in d:
            d[f"cfg_{k}"] = v
    return d


def _aggregate(rows: list[dict], metric: str = "final_accuracy") -> dict:
    """Aggregate rows for the same label across seeds: mean, std, n."""
    if not rows:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    vals = [r[metric] for r in rows if metric in r and r[metric] is not None]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "values": vals,
    }


def _paired_ttest(group_a: list[float], group_b: list[float]) -> dict:
    """Paired t-test between two groups. Returns dict with p-value and significance."""
    from scipy import stats
    if len(group_a) < 2 or len(group_b) < 2 or len(group_a) != len(group_b):
        return {"p_value": None, "significant": None, "note": "insufficient data"}
    t_stat, p_value = stats.ttest_rel(group_a, group_b)
    return {
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "t_stat": float(t_stat),
    }


def _print_progress(label: str, seed: int, n_seeds: int,
                     acc: float, f1: float, ws: int, human: int,
                     elapsed: float):
    """Print a concise progress line."""
    print(f"  [{label}] seed={seed} ({seed}/{n_seeds} seeds) | "
          f"Acc={acc:.4f} F1={f1:.4f} | WS={ws} Human={human} | {elapsed:.1f}s",
          flush=True)


def _run_single(label: str, PipelineClass, cfg: PipelineConfig,
                dataset_name: str, step: str, seed: int, n_seeds: int,
                config_as_kwarg: bool = False,
                **pipe_kwargs) -> dict | None:
    """Run a single pipeline and return result dict, or None on failure.

    Args:
        config_as_kwarg: If True, pass config as keyword arg (config=cfg).
            Technique pipelines (T1-T14, Combo) use this style.
            Base pipelines (HybridPipeline, etc.) use positional style.
    """
    try:
        t0 = time.time()
        dataset = load_dataset(cfg)
        if config_as_kwarg:
            pipe = PipelineClass(config=cfg, **pipe_kwargs)
        else:
            pipe = PipelineClass(cfg)
        result = pipe.run(dataset)
        elapsed = time.time() - t0
        _print_progress(label, seed, n_seeds,
                        result.final_accuracy, result.final_f1_macro,
                        result.total_ws_labels, result.total_human_labels,
                        elapsed)
        return _result_to_dict(result, label, seed, dataset_name, step)
    except ImportError as e:
        print(f"  ⚠️  {label}/seed={seed}: SKIP (missing dep: {e})", flush=True)
        return None
    except Exception as e:
        print(f"  ❌ {label}/seed={seed}: {e}", flush=True)
        return None


# =========================================================================
# PHASE A: STEP-WISE ABLATION (with sequential step-locking)
# =========================================================================

def run_phase_a(
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    max_samples: int = FULL_MAX_SAMPLES,
    max_human: int = FULL_MAX_HUMAN,
    batch_size: int = FULL_BATCH_SIZE,
    steps: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Run Phase A: Step-wise ablation with sequential step-locking.

    Returns:
        (all_rows, locked_configs_per_dataset)
        - all_rows: flat list of result dicts for CSV
        - locked_configs_per_dataset: {dataset: {param: value}}
    """
    if datasets is None:
        datasets = FULL_DATASETS
    if seeds is None:
        seeds = FULL_SEEDS
    n_seeds = len(seeds)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)

    all_rows: list[dict] = []
    locked_configs: dict[str, dict[str, Any]] = {}

    # All available steps in execution order
    all_steps = ["1", "1b", "2", "4", "5", "3", "6"]
    if steps is None:
        steps = all_steps

    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"PHASE A — Step-wise Ablation: {dataset}")
        print(f"  Seeds: {seeds} | max_samples={max_samples} | max_human={max_human}")
        print(f"{'=' * 70}")

        # Initialize locked config for this dataset
        locked = {
            "classifier_type": "lr",
            "ws_accuracy_threshold": 0.6,
            "weak_cert_alpha": 0.9,
            "lf_confidence_threshold": 0.7,
            "ws_confidence_filter": 0.8,
            # To be locked:
            "encoder_type": None,
            "query_strategy": None,
            "label_model": None,
            "use_weak_cert": None,
            "use_weak_clust": None,
            "use_topic_lf": None,
        }
        locked_configs[dataset] = locked

        def cfg_for_step(**overrides) -> PipelineConfig:
            """Build config using currently locked values + overrides."""
            base = {k: v for k, v in locked.items() if v is not None}
            return make_config(
                dataset=dataset, max_samples=max_samples,
                max_human=max_human, batch_size=batch_size,
                seed=42,  # overridden per-seed below
                **base, **overrides,
            )

        # ── STEP 1: TEXT ENCODING ─────────────────────────────
        if "1" in steps:
            print(f"\n{'─' * 60}")
            print("STEP 1: Text Encoding (R1: does encoding matter? R2: which is best?)")
            print(f"{'─' * 60}")

            encoders = ["tfidf", "fasttext_dense"]
            optional = ["bm25", "fasttext_sparse", "dense", "hybrid"]
            step_results: dict[str, list[float]] = {}

            for enc in encoders + optional:
                print(f"\n  >> Encoding: {enc} ...", flush=True)
                # ── #5 OPTIMIZATION: Parallel seed runs via joblib ──
                def _run_s1_seed(seed):
                    cfg = cfg_for_step(encoder_type=enc, random_seed=seed)
                    return _run_single(f"S1/{enc}", HybridPipeline, cfg,
                                      dataset, "step1_encoding", seed, n_seeds)

                rows = Parallel(n_jobs=min(len(seeds), 4), backend="loky")(
                    delayed(_run_s1_seed)(seed) for seed in seeds
                )
                accs = []
                for row in rows:
                    if row is not None:
                        all_rows.append(row)
                        accs.append(row["final_accuracy"])

                if accs:
                    step_results[enc] = accs
                    agg = _aggregate(
                        [r for r in all_rows
                         if r["label"] == f"S1/{enc}" and r["step"] == "step1_encoding"],
                    )
                    print(f"    {enc}: {agg['mean']:.4f} ± {agg['std']:.4f} (n={agg['n']})", flush=True)

            # Lock best encoder
            if step_results:
                best_enc = max(step_results, key=lambda e: np.mean(step_results[e]))
                locked["encoder_type"] = best_enc
                print(f"\n  🔒 LOCKED encoder_type = {best_enc} "
                      f"(mean_acc={np.mean(step_results[best_enc]):.4f})")

                # Significance test: best vs default tfidf
                if best_enc != "tfidf" and "tfidf" in step_results:
                    sig = _paired_ttest(step_results[best_enc], step_results["tfidf"])
                    if sig['p_value'] is not None:
                        print(f"    vs tfidf: p={sig['p_value']:.4f} "
                              f"{'✅ significant' if sig.get('significant') else '(not significant)'}")
                    else:
                        print(f"    vs tfidf: p=N/A (need ≥2 seeds for t-test)")
            else:
                locked["encoder_type"] = "tfidf"
                print(f"\n  🔒 LOCKED encoder_type = tfidf (fallback)")

            _save_incremental(all_rows, output_dir, "step1_encoding")

        # ── STEP 1b: HYBRID ALPHA ─────────────────────────────
        if "1b" in steps and locked.get("encoder_type") == "hybrid":
            print(f"\n{'─' * 60}")
            print("STEP 1b: Hybrid Alpha Sensitivity")
            print(f"{'─' * 60}")

            alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            alpha_results: dict[str, list[float]] = {}

            for alpha in alphas:
                accs = []
                for seed in seeds:
                    cfg = cfg_for_step(hybrid_alpha=alpha, random_seed=seed)
                    row = _run_single(f"S1b/α={alpha}", HybridPipeline, cfg,
                                      dataset, "step1b_alpha", seed, n_seeds)
                    if row is not None:
                        all_rows.append(row)
                        accs.append(row["final_accuracy"])
                if accs:
                    alpha_results[f"alpha_{alpha}"] = accs

            if alpha_results:
                best_alpha_key = max(alpha_results, key=lambda k: np.mean(alpha_results[k]))
                best_alpha = float(best_alpha_key.replace("alpha_", ""))
                print(f"\n  Best hybrid_alpha = {best_alpha} "
                      f"(mean_acc={np.mean(alpha_results[best_alpha_key]):.4f})")
        elif "1b" in steps:
            print(f"\n  ⏭  Step 1b SKIP: encoder is '{locked.get('encoder_type')}', not 'hybrid'")

        # ── STEP 2: AL QUERY STRATEGY ─────────────────────────
        if "2" in steps:
            print(f"\n{'─' * 60}")
            print("STEP 2: AL Query Strategy (R1: AL vs Random? R2: which strategy?)")
            print(f"{'─' * 60}")

            strategies = [
                "random",
                "uncertainty_least_confident",
                "uncertainty_margin",
                "uncertainty_entropy",
            ]
            strat_results: dict[str, list[float]] = {}

            for strat in strategies:
                accs = []
                for seed in seeds:
                    cfg = cfg_for_step(query_strategy=strat, random_seed=seed)
                    row = _run_single(f"S2/{strat}", HybridPipeline, cfg,
                                      dataset, "step2_strategy", seed, n_seeds)
                    if row is not None:
                        all_rows.append(row)
                        accs.append(row["final_accuracy"])
                if accs:
                    strat_results[strat] = accs
                    agg = _aggregate(
                        [r for r in all_rows
                         if r["label"] == f"S2/{strat}" and r["step"] == "step2_strategy"],
                    )
                    print(f"    {strat}: {agg['mean']:.4f} ± {agg['std']:.4f}")

            # Lock best strategy
            if strat_results:
                best_strat = max(strat_results, key=lambda s: np.mean(strat_results[s]))
                locked["query_strategy"] = best_strat
                print(f"\n  🔒 LOCKED query_strategy = {best_strat} "
                      f"(mean_acc={np.mean(strat_results[best_strat]):.4f})")

                # Significance: best vs random
                if best_strat != "random" and "random" in strat_results:
                    sig = _paired_ttest(strat_results[best_strat], strat_results["random"])
                    if sig['p_value'] is not None:
                        print(f"    vs random: p={sig['p_value']:.4f} "
                              f"{'✅ significant' if sig.get('significant') else '(not significant)'}")
                    else:
                        print(f"    vs random: p=N/A (need ≥2 seeds for t-test)")
            else:
                locked["query_strategy"] = "uncertainty_entropy"
                print(f"\n  🔒 LOCKED query_strategy = uncertainty_entropy (fallback)")

            _save_incremental(all_rows, output_dir, "step2_strategy")

        # ── STEP 4: WS LABEL AGGREGATION ──────────────────────
        if "4" in steps:
            print(f"\n{'─' * 60}")
            print("STEP 4: WS Label Aggregation (R1: DS vs MV? R2: which aggregator?)")
            print(f"{'─' * 60}")

            agg_methods = ["majority_vote", "dawid_skene"]
            agg_results: dict[str, list[float]] = {}

            for method in agg_methods:
                accs = []
                for seed in seeds:
                    cfg = cfg_for_step(label_model=method, random_seed=seed)
                    row = _run_single(f"S4/{method}", HybridPipeline, cfg,
                                      dataset, "step4_aggregation", seed, n_seeds)
                    if row is not None:
                        all_rows.append(row)
                        accs.append(row["final_accuracy"])
                if accs:
                    agg_results[method] = accs
                    agg = _aggregate(
                        [r for r in all_rows
                         if r["label"] == f"S4/{method}" and r["step"] == "step4_aggregation"],
                    )
                    print(f"    {method}: {agg['mean']:.4f} ± {agg['std']:.4f}")

            # Lock best aggregation
            if agg_results:
                best_agg = max(agg_results, key=lambda m: np.mean(agg_results[m]))
                locked["label_model"] = best_agg
                print(f"\n  🔒 LOCKED label_model = {best_agg} "
                      f"(mean_acc={np.mean(agg_results[best_agg]):.4f})")
                if len(agg_methods) == 2:
                    sig = _paired_ttest(
                        agg_results.get("dawid_skene", []),
                        agg_results.get("majority_vote", []),
                    )
                    if sig["p_value"] is not None:
                        print(f"    DS vs MV: p={sig['p_value']:.4f} "
                              f"{'✅ significant' if sig.get('significant') else '(not significant)'}")
            else:
                locked["label_model"] = "dawid_skene"
                print(f"\n  🔒 LOCKED label_model = dawid_skene (fallback)")

            _save_incremental(all_rows, output_dir, "step4_aggregation")

        # ── STEP 5: AUTO-LABELING ─────────────────────────────
        if "5" in steps:
            print(f"\n{'─' * 60}")
            print("STEP 5: Auto-labeling (R1: does it help? R2: WeakCert vs WeakClust?)")
            print(f"{'─' * 60}")

            auto_configs = {
                "none":       dict(use_weak_cert=False, use_weak_clust=False),
                "weak_cert":  dict(use_weak_cert=True,  use_weak_clust=False),
                "weak_clust": dict(use_weak_cert=False, use_weak_clust=True),
                "both":       dict(use_weak_cert=True,  use_weak_clust=True),
            }
            auto_results: dict[str, list[float]] = {}

            for label, overrides in auto_configs.items():
                accs = []
                for seed in seeds:
                    cfg = cfg_for_step(**overrides, random_seed=seed)
                    row = _run_single(f"S5/{label}", HybridPipeline, cfg,
                                      dataset, "step5_autolabel", seed, n_seeds)
                    if row is not None:
                        all_rows.append(row)
                        accs.append(row["final_accuracy"])
                if accs:
                    auto_results[label] = accs
                    agg = _aggregate(
                        [r for r in all_rows
                         if r["label"] == f"S5/{label}" and r["step"] == "step5_autolabel"],
                    )
                    print(f"    {label}: {agg['mean']:.4f} ± {agg['std']:.4f}")

            # Lock best auto-labeling config
            if auto_results:
                best_auto = max(auto_results, key=lambda l: np.mean(auto_results[l]))
                best_overrides = auto_configs[best_auto]
                locked["use_weak_cert"] = best_overrides["use_weak_cert"]
                locked["use_weak_clust"] = best_overrides["use_weak_clust"]
                print(f"\n  🔒 LOCKED use_weak_cert={best_overrides['use_weak_cert']}, "
                      f"use_weak_clust={best_overrides['use_weak_clust']} "
                      f"(from '{best_auto}', mean_acc={np.mean(auto_results[best_auto]):.4f})")

                # Significance: best vs none
                if best_auto != "none" and "none" in auto_results:
                    sig = _paired_ttest(auto_results[best_auto], auto_results["none"])
                    if sig['p_value'] is not None:
                        print(f"    vs none: p={sig['p_value']:.4f} "
                              f"{'✅ significant' if sig.get('significant') else '(not significant)'}")
                    else:
                        print(f"    vs none: p=N/A (need ≥2 seeds for t-test)")
            else:
                locked["use_weak_cert"] = True
                locked["use_weak_clust"] = False
                print(f"\n  🔒 LOCKED use_weak_cert=True, use_weak_clust=False (fallback)")

            _save_incremental(all_rows, output_dir, "step5_autolabel")

        # ── STEP 3: WS LABELING FUNCTIONS ─────────────────────
        if "3" in steps:
            print(f"\n{'─' * 60}")
            print("STEP 3: WS Labeling Functions (R1: WS helps vs AL-only? R2: TopicLF?)")
            print(f"{'─' * 60}")

            lf_configs = {
                "al_only":          ("al_only", False),
                "hybrid_no_topic":  ("hybrid", False),
                "hybrid_with_topic": ("hybrid", True),
            }
            lf_results: dict[str, list[float]] = {}

            for label, (mode, use_topic) in lf_configs.items():
                accs = []
                for seed in seeds:
                    cfg = cfg_for_step(use_topic_lf=use_topic, random_seed=seed)
                    dataset_obj = load_dataset(cfg)
                    try:
                        t0 = time.time()
                        if mode == "al_only":
                            pipe = ALOnlyPipeline(cfg)
                        else:
                            pipe = HybridPipeline(cfg)
                        result = pipe.run(dataset_obj)
                        elapsed = time.time() - t0
                        _print_progress(f"S3/{label}", seed, n_seeds,
                                        result.final_accuracy, result.final_f1_macro,
                                        result.total_ws_labels, result.total_human_labels,
                                        elapsed)
                        row = _result_to_dict(result, f"S3/{label}", seed,
                                              dataset, "step3_ws_lfs")
                        all_rows.append(row)
                        accs.append(row["final_accuracy"])
                    except Exception as e:
                        print(f"  ❌ S3/{label}/seed={seed}: {e}")

                if accs:
                    lf_results[label] = accs
                    agg = _aggregate(
                        [r for r in all_rows
                         if r["label"] == f"S3/{label}" and r["step"] == "step3_ws_lfs"],
                    )
                    print(f"    {label}: {agg['mean']:.4f} ± {agg['std']:.4f}")

            # Lock TopicLF decision
            topic_on = lf_results.get("hybrid_with_topic", [])
            topic_off = lf_results.get("hybrid_no_topic", [])
            if topic_on and topic_off:
                if np.mean(topic_on) >= np.mean(topic_off):
                    locked["use_topic_lf"] = True
                    print(f"\n  🔒 LOCKED use_topic_lf = True "
                          f"(on: {np.mean(topic_on):.4f} >= off: {np.mean(topic_off):.4f})")
                else:
                    locked["use_topic_lf"] = False
                    print(f"\n  🔒 LOCKED use_topic_lf = False "
                          f"(off: {np.mean(topic_off):.4f} > on: {np.mean(topic_on):.4f})")

                sig = _paired_ttest(topic_on, topic_off)
                if sig["p_value"] is not None:
                    print(f"    with_topic vs without: p={sig['p_value']:.4f} "
                          f"{'✅ significant' if sig.get('significant') else '(not significant)'}")

                # R1 significance: hybrid vs al_only
                hybrid_accs = lf_results.get("hybrid_no_topic", lf_results.get("hybrid_with_topic", []))
                al_only_accs = lf_results.get("al_only", [])
                if hybrid_accs and al_only_accs:
                    sig_r1 = _paired_ttest(hybrid_accs, al_only_accs)
                    if sig_r1["p_value"] is not None:
                        print(f"    R1 (hybrid vs AL-only): p={sig_r1['p_value']:.4f} "
                              f"{'✅ significant' if sig_r1.get('significant') else '(not significant)'}")
            else:
                locked["use_topic_lf"] = False
                print(f"\n  🔒 LOCKED use_topic_lf = False (fallback)")

            _save_incremental(all_rows, output_dir, "step3_ws_lfs")

        # ── STEP 6: PIPELINE INTEGRATION ──────────────────────
        if "6" in steps:
            print(f"\n{'─' * 60}")
            print("STEP 6: Pipeline Integration (R1: Hybrid vs AL-only vs WS-only)")
            print(f"{'─' * 60}")

            modes = {
                "al_only": ALOnlyPipeline,
                "ws_only": WSOnlyPipeline,
                "hybrid": HybridPipeline,
            }
            mode_results: dict[str, list[float]] = {}

            for label, PipelineClass in modes.items():
                accs = []
                for seed in seeds:
                    cfg = cfg_for_step(random_seed=seed)
                    ds = load_dataset(cfg)
                    try:
                        t0 = time.time()
                        if PipelineClass == HybridPipeline:
                            pipe = PipelineClass(cfg)
                        else:
                            pipe = PipelineClass(cfg)
                        result = pipe.run(ds)
                        elapsed = time.time() - t0
                        _print_progress(f"S6/{label}", seed, n_seeds,
                                        result.final_accuracy, result.final_f1_macro,
                                        result.total_ws_labels, result.total_human_labels,
                                        elapsed)
                        row = _result_to_dict(result, f"S6/{label}", seed,
                                              dataset, "step6_integration")
                        all_rows.append(row)
                        accs.append(row["final_accuracy"])
                    except Exception as e:
                        print(f"  ❌ S6/{label}/seed={seed}: {e}")

                if accs:
                    mode_results[label] = accs
                    agg = _aggregate(
                        [r for r in all_rows
                         if r["label"] == f"S6/{label}" and r["step"] == "step6_integration"],
                    )
                    print(f"    {label}: {agg['mean']:.4f} ± {agg['std']:.4f}")

            # Significance: hybrid vs al_only
            hybrid_accs = mode_results.get("hybrid", [])
            al_only_accs = mode_results.get("al_only", [])
            if hybrid_accs and al_only_accs:
                sig = _paired_ttest(hybrid_accs, al_only_accs)
                if sig["p_value"] is not None:
                    print(f"    Hybrid vs AL-only: p={sig['p_value']:.4f} "
                          f"{'✅ significant' if sig.get('significant') else '(not significant)'}")

        # Print final locked config for this dataset
        print(f"\n{'=' * 70}")
        print(f"LOCKED REFERENCE CONFIG for {dataset}:")
        print(f"{'=' * 70}")
        for k, v in locked.items():
            status = "✅ locked" if v is not None else "❌ NOT LOCKED"
            print(f"  {k:30s} = {str(v):30s}  [{status}]")

        _save_incremental(all_rows, output_dir, f"phase_a_{dataset}")

    return all_rows, locked_configs


# =========================================================================
# PHASE B: TECHNIQUE EVALUATION (T1–T14 + Combo)
# =========================================================================

def run_phase_b(
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    max_samples: int = FULL_MAX_SAMPLES,
    max_human: int = FULL_MAX_HUMAN,
    batch_size: int = FULL_BATCH_SIZE,
    locked_configs: dict[str, dict] | None = None,
    techniques: list[str] | None = None,
    output_dir: str | Path | None = None,
    phase_a_rows: list[dict] | None = None,
) -> list[dict]:
    """
    Run Phase B: Evaluate all techniques against the Reference Config.

    Args:
        locked_configs: Output from Phase A. If None, uses DEFAULT_REFERENCE.
        techniques: Subset of techniques to run. None = all.
        phase_a_rows: Phase A rows to preserve in incremental saves.

    Returns:
        Flat list of result dicts for CSV.
    """
    if datasets is None:
        datasets = FULL_DATASETS
    if seeds is None:
        seeds = FULL_SEEDS
    n_seeds = len(seeds)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)

    all_rows: list[dict] = []

    # Import technique pipelines
    from ..pipeline.enhanced_hybrid import (
        T1WeightedPipeline,
        T2VerificationPipeline,
        T3CalibratedPipeline,
        T4SelfTrainingPipeline,
        T5UnanimousPipeline,
        T6PerClassThresholdPipeline,
        T7IsotonicPipeline,
        T8BERTLFPipeline,
        T9BADGEPipeline,
        T10LabelPropagationPipeline,
        T11CostSensitivePipeline,
        T12AdaptiveBudgetPipeline,
        T13FlyingSquidPipeline,
        T14CalibratedPseudoPipeline,
        ComboPipeline,
    )

    all_techniques = [
        ("T1_weighted",          T1WeightedPipeline,       {}),
        ("T2_verification",      T2VerificationPipeline,   {}),
        ("T3_calibrated",        T3CalibratedPipeline,     {}),
        ("T4_self_training",     T4SelfTrainingPipeline,   {}),
        ("T5_unanimous",         T5UnanimousPipeline,      {}),
        ("T6_per_class_thresh",  T6PerClassThresholdPipeline, {}),
        ("T7_isotonic",          T7IsotonicPipeline,       {}),
        ("T8_BERT_LF",           T8BERTLFPipeline,         {}),
        ("T9_BADGE",             T9BADGEPipeline,          {}),
        ("T10_label_propagation", T10LabelPropagationPipeline, {}),
        ("T11_cost_sensitive",   T11CostSensitivePipeline, {}),
        ("T12_adaptive_budget",  T12AdaptiveBudgetPipeline, {}),
        ("T13_FlyingSquid",      T13FlyingSquidPipeline,   {}),
        ("T14_calibrated_pseudo", T14CalibratedPseudoPipeline, {}),
        ("Combo_T1T2T3T6",       ComboPipeline,            {}),
    ]

    # Filter techniques if requested
    if techniques is not None:
        all_techniques = [(l, p, k) for l, p, k in all_techniques if l in techniques]

    for dataset in datasets:
        # Get locked config for this dataset
        if locked_configs and dataset in locked_configs:
            ref = {k: v for k, v in locked_configs[dataset].items() if v is not None}
        else:
            ref = {k: v for k, v in DEFAULT_REFERENCE.items() if v is not None}

        print(f"\n{'=' * 70}")
        print(f"PHASE B — Technique Evaluation: {dataset}")
        print(f"  Reference Config:")
        for k, v in ref.items():
            print(f"    {k} = {v}")
        print(f"  Seeds: {seeds} | max_samples={max_samples} | max_human={max_human}")
        print(f"  Techniques: {len(all_techniques)}")
        print(f"{'=' * 70}")

        # ── Baseline: Reference Config (HybridPipeline, no technique) ──
        print(f"\n  ── Reference Baseline ──")
        ref_accs: list[float] = []

        # ── #5 OPTIMIZATION: Parallel seed runs for reference baseline ──
        def _run_ref_seed(seed):
            cfg = make_config(
                dataset=dataset, max_samples=max_samples,
                max_human=max_human, batch_size=batch_size,
                seed=seed, **ref,
            )
            return _run_single("Reference", HybridPipeline, cfg,
                              dataset, "technique_reference", seed, n_seeds)

        ref_rows = Parallel(n_jobs=min(len(seeds), 4), backend="loky")(
            delayed(_run_ref_seed)(seed) for seed in seeds
        )
        for row in ref_rows:
            if row is not None:
                all_rows.append(row)
                ref_accs.append(row["final_accuracy"])

        if ref_accs:
            agg = _aggregate(
                [r for r in all_rows
                 if r["label"] == "Reference" and r["step"] == "technique_reference" and r["dataset"] == dataset],
            )
            print(f"    Reference: {agg['mean']:.4f} ± {agg['std']:.4f}")

        # ── Each technique ──
        technique_results: dict[str, list[float]] = {"Reference": ref_accs}

        for label, PipelineClass, pipe_kwargs in all_techniques:
            print(f"\n  ── {label} ──")
            tech_accs: list[float] = []

            # ── #5 OPTIMIZATION: Parallel seed runs per technique ──
            def _run_tech_seed(seed, _label=label, _PipelineClass=PipelineClass, _pipe_kwargs=pipe_kwargs):
                cfg = make_config(
                    dataset=dataset, max_samples=max_samples,
                    max_human=max_human, batch_size=batch_size,
                    seed=seed, **ref,
                )
                return _run_single(_label, _PipelineClass, cfg,
                                  dataset, f"technique_{_label}", seed, n_seeds,
                                  config_as_kwarg=True,
                                  **_pipe_kwargs)

            tech_rows = Parallel(n_jobs=min(len(seeds), 4), backend="loky")(
                delayed(_run_tech_seed)(seed) for seed in seeds
            )
            for row in tech_rows:
                if row is not None:
                    all_rows.append(row)
                    tech_accs.append(row["final_accuracy"])

            if tech_accs:
                technique_results[label] = tech_accs
                agg = _aggregate(
                    [r for r in all_rows
                     if r["label"] == label and r["step"] == f"technique_{label}" and r["dataset"] == dataset],
                )
                delta = agg["mean"] - (np.mean(ref_accs) if ref_accs else 0)
                sig_marker = ""
                if ref_accs and len(tech_accs) == len(ref_accs) and len(tech_accs) >= 2:
                    sig = _paired_ttest(tech_accs, ref_accs)
                    if sig["p_value"] is not None:
                        sig_marker = " ✅" if sig["significant"] else ""
                        print(f"    {label}: {agg['mean']:.4f} ± {agg['std']:.4f} "
                              f"(Δ={delta:+.4f}, p={sig['p_value']:.4f}{sig_marker})")
                    else:
                        print(f"    {label}: {agg['mean']:.4f} ± {agg['std']:.4f} (Δ={delta:+.4f})")
                else:
                    print(f"    {label}: {agg['mean']:.4f} ± {agg['std']:.4f} (Δ={delta:+.4f})")

            _save_incremental(all_rows, output_dir, f"technique_{label}", phase_a_rows=phase_a_rows)

        # ── Technique summary table ──
        print(f"\n{'─' * 70}")
        print(f"TECHNIQUE SUMMARY — {dataset}")
        print(f"{'─' * 70}")
        print(f"  {'Technique':25s} {'Acc':>8s} {'±':>4s} {'F1':>8s} {'Δ Acc':>8s} {'p-value':>10s}")
        print(f"  {'─' * 25} {'─' * 8} {'─' * 4} {'─' * 8} {'─' * 8} {'─' * 10}")

        ref_mean = np.mean(ref_accs) if ref_accs else 0
        for label, tech_accs in technique_results.items():
            agg = _aggregate(
                [r for r in all_rows
                 if r["label"] == label and r["dataset"] == dataset
                 and r["step"].startswith("technique_")],
                metric="final_accuracy",
            )
            f1_agg = _aggregate(
                [r for r in all_rows
                 if r["label"] == label and r["dataset"] == dataset
                 and r["step"].startswith("technique_")],
                metric="final_f1_macro",
            )
            delta = agg["mean"] - ref_mean
            p_str = "—"
            if label != "Reference" and ref_accs and len(tech_accs) == len(ref_accs) and len(tech_accs) >= 2:
                sig = _paired_ttest(tech_accs, ref_accs)
                if sig["p_value"] is not None:
                    p_str = f"{sig['p_value']:.4f}"
                    if sig["significant"]:
                        p_str += " *"

            print(f"  {label:25s} {agg['mean']:8.4f} ±{agg['std']:4.4f} "
                  f"{f1_agg['mean']:8.4f} {delta:+8.4f} {p_str:>10s}")

        _save_incremental(all_rows, output_dir, f"phase_b_{dataset}", phase_a_rows=phase_a_rows)

    return all_rows


# =========================================================================
# FULL RUNNER
# =========================================================================

def run_full_experiments(
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    max_samples: int = FULL_MAX_SAMPLES,
    max_human: int = FULL_MAX_HUMAN,
    batch_size: int = FULL_BATCH_SIZE,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Run the complete thesis experiment suite: Phase A + Phase B.

    Returns:
        dict with keys:
          - "raw_rows": list of flat result dicts
          - "locked_configs": per-dataset locked config from Phase A
          - "summary_df": pandas DataFrame with aggregated results
          - "output_dir": path where results were saved
    """
    if datasets is None:
        datasets = FULL_DATASETS
    if seeds is None:
        seeds = FULL_SEEDS
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║  FULL THESIS EXPERIMENT SUITE                                  ║")
    print(f"║  Datasets: {str(datasets):55s}║")
    print(f"║  Seeds: {str(seeds):58s}║")
    print(f"║  Scale: max_samples={max_samples}, max_human={max_human}, batch={batch_size}  ║")
    print("╚" + "═" * 68 + "╝")

    # ── PHASE A ──
    print("\n\n" + "🔵" * 35)
    print("PHASE A: Step-wise Ablation")
    print("🔵" * 35)

    phase_a_rows, locked_configs = run_phase_a(
        datasets=datasets, seeds=seeds,
        max_samples=max_samples, max_human=max_human,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    # ── PHASE B ──
    print("\n\n" + "🟢" * 35)
    print("PHASE B: Technique Evaluation")
    print("🟢" * 35)

    phase_b_rows = run_phase_b(
        datasets=datasets, seeds=seeds,
        max_samples=max_samples, max_human=max_human,
        batch_size=batch_size,
        locked_configs=locked_configs,
        output_dir=output_dir,
        phase_a_rows=phase_a_rows,
    )

    # ── COMBINE & SAVE ──
    all_rows = phase_a_rows + phase_b_rows

    total_elapsed = time.time() - total_start

    # Save raw CSV
    raw_df = pd.DataFrame(all_rows)
    raw_path = output_dir / "raw_results.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n  📊 Raw results saved to: {raw_path}")

    # Build summary CSV (mean ± std per label/dataset)
    summary_rows = []
    for (dataset, step, label), group in raw_df.groupby(["dataset", "step", "label"]):
        for metric in ["final_accuracy", "final_f1_macro", "total_ws_labels",
                        "total_human_labels", "ws_label_accuracy", "ws_contribution_pct"]:
            vals = group[metric].dropna().values
            if len(vals) > 0:
                summary_rows.append({
                    "dataset": dataset,
                    "step": step,
                    "label": label,
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  📊 Summary saved to: {summary_path}")

    # Save locked configs as JSON
    config_path = output_dir / "locked_configs.json"
    with open(config_path, "w") as f:
        json.dump(locked_configs, f, indent=2, default=str)
    print(f"  📊 Locked configs saved to: {config_path}")

    # Save per-step learning curves as separate JSON (for easy plotting)
    history_rows = []
    for row in all_rows:
        if row.get("history_json"):
            try:
                history = json.loads(row["history_json"])
                for h in history:
                    history_rows.append({
                        "label": row["label"],
                        "seed": row["seed"],
                        "dataset": row["dataset"],
                        "step_name": row["step"],
                        **h,
                    })
            except (json.JSONDecodeError, TypeError):
                pass
    if history_rows:
        history_path = output_dir / "learning_curves.json"
        with open(history_path, "w") as f:
            json.dump(history_rows, f, indent=2, default=str)
        print(f"  📊 Learning curves saved to: {history_path} ({len(history_rows)} data points)")

    # ── FINAL SUMMARY ──
    print(f"\n\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    print(f"  Total rows: {len(all_rows)}")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds: {seeds}")
    print(f"  Results in: {output_dir}")

    # Print locked config per dataset
    for ds, cfg in locked_configs.items():
        print(f"\n  {ds} — Locked Reference Config:")
        for k, v in cfg.items():
            print(f"    {k} = {v}")

    return {
        "raw_rows": all_rows,
        "locked_configs": locked_configs,
        "summary_df": summary_df,
        "raw_df": raw_df,
        "output_dir": str(output_dir),
    }


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full Thesis Experiment Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--phase", type=str, choices=["A", "B", "both"],
                        default="both", help="Which phase to run (default: both)")
    parser.add_argument("--step", type=str, default=None,
                        help="Run a single step: 1, 1b, 2, 3, 4, 5, 6 (Phase A only)")
    parser.add_argument("--dataset", type=str, nargs="+",
                        default=FULL_DATASETS,
                        help=f"Dataset(s) to use (default: {FULL_DATASETS})")
    parser.add_argument("--seeds", type=int, default=FULL_N_SEEDS,
                        help=f"Number of seeds (default: {FULL_N_SEEDS})")
    parser.add_argument("--max-samples", type=int, default=FULL_MAX_SAMPLES,
                        help=f"Max samples per dataset (default: {FULL_MAX_SAMPLES})")
    parser.add_argument("--max-human", type=int, default=FULL_MAX_HUMAN,
                        help=f"Max human labels budget (default: {FULL_MAX_HUMAN})")
    parser.add_argument("--batch-size", type=int, default=FULL_BATCH_SIZE,
                        help=f"AL batch size (default: {FULL_BATCH_SIZE})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--technique", type=str, nargs="+", default=None,
                        help="Run specific techniques only (Phase B): T1_weighted T7_isotonic ...")

    args = parser.parse_args()

    # Ensure stdout is unbuffered in Colab/notebook environments
    # so progress appears in real-time instead of being swallowed
    sys.stdout.reconfigure(line_buffering=True)

    # Generate seed list
    seeds = [42 + i * 100 for i in range(args.seeds)]

    # Determine steps for Phase A
    steps = [args.step] if args.step else None

    if args.phase == "both":
        run_full_experiments(
            datasets=args.dataset,
            seeds=seeds,
            max_samples=args.max_samples,
            max_human=args.max_human,
            batch_size=args.batch_size,
            output_dir=args.output,
        )
    elif args.phase == "A":
        run_phase_a(
            datasets=args.dataset,
            seeds=seeds,
            max_samples=args.max_samples,
            max_human=args.max_human,
            batch_size=args.batch_size,
            steps=steps,
            output_dir=args.output,
        )
    elif args.phase == "B":
        run_phase_b(
            datasets=args.dataset,
            seeds=seeds,
            max_samples=args.max_samples,
            max_human=args.max_human,
            batch_size=args.batch_size,
            techniques=args.technique,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
