"""
Smoke Tests for Complete Thesis Experiment Suite.

Validates that EVERY pipeline step and technique runs end-to-end without
crashing, producing non-degenerate results (not NaN, not 0%, not crashes).

Uses a SEQUENTIAL STEP-LOCKING strategy:
  - Step 1 (Encoding) runs first → best encoder is LOCKED
  - Step 2 (AL Strategy) runs with locked encoder → best strategy LOCKED
  - Step 4 (Aggregation) runs with locked enc+strat → best agg LOCKED
  - Step 5 (Auto-labeling) runs with locked base → best auto-label LOCKED
  - Step 3 (WS LFs) runs with locked base → TopicLF decision LOCKED
  - Step 6 (Integration) runs with full locked base
  - Techniques T1-T14 all run against the full locked Reference Config

Each test uses:
  - 1 dataset (customer_tickets — fast, 4 classes)
  - max_samples=200 (tiny)
  - max_human_labels=30 (minimal budget)
  - batch_size=5
  - 1 seed (no multi-seed)
  - ~10-30 seconds per test

Run:
    python -m Weak_AL_Pipeline.experiments.smoke_tests
    python -m Weak_AL_Pipeline.experiments.smoke_tests --step 1
    python -m Weak_AL_Pipeline.experiments.smoke_tests --step techniques
    python -m Weak_AL_Pipeline.experiments.smoke_tests --all
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..data import load_dataset
from ..pipeline import HybridPipeline, ALOnlyPipeline, WSOnlyPipeline, PipelineResult


# =========================================================================
# REFERENCE CONFIG — sequential step-locking
# =========================================================================
# This dict accumulates the "best so far" as each step completes.
# Each subsequent step uses all previously locked values.

LOCKED_CONFIG = {
    # ── Already decided (not tested) ──
    "classifier_type": "lr",
    "ws_accuracy_threshold": 0.6,
    "weak_cert_alpha": 0.9,
    "lf_confidence_threshold": 0.7,
    # ── To be locked by Step 1 ──
    "encoder_type": None,
    # ── To be locked by Step 2 ──
    "query_strategy": None,
    # ── To be locked by Step 4 ──
    "label_model": None,
    # ── To be locked by Step 5 ──
    "use_weak_cert": None,
    "use_weak_clust": None,
    # ── To be locked by Step 3 ──
    "use_topic_lf": None,
}


# =========================================================================
# SMOKE CONFIG — intentionally tiny
# =========================================================================

SMOKE_DATASET = "customer_tickets"   # 4 classes, fast
SMOKE_MAX_SAMPLES = 200
SMOKE_MAX_HUMAN = 30
SMOKE_BATCH = 5
SMOKE_SEED = 42


def smoke_cfg(**overrides) -> PipelineConfig:
    """Create a tiny config for smoke testing.

    Uses LOCKED_CONFIG values for all steps that have been decided so far.
    Any override takes precedence (used to test a variation for the current step).
    """
    defaults = dict(
        dataset_name=SMOKE_DATASET,
        max_samples=SMOKE_MAX_SAMPLES,
        max_human_labels=SMOKE_MAX_HUMAN,
        batch_size=SMOKE_BATCH,
        initial_per_class=2,
        random_seed=SMOKE_SEED,
        # Start with reasonable defaults for unlocked steps
        encoder_type="fasttext_dense",
        classifier_type="lr",
        query_strategy="uncertainty_entropy",
        label_model="dawid_skene",
        use_weak_cert=True,
        use_weak_clust=False,
        use_topic_lf=False,
    )
    # Apply locked values (override defaults for already-decided steps)
    for key, val in LOCKED_CONFIG.items():
        if val is not None:
            defaults[key] = val
    # Apply test-specific overrides (highest priority)
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _check_result(name: str, result: PipelineResult | None, min_acc: float = 0.05) -> bool:
    """Validate a result is non-degenerate."""
    if result is None:
        print(f"  ❌ {name}: result is None (crash)")
        return False

    acc = result.final_accuracy
    f1 = result.final_f1_macro

    if np.isnan(acc) or np.isnan(f1):
        print(f"  ❌ {name}: NaN detected (acc={acc}, f1={f1})")
        return False

    if acc < min_acc:
        print(f"  ❌ {name}: accuracy too low ({acc:.4f} < {min_acc})")
        return False

    if acc > 1.0 or f1 > 1.0:
        print(f"  ❌ {name}: metric > 1.0 (acc={acc}, f1={f1})")
        return False

    print(f"  ✅ {name}: acc={acc:.4f}, f1={f1:.4f}, ws={result.total_ws_labels}, "
          f"human={result.total_human_labels}")
    return True


def _pick_best(results: dict[str, PipelineResult], metric: str = "final_accuracy") -> str:
    """Pick the best config name from a dict of results."""
    best_name = None
    best_val = -1.0
    for name, result in results.items():
        if result is None:
            continue
        val = getattr(result, metric, 0.0)
        if val > best_val:
            best_val = val
            best_name = name
    return best_name


# =========================================================================
# STEP 1: TEXT ENCODING — LOCKS encoder_type
# =========================================================================

def smoke_step1_encoding() -> dict[str, bool]:
    """R1: Encoder choice matters. R2: Which encoder is best?

    Tests encoders with default config (nothing locked yet).
    LOCKS: best encoder_type into LOCKED_CONFIG.
    """
    print("\n" + "=" * 60)
    print("STEP 1: TEXT ENCODING (R1: presence, R2: technique)")
    print("  Config so far: classifier=lr (locked)")
    print("  Testing: tfidf, fasttext_dense, [+ bm25, dense, hybrid if deps]")
    print("=" * 60)

    results = {}
    best_results = {}  # name → PipelineResult for picking best

    # Core encoders (always available)
    encoders = ["tfidf", "fasttext_dense"]
    # Optional encoders (need deps)
    optional_encoders = ["bm25", "fasttext_sparse", "dense", "hybrid"]

    for enc in encoders + optional_encoders:
        cfg = smoke_cfg(encoder_type=enc)
        try:
            dataset = load_dataset(cfg)
            pipe = HybridPipeline(cfg)
            result = pipe.run(dataset)
            results[f"encoder_{enc}"] = _check_result(f"Step1/{enc}", result)
            if result is not None:
                best_results[enc] = result
        except ImportError as e:
            print(f"  ⚠️  Step1/{enc}: SKIP (missing dependency: {e})")
            results[f"encoder_{enc}"] = None
        except Exception as e:
            print(f"  ❌ Step1/{enc}: {e}")
            results[f"encoder_{enc}"] = False

    # Lock best encoder
    if best_results:
        best_enc = _pick_best(best_results)
        LOCKED_CONFIG["encoder_type"] = best_enc
        print(f"\n  🔒 LOCKED encoder_type = {best_enc} (acc={best_results[best_enc].final_accuracy:.4f})")
    else:
        # Fallback
        LOCKED_CONFIG["encoder_type"] = "fasttext_dense"
        print(f"\n  🔒 LOCKED encoder_type = fasttext_dense (fallback, no results)")

    return results


# =========================================================================
# STEP 1b: HYBRID ALPHA SENSITIVITY
# =========================================================================

def smoke_step1b_alpha() -> dict[str, bool]:
    """Alpha sensitivity for hybrid encoder.

    Only runs if encoder_type was locked to "hybrid" in Step 1.
    Otherwise, skips with a note.
    """
    print("\n" + "=" * 60)
    print("STEP 1b: HYBRID ALPHA SENSITIVITY")
    if LOCKED_CONFIG["encoder_type"] != "hybrid":
        print(f"  ⏭  SKIP: locked encoder is '{LOCKED_CONFIG['encoder_type']}', not 'hybrid'")
        return {"alpha_skip": None}
    print(f"  Config: encoder=hybrid (locked), classifier=lr (locked)")
    print("=" * 60)

    results = {}
    best_results = {}

    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        cfg = smoke_cfg(encoder_type="hybrid", hybrid_alpha=alpha)
        try:
            dataset = load_dataset(cfg)
            pipe = HybridPipeline(cfg)
            result = pipe.run(dataset)
            results[f"alpha_{alpha}"] = _check_result(f"Step1b/alpha={alpha}", result)
            if result is not None:
                best_results[f"alpha_{alpha}"] = result
        except Exception as e:
            print(f"  ❌ Step1b/alpha={alpha}: {e}")
            results[f"alpha_{alpha}"] = False

    return results


# =========================================================================
# STEP 2: AL QUERY STRATEGY — LOCKS query_strategy
# =========================================================================

def smoke_step2_al_strategy() -> dict[str, bool]:
    """R1: AL vs Random. R2: LC vs Margin vs Entropy.

    Uses locked encoder_type from Step 1.
    LOCKS: best query_strategy into LOCKED_CONFIG.
    """
    print("\n" + "=" * 60)
    print("STEP 2: AL QUERY STRATEGY (R1: presence, R2: technique)")
    print(f"  Config: encoder={LOCKED_CONFIG['encoder_type']} (locked), classifier=lr (locked)")
    print("=" * 60)

    results = {}
    best_results = {}

    strategies = [
        "random",
        "uncertainty_least_confident",
        "uncertainty_margin",
        "uncertainty_entropy",
    ]

    for strategy in strategies:
        cfg = smoke_cfg(query_strategy=strategy)
        try:
            dataset = load_dataset(cfg)
            pipe = HybridPipeline(cfg)
            result = pipe.run(dataset)
            label = strategy.replace("uncertainty_", "")
            results[f"strategy_{label}"] = _check_result(f"Step2/{label}", result)
            if result is not None:
                best_results[strategy] = result
        except Exception as e:
            print(f"  ❌ Step2/{strategy}: {e}")
            results[f"strategy_{label}"] = False

    # Lock best strategy
    if best_results:
        best_strat = _pick_best(best_results)
        LOCKED_CONFIG["query_strategy"] = best_strat
        print(f"\n  🔒 LOCKED query_strategy = {best_strat} (acc={best_results[best_strat].final_accuracy:.4f})")
    else:
        LOCKED_CONFIG["query_strategy"] = "uncertainty_entropy"
        print(f"\n  🔒 LOCKED query_strategy = uncertainty_entropy (fallback)")

    return results


# =========================================================================
# STEP 3: WS LABELING FUNCTIONS — LOCKS use_topic_lf
# =========================================================================

def smoke_step3_ws_lfs() -> dict[str, bool]:
    """R1: WS helps vs AL-only. R2: TopicLF adds value.

    Uses locked encoder + strategy from Steps 1-2.
    LOCKS: use_topic_lf into LOCKED_CONFIG.
    """
    print("\n" + "=" * 60)
    print("STEP 3: WS LABELING FUNCTIONS (R1: presence, R2: technique)")
    print(f"  Config: encoder={LOCKED_CONFIG['encoder_type']}, "
          f"strategy={LOCKED_CONFIG['query_strategy']} (locked)")
    print("=" * 60)

    results = {}

    configs = {
        "al_only": None,           # ALOnlyPipeline (R1 baseline)
        "hybrid_no_topic": False,  # HybridPipeline, use_topic_lf=False
        "hybrid_with_topic": True, # HybridPipeline, use_topic_lf=True
    }

    best_results = {}

    for label, use_topic in configs.items():
        try:
            if use_topic is None:
                cfg = smoke_cfg()
                dataset = load_dataset(cfg)
                pipe = ALOnlyPipeline(cfg)
            else:
                cfg = smoke_cfg(use_topic_lf=use_topic, topic_n_topics=4)
                dataset = load_dataset(cfg)
                pipe = HybridPipeline(cfg)

            result = pipe.run(dataset)
            results[label] = _check_result(f"Step3/{label}", result)
            if result is not None:
                best_results[label] = result
        except Exception as e:
            print(f"  ❌ Step3/{label}: {e}")
            results[label] = False

    # Lock TopicLF decision: compare hybrid_with_topic vs hybrid_no_topic
    topic_on = best_results.get("hybrid_with_topic")
    topic_off = best_results.get("hybrid_no_topic")
    if topic_on and topic_off:
        if topic_on.final_accuracy >= topic_off.final_accuracy:
            LOCKED_CONFIG["use_topic_lf"] = True
            print(f"\n  🔒 LOCKED use_topic_lf = True "
                  f"(topic_on: {topic_on.final_accuracy:.4f} >= topic_off: {topic_off.final_accuracy:.4f})")
        else:
            LOCKED_CONFIG["use_topic_lf"] = False
            print(f"\n  🔒 LOCKED use_topic_lf = False "
                  f"(topic_off: {topic_off.final_accuracy:.4f} > topic_on: {topic_on.final_accuracy:.4f})")
    else:
        LOCKED_CONFIG["use_topic_lf"] = False
        print(f"\n  🔒 LOCKED use_topic_lf = False (fallback)")

    return results


# =========================================================================
# STEP 4: WS LABEL AGGREGATION — LOCKS label_model
# =========================================================================

def smoke_step4_aggregation() -> dict[str, bool]:
    """R1: Dawid-Skene vs Majority Vote. R2: Which is better?

    Uses locked encoder + strategy + topic_lf from Steps 1-3.
    LOCKS: best label_model into LOCKED_CONFIG.
    """
    print("\n" + "=" * 60)
    print("STEP 4: WS LABEL AGGREGATION (R1: presence, R2: technique)")
    print(f"  Config: encoder={LOCKED_CONFIG['encoder_type']}, "
          f"strategy={LOCKED_CONFIG['query_strategy']}, "
          f"topic_lf={LOCKED_CONFIG['use_topic_lf']} (locked)")
    print("=" * 60)

    results = {}
    best_results = {}

    for method in ["majority_vote", "dawid_skene"]:
        cfg = smoke_cfg(label_model=method)
        try:
            dataset = load_dataset(cfg)
            pipe = HybridPipeline(cfg)
            result = pipe.run(dataset)
            results[f"agg_{method}"] = _check_result(f"Step4/{method}", result)
            if result is not None:
                best_results[method] = result
        except Exception as e:
            print(f"  ❌ Step4/{method}: {e}")
            results[f"agg_{method}"] = False

    # Lock best aggregation
    if best_results:
        best_agg = _pick_best(best_results)
        LOCKED_CONFIG["label_model"] = best_agg
        print(f"\n  🔒 LOCKED label_model = {best_agg} (acc={best_results[best_agg].final_accuracy:.4f})")
    else:
        LOCKED_CONFIG["label_model"] = "dawid_skene"
        print(f"\n  🔒 LOCKED label_model = dawid_skene (fallback)")

    return results


# =========================================================================
# STEP 5: AUTO-LABELING — LOCKS use_weak_cert, use_weak_clust
# =========================================================================

def smoke_step5_auto_labeling() -> dict[str, bool]:
    """R1: Auto-labeling helps. R2: WeakCert vs WeakClust vs both.

    Uses locked config from Steps 1-4.
    LOCKS: best use_weak_cert + use_weak_clust into LOCKED_CONFIG.
    """
    print("\n" + "=" * 60)
    print("STEP 5: AUTO-LABELING (R1: presence, R2: technique)")
    print(f"  Config: encoder={LOCKED_CONFIG['encoder_type']}, "
          f"strategy={LOCKED_CONFIG['query_strategy']}, "
          f"topic_lf={LOCKED_CONFIG['use_topic_lf']}, "
          f"agg={LOCKED_CONFIG['label_model']} (locked)")
    print("=" * 60)

    results = {}
    best_results = {}

    configs = {
        "none":       dict(use_weak_cert=False, use_weak_clust=False),
        "weak_cert":  dict(use_weak_cert=True,  use_weak_clust=False),
        "weak_clust": dict(use_weak_cert=False, use_weak_clust=True),
        "both":       dict(use_weak_cert=True,  use_weak_clust=True),
    }

    for label, overrides in configs.items():
        cfg = smoke_cfg(**overrides)
        try:
            dataset = load_dataset(cfg)
            pipe = HybridPipeline(cfg)
            result = pipe.run(dataset)
            results[label] = _check_result(f"Step5/{label}", result)
            if result is not None:
                best_results[label] = result
        except Exception as e:
            print(f"  ❌ Step5/{label}: {e}")
            results[label] = False

    # Lock best auto-labeling config
    if best_results:
        best_label = _pick_best(best_results)
        best_overrides = configs[best_label]
        LOCKED_CONFIG["use_weak_cert"] = best_overrides["use_weak_cert"]
        LOCKED_CONFIG["use_weak_clust"] = best_overrides["use_weak_clust"]
        print(f"\n  🔒 LOCKED use_weak_cert={best_overrides['use_weak_cert']}, "
              f"use_weak_clust={best_overrides['use_weak_clust']} "
              f"(from '{best_label}', acc={best_results[best_label].final_accuracy:.4f})")
    else:
        LOCKED_CONFIG["use_weak_cert"] = True
        LOCKED_CONFIG["use_weak_clust"] = False
        print(f"\n  🔒 LOCKED use_weak_cert=True, use_weak_clust=False (fallback)")

    return results


# =========================================================================
# STEP 6: PIPELINE INTEGRATION — validates full locked config
# =========================================================================

def smoke_step6_integration() -> dict[str, bool]:
    """R1: Hybrid vs AL-only vs WS-only. R2: Integration strategy.

    Uses FULL locked config from Steps 1-5.
    Does NOT lock anything — this is a validation step.
    """
    print("\n" + "=" * 60)
    print("STEP 6: PIPELINE INTEGRATION (R1: presence, R2: technique)")
    print(f"  FULL REFERENCE CONFIG:")
    for k, v in LOCKED_CONFIG.items():
        if v is not None:
            print(f"    {k} = {v}")
    print("=" * 60)

    results = {}

    modes = {
        "al_only": ALOnlyPipeline,
        "ws_only": WSOnlyPipeline,
        "hybrid": HybridPipeline,
    }

    for label, PipelineClass in modes.items():
        cfg = smoke_cfg()
        try:
            dataset = load_dataset(cfg)
            pipe = PipelineClass(cfg)
            result = pipe.run(dataset)
            results[label] = _check_result(f"Step6/{label}", result)
        except Exception as e:
            print(f"  ❌ Step6/{label}: {e}")
            results[label] = False

    return results


# =========================================================================
# TECHNIQUES T1–T14 + COMBO — tested against full Reference Config
# =========================================================================

def smoke_techniques() -> dict[str, bool]:
    """Smoke test all 14 techniques + Combo against the Reference Config."""
    print("\n" + "=" * 60)
    print("TECHNIQUES T1–T14 + COMBO (against Reference Config)")
    print(f"  REFERENCE CONFIG:")
    for k, v in LOCKED_CONFIG.items():
        if v is not None:
            print(f"    {k} = {v}")
    print("=" * 60)

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

    results = {}

    techniques = [
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

    for label, PipelineClass, kwargs in techniques:
        cfg = smoke_cfg()  # Uses full locked Reference Config
        try:
            dataset = load_dataset(cfg)
            pipe = PipelineClass(config=cfg, **kwargs)
            result = pipe.run(dataset)
            results[label] = _check_result(f"Technique/{label}", result)
        except ImportError as e:
            print(f"  ⚠️  Technique/{label}: SKIP (missing dependency: {e})")
            results[label] = None  # Not a failure — just missing dep
        except Exception as e:
            print(f"  ❌ Technique/{label}: {e}")
            results[label] = False

    return results


# =========================================================================
# ENCODER COMPREHENSIVE (if deps available)
# =========================================================================

def smoke_all_encoders() -> dict[str, bool]:
    """Test every encoder the system supports (standalone, no locking)."""
    print("\n" + "=" * 60)
    print("ENCODER COMPREHENSIVE (all available)")
    print("=" * 60)

    all_encoders = [
        "tfidf", "bm25", "fasttext_sparse", "fasttext_dense",
        "dense", "hybrid", "splade",
    ]

    results = {}
    for enc in all_encoders:
        cfg = smoke_cfg(encoder_type=enc)
        try:
            dataset = load_dataset(cfg)
            pipe = HybridPipeline(cfg)
            result = pipe.run(dataset)
            results[f"encoder_{enc}"] = _check_result(f"Encoder/{enc}", result)
        except ImportError as e:
            print(f"  ⚠️  Encoder/{enc}: SKIP (missing dependency: {e})")
            results[f"encoder_{enc}"] = None
        except Exception as e:
            print(f"  ❌ Encoder/{enc}: {e}")
            results[f"encoder_{enc}"] = False

    return results


# =========================================================================
# MAIN RUNNER
# =========================================================================

def run_all_smoke_tests() -> dict[str, dict[str, bool]]:
    """Run every smoke test sequentially (with step-locking) and return summary."""
    global LOCKED_CONFIG
    LOCKED_CONFIG = {  # Reset
        "classifier_type": "lr",
        "ws_accuracy_threshold": 0.6,
        "weak_cert_alpha": 0.9,
        "lf_confidence_threshold": 0.7,
        "encoder_type": None,
        "query_strategy": None,
        "label_model": None,
        "use_weak_cert": None,
        "use_weak_clust": None,
        "use_topic_lf": None,
    }

    all_results = {}

    # ORDER MATTERS — each step locks config for the next
    steps = [
        ("Step1: Encoding",      smoke_step1_encoding),
        ("Step1b: Alpha",        smoke_step1b_alpha),
        ("Step2: AL Strategy",   smoke_step2_al_strategy),
        ("Step4: Aggregation",   smoke_step4_aggregation),
        ("Step5: Auto-labeling", smoke_step5_auto_labeling),
        ("Step3: WS LFs",        smoke_step3_ws_lfs),
        ("Step6: Integration",   smoke_step6_integration),
        ("Techniques: T1-T14",   smoke_techniques),
    ]

    total_start = time.time()

    for name, fn in steps:
        step_start = time.time()
        print(f"\n{'─' * 60}")
        print(f"Running: {name}")
        print(f"{'─' * 60}")
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"  ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"_step": False}

        elapsed = time.time() - step_start
        print(f"  ⏱  {name}: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # ── PRINT FINAL REFERENCE CONFIG ──
    print("\n\n" + "=" * 70)
    print("FINAL REFERENCE CONFIG (locked by sequential step evaluation)")
    print("=" * 70)
    for k, v in LOCKED_CONFIG.items():
        status = "✅ locked" if v is not None else "❌ NOT LOCKED"
        print(f"  {k:30s} = {str(v):30s}  [{status}]")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for step_name, step_results in all_results.items():
        print(f"\n  {step_name}:")
        for test_name, status in step_results.items():
            if status is None:
                print(f"    ⚠️  {test_name}: SKIP")
                skipped += 1
            elif status:
                print(f"    ✅ {test_name}: PASS")
                passed += 1
            else:
                print(f"    ❌ {test_name}: FAIL")
                failed += 1
            total += 1

    print(f"\n{'─' * 70}")
    print(f"  Total: {total} | ✅ Pass: {passed} | ❌ Fail: {failed} | ⚠️  Skip: {skipped}")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"{'─' * 70}")

    if failed > 0:
        print("\n  ⛔ SOME TESTS FAILED — fix before running full experiments!")
    else:
        print("\n  🟢 ALL TESTS PASSED — safe to run full experiments!")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smoke Tests for Thesis Pipeline")
    parser.add_argument("--step", type=str, default=None,
                        help="Run a specific step: 1, 1b, 2, 3, 4, 5, 6, techniques, encoders")
    parser.add_argument("--all", action="store_true",
                        help="Run all smoke tests (sequential step-locking)")

    args = parser.parse_args()

    if args.all:
        run_all_smoke_tests()
    elif args.step == "1":
        smoke_step1_encoding()
    elif args.step == "1b":
        smoke_step1b_alpha()
    elif args.step == "2":
        smoke_step2_al_strategy()
    elif args.step == "3":
        smoke_step3_ws_lfs()
    elif args.step == "4":
        smoke_step4_aggregation()
    elif args.step == "5":
        smoke_step5_auto_labeling()
    elif args.step == "6":
        smoke_step6_integration()
    elif args.step == "techniques":
        smoke_techniques()
    elif args.step == "encoders":
        smoke_all_encoders()
    else:
        print("Use --all to run sequential step-locking smoke tests, or --step <N> for a single step.")
        print("Valid steps: 1, 1b, 2, 3, 4, 5, 6, techniques, encoders")
        sys.exit(1)


if __name__ == "__main__":
    main()
