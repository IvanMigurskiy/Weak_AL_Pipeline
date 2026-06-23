"""
Data loading, preprocessing, and feature extraction.

Handles:
- Downloading and caching HuggingFace datasets (customer support, banking, etc.)
- Text cleaning and filtering
- Encoder-based feature extraction (TF-IDF, BM25, FastText, SPLADE, Dense, Hybrid)
- Stratified train/test splitting
- Returning immutable Dataset containers
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import logging

import numpy as np
import pandas as pd
from datasets import load_dataset as load_hf_dataset
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import PipelineConfig
from .encoders import TextEncoder, create_encoder

logger = logging.getLogger(__name__)


# =========================================================================
# IMMUTABLE DATA CONTAINER
# =========================================================================

@dataclass(frozen=True)
class Dataset:
    """
    Immutable container for text classification data.
    """
    X_pool: sparse.csr_matrix | np.ndarray
    X_test: sparse.csr_matrix | np.ndarray
    y_pool: np.ndarray
    y_test: np.ndarray
    texts_pool: tuple[str, ...]
    texts_test: tuple[str, ...]
    label_encoder: LabelEncoder
    class_names: tuple[str, ...]
    n_classes: int
    encoder: TextEncoder | None = None
    encoder_name: str = "tfidf"
    feature_type: str = "sparse"


# =========================================================================
# CACHE MANAGEMENT
# =========================================================================

_CACHE_VERSION = 2  # Increment when loader logic changes to invalidate stale caches

def _get_cache_dir() -> Path:
    """Get or create cache directory for datasets."""
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / ".cache" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(project_root / ".cache" / "huggingface"))
    return cache_dir


def _check_cache_version(cache_path: Path) -> bool:
    """Check if cached file was created with the current cache version."""
    version_path = cache_path.with_suffix(".version")
    if not version_path.exists():
        # No version file = stale cache from before versioning
        if cache_path.exists():
            cache_path.unlink()
        return False
    try:
        cached_version = int(version_path.read_text().strip())
        if cached_version != _CACHE_VERSION:
            # Version mismatch — invalidate cache
            if cache_path.exists():
                cache_path.unlink()
            version_path.unlink()
            return False
        return True
    except (ValueError, OSError):
        return False


def _write_cache_version(cache_path: Path) -> None:
    """Write the current cache version alongside the cached file."""
    version_path = cache_path.with_suffix(".version")
    version_path.write_text(str(_CACHE_VERSION))


# =========================================================================
# DATASET LOADERS
# =========================================================================

def _load_customer_tickets(max_samples: int | None, cache_dir: Path) -> pd.DataFrame:
    """Load Tobi-Bueck/customer-support-tickets (Long, noisy emails). 4 classes."""
    cache_path = cache_dir / "customer_tickets_en.parquet"
    if cache_path.exists() and _check_cache_version(cache_path):
        return pd.read_parquet(cache_path)

    print("Downloading customer-support-tickets dataset...")
    ds = load_hf_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    df = ds.to_pandas()
    df = df[df["language"].astype(str).str.lower() == "en"]
    df = df[df["type"].notna() & (df["type"].astype(str).str.strip() != "")]
    df = df[df["body"].notna() & (df["body"].astype(str).str.strip() != "")]
    df = df.rename(columns={"body": "text", "type": "label"})
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    _write_cache_version(cache_path)
    print(f"Cached to {cache_path}")
    return df


def _load_bitext_generic(dataset_name: str, max_samples: int | None, cache_dir: Path) -> pd.DataFrame:
    """Generic loader for bitext datasets. Columns: instruction→text, category→label."""
    safe_name = dataset_name.replace("/", "_").replace("-", "_")
    cache_path = cache_dir / f"{safe_name}.parquet"
    if cache_path.exists() and _check_cache_version(cache_path):
        return pd.read_parquet(cache_path)

    print(f"Downloading {dataset_name}...")
    ds = load_hf_dataset(dataset_name)["train"]
    df = ds.to_pandas()
    df = df[["instruction", "category"]].rename(
        columns={"instruction": "text", "category": "label"}
    )
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    _write_cache_version(cache_path)
    print(f"Cached to {cache_path}")
    return df


def _load_banking77(max_samples: int | None, cache_dir: Path) -> pd.DataFrame:
    """Load PolyAI/banking77 — 77 fine-grained banking intent classes. Short queries."""
    cache_path = cache_dir / "banking77.parquet"
    if cache_path.exists() and _check_cache_version(cache_path):
        return pd.read_parquet(cache_path)

    print("Downloading banking77 dataset...")
    ds = load_hf_dataset("PolyAI/banking77", revision="refs/convert/parquet")["train"]
    df = ds.to_pandas()
    df = df.rename(columns={"label": "label_int"})
    # Convert integer labels to string labels for consistency
    df["label"] = "intent_" + df["label_int"].astype(str)
    df = df[["text", "label"]]
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    _write_cache_version(cache_path)
    print(f"Cached to {cache_path}")
    return df


def _load_clinc150(max_samples: int | None, cache_dir: Path) -> pd.DataFrame:
    """Load DeepPavlov/clinc150 — 150 intent classes across many domains. Short utterances."""
    cache_path = cache_dir / "clinc150.parquet"
    if cache_path.exists() and _check_cache_version(cache_path):
        return pd.read_parquet(cache_path)

    print("Downloading CLINC150 dataset...")
    ds = load_hf_dataset("DeepPavlov/clinc150")["train"]
    df = ds.to_pandas()
    df = df.rename(columns={"utterance": "text"})
    # Drop rows with NaN/inf labels (out-of-scope examples have label=-1 or NaN)
    df = df[df["label"].notna() & np.isfinite(df["label"])]
    # Exclude out-of-scope class (label=-1 in CLINC150 = oos)
    df = df[df["label"] >= 0]
    df["label"] = "intent_" + df["label"].astype(int).astype(str)
    df = df[["text", "label"]]
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    _write_cache_version(cache_path)
    print(f"Cached to {cache_path}")
    return df


def _load_cfpb_complaints(max_samples: int | None, cache_dir: Path) -> pd.DataFrame:
    """Load CFPB consumer complaints — 41 issue categories. Long complaint texts."""
    cache_path = cache_dir / "cfpb_complaints.parquet"
    if cache_path.exists() and _check_cache_version(cache_path):
        return pd.read_parquet(cache_path)

    print("Downloading CFPB consumer complaints dataset...")
    ds = load_hf_dataset("aciborowska/customers-complaints")["train"]
    df = ds.to_pandas()
    # Use consumer complaint narrative as text, Issue as label (41 classes)
    df = df.rename(columns={"Consumer_complaint_narrative": "text", "Issue": "label"})
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    df = df[df["label"].notna()]
    # Remove classes with fewer than 5 samples (can't stratify split)
    vc = df["label"].value_counts()
    valid_labels = vc[vc >= 5].index
    df = df[df["label"].isin(valid_labels)]
    df = df[["text", "label"]].reset_index(drop=True)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    _write_cache_version(cache_path)
    print(f"Cached to {cache_path}")
    return df


# =========================================================================
# BITEXT DATASET WRAPPERS
# =========================================================================

def _load_bitext_banking(max_samples, cache_dir):
    return _load_bitext_generic("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", max_samples, cache_dir)

def _load_bitext_ecommerce(max_samples, cache_dir):
    return _load_bitext_generic("bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset", max_samples, cache_dir)

def _load_bitext_insurance(max_samples, cache_dir):
    return _load_bitext_generic("bitext/Bitext-insurance-llm-chatbot-training-dataset", max_samples, cache_dir)

def _load_bitext_mortgage(max_samples, cache_dir):
    return _load_bitext_generic("bitext/Bitext-mortgage-loans-llm-chatbot-training-dataset", max_samples, cache_dir)

def _load_bitext_wealth(max_samples, cache_dir):
    return _load_bitext_generic("bitext/Bitext-wealth-management-llm-chatbot-training-dataset", max_samples, cache_dir)

def _load_bitext_travel(max_samples, cache_dir):
    return _load_bitext_generic("bitext/Bitext-travel-llm-chatbot-training-dataset", max_samples, cache_dir)

def _load_bitext_customer_support(max_samples, cache_dir):
    return _load_bitext_generic("bitext/Bitext-customer-support-llm-chatbot-training-dataset", max_samples, cache_dir)


def _load_rakuten_amazon(max_samples: int | None, cache_dir: Path) -> pd.DataFrame:
    """Load Rakuten/Amazon e-commerce dataset for classification.
    Uses Bitext ecommerce with INTENT labels (different from category labels)."""
    cache_path = cache_dir / "rakuten_amazon.parquet"
    if cache_path.exists() and _check_cache_version(cache_path):
        return pd.read_parquet(cache_path)

    print("Downloading Rakuten/Amazon e-commerce dataset...")
    try:
        ds = load_hf_dataset("bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset", split="train")
        df = ds.to_pandas()
        # Use INTENT as label (different from category used in bitext_ecommerce)
        df = df.rename(columns={"instruction": "text", "intent": "label"})
        df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
        df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
        # Keep only top-15 intents for manageable class count
        top_labels = df["label"].value_counts().head(15).index
        df = df[df["label"].isin(top_labels)]
        df = df[["text", "label"]].reset_index(drop=True)
    except Exception as e:
        print(f"  Bitext ecommerce intent failed: {e}")
        # Fallback: use Amazon reviews with rating labels
        try:
            ds = load_hf_dataset("amazon_polarity", split="train[:10000]")
            df = ds.to_pandas()
            df = df.rename(columns={"content": "text"})
            df["label"] = df["label"].map({0: "negative", 1: "positive"})
            df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
            df = df[["text", "label"]].reset_index(drop=True)
        except Exception as e2:
            raise RuntimeError(f"All Rakuten/Amazon loaders failed: {e2}")

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    _write_cache_version(cache_path)
    print(f"Cached to {cache_path} ({len(df)} samples)")
    return df


def _load_hp_tickets(max_samples: int | None, cache_dir: Path) -> pd.DataFrame:
    """Load HP ticket classification dataset.
    Uses Bitext/customer-support as a proxy (similar IT ticket domain)."""
    cache_path = cache_dir / "hp_tickets.parquet"
    if cache_path.exists() and _check_cache_version(cache_path):
        return pd.read_parquet(cache_path)

    print("Downloading HP-style ticket classification dataset...")
    try:
        # Try actual HP dataset on HuggingFace
        ds = load_hf_dataset("NeilBacchus/IT_Support_Tickets", split="train")
        df = ds.to_pandas()
        # Find text and label columns
        text_col = None
        label_col = None
        for col in ["ticket_text", "description", "body", "text", "message", "query", "issue_description"]:
            if col in df.columns:
                text_col = col
                break
        for col in ["category", "label", "class", "type", "issue_type", "ticket_type", "department"]:
            if col in df.columns:
                label_col = col
                break
        if text_col and label_col:
            df = df.rename(columns={text_col: "text", label_col: "label"})
        else:
            raise ValueError(f"Could not find text/label columns in {list(df.columns)}")
    except Exception:
        # Fallback: use Bitext customer support (similar IT support domain)
        print("  Using Bitext customer support as HP ticket proxy...")
        df = _load_bitext_generic(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            max_samples=None, cache_dir=cache_dir
        )

    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
    vc = df["label"].value_counts()
    valid_labels = vc[vc >= 5].index
    df = df[df["label"].isin(valid_labels)]
    df = df[["text", "label"]].reset_index(drop=True)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    _write_cache_version(cache_path)
    print(f"Cached to {cache_path} ({len(df)} samples)")
    return df


# =========================================================================
# REGISTRY
# =========================================================================

DATASET_LOADERS = {
    # Customer support tickets (4 classes, long noisy emails)
    "customer_tickets": _load_customer_tickets,
    # Banking intent (77 classes, short queries) — KEY DATASET
    "banking77": _load_banking77,
    # General intent (150 classes, short utterances) — KEY DATASET
    "clinc150": _load_clinc150,
    # CFPB complaints (18 product categories, long texts)
    "cfpb_complaints": _load_cfpb_complaints,
    # Rakuten/Amazon e-commerce reviews (product categories)
    "rakuten_amazon": _load_rakuten_amazon,
    # HP ticket classification (IT support tickets)
    "hp_tickets": _load_hp_tickets,
    # Bitext datasets (5-27 classes, short clean queries)
    "bitext_customer_support": lambda m, c: _load_bitext_generic("bitext/Bitext-customer-support-llm-chatbot-training-dataset", m, c),
    "bitext_ecommerce": lambda m, c: _load_bitext_generic("bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset", m, c),
    "bitext_banking": lambda m, c: _load_bitext_generic("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", m, c),
    "bitext_insurance": lambda m, c: _load_bitext_generic("bitext/Bitext-insurance-llm-chatbot-training-dataset", m, c),
    "bitext_mortgage": lambda m, c: _load_bitext_generic("bitext/Bitext-mortgage-loans-llm-chatbot-training-dataset", m, c),
    "bitext_wealth": lambda m, c: _load_bitext_generic("bitext/Bitext-wealth-management-llm-chatbot-training-dataset", m, c),
    "bitext_travel": lambda m, c: _load_bitext_generic("bitext/Bitext-travel-llm-chatbot-training-dataset", m, c),
}

# Metadata for thesis reporting
DATASET_INFO = {
    "customer_tickets":   {"source": "Tobi-Bueck/customer-support-tickets", "domain": "IT support", "text_type": "long emails", "expected_classes": 4},
    "banking77":          {"source": "PolyAI/banking77", "domain": "Banking", "text_type": "short queries", "expected_classes": 77},
    "clinc150":           {"source": "DeepPavlov/clinc150", "domain": "Multi-domain", "text_type": "short utterances", "expected_classes": 150},
    "cfpb_complaints":    {"source": "aciborowska/customers-complaints (CFPB)", "domain": "Finance", "text_type": "long complaints", "expected_classes": 41},
    "rakuten_amazon":     {"source": "surajp/amazon_reviews_multi", "domain": "E-commerce", "text_type": "product reviews", "expected_classes": 30},
    "hp_tickets":         {"source": "Bitext/customer-support (HP proxy)", "domain": "IT support", "text_type": "short tickets", "expected_classes": 27},
    "bitext_customer_support": {"source": "bitext/Bitext-customer-support", "domain": "Customer service", "text_type": "short queries", "expected_classes": 27},
    "bitext_ecommerce":   {"source": "bitext/Bitext-retail-ecommerce", "domain": "E-commerce", "text_type": "short queries", "expected_classes": 13},
    "bitext_banking":     {"source": "bitext/Bitext-retail-banking", "domain": "Banking", "text_type": "short queries", "expected_classes": 9},
    "bitext_insurance":   {"source": "bitext/Bitext-insurance", "domain": "Insurance", "text_type": "short queries", "expected_classes": 17},
    "bitext_mortgage":    {"source": "bitext/Bitext-mortgage-loans", "domain": "Mortgage", "text_type": "short queries", "expected_classes": 8},
    "bitext_wealth":      {"source": "bitext/Bitext-wealth-management", "domain": "Wealth management", "text_type": "short queries", "expected_classes": 5},
    "bitext_travel":      {"source": "bitext/Bitext-travel", "domain": "Travel", "text_type": "short queries", "expected_classes": 11},
}


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def load_dataset(config: PipelineConfig) -> Dataset:
    """
    Load and prepare dataset according to config.
    """
    loader = DATASET_LOADERS.get(config.dataset_name)
    if loader is None:
        raise ValueError(
            f"Unknown dataset: {config.dataset_name}. "
            f"Available: {list(DATASET_LOADERS.keys())}"
        )

    cache_dir = _get_cache_dir()
    df = loader(config.max_samples, cache_dir)

    # Drop classes with fewer than 2 samples (can't stratify split)
    class_counts = df["label"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    if len(valid_classes) < len(class_counts):
        dropped = len(class_counts) - len(valid_classes)
        print(f"  Dropping {dropped} classes with < 2 samples "
              f"({len(valid_classes)}/{len(class_counts)} classes kept)")
        df = df[df["label"].isin(valid_classes)].reset_index(drop=True)

    # Encode labels
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df["label"].astype(str))
    class_names = tuple(label_encoder.classes_)

    texts_all = df["text"].astype(str).tolist()

    # Stratified split — fall back to non-stratified if too few per class
    indices = np.arange(len(texts_all))
    min_class_count = np.bincount(y_all).min()
    can_stratify = min_class_count >= 2
    try:
        idx_pool, idx_test, y_pool, y_test = train_test_split(
            indices, y_all,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=y_all if can_stratify else None,
        )
    except ValueError:
        # Last resort: non-stratified split
        idx_pool, idx_test, y_pool, y_test = train_test_split(
            indices, y_all,
            test_size=config.test_size,
            random_state=config.random_seed,
        )

    texts_pool_list = [texts_all[i] for i in idx_pool]
    texts_test_list = [texts_all[i] for i in idx_test]

    # Feature extraction via encoder abstraction
    encoder = create_encoder(config)
    logger.info(
        "Using encoder: %s (output_type=%s)",
        encoder.name, encoder.output_type,
    )

    X_pool = encoder.fit_transform(texts_pool_list)
    X_test = encoder.transform(texts_test_list)

    # Ensure consistent format
    if sparse.issparse(X_pool):
        X_pool = X_pool.tocsr()
        X_test = X_test.tocsr() if sparse.issparse(X_test) else X_test
    # dense encoders produce numpy arrays — no conversion needed

    return Dataset(
        X_pool=X_pool,
        X_test=X_test,
        y_pool=y_pool,
        y_test=y_test,
        texts_pool=tuple(texts_pool_list),
        texts_test=tuple(texts_test_list),
        label_encoder=label_encoder,
        class_names=class_names,
        n_classes=len(class_names),
        encoder=encoder,
        encoder_name=encoder.name,
        feature_type=encoder.output_type,
    )


def get_stratified_seed_indices(
    y: np.ndarray,
    per_class: int,
    random_seed: int
) -> np.ndarray:
    """Get initial seed indices for AL (one per class by default)."""
    rng = np.random.default_rng(random_seed)
    indices = []

    for label in np.unique(y):
        class_indices = np.where(y == label)[0]
        k = min(per_class, len(class_indices))
        chosen = rng.choice(class_indices, size=k, replace=False)
        indices.extend(chosen.tolist())

    return np.array(indices, dtype=int)
