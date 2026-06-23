"""
Text encoder abstraction layer.

Provides a unified interface for different text feature extraction methods,
allowing the pipeline to swap between TF-IDF, BM25, FastText (sparse/dense),
SPLADE, SentenceTransformer, and hybrid encoders via a single config field.

All encoders follow the sklearn-style fit/transform API:
  - fit(texts) -> self
  - transform(texts) -> feature matrix (sparse or dense)
  - fit_transform(texts) -> feature matrix
  - get_feature_names() -> list[str] | None
  - get_vocabulary() -> dict | None
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


# =========================================================================
# ABC
# =========================================================================

class TextEncoder(ABC):
    """Base class for all text encoders.

    Subclasses must implement fit, transform, and describe their output type.
    Optionally override get_feature_names / get_vocabulary for encoders that
    maintain a vocabulary (e.g. TF-IDF, BM25, FastTextSparse, SPLADE).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable encoder name."""
        ...

    @property
    @abstractmethod
    def output_type(self) -> Literal["sparse", "dense"]:
        """Whether transform() returns a sparse or dense matrix."""
        ...

    @abstractmethod
    def fit(self, texts: list[str]) -> TextEncoder:
        """Fit encoder on training texts. Returns self."""
        ...

    @abstractmethod
    def transform(self, texts: list[str]) -> sparse.csr_matrix | np.ndarray:
        """Transform texts to feature matrix using fitted encoder."""
        ...

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix | np.ndarray:
        """Fit and transform in one step (override for efficiency)."""
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> list[str] | None:
        """Return feature names if available (e.g. vocabulary terms)."""
        return None

    def get_vocabulary(self) -> dict[str, int] | None:
        """Return vocabulary mapping {term: index} if available."""
        return None

    @property
    def n_features(self) -> int | None:
        """Number of features in the output matrix (None if not yet fitted)."""
        return None


# =========================================================================
# TF-IDF ENCODER
# =========================================================================

class TfidfEncoder(TextEncoder):
    """TF-IDF vectorization using sklearn TfidfVectorizer.

    This is the default encoder and produces identical results to the
    previous hardcoded TF-IDF path in data.py.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self._vectorizer: Any = None

    @property
    def name(self) -> str:
        return "tfidf"

    @property
    def output_type(self) -> Literal["sparse"]:
        return "sparse"

    @property
    def n_features(self) -> int | None:
        if self._vectorizer is None:
            return None
        return len(self._vectorizer.vocabulary_)

    def fit(self, texts: list[str]) -> TfidfEncoder:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )
        self._vectorizer.fit(texts)
        logger.info(
            "TfidfEncoder fitted: vocab=%d, ngram_range=%s",
            len(self._vectorizer.vocabulary_),
            self.ngram_range,
        )
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        if self._vectorizer is None:
            raise RuntimeError("TfidfEncoder not fitted. Call fit() first.")
        return self._vectorizer.transform(texts).tocsr()

    def get_feature_names(self) -> list[str] | None:
        if self._vectorizer is None:
            return None
        return list(self._vectorizer.get_feature_names_out())

    def get_vocabulary(self) -> dict[str, int] | None:
        if self._vectorizer is None:
            return None
        return self._vectorizer.vocabulary_

    @property
    def vectorizer(self) -> Any:
        """Access the underlying TfidfVectorizer (for backward compat)."""
        return self._vectorizer


# =========================================================================
# BM25 ENCODER
# =========================================================================

class BM25Encoder(TextEncoder):
    """BM25-style sparse encoding with term-frequency saturation.

    Applies the BM25 saturation function to raw term frequencies:
        tf_sat = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

    This handles long documents better than TF-IDF, where raw TF can
    inflate uninformative terms. Uses IDF weighting from TfidfVectorizer
    and then re-weights TF using BM25 saturation.

    Produces the same sparse shape as TF-IDF (same vocabulary).
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.k1 = k1
        self.b = b
        self._vectorizer: Any = None
        self._avgdl: float = 0.0

    @property
    def name(self) -> str:
        return "bm25"

    @property
    def output_type(self) -> Literal["sparse"]:
        return "sparse"

    @property
    def n_features(self) -> int | None:
        if self._vectorizer is None:
            return None
        return len(self._vectorizer.vocabulary_)

    def fit(self, texts: list[str]) -> BM25Encoder:
        from sklearn.feature_extraction.text import CountVectorizer

        # Use CountVectorizer (no TF weighting) — we apply BM25 ourselves
        self._vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            binary=False,  # need raw counts for BM25
        )
        X_counts = self._vectorizer.fit_transform(texts)
        # Compute average document length (in terms)
        self._avgdl = X_counts.sum(axis=1).mean()
        logger.info(
            "BM25Encoder fitted: vocab=%d, k1=%.2f, b=%.2f, avgdl=%.1f",
            len(self._vectorizer.vocabulary_), self.k1, self.b, self._avgdl,
        )
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        if self._vectorizer is None:
            raise RuntimeError("BM25Encoder not fitted. Call fit() first.")

        X_counts = self._vectorizer.transform(texts).tocsr()
        # BM25 saturation per document
        X_bm25 = X_counts.copy().astype(np.float64)

        for i in range(X_bm25.shape[0]):
            start, end = X_bm25.indptr[i], X_bm25.indptr[i + 1]
            dl = X_counts[i].sum()
            for j in range(start, end):
                tf = X_counts[i].data[j - X_counts[0].indptr[0]] if i == 0 else X_counts.data[start + (j - start)]
                # This is too slow per-element; use vectorized approach below

        # Vectorized BM25 saturation
        tf = X_counts.astype(np.float64)
        dl = np.asarray(tf.sum(axis=1)).flatten()  # doc lengths
        tf_data = tf.data
        dl_row = np.repeat(dl, np.diff(tf.indptr))

        # BM25 TF saturation: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
        denominator = tf_data + self.k1 * (1.0 - self.b + self.b * dl_row / self._avgdl)
        X_bm25 = tf.copy()
        X_bm25.data = (tf_data * (self.k1 + 1.0)) / denominator

        # Apply IDF weighting
        n_docs = X_counts.shape[0]
        df = np.asarray((X_counts > 0).sum(axis=0)).flatten()
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        X_bm25 = X_bm25.multiply(idf)

        return X_bm25.tocsr()

    def get_feature_names(self) -> list[str] | None:
        if self._vectorizer is None:
            return None
        return list(self._vectorizer.get_feature_names_out())

    def get_vocabulary(self) -> dict[str, int] | None:
        if self._vectorizer is None:
            return None
        return self._vectorizer.vocabulary_


# =========================================================================
# FASTTEXT SPARSE ENCODER
# =========================================================================

class FastTextSparseEncoder(TextEncoder):
    """Typo-resilient sparse encoder via FastText subword nearest-neighbor mapping.

    Trains a FastText model on the corpus, then for each token in a document
    maps it to its nearest vocabulary term using subword similarity:
        "l0gin" → "login", "acount" → "account"

    This produces a sparse matrix in the same shape as a vocabulary-based
    encoder, so it's compatible with MultinomialNB and KeywordLF.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 2,
        max_df: float = 0.95,
        fasttext_dim: int = 100,
        fasttext_window: int = 5,
        fasttext_min_count: int = 1,
        fasttext_epochs: int = 5,
        fasttext_sg: int = 1,
        nn_top_k: int = 1,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.fasttext_dim = fasttext_dim
        self.fasttext_window = fasttext_window
        self.fasttext_min_count = fasttext_min_count
        self.fasttext_epochs = fasttext_epochs
        self.fasttext_sg = fasttext_sg
        self.nn_top_k = nn_top_k
        self._ft_model: Any = None
        self._vocab: dict[str, int] = {}
        self._feature_names: list[str] = []
        self._vocab_matrix: np.ndarray | None = None  # Pre-computed normalized vocab vectors
        self._vocab_matrix_indices: np.ndarray | None = None  # Mapping matrix rows → vocab indices

    @property
    def name(self) -> str:
        return "fasttext_sparse"

    @property
    def output_type(self) -> Literal["sparse"]:
        return "sparse"

    @property
    def n_features(self) -> int | None:
        if not self._vocab:
            return None
        return len(self._vocab)

    def fit(self, texts: list[str]) -> FastTextSparseEncoder:
        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(
                "gensim is required for FastTextSparseEncoder. "
                "Install it with: pip install gensim"
            )

        # Train FastText on the corpus
        tokenized = [text.lower().split() for text in texts]
        self._ft_model = FastText(
            sentences=tokenized,
            vector_size=self.fasttext_dim,
            window=self.fasttext_window,
            min_count=self.fasttext_min_count,
            epochs=self.fasttext_epochs,
            sg=self.fasttext_sg,
        )

        # Build vocabulary from the corpus (using CountVectorizer for consistent vocab)
        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        cv.fit(texts)
        self._vocab = cv.vocabulary_
        self._feature_names = list(cv.get_feature_names_out())

        logger.info(
            "FastTextSparseEncoder fitted: vocab=%d, ft_dim=%d",
            len(self._vocab), self.fasttext_dim,
        )
        return self

    def _build_vocab_matrix(self) -> None:
        """Pre-compute a normalized matrix of vocab term vectors for fast NN lookup."""
        if self._vocab_matrix is not None:
            return
        # Collect vectors for vocab terms that exist in FastText model
        vocab_vecs = []
        vocab_indices = []
        for term in self._feature_names:
            if term in self._ft_model.wv:
                vocab_vecs.append(self._ft_model.wv[term])
                vocab_indices.append(self._vocab[term])
        if not vocab_vecs:
            self._vocab_matrix = None
            self._vocab_matrix_indices = np.array([], dtype=int)
            return
        mat = np.array(vocab_vecs, dtype=np.float32)
        # L2-normalize rows for cosine similarity via dot product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._vocab_matrix = mat / norms
        self._vocab_matrix_indices = np.array(vocab_indices, dtype=int)

    def _map_token_to_vocab(self, token: str) -> list[tuple[str, float]]:
        """Map a (possibly misspelled) token to nearest vocab terms."""
        if token in self._vocab:
            return [(token, 1.0)]

        # Use FastText subword similarity to find nearest vocab term
        if self._ft_model is None or token not in self._ft_model.wv:
            return []

        try:
            self._build_vocab_matrix()
            if self._vocab_matrix is None or len(self._vocab_matrix) == 0:
                return []

            token_vec = self._ft_model.wv[token].astype(np.float32)
            norm = np.linalg.norm(token_vec)
            if norm < 1e-10:
                return []
            token_vec_normed = token_vec / norm

            # Vectorized cosine similarity against all vocab vectors
            sims = self._vocab_matrix @ token_vec_normed
            top_k = min(self.nn_top_k, len(sims))
            if top_k <= 0:
                return []
            best_idx = np.argpartition(sims, -top_k)[-top_k:]
            best_idx = best_idx[np.argsort(sims[best_idx])[::-1]]

            results = []
            for idx in best_idx:
                vocab_idx = self._vocab_matrix_indices[idx]
                term = self._feature_names[vocab_idx] if vocab_idx < len(self._feature_names) else None
                if term is not None:
                    results.append((term, float(sims[idx])))
            return results[:self.nn_top_k]
        except Exception:
            return []

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        if self._ft_model is None or not self._vocab:
            raise RuntimeError("FastTextSparseEncoder not fitted. Call fit() first.")

        self._build_vocab_matrix()
        vocab_size = len(self._vocab)
        n_texts = len(texts)

        rows, cols, data = [], [], []

        for i, text in enumerate(texts):
            tokens = text.lower().split()
            term_weights: dict[int, float] = {}

            # Fast path: exact vocab matches (most tokens)
            oov_tokens = []
            for token in tokens:
                if token in self._vocab:
                    idx = self._vocab[token]
                    term_weights[idx] = term_weights.get(idx, 0.0) + 1.0
                else:
                    oov_tokens.append(token)

            # Slow path: batch vectorized NN for OOV tokens
            if oov_tokens and self._vocab_matrix is not None and len(self._vocab_matrix) > 0:
                # Get vectors for all OOV tokens that exist in FastText
                oov_vecs = []
                oov_valid = []
                for token in oov_tokens:
                    if token in self._ft_model.wv:
                        vec = self._ft_model.wv[token].astype(np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 1e-10:
                            oov_vecs.append(vec / norm)
                            oov_valid.append(token)

                if oov_vecs:
                    # Batch cosine similarity: (n_oov, n_vocab)
                    oov_mat = np.array(oov_vecs, dtype=np.float32)
                    sim_matrix = oov_mat @ self._vocab_matrix.T

                    for j, token in enumerate(oov_valid):
                        sims = sim_matrix[j]
                        top_k = min(self.nn_top_k, len(sims))
                        best_idx = np.argpartition(sims, -top_k)[-top_k:]
                        best_idx = best_idx[np.argsort(sims[best_idx])[::-1]]
                        for idx in best_idx[:self.nn_top_k]:
                            vocab_idx = self._vocab_matrix_indices[idx]
                            term = self._feature_names[vocab_idx] if vocab_idx < len(self._feature_names) else None
                            if term is not None and term in self._vocab:
                                vidx = self._vocab[term]
                                term_weights[vidx] = term_weights.get(vidx, 0.0) + float(sims[idx])
            elif oov_tokens:
                # Fallback: per-token lookup (no vocab matrix)
                for token in oov_tokens:
                    mappings = self._map_token_to_vocab(token)
                    for mapped_term, weight in mappings:
                        if mapped_term in self._vocab:
                            idx = self._vocab[mapped_term]
                            term_weights[idx] = term_weights.get(idx, 0.0) + weight

            for idx, weight in term_weights.items():
                rows.append(i)
                cols.append(idx)
                data.append(weight)

            if (i + 1) % 500 == 0:
                logger.info("FastTextSparseEncoder: transformed %d/%d docs", i + 1, n_texts)
                print(f"    [fasttext_sparse] transformed {i + 1}/{n_texts} docs...", flush=True)

        X = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(texts), vocab_size),
            dtype=np.float64,
        )
        logger.info("FastTextSparseEncoder: transform complete, shape=%s", X.shape)
        return X

    def get_feature_names(self) -> list[str] | None:
        return self._feature_names if self._feature_names else None

    def get_vocabulary(self) -> dict[str, int] | None:
        return self._vocab if self._vocab else None


# =========================================================================
# FASTTEXT DENSE ENCODER
# =========================================================================

class FastTextDenseEncoder(TextEncoder):
    """FastText subword embeddings with mean pooling.

    Uses pre-trained Facebook FastText vectors (wiki-news-300d-1M-subword)
    by default, which gives meaningful embeddings without training from scratch.
    Falls back to training from scratch only if download fails.
    Produces dense numpy arrays.
    """

    # Pre-trained model name from gensim-data
    PRETRAINED_NAME = "fasttext-wiki-news-subwords-300"

    def __init__(
        self,
        model_path: str | None = None,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 5,
        sg: int = 1,
        use_pretrained: bool = True,
    ) -> None:
        self.model_path = model_path
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.use_pretrained = use_pretrained
        self._model: Any = None
        self._keyed_vectors: Any = None  # for pre-trained gensim KeyedVectors

    @property
    def name(self) -> str:
        return "fasttext_dense"

    @property
    def output_type(self) -> Literal["dense"]:
        return "dense"

    @property
    def n_features(self) -> int | None:
        if self._keyed_vectors is not None:
            return self._keyed_vectors.vector_size
        if self._model is not None:
            return self._model.vector_size
        return self.vector_size

    def fit(self, texts: list[str]) -> FastTextDenseEncoder:
        # Try pre-trained vectors first
        if self.use_pretrained and self.model_path is None:
            try:
                import gensim.downloader as api
                logger.info("FastTextDenseEncoder: downloading pre-trained vectors '%s'...",
                            self.PRETRAINED_NAME)
                self._keyed_vectors = api.load(self.PRETRAINED_NAME)
                self.vector_size = self._keyed_vectors.vector_size
                logger.info("FastTextDenseEncoder: loaded pre-trained vectors, dim=%d",
                            self.vector_size)
                return self
            except Exception as e:
                logger.warning("FastTextDenseEncoder: pre-trained download failed (%s), "
                               "falling back to training from scratch", e)

        # Try loading from local path
        if self.model_path and Path(self.model_path).exists():
            try:
                from gensim.models import FastText
                logger.info("FastTextDenseEncoder: loading model from %s", self.model_path)
                self._model = FastText.load(self.model_path)
                self.vector_size = self._model.vector_size
                return self
            except Exception:
                pass

        # Fall back: train from scratch (less ideal but works)
        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(
                "gensim is required for FastTextDenseEncoder. "
                "Install it with: pip install gensim"
            )
        tokenized = [text.lower().split() for text in texts]
        logger.info(
            "FastTextDenseEncoder: training from scratch on %d docs, dim=%d",
            len(tokenized), self.vector_size,
        )
        self._model = FastText(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._keyed_vectors is not None:
            # Pre-trained KeyedVectors path
            dim = self._keyed_vectors.vector_size
            embeddings = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                words = text.lower().split()
                word_vectors = [self._keyed_vectors[w] for w in words if w in self._keyed_vectors]
                if word_vectors:
                    embeddings[i] = np.mean(word_vectors, axis=0)
            return embeddings

        if self._model is None:
            raise RuntimeError("FastTextDenseEncoder not fitted. Call fit() first.")

        embeddings = np.zeros((len(texts), self._model.vector_size), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            if words:
                word_vectors = [self._model.wv[w] for w in words if w in self._model.wv]
                if word_vectors:
                    embeddings[i] = np.mean(word_vectors, axis=0)
        return embeddings

    def get_feature_names(self) -> list[str] | None:
        """Dense embeddings have no interpretable feature names."""
        return None

    def get_vocabulary(self) -> dict[str, int] | None:
        """Return word-to-index mapping from the FastText model."""
        if self._model is None:
            return None
        return {word: i for i, word in enumerate(self._model.wv.index_to_key)}


# =========================================================================
# SPLADE ENCODER
# =========================================================================

class SpladeEncoder(TextEncoder):
    """SPLADE v2 learned sparse encoder.

    SPLADE (Sparse Lexical Anomaly DEtection) produces sparse vectors where
    each dimension corresponds to a vocabulary term, but weights are learned
    via a transformer + MLM head. This solves TF-IDF's "synonym blindness":
    in TF-IDF, "car" and "automobile" are orthogonal; in SPLADE, both
    activate the same terms via expansion.

    Uses disk caching to avoid re-computing the expensive transformer
    forward pass on re-runs.

    Requires: torch, transformers
    """

    def __init__(
        self,
        model_name: str = "naver/splade_v2_max",
        batch_size: int = 32,
        device: str = "auto",
        cache_dir: str = ".cache/splade",
        max_length: int = 256,
        agg_strategy: str = "max",  # "max" or "sum"
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device_val = device
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.agg_strategy = agg_strategy
        self._model: Any = None
        self._tokenizer: Any = None
        self._vocab: dict[str, int] = {}
        self._feature_names: list[str] = []
        self._fitted = False

    @property
    def name(self) -> str:
        return "splade"

    @property
    def output_type(self) -> Literal["sparse"]:
        return "sparse"

    @property
    def n_features(self) -> int | None:
        if not self._vocab:
            return None
        return len(self._vocab)

    def _get_device(self) -> str:
        if self.device_val == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device_val

    def _load_model(self) -> None:
        """Load the SPLADE model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "torch and transformers are required for SpladeEncoder. "
                "Install them with: pip install torch transformers"
            )

        device = self._get_device()
        logger.info("SpladeEncoder: loading '%s' on %s", self.model_name, device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model.to(device)
        self._model.eval()

        # ── #6 OPTIMIZATION: torch.compile for faster inference ──
        try:
            import torch as _torch
            if hasattr(_torch, 'compile'):
                self._model = _torch.compile(self._model, mode="reduce-overhead")
                logger.info("SpladeEncoder: torch.compile enabled (reduce-overhead)")
        except Exception as e:
            logger.info("SpladeEncoder: torch.compile skipped (%s)", e)

        # SPLADE vocabulary = tokenizer vocabulary
        self._vocab = dict(self._tokenizer.get_vocab())
        self._feature_names = [
            self._tokenizer.convert_ids_to_tokens(i)
            for i in range(len(self._vocab))
        ]

    def _compute_splade(self, texts: list[str]) -> sparse.csr_matrix:
        """Run SPLADE forward pass and produce sparse matrix."""
        import torch

        all_rows = []
        device_str = self._get_device()

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(device_str) for k, v in tokens.items()}

            with torch.no_grad():
                output = self._model(**tokens)

            logits = output.logits  # (batch, seq_len, vocab_size)

            # Aggregate across tokens: max or sum over sequence dim
            if self.agg_strategy == "max":
                agg = torch.max(torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1), dim=1).values
            else:
                agg = torch.sum(torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1), dim=1)

            all_rows.append(agg.cpu().numpy())

        X_dense = np.vstack(all_rows)
        # Threshold small values for sparsity
        X_dense[X_dense < 1e-3] = 0
        X_sparse = sparse.csr_matrix(X_dense)
        return X_sparse

    def _cache_path(self, prefix: str) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{prefix}.npz"

    def _meta_path(self, prefix: str) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{prefix}_meta.json"

    def fit(self, texts: list[str]) -> SpladeEncoder:
        self._load_model()
        self._fitted = True
        logger.info("SpladeEncoder fitted: vocab=%d", len(self._vocab))
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        if not self._fitted:
            raise RuntimeError("SpladeEncoder not fitted. Call fit() first.")
        return self._compute_splade(texts)

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix:
        """Override: compute SPLADE and cache the result."""
        self._load_model()
        self._fitted = True

        cache_key = f"splade_train_{len(texts)}_{hash(tuple(texts[:10])) & 0xFFFFFFFF}"
        cache_p = self._cache_path(cache_key)
        if cache_p.exists():
            logger.info("SpladeEncoder: loading cached matrix from %s", cache_p)
            return sparse.load_npz(cache_p)

        X = self._compute_splade(texts)
        sparse.save_npz(cache_p, X)
        logger.info("SpladeEncoder: cached matrix to %s", cache_p)
        return X

    def get_feature_names(self) -> list[str] | None:
        return self._feature_names if self._feature_names else None

    def get_vocabulary(self) -> dict[str, int] | None:
        return self._vocab if self._vocab else None


# =========================================================================
# DENSE ENCODER (SentenceTransformer)
# =========================================================================

class DenseEncoder(TextEncoder):
    """Sentence-transformers dense embeddings.

    Uses pre-trained models from the sentence-transformers library.
    Produces fixed-size dense vectors, excellent for semantic similarity.
    Default model: BAAI/bge-small-en-v1.5 (384-dim, fast, strong).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        device: str = "auto",
        cache_dir: str = ".cache/sentence_transformers",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device_val = device
        self.cache_dir = cache_dir
        self._model: Any = None
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return "dense"

    @property
    def output_type(self) -> Literal["dense"]:
        return "dense"

    @property
    def n_features(self) -> int | None:
        return self._dim

    def _get_device(self) -> str:
        if self.device_val == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device_val

    def fit(self, texts: list[str]) -> DenseEncoder:
        """Load the model (no training needed — pre-trained)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for DenseEncoder. "
                "Install it with: pip install sentence-transformers"
            )

        device = self._get_device()
        logger.info("DenseEncoder: loading '%s' on %s", self.model_name, device)
        self._model = SentenceTransformer(
            self.model_name,
            cache_folder=self.cache_dir,
            device=device,
        )
        dummy = self._model.encode(["test"], convert_to_numpy=True)
        self._dim = dummy.shape[1]
        logger.info("DenseEncoder ready: dim=%d", self._dim)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("DenseEncoder not fitted. Call fit() first.")
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def get_feature_names(self) -> list[str] | None:
        """Dense embeddings have no interpretable feature names."""
        return None

    def get_vocabulary(self) -> dict[str, int] | None:
        """Sentence-transformers have no vocabulary mapping."""
        return None


# =========================================================================
# HYBRID ENCODER (sparse + dense)
# =========================================================================

class HybridEncoder(TextEncoder):
    """Hybrid sparse + dense encoder with weighted interpolation.

    Combines a sparse encoder (TF-IDF/BM25/SPLADE/FT-sparse) with a dense
    encoder (Dense/FT-dense) via L2-normalized weighted concatenation:

        X_hybrid = hstack([alpha * normalize(X_sparse),
                           (1-alpha) * normalize(X_dense)])

    where each modality is L2-normalized per-row before weighting, and
    alpha controls the sparse/dense balance:
        alpha = 1.0  → sparse only (dense contribution zeroed out)
        alpha = 0.0  → dense only (sparse contribution zeroed out)
        alpha = 0.5  → equal weight (default)

    This is better than plain hstack because:
    - L2 normalization prevents one modality from dominating due to scale
      (e.g., TF-IDF values ~0.1-1.0 vs dense embeddings ~-2.0 to 2.0)
    - alpha gives explicit control over sparse vs dense contribution
    - The resulting features work well with both sparse-aware classifiers
      (MultinomialNB) and general classifiers (RF, LR, SVM)

    Non-negative shift: when shift_dense=True, adds abs(min_value) to
    the dense component so MultinomialNB (requires non-negative inputs)
    can work. The shift value is computed once during fit_transform and
    reused consistently during transform.

    Has vocabulary from the sparse side → KeywordLF partially works.
    """

    def __init__(
        self,
        sparse_encoder: TextEncoder,
        dense_encoder: TextEncoder,
        shift_dense: bool = True,
        alpha: float = 0.5,
    ) -> None:
        self.sparse_encoder = sparse_encoder
        self.dense_encoder = dense_encoder
        self.shift_dense = shift_dense
        self.alpha = alpha  # sparse weight; (1-alpha) = dense weight
        self._fitted = False
        self._shift_value: float = 0.0  # cached from training data

    @property
    def name(self) -> str:
        return f"hybrid({self.sparse_encoder.name}+{self.dense_encoder.name})"

    @property
    def output_type(self) -> Literal["sparse"]:
        # After hstack, result is always sparse
        return "sparse"

    @property
    def n_features(self) -> int | None:
        s = self.sparse_encoder.n_features
        d = self.dense_encoder.n_features
        if s is not None and d is not None:
            return s + d
        return None

    def fit(self, texts: list[str]) -> HybridEncoder:
        self.sparse_encoder.fit(texts)
        self.dense_encoder.fit(texts)
        self._fitted = True
        logger.info(
            "HybridEncoder fitted: sparse=%s(%d), dense=%s(%d)",
            self.sparse_encoder.name, self.sparse_encoder.n_features or 0,
            self.dense_encoder.name, self.dense_encoder.n_features or 0,
        )
        return self

    def _normalize_and_combine(self, X_sparse, X_dense, is_training: bool = False) -> sparse.csr_matrix:
        """L2-normalize and weighted-concatenate sparse + dense features.

        Steps:
        1. Ensure sparse is csr, dense is ndarray
        2. Apply non-negative shift to dense (cached from training)
        3. L2-normalize each modality per row
        4. Scale: alpha * sparse_norm, (1-alpha) * dense_norm
        5. hstack and return as csr
        """
        if not sparse.issparse(X_sparse):
            X_sparse = sparse.csr_matrix(X_sparse)
        if sparse.issparse(X_dense):
            X_dense = X_dense.toarray()

        # Non-negative shift for MultinomialNB compatibility
        # During training: compute and cache the shift value
        # During inference: reuse the cached shift value for consistency
        if self.shift_dense and X_dense.min() < 0:
            if is_training:
                self._shift_value = abs(X_dense.min())
            X_dense = X_dense + self._shift_value

        # ── #4 OPTIMIZATION: L2-normalize sparse without .toarray() ──
        # Compute L2 norms from sparse data directly
        sparse_norms = np.sqrt(np.asarray(X_sparse.multiply(X_sparse).sum(axis=1)).flatten())
        sparse_norms[sparse_norms == 0] = 1.0
        # Normalize in-place on sparse (divide each row by its norm)
        diag_inv = sparse.diags(1.0 / sparse_norms)
        X_sparse_norm = diag_inv @ X_sparse

        # L2-normalize dense features per row
        dense_norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
        dense_norms[dense_norms == 0] = 1.0
        X_dense_norm = X_dense / dense_norms

        # Weighted scaling
        X_sparse_weighted = self.alpha * X_sparse_norm
        X_dense_weighted = (1.0 - self.alpha) * X_dense_norm

        # Concatenate (hstack) sparse and dense modalities
        X_hybrid = sparse.hstack(
            [X_sparse_weighted,
             sparse.csr_matrix(X_dense_weighted)],
            format="csr",
        )

        return X_hybrid

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        if not self._fitted:
            raise RuntimeError("HybridEncoder not fitted. Call fit() first.")

        X_sparse = self.sparse_encoder.transform(texts)
        X_dense = self.dense_encoder.transform(texts)

        return self._normalize_and_combine(X_sparse, X_dense, is_training=False)

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix:
        self.sparse_encoder.fit(texts)
        X_sparse = self.sparse_encoder.transform(texts)
        X_dense = self.dense_encoder.fit(texts).transform(texts)

        X_hybrid = self._normalize_and_combine(X_sparse, X_dense, is_training=True)
        self._fitted = True

        logger.info(
            "HybridEncoder fit_transform: shape=%s, alpha=%.2f",
            X_hybrid.shape, self.alpha,
        )
        return X_hybrid

    def get_feature_names(self) -> list[str] | None:
        sparse_names = self.sparse_encoder.get_feature_names()
        if sparse_names is None:
            return None
        n_dense = self.dense_encoder.n_features or 0
        dense_names = [f"dense_{i}" for i in range(n_dense)]
        return sparse_names + dense_names

    def get_vocabulary(self) -> dict[str, int] | None:
        sparse_vocab = self.sparse_encoder.get_vocabulary()
        if sparse_vocab is None:
            return None
        # Offset dense feature indices
        n_sparse = len(sparse_vocab)
        n_dense = self.dense_encoder.n_features or 0
        vocab = dict(sparse_vocab)
        for i in range(n_dense):
            vocab[f"dense_{i}"] = n_sparse + i
        return vocab


# =========================================================================
# FACTORY
# =========================================================================

_ENCODER_REGISTRY: dict[str, type[TextEncoder]] = {
    "tfidf": TfidfEncoder,
    "bm25": BM25Encoder,
    "fasttext_sparse": FastTextSparseEncoder,
    "fasttext_dense": FastTextDenseEncoder,
    "splade": SpladeEncoder,
    "dense": DenseEncoder,
}


def create_encoder(config: Any) -> TextEncoder:
    """Factory: create an encoder from a PipelineConfig instance.

    Reads encoder_type and all encoder-specific parameters from the config.
    For hybrid mode, recursively creates sparse + dense sub-encoders.
    """
    encoder_type = getattr(config, "encoder_type", "tfidf")

    if encoder_type == "tfidf":
        return TfidfEncoder(
            max_features=getattr(config, "max_features", 5000),
            ngram_range=getattr(config, "ngram_range", (1, 2)),
            min_df=getattr(config, "tfidf_min_df", 2),
            max_df=getattr(config, "tfidf_max_df", 0.95),
            sublinear_tf=getattr(config, "tfidf_sublinear_tf", True),
        )
    elif encoder_type == "bm25":
        return BM25Encoder(
            max_features=getattr(config, "max_features", 5000),
            ngram_range=getattr(config, "ngram_range", (1, 2)),
            min_df=getattr(config, "tfidf_min_df", 2),
            max_df=getattr(config, "tfidf_max_df", 0.95),
            k1=getattr(config, "bm25_k1", 1.5),
            b=getattr(config, "bm25_b", 0.75),
        )
    elif encoder_type == "fasttext_sparse":
        return FastTextSparseEncoder(
            max_features=getattr(config, "max_features", 5000),
            ngram_range=getattr(config, "ngram_range", (1, 1)),
            min_df=getattr(config, "tfidf_min_df", 2),
            max_df=getattr(config, "tfidf_max_df", 0.95),
            fasttext_dim=getattr(config, "fasttext_dim", 100),
            fasttext_window=getattr(config, "fasttext_window", 5),
            fasttext_min_count=getattr(config, "fasttext_min_count", 1),
            fasttext_epochs=getattr(config, "fasttext_epochs", 5),
            fasttext_sg=getattr(config, "fasttext_sg", 1),
        )
    elif encoder_type == "fasttext_dense":
        return FastTextDenseEncoder(
            model_path=getattr(config, "fasttext_model_path", None),
            vector_size=getattr(config, "fasttext_dim", 300),
            window=getattr(config, "fasttext_window", 5),
            min_count=getattr(config, "fasttext_min_count", 1),
            epochs=getattr(config, "fasttext_epochs", 5),
            sg=getattr(config, "fasttext_sg", 1),
            use_pretrained=getattr(config, "fasttext_use_pretrained", True),
        )
    elif encoder_type == "splade":
        return SpladeEncoder(
            model_name=getattr(config, "splade_model_name", "naver/splade_v2_max"),
            batch_size=getattr(config, "splade_batch_size", 32),
            device=getattr(config, "splade_device", "auto"),
            cache_dir=getattr(config, "splade_cache_dir", ".cache/splade"),
            max_length=getattr(config, "splade_max_length", 256),
            agg_strategy=getattr(config, "splade_agg_strategy", "max"),
        )
    elif encoder_type == "dense":
        return DenseEncoder(
            model_name=getattr(config, "st_model_name", "BAAI/bge-small-en-v1.5"),
            batch_size=getattr(config, "st_batch_size", 64),
            device=getattr(config, "st_device", "auto"),
            cache_dir=getattr(config, "st_cache_dir", ".cache/sentence_transformers"),
        )
    elif encoder_type == "hybrid":
        # Create sparse + dense sub-encoders
        sparse_type = getattr(config, "hybrid_sparse_encoder", "tfidf")
        dense_type = getattr(config, "hybrid_dense_encoder", "dense")

        # Create sub-configs with the right encoder_type
        sparse_config = config.with_overrides(encoder_type=sparse_type) if hasattr(config, "with_overrides") else config
        dense_config = config.with_overrides(encoder_type=dense_type) if hasattr(config, "with_overrides") else config

        sparse_enc = create_encoder(sparse_config)
        dense_enc = create_encoder(dense_config)

        return HybridEncoder(
            sparse_encoder=sparse_enc,
            dense_encoder=dense_enc,
            shift_dense=getattr(config, "hybrid_shift_dense", True),
            alpha=getattr(config, "hybrid_alpha", 0.5),
        )
    else:
        raise ValueError(
            f"Unknown encoder type: '{encoder_type}'. "
            f"Available: {list(_ENCODER_REGISTRY.keys()) + ['hybrid']}"
        )
