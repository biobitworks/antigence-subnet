"""Sentence-transformers embedding wrapper with graceful import handling.

Provides dense semantic embeddings (384-dim from all-MiniLM-L6-v2) as an
upgrade path from TF-IDF sparse features. Falls back gracefully when
sentence-transformers is not installed.
"""

from __future__ import annotations

import numpy as np

# Graceful import of sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    _sbert_available = True
except ImportError:
    _sbert_available = False

# Module-level model cache keyed by model_name
_model_cache: dict[str, object] = {}


def is_sbert_available() -> bool:
    """Check whether sentence-transformers is installed and importable.

    Returns:
        True if sentence-transformers is available, False otherwise.
    """
    return _sbert_available


def encode_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode texts into dense semantic embeddings.

    Uses sentence-transformers SentenceTransformer models. Model instances
    are cached in ``_model_cache`` by model_name to avoid repeated loading.

    Args:
        texts: List of text strings to encode.
        model_name: HuggingFace model name for sentence-transformers.
            Defaults to all-MiniLM-L6-v2 (384-dim output).
        batch_size: Batch size for encoding. Defaults to 32.

    Returns:
        Numpy array of shape (n_texts, embedding_dim) with float32 values.

    Raises:
        RuntimeError: If sentence-transformers is not installed.
    """
    if not _sbert_available:
        raise RuntimeError(
            "sentence-transformers not installed. "
            "Install with: pip install 'antigence-subnet[sbert]'"
        )

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)

    model = _model_cache[model_name]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)
