"""Model manager for SLM embeddings and semantic similarity scoring.

Provides a unified ``embed()``/``score()`` API for immune cells. Uses
sentence-transformers for embedding generation with lazy initialization --
no model download or loading occurs until the first ``embed()`` or ``score()``
call, so miners who don't use SLM cells pay zero startup cost.

Design decisions:
- D-01: Lazy init -- ModelManager.__init__ sets _model = None
- D-02: Uses sentence-transformers all-MiniLM-L6-v2 (384-dim)
- D-04: CPU fallback via torch.cuda.is_available() auto-detection
- D-05: embed(text) -> ndarray(384,) float32; score(a, b) -> float [0, 1]
- D-06: Reuses existing sentence-transformers dependency
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from antigence_subnet.miner.orchestrator.config import ModelConfig

logger = logging.getLogger(__name__)

# Graceful import of sentence-transformers (same pattern as embeddings.py)
try:
    from sentence_transformers import SentenceTransformer

    _sbert_available = True
except ImportError:
    _sbert_available = False

# Module-level model cache keyed by model_name for singleton instances.
# Multiple ModelManager instances with the same model_name share the
# underlying SentenceTransformer object.
_model_cache: dict[str, object] = {}


def _resolve_device(device: str) -> str:
    """Resolve device string to actual device.

    Args:
        device: "auto", "cpu", or "cuda".

    Returns:
        Resolved device string ("cpu" or "cuda").
    """
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class ModelManager:
    """Manages SLM model lifecycle: download, cache, embed, score.

    Lazy initialization: the model is not downloaded or loaded until the
    first ``embed()`` or ``score()`` call. This ensures miners who don't
    use SLM-based immune cells pay zero startup cost.

    The underlying ``SentenceTransformer`` instance is cached at module
    level keyed by model_name, so multiple ``ModelManager`` instances with
    the same config share the same model object.

    Args:
        config: ModelConfig with model_name, cache_dir, device settings.
            If None, uses default ModelConfig().

    Example::

        mgr = ModelManager()
        vec = mgr.embed("some text")        # shape (384,), float32
        sim = mgr.score("prompt", "output")  # float in [0.0, 1.0]
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        if config is None:
            from antigence_subnet.miner.orchestrator.config import ModelConfig

            config = ModelConfig()
        self._config = config
        self._model: object | None = None
        self._device = _resolve_device(config.device)

    def _ensure_loaded(self) -> None:
        """Load the model if not already loaded.

        Checks module-level ``_model_cache`` first. If the model is not
        cached, creates a new ``SentenceTransformer`` and stores it.

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        if self._model is not None:
            return

        if not _sbert_available:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install 'antigence-subnet[sbert]'"
            )

        model_name = self._config.model_name
        if model_name not in _model_cache:
            logger.info(
                "Loading model '%s' on device '%s'",
                model_name,
                self._device,
            )
            kwargs: dict = {
                "model_name_or_path": model_name,
                "device": self._device,
            }
            if self._config.cache_dir is not None:
                kwargs["cache_folder"] = self._config.cache_dir
            _model_cache[model_name] = SentenceTransformer(**kwargs)

        self._model = _model_cache[model_name]

    def is_available(self) -> bool:
        """Check whether sentence-transformers is installed.

        This is a class-level check that does NOT trigger model loading.

        Returns:
            True if sentence-transformers is importable, False otherwise.
        """
        return _sbert_available

    def embed(self, text: str) -> np.ndarray:
        """Generate a dense embedding for the given text.

        Args:
            text: Input text to embed.

        Returns:
            1-D numpy array of shape ``(384,)`` with dtype float32
            (for the default all-MiniLM-L6-v2 model).

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        self._ensure_loaded()
        result = self._model.encode([text], show_progress_bar=False)[0]
        return np.asarray(result, dtype=np.float32)

    def score(self, prompt: str, output: str) -> float:
        """Compute semantic similarity between prompt and output.

        Uses cosine similarity of the embeddings, clipped to [0.0, 1.0].

        Args:
            prompt: The input prompt text.
            output: The generated output text.

        Returns:
            Float in [0.0, 1.0] representing cosine similarity.

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        self._ensure_loaded()
        embeddings = self._model.encode(
            [prompt, output], show_progress_bar=False
        )
        a = np.asarray(embeddings[0], dtype=np.float32)
        b = np.asarray(embeddings[1], dtype=np.float32)

        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = dot / (norm_a * norm_b)
        return float(np.clip(similarity, 0.0, 1.0))

    @property
    def loaded(self) -> bool:
        """Whether the underlying model has been loaded.

        Returns:
            True if the model is loaded, False otherwise.
        """
        return self._model is not None
