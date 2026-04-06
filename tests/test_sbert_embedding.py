"""Tests for sentence-transformers embedding module.

Validates the sbert embedding wrapper with graceful import handling,
model caching, correct output shapes, and error behavior when
sentence-transformers is not available.
"""

import numpy as np
import pytest

# Skip entire module if sentence-transformers not installed
pytest.importorskip("sentence_transformers", reason="sbert tests require sentence-transformers")


class TestSbertEmbedding:
    """Tests for the embeddings module functions."""

    def test_is_sbert_available_returns_true(self):
        """Test 1: is_sbert_available() returns True when sentence-transformers is installed."""
        from antigence_subnet.miner.detectors.embeddings import is_sbert_available

        assert is_sbert_available() is True

    def test_encode_texts_single_text_shape(self):
        """Test 2: encode_texts(["hello world"]) returns numpy array of shape (1, 384)."""
        from antigence_subnet.miner.detectors.embeddings import encode_texts

        result = encode_texts(["hello world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384)

    def test_encode_texts_multiple_texts_shape(self):
        """Test 3: encode_texts(["a", "b", "c"]) returns shape (3, 384)."""
        from antigence_subnet.miner.detectors.embeddings import encode_texts

        result = encode_texts(["a", "b", "c"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)

    def test_encode_texts_raises_when_unavailable(self):
        """Test 4: encode_texts raises RuntimeError when sentence-transformers not importable."""
        import antigence_subnet.miner.detectors.embeddings as emb_module

        original = emb_module._sbert_available
        try:
            emb_module._sbert_available = False
            with pytest.raises(RuntimeError, match="sentence-transformers not installed"):
                emb_module.encode_texts(["test"])
        finally:
            emb_module._sbert_available = original

    def test_model_caching(self):
        """Test 5: Model caching works -- calling encode_texts twice uses same model instance."""
        import antigence_subnet.miner.detectors.embeddings as emb_module
        from antigence_subnet.miner.detectors.embeddings import encode_texts

        # Clear cache first
        emb_module._model_cache.clear()

        encode_texts(["first call"])
        encode_texts(["second call"])

        assert len(emb_module._model_cache) == 1
