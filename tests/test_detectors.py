"""Unit tests for BaseDetector extensions, shared features, and detector implementations."""

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: Load seed dataset for training/test fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "hallucination"


def _load_samples():
    """Load samples.json seed data."""
    with open(DATA_DIR / "samples.json") as f:
        return json.load(f)["samples"]


def _load_manifest():
    """Load manifest.json ground truth."""
    with open(DATA_DIR / "manifest.json") as f:
        return json.load(f)


@pytest.fixture
def all_samples():
    return _load_samples()


@pytest.fixture
def manifest():
    return _load_manifest()


@pytest.fixture
def normal_samples(all_samples, manifest):
    """Return only normal-labeled samples for training."""
    return [s for s in all_samples if manifest[s["id"]]["ground_truth_label"] == "normal"]


@pytest.fixture
def anomalous_samples(all_samples, manifest):
    """Return only anomalous-labeled samples for testing."""
    return [s for s in all_samples if manifest[s["id"]]["ground_truth_label"] == "anomalous"]


# ===========================================================================
# BaseDetector Extension Tests
# ===========================================================================

class TestBaseDetectorExtension:
    """Tests for BaseDetector fit() and get_info() methods."""

    def test_subclass_without_fit_raises_type_error(self):
        """A subclass implementing only detect() should fail because fit() is abstract."""
        from antigence_subnet.miner.detector import BaseDetector, DetectionResult

        class IncompleteDetector(BaseDetector):
            domain = "test"

            async def detect(self, prompt, output, code=None, context=None):
                return DetectionResult(score=0.5, confidence=0.5, anomaly_type="test")

        with pytest.raises(TypeError):
            IncompleteDetector()

    def test_subclass_with_fit_and_detect_instantiates(self):
        """A subclass implementing both fit() and detect() should instantiate."""
        from antigence_subnet.miner.detector import BaseDetector, DetectionResult

        class CompleteDetector(BaseDetector):
            domain = "test"

            def fit(self, samples):
                pass

            async def detect(self, prompt, output, code=None, context=None):
                return DetectionResult(score=0.5, confidence=0.5, anomaly_type="test")

        d = CompleteDetector()
        assert d.domain == "test"

    def test_get_info_returns_expected_keys(self):
        """get_info() should return dict with name, domain, version, backend, is_fitted."""
        from antigence_subnet.miner.detector import BaseDetector, DetectionResult

        class InfoDetector(BaseDetector):
            domain = "test"

            def fit(self, samples):
                pass

            async def detect(self, prompt, output, code=None, context=None):
                return DetectionResult(score=0.5, confidence=0.5, anomaly_type="test")

        d = InfoDetector()
        info = d.get_info()
        assert set(info.keys()) == {"name", "domain", "version", "backend", "is_fitted"}

    def test_default_get_info_is_not_fitted(self):
        """Default get_info() should report is_fitted=False."""
        from antigence_subnet.miner.detector import BaseDetector, DetectionResult

        class DefaultInfoDetector(BaseDetector):
            domain = "test"

            def fit(self, samples):
                pass

            async def detect(self, prompt, output, code=None, context=None):
                return DetectionResult(score=0.5, confidence=0.5, anomaly_type="test")

        d = DefaultInfoDetector()
        info = d.get_info()
        assert info["is_fitted"] is False


# ===========================================================================
# Shared Features Module Tests
# ===========================================================================

class TestSharedFeatures:
    """Tests for antigence_subnet.miner.detectors.features module."""

    def test_create_vectorizer_returns_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        from antigence_subnet.miner.detectors.features import create_vectorizer

        v = create_vectorizer()
        assert isinstance(v, TfidfVectorizer)

    def test_create_vectorizer_default_params(self):
        from antigence_subnet.miner.detectors.features import create_vectorizer

        v = create_vectorizer()
        assert v.max_features == 5000
        assert v.ngram_range == (1, 2)

    def test_samples_to_texts_converts_dicts(self):
        from antigence_subnet.miner.detectors.features import samples_to_texts

        samples = [
            {"prompt": "Hello", "output": "World"},
            {"prompt": "Foo", "output": "Bar"},
        ]
        texts = samples_to_texts(samples)
        assert texts == ["Hello World", "Foo Bar"]

    def test_samples_to_texts_handles_missing_keys(self):
        from antigence_subnet.miner.detectors.features import samples_to_texts

        samples = [{"prompt": "Hello"}, {"output": "World"}, {}]
        texts = samples_to_texts(samples)
        assert texts == ["Hello ", " World", " "]


# ===========================================================================
# IsolationForest Detector Tests
# ===========================================================================

class TestIsolationForestDetector:
    """Tests for IsolationForestDetector implementation."""

    def test_domain_is_hallucination(self):
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        assert d.domain == "hallucination"

    def test_fit_sets_is_fitted(self, normal_samples):
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        assert d._is_fitted is True

    def test_fit_stores_sorted_baseline(self, normal_samples):
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        assert d._baseline_scores_sorted is not None
        assert len(d._baseline_scores_sorted) == len(normal_samples)
        # Verify sorted
        assert np.all(d._baseline_scores_sorted[:-1] <= d._baseline_scores_sorted[1:])

    async def test_detect_produces_valid_score(self, normal_samples):
        """detect() on any input produces a valid anomaly_score in [0, 1]."""
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        result = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )
        assert 0.0 <= result.score <= 1.0

    async def test_detect_normal_training_sample_produces_score(self, normal_samples):
        """detect() on a training sample produces a valid score."""
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        # Use a normal sample from training data
        s = normal_samples[0]
        result = await d.detect(prompt=s["prompt"], output=s["output"])
        assert 0.0 <= result.score <= 1.0

    async def test_detect_returns_complete_result(self, normal_samples):
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        result = await d.detect(prompt="Test prompt", output="Test output")
        assert isinstance(result, DetectionResult)
        assert result.score is not None
        assert result.confidence is not None
        assert result.anomaly_type is not None

    async def test_anomaly_score_in_range(self, normal_samples):
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        result = await d.detect(prompt="Test", output="Output")
        assert 0.0 <= result.score <= 1.0

    async def test_confidence_in_range(self, normal_samples):
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        result = await d.detect(prompt="Test", output="Output")
        assert 0.0 <= result.confidence <= 1.0

    def test_get_info_after_fit(self, normal_samples):
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        info = d.get_info()
        assert info["backend"] == "scikit-learn"
        assert info["is_fitted"] is True

    def test_save_state_creates_file(self, normal_samples, tmp_path):
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        d.save_state(str(tmp_path))
        assert (tmp_path / "iforest_state.joblib").exists()

    async def test_load_state_restores_scores(self, normal_samples, tmp_path):
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)

        # Get score before save
        result_before = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )

        d.save_state(str(tmp_path))

        # Load into new instance
        d2 = IsolationForestDetector()
        d2.load_state(str(tmp_path))
        result_after = await d2.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )

        assert abs(result_before.score - result_after.score) < 1e-6

    async def test_detect_anomaly_type_field(self, normal_samples):
        """anomaly_type should be 'hallucination' or 'normal' based on score."""
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        result = await d.detect(prompt="Test", output="Test output")
        assert result.anomaly_type in ("hallucination", "normal")

    async def test_detect_multiple_samples_vary(self, normal_samples, all_samples):
        """Scores across different inputs should show some variation."""
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

        d = IsolationForestDetector()
        d.fit(normal_samples)
        scores = []
        for s in all_samples[:10]:
            result = await d.detect(prompt=s["prompt"], output=s["output"])
            scores.append(result.score)
        # At least some variation in scores (not all identical)
        assert len(set(scores)) > 1, f"All scores identical: {scores}"


# ===========================================================================
# Autoencoder Detector Tests
# ===========================================================================

_torch_available = True
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _torch_available = False


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
class TestAutoencoderDetector:
    """Tests for AutoencoderDetector implementation."""

    def test_domain_is_hallucination(self):
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector()
        assert d.domain == "hallucination"

    def test_fit_sets_is_fitted(self, normal_samples):
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector(epochs=20)
        d.fit(normal_samples)
        assert d._is_fitted is True

    def test_fit_stores_sorted_baseline_errors(self, normal_samples):
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector(epochs=20)
        d.fit(normal_samples)
        assert d._baseline_errors_sorted is not None
        assert len(d._baseline_errors_sorted) == len(normal_samples)
        # Verify sorted
        assert np.all(d._baseline_errors_sorted[:-1] <= d._baseline_errors_sorted[1:])

    async def test_detect_returns_valid_result(self, normal_samples):
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector(epochs=20)
        d.fit(normal_samples)
        result = await d.detect(prompt="Test prompt", output="Test output")
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0

    def test_get_info_after_fit(self, normal_samples):
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector(epochs=20)
        d.fit(normal_samples)
        info = d.get_info()
        assert info["backend"] == "pytorch"
        assert info["is_fitted"] is True

    def test_save_state_creates_files(self, normal_samples, tmp_path):
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector(epochs=20)
        d.fit(normal_samples)
        d.save_state(str(tmp_path))
        assert (tmp_path / "ae_vectorizer.joblib").exists()
        assert (tmp_path / "ae_state.pt").exists()

    async def test_load_state_restores_scores(self, normal_samples, tmp_path):
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector(epochs=20)
        d.fit(normal_samples)

        result_before = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )

        d.save_state(str(tmp_path))

        d2 = AutoencoderDetector(epochs=20)
        d2.load_state(str(tmp_path))
        result_after = await d2.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )

        assert abs(result_before.score - result_after.score) < 1e-6

    def test_defaults_to_cpu(self):
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector()
        assert d.device == "cpu"

    async def test_detect_anomaly_type_field(self, normal_samples):
        """anomaly_type should be 'hallucination' or 'normal' based on score."""
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector(epochs=20)
        d.fit(normal_samples)
        result = await d.detect(prompt="Test", output="Test output")
        assert result.anomaly_type in ("hallucination", "normal")

    async def test_detect_confidence_in_range(self, normal_samples):
        """Confidence should be in [0, 1]."""
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector

        d = AutoencoderDetector(epochs=20)
        d.fit(normal_samples)
        result = await d.detect(prompt="Test", output="Test output")
        assert 0.0 <= result.confidence <= 1.0
