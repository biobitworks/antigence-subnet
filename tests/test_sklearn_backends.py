"""Unit tests for LOFDetector and OCSVMDetector sklearn backend detectors."""

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
# LOFDetector Tests
# ===========================================================================


class TestLOFDetector:
    """Tests for LOFDetector implementation."""

    def test_domain_is_hallucination(self):
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        assert d.domain == "hallucination"

    def test_fit_sets_is_fitted(self, normal_samples):
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        d.fit(normal_samples)
        assert d._is_fitted is True

    async def test_detect_score_in_range(self, normal_samples):
        """detect() returns DetectionResult with 0.0 <= score <= 1.0."""
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        d.fit(normal_samples)
        result = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )
        assert 0.0 <= result.score <= 1.0

    async def test_detect_confidence_in_range(self, normal_samples):
        """detect() returns DetectionResult with 0.0 <= confidence <= 1.0."""
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        d.fit(normal_samples)
        result = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )
        assert 0.0 <= result.confidence <= 1.0

    async def test_detect_anomaly_type_field(self, normal_samples):
        """anomaly_type should be 'hallucination' or 'normal'."""
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        d.fit(normal_samples)
        result = await d.detect(prompt="Test", output="Test output")
        assert result.anomaly_type in ("hallucination", "normal")

    def test_get_info_after_fit(self, normal_samples):
        """get_info() returns backend='scikit-learn' and is_fitted=True after fit."""
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        d.fit(normal_samples)
        info = d.get_info()
        assert info["backend"] == "scikit-learn"
        assert info["is_fitted"] is True

    def test_save_state_creates_file(self, normal_samples, tmp_path):
        """save_state() creates lof_state.joblib in the given directory."""
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        d.fit(normal_samples)
        d.save_state(str(tmp_path))
        assert (tmp_path / "lof_state.joblib").exists()

    async def test_load_state_restores_scores(self, normal_samples, tmp_path):
        """load_state() restores detector to produce identical scores."""
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        d.fit(normal_samples)

        result_before = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )

        d.save_state(str(tmp_path))

        d2 = LOFDetector()
        d2.load_state(str(tmp_path))
        result_after = await d2.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )

        assert abs(result_before.score - result_after.score) < 1e-6

    async def test_detect_multiple_samples_vary(self, normal_samples, all_samples):
        """Scores across different inputs should show some variation."""
        from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector

        d = LOFDetector()
        d.fit(normal_samples)
        scores = []
        for s in all_samples[:10]:
            result = await d.detect(prompt=s["prompt"], output=s["output"])
            scores.append(result.score)
        assert len(set(scores)) > 1, f"All scores identical: {scores}"


# ===========================================================================
# OCSVMDetector Tests
# ===========================================================================


class TestOCSVMDetector:
    """Tests for OCSVMDetector implementation."""

    def test_domain_is_hallucination(self):
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        assert d.domain == "hallucination"

    def test_fit_sets_is_fitted(self, normal_samples):
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        d.fit(normal_samples)
        assert d._is_fitted is True

    async def test_detect_score_in_range(self, normal_samples):
        """detect() returns DetectionResult with 0.0 <= score <= 1.0."""
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        d.fit(normal_samples)
        result = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )
        assert 0.0 <= result.score <= 1.0

    async def test_detect_confidence_in_range(self, normal_samples):
        """detect() returns DetectionResult with 0.0 <= confidence <= 1.0."""
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        d.fit(normal_samples)
        result = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )
        assert 0.0 <= result.confidence <= 1.0

    async def test_detect_anomaly_type_field(self, normal_samples):
        """anomaly_type should be 'hallucination' or 'normal'."""
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        d.fit(normal_samples)
        result = await d.detect(prompt="Test", output="Test output")
        assert result.anomaly_type in ("hallucination", "normal")

    def test_get_info_after_fit(self, normal_samples):
        """get_info() returns backend='scikit-learn' and is_fitted=True after fit."""
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        d.fit(normal_samples)
        info = d.get_info()
        assert info["backend"] == "scikit-learn"
        assert info["is_fitted"] is True

    def test_save_state_creates_file(self, normal_samples, tmp_path):
        """save_state() creates ocsvm_state.joblib in the given directory."""
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        d.fit(normal_samples)
        d.save_state(str(tmp_path))
        assert (tmp_path / "ocsvm_state.joblib").exists()

    async def test_load_state_restores_scores(self, normal_samples, tmp_path):
        """load_state() restores detector to produce identical scores."""
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        d.fit(normal_samples)

        result_before = await d.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )

        d.save_state(str(tmp_path))

        d2 = OCSVMDetector()
        d2.load_state(str(tmp_path))
        result_after = await d2.detect(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
        )

        assert abs(result_before.score - result_after.score) < 1e-6

    async def test_detect_multiple_samples_vary(self, normal_samples, all_samples):
        """Scores across different inputs should show some variation."""
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        d = OCSVMDetector()
        d.fit(normal_samples)
        scores = []
        for s in all_samples[:10]:
            result = await d.detect(prompt=s["prompt"], output=s["output"])
            scores.append(result.score)
        assert len(set(scores)) > 1, f"All scores identical: {scores}"


# ===========================================================================
# Import Tests
# ===========================================================================


class TestSklearnBackendsImport:
    """Test that LOFDetector and OCSVMDetector are importable from the detectors package."""

    def test_lof_importable_from_detectors(self):
        from antigence_subnet.miner.detectors import LOFDetector

        assert LOFDetector is not None

    def test_ocsvm_importable_from_detectors(self):
        from antigence_subnet.miner.detectors import OCSVMDetector

        assert OCSVMDetector is not None
