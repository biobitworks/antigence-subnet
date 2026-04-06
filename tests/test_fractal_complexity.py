"""Tests for fractal complexity feature extraction and detector.

Covers the 8-feature fractal vector (Higuchi FD, Hurst exponent, Shannon entropy)
and the FractalComplexityDetector that wraps IsolationForest over those features.
"""

import numpy as np
import pytest

from antigence_subnet.miner.detectors.fractal_features import extract_fractal_features

FEATURE_NAMES = [
    "hfd_char_dist",
    "hfd_word_lengths",
    "hurst_word_lengths",
    "hurst_sentence_lengths",
    "shannon_char",
    "shannon_word",
    "shannon_bigram",
    "complexity_index",
]


class TestExtractFractalFeatures:
    def test_returns_8_element_array(self):
        result = extract_fractal_features("The quick brown fox jumps over the lazy dog")
        assert isinstance(result, np.ndarray)
        assert result.shape == (8,)

    def test_all_features_finite(self):
        result = extract_fractal_features(
            "The quick brown fox jumps over the lazy dog and more words "
            "to ensure length is sufficient for all metrics"
        )
        assert np.all(np.isfinite(result)), f"Non-finite values: {result}"

    def test_empty_string_returns_defaults(self):
        result = extract_fractal_features("")
        expected = np.array([1.5, 1.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_short_text_uses_hfd_default(self):
        result = extract_fractal_features("short text")
        # HFD needs >= 10 data points, word lengths has 2 -- should fallback
        assert result[0] == pytest.approx(1.5, abs=0.01)  # hfd_char_dist might compute (chars > 10)
        assert result[1] == pytest.approx(1.5, abs=0.01)  # hfd_word_lengths (2 words < 10)

    def test_shannon_entropy_single_char_is_zero(self):
        result = extract_fractal_features(
            "aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa"
        )
        # shannon_char for all-a string should be very low (plus spaces)
        assert result[4] < 1.0  # shannon_char near zero

    def test_shannon_entropy_diverse_text_is_higher(self):
        uniform = extract_fractal_features(
            "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        )
        repetitive = extract_fractal_features(
            "a a a a a a a a a a a a a a a a a a a a a a a a a a"
        )
        assert uniform[5] > repetitive[5]  # shannon_word: diverse > repetitive

    def test_complexity_index_formula(self):
        result = extract_fractal_features(
            "The quick brown fox jumps over the lazy dog. "
            "She sells sea shells by the sea shore. "
            "How much wood would a woodchuck chuck."
        )
        hfd = result[1]  # hfd_word_lengths
        hurst = result[2]  # hurst_word_lengths
        expected_ci = (hfd - 1.0) * (1.0 - hurst + 0.5)
        assert result[7] == pytest.approx(expected_ci, abs=1e-6)

    def test_non_ascii_text_does_not_crash(self):
        result = extract_fractal_features(
            "Unicode text: caf\u00e9 r\u00e9sum\u00e9 na\u00efve z\u00fcrich"
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (8,)
        assert np.all(np.isfinite(result))


# --- Detector tests (Task 2) ---

from antigence_subnet.miner.detectors.fractal_complexity import FractalComplexityDetector  # noqa: E402
from antigence_subnet.miner.detector import DetectionResult  # noqa: E402

# Minimal training samples for tests
TRAIN_SAMPLES = [
    {
        "prompt": "What is the capital of France?",
        "output": "The capital of France is Paris, located in the Ile-de-France region.",
    },
    {
        "prompt": "Explain photosynthesis",
        "output": (
            "Photosynthesis is the process by which plants convert sunlight, "
            "carbon dioxide, and water into glucose and oxygen."
        ),
    },
    {
        "prompt": "What is gravity?",
        "output": (
            "Gravity is a fundamental force of nature that attracts objects "
            "with mass toward each other."
        ),
    },
    {
        "prompt": "Describe the water cycle",
        "output": (
            "The water cycle involves evaporation, condensation, and precipitation, "
            "continuously circulating water through the atmosphere and Earth."
        ),
    },
    {
        "prompt": "What is DNA?",
        "output": (
            "DNA (deoxyribonucleic acid) is a molecule that carries the genetic "
            "instructions used in the growth and development of all living organisms."
        ),
    },
]


class TestFractalComplexityDetector:
    def test_domain_is_fractal(self):
        det = FractalComplexityDetector()
        assert det.domain == "fractal"

    def test_implements_base_detector(self):
        from antigence_subnet.miner.detector import BaseDetector

        det = FractalComplexityDetector()
        assert isinstance(det, BaseDetector)

    def test_fit_sets_is_fitted(self):
        det = FractalComplexityDetector()
        det.fit(TRAIN_SAMPLES)
        assert det.get_info()["is_fitted"] is True

    @pytest.mark.asyncio
    async def test_detect_returns_detection_result(self):
        det = FractalComplexityDetector()
        det.fit(TRAIN_SAMPLES)
        result = await det.detect(
            "What is the capital of France?", "Paris is the capital of France."
        )
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.anomaly_type, str)

    @pytest.mark.asyncio
    async def test_detect_score_variation(self):
        det = FractalComplexityDetector()
        det.fit(TRAIN_SAMPLES)
        r1 = await det.detect("What is 2+2?", "Four.")
        r2 = await det.detect(
            "Explain quantum mechanics",
            (
                "Quantum mechanics describes the behavior of particles at the "
                "subatomic level, where classical physics breaks down and particles "
                "exhibit wave-particle duality, superposition, and entanglement "
                "phenomena that challenge our everyday intuitions about how the "
                "physical world operates at its most fundamental level."
            ),
        )
        # Different length/complexity inputs should produce different scores
        assert r1.score != r2.score or r1.confidence != r2.confidence

    @pytest.mark.asyncio
    async def test_detect_feature_attribution(self):
        det = FractalComplexityDetector()
        det.fit(TRAIN_SAMPLES)
        result = await det.detect("Test prompt", "Test output with some words.")
        assert result.feature_attribution is not None
        assert isinstance(result.feature_attribution, dict)
        # Should have fractal feature names as keys
        assert any(
            "hfd" in k or "hurst" in k or "shannon" in k or "complexity" in k
            for k in result.feature_attribution
        )

    def test_get_info(self):
        det = FractalComplexityDetector()
        info = det.get_info()
        assert info["name"] == "FractalComplexityDetector"
        assert info["domain"] == "fractal"
        assert info["backend"] == "scikit-learn+nolds"
        assert info["is_fitted"] is False

    def test_save_load_state(self, tmp_path):
        det = FractalComplexityDetector()
        det.fit(TRAIN_SAMPLES)
        det.save_state(str(tmp_path))
        det2 = FractalComplexityDetector()
        det2.load_state(str(tmp_path))
        assert det2.get_info()["is_fitted"] is True


class TestFractalDetectorImports:
    def test_importable_from_detectors_package(self):
        from antigence_subnet.miner.detectors import FractalComplexityDetector as FC

        assert FC is not None
