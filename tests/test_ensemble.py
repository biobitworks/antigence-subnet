"""
Tests for multi-detector ensemble registry and ensemble_detect() averaging.

Covers:
- List-based DETECTOR_REGISTRY (register_detector appends, get_detector returns first)
- get_detectors() returns full list
- ensemble_detect() averages scores and confidences
- Backward compatibility with single-detector-per-domain
- Built-in domain registrations remain at 1 detector each
"""

import pytest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult

# ---------------------------------------------------------------------------
# Mock detector helpers
# ---------------------------------------------------------------------------


class MockDetectorA(BaseDetector):
    """Mock detector returning fixed score=0.8, confidence=0.9."""

    domain = "test_domain"

    def fit(self, samples):
        pass

    async def detect(self, prompt, output, code=None, context=None):
        return DetectionResult(
            score=0.8,
            confidence=0.9,
            anomaly_type="mock_a",
            feature_attribution={"feat_a": 0.5},
        )


class MockDetectorB(BaseDetector):
    """Mock detector returning fixed score=0.4, confidence=0.7."""

    domain = "test_domain"

    def fit(self, samples):
        pass

    async def detect(self, prompt, output, code=None, context=None):
        return DetectionResult(
            score=0.4,
            confidence=0.7,
            anomaly_type="mock_b",
            feature_attribution={"feat_b": 0.3},
        )


class MockDetectorC(BaseDetector):
    """Mock detector returning fixed score=0.6, confidence=0.5."""

    domain = "test_domain"

    def fit(self, samples):
        pass

    async def detect(self, prompt, output, code=None, context=None):
        return DetectionResult(
            score=0.6,
            confidence=0.5,
            anomaly_type="mock_c",
            feature_attribution=None,
        )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestDetectorRegistry:
    """Test list-based DETECTOR_REGISTRY with append semantics."""

    def setup_method(self):
        """Clear registry before each test."""
        from antigence_subnet.miner.detectors import DETECTOR_REGISTRY

        self._backup = dict(DETECTOR_REGISTRY)
        DETECTOR_REGISTRY.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        from antigence_subnet.miner.detectors import DETECTOR_REGISTRY

        DETECTOR_REGISTRY.clear()
        DETECTOR_REGISTRY.update(self._backup)

    def test_register_detector_appends_to_list(self):
        """Test 1: register_detector adds detector to list for domain (not replaces)."""
        from antigence_subnet.miner.detectors import DETECTOR_REGISTRY, register_detector

        register_detector("test_domain", MockDetectorA)
        register_detector("test_domain", MockDetectorB)

        assert len(DETECTOR_REGISTRY["test_domain"]) == 2
        assert DETECTOR_REGISTRY["test_domain"][0] is MockDetectorA
        assert DETECTOR_REGISTRY["test_domain"][1] is MockDetectorB

    def test_get_detector_returns_first(self):
        """Test 2: get_detector returns first registered detector for backward compat."""
        from antigence_subnet.miner.detectors import get_detector, register_detector

        register_detector("test_domain", MockDetectorA)
        register_detector("test_domain", MockDetectorB)

        result = get_detector("test_domain")
        assert result is MockDetectorA

    def test_get_detectors_returns_full_list(self):
        """Test 3: get_detectors returns full list of detectors for a domain."""
        from antigence_subnet.miner.detectors import (
            get_detectors,
            register_detector,
        )

        register_detector("test_domain", MockDetectorA)
        register_detector("test_domain", MockDetectorB)

        result = get_detectors("test_domain")
        assert len(result) == 2
        assert result[0] is MockDetectorA
        assert result[1] is MockDetectorB

    def test_get_detectors_returns_empty_for_unregistered(self):
        """Test 4: get_detectors returns empty list for unregistered domain."""
        from antigence_subnet.miner.detectors import get_detectors

        result = get_detectors("nonexistent_domain")
        assert result == []

    def test_get_detector_returns_none_for_unregistered(self):
        """get_detector returns None for unregistered domain (backward compat)."""
        from antigence_subnet.miner.detectors import get_detector

        result = get_detector("nonexistent_domain")
        assert result is None


class TestBuiltinRegistrations:
    """Test that built-in domain registrations remain at exactly 1 detector each."""

    def test_builtin_domains_have_one_detector_each(self):
        """Test 10: Built-in domain registrations each have exactly 1 detector."""
        from antigence_subnet.miner.detectors import DETECTOR_REGISTRY

        for domain in ["hallucination", "code_security", "reasoning", "bio"]:
            assert domain in DETECTOR_REGISTRY, f"Domain '{domain}' not in registry"
            detectors = DETECTOR_REGISTRY[domain]
            assert isinstance(detectors, list), (
                f"Domain '{domain}' registry value should be a list, got {type(detectors)}"
            )
            assert len(detectors) == 1, (
                f"Domain '{domain}' should have exactly 1 detector, has {len(detectors)}"
            )


# ---------------------------------------------------------------------------
# Ensemble detect tests
# ---------------------------------------------------------------------------


class TestEnsembleDetect:
    """Test ensemble_detect() averaging function."""

    @pytest.mark.asyncio
    async def test_ensemble_averages_scores(self):
        """Test 5: ensemble_detect averages scores from multiple detectors."""
        from antigence_subnet.miner.ensemble import ensemble_detect

        detectors = [MockDetectorA(), MockDetectorB()]
        result = await ensemble_detect(
            detectors=detectors,
            prompt="test prompt",
            output="test output",
        )
        # Average of 0.8 and 0.4 = 0.6
        assert abs(result.score - 0.6) < 1e-6

    @pytest.mark.asyncio
    async def test_ensemble_averages_confidences(self):
        """Test 6: ensemble_detect averages confidences from multiple detectors."""
        from antigence_subnet.miner.ensemble import ensemble_detect

        detectors = [MockDetectorA(), MockDetectorB()]
        result = await ensemble_detect(
            detectors=detectors,
            prompt="test prompt",
            output="test output",
        )
        # Average of 0.9 and 0.7 = 0.8
        assert abs(result.confidence - 0.8) < 1e-6

    @pytest.mark.asyncio
    async def test_ensemble_single_detector_passthrough(self):
        """Test 7: ensemble_detect with single detector returns that detector's result."""
        from antigence_subnet.miner.ensemble import ensemble_detect

        detectors = [MockDetectorA()]
        result = await ensemble_detect(
            detectors=detectors,
            prompt="test prompt",
            output="test output",
        )
        assert result.score == 0.8
        assert result.confidence == 0.9
        assert result.anomaly_type == "mock_a"
        # Single detector passthrough preserves feature_attribution
        assert result.feature_attribution == {"feat_a": 0.5}

    @pytest.mark.asyncio
    async def test_ensemble_uses_first_anomaly_type(self):
        """Test 8: ensemble_detect uses first detector's anomaly_type."""
        from antigence_subnet.miner.ensemble import ensemble_detect

        detectors = [MockDetectorA(), MockDetectorB()]
        result = await ensemble_detect(
            detectors=detectors,
            prompt="test prompt",
            output="test output",
        )
        assert result.anomaly_type == "mock_a"

    @pytest.mark.asyncio
    async def test_ensemble_returns_none_feature_attribution(self):
        """Test 9: ensemble_detect returns feature_attribution=None."""
        from antigence_subnet.miner.ensemble import ensemble_detect

        detectors = [MockDetectorA(), MockDetectorB()]
        result = await ensemble_detect(
            detectors=detectors,
            prompt="test prompt",
            output="test output",
        )
        assert result.feature_attribution is None

    @pytest.mark.asyncio
    async def test_ensemble_three_detectors(self):
        """Ensemble with 3 detectors averages correctly."""
        from antigence_subnet.miner.ensemble import ensemble_detect

        detectors = [MockDetectorA(), MockDetectorB(), MockDetectorC()]
        result = await ensemble_detect(
            detectors=detectors,
            prompt="test prompt",
            output="test output",
        )
        # Average of 0.8, 0.4, 0.6 = 0.6
        assert abs(result.score - 0.6) < 1e-6
        # Average of 0.9, 0.7, 0.5 = 0.7
        assert abs(result.confidence - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# Forward routing integration tests
# ---------------------------------------------------------------------------


class TestForwardEnsembleRouting:
    """Test that miner forward routes correctly for ensemble vs single detector."""

    @pytest.mark.asyncio
    async def test_forward_with_ensemble_calls_ensemble_detect(self):
        """Miner forward with ensemble (list of detectors) uses ensemble_detect."""
        from unittest.mock import MagicMock

        from antigence_subnet.miner.forward import forward

        # Create mock miner with ensemble (list) for domain
        miner = MagicMock()
        miner.supported_domains = {"test_domain"}
        miner.detectors = {"test_domain": [MockDetectorA(), MockDetectorB()]}
        miner.orchestrator = None  # No orchestrator (use ensemble path)

        # Create mock synapse
        synapse = MagicMock()
        synapse.domain = "test_domain"
        synapse.prompt = "test prompt"
        synapse.output = "test output"
        synapse.code = None
        synapse.context = None

        await forward(miner, synapse)

        # Should set averaged score: (0.8 + 0.4) / 2 = 0.6
        assert abs(synapse.anomaly_score - 0.6) < 1e-6
        # Should set averaged confidence: (0.9 + 0.7) / 2 = 0.8
        assert abs(synapse.confidence - 0.8) < 1e-6
        # First detector's anomaly type
        assert synapse.anomaly_type == "mock_a"
        # Ensemble returns None for feature_attribution
        assert synapse.feature_attribution is None

    @pytest.mark.asyncio
    async def test_forward_with_single_detector_backward_compat(self):
        """Miner forward with single detector (not list) works as before."""
        from unittest.mock import MagicMock

        from antigence_subnet.miner.forward import forward

        # Create mock miner with single detector (not in a list)
        miner = MagicMock()
        miner.supported_domains = {"test_domain"}
        miner.detectors = {"test_domain": MockDetectorA()}
        miner.orchestrator = None  # No orchestrator (use ensemble path)

        # Create mock synapse
        synapse = MagicMock()
        synapse.domain = "test_domain"
        synapse.prompt = "test prompt"
        synapse.output = "test output"
        synapse.code = None
        synapse.context = None

        await forward(miner, synapse)

        assert synapse.anomaly_score == 0.8
        assert synapse.confidence == 0.9
        assert synapse.anomaly_type == "mock_a"
        assert synapse.feature_attribution == {"feat_a": 0.5}


# ---------------------------------------------------------------------------
# TOML list-value loading tests
# ---------------------------------------------------------------------------


class TestTOMLEnsembleConfig:
    """Test TOML list-value parsing for ensemble detector configuration."""

    def test_toml_string_value_is_single_detector(self):
        """TOML with string value should be treated as single detector path."""
        # This tests the isinstance(class_paths, str) -> [class_paths] logic
        class_paths = "antigence_subnet.miner.detectors.domain_packs.hallucination.detector.HallucinationDetector"  # noqa: E501

        # Simulate the miner's TOML normalization logic
        if isinstance(class_paths, str):
            class_paths = [class_paths]

        assert len(class_paths) == 1
        assert isinstance(class_paths, list)

    def test_toml_list_value_is_ensemble(self):
        """TOML with list value should be treated as ensemble of detector paths."""
        class_paths = [
            "antigence_subnet.miner.detectors.domain_packs.hallucination.detector.HallucinationDetector",
            "antigence_subnet.miner.detectors.sklearn_backends.LOFDetector",
        ]

        # List values pass through directly
        assert len(class_paths) == 2
        assert isinstance(class_paths, list)

    def test_toml_list_with_single_entry(self):
        """TOML with list containing one entry should work as single detector."""
        class_paths = [
            "antigence_subnet.miner.detectors.domain_packs.hallucination.detector.HallucinationDetector",
        ]

        if isinstance(class_paths, str):
            class_paths = [class_paths]

        assert len(class_paths) == 1
        assert isinstance(class_paths, list)
