"""
Tests for domain routing in miner forward (PROTO-04).

Verifies correct detector dispatch, unknown domain rejection,
multiple domain routing, and detector registry operations.
"""

from unittest.mock import MagicMock

import pytest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors import (
    DETECTOR_REGISTRY,
    get_detector,
    register_detector,
)
from antigence_subnet.miner.forward import forward as miner_forward
from antigence_subnet.protocol import (
    KNOWN_DOMAINS,
    VerificationSynapse,
)


class StubDetector(BaseDetector):
    """Stub detector returning fixed scores for testing."""

    def __init__(self, domain: str, score: float = 0.7, confidence: float = 0.9):
        self.domain = domain
        self._score = score
        self._confidence = confidence

    def fit(self, samples: list[dict]) -> None:
        pass

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        return DetectionResult(
            score=self._score,
            confidence=self._confidence,
            anomaly_type=f"stub_{self.domain}",
            feature_attribution={f"{self.domain}_feature": self._score},
        )


def _make_mock_miner(detectors: dict[str, BaseDetector]):
    """Create a mock miner object with detectors and supported_domains."""
    miner = MagicMock()
    miner.detectors = detectors
    miner.supported_domains = set(detectors.keys())
    miner.orchestrator = None  # No orchestrator by default (use ensemble path)
    return miner


@pytest.fixture
def hallucination_synapse():
    """Synapse for hallucination domain."""
    return VerificationSynapse(
        prompt="What is 2+2?",
        output="2+2 is 5",
        domain="hallucination",
    )


@pytest.fixture
def code_security_synapse():
    """Synapse for code_security domain."""
    return VerificationSynapse(
        prompt="Review this code",
        output="import os; os.system('rm -rf /')",
        domain="code_security",
        code="import os; os.system('rm -rf /')",
    )


@pytest.fixture
def unknown_domain_synapse():
    """Synapse for an unknown domain."""
    return VerificationSynapse(
        prompt="Test prompt",
        output="Test output",
        domain="unknown_domain",
    )


@pytest.mark.asyncio
async def test_route_known_domain(hallucination_synapse):
    """Test that a known domain routes to the correct detector."""
    detector = StubDetector(domain="hallucination", score=0.85, confidence=0.92)
    miner = _make_mock_miner({"hallucination": detector})

    result = await miner_forward(miner, hallucination_synapse)

    assert result.anomaly_score == 0.85
    assert result.confidence == 0.92
    assert result.anomaly_type == "stub_hallucination"
    assert result.feature_attribution == {"hallucination_feature": 0.85}


@pytest.mark.asyncio
async def test_route_unknown_domain_rejected(unknown_domain_synapse):
    """Test that an unknown domain is rejected with status 400 (D-07)."""
    detector = StubDetector(domain="hallucination")
    miner = _make_mock_miner({"hallucination": detector})

    result = await miner_forward(miner, unknown_domain_synapse)

    assert result.axon.status_code == 400
    assert "Unsupported domain" in result.axon.status_message
    assert "unknown_domain" in result.axon.status_message
    # Response fields should remain None (not populated)
    assert result.anomaly_score is None


@pytest.mark.asyncio
async def test_route_multiple_domains(hallucination_synapse, code_security_synapse):
    """Test routing to correct detector for multiple domains."""
    hallucination_det = StubDetector(domain="hallucination", score=0.3, confidence=0.7)
    code_det = StubDetector(domain="code_security", score=0.9, confidence=0.95)
    miner = _make_mock_miner({
        "hallucination": hallucination_det,
        "code_security": code_det,
    })

    # Route hallucination
    result_h = await miner_forward(miner, hallucination_synapse)
    assert result_h.anomaly_score == 0.3
    assert result_h.anomaly_type == "stub_hallucination"

    # Route code_security
    result_c = await miner_forward(miner, code_security_synapse)
    assert result_c.anomaly_score == 0.9
    assert result_c.anomaly_type == "stub_code_security"


def test_all_v1_domains_defined():
    """Test that all v1 domains are defined (D-05)."""
    assert {"hallucination", "code_security", "reasoning", "bio"} == KNOWN_DOMAINS


def test_detector_registry_operations():
    """Test register_detector and get_detector work correctly."""
    # Clear any previous registrations
    original_registry = DETECTOR_REGISTRY.copy()
    DETECTOR_REGISTRY.clear()

    try:
        # Register a detector
        register_detector("test_domain", StubDetector)
        assert get_detector("test_domain") is StubDetector

        # Non-registered domain returns None
        assert get_detector("nonexistent") is None
    finally:
        # Restore original registry
        DETECTOR_REGISTRY.clear()
        DETECTOR_REGISTRY.update(original_registry)


def test_detection_result_dataclass():
    """Test DetectionResult dataclass has expected fields."""
    result = DetectionResult(score=0.7, confidence=0.9, anomaly_type="test")
    assert result.score == 0.7
    assert result.confidence == 0.9
    assert result.anomaly_type == "test"
    assert result.feature_attribution is None

    # With feature attribution
    result_with_features = DetectionResult(
        score=0.5,
        confidence=0.8,
        anomaly_type="test_feat",
        feature_attribution={"feat1": 0.5, "feat2": 0.3},
    )
    assert result_with_features.feature_attribution == {"feat1": 0.5, "feat2": 0.3}
