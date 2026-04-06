"""
Tests for miner forward handler resilience (RESIL-03).

Verifies that the miner handles malformed Synapse inputs and detector
exceptions gracefully -- never crashing from a single bad request, and
recovering for subsequent valid requests.
"""

from unittest.mock import AsyncMock, MagicMock

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.forward import forward as miner_forward
from antigence_subnet.protocol import VerificationSynapse


def _make_miner_stub(detector_mock=None):
    """Create a minimal miner stub with a single hallucination detector."""
    miner = MagicMock()
    miner.supported_domains = {"hallucination"}

    if detector_mock is None:
        detector_mock = AsyncMock()
        detector_mock.detect.return_value = DetectionResult(
            score=0.85,
            confidence=0.9,
            anomaly_type="factual_error",
            feature_attribution={"city_name": 0.95},
        )

    miner.detectors = {"hallucination": detector_mock}
    miner.orchestrator = None  # No orchestrator (use ensemble path)
    return miner


async def test_miner_forward_empty_prompt():
    """When synapse.prompt is empty, miner returns 400 with 'empty prompt'."""
    miner = _make_miner_stub()
    synapse = VerificationSynapse(
        prompt="",
        output="Some output text",
        domain="hallucination",
    )

    result = await miner_forward(miner, synapse)

    assert result.axon.status_code == 400
    assert "empty prompt" in result.axon.status_message.lower()


async def test_miner_forward_empty_output():
    """When synapse.output is empty, miner returns 400 with 'empty output'."""
    miner = _make_miner_stub()
    synapse = VerificationSynapse(
        prompt="What is the capital of France?",
        output="",
        domain="hallucination",
    )

    result = await miner_forward(miner, synapse)

    assert result.axon.status_code == 400
    assert "empty output" in result.axon.status_message.lower()


async def test_miner_forward_whitespace_only_prompt():
    """When synapse.prompt is whitespace-only, miner returns 400."""
    miner = _make_miner_stub()
    synapse = VerificationSynapse(
        prompt="   \t\n  ",
        output="Some output text",
        domain="hallucination",
    )

    result = await miner_forward(miner, synapse)

    assert result.axon.status_code == 400
    assert "empty prompt" in result.axon.status_message.lower()


async def test_miner_forward_detector_exception():
    """When detector.detect() raises RuntimeError, miner returns 500."""
    detector_mock = AsyncMock()
    detector_mock.detect.side_effect = RuntimeError("model crashed")
    miner = _make_miner_stub(detector_mock=detector_mock)

    synapse = VerificationSynapse(
        prompt="What is the capital of France?",
        output="The capital of France is Berlin.",
        domain="hallucination",
    )

    result = await miner_forward(miner, synapse)

    assert result.axon.status_code == 500
    assert "detection error" in result.axon.status_message.lower()
    assert result.anomaly_score is None


async def test_miner_forward_detector_generic_exception():
    """When detector.detect() raises any Exception subclass, miner catches it."""
    detector_mock = AsyncMock()
    detector_mock.detect.side_effect = ValueError("unexpected value")
    miner = _make_miner_stub(detector_mock=detector_mock)

    synapse = VerificationSynapse(
        prompt="What is the capital of France?",
        output="The capital of France is Berlin.",
        domain="hallucination",
    )

    result = await miner_forward(miner, synapse)

    assert result.axon.status_code == 500
    assert "detection error" in result.axon.status_message.lower()
    assert result.anomaly_score is None


async def test_miner_forward_recovery_after_error():
    """After a failed detect() call, the next valid request succeeds."""
    detector_mock = AsyncMock()
    # First call raises, second call succeeds
    detector_mock.detect.side_effect = [
        RuntimeError("model crashed"),
        DetectionResult(
            score=0.75,
            confidence=0.88,
            anomaly_type="factual_error",
            feature_attribution=None,
        ),
    ]
    miner = _make_miner_stub(detector_mock=detector_mock)

    # First request -- should fail gracefully
    bad_synapse = VerificationSynapse(
        prompt="What is 2+2?",
        output="2+2 is 5",
        domain="hallucination",
    )
    result1 = await miner_forward(miner, bad_synapse)
    assert result1.axon.status_code == 500

    # Second request -- should succeed normally
    good_synapse = VerificationSynapse(
        prompt="What is the capital of France?",
        output="The capital of France is Berlin.",
        domain="hallucination",
    )
    result2 = await miner_forward(miner, good_synapse)

    assert result2.anomaly_score == 0.75
    assert result2.confidence == 0.88
    assert result2.anomaly_type == "factual_error"
