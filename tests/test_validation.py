"""Tests for structural synapse validation module.

Covers NET-07: Synapse validation inspecting miner responses for
structural issues before scoring.
"""

from types import SimpleNamespace

from antigence_subnet.validator.validation import (
    KNOWN_ANOMALY_TYPES,
    validate_response,
)


class TestValidateResponse:
    """Tests for validate_response function."""

    def _make_response(self, anomaly_score=None, anomaly_type=None, confidence=None):
        """Create a mock response object."""
        return SimpleNamespace(
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            confidence=confidence,
        )

    def test_valid_response_passes(self):
        """A response with valid anomaly_score and anomaly_type passes."""
        response = self._make_response(anomaly_score=0.8, anomaly_type="factual_error")
        is_valid, reason = validate_response(response)
        assert is_valid is True
        assert reason == ""

    def test_none_anomaly_score_rejected(self):
        """A response with anomaly_score=None is rejected."""
        response = self._make_response(anomaly_score=None, anomaly_type="factual_error")
        is_valid, reason = validate_response(response)
        assert is_valid is False
        assert reason == "missing_anomaly_score"

    def test_out_of_range_score_rejected(self):
        """A response with anomaly_score > 1.0 is rejected."""
        response = self._make_response(anomaly_score=1.5, anomaly_type="factual_error")
        is_valid, reason = validate_response(response)
        assert is_valid is False
        assert "anomaly_score_out_of_range" in reason

    def test_negative_score_rejected(self):
        """A response with anomaly_score < 0.0 is rejected."""
        response = self._make_response(anomaly_score=-0.1, anomaly_type="factual_error")
        is_valid, reason = validate_response(response)
        assert is_valid is False
        assert "anomaly_score_out_of_range" in reason

    def test_unknown_anomaly_type_rejected(self):
        """A response with an unknown anomaly_type is rejected."""
        response = self._make_response(anomaly_score=0.8, anomaly_type="made_up_type")
        is_valid, reason = validate_response(response)
        assert is_valid is False
        assert "unknown_anomaly_type" in reason

    def test_none_anomaly_type_valid(self):
        """A response with anomaly_type=None passes (no anomaly detected is valid)."""
        response = self._make_response(anomaly_score=0.2, anomaly_type=None)
        is_valid, reason = validate_response(response)
        assert is_valid is True
        assert reason == ""

    def test_rejection_logged(self, mocker):
        """When a response is rejected, bt.logging.warning is called."""
        mock_warning = mocker.patch("bittensor.logging.warning")
        response = self._make_response(anomaly_score=None)
        validate_response(response)
        mock_warning.assert_called_once()
        assert "missing_anomaly_score" in mock_warning.call_args[0][0]

    def test_boundary_score_zero_valid(self):
        """anomaly_score=0.0 is valid (edge of range)."""
        response = self._make_response(anomaly_score=0.0)
        is_valid, reason = validate_response(response)
        assert is_valid is True

    def test_boundary_score_one_valid(self):
        """anomaly_score=1.0 is valid (edge of range)."""
        response = self._make_response(anomaly_score=1.0)
        is_valid, reason = validate_response(response)
        assert is_valid is True

    def test_mock_anomaly_type_valid(self):
        """mock_anomaly type is valid (for testing)."""
        response = self._make_response(anomaly_score=0.5, anomaly_type="mock_anomaly")
        is_valid, reason = validate_response(response)
        assert is_valid is True


class TestKnownAnomalyTypes:
    """Tests for KNOWN_ANOMALY_TYPES constant."""

    def test_is_frozenset(self):
        assert isinstance(KNOWN_ANOMALY_TYPES, frozenset)

    def test_contains_expected_types(self):
        expected = {
            "factual_error", "fabricated_citation", "unsupported_claim",
            "sql_injection", "xss", "code_backdoor", "buffer_overflow",
            "logic_inconsistency", "constraint_violation",
            "data_anomaly", "pipeline_error",
            "mock_anomaly", None,
        }
        assert expected.issubset(KNOWN_ANOMALY_TYPES)
