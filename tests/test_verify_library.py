"""Tests for the standalone verify() library function (NET-04).

Covers VerificationResult structure, miner aggregation,
empty response handling, type correctness, and stake-based selection.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from antigence_subnet.api.verify import VerificationResult, verify

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_metagraph(n: int = 8):
    """Build a mock metagraph with n neurons and varied stakes."""
    metagraph = MagicMock()
    metagraph.n = n
    metagraph.hotkeys = [f"hotkey-{i}" for i in range(n)]
    metagraph.axons = [MagicMock() for _ in range(n)]
    # Varied stakes so top-K selection is testable
    metagraph.S = np.array(
        [1000.0 * (i + 1) for i in range(n)], dtype=np.float32
    )
    return metagraph


def _make_mock_dendrite_with_responses(anomaly_scores, confidences, anomaly_types):
    """Build a mock dendrite returning specific response values."""
    async def _mock_call(axons, synapse, timeout=12.0, deserialize=False, **kw):
        results = []
        for idx, _ in enumerate(axons):
            resp = synapse.model_copy()
            if idx < len(anomaly_scores):
                resp.anomaly_score = anomaly_scores[idx]
                resp.confidence = confidences[idx]
                resp.anomaly_type = anomaly_types[idx]
            else:
                resp.anomaly_score = None
                resp.confidence = None
                resp.anomaly_type = None
            results.append(resp)
        return results

    dendrite = AsyncMock(side_effect=_mock_call)
    return dendrite


def _make_mock_dendrite_no_responses():
    """Build a mock dendrite where all miners return None scores."""
    async def _mock_call(axons, synapse, timeout=12.0, deserialize=False, **kw):
        results = []
        for _ in axons:
            resp = synapse.model_copy()
            resp.anomaly_score = None
            resp.confidence = None
            resp.anomaly_type = None
            results.append(resp)
        return results

    return AsyncMock(side_effect=_mock_call)


# ---------------------------------------------------------------------------
# Test 1: verify() returns VerificationResult with correct fields
# ---------------------------------------------------------------------------

class TestVerificationResultFields:
    def test_verify_returns_verification_result(self):
        metagraph = _make_mock_metagraph()
        dendrite = _make_mock_dendrite_with_responses(
            anomaly_scores=[0.7, 0.8, 0.6, 0.9, 0.5],
            confidences=[0.9, 0.85, 0.8, 0.95, 0.7],
            anomaly_types=[
                "hallucination", "hallucination", None,
                "factual_error", "hallucination",
            ],
        )

        with patch("antigence_subnet.api.verify._create_subtensor") as mock_st, \
             patch("antigence_subnet.api.verify._create_metagraph", return_value=metagraph), \
             patch("antigence_subnet.api.verify._create_dendrite", return_value=dendrite):
            mock_st.return_value = MagicMock()

            result = verify(
                prompt="What is 2+2?",
                output="2+2 is 5",
                domain="hallucination",
                subtensor_network="mock",
            )

        assert isinstance(result, VerificationResult)
        assert hasattr(result, "trust_score")
        assert hasattr(result, "confidence")
        assert hasattr(result, "anomaly_types")
        assert hasattr(result, "contributing_miners")
        assert hasattr(result, "raw_responses")


# ---------------------------------------------------------------------------
# Test 2: verify() queries top validators and aggregates responses
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_verify_aggregates_miner_responses(self):
        metagraph = _make_mock_metagraph()
        dendrite = _make_mock_dendrite_with_responses(
            anomaly_scores=[0.7, 0.8, 0.6, 0.9, 0.5],
            confidences=[0.9, 0.85, 0.8, 0.95, 0.7],
            anomaly_types=[
                "hallucination", "hallucination", None,
                "factual_error", "hallucination",
            ],
        )

        with patch("antigence_subnet.api.verify._create_subtensor") as mock_st, \
             patch("antigence_subnet.api.verify._create_metagraph", return_value=metagraph), \
             patch("antigence_subnet.api.verify._create_dendrite", return_value=dendrite):
            mock_st.return_value = MagicMock()

            result = verify(
                prompt="What is 2+2?",
                output="2+2 is 5",
                domain="hallucination",
                subtensor_network="mock",
            )

        assert result.contributing_miners > 0
        assert 0.0 <= result.trust_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.anomaly_types) > 0


# ---------------------------------------------------------------------------
# Test 3: No responding miners -> trust_score=0.5, confidence=0.0
# ---------------------------------------------------------------------------

class TestNoResponders:
    def test_verify_with_no_responses_returns_defaults(self):
        metagraph = _make_mock_metagraph()
        dendrite = _make_mock_dendrite_no_responses()

        with patch("antigence_subnet.api.verify._create_subtensor") as mock_st, \
             patch("antigence_subnet.api.verify._create_metagraph", return_value=metagraph), \
             patch("antigence_subnet.api.verify._create_dendrite", return_value=dendrite):
            mock_st.return_value = MagicMock()

            result = verify(
                prompt="test",
                output="test",
                domain="hallucination",
                subtensor_network="mock",
            )

        assert result.trust_score == 0.5
        assert result.confidence == 0.0
        assert result.contributing_miners == 0
        assert result.anomaly_types == []
        assert result.raw_responses == []


# ---------------------------------------------------------------------------
# Test 4: VerificationResult field types are correct
# ---------------------------------------------------------------------------

class TestFieldTypes:
    def test_verification_result_field_types(self):
        metagraph = _make_mock_metagraph()
        dendrite = _make_mock_dendrite_with_responses(
            anomaly_scores=[0.7, 0.8, 0.6, 0.9, 0.5],
            confidences=[0.9, 0.85, 0.8, 0.95, 0.7],
            anomaly_types=[
                "hallucination", "hallucination", None,
                "factual_error", "hallucination",
            ],
        )

        with patch("antigence_subnet.api.verify._create_subtensor") as mock_st, \
             patch("antigence_subnet.api.verify._create_metagraph", return_value=metagraph), \
             patch("antigence_subnet.api.verify._create_dendrite", return_value=dendrite):
            mock_st.return_value = MagicMock()

            result = verify(
                prompt="test",
                output="test",
                domain="hallucination",
                subtensor_network="mock",
            )

        assert isinstance(result.trust_score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.anomaly_types, list)
        assert all(isinstance(t, str) for t in result.anomaly_types)
        assert isinstance(result.contributing_miners, int)
        assert isinstance(result.raw_responses, list)
        for raw in result.raw_responses:
            assert isinstance(raw, dict)
            assert "anomaly_score" in raw
            assert "confidence" in raw
            assert "anomaly_type" in raw


# ---------------------------------------------------------------------------
# Test 5: verify() selects top validators by stake
# ---------------------------------------------------------------------------

class TestStakeSelection:
    def test_verify_selects_top_by_stake(self):
        """Top-K miners are selected by stake (metagraph.S)."""
        n = 8
        metagraph = _make_mock_metagraph(n=n)
        # Assign dramatically different stakes
        metagraph.S = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0, 300.0],
            dtype=np.float32,
        )

        queried_axons = []

        async def _tracking_dendrite(axons, synapse, timeout=12.0, deserialize=False, **kw):
            queried_axons.extend(axons)
            results = []
            for _ in axons:
                resp = synapse.model_copy()
                resp.anomaly_score = 0.7
                resp.confidence = 0.9
                resp.anomaly_type = "test"
                results.append(resp)
            return results

        dendrite = AsyncMock(side_effect=_tracking_dendrite)

        with patch("antigence_subnet.api.verify._create_subtensor") as mock_st, \
             patch("antigence_subnet.api.verify._create_metagraph", return_value=metagraph), \
             patch("antigence_subnet.api.verify._create_dendrite", return_value=dendrite):
            mock_st.return_value = MagicMock()

            verify(
                prompt="test",
                output="test",
                domain="hallucination",
                subtensor_network="mock",
                top_k=3,
            )

        # Should have queried the 3 highest-stake miners (UIDs 5, 6, 7)
        assert len(queried_axons) == 3
        expected_axons = [metagraph.axons[5], metagraph.axons[6], metagraph.axons[7]]
        for ax in expected_axons:
            assert ax in queried_axons
