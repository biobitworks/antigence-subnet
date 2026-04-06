"""Tests for weight utility functions.

Covers RWRD-07: EMA score tracking with weight setting via subtensor.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from antigence_subnet.base.utils.weight_utils import (
    U16_MAX,
    U32_MAX,
    convert_weights_and_uids_for_emit,
    normalize_max_weight,
    process_weights_for_netuid,
)


class TestNormalizeMaxWeight:
    """Tests for normalize_max_weight function."""

    def test_normalize_max_weight_sums_to_one(self):
        """Output sums to 1.0."""
        x = np.array([0.3, 0.5, 0.2, 0.1], dtype=np.float32)
        result = normalize_max_weight(x, limit=0.4)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)

    def test_normalize_max_weight_respects_limit(self):
        """No single weight exceeds the limit parameter."""
        x = np.array([1.0, 0.1, 0.1, 0.1], dtype=np.float32)
        limit = 0.4
        result = normalize_max_weight(x, limit=limit)
        assert result.max() <= limit + 1e-6

    def test_normalize_all_zeros_returns_uniform(self):
        """All-zero input returns uniform distribution."""
        x = np.zeros(5, dtype=np.float32)
        result = normalize_max_weight(x, limit=0.5)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.allclose(result, 1.0 / 5.0)

    def test_normalize_single_element(self):
        """Single element normalizes to 1.0."""
        x = np.array([3.0], dtype=np.float32)
        result = normalize_max_weight(x, limit=1.0)
        assert result[0] == pytest.approx(1.0)

    def test_normalize_preserves_relative_order(self):
        """Larger weights remain larger after normalization."""
        x = np.array([0.5, 0.3, 0.1], dtype=np.float32)
        result = normalize_max_weight(x, limit=0.5)
        assert result[0] >= result[1] >= result[2]


class TestConvertWeightsAndUidsForEmit:
    """Tests for convert_weights_and_uids_for_emit function."""

    def test_convert_weights_uids_filters_zeros(self):
        """Zero weights are filtered from output."""
        uids = np.array([0, 1, 2, 3])
        weights = np.array([0.5, 0.0, 0.3, 0.0])
        result_uids, result_weights = convert_weights_and_uids_for_emit(uids, weights)
        assert 1 not in result_uids
        assert 3 not in result_uids
        assert len(result_uids) == 2

    def test_convert_weights_uids_uint16_range(self):
        """All output weight values are in [0, 65535]."""
        uids = np.array([0, 1, 2])
        weights = np.array([0.5, 0.3, 0.2])
        _, result_weights = convert_weights_and_uids_for_emit(uids, weights)
        for w in result_weights:
            assert 0 <= w <= 65535

    def test_convert_all_zeros_returns_empty(self):
        """All-zero weights return empty lists."""
        uids = np.array([0, 1, 2])
        weights = np.array([0.0, 0.0, 0.0])
        result_uids, result_weights = convert_weights_and_uids_for_emit(uids, weights)
        assert result_uids == []
        assert result_weights == []

    def test_convert_negative_raises(self):
        """Negative weights raise ValueError."""
        uids = np.array([0, 1])
        weights = np.array([0.5, -0.1])
        with pytest.raises(ValueError, match="negative"):
            convert_weights_and_uids_for_emit(uids, weights)

    def test_convert_max_weight_gets_u16_max(self):
        """The maximum weight value maps to U16_MAX (65535)."""
        uids = np.array([0, 1])
        weights = np.array([1.0, 0.5])
        _, result_weights = convert_weights_and_uids_for_emit(uids, weights)
        assert result_weights[0] == U16_MAX


class TestProcessWeightsForNetuid:
    """Tests for process_weights_for_netuid function."""

    def _mock_subtensor(self, min_allowed_weights=1, max_weight_limit=0.5):
        """Create a mock subtensor with configurable chain parameters."""
        def _min_allowed_weights(netuid):
            return min_allowed_weights

        def _max_weight_limit(netuid):
            return max_weight_limit

        return SimpleNamespace(
            min_allowed_weights=_min_allowed_weights,
            max_weight_limit=_max_weight_limit,
        )

    def _mock_metagraph(self, n=16):
        """Create a mock metagraph."""
        return SimpleNamespace(n=n)

    def test_process_weights_for_netuid_returns_normalized(self):
        """Full pipeline returns normalized weights that sum to ~1.0."""
        uids = np.array([0, 1, 2, 3])
        weights = np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float32)
        subtensor = self._mock_subtensor(min_allowed_weights=1, max_weight_limit=0.5)
        metagraph = self._mock_metagraph(n=16)

        result_uids, result_weights = process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=1,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        assert result_weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_process_weights_respects_max_weight_limit(self):
        """No weight exceeds max_weight_limit after processing."""
        uids = np.array([0, 1, 2])
        weights = np.array([0.9, 0.05, 0.05], dtype=np.float32)
        max_limit = 0.5
        subtensor = self._mock_subtensor(min_allowed_weights=1, max_weight_limit=max_limit)
        metagraph = self._mock_metagraph(n=16)

        _, result_weights = process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=1,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        assert result_weights.max() <= max_limit + 1e-6

    def test_process_weights_all_zeros_returns_uniform(self):
        """All-zero weights produce uniform distribution."""
        uids = np.array([0, 1, 2])
        weights = np.zeros(3, dtype=np.float32)
        subtensor = self._mock_subtensor(min_allowed_weights=1, max_weight_limit=0.5)
        metagraph = self._mock_metagraph(n=16)

        _, result_weights = process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=1,
            subtensor=subtensor,
            metagraph=metagraph,
        )

        assert result_weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_process_weights_with_exclude_quantile(self):
        """Exclude quantile removes lowest weights while maintaining constraints."""
        uids = np.array([0, 1, 2, 3, 4])
        weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05], dtype=np.float32)
        subtensor = self._mock_subtensor(min_allowed_weights=1, max_weight_limit=0.5)
        metagraph = self._mock_metagraph(n=16)

        result_uids, result_weights = process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=1,
            subtensor=subtensor,
            metagraph=metagraph,
            exclude_quantile=10000,  # Some quantile
        )

        assert result_weights.sum() == pytest.approx(1.0, abs=1e-5)


class TestConstants:
    """Verify weight utility constants."""

    def test_u32_max(self):
        assert U32_MAX == 4294967295

    def test_u16_max(self):
        assert U16_MAX == 65535
