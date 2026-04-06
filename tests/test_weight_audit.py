"""Tests for weight audit and commit-reveal verification.

Covers CHEAT-05 (weight copying detection), CHEAT-06 (weight anomaly audit),
and NET-06 (commit-reveal verification).
"""

from unittest.mock import MagicMock

import numpy as np

from antigence_subnet.validator.weight_audit import (
    audit_weights,
    check_commit_reveal_enabled,
)


class TestAuditWeights:
    """Tests for audit_weights function."""

    def test_clean_weights_no_warnings(self):
        """Normal, well-distributed weight vector returns empty list."""
        weights = np.array([0.1, 0.2, 0.15, 0.25, 0.3], dtype=np.float32)
        warnings = audit_weights(weights)
        assert warnings == []

    def test_uniform_weights_warned(self):
        """All equal weights produce near-uniform warning."""
        weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        warnings = audit_weights(weights)
        assert any("near-uniform" in w for w in warnings)

    def test_extreme_concentration_warned(self):
        """One weight > 0.5 produces extreme concentration warning."""
        weights = np.array([0.8, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        warnings = audit_weights(weights)
        assert any("extreme weight concentration" in w for w in warnings)

    def test_both_warnings(self):
        """Weight vector triggering both uniform and concentrated checks."""
        # Single high weight with all others zero -- extreme concentration only
        # For both: need uniform non-zero AND max > 0.5
        # This is contradictory (uniform can't have max > 0.5 easily)
        # But we can test independently that both warning types fire
        # Let's test concentration first with one big weight
        weights_conc = np.array([0.9, 0.025, 0.025, 0.025, 0.025], dtype=np.float32)
        warnings_conc = audit_weights(weights_conc)
        assert any("extreme weight concentration" in w for w in warnings_conc)

        # And uniform separately
        weights_uni = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        warnings_uni = audit_weights(weights_uni)
        assert any("near-uniform" in w for w in warnings_uni)

    def test_similar_validator_warned(self):
        """Our weights with cosine sim > 0.99 to another validator triggers warning."""
        our_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15], dtype=np.float32)
        # Near-identical vector
        other_weights = our_weights + np.array([0.001, -0.001, 0.0, 0.0, 0.0], dtype=np.float32)
        recent = {"5C4hrfjw9DjXZTz": other_weights}
        warnings = audit_weights(our_weights, recent_validator_weights=recent)
        assert any("weight vector similarity" in w for w in warnings)

    def test_dissimilar_validator_ok(self):
        """Our weights with cosine sim < 0.99 to another validator: no cross-validator warning."""
        our_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15], dtype=np.float32)
        other_weights = np.array([0.05, 0.05, 0.3, 0.3, 0.3], dtype=np.float32)
        recent = {"5C4hrfjw9DjXZTz": other_weights}
        warnings = audit_weights(our_weights, recent_validator_weights=recent)
        # Should not have cross-validator warning
        assert not any("weight vector similarity" in w for w in warnings)

    def test_different_length_skipped(self):
        """Validator with different-length weight vector: no crash."""
        our_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15], dtype=np.float32)
        other_weights = np.array([0.5, 0.5], dtype=np.float32)  # Different length
        recent = {"5C4hrfjw9DjXZTz": other_weights}
        warnings = audit_weights(our_weights, recent_validator_weights=recent)
        # Should not crash, no cross-validator warning
        assert not any("weight vector similarity" in w for w in warnings)

    def test_zero_weights_no_false_positive(self):
        """All-zero weight vector returns no warnings (edge case)."""
        weights = np.zeros(5, dtype=np.float32)
        warnings = audit_weights(weights)
        assert warnings == []

    def test_single_nonzero_no_uniform_warning(self):
        """Only one non-zero weight: no 'uniform' warning (can't be uniform with 1 entry)."""
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        warnings = audit_weights(weights)
        # Should not have near-uniform warning (only 1 non-zero entry)
        assert not any("near-uniform" in w for w in warnings)
        # May have concentration warning (max=1.0 > 0.5) -- that's fine
        assert any("extreme weight concentration" in w for w in warnings)


class TestCheckCommitRevealEnabled:
    """Tests for check_commit_reveal_enabled function."""

    def test_commit_reveal_enabled_true(self):
        """Mock subtensor with commit_reveal_enabled returning True."""
        mock_subtensor = MagicMock()
        mock_subtensor.commit_reveal_enabled.return_value = True
        result = check_commit_reveal_enabled(mock_subtensor, netuid=1)
        assert result is True
        mock_subtensor.commit_reveal_enabled.assert_called_once_with(netuid=1)

    def test_commit_reveal_enabled_false(self):
        """Mock subtensor returning False."""
        mock_subtensor = MagicMock()
        mock_subtensor.commit_reveal_enabled.return_value = False
        result = check_commit_reveal_enabled(mock_subtensor, netuid=1)
        assert result is False

    def test_commit_reveal_enabled_no_method(self):
        """Mock subtensor without commit_reveal_enabled method: graceful False."""
        mock_subtensor = MagicMock(spec=[])
        # spec=[] means no attributes exist, so accessing commit_reveal_enabled
        # will raise AttributeError
        result = check_commit_reveal_enabled(mock_subtensor, netuid=1)
        assert result is False
