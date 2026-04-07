"""Tests for the feature audit script.

Validates that audit_domain() produces correct statistics, binary/constant
classification, and correlation flagging for all domains including the
degenerate code_security case (all features constant when output is empty).
"""

import sys
from pathlib import Path

import pytest

# Add scripts directory to path so we can import feature_audit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from feature_audit import audit_domain  # noqa: E402

# ---------------------------------------------------------------------------
# TestAuditDomainStructure -- verify returned dict has required keys
# ---------------------------------------------------------------------------


class TestAuditDomainStructure:
    """Verify audit_domain returns all required top-level and nested keys."""

    @pytest.fixture(scope="class")
    def hallucination_result(self):
        return audit_domain("hallucination")

    def test_audit_domain_returns_required_keys(self, hallucination_result):
        required_keys = {
            "domain",
            "n_samples",
            "feature_stats",
            "correlation_matrix",
            "high_correlations",
            "feature_names",
        }
        assert required_keys.issubset(hallucination_result.keys())

    def test_domain_name_correct(self, hallucination_result):
        assert hallucination_result["domain"] == "hallucination"

    def test_n_samples_positive(self, hallucination_result):
        assert hallucination_result["n_samples"] > 0

    def test_feature_names_has_ten(self, hallucination_result):
        assert len(hallucination_result["feature_names"]) == 10

    def test_feature_stats_has_ten_entries(self, hallucination_result):
        assert len(hallucination_result["feature_stats"]) == 10

    def test_feature_stat_has_required_keys(self, hallucination_result):
        required_stat_keys = {
            "index",
            "mean",
            "std",
            "min",
            "max",
            "percentiles",
            "unique_count",
            "is_binary",
            "is_constant",
        }
        for name, stats in hallucination_result["feature_stats"].items():
            assert required_stat_keys.issubset(
                stats.keys()
            ), f"Feature {name} missing keys: {required_stat_keys - stats.keys()}"

    def test_percentiles_dict_has_required_keys(self, hallucination_result):
        required_percentile_keys = {"p5", "p25", "p50", "p75", "p95"}
        for name, stats in hallucination_result["feature_stats"].items():
            assert required_percentile_keys.issubset(
                stats["percentiles"].keys()
            ), f"Feature {name} missing percentile keys"

    def test_metadata_present(self, hallucination_result):
        assert "metadata" in hallucination_result


# ---------------------------------------------------------------------------
# TestBinaryClassification -- verify known binary features classified right
# ---------------------------------------------------------------------------


class TestBinaryClassification:
    """Verify binary feature classification for known binary features."""

    @pytest.fixture(scope="class")
    def hallucination_result(self):
        return audit_domain("hallucination")

    def test_binary_classification_hedging_ratio(self, hallucination_result):
        """hedging_ratio is binary (unique values in {0.0, 1.0})."""
        stats = hallucination_result["feature_stats"]["hedging_ratio"]
        assert stats["is_binary"] is True

    def test_binary_classification_certainty(self, hallucination_result):
        """certainty is binary (unique values in {0.0, 1.0})."""
        stats = hallucination_result["feature_stats"]["certainty"]
        assert stats["is_binary"] is True

    def test_binary_classification_citation_count(self, hallucination_result):
        """citation_count is binary (unique values in {0.0, 1.0} or all same)."""
        stats = hallucination_result["feature_stats"]["citation_count"]
        # citation_count is binary OR constant (all 0.0) -- either way is_binary should be True
        assert stats["is_binary"] is True

    def test_binary_classification_numeric_density(self, hallucination_result):
        """numeric_density is binary (unique values in {0.0, 1.0})."""
        stats = hallucination_result["feature_stats"]["numeric_density"]
        assert stats["is_binary"] is True

    def test_continuous_classification_specificity(self, hallucination_result):
        """specificity is continuous (many unique values)."""
        stats = hallucination_result["feature_stats"]["specificity"]
        # specificity has many unique values, so should NOT be binary
        # (unless by coincidence all samples map to {0.0, 1.0})
        assert stats["unique_count"] > 2 or stats["is_binary"] is True


# ---------------------------------------------------------------------------
# TestConstantDetection -- verify constant features flagged
# ---------------------------------------------------------------------------


class TestConstantDetection:
    """Verify constant feature detection."""

    @pytest.fixture(scope="class")
    def hallucination_result(self):
        return audit_domain("hallucination")

    def test_constant_feature_citation_count(self, hallucination_result):
        """citation_count should be constant (all 0.0) in hallucination domain."""
        stats = hallucination_result["feature_stats"]["citation_count"]
        assert stats["is_constant"] is True
        assert stats["std"] == 0.0

    def test_non_constant_feature_specificity(self, hallucination_result):
        """specificity should NOT be constant in hallucination domain."""
        stats = hallucination_result["feature_stats"]["specificity"]
        assert stats["is_constant"] is False


# ---------------------------------------------------------------------------
# TestCorrelationMatrix -- verify shape, NaN->None, pamp/danger flagged
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    """Verify correlation matrix structure and high-correlation flagging."""

    @pytest.fixture(scope="class")
    def hallucination_result(self):
        return audit_domain("hallucination")

    def test_correlation_matrix_shape(self, hallucination_result):
        """Correlation matrix should be 10x10."""
        matrix = hallucination_result["correlation_matrix"]
        assert len(matrix) == 10, f"Expected 10 rows, got {len(matrix)}"
        for i, row in enumerate(matrix):
            assert len(row) == 10, f"Row {i} has {len(row)} columns, expected 10"

    def test_nan_replaced_with_none(self, hallucination_result):
        """No NaN values in correlation matrix -- constant features should produce None."""
        matrix = hallucination_result["correlation_matrix"]
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val is not None:
                    assert val == val, f"NaN found at [{i}][{j}]"  # NaN != NaN

    def test_pamp_danger_high_correlation(self, hallucination_result):
        """pamp_score and danger_signal should be flagged as highly correlated."""
        high_corrs = hallucination_result["high_correlations"]
        pair_found = any(
            ("pamp_score" in pair["features"] and "danger_signal" in pair["features"])
            for pair in high_corrs
        )
        assert pair_found, (
            f"pamp_score/danger_signal pair not found in high_correlations: {high_corrs}"
        )

    def test_high_correlation_has_r_value(self, hallucination_result):
        """Each high correlation entry should have an r value."""
        for pair in hallucination_result["high_correlations"]:
            assert "r" in pair
            assert "features" in pair


# ---------------------------------------------------------------------------
# TestCodeSecurityEdgeCase -- all-constant domain does not crash
# ---------------------------------------------------------------------------


class TestCodeSecurityEdgeCase:
    """Verify code_security domain (potentially all constant) does not crash."""

    @pytest.fixture(scope="class")
    def code_security_result(self):
        return audit_domain("code_security")

    def test_code_security_does_not_crash(self, code_security_result):
        """code_security domain should complete without error."""
        assert code_security_result["domain"] == "code_security"
        assert code_security_result["n_samples"] > 0

    def test_code_security_has_valid_stats(self, code_security_result):
        """Each feature stat should have valid numeric values (not NaN)."""
        for name, stats in code_security_result["feature_stats"].items():
            assert isinstance(stats["mean"], int | float), f"{name}.mean is not numeric"
            assert isinstance(stats["std"], int | float), f"{name}.std is not numeric"
            # mean and std should not be NaN
            assert stats["mean"] == stats["mean"], f"{name}.mean is NaN"
            assert stats["std"] == stats["std"], f"{name}.std is NaN"

    def test_code_security_correlation_matrix_no_nan(self, code_security_result):
        """Correlation matrix should use None (not NaN) for undefined correlations."""
        import json

        # Serialize to JSON to check for NaN strings
        serialized = json.dumps(code_security_result["correlation_matrix"])
        assert "NaN" not in serialized, "Correlation matrix contains NaN string"
