"""Tests for NKCell fast-path anomaly gate.

Covers: Protocol conformance, z-score thresholds, binary feature handling,
confidence scaling, audit JSON loading, skip features, feature attribution.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.orchestrator.cells import ImmuneCellType
from antigence_subnet.miner.orchestrator.nk_cell import (
    DEFAULT_SKIP_FEATURES,
    FeatureStatistics,
    NKCell,
)


@pytest.fixture
def synthetic_stats():
    """Synthetic FeatureStatistics matching hallucination audit schema."""
    return [
        FeatureStatistics(name="claim_density", index=0, mean=0.5, std=0.025, is_binary=False, is_constant=False),
        FeatureStatistics(name="citation_count", index=1, mean=0.0, std=0.0, is_binary=True, is_constant=True),
        FeatureStatistics(name="hedging_ratio", index=2, mean=0.067, std=0.249, is_binary=True, is_constant=False),
        FeatureStatistics(name="specificity", index=3, mean=0.74, std=0.247, is_binary=False, is_constant=False),
        FeatureStatistics(name="numeric_density", index=4, mean=0.233, std=0.423, is_binary=True, is_constant=False),
        FeatureStatistics(name="pamp_score", index=5, mean=0.006, std=0.043, is_binary=False, is_constant=False),
        FeatureStatistics(name="exaggeration", index=6, mean=0.0, std=0.0, is_binary=False, is_constant=True),
        FeatureStatistics(name="certainty", index=7, mean=0.033, std=0.180, is_binary=True, is_constant=False),
        FeatureStatistics(name="controversy", index=8, mean=0.0, std=0.0, is_binary=False, is_constant=True),
        FeatureStatistics(name="danger_signal", index=9, mean=0.004, std=0.026, is_binary=False, is_constant=False),
    ]


class TestNKCellProtocol:
    """NKCell satisfies ImmuneCellType Protocol."""

    def test_nk_cell_satisfies_protocol(self, synthetic_stats):
        """NKCell instance passes isinstance(cell, ImmuneCellType)."""
        cell = NKCell(feature_stats=synthetic_stats)
        assert isinstance(cell, ImmuneCellType)


class TestNKCellProcess:
    """NKCell.process() returns None for normal features, DetectionResult for extreme."""

    def test_returns_none_for_normal_features(self, synthetic_stats):
        """All features within normal range -> returns None (defer to ensemble)."""
        cell = NKCell(feature_stats=synthetic_stats)
        # All zeros: claim_density z=(0-0.5)/0.025=20, but let's use mean values
        features = np.array([0.5, 0.0, 0.0, 0.74, 0.0, 0.006, 0.0, 0.0, 0.0, 0.004])
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_returns_detection_result_for_extreme_feature(self, synthetic_stats):
        """pamp_score=1.0 when mean=0.006, std=0.043 -> z=23.1 >> 3.0 -> DetectionResult."""
        cell = NKCell(feature_stats=synthetic_stats)
        features = np.array([0.5, 0.0, 0.0, 0.74, 0.0, 1.0, 0.0, 0.0, 0.0, 0.004])
        result = cell.process(features, "prompt", "output")
        assert result is not None
        assert isinstance(result, DetectionResult)
        assert result.score == 1.0
        assert result.anomaly_type == "nk_fast_path"

    def test_detection_result_has_feature_attribution(self, synthetic_stats):
        """DetectionResult.feature_attribution contains triggering feature name and z-score."""
        cell = NKCell(feature_stats=synthetic_stats)
        features = np.array([0.5, 0.0, 0.0, 0.74, 0.0, 1.0, 0.0, 0.0, 0.0, 0.004])
        result = cell.process(features, "prompt", "output")
        assert result is not None
        assert result.feature_attribution is not None
        assert "pamp_score" in result.feature_attribution
        # z = abs(1.0 - 0.006) / 0.043 = 23.1163
        expected_z = round(abs(1.0 - 0.006) / 0.043, 4)
        assert result.feature_attribution["pamp_score"] == expected_z

    def test_skips_constant_features(self, synthetic_stats):
        """Constant features (std=0) are skipped even when value is extreme."""
        cell = NKCell(feature_stats=synthetic_stats)
        # exaggeration (index=6) is constant (std=0), set to extreme value 1.0
        # controversy (index=8) is constant (std=0), set to extreme value 1.0
        # All other continuous features at their mean
        features = np.array([0.5, 0.0, 0.0, 0.74, 0.0, 0.006, 1.0, 0.0, 1.0, 0.004])
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_empty_feature_stats_returns_none(self):
        """NKCell with empty feature_stats returns None (graceful passthrough)."""
        cell = NKCell(feature_stats=[])
        features = np.ones(10)
        result = cell.process(features, "prompt", "output")
        assert result is None


class TestNKCellZScore:
    """Z-score math correctness and threshold boundary behavior."""

    def test_zscore_below_threshold_returns_none(self):
        """z=2.99 (below default threshold 3.0) -> None."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats)
        # z = abs(2.99 - 0.0) / 1.0 = 2.99 < 3.0
        features = np.array([2.99])
        result = cell.process(features, "p", "o")
        assert result is None

    def test_zscore_above_threshold_returns_detection(self):
        """z=3.01 (above default threshold 3.0) -> DetectionResult."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats)
        # z = abs(3.01 - 0.0) / 1.0 = 3.01 > 3.0
        features = np.array([3.01])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert result.score == 1.0
        assert result.anomaly_type == "nk_fast_path"

    def test_zscore_exactly_at_threshold_returns_none(self):
        """z=3.0 (exactly at threshold, not exceeding) -> None."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats)
        features = np.array([3.0])
        result = cell.process(features, "p", "o")
        assert result is None

    def test_custom_z_threshold(self):
        """Custom z_threshold=2.0 changes trigger sensitivity."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats, z_threshold=2.0)
        # z=2.5 > 2.0 -> should trigger
        features = np.array([2.5])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert result.score == 1.0

    def test_negative_deviation_triggers(self):
        """Negative z-score (value << mean) also triggers."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=10.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats)
        # z = abs(0.0 - 10.0) / 1.0 = 10.0 >> 3.0
        features = np.array([0.0])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert result.feature_attribution is not None
        assert result.feature_attribution["feat_a"] == 10.0


class TestNKCellConfidence:
    """Confidence scales with z-score per D-03: min(1.0, z / (2*threshold))."""

    def test_confidence_at_threshold(self):
        """At exactly z=threshold+epsilon -> confidence ~= 0.5."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats, z_threshold=3.0)
        # z = 3.01, confidence = min(1.0, 3.01 / 6.0) = 0.5017
        features = np.array([3.01])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert abs(result.confidence - 0.5) < 0.01

    def test_confidence_at_double_threshold(self):
        """At z=2*threshold -> confidence = 1.0."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats, z_threshold=3.0)
        # z = 6.0, confidence = min(1.0, 6.0 / 6.0) = 1.0
        features = np.array([6.0])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert result.confidence == 1.0

    def test_confidence_capped_at_1(self):
        """At z >> 2*threshold -> confidence capped at 1.0."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats, z_threshold=3.0)
        # z = 100.0, confidence = min(1.0, 100.0 / 6.0) = 1.0
        features = np.array([100.0])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert result.confidence == 1.0

    def test_confidence_formula_exact(self):
        """Verify exact confidence formula: min(1.0, z_score / (2.0 * z_threshold))."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats, z_threshold=3.0)
        # z = 4.5, confidence = min(1.0, 4.5 / 6.0) = 0.75
        features = np.array([4.5])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert result.confidence == 0.75


class TestNKCellBinary:
    """Binary feature handling."""

    def test_binary_nonzero_prevalence_no_trigger(self):
        """Binary feature with non-zero prevalence (is_binary=True, is_constant=False) -> skip."""
        stats = [
            FeatureStatistics(name="hedging_ratio", index=0, mean=0.067, std=0.249, is_binary=True, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats)
        # Even with extreme value, binary non-constant should not trigger
        features = np.array([1.0])
        result = cell.process(features, "p", "o")
        assert result is None

    def test_binary_constant_at_zero_skipped(self):
        """Binary feature constant at 0.0 (is_binary=True, is_constant=True) -> skipped by is_constant."""
        stats = [
            FeatureStatistics(name="citation_count", index=0, mean=0.0, std=0.0, is_binary=True, is_constant=True),
        ]
        cell = NKCell(feature_stats=stats)
        features = np.array([1.0])
        result = cell.process(features, "p", "o")
        assert result is None


class TestNKCellLoading:
    """NKCell.from_audit_json() loads statistics from audit JSON files."""

    def test_load_hallucination_audit(self):
        """from_audit_json() with real data/audit/hallucination.json produces correct stats."""
        audit_path = Path(__file__).parent.parent / "data" / "audit" / "hallucination.json"
        if not audit_path.exists():
            pytest.skip("Audit file not found")

        cell = NKCell.from_audit_json(audit_path)
        assert len(cell._stats) == 10

        # Verify first feature
        claim = cell._stats[0]
        assert claim.name == "claim_density"
        assert claim.index == 0
        assert abs(claim.mean - 0.5067) < 0.001
        assert abs(claim.std - 0.0249) < 0.001
        assert claim.is_binary is False
        assert claim.is_constant is False

        # Verify a binary constant feature
        citation = cell._stats[1]
        assert citation.name == "citation_count"
        assert citation.is_binary is True
        assert citation.is_constant is True

    def test_load_with_custom_threshold(self):
        """from_audit_json() passes z_threshold through."""
        audit_path = Path(__file__).parent.parent / "data" / "audit" / "hallucination.json"
        if not audit_path.exists():
            pytest.skip("Audit file not found")

        cell = NKCell.from_audit_json(audit_path, z_threshold=2.0)
        assert cell._z_threshold == 2.0

    def test_load_preserves_feature_order(self):
        """from_audit_json() preserves feature_names order from JSON."""
        audit_path = Path(__file__).parent.parent / "data" / "audit" / "hallucination.json"
        if not audit_path.exists():
            pytest.skip("Audit file not found")

        cell = NKCell.from_audit_json(audit_path)
        expected_names = [
            "claim_density", "citation_count", "hedging_ratio", "specificity",
            "numeric_density", "pamp_score", "exaggeration", "certainty",
            "controversy", "danger_signal",
        ]
        actual_names = [s.name for s in cell._stats]
        assert actual_names == expected_names


class TestNKCellSkipFeatures:
    """DEFAULT_SKIP_FEATURES and custom skip_features parameter."""

    def test_default_skip_features_contains_danger_signal(self):
        """DEFAULT_SKIP_FEATURES = {'danger_signal'}."""
        assert "danger_signal" in DEFAULT_SKIP_FEATURES

    def test_danger_signal_skipped_by_default(self, synthetic_stats):
        """danger_signal is skipped by default even when extreme."""
        cell = NKCell(feature_stats=synthetic_stats)
        # danger_signal (index=9) is continuous non-constant with mean=0.004, std=0.026
        # Set to 1.0: z = (1.0 - 0.004)/0.026 = 38.3 >> 3.0
        # But should be skipped because it's in DEFAULT_SKIP_FEATURES
        features = np.array([0.5, 0.0, 0.0, 0.74, 0.0, 0.006, 0.0, 0.0, 0.0, 1.0])
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_custom_skip_features(self):
        """Custom skip_features parameter overrides default."""
        stats = [
            FeatureStatistics(name="feat_a", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
            FeatureStatistics(name="feat_b", index=1, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats, skip_features={"feat_a"})
        # feat_a skipped, feat_b extreme
        features = np.array([100.0, 100.0])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert "feat_b" in result.feature_attribution

    def test_empty_skip_features_processes_all(self):
        """Empty skip_features set means all features are processed."""
        stats = [
            FeatureStatistics(name="danger_signal", index=0, mean=0.0, std=1.0, is_binary=False, is_constant=False),
        ]
        cell = NKCell(feature_stats=stats, skip_features=set())
        features = np.array([100.0])
        result = cell.process(features, "p", "o")
        assert result is not None
        assert "danger_signal" in result.feature_attribution

    def test_feature_attribution_zscore_rounded(self, synthetic_stats):
        """feature_attribution z-score is rounded to 4 decimal places."""
        cell = NKCell(feature_stats=synthetic_stats)
        # pamp_score: z = abs(1.0 - 0.006) / 0.043 = 23.116279...
        features = np.array([0.5, 0.0, 0.0, 0.74, 0.0, 1.0, 0.0, 0.0, 0.0, 0.004])
        result = cell.process(features, "prompt", "output")
        assert result is not None
        z_val = result.feature_attribution["pamp_score"]
        # Check it's rounded to 4 decimal places
        assert z_val == round(z_val, 4)
        expected = round(abs(1.0 - 0.006) / 0.043, 4)
        assert z_val == expected
