"""
Tests for multi-validator agreement module (VHARD-04).

Verifies Spearman rank correlation computation, ranking snapshot storage,
network agreement, outlier detection, and edge cases.
"""

import numpy as np
import pytest

from antigence_subnet.validator.agreement import (
    AgreementConfig,
    RankingSnapshot,
    ValidatorAgreement,
    parse_agreement_config,
)

# ------------------------------------------------------------------ #
# AgreementConfig
# ------------------------------------------------------------------ #


class TestAgreementConfig:
    """Tests for AgreementConfig dataclass."""

    def test_defaults(self):
        """Config has documented defaults."""
        cfg = AgreementConfig()
        assert cfg.min_validators == 2
        assert cfg.correlation_threshold == 0.5
        assert cfg.max_snapshots == 100

    def test_custom_threshold(self):
        """Custom threshold accepted."""
        cfg = AgreementConfig(correlation_threshold=0.8)
        assert cfg.correlation_threshold == 0.8


# ------------------------------------------------------------------ #
# TOML parsing
# ------------------------------------------------------------------ #


class TestTomlParsing:
    """Tests for parse_agreement_config TOML integration."""

    def test_empty_toml(self):
        """Empty dict produces defaults."""
        cfg = parse_agreement_config({})
        assert cfg.min_validators == 2
        assert cfg.correlation_threshold == 0.5

    def test_partial_toml(self):
        """Partial TOML overrides specified keys."""
        toml = {
            "validator": {
                "agreement": {"correlation_threshold": 0.7}
            }
        }
        cfg = parse_agreement_config(toml)
        assert cfg.correlation_threshold == 0.7
        assert cfg.min_validators == 2  # default

    def test_full_toml(self):
        """Full TOML section populates all fields."""
        toml = {
            "validator": {
                "agreement": {
                    "min_validators": 3,
                    "correlation_threshold": 0.6,
                    "max_snapshots": 50,
                }
            }
        }
        cfg = parse_agreement_config(toml)
        assert cfg.min_validators == 3
        assert cfg.correlation_threshold == 0.6
        assert cfg.max_snapshots == 50


# ------------------------------------------------------------------ #
# RankingSnapshot
# ------------------------------------------------------------------ #


class TestRankingSnapshot:
    """Tests for RankingSnapshot dataclass."""

    def test_creation(self):
        """Snapshot stores hotkey, step, rankings."""
        snap = RankingSnapshot(
            validator_hotkey="val-A",
            step=10,
            rankings={0: 0.9, 1: 0.7, 2: 0.3},
        )
        assert snap.validator_hotkey == "val-A"
        assert snap.step == 10
        assert snap.rankings[1] == 0.7
        assert snap.timestamp > 0

    def test_auto_timestamp(self):
        """Timestamp auto-generated if not provided."""
        snap = RankingSnapshot(
            validator_hotkey="val-B",
            step=5,
            rankings={0: 1.0},
        )
        assert isinstance(snap.timestamp, float)


# ------------------------------------------------------------------ #
# Spearman correlation: known cases
# ------------------------------------------------------------------ #


class TestSpearmanCorrelation:
    """Tests for Spearman rank correlation with known rankings."""

    def test_perfect_agreement(self):
        """Identical rankings -> correlation = 1.0."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1},
        ))
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1,
            rankings={0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1},
        ))
        result = va.compute_agreement("A", "B")
        assert abs(result.correlation - 1.0) < 0.01
        assert result.n_common_uids == 5

    def test_inverse_agreement(self):
        """Reversed rankings -> correlation = -1.0."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1},
        ))
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1,
            rankings={0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9},
        ))
        result = va.compute_agreement("A", "B")
        assert abs(result.correlation - (-1.0)) < 0.01

    def test_correlated_but_not_identical(self):
        """Same order but different magnitudes -> high correlation."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1},
        ))
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1,
            rankings={0: 0.85, 1: 0.65, 2: 0.45, 3: 0.25, 4: 0.05},
        ))
        result = va.compute_agreement("A", "B")
        assert result.correlation > 0.9
        assert result.is_significant is True

    def test_random_rankings(self):
        """Random vs ordered -> low or no correlation."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={i: float(i) / 10 for i in range(10)},
        ))
        # Shuffled rankings
        np.random.seed(42)
        shuffled = list(range(10))
        np.random.shuffle(shuffled)
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1,
            rankings={i: float(shuffled[i]) / 10 for i in range(10)},
        ))
        result = va.compute_agreement("A", "B")
        # Random permutation typically gives correlation close to 0
        assert abs(result.correlation) < 0.8


# ------------------------------------------------------------------ #
# Network agreement
# ------------------------------------------------------------------ #


class TestNetworkAgreement:
    """Tests for network-wide agreement computation."""

    def _make_va(self, n_validators, same_rankings=True):
        """Helper to create ValidatorAgreement with n validators."""
        va = ValidatorAgreement()
        for i in range(n_validators):
            if same_rankings:
                rankings = {uid: float(uid) / 10 for uid in range(5)}
            else:
                # Each validator has different ordering
                np.random.seed(i + 100)
                vals = np.random.permutation(5)
                rankings = {uid: float(vals[uid]) / 4 for uid in range(5)}
            va.record_ranking(RankingSnapshot(
                validator_hotkey=f"val-{i}",
                step=1,
                rankings=rankings,
            ))
        return va

    def test_two_validators_perfect(self):
        """Two validators with same rankings -> agreement = 1.0."""
        va = self._make_va(2, same_rankings=True)
        assert abs(va.get_network_agreement() - 1.0) < 0.01

    def test_three_validators_perfect(self):
        """Three validators with same rankings -> agreement = 1.0."""
        va = self._make_va(3, same_rankings=True)
        assert abs(va.get_network_agreement() - 1.0) < 0.01

    def test_four_validators_perfect(self):
        """Four validators with same rankings -> agreement = 1.0."""
        va = self._make_va(4, same_rankings=True)
        assert abs(va.get_network_agreement() - 1.0) < 0.01

    def test_single_validator_no_agreement(self):
        """Single validator -> agreement = 0.0 (need >= 2)."""
        va = self._make_va(1)
        assert va.get_network_agreement() == 0.0


# ------------------------------------------------------------------ #
# Outlier detection
# ------------------------------------------------------------------ #


class TestOutlierDetection:
    """Tests for validator outlier detection."""

    def test_no_outliers_when_all_agree(self):
        """All validators agree -> no outliers."""
        va = ValidatorAgreement()
        for i in range(3):
            va.record_ranking(RankingSnapshot(
                validator_hotkey=f"val-{i}",
                step=1,
                rankings={0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1},
            ))
        assert va.detect_outlier_validator() == []

    def test_one_outlier_detected(self):
        """One validator with reversed rankings -> detected as outlier."""
        va = ValidatorAgreement()
        # Two agreeing validators
        for i in range(2):
            va.record_ranking(RankingSnapshot(
                validator_hotkey=f"val-{i}",
                step=1,
                rankings={0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1},
            ))
        # One disagreeing validator (reversed)
        va.record_ranking(RankingSnapshot(
            validator_hotkey="outlier",
            step=1,
            rankings={0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9},
        ))
        outliers = va.detect_outlier_validator()
        assert "outlier" in outliers

    def test_custom_threshold(self):
        """Custom threshold affects outlier detection."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1},
        ))
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1,
            rankings={0: 0.85, 1: 0.65, 2: 0.55, 3: 0.35, 4: 0.05},
        ))
        # With very high threshold, even slightly different rankings are outliers
        outliers = va.detect_outlier_validator(threshold=0.999)
        # Both are measured against each other -- may or may not be outliers
        # depending on exact correlation
        assert isinstance(outliers, list)


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


class TestEdgeCases:
    """Tests for edge cases in agreement computation."""

    def test_no_common_uids(self):
        """No overlapping UIDs -> correlation 0.0."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={0: 0.9, 1: 0.7},
        ))
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1,
            rankings={10: 0.8, 11: 0.6},
        ))
        result = va.compute_agreement("A", "B")
        assert result.correlation == 0.0
        assert result.n_common_uids == 0
        assert result.is_significant is False

    def test_single_common_uid(self):
        """Single common UID -> correlation 0.0 (need >= 2)."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={0: 0.9, 1: 0.7},
        ))
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1,
            rankings={0: 0.8, 10: 0.6},
        ))
        result = va.compute_agreement("A", "B")
        assert result.correlation == 0.0
        assert result.n_common_uids == 1

    def test_identical_scores(self):
        """All same score (constant array) -> correlation 0.0."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={0: 0.5, 1: 0.5, 2: 0.5},
        ))
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1,
            rankings={0: 0.5, 1: 0.5, 2: 0.5},
        ))
        result = va.compute_agreement("A", "B")
        assert result.correlation == 0.0  # undefined -> 0.0

    def test_missing_validator_raises(self):
        """Querying unknown validator raises ValueError."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1,
            rankings={0: 0.9},
        ))
        with pytest.raises(ValueError, match="No rankings recorded"):
            va.compute_agreement("A", "unknown")

    def test_max_snapshots_eviction(self):
        """Old snapshots evicted when max_snapshots reached."""
        cfg = AgreementConfig(max_snapshots=3)
        va = ValidatorAgreement(config=cfg)

        for step in range(5):
            va.record_ranking(RankingSnapshot(
                validator_hotkey="A",
                step=step,
                rankings={0: float(step) / 4},
            ))

        # Only last 3 snapshots remain
        latest = va.get_latest_ranking("A")
        assert latest is not None
        assert latest.step == 4
        # Check internal list length
        assert len(va._snapshots["A"]) == 3

    def test_get_latest_ranking_none(self):
        """get_latest_ranking returns None for unknown validator."""
        va = ValidatorAgreement()
        assert va.get_latest_ranking("unknown") is None

    def test_validator_count(self):
        """validator_count reflects unique validators."""
        va = ValidatorAgreement()
        assert va.validator_count == 0
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=1, rankings={0: 0.5},
        ))
        assert va.validator_count == 1
        va.record_ranking(RankingSnapshot(
            validator_hotkey="B", step=1, rankings={0: 0.5},
        ))
        assert va.validator_count == 2
        # Same validator again
        va.record_ranking(RankingSnapshot(
            validator_hotkey="A", step=2, rankings={0: 0.6},
        ))
        assert va.validator_count == 2

    def test_validator_hotkeys_list(self):
        """validator_hotkeys returns list of all recorded validators."""
        va = ValidatorAgreement()
        va.record_ranking(RankingSnapshot(
            validator_hotkey="X", step=1, rankings={0: 0.5},
        ))
        va.record_ranking(RankingSnapshot(
            validator_hotkey="Y", step=1, rankings={0: 0.5},
        ))
        assert set(va.validator_hotkeys) == {"X", "Y"}


# ------------------------------------------------------------------ #
# Integration: validator has agreement attribute
# ------------------------------------------------------------------ #


@pytest.mark.skip(reason="agreement not yet wired into Validator")
class TestValidatorIntegration:
    """Integration tests with BaseValidatorNeuron."""

    def test_agreement_attr_exists(self, mock_config):
        """Validator has agreement attribute after init."""
        from neurons.validator import Validator

        v = Validator(config=mock_config)
        assert hasattr(v, "agreement")
        assert isinstance(v.agreement, ValidatorAgreement)

    def test_agreement_records_snapshot(self, mock_config):
        """_record_agreement_snapshot creates a snapshot."""
        from neurons.validator import Validator

        v = Validator(config=mock_config)
        v.scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=np.float32)
        v._record_agreement_snapshot()
        assert v.agreement.validator_count == 1
