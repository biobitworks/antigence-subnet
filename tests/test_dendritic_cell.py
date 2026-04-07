"""Tests for DendriticCell (dDCA) signal classification, maturation state, and tier routing.

Covers: DCA-01 (signal classification), DCA-02 (tier routing), DCA-03 (determinism),
Protocol conformance, edge cases, from_config factory.
"""

from __future__ import annotations

import numpy as np
import pytest

from antigence_subnet.miner.orchestrator.cells import ImmuneCellType
from antigence_subnet.miner.orchestrator.dendritic_cell import (
    DANGER_FEATURES,
    EXCLUDED_FEATURES,
    PAMP_FEATURES,
    SAFE_FEATURES,
    TIER_MAP,
    DCAResult,
    DendriticCell,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(**kwargs: float) -> np.ndarray:
    """Build a 10-dim feature vector, defaulting all features to 0.0.

    Accepts keyword arguments keyed by feature name, e.g.
    ``_make_features(pamp_score=0.8, claim_density=0.3)``.
    """
    name_to_idx = {
        "claim_density": 0,
        "citation_count": 1,
        "hedging_ratio": 2,
        "specificity": 3,
        "numeric_density": 4,
        "pamp_score": 5,
        "exaggeration": 6,
        "certainty": 7,
        "controversy": 8,
        "danger_signal": 9,
    }
    vec = np.zeros(10, dtype=np.float64)
    for name, value in kwargs.items():
        vec[name_to_idx[name]] = value
    return vec


# ---------------------------------------------------------------------------
# Signal Category Constants
# ---------------------------------------------------------------------------


class TestSignalCategories:
    """Module-level signal category constants are correct and disjoint."""

    def test_pamp_contains_pamp_score_only(self):
        """PAMP_FEATURES maps pamp_score [5] with weight 1.0."""
        assert "pamp_score" in PAMP_FEATURES
        assert PAMP_FEATURES["pamp_score"] == (5, 1.0)
        assert len(PAMP_FEATURES) == 1

    def test_danger_contains_three_features(self):
        """DANGER_FEATURES maps exaggeration [6], controversy [8], claim_density [0]."""
        assert "exaggeration" in DANGER_FEATURES
        assert "controversy" in DANGER_FEATURES
        assert "claim_density" in DANGER_FEATURES
        assert len(DANGER_FEATURES) == 3
        # Equal weights summing to ~1.0
        total = sum(w for _, w in DANGER_FEATURES.values())
        assert abs(total - 1.0) < 1e-9

    def test_safe_contains_five_features(self):
        """SAFE_FEATURES maps citation_count [1], hedging_ratio [2], numeric_density [4], certainty [7], specificity [3]."""  # noqa: E501
        expected_names = {
            "citation_count",
            "hedging_ratio",
            "numeric_density",
            "certainty",
            "specificity",
        }
        assert set(SAFE_FEATURES.keys()) == expected_names
        # Equal weights summing to ~1.0
        total = sum(w for _, w in SAFE_FEATURES.values())
        assert abs(total - 1.0) < 1e-9

    def test_excluded_contains_danger_signal(self):
        """EXCLUDED_FEATURES = {'danger_signal'}."""
        assert "danger_signal" in EXCLUDED_FEATURES

    def test_categories_are_disjoint(self):
        """No feature appears in multiple categories (disjoint partition of 9 active features)."""
        pamp_names = set(PAMP_FEATURES.keys())
        danger_names = set(DANGER_FEATURES.keys())
        safe_names = set(SAFE_FEATURES.keys())
        # No overlaps
        assert pamp_names & danger_names == set()
        assert pamp_names & safe_names == set()
        assert danger_names & safe_names == set()
        # Union covers exactly 9 features (all except danger_signal)
        all_covered = pamp_names | danger_names | safe_names
        assert len(all_covered) == 9
        assert "danger_signal" not in all_covered

    def test_danger_signal_excluded_from_all_categories(self):
        """danger_signal [9] must not appear in PAMP, Danger, or Safe."""
        assert "danger_signal" not in PAMP_FEATURES
        assert "danger_signal" not in DANGER_FEATURES
        assert "danger_signal" not in SAFE_FEATURES


# ---------------------------------------------------------------------------
# Tier Map
# ---------------------------------------------------------------------------


class TestTierMap:
    """TIER_MAP routes maturation states to detector tiers."""

    def test_immature_routes_to_ocsvm_and_negsel(self):
        assert TIER_MAP["immature"] == ["ocsvm", "negsel"]

    def test_semi_mature_routes_to_ocsvm_and_negsel(self):
        assert TIER_MAP["semi-mature"] == ["ocsvm", "negsel"]

    def test_mature_routes_to_full_ensemble(self):
        assert TIER_MAP["mature"] == []


# ---------------------------------------------------------------------------
# DCAResult
# ---------------------------------------------------------------------------


class TestDCAResult:
    """DCAResult is frozen (immutable) and has correct fields."""

    def test_dca_result_fields(self):
        result = DCAResult(
            maturation_state="immature",
            signal_scores={"pamp": 0.0, "danger": 0.0, "safe": 0.0},
            recommended_tier=["ocsvm"],
        )
        assert result.maturation_state == "immature"
        assert result.signal_scores == {"pamp": 0.0, "danger": 0.0, "safe": 0.0}
        assert result.recommended_tier == ["ocsvm"]

    def test_dca_result_is_frozen(self):
        result = DCAResult(
            maturation_state="immature",
            signal_scores={"pamp": 0.0, "danger": 0.0, "safe": 0.0},
            recommended_tier=["ocsvm"],
        )
        with pytest.raises(AttributeError):
            result.maturation_state = "mature"

    def test_dca_result_equality(self):
        """Two DCAResults with same values are equal (frozen dataclass)."""
        r1 = DCAResult(
            maturation_state="semi-mature",
            signal_scores={"pamp": 0.1, "danger": 0.3, "safe": 0.2},
            recommended_tier=["ocsvm", "negsel"],
        )
        r2 = DCAResult(
            maturation_state="semi-mature",
            signal_scores={"pamp": 0.1, "danger": 0.3, "safe": 0.2},
            recommended_tier=["ocsvm", "negsel"],
        )
        assert r1 == r2


# ---------------------------------------------------------------------------
# Protocol Conformance
# ---------------------------------------------------------------------------


class TestDendriticCellProtocol:
    """DendriticCell satisfies ImmuneCellType Protocol."""

    def test_satisfies_protocol(self):
        """DendriticCell instance passes isinstance(cell, ImmuneCellType)."""
        cell = DendriticCell()
        assert isinstance(cell, ImmuneCellType)

    def test_process_returns_none(self):
        """process() always returns None (DCA is a router, not a detector)."""
        cell = DendriticCell()
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_process_with_all_optional_args(self):
        """process() accepts code and context kwargs."""
        cell = DendriticCell()
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output", code="x=1", context='{"k":"v"}')
        assert result is None


# ---------------------------------------------------------------------------
# Signal Classification
# ---------------------------------------------------------------------------


class TestClassifySignals:
    """DendriticCell.classify_signals() computes weighted signal scores."""

    def test_all_zero_features(self):
        """All-zero features produce all-zero signal scores."""
        cell = DendriticCell()
        scores = cell.classify_signals(np.zeros(10))
        assert scores["pamp"] == 0.0
        assert scores["danger"] == 0.0
        assert scores["safe"] == 0.0

    def test_pamp_score_from_feature(self):
        """pamp_score [5] = 0.8 -> signal_scores['pamp'] = 0.8."""
        cell = DendriticCell()
        features = _make_features(pamp_score=0.8)
        scores = cell.classify_signals(features)
        assert abs(scores["pamp"] - 0.8) < 1e-9

    def test_danger_score_from_features(self):
        """exaggeration=0.6, controversy=0.3, claim_density=0.9 -> danger = (0.6+0.3+0.9)/3."""
        cell = DendriticCell()
        features = _make_features(exaggeration=0.6, controversy=0.3, claim_density=0.9)
        scores = cell.classify_signals(features)
        expected_danger = (0.6 + 0.3 + 0.9) / 3.0
        assert abs(scores["danger"] - expected_danger) < 1e-9

    def test_safe_score_from_features(self):
        """All safe features at 1.0 -> safe = 1.0."""
        cell = DendriticCell()
        features = _make_features(
            citation_count=1.0,
            hedging_ratio=1.0,
            numeric_density=1.0,
            certainty=1.0,
            specificity=1.0,
        )
        scores = cell.classify_signals(features)
        assert abs(scores["safe"] - 1.0) < 1e-9

    def test_danger_signal_excluded(self):
        """danger_signal [9] does not affect any signal score."""
        cell = DendriticCell()
        f1 = _make_features(danger_signal=0.0)
        f2 = _make_features(danger_signal=1.0)
        s1 = cell.classify_signals(f1)
        s2 = cell.classify_signals(f2)
        assert s1 == s2

    def test_scores_clamped_to_unit_interval(self):
        """Signal scores are clamped to [0.0, 1.0] even with extreme inputs."""
        cell = DendriticCell()
        # Features can technically exceed 1.0 if constructed manually
        features = np.full(10, 5.0, dtype=np.float64)
        scores = cell.classify_signals(features)
        assert 0.0 <= scores["pamp"] <= 1.0
        assert 0.0 <= scores["danger"] <= 1.0
        assert 0.0 <= scores["safe"] <= 1.0

    def test_scores_have_three_keys(self):
        """classify_signals() returns dict with exactly keys 'pamp', 'danger', 'safe'."""
        cell = DendriticCell()
        scores = cell.classify_signals(np.zeros(10))
        assert set(scores.keys()) == {"pamp", "danger", "safe"}


# ---------------------------------------------------------------------------
# Maturation State
# ---------------------------------------------------------------------------


class TestDetermineMaturation:
    """DendriticCell.determine_maturation() applies D-03 logic."""

    def test_immature_when_safe_dominates(self):
        """safe > pamp + danger -> immature."""
        cell = DendriticCell()
        state = cell.determine_maturation({"pamp": 0.1, "danger": 0.1, "safe": 0.5})
        assert state == "immature"

    def test_mature_when_pamp_exceeds_threshold(self):
        """pamp >= pamp_threshold (0.3) -> mature."""
        cell = DendriticCell()
        state = cell.determine_maturation({"pamp": 0.3, "danger": 0.0, "safe": 0.0})
        assert state == "mature"

    def test_mature_when_pamp_above_threshold(self):
        """pamp > pamp_threshold (0.5 > 0.3) -> mature."""
        cell = DendriticCell()
        state = cell.determine_maturation({"pamp": 0.5, "danger": 0.0, "safe": 0.0})
        assert state == "mature"

    def test_semi_mature_otherwise(self):
        """pamp < threshold AND safe <= pamp+danger -> semi-mature."""
        cell = DendriticCell()
        # pamp=0.1 < 0.3, danger=0.3, safe=0.2 <= 0.1+0.3=0.4
        state = cell.determine_maturation({"pamp": 0.1, "danger": 0.3, "safe": 0.2})
        assert state == "semi-mature"

    def test_custom_pamp_threshold(self):
        """Custom pamp_threshold changes mature boundary."""
        cell = DendriticCell(pamp_threshold=0.5)
        # pamp=0.3 < 0.5 threshold
        state = cell.determine_maturation({"pamp": 0.3, "danger": 0.0, "safe": 0.0})
        assert state != "mature"

    def test_boundary_safe_equals_pamp_plus_danger(self):
        """When safe == pamp + danger exactly, not immature (safe must be strictly greater)."""
        cell = DendriticCell()
        state = cell.determine_maturation({"pamp": 0.1, "danger": 0.2, "safe": 0.3})
        assert state != "immature"


# ---------------------------------------------------------------------------
# classify() Integration
# ---------------------------------------------------------------------------


class TestClassify:
    """DendriticCell.classify() end-to-end: features -> DCAResult."""

    def test_all_zero_features_immature(self):
        """All-zero features -> safe_score(0) > pamp(0)+danger(0) is False, and pamp < 0.3 -> semi-mature.  # noqa: E501
        Actually: safe=0, pamp=0, danger=0. safe(0) > pamp(0)+danger(0) = 0 > 0 = False. pamp(0) < 0.3 -> semi-mature."""  # noqa: E501
        cell = DendriticCell()
        result = cell.classify(np.zeros(10))
        assert isinstance(result, DCAResult)
        # All scores are 0, safe(0) is NOT > pamp(0)+danger(0)=0, and pamp(0) < 0.3
        assert result.maturation_state == "semi-mature"

    def test_high_safe_features_immature(self):
        """High safe features with low pamp/danger -> immature."""
        cell = DendriticCell()
        features = _make_features(
            citation_count=1.0,
            hedging_ratio=1.0,
            specificity=1.0,
            numeric_density=1.0,
            certainty=1.0,
        )
        result = cell.classify(features)
        assert result.maturation_state == "immature"
        assert result.recommended_tier == ["ocsvm", "negsel"]

    def test_high_pamp_mature(self):
        """pamp_score >= 0.3 -> mature, regardless of other features."""
        cell = DendriticCell()
        features = _make_features(pamp_score=0.5)
        result = cell.classify(features)
        assert result.maturation_state == "mature"
        assert result.recommended_tier == []

    def test_all_features_at_one_mature(self):
        """All features at 1.0 -> pamp_score=1.0 >= 0.3 -> mature."""
        cell = DendriticCell()
        features = np.ones(10)
        result = cell.classify(features)
        assert result.maturation_state == "mature"
        assert result.recommended_tier == []

    def test_semi_mature_case(self):
        """Moderate danger, low pamp, low safe -> semi-mature."""
        cell = DendriticCell()
        features = _make_features(exaggeration=0.9, controversy=0.6, claim_density=0.3)
        result = cell.classify(features)
        assert result.maturation_state == "semi-mature"
        assert result.recommended_tier == ["ocsvm", "negsel"]

    def test_signal_scores_in_result(self):
        """classify() result contains correct signal_scores dict."""
        cell = DendriticCell()
        features = _make_features(pamp_score=0.2)
        result = cell.classify(features)
        assert "pamp" in result.signal_scores
        assert "danger" in result.signal_scores
        assert "safe" in result.signal_scores
        assert abs(result.signal_scores["pamp"] - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Determinism (DCA-03)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same input always produces same DCAResult (DCA-03)."""

    def test_deterministic_classify(self):
        """Calling classify() twice with identical features -> identical DCAResult."""
        cell = DendriticCell()
        features = _make_features(pamp_score=0.2, exaggeration=0.5, citation_count=1.0)
        r1 = cell.classify(features)
        r2 = cell.classify(features)
        assert r1 == r2

    def test_deterministic_across_instances(self):
        """Two DendriticCell instances with same config -> same results."""
        cell_a = DendriticCell()
        cell_b = DendriticCell()
        features = _make_features(pamp_score=0.1, controversy=0.8, specificity=0.5)
        r_a = cell_a.classify(features)
        r_b = cell_b.classify(features)
        assert r_a == r_b

    def test_deterministic_many_runs(self):
        """100 runs with same input produce identical results."""
        cell = DendriticCell()
        features = _make_features(pamp_score=0.15, exaggeration=0.3, hedging_ratio=1.0)
        first = cell.classify(features)
        for _ in range(99):
            assert cell.classify(features) == first


# ---------------------------------------------------------------------------
# from_config Factory
# ---------------------------------------------------------------------------


class TestFromConfig:
    """DendriticCell.from_config() creates instances from dca_config dicts."""

    def test_default_config(self):
        """Empty config dict -> default DendriticCell."""
        cell = DendriticCell.from_config({})
        result = cell.classify(np.zeros(10))
        assert isinstance(result, DCAResult)

    def test_custom_pamp_threshold(self):
        """Custom pamp_threshold from config."""
        cell = DendriticCell.from_config({"pamp_threshold": 0.5})
        # pamp_score=0.4 < 0.5 threshold -> not mature
        features = _make_features(pamp_score=0.4)
        result = cell.classify(features)
        assert result.maturation_state != "mature"

    def test_custom_tier_map(self):
        """Custom tier_map from config."""
        custom_map = {
            "immature": ["fast_only"],
            "semi-mature": ["fast_only", "deep"],
            "mature": ["all"],
        }
        cell = DendriticCell.from_config({"tier_map": custom_map})
        features = _make_features(pamp_score=0.5)
        result = cell.classify(features)
        assert result.recommended_tier == ["all"]

    def test_config_none_values_use_defaults(self):
        """None values in config use module defaults."""
        cell = DendriticCell.from_config(
            {
                "pamp_threshold": None,
                "signal_weights": None,
                "tier_map": None,
            }
        )
        # Should behave identically to default
        default_cell = DendriticCell()
        features = _make_features(pamp_score=0.2, exaggeration=0.5)
        assert cell.classify(features) == default_cell.classify(features)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case behavior for boundary inputs."""

    def test_pamp_at_threshold_boundary(self):
        """pamp exactly at threshold (0.3) -> mature (>= comparison)."""
        cell = DendriticCell()
        features = _make_features(pamp_score=0.3)
        result = cell.classify(features)
        assert result.maturation_state == "mature"

    def test_constant_features_contribute_zero(self):
        """Features with value 0.0 contribute nothing to their category score."""
        cell = DendriticCell()
        # exaggeration=0 in hallucination domain -> contributes 0 to danger
        features = _make_features(exaggeration=0.0, controversy=0.0, claim_density=0.0)
        scores = cell.classify_signals(features)
        assert scores["danger"] == 0.0

    def test_negative_feature_values_clamped(self):
        """Negative feature values produce clamped scores (never < 0)."""
        cell = DendriticCell()
        features = np.full(10, -1.0, dtype=np.float64)
        scores = cell.classify_signals(features)
        assert scores["pamp"] >= 0.0
        assert scores["danger"] >= 0.0
        assert scores["safe"] >= 0.0


# ---------------------------------------------------------------------------
# DendriticCell with AdaptiveWeightManager (Phase 44-02)
# ---------------------------------------------------------------------------


class TestClassifyWithWeightManager:
    """DendriticCell.classify() with pre-adapted weight manager."""

    def test_classify_with_weight_manager(self):
        """classify() with a pre-adapted weight manager produces different
        danger scores than default (danger has 3 features -> adaptation visible)."""
        from antigence_subnet.miner.orchestrator.adaptive_weights import AdaptiveWeightManager

        mgr = AdaptiveWeightManager(alpha=0.5)
        # Train: high exaggeration is predictive, others not
        for _ in range(20):
            features = _make_features(exaggeration=0.9, controversy=0.1, claim_density=0.1)
            mgr.adapt(features, outcome=1.0)

        dc_adapted = DendriticCell(weight_manager=mgr)
        dc_default = DendriticCell()

        test_features = _make_features(exaggeration=0.6, controversy=0.3, claim_density=0.2)
        adapted_result = dc_adapted.classify(test_features)
        default_result = dc_default.classify(test_features)

        # Adapted DCA should have shifted danger score due to different weighting
        assert adapted_result.signal_scores["danger"] != default_result.signal_scores["danger"]
