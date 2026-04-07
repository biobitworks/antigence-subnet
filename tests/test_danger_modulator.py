"""Tests for DangerTheoryModulator costimulation modulation.

Covers: DANGER-01 (costimulation uses max(pamp, danger)),
        DANGER-02 (scores stay in [0,1]),
        DANGER-03 (monotonicity preserved).
"""

from __future__ import annotations

import numpy as np

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.orchestrator.danger import DangerTheoryModulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(**kwargs: float) -> np.ndarray:
    """Build a 10-dim feature vector, defaulting all features to 0.0.

    Accepts keyword arguments keyed by feature name.
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
# Core Modulation Formula
# ---------------------------------------------------------------------------


class TestModulation:
    """DangerTheoryModulator.modulate() applies D-02 formula correctly."""

    def test_modulation_boosts_score(self):
        """raw=0.6, costim=0.8, alpha=0.3 -> 0.6 + (1-0.6)*0.8*0.3 = 0.696."""
        mod = DangerTheoryModulator(alpha=0.3)
        result = mod.modulate(0.6, 0.8)
        expected = 0.6 + (1.0 - 0.6) * 0.8 * 0.3
        assert abs(result - expected) < 1e-9

    def test_zero_costimulation_no_change(self):
        """raw=0.6, costim=0.0, alpha=0.3 -> 0.6 (no boost)."""
        mod = DangerTheoryModulator(alpha=0.3)
        result = mod.modulate(0.6, 0.0)
        assert abs(result - 0.6) < 1e-9

    def test_score_stays_in_unit_range(self):
        """raw=0.99, costim=1.0, alpha=0.3 -> 0.99 + 0.01*1.0*0.3 = 0.993."""
        mod = DangerTheoryModulator(alpha=0.3)
        result = mod.modulate(0.99, 1.0)
        expected = 0.99 + 0.01 * 1.0 * 0.3
        assert abs(result - expected) < 1e-9
        assert 0.0 <= result <= 1.0

    def test_monotonicity_preserved(self):
        """For any pair where raw_A > raw_B, modulated_A >= modulated_B (same costim)."""
        mod = DangerTheoryModulator(alpha=0.3)
        costim = 0.7
        raw_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        modulated = [mod.modulate(r, costim) for r in raw_scores]
        for i in range(len(modulated) - 1):
            assert modulated[i] <= modulated[i + 1], (
                f"Monotonicity violated: modulated[{i}]={modulated[i]} > modulated[{i + 1}]={modulated[i + 1]}"  # noqa: E501
            )

    def test_high_confidence_not_suppressed(self):
        """raw=0.95, any costim -> modulated >= 0.95 (formula only adds)."""
        mod = DangerTheoryModulator(alpha=0.3)
        for costim in [0.0, 0.3, 0.5, 0.8, 1.0]:
            result = mod.modulate(0.95, costim)
            assert result >= 0.95, f"High-confidence suppressed with costim={costim}: {result}"


# ---------------------------------------------------------------------------
# Costimulation Extraction
# ---------------------------------------------------------------------------


class TestCostimulation:
    """DangerTheoryModulator.costimulation() extracts max(pamp, danger)."""

    def test_costimulation_uses_max_pamp_danger(self):
        """Features with pamp=0.7, danger=0.3 -> costim = max(0.7, 0.3) = 0.7."""
        mod = DangerTheoryModulator()
        features = _make_features(pamp_score=0.7, danger_signal=0.3)
        costim = mod.costimulation(features)
        assert abs(costim - 0.7) < 1e-9

    def test_costimulation_danger_higher(self):
        """When danger_signal > pamp_score, costim = danger_signal."""
        mod = DangerTheoryModulator()
        features = _make_features(pamp_score=0.2, danger_signal=0.9)
        costim = mod.costimulation(features)
        assert abs(costim - 0.9) < 1e-9

    def test_costimulation_both_zero(self):
        """When both are zero, costim = 0.0."""
        mod = DangerTheoryModulator()
        features = _make_features(pamp_score=0.0, danger_signal=0.0)
        costim = mod.costimulation(features)
        assert costim == 0.0


# ---------------------------------------------------------------------------
# Batch Modulation
# ---------------------------------------------------------------------------


class TestModulateBatch:
    """modulate() applied to multiple scores, each clamped to [0, 1]."""

    def test_modulate_batch(self):
        """Modulate list of scores, each stays in [0, 1]."""
        mod = DangerTheoryModulator(alpha=0.3)
        raw_scores = [0.1, 0.5, 0.9, 0.99]
        costim = 0.8
        for raw in raw_scores:
            result = mod.modulate(raw, costim)
            assert 0.0 <= result <= 1.0, f"Score out of range: {result}"
            assert result >= raw, f"Score decreased: {result} < {raw}"


# ---------------------------------------------------------------------------
# Defaults and Configuration
# ---------------------------------------------------------------------------


class TestDefaults:
    """Default alpha and from_config factory."""

    def test_default_alpha(self):
        """DangerTheoryModulator() has alpha=0.3."""
        mod = DangerTheoryModulator()
        # Verify via modulation: raw=0.5, costim=1.0 -> 0.5 + 0.5*1.0*0.3 = 0.65
        result = mod.modulate(0.5, 1.0)
        expected = 0.5 + 0.5 * 1.0 * 0.3
        assert abs(result - expected) < 1e-9

    def test_alpha_zero_passthrough(self):
        """alpha=0.0 -> modulated == raw (no modulation)."""
        mod = DangerTheoryModulator(alpha=0.0)
        for raw in [0.1, 0.5, 0.9]:
            for costim in [0.0, 0.5, 1.0]:
                result = mod.modulate(raw, costim)
                assert abs(result - raw) < 1e-9, f"alpha=0 should passthrough: {result} != {raw}"

    def test_from_config(self):
        """DangerTheoryModulator.from_config({"alpha": 0.5, "enabled": True})."""
        mod = DangerTheoryModulator.from_config({"alpha": 0.5, "enabled": True})
        # Verify alpha=0.5: raw=0.5, costim=1.0 -> 0.5 + 0.5*1.0*0.5 = 0.75
        result = mod.modulate(0.5, 1.0)
        expected = 0.5 + 0.5 * 1.0 * 0.5
        assert abs(result - expected) < 1e-9

    def test_from_config_defaults(self):
        """from_config({}) uses alpha=0.3, enabled=True."""
        mod = DangerTheoryModulator.from_config({})
        result = mod.modulate(0.5, 1.0)
        expected = 0.5 + 0.5 * 1.0 * 0.3
        assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# Disabled Mode
# ---------------------------------------------------------------------------


class TestDisabledPassthrough:
    """When enabled=False, modulate() returns raw score unchanged."""

    def test_disabled_passthrough(self):
        """Disabled modulator returns raw score for all inputs."""
        mod = DangerTheoryModulator(alpha=0.3, enabled=False)
        for raw in [0.0, 0.3, 0.6, 0.95, 1.0]:
            for costim in [0.0, 0.5, 1.0]:
                result = mod.modulate(raw, costim)
                assert abs(result - raw) < 1e-9, (
                    f"Disabled modulator should passthrough: {result} != {raw}"
                )

    def test_disabled_modulate_result(self):
        """Disabled modulator returns DetectionResult with unchanged score."""
        mod = DangerTheoryModulator(alpha=0.3, enabled=False)
        original = DetectionResult(score=0.6, confidence=0.8, anomaly_type="test")
        features = _make_features(pamp_score=0.9, danger_signal=0.9)
        result = mod.modulate_result(original, features)
        assert abs(result.score - 0.6) < 1e-9
        assert result.confidence == 0.8
        assert result.anomaly_type == "test"


# ---------------------------------------------------------------------------
# modulate_result Integration
# ---------------------------------------------------------------------------


class TestModulateResult:
    """DangerTheoryModulator.modulate_result() wraps DetectionResult."""

    def test_modulate_result_boosts(self):
        """modulate_result applies modulation to DetectionResult.score."""
        mod = DangerTheoryModulator(alpha=0.3)
        original = DetectionResult(
            score=0.6,
            confidence=0.8,
            anomaly_type="ensemble",
            feature_attribution={"feat1": 0.5},
        )
        features = _make_features(pamp_score=0.7, danger_signal=0.3)
        result = mod.modulate_result(original, features)
        # costim = max(0.7, 0.3) = 0.7
        expected = 0.6 + (1.0 - 0.6) * 0.7 * 0.3
        assert abs(result.score - expected) < 1e-9
        assert result.confidence == 0.8
        assert result.anomaly_type == "ensemble"
        assert result.feature_attribution == {"feat1": 0.5}

    def test_modulate_result_preserves_none_attribution(self):
        """modulate_result preserves None feature_attribution."""
        mod = DangerTheoryModulator(alpha=0.3)
        original = DetectionResult(score=0.5, confidence=0.7, anomaly_type="test")
        features = _make_features(pamp_score=0.5)
        result = mod.modulate_result(original, features)
        assert result.feature_attribution is None
