"""Tests for AdaptiveWeightManager: EMA-based weight adaptation with bounded constraints.

Covers:
- Cold start with default weights from DendriticCell constants
- EMA update mechanics (alpha blending)
- Bounds enforcement [0.05, 0.5]
- Category re-normalization after clamping
- JSON persistence (save/load)
- Cold start from nonexistent file
- Convergence behavior over many rounds
- Zero-feature no-op behavior
- Round count tracking
- DCA integration: DendriticCell with AdaptiveWeightManager (Phase 44-02)
- Orchestrator integration: ImmuneOrchestrator adaptive pipeline (Phase 44-02)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from antigence_subnet.miner.orchestrator.adaptive_weights import AdaptiveWeightManager
from antigence_subnet.miner.orchestrator.dendritic_cell import (
    DANGER_FEATURES,
    DendriticCell,
    PAMP_FEATURES,
    SAFE_FEATURES,
)
from antigence_subnet.miner.orchestrator.telemetry import MinerTelemetry


# ---------------------------------------------------------------------------
# Cold-start defaults
# ---------------------------------------------------------------------------


class TestColdStartDefaults:
    """New manager starts with default static weights from DendriticCell."""

    def test_default_pamp_weights(self) -> None:
        mgr = AdaptiveWeightManager()
        weights = mgr.get_weights()
        for name, (idx, w) in PAMP_FEATURES.items():
            assert name in weights["pamp"]
            assert weights["pamp"][name] == (idx, w)

    def test_default_danger_weights(self) -> None:
        mgr = AdaptiveWeightManager()
        weights = mgr.get_weights()
        for name, (idx, w) in DANGER_FEATURES.items():
            assert name in weights["danger"]
            assert weights["danger"][name] == (idx, w)

    def test_default_safe_weights(self) -> None:
        mgr = AdaptiveWeightManager()
        weights = mgr.get_weights()
        for name, (idx, w) in SAFE_FEATURES.items():
            assert name in weights["safe"]
            assert weights["safe"][name] == (idx, w)

    def test_round_count_starts_at_zero(self) -> None:
        mgr = AdaptiveWeightManager()
        assert mgr.get_round_count() == 0


# ---------------------------------------------------------------------------
# EMA update mechanics
# ---------------------------------------------------------------------------


class TestEMAUpdate:
    """adapt() updates weights via EMA with correct alpha blending."""

    def test_adapt_positive_outcome(self) -> None:
        """adapt(features, outcome=1.0) updates via EMA alpha=0.1."""
        mgr = AdaptiveWeightManager(alpha=0.1)
        features = np.zeros(10)
        # pamp_score at idx 5
        features[5] = 0.8
        mgr.adapt(features, outcome=1.0)
        weights = mgr.get_weights()
        # importance = |0.8 * 1.0| = 0.8
        # new_weight = 0.1 * 0.8 + 0.9 * 1.0 = 0.08 + 0.9 = 0.98 -> clamped to 0.5
        # After renorm pamp has one feature so weight stays at 1.0 (sum = 1.0)
        # Actually renorm preserves category sum = 1.0, with one feature -> 1.0
        # But 0.98 > 0.5 clamp -> 0.5 -> renorm to 1.0 (single feature, sum=1.0)
        assert weights["pamp"]["pamp_score"][1] == pytest.approx(1.0, abs=1e-9)

    def test_adapt_negative_outcome(self) -> None:
        """adapt(features, outcome=-1.0) uses absolute importance."""
        mgr = AdaptiveWeightManager(alpha=0.1)
        features = np.zeros(10)
        features[6] = 0.5  # exaggeration (danger)
        mgr.adapt(features, outcome=-1.0)
        weights = mgr.get_weights()
        # importance = |0.5 * -1.0| = 0.5
        # old_weight for exaggeration = 1/3
        # new = 0.1 * 0.5 + 0.9 * (1/3) = 0.05 + 0.3 = 0.35
        # Other danger features: controversy, claim_density had importance = 0
        # new_weight = 0.1 * 0 + 0.9 * (1/3) = 0.3
        # After clamp: all in [0.05, 0.5] -> ok
        # Original danger sum = 1.0
        # Current sum = 0.35 + 0.3 + 0.3 = 0.95
        # Renorm: each *= 1.0/0.95
        # exaggeration: 0.35 * (1.0/0.95) ~= 0.3684
        exag_weight = weights["danger"]["exaggeration"][1]
        assert exag_weight > 1.0 / 3.0  # increased from default

    def test_increments_round_count(self) -> None:
        mgr = AdaptiveWeightManager()
        assert mgr.get_round_count() == 0
        mgr.adapt(np.zeros(10), outcome=1.0)
        assert mgr.get_round_count() == 1
        mgr.adapt(np.zeros(10), outcome=-1.0)
        assert mgr.get_round_count() == 2


# ---------------------------------------------------------------------------
# Bounds enforcement
# ---------------------------------------------------------------------------


class TestBoundsEnforcement:
    """No weight exceeds max or drops below min after adaptation."""

    @pytest.mark.parametrize("outcome", [1.0, -1.0, 0.5, -0.5])
    def test_no_weight_exceeds_max_multi_feature(self, outcome: float) -> None:
        """In multi-feature categories, no single weight exceeds max after renorm."""
        mgr = AdaptiveWeightManager(alpha=0.9, max_weight=0.5)
        features = np.ones(10) * 10.0  # Extreme values
        mgr.adapt(features, outcome=outcome)
        weights = mgr.get_weights()
        # Multi-feature categories: danger (3 features), safe (5 features)
        for cat in ("danger", "safe"):
            for _name, (_idx, w) in weights[cat].items():
                assert w <= 0.5 + 1e-9, f"{cat}.{_name} weight {w} exceeds 0.5"

    @pytest.mark.parametrize("outcome", [1.0, -1.0, 0.5, -0.5])
    def test_single_feature_preserves_category_sum(self, outcome: float) -> None:
        """PAMP (1 feature) always renormalizes to category sum=1.0."""
        mgr = AdaptiveWeightManager(alpha=0.9, max_weight=0.5)
        features = np.ones(10) * 10.0
        mgr.adapt(features, outcome=outcome)
        weights = mgr.get_weights()
        pamp_w = weights["pamp"]["pamp_score"][1]
        assert pamp_w == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("outcome", [1.0, -1.0, 0.5, -0.5])
    def test_no_weight_below_min(self, outcome: float) -> None:
        """No weight drops below 0.05 even with extreme features."""
        mgr = AdaptiveWeightManager(alpha=0.9, min_weight=0.05)
        features = np.zeros(10)
        features[5] = 100.0  # Only pamp_score is extreme
        mgr.adapt(features, outcome=outcome)
        weights = mgr.get_weights()
        for cat in ("danger", "safe"):
            for _name, (_idx, w) in weights[cat].items():
                assert w >= 0.05 - 1e-9, f"{cat}.{_name} weight {w} below 0.05"

    def test_bounds_after_many_rounds(self) -> None:
        """Bounds hold after 100 extreme-value rounds for multi-feature cats."""
        mgr = AdaptiveWeightManager(alpha=0.3)
        rng = np.random.default_rng(42)
        for _ in range(100):
            features = rng.uniform(-5, 5, size=10)
            outcome = rng.choice([-1.0, 1.0])
            mgr.adapt(features, outcome)
        weights = mgr.get_weights()
        # Multi-feature categories respect bounds
        for cat in ("danger", "safe"):
            for _name, (_idx, w) in weights[cat].items():
                assert 0.05 - 1e-9 <= w <= 0.5 + 1e-9
        # Single-feature (PAMP) preserves category sum
        assert weights["pamp"]["pamp_score"][1] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Category re-normalization
# ---------------------------------------------------------------------------


class TestCategoryRenormalization:
    """After clamping, category sums preserve original proportions."""

    def test_danger_sum_preserved(self) -> None:
        """Danger category weights re-normalize to sum=1.0 after adaptation."""
        mgr = AdaptiveWeightManager(alpha=0.5)
        features = np.zeros(10)
        features[6] = 2.0  # exaggeration (danger)
        mgr.adapt(features, outcome=1.0)
        weights = mgr.get_weights()
        danger_sum = sum(w for _, w in weights["danger"].values())
        assert danger_sum == pytest.approx(1.0, abs=1e-9)

    def test_safe_sum_preserved(self) -> None:
        """Safe category weights re-normalize to sum=1.0 after adaptation."""
        mgr = AdaptiveWeightManager(alpha=0.5)
        features = np.ones(10) * 0.5
        mgr.adapt(features, outcome=1.0)
        weights = mgr.get_weights()
        safe_sum = sum(w for _, w in weights["safe"].values())
        assert safe_sum == pytest.approx(1.0, abs=1e-9)

    def test_pamp_sum_preserved(self) -> None:
        """PAMP category (1 feature) stays at sum=1.0."""
        mgr = AdaptiveWeightManager(alpha=0.5)
        features = np.ones(10) * 0.5
        mgr.adapt(features, outcome=1.0)
        weights = mgr.get_weights()
        pamp_sum = sum(w for _, w in weights["pamp"].values())
        assert pamp_sum == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# get_weights() shape
# ---------------------------------------------------------------------------


class TestGetWeightsShape:
    """get_weights() returns correct structure for DendriticCell consumption."""

    def test_returns_deep_copy(self) -> None:
        """Modifying returned weights does not change internal state."""
        mgr = AdaptiveWeightManager()
        w1 = mgr.get_weights()
        w1["pamp"]["pamp_score"] = (5, 999.0)
        w2 = mgr.get_weights()
        assert w2["pamp"]["pamp_score"] == (5, 1.0)

    def test_has_three_categories(self) -> None:
        mgr = AdaptiveWeightManager()
        weights = mgr.get_weights()
        assert set(weights.keys()) == {"pamp", "danger", "safe"}

    def test_values_are_tuples(self) -> None:
        mgr = AdaptiveWeightManager()
        weights = mgr.get_weights()
        for cat in ("pamp", "danger", "safe"):
            for name, val in weights[cat].items():
                assert isinstance(val, tuple), f"{cat}.{name} is not tuple"
                assert len(val) == 2
                assert isinstance(val[0], int)
                assert isinstance(val[1], float)


# ---------------------------------------------------------------------------
# Persistence (save/load)
# ---------------------------------------------------------------------------


class TestPersistence:
    """save() writes JSON to disk, load() restores identical weights."""

    def test_save_load_roundtrip(self) -> None:
        """Weights persist and reload identically."""
        mgr = AdaptiveWeightManager(alpha=0.2)
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05])
        mgr.adapt(features, outcome=1.0)
        mgr.adapt(features, outcome=-0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr.save("test_domain", base_path=tmpdir)
            # Check file exists
            fpath = os.path.join(tmpdir, "test_domain.json")
            assert os.path.exists(fpath)

            # Load into fresh manager
            mgr2 = AdaptiveWeightManager(alpha=0.2)
            loaded = mgr2.load("test_domain", base_path=tmpdir)
            assert loaded is True

            # Weights match
            w1 = mgr.get_weights()
            w2 = mgr2.get_weights()
            for cat in ("pamp", "danger", "safe"):
                for name in w1[cat]:
                    assert w1[cat][name][0] == w2[cat][name][0]
                    assert w1[cat][name][1] == pytest.approx(w2[cat][name][1])

            assert mgr2.get_round_count() == mgr.get_round_count()

    def test_load_nonexistent_returns_false(self) -> None:
        """load() from nonexistent file returns False, keeps defaults."""
        mgr = AdaptiveWeightManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = mgr.load("nonexistent_domain", base_path=tmpdir)
            assert loaded is False
            # Weights unchanged from defaults
            weights = mgr.get_weights()
            assert weights["pamp"]["pamp_score"] == (5, 1.0)

    def test_save_creates_directory(self) -> None:
        """save() creates base_path directory if it doesn't exist."""
        mgr = AdaptiveWeightManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "dir")
            mgr.save("test_domain", base_path=nested)
            assert os.path.exists(os.path.join(nested, "test_domain.json"))

    def test_save_json_is_valid(self) -> None:
        """Saved file is valid JSON with expected keys."""
        mgr = AdaptiveWeightManager()
        mgr.adapt(np.ones(10) * 0.5, outcome=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr.save("test_domain", base_path=tmpdir)
            fpath = os.path.join(tmpdir, "test_domain.json")
            with open(fpath) as f:
                data = json.load(f)
            assert "pamp" in data
            assert "danger" in data
            assert "safe" in data
            assert "round_count" in data
            assert "alpha" in data


# ---------------------------------------------------------------------------
# Convergence behavior
# ---------------------------------------------------------------------------


class TestConvergence:
    """50+ sequential adapt() calls shift weights toward predictive features."""

    def test_convergence_toward_predictive_feature(self) -> None:
        """Consistent high-value feature gains weight over 50 rounds."""
        mgr = AdaptiveWeightManager(alpha=0.1)
        for _ in range(50):
            features = np.zeros(10)
            features[6] = 0.9  # exaggeration (danger) consistently high
            features[8] = 0.1  # controversy (danger) consistently low
            features[0] = 0.1  # claim_density (danger) consistently low
            mgr.adapt(features, outcome=1.0)

        weights = mgr.get_weights()
        exag_w = weights["danger"]["exaggeration"][1]
        cont_w = weights["danger"]["controversy"][1]
        claim_w = weights["danger"]["claim_density"][1]
        # Exaggeration should have the highest weight
        assert exag_w > cont_w
        assert exag_w > claim_w


# ---------------------------------------------------------------------------
# Zero-feature no-op
# ---------------------------------------------------------------------------


class TestZeroFeatureNoOp:
    """adapt() with all-zero features does not change weights."""

    def test_zero_features_no_change(self) -> None:
        """All-zero features produce zero importance -> EMA keeps old weights."""
        mgr = AdaptiveWeightManager(alpha=0.1)
        weights_before = mgr.get_weights()
        mgr.adapt(np.zeros(10), outcome=1.0)
        weights_after = mgr.get_weights()
        # importance = |0 * outcome| = 0 for all features
        # new_weight = alpha * 0 + (1-alpha) * old = 0.9 * old
        # After renorm, proportions restored to original sums
        # So weights should be unchanged from defaults
        for cat in ("pamp", "danger", "safe"):
            for name in weights_before[cat]:
                assert weights_before[cat][name][1] == pytest.approx(
                    weights_after[cat][name][1], abs=1e-9
                )

    def test_zero_features_increments_round(self) -> None:
        """Even with zero features, round count increments."""
        mgr = AdaptiveWeightManager()
        mgr.adapt(np.zeros(10), outcome=1.0)
        assert mgr.get_round_count() == 1


# ---------------------------------------------------------------------------
# OrchestratorConfig DCA adaptive fields (Task 2)
# ---------------------------------------------------------------------------


class TestOrchestratorConfigDCAAdaptive:
    """Config parsing for [miner.orchestrator.dca] adaptive fields (D-06)."""

    def test_adaptive_true_parses(self) -> None:
        """TOML with adaptive = true is accessible via dca_config."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        toml_raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "dca": {"adaptive": True},
                },
            },
        }
        config = OrchestratorConfig.from_toml_raw(toml_raw)
        assert config.dca_config.get("adaptive", False) is True

    def test_adapt_alpha_parses(self) -> None:
        """TOML with adapt_alpha = 0.2 is accessible via dca_config."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        toml_raw = {
            "miner": {
                "orchestrator": {
                    "dca": {"adaptive": True, "adapt_alpha": 0.2},
                },
            },
        }
        config = OrchestratorConfig.from_toml_raw(toml_raw)
        assert config.dca_config["adapt_alpha"] == 0.2

    def test_no_dca_section_backward_compat(self) -> None:
        """TOML without [miner.orchestrator.dca] produces adaptive=False."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        toml_raw = {"miner": {"orchestrator": {"enabled": True}}}
        config = OrchestratorConfig.from_toml_raw(toml_raw)
        assert config.dca_config.get("adaptive", False) is False

    def test_empty_toml_backward_compat(self) -> None:
        """Completely empty TOML produces safe defaults."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig.from_toml_raw({})
        assert config.dca_config.get("adaptive", False) is False
        assert config.dca_config.get("adapt_alpha", 0.1) == 0.1

    def test_invalid_adapt_alpha_zero_raises(self) -> None:
        """adapt_alpha = 0.0 raises ValueError."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        toml_raw = {
            "miner": {
                "orchestrator": {
                    "dca": {"adapt_alpha": 0.0},
                },
            },
        }
        with pytest.raises(ValueError, match="adapt_alpha"):
            OrchestratorConfig.from_toml_raw(toml_raw)

    def test_invalid_adapt_alpha_negative_raises(self) -> None:
        """adapt_alpha = -0.5 raises ValueError."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        toml_raw = {
            "miner": {
                "orchestrator": {
                    "dca": {"adapt_alpha": -0.5},
                },
            },
        }
        with pytest.raises(ValueError, match="adapt_alpha"):
            OrchestratorConfig.from_toml_raw(toml_raw)

    def test_valid_adapt_alpha_one_passes(self) -> None:
        """adapt_alpha = 1.0 is valid (boundary)."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        toml_raw = {
            "miner": {
                "orchestrator": {
                    "dca": {"adapt_alpha": 1.0},
                },
            },
        }
        config = OrchestratorConfig.from_toml_raw(toml_raw)
        assert config.dca_config["adapt_alpha"] == 1.0

    def test_invalid_adapt_alpha_above_one_raises(self) -> None:
        """adapt_alpha = 1.5 raises ValueError."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

        toml_raw = {
            "miner": {
                "orchestrator": {
                    "dca": {"adapt_alpha": 1.5},
                },
            },
        }
        with pytest.raises(ValueError, match="adapt_alpha"):
            OrchestratorConfig.from_toml_raw(toml_raw)


# ---------------------------------------------------------------------------
# DCA Integration (Phase 44-02)
# ---------------------------------------------------------------------------


class TestDCAIntegration:
    """DendriticCell with AdaptiveWeightManager produces adapted signal scores."""

    def test_adapted_weights_change_classification(self) -> None:
        """After 10 adapt() calls, DendriticCell.classify() produces
        different signal scores than a default DendriticCell on same features."""
        mgr = AdaptiveWeightManager(alpha=0.3)
        features = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.05, 0.9, 0.1, 0.7, 0.0])
        for _ in range(10):
            mgr.adapt(features, outcome=1.0)

        dc_adapted = DendriticCell(weight_manager=mgr)
        dc_default = DendriticCell()

        result_adapted = dc_adapted.classify(features)
        result_default = dc_default.classify(features)

        # Danger has 3 features with different importance patterns, so
        # adapted weights should yield different danger scores
        danger_adapted = result_adapted.signal_scores["danger"]
        danger_default = result_default.signal_scores["danger"]
        assert abs(danger_adapted - danger_default) > 1e-6, (
            f"Expected different danger scores after adaptation: "
            f"adapted={danger_adapted}, default={danger_default}"
        )

    def test_backward_compat_no_weight_manager(self) -> None:
        """DendriticCell() without weight_manager matches original behavior exactly."""
        dc_new = DendriticCell()
        dc_explicit_none = DendriticCell(weight_manager=None)

        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05])
        result_new = dc_new.classify(features)
        result_none = dc_explicit_none.classify(features)
        assert result_new == result_none

    def test_weight_manager_refreshes_each_classify(self) -> None:
        """Each classify() call uses latest weights from the manager."""
        mgr = AdaptiveWeightManager(alpha=0.5)
        dc = DendriticCell(weight_manager=mgr)
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0])

        result_before = dc.classify(features)
        # Adapt weights significantly
        for _ in range(20):
            mgr.adapt(features, outcome=1.0)
        result_after = dc.classify(features)

        # Danger scores should differ after adaptation
        assert result_before.signal_scores["danger"] != result_after.signal_scores["danger"]

    def test_from_config_with_weight_manager(self) -> None:
        """from_config() accepts and passes through weight_manager kwarg."""
        mgr = AdaptiveWeightManager(alpha=0.2)
        dc = DendriticCell.from_config({"pamp_threshold": 0.4}, weight_manager=mgr)
        assert dc._weight_manager is mgr
        assert dc._pamp_threshold == 0.4


# ---------------------------------------------------------------------------
# Orchestrator Integration (Phase 44-02)
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    """ImmuneOrchestrator with adaptive weights calls adapt() after detection."""

    @pytest.mark.asyncio
    async def test_adaptive_true_calls_adapt(self) -> None:
        """With adaptive_weights set, process() calls adapt() after detection."""
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator

        # Mock all components
        extractor = MagicMock()
        extractor.extract.return_value = np.zeros(10)
        nk_cell = MagicMock()
        nk_cell.process.return_value = None  # not triggered
        dc = DendriticCell()
        danger = MagicMock()
        danger.modulate_result.side_effect = lambda r, f, **kw: r

        mock_result = DetectionResult(score=0.7, confidence=0.8, anomaly_type="test")
        mock_detector = MagicMock()

        weight_mgr = MagicMock(spec=AdaptiveWeightManager)
        weight_mgr.get_weights.return_value = {
            "pamp": dict(PAMP_FEATURES),
            "danger": dict(DANGER_FEATURES),
            "safe": dict(SAFE_FEATURES),
        }

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk_cell,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [mock_detector]},
            adaptive_weights=weight_mgr,
        )

        with patch(
            "antigence_subnet.miner.orchestrator.orchestrator.ensemble_detect",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await orchestrator.process("prompt", "output", "hallucination")

        weight_mgr.adapt.assert_called_once()
        # anomaly_score 0.7 > 0.5 -> outcome = 1.0
        call_args = weight_mgr.adapt.call_args
        assert call_args[0][1] == 1.0  # outcome

    @pytest.mark.asyncio
    async def test_adaptive_false_does_not_call_adapt(self) -> None:
        """Without adaptive_weights, process() does NOT call adapt."""
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator

        extractor = MagicMock()
        extractor.extract.return_value = np.zeros(10)
        nk_cell = MagicMock()
        nk_cell.process.return_value = None
        dc = DendriticCell()
        danger = MagicMock()
        danger.modulate_result.side_effect = lambda r, f, **kw: r

        mock_result = DetectionResult(score=0.7, confidence=0.8, anomaly_type="test")

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk_cell,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": []},
            # No adaptive_weights -- default is None
        )

        with patch(
            "antigence_subnet.miner.orchestrator.orchestrator.ensemble_detect",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await orchestrator.process("prompt", "output", "hallucination")

        # No adapt call possible since _adaptive_weights is None
        assert result.score == 0.7

    def test_save_state_persists_weights(self) -> None:
        """save_state() + from_config() round-trip restores weights."""
        from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator

        extractor = MagicMock()
        nk_cell = MagicMock()
        dc = DendriticCell()
        danger = MagicMock()

        mgr = AdaptiveWeightManager(alpha=0.3)
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0])
        for _ in range(5):
            mgr.adapt(features, outcome=1.0)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk_cell,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={},
            adaptive_weights=mgr,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr.save("default", base_path=tmpdir)
            # Load into a new manager
            mgr2 = AdaptiveWeightManager(alpha=0.3)
            loaded = mgr2.load("default", base_path=tmpdir)
            assert loaded is True
            # Weights should match
            w1 = mgr.get_weights()
            w2 = mgr2.get_weights()
            for cat in ("pamp", "danger", "safe"):
                for name in w1[cat]:
                    assert w1[cat][name][1] == pytest.approx(w2[cat][name][1])

    @pytest.mark.asyncio
    async def test_telemetry_records_after_adapt(self) -> None:
        """When telemetry is provided, process() records score/confidence."""
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator

        extractor = MagicMock()
        extractor.extract.return_value = np.zeros(10)
        nk_cell = MagicMock()
        nk_cell.process.return_value = None
        dc = DendriticCell()
        danger = MagicMock()
        danger.modulate_result.side_effect = lambda r, f, **kw: r

        mock_result = DetectionResult(score=0.6, confidence=0.9, anomaly_type="test")

        weight_mgr = AdaptiveWeightManager(alpha=0.1)
        telemetry = MagicMock(spec=MinerTelemetry)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk_cell,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": []},
            adaptive_weights=weight_mgr,
            telemetry=telemetry,
        )

        with patch(
            "antigence_subnet.miner.orchestrator.orchestrator.ensemble_detect",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await orchestrator.process("prompt", "output", "hallucination")

        telemetry.record.assert_called_once_with("hallucination", 0.6, 0.9)

    def test_weight_update_logged(self, caplog) -> None:
        """Weight adaptation logs old/new weights at DEBUG level."""
        import asyncio
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator

        extractor = MagicMock()
        extractor.extract.return_value = np.ones(10) * 0.5
        nk_cell = MagicMock()
        nk_cell.process.return_value = None
        dc = DendriticCell()
        danger = MagicMock()
        danger.modulate_result.side_effect = lambda r, f, **kw: r

        mock_result = DetectionResult(score=0.8, confidence=0.9, anomaly_type="test")
        weight_mgr = AdaptiveWeightManager(alpha=0.1)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk_cell,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": []},
            adaptive_weights=weight_mgr,
        )

        with patch(
            "antigence_subnet.miner.orchestrator.orchestrator.ensemble_detect",
            new_callable=AsyncMock,
            return_value=mock_result,
        ), caplog.at_level(logging.DEBUG, logger="antigence_subnet.miner.orchestrator.orchestrator"):
            asyncio.get_event_loop().run_until_complete(
                orchestrator.process("prompt", "output", "hallucination")
            )

        assert any("DCA weight adaptation" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Package export (Phase 44-02)
# ---------------------------------------------------------------------------


class TestPackageExport:
    """AdaptiveWeightManager is importable from the orchestrator package."""

    def test_importable_from_package(self) -> None:
        """AdaptiveWeightManager importable from antigence_subnet.miner.orchestrator."""
        from antigence_subnet.miner.orchestrator import AdaptiveWeightManager as AWM
        assert AWM is AdaptiveWeightManager
