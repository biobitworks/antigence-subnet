"""Tests for ImmuneOrchestrator coordinator pipeline.

Covers: D-06 (orchestrator pipeline), NK gate short-circuit, DCA tier routing,
Danger Theory modulation application, from_config factory, per-domain config
application (Phase 36).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.orchestrator.config import DomainConfig, OrchestratorConfig
from antigence_subnet.miner.orchestrator.danger import DangerTheoryModulator
from antigence_subnet.miner.orchestrator.dendritic_cell import DCAResult, DendriticCell
from antigence_subnet.miner.orchestrator.nk_cell import FeatureStatistics, NKCell
from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(**kwargs: float) -> np.ndarray:
    """Build a 10-dim feature vector, defaulting all features to 0.0."""
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


def _mock_extractor(features: np.ndarray | None = None):
    """Create a mock DendriticFeatureExtractor."""
    if features is None:
        features = np.zeros(10, dtype=np.float64)
    extractor = MagicMock()
    extractor.extract.return_value = features
    return extractor


def _mock_nk_cell(result: DetectionResult | None = None):
    """Create a mock NKCell. Returns given result or None."""
    nk = MagicMock(spec=NKCell)
    nk.process.return_value = result
    return nk


def _mock_dendritic_cell(dca_result: DCAResult | None = None):
    """Create a mock DendriticCell returning given DCAResult."""
    if dca_result is None:
        dca_result = DCAResult(
            maturation_state="semi-mature",
            signal_scores={"pamp": 0.1, "danger": 0.3, "safe": 0.2},
            recommended_tier=["ocsvm", "negsel"],
        )
    dc = MagicMock(spec=DendriticCell)
    dc.classify.return_value = dca_result
    return dc


def _mock_detector(name: str, score: float = 0.5, confidence: float = 0.8) -> BaseDetector:
    """Create a mock BaseDetector with given score."""
    detector = MagicMock(spec=BaseDetector)
    detector.__class__.__name__ = name
    type(detector).__name__ = name
    detector.detect = AsyncMock(
        return_value=DetectionResult(
            score=score, confidence=confidence, anomaly_type="test",
        )
    )
    return detector


# ---------------------------------------------------------------------------
# NK Gate Short-Circuit
# ---------------------------------------------------------------------------

class TestNKGateShortCircuit:
    """When NKCell.process() returns a DetectionResult, orchestrator returns immediately."""

    @pytest.mark.asyncio
    async def test_nk_gate_short_circuits(self):
        """NK Cell returns DetectionResult -> orchestrator returns it without DCA/detectors."""
        nk_result = DetectionResult(
            score=1.0, confidence=0.9, anomaly_type="nk_fast_path",
        )
        features = _make_features(pamp_score=0.8)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(nk_result)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.3)
        detector = _mock_detector("OCSVMDetector")

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        assert result.score == 1.0
        assert result.anomaly_type == "nk_fast_path"
        # DCA and detectors should NOT have been called
        dc.classify.assert_not_called()
        detector.detect.assert_not_called()


# ---------------------------------------------------------------------------
# DCA Tier Routing
# ---------------------------------------------------------------------------

class TestDCATierRouting:
    """DCA classifies features and orchestrator selects detectors from recommended_tier."""

    @pytest.mark.asyncio
    async def test_dca_routes_to_tier(self):
        """When NK defers (None), DCA classifies features, orchestrator selects detectors."""
        dca_result = DCAResult(
            maturation_state="semi-mature",
            signal_scores={"pamp": 0.1, "danger": 0.3, "safe": 0.2},
            recommended_tier=["ocsvm"],
        )
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)  # NK defers
        dc = _mock_dendritic_cell(dca_result)
        danger = DangerTheoryModulator(alpha=0.0)  # passthrough

        ocsvm_det = _mock_detector("OCSVMDetector", score=0.6)
        negsel_det = _mock_detector("NegSelAISDetector", score=0.4)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [ocsvm_det, negsel_det]},
        )

        await orchestrator.process("prompt", "output", "hallucination")
        # Only OCSVM should have been called (tier=["ocsvm"])
        ocsvm_det.detect.assert_called_once()
        negsel_det.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_immature_routes_ocsvm_only(self):
        """Immature maturation -> only OCSVM detector runs."""
        dca_result = DCAResult(
            maturation_state="immature",
            signal_scores={"pamp": 0.0, "danger": 0.0, "safe": 0.8},
            recommended_tier=["ocsvm"],
        )
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell(dca_result)
        danger = DangerTheoryModulator(alpha=0.0)

        ocsvm_det = _mock_detector("OCSVMDetector", score=0.3)
        negsel_det = _mock_detector("NegSelAISDetector", score=0.5)
        iforest_det = _mock_detector("IsolationForestDetector", score=0.4)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [ocsvm_det, negsel_det, iforest_det]},
        )

        await orchestrator.process("prompt", "output", "hallucination")
        ocsvm_det.detect.assert_called_once()
        negsel_det.detect.assert_not_called()
        iforest_det.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_mature_routes_full_ensemble(self):
        """Mature maturation with empty tier -> all domain detectors run."""
        dca_result = DCAResult(
            maturation_state="mature",
            signal_scores={"pamp": 0.5, "danger": 0.3, "safe": 0.1},
            recommended_tier=[],  # empty = full ensemble
        )
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell(dca_result)
        danger = DangerTheoryModulator(alpha=0.0)

        ocsvm_det = _mock_detector("OCSVMDetector", score=0.7)
        negsel_det = _mock_detector("NegSelAISDetector", score=0.6)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [ocsvm_det, negsel_det]},
        )

        await orchestrator.process("prompt", "output", "hallucination")
        # All detectors should run
        ocsvm_det.detect.assert_called_once()
        negsel_det.detect.assert_called_once()


# ---------------------------------------------------------------------------
# Danger Theory Modulation
# ---------------------------------------------------------------------------

class TestDangerModulation:
    """After detector execution, DangerTheoryModulator is applied."""

    @pytest.mark.asyncio
    async def test_danger_modulation_applied(self):
        """Danger modulation boosts ensemble score based on costimulation."""
        features = _make_features(pamp_score=0.8, danger_signal=0.3)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dca_result = DCAResult(
            maturation_state="mature",
            signal_scores={"pamp": 0.8, "danger": 0.3, "safe": 0.0},
            recommended_tier=[],
        )
        dc = _mock_dendritic_cell(dca_result)
        danger = DangerTheoryModulator(alpha=0.3)

        detector = _mock_detector("OCSVMDetector", score=0.6)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        # costim = max(0.8, 0.3) = 0.8
        # modulated = 0.6 + (1-0.6)*0.8*0.3 = 0.6 + 0.096 = 0.696
        expected = 0.6 + (1.0 - 0.6) * 0.8 * 0.3
        assert abs(result.score - expected) < 1e-9

    @pytest.mark.asyncio
    async def test_danger_disabled_no_modulation(self):
        """When danger_config.enabled=False, raw ensemble score returned unchanged."""
        features = _make_features(pamp_score=0.9, danger_signal=0.9)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dca_result = DCAResult(
            maturation_state="mature",
            signal_scores={"pamp": 0.9, "danger": 0.3, "safe": 0.0},
            recommended_tier=[],
        )
        dc = _mock_dendritic_cell(dca_result)
        danger = DangerTheoryModulator(alpha=0.3, enabled=False)

        detector = _mock_detector("OCSVMDetector", score=0.6)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        assert abs(result.score - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# Return Type and Feature Extraction
# ---------------------------------------------------------------------------

class TestProcessContract:
    """process() returns DetectionResult and calls feature extraction."""

    @pytest.mark.asyncio
    async def test_process_returns_detection_result(self):
        """process() always returns a DetectionResult with score, confidence, anomaly_type."""
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0)
        detector = _mock_detector("OCSVMDetector", score=0.5, confidence=0.7)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        assert isinstance(result, DetectionResult)
        assert hasattr(result, "score")
        assert hasattr(result, "confidence")
        assert hasattr(result, "anomaly_type")

    @pytest.mark.asyncio
    async def test_feature_extraction_called(self):
        """DendriticFeatureExtractor.extract() called with output text."""
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0)
        detector = _mock_detector("OCSVMDetector")

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
        )

        await orchestrator.process("prompt", "output text", "hallucination")
        extractor.extract.assert_called_once_with("output text")

    @pytest.mark.asyncio
    async def test_feature_extraction_with_code(self):
        """When code is provided, extract receives output + code concatenated."""
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0)
        detector = _mock_detector("OCSVMDetector")

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"code_security": [detector]},
        )

        await orchestrator.process("prompt", "output", "code_security", code="x=1")
        extractor.extract.assert_called_once_with("output\nx=1")


# ---------------------------------------------------------------------------
# BCellStub Integration
# ---------------------------------------------------------------------------

class TestBCellStub:
    """BCellStub.process() returns None, orchestrator ignores it."""

    @pytest.mark.asyncio
    async def test_bcell_stub_no_effect(self):
        """BCellStub returning None does not affect the pipeline."""
        from antigence_subnet.miner.orchestrator.cells import BCellStub

        bcell = BCellStub()
        assert bcell.process(np.zeros(10), "prompt", "output") is None
        # The orchestrator pipeline does not use BCellStub directly
        # (it's not wired into the process() flow), but confirm it's harmless


# ---------------------------------------------------------------------------
# from_config Factory
# ---------------------------------------------------------------------------

class TestFromConfig:
    """ImmuneOrchestrator.from_config() creates orchestrator from OrchestratorConfig."""

    @pytest.mark.asyncio
    async def test_orchestrator_from_config(self):
        """from_config creates orchestrator with NK, DCA, danger modulator."""
        config = OrchestratorConfig(
            enabled=True,
            nk_config={},
            dca_config={"pamp_threshold": 0.3},
            danger_config={"alpha": 0.3, "enabled": True},
        )
        detectors = {"hallucination": [_mock_detector("OCSVMDetector")]}

        orchestrator = ImmuneOrchestrator.from_config(config, detectors)
        assert isinstance(orchestrator, ImmuneOrchestrator)

        # Verify it can process (smoke test)
        result = await orchestrator.process(
            "test prompt", "test output", "hallucination",
        )
        assert isinstance(result, DetectionResult)


# ---------------------------------------------------------------------------
# NK Cell per-request z_threshold override
# ---------------------------------------------------------------------------

class TestNKCellOverride:
    """NKCell.process() accepts optional z_threshold kwarg."""

    def test_override_z_threshold_triggers_on_lower_z(self):
        """NKCell.process() with z_threshold=1.0 override triggers on z=1.5
        that would not trigger at default 3.0."""
        stats = [
            FeatureStatistics(
                name="feat_a", index=0, mean=0.0, std=1.0,
                is_binary=False, is_constant=False,
            ),
        ]
        cell = NKCell(feature_stats=stats, z_threshold=3.0)

        features = np.array([1.5])  # z=1.5

        # Default threshold 3.0: should NOT trigger
        result_default = cell.process(features, "p", "o")
        assert result_default is None

        # Override threshold 1.0: should trigger
        result_override = cell.process(features, "p", "o", z_threshold=1.0)
        assert result_override is not None
        assert result_override.score == 1.0
        assert result_override.anomaly_type == "nk_fast_path"

    def test_no_override_uses_instance_threshold(self):
        """NKCell.process() without z_threshold override uses self._z_threshold."""
        stats = [
            FeatureStatistics(
                name="feat_a", index=0, mean=0.0, std=1.0,
                is_binary=False, is_constant=False,
            ),
        ]
        cell = NKCell(feature_stats=stats, z_threshold=3.0)
        features = np.array([3.01])  # z=3.01 > 3.0
        result = cell.process(features, "p", "o")
        assert result is not None
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# DendriticCell per-request pamp_threshold override
# ---------------------------------------------------------------------------

class TestDendriticCellOverride:
    """DendriticCell.classify() accepts optional pamp_threshold kwarg."""

    def test_override_pamp_threshold_classifies_as_mature(self):
        """DendriticCell.classify() with pamp_threshold=0.1 override classifies
        as mature where default 0.3 would not."""
        dc = DendriticCell(pamp_threshold=0.3)

        # Features with pamp_score=0.2 (index 5)
        features = _make_features(pamp_score=0.2)

        # Default 0.3: pamp=0.2 < 0.3 -> NOT mature
        result_default = dc.classify(features)
        assert result_default.maturation_state != "mature"

        # Override 0.1: pamp=0.2 >= 0.1 -> mature
        result_override = dc.classify(features, pamp_threshold=0.1)
        assert result_override.maturation_state == "mature"

    def test_no_override_uses_instance_threshold(self):
        """DendriticCell.classify() without pamp_threshold override uses
        self._pamp_threshold."""
        dc = DendriticCell(pamp_threshold=0.3)
        features = _make_features(pamp_score=0.5)
        result = dc.classify(features)
        assert result.maturation_state == "mature"


# ---------------------------------------------------------------------------
# DangerTheoryModulator per-request alpha/enabled override
# ---------------------------------------------------------------------------

class TestDangerModulatorOverride:
    """DangerTheoryModulator.modulate_result() accepts optional alpha/enabled kwargs."""

    def test_override_alpha_produces_higher_modulation(self):
        """modulate_result() with alpha=0.8 override produces higher modulation
        than default 0.3."""
        danger = DangerTheoryModulator(alpha=0.3, enabled=True)
        features = _make_features(pamp_score=0.8, danger_signal=0.3)
        raw_result = DetectionResult(score=0.5, confidence=0.8, anomaly_type="test")

        result_default = danger.modulate_result(raw_result, features)
        result_override = danger.modulate_result(raw_result, features, alpha=0.8)

        assert result_override.score > result_default.score

    def test_override_enabled_false_returns_raw(self):
        """modulate_result() with enabled=False override returns raw score."""
        danger = DangerTheoryModulator(alpha=0.3, enabled=True)
        features = _make_features(pamp_score=0.8, danger_signal=0.3)
        raw_result = DetectionResult(score=0.5, confidence=0.8, anomaly_type="test")

        result = danger.modulate_result(raw_result, features, enabled=False)
        assert abs(result.score - 0.5) < 1e-9

    def test_no_override_uses_instance_values(self):
        """modulate_result() without overrides uses self._alpha/self._enabled."""
        danger = DangerTheoryModulator(alpha=0.3, enabled=True)
        features = _make_features(pamp_score=0.8, danger_signal=0.3)
        raw_result = DetectionResult(score=0.5, confidence=0.8, anomaly_type="test")

        result = danger.modulate_result(raw_result, features)
        # costim = max(0.8, 0.3) = 0.8
        # modulated = 0.5 + (1-0.5)*0.8*0.3 = 0.5 + 0.12 = 0.62
        expected = 0.5 + (1.0 - 0.5) * 0.8 * 0.3
        assert abs(result.score - expected) < 1e-9


# ---------------------------------------------------------------------------
# ImmuneOrchestrator per-domain config application
# ---------------------------------------------------------------------------

class TestOrchestratorDomainConfig:
    """ImmuneOrchestrator.process() applies domain-specific config overrides."""

    @pytest.mark.asyncio
    async def test_domain_config_nk_threshold_applied(self):
        """ImmuneOrchestrator.process() with domain_configs['hallucination'].nk_z_threshold=1.0
        applies that threshold for hallucination domain."""
        # NK Cell with real stats -- feat_a at index 0 with mean=0, std=1
        stats = [
            FeatureStatistics(
                name="feat_a", index=0, mean=0.0, std=1.0,
                is_binary=False, is_constant=False,
            ),
        ]
        nk = NKCell(feature_stats=stats, z_threshold=3.0)  # default 3.0

        # Features where feat_a z=1.5 (triggers at 1.0 but not 3.0)
        features = np.zeros(10, dtype=np.float64)
        features[0] = 1.5

        extractor = _mock_extractor(features)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0)
        detector = _mock_detector("OCSVMDetector")

        config = OrchestratorConfig(
            enabled=True,
            domain_configs={
                "hallucination": DomainConfig(nk_z_threshold=1.0),
            },
        )

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
            config=config,
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        # NK should trigger at z=1.5 with threshold=1.0
        assert result.score == 1.0
        assert result.anomaly_type == "nk_fast_path"

    @pytest.mark.asyncio
    async def test_unconfigured_domain_uses_global_defaults(self):
        """ImmuneOrchestrator.process() with unconfigured domain uses global defaults."""
        stats = [
            FeatureStatistics(
                name="feat_a", index=0, mean=0.0, std=1.0,
                is_binary=False, is_constant=False,
            ),
        ]
        nk = NKCell(feature_stats=stats, z_threshold=3.0)

        features = np.zeros(10, dtype=np.float64)
        features[0] = 1.5  # z=1.5, below default 3.0

        extractor = _mock_extractor(features)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0)
        detector = _mock_detector("OCSVMDetector")

        config = OrchestratorConfig(
            enabled=True,
            domain_configs={
                "hallucination": DomainConfig(nk_z_threshold=1.0),
            },
        )

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"code_security": [detector]},
            config=config,
        )

        # code_security not in domain_configs -> uses global default z_threshold=3.0
        result = await orchestrator.process("prompt", "output", "code_security")
        # z=1.5 < 3.0 -> NK should NOT trigger
        assert result.anomaly_type != "nk_fast_path"

    @pytest.mark.asyncio
    async def test_domain_not_in_configs_falls_back(self):
        """ImmuneOrchestrator.process() with domain_configs but domain not in
        configs falls back to global defaults."""
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.3, enabled=True)
        detector = _mock_detector("OCSVMDetector", score=0.5)

        config = OrchestratorConfig(
            enabled=True,
            domain_configs={
                "hallucination": DomainConfig(danger_alpha=0.8),
            },
        )

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"reasoning": [detector]},
            config=config,
        )

        # reasoning not in domain_configs -> uses global danger alpha=0.3
        result = await orchestrator.process("prompt", "output", "reasoning")
        # With zero features, costim=0, so modulation has no effect regardless
        # The key assertion is that it doesn't crash and uses global defaults
        assert isinstance(result, DetectionResult)


# ---------------------------------------------------------------------------
# Feature-Only BCell Integration (Phase 37)
# ---------------------------------------------------------------------------


class TestFeatureOnlyBCellIntegration:
    """Phase 37: ImmuneOrchestrator wires feature-only BCell into pipeline."""

    @pytest.mark.asyncio
    async def test_process_with_bcell_influence(self):
        """Orchestrator with a BCell that has stored signatures shifts the score."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        # Create a feature-only BCell with stored signatures (high anomaly)
        bcell = BCell(bcell_weight=0.3, k=3)
        for _ in range(5):
            feats = np.zeros(10, dtype=np.float64)  # match query features
            bcell.store_signature(feats, 0.9, 1.0)  # high anomaly score

        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0, enabled=False)
        detector = _mock_detector("OCSVMDetector", score=0.5)

        # With BCell
        orchestrator_with = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
            b_cell=bcell,
        )

        result_with = await orchestrator_with.process(
            "prompt", "output", "hallucination",
        )

        # Without BCell (baseline)
        extractor2 = _mock_extractor(features)
        nk2 = _mock_nk_cell(None)
        dc2 = _mock_dendritic_cell()
        danger2 = DangerTheoryModulator(alpha=0.0, enabled=False)
        detector2 = _mock_detector("OCSVMDetector", score=0.5)

        orchestrator_without = ImmuneOrchestrator(
            feature_extractor=extractor2,
            nk_cell=nk2,
            dendritic_cell=dc2,
            danger_modulator=danger2,
            detectors={"hallucination": [detector2]},
        )

        result_without = await orchestrator_without.process(
            "prompt", "output", "hallucination",
        )

        # BCell with high-anomaly stored signatures should shift the score up
        assert result_with.score != result_without.score
        assert result_with.score > result_without.score  # prior=0.9, weight=0.3

    @pytest.mark.asyncio
    async def test_process_with_empty_bcell_no_influence(self):
        """Orchestrator with empty BCell (cold start) produces identical result to no-BCell case."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        bcell = BCell(bcell_weight=0.3, k=3)
        assert bcell.memory_size == 0

        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0, enabled=False)
        detector = _mock_detector("OCSVMDetector", score=0.5, confidence=0.8)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
            b_cell=bcell,
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        # Cold start: ensemble result unchanged
        assert abs(result.score - 0.5) < 1e-9

    @pytest.mark.asyncio
    async def test_from_config_creates_bcell(self):
        """from_config() with non-empty bcell_config creates orchestrator with BCell."""
        config = OrchestratorConfig(
            enabled=True,
            bcell_config={"bcell_weight": 0.2, "max_memory": 500},
        )
        detectors = {"hallucination": [_mock_detector("OCSVMDetector")]}

        orchestrator = ImmuneOrchestrator.from_config(config, detectors)
        assert orchestrator._b_cell is not None
        assert orchestrator._b_cell._bcell_weight == 0.2
        assert orchestrator._b_cell._max_memory == 500
        # Feature-only mode (no embedding_mode key)
        assert orchestrator._b_cell._embedding_mode is False


# ---------------------------------------------------------------------------
# Embedding BCell Integration (Phase 43)
# ---------------------------------------------------------------------------

class TestOrchestratorEmbeddingBCell:
    """Phase 43: ImmuneOrchestrator wires embedding-mode BCell via model_manager."""

    @pytest.mark.asyncio
    async def test_process_with_embedding_bcell_influence(self):
        """Orchestrator with embedding-mode BCell modifies score via prior."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        # Create a mock ModelManager that returns a fixed 384-dim embedding
        mock_mm = MagicMock()
        mock_mm.embed.return_value = np.ones(384, dtype=np.float32) * 0.5
        mock_mm.is_available.return_value = True

        # Create BCell in embedding mode with mock model_manager
        bcell = BCell(
            embedding_mode=True,
            model_manager=mock_mm,
            bcell_weight=0.3,
            k=3,
        )

        # Store some signatures with embeddings so prior has data
        for _ in range(5):
            feats = np.random.rand(10).astype(np.float64)
            emb = np.ones(384, dtype=np.float32) * 0.5  # similar to query
            bcell.store_signature(feats, 0.9, 1.0, embedding=emb)

        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0, enabled=False)
        detector = _mock_detector("OCSVMDetector", score=0.5)

        # Create orchestrator WITH BCell and model_manager
        orchestrator_with = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
            b_cell=bcell,
            model_manager=mock_mm,
        )

        result_with = await orchestrator_with.process(
            "prompt", "output", "hallucination",
        )

        # Create orchestrator WITHOUT BCell for baseline
        extractor2 = _mock_extractor(features)
        nk2 = _mock_nk_cell(None)
        dc2 = _mock_dendritic_cell()
        danger2 = DangerTheoryModulator(alpha=0.0, enabled=False)
        detector2 = _mock_detector("OCSVMDetector", score=0.5)

        orchestrator_without = ImmuneOrchestrator(
            feature_extractor=extractor2,
            nk_cell=nk2,
            dendritic_cell=dc2,
            danger_modulator=danger2,
            detectors={"hallucination": [detector2]},
        )

        result_without = await orchestrator_without.process(
            "prompt", "output", "hallucination",
        )

        # BCell with stored high-anomaly signatures should shift the score
        assert result_with.score != result_without.score
        # model_manager.embed() should have been called
        mock_mm.embed.assert_called_once_with("output")

    @pytest.mark.asyncio
    async def test_process_embedding_bcell_cold_start_no_change(self):
        """Orchestrator with embedding-mode BCell but empty memory = no influence."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        mock_mm = MagicMock()
        mock_mm.embed.return_value = np.ones(384, dtype=np.float32) * 0.5
        mock_mm.is_available.return_value = True

        # Empty BCell (cold start)
        bcell = BCell(
            embedding_mode=True,
            model_manager=mock_mm,
            bcell_weight=0.3,
        )
        assert bcell.memory_size == 0

        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0, enabled=False)
        detector = _mock_detector("OCSVMDetector", score=0.5, confidence=0.8)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
            b_cell=bcell,
            model_manager=mock_mm,
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        # Cold start: BCell returns ensemble result unchanged
        assert abs(result.score - 0.5) < 1e-9

    @pytest.mark.asyncio
    async def test_process_model_manager_embed_failure_fallback(self):
        """When model_manager.embed() raises, orchestrator falls back gracefully."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        mock_mm = MagicMock()
        mock_mm.embed.side_effect = RuntimeError("Model not loaded")
        mock_mm.is_available.return_value = True

        bcell = BCell(
            embedding_mode=True,
            model_manager=mock_mm,
            bcell_weight=0.3,
        )
        # Store a signature so BCell has memory
        feats = np.random.rand(10).astype(np.float64)
        bcell.store_signature(feats, 0.7, 1.0)

        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0, enabled=False)
        detector = _mock_detector("OCSVMDetector", score=0.5)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
            b_cell=bcell,
            model_manager=mock_mm,
        )

        # Should NOT crash -- falls back to feature-only BCell influence
        result = await orchestrator.process("prompt", "output", "hallucination")
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_from_config_with_embedding_bcell_config(self):
        """from_config with bcell_config.embedding_mode=True creates embedding BCell."""
        config = OrchestratorConfig(
            enabled=True,
            bcell_config={
                "embedding_mode": True,
                "bcell_weight": 0.3,
                "max_memory": 500,
            },
        )
        detectors = {"hallucination": [_mock_detector("OCSVMDetector")]}

        orchestrator = ImmuneOrchestrator.from_config(config, detectors)
        assert isinstance(orchestrator, ImmuneOrchestrator)
        # BCell should have been created with embedding_mode
        assert orchestrator._b_cell is not None
        assert orchestrator._model_manager is not None
        # BCell should have embedding_mode True (since model_manager is available)
        assert orchestrator._b_cell._embedding_mode is True

    @pytest.mark.asyncio
    async def test_from_config_without_bcell_config_no_bcell(self):
        """from_config with empty bcell_config creates no BCell."""
        config = OrchestratorConfig(
            enabled=True,
            bcell_config={},
        )
        detectors = {"hallucination": [_mock_detector("OCSVMDetector")]}

        orchestrator = ImmuneOrchestrator.from_config(config, detectors)
        assert orchestrator._b_cell is None
