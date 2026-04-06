"""End-to-end integration tests for the immune orchestrator pipeline.

Exercises all v9.0 components wired together through ImmuneOrchestrator.process():
- Pipeline path coverage: NK fast-path, SLM NK disconnect, DCA tier routing, normal input
- Component interaction: telemetry recording, B Cell score modification, adaptive DCA updates
- Real evaluation data: orchestrator processes hallucination samples with fitted detectors

Satisfies INTEG-01 (pipeline paths + component interactions) and
INTEG-02 (cold-start, backward compat, graceful degradation).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.orchestrator.adaptive_weights import AdaptiveWeightManager
from antigence_subnet.miner.orchestrator.b_cell import BCell
from antigence_subnet.miner.orchestrator.config import OrchestratorConfig, SLMNKConfig
from antigence_subnet.miner.orchestrator.danger import DangerTheoryModulator
from antigence_subnet.miner.orchestrator.dendritic_cell import DCAResult, DendriticCell
from antigence_subnet.miner.orchestrator.nk_cell import FeatureStatistics, NKCell
from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator
from antigence_subnet.miner.orchestrator.slm_nk_cell import SLMNKCell
from antigence_subnet.miner.orchestrator.telemetry import MinerTelemetry

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"


# ---------------------------------------------------------------------------
# Shared Helpers
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


def _mock_model_manager(score_value: float = 0.8, available: bool = True):
    """Create a mock ModelManager for SLM NK and BCell embedding tests."""
    mm = MagicMock()
    mm.embed.return_value = np.ones(384, dtype=np.float32) * 0.5
    mm.score.return_value = score_value
    mm.is_available.return_value = available
    return mm


def _build_full_orchestrator(
    features: np.ndarray | None = None,
    nk_result: DetectionResult | None = None,
    dca_result: DCAResult | None = None,
    detector_names_scores: list[tuple[str, float]] | None = None,
    danger_alpha: float = 0.0,
    danger_enabled: bool = False,
    slm_nk_enabled: bool = True,
    slm_nk_score: float = 0.8,
    slm_nk_threshold: float = 0.3,
    bcell: BCell | None = "default",
    bcell_stored_sigs: int = 0,
    adaptive_weights: AdaptiveWeightManager | None = "default",
    model_available: bool = True,
    config: OrchestratorConfig | None = None,
):
    """Factory that constructs a full ImmuneOrchestrator with all v9.0 components.

    Returns (orchestrator, components_dict) where components_dict has references
    to all components for assertion in tests.
    """
    if features is None:
        features = np.zeros(10, dtype=np.float64)

    extractor = _mock_extractor(features)

    # NK Cell: use mock unless caller supplies real one via nk_result sentinel
    nk = _mock_nk_cell(nk_result)

    # DCA: use mock
    dc = _mock_dendritic_cell(dca_result)

    # Danger modulator
    danger = DangerTheoryModulator(alpha=danger_alpha, enabled=danger_enabled)

    # Detectors
    if detector_names_scores is None:
        detector_names_scores = [("OCSVMDetector", 0.4), ("NegSelAISDetector", 0.3)]
    detectors = [_mock_detector(name, score) for name, score in detector_names_scores]

    # Model manager
    mock_mm = _mock_model_manager(score_value=slm_nk_score, available=model_available)

    # SLM NK Cell
    slm_nk = SLMNKCell(
        model_manager=mock_mm,
        similarity_threshold=slm_nk_threshold,
        enabled=slm_nk_enabled,
    )

    # BCell
    if bcell == "default":
        bcell = BCell(
            embedding_mode=True,
            model_manager=mock_mm,
            bcell_weight=0.2,
            k=3,
            max_memory=100,
        )
        for i in range(bcell_stored_sigs):
            feats = np.random.rand(10).astype(np.float64)
            emb = np.ones(384, dtype=np.float32) * 0.5
            bcell.store_signature(feats, 0.9, 1.0, embedding=emb)
    elif bcell is None:
        pass  # no bcell

    # Adaptive weights
    if adaptive_weights == "default":
        adaptive_weights = AdaptiveWeightManager(alpha=0.1)

    orchestrator = ImmuneOrchestrator(
        feature_extractor=extractor,
        nk_cell=nk,
        dendritic_cell=dc,
        danger_modulator=danger,
        detectors={"hallucination": detectors},
        config=config,
        slm_nk_cell=slm_nk,
        b_cell=bcell,
        adaptive_weights=adaptive_weights,
        model_manager=mock_mm,
    )

    components = {
        "extractor": extractor,
        "nk": nk,
        "dc": dc,
        "danger": danger,
        "detectors": detectors,
        "slm_nk": slm_nk,
        "bcell": bcell,
        "adaptive_weights": adaptive_weights,
        "model_manager": mock_mm,
    }

    return orchestrator, components


# ---------------------------------------------------------------------------
# INTEG-01: Pipeline Path Tests
# ---------------------------------------------------------------------------

class TestPipelinePathNKFastPath:
    """NK fast-path catches obvious anomaly (z-score > threshold)."""

    @pytest.mark.asyncio
    async def test_nk_fast_path_short_circuits(self):
        """NK Cell returns nk_fast_path -> DCA/detectors NOT called."""
        # Build NK Cell with real stats: feature at index 0, mean=0, std=1
        stats = [
            FeatureStatistics(
                name="claim_density", index=0, mean=0.0, std=1.0,
                is_binary=False, is_constant=False,
            ),
        ]
        nk = NKCell(feature_stats=stats, z_threshold=2.0)

        # Feature with z=3.0 > 2.0 threshold
        features = _make_features(claim_density=3.0)
        extractor = _mock_extractor(features)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0)
        detector = _mock_detector("OCSVMDetector")

        mock_mm = _mock_model_manager()
        slm_nk = SLMNKCell(model_manager=mock_mm, enabled=True)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
            slm_nk_cell=slm_nk,
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        assert result.anomaly_type == "nk_fast_path"
        assert result.score == 1.0
        # DCA and detectors should NOT have been called
        dc.classify.assert_not_called()
        detector.detect.assert_not_called()
        # SLM NK should NOT have been called
        mock_mm.score.assert_not_called()


class TestPipelinePathSLMNKDisconnect:
    """SLM NK catches semantically disconnected output."""

    @pytest.mark.asyncio
    async def test_slm_nk_semantic_disconnect(self):
        """SLM NK returns slm_nk_semantic_disconnect when similarity < threshold.
        DCA/detectors NOT called."""
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        # NK returns None (no rule-based trigger)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0)
        detector = _mock_detector("OCSVMDetector")

        # Mock ModelManager with low similarity score (0.1 < 0.3 threshold)
        mock_mm = _mock_model_manager(score_value=0.1, available=True)
        slm_nk = SLMNKCell(
            model_manager=mock_mm,
            similarity_threshold=0.3,
            enabled=True,
        )

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [detector]},
            slm_nk_cell=slm_nk,
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        assert result.anomaly_type == "slm_nk_semantic_disconnect"
        assert result.score == 0.9  # 1.0 - 0.1
        # DCA and detectors should NOT have been called
        dc.classify.assert_not_called()
        detector.detect.assert_not_called()


class TestPipelinePathDCATierRouting:
    """DCA routes ambiguous input through correct detector tier."""

    @pytest.mark.asyncio
    async def test_dca_routes_to_ocsvm_tier(self):
        """DCA returns tier=["ocsvm"] -> only OCSVMDetector called, not NegSel."""
        dca_result = DCAResult(
            maturation_state="semi-mature",
            signal_scores={"pamp": 0.1, "danger": 0.3, "safe": 0.2},
            recommended_tier=["ocsvm"],
        )
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell(dca_result)
        danger = DangerTheoryModulator(alpha=0.0, enabled=False)

        # SLM NK passes (high similarity)
        mock_mm = _mock_model_manager(score_value=0.9, available=True)
        slm_nk = SLMNKCell(model_manager=mock_mm, similarity_threshold=0.3, enabled=True)

        ocsvm_det = _mock_detector("OCSVMDetector", score=0.6)
        negsel_det = _mock_detector("NegSelAISDetector", score=0.4)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [ocsvm_det, negsel_det]},
            slm_nk_cell=slm_nk,
        )

        result = await orchestrator.process("prompt", "output", "hallucination")
        # Only OCSVM should have been called
        ocsvm_det.detect.assert_called_once()
        negsel_det.detect.assert_not_called()


class TestPipelinePathNormalInput:
    """Normal input passes full pipeline with low anomaly score."""

    @pytest.mark.asyncio
    async def test_normal_input_full_pipeline(self):
        """Normal input: NK=None, SLM NK=None (high similarity), DCA empty tier -> all detectors run."""
        dca_result = DCAResult(
            maturation_state="mature",
            signal_scores={"pamp": 0.5, "danger": 0.1, "safe": 0.1},
            recommended_tier=[],  # empty = full ensemble
        )
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell(dca_result)
        danger = DangerTheoryModulator(alpha=0.0, enabled=False)

        # SLM NK passes (similarity 0.8 > 0.3)
        mock_mm = _mock_model_manager(score_value=0.8, available=True)
        slm_nk = SLMNKCell(model_manager=mock_mm, similarity_threshold=0.3, enabled=True)

        det1 = _mock_detector("OCSVMDetector", score=0.2)
        det2 = _mock_detector("NegSelAISDetector", score=0.3)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={"hallucination": [det1, det2]},
            slm_nk_cell=slm_nk,
        )

        result = await orchestrator.process("prompt", "benign output", "hallucination")
        assert result.score < 0.5
        # Both detectors should have been called
        det1.detect.assert_called_once()
        det2.detect.assert_called_once()


# ---------------------------------------------------------------------------
# INTEG-01: Component Interaction Tests
# ---------------------------------------------------------------------------

class TestComponentInteraction:
    """Telemetry, B Cell influence, and adaptive DCA weight updates."""

    @pytest.mark.asyncio
    async def test_telemetry_recorded(self):
        """Telemetry.record() after process() makes get_stats() return data."""
        orchestrator, components = _build_full_orchestrator(
            slm_nk_score=0.9,  # SLM NK passes
        )
        telemetry = MinerTelemetry(window_size=100)

        result = await orchestrator.process("prompt", "output", "hallucination")
        # Mirror forward.py pattern: record telemetry after detection
        telemetry.record("hallucination", result.score, result.confidence)

        stats = telemetry.get_stats("hallucination")
        assert stats is not None
        assert stats["count"] == 1
        assert "mean" in stats

    @pytest.mark.asyncio
    async def test_bcell_modifies_score(self):
        """BCell with stored high-anomaly signatures modifies score vs no-BCell baseline."""
        # Orchestrator WITH BCell having 5 stored high-anomaly signatures
        orch_with, comp_with = _build_full_orchestrator(
            slm_nk_score=0.9,  # SLM NK passes
            bcell_stored_sigs=5,
        )
        result_with = await orch_with.process("prompt", "output", "hallucination")

        # Orchestrator WITHOUT BCell
        orch_without, comp_without = _build_full_orchestrator(
            slm_nk_score=0.9,
            bcell=None,
        )
        result_without = await orch_without.process("prompt", "output", "hallucination")

        # BCell with high-anomaly signatures should shift the score
        assert result_with.score != result_without.score

    @pytest.mark.asyncio
    async def test_adaptive_weights_update(self):
        """AdaptiveWeightManager.adapt() updates round_count and modifies weights."""
        awm = AdaptiveWeightManager(alpha=0.3)
        initial_weights = awm.get_weights()
        initial_round = awm.get_round_count()
        assert initial_round == 0

        features = _make_features(pamp_score=0.8, claim_density=0.5, exaggeration=0.6)
        awm.adapt(features, outcome=1.0)

        assert awm.get_round_count() == 1
        updated_weights = awm.get_weights()
        # Danger has 3 features (exaggeration, controversy, claim_density) with
        # different feature values, so EMA + renormalization should redistribute
        # weights within the category (unlike single-feature PAMP which stays fixed).
        initial_danger = initial_weights["danger"]
        updated_danger = updated_weights["danger"]
        # At least one weight should differ within the danger category
        any_changed = False
        for name in initial_danger:
            if abs(initial_danger[name][1] - updated_danger[name][1]) > 1e-12:
                any_changed = True
                break
        assert any_changed, "Expected at least one danger weight to change after adapt()"


# ---------------------------------------------------------------------------
# INTEG-01: Real Evaluation Data Tests
# ---------------------------------------------------------------------------

class TestRealEvalData:
    """Full pipeline with real evaluation data produces valid DetectionResults."""

    @pytest.fixture(scope="class")
    def fitted_detectors(self):
        """Fit OCSVM + NegSel on hallucination training data."""
        from antigence_subnet.miner.data import load_training_samples
        from antigence_subnet.miner.detectors.negsel import NegSelAISDetector
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        training = load_training_samples(str(DATA_DIR), "hallucination")
        ocsvm = OCSVMDetector()
        negsel = NegSelAISDetector()
        ocsvm.fit(training)
        negsel.fit(training)
        return [ocsvm, negsel]

    @pytest.fixture(scope="class")
    def eval_samples(self):
        """Load hallucination evaluation samples."""
        with open(DATA_DIR / "hallucination" / "samples.json") as f:
            return json.load(f)["samples"]

    @pytest.mark.asyncio
    async def test_real_data_produces_valid_results(self, fitted_detectors, eval_samples):
        """Orchestrator processes real samples with score in [0.0, 1.0]."""
        mock_mm = _mock_model_manager(score_value=0.8, available=True)
        config = OrchestratorConfig(
            enabled=True,
            nk_config={"z_threshold": 100.0},  # high threshold to avoid NK triggers
            danger_config={"alpha": 0.0, "enabled": False},
            slm_nk_config=SLMNKConfig(enabled=False),  # disable SLM NK for real data test
        )
        orchestrator = ImmuneOrchestrator.from_config(
            config, {"hallucination": fitted_detectors}
        )

        for sample in eval_samples[:5]:
            result = await orchestrator.process(
                sample.get("prompt", ""),
                sample.get("output", ""),
                "hallucination",
            )
            assert isinstance(result, DetectionResult)
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0


# ===========================================================================
# INTEG-02: Cold-Start, Backward Compat, Graceful Degradation
# ===========================================================================

class TestColdStart:
    """Fresh orchestrator (empty BCell, no telemetry) produces no regression."""

    @pytest.fixture(scope="class")
    def fitted_detectors(self):
        """Fit OCSVM + NegSel on hallucination training data."""
        from antigence_subnet.miner.data import load_training_samples
        from antigence_subnet.miner.detectors.negsel import NegSelAISDetector
        from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector

        training = load_training_samples(str(DATA_DIR), "hallucination")
        ocsvm = OCSVMDetector()
        negsel = NegSelAISDetector()
        ocsvm.fit(training)
        negsel.fit(training)
        return [ocsvm, negsel]

    @pytest.fixture(scope="class")
    def samples_and_manifest(self):
        """Load hallucination evaluation data with ground truth."""
        with open(DATA_DIR / "hallucination" / "samples.json") as f:
            samples = json.load(f)["samples"]
        with open(DATA_DIR / "hallucination" / "manifest.json") as f:
            manifest = json.load(f)
        return samples, manifest

    @pytest.mark.asyncio
    async def test_cold_start_no_regression(self, fitted_detectors, samples_and_manifest):
        """Cold-start orchestrator F1 >= flat ensemble F1 (zero regression).

        Per D-04: empty BCell has bcell_weight forced to 0.0, so orchestrator
        scores should be very close to flat ensemble scores.
        """
        from antigence_subnet.miner.ensemble import ensemble_detect

        samples, manifest = samples_and_manifest

        mock_mm = _mock_model_manager(score_value=0.8, available=True)

        config = OrchestratorConfig(
            enabled=True,
            nk_config={"z_threshold": 100.0},  # high threshold = no NK triggers
            danger_config={"alpha": 0.3, "enabled": True},
            bcell_config={
                "embedding_mode": True,
                "max_memory": 100,
                "bcell_weight": 0.2,
            },
            slm_nk_config=SLMNKConfig(enabled=False),  # disable SLM NK for fair comparison
        )
        orchestrator = ImmuneOrchestrator.from_config(
            config, {"hallucination": fitted_detectors}
        )
        # Verify BCell is cold start (empty memory)
        assert orchestrator._b_cell is not None
        assert orchestrator._b_cell.memory_size == 0

        # Select 10 evaluation samples (mix of normal and anomalous)
        test_samples = samples[:10]

        # Run orchestrator on samples
        orch_predictions = []
        ensemble_predictions = []
        ground_truths = []
        for sample in test_samples:
            prompt = sample.get("prompt", "")
            output = sample.get("output", "")
            sid = sample["id"]
            gt = manifest[sid]["ground_truth_label"]
            ground_truths.append(1 if gt == "anomalous" else 0)

            orch_r = await orchestrator.process(prompt, output, "hallucination")
            orch_predictions.append(1 if orch_r.score > 0.5 else 0)

            flat_r = await ensemble_detect(fitted_detectors, prompt, output)
            ensemble_predictions.append(1 if flat_r.score > 0.5 else 0)

        # Compute F1 for both
        def _f1(preds, truth):
            tp = sum(1 for p, t in zip(preds, truth) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(preds, truth) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(preds, truth) if p == 0 and t == 1)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        orch_f1 = _f1(orch_predictions, ground_truths)
        ensemble_f1 = _f1(ensemble_predictions, ground_truths)

        # Cold start: orchestrator F1 >= ensemble F1 - 0.02 tolerance
        assert orch_f1 >= ensemble_f1 - 0.02, (
            f"Cold-start regression: orchestrator F1={orch_f1:.3f} < "
            f"ensemble F1={ensemble_f1:.3f} - 0.02"
        )


class TestBackwardCompatDisabled:
    """Disabled orchestrator config produces correct state."""

    def test_disabled_config_default(self):
        """OrchestratorConfig with enabled=False (default) builds without error."""
        config = OrchestratorConfig()
        assert config.enabled is False

    def test_from_config_disabled_constructs(self):
        """from_config with enabled=False constructs orchestrator without error."""
        config = OrchestratorConfig(enabled=False)
        detectors = {"hallucination": [_mock_detector("OCSVMDetector")]}
        orchestrator = ImmuneOrchestrator.from_config(config, detectors)
        assert isinstance(orchestrator, ImmuneOrchestrator)
        assert config.enabled is False

    def test_forward_pattern_disabled_falls_through(self):
        """Simulate forward.py pattern: disabled orchestrator -> flat ensemble path."""
        config = OrchestratorConfig(enabled=False)
        orchestrator = ImmuneOrchestrator.from_config(
            config, {"hallucination": [_mock_detector("OCSVMDetector")]}
        )
        # forward.py checks: if orchestrator is not None and config.enabled
        # When enabled=False, this evaluates to False -> flat ensemble path
        takes_orchestrator_path = orchestrator is not None and config.enabled
        assert takes_orchestrator_path is False


class TestGracefulDegradation:
    """Disabling individual components does not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_slm_nk_disabled_no_crash(self):
        """SLM NK with enabled=False: rule-based NK as sole gate, no crash."""
        orchestrator, components = _build_full_orchestrator(
            slm_nk_enabled=False,
            slm_nk_score=0.1,  # would trigger if enabled
        )
        result = await orchestrator.process("prompt", "output", "hallucination")
        assert isinstance(result, DetectionResult)
        assert result.anomaly_type != "slm_nk_semantic_disconnect"

    @pytest.mark.asyncio
    async def test_bcell_none_no_crash(self):
        """Orchestrator without b_cell param: pipeline works without memory influence."""
        orchestrator, components = _build_full_orchestrator(
            bcell=None,
            slm_nk_score=0.9,  # SLM NK passes
        )
        result = await orchestrator.process("prompt", "output", "hallucination")
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_adaptive_weights_none_no_crash(self):
        """Orchestrator without adaptive_weights: pipeline works with static weights."""
        orchestrator, components = _build_full_orchestrator(
            adaptive_weights=None,
            slm_nk_score=0.9,  # SLM NK passes
        )
        result = await orchestrator.process("prompt", "output", "hallucination")
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_model_unavailable_graceful_fallback(self):
        """ModelManager unavailable: SLM NK returns None, BCell uses feature-only mode."""
        orchestrator, components = _build_full_orchestrator(
            model_available=False,
            slm_nk_score=0.1,  # would trigger if model were available
        )
        result = await orchestrator.process("prompt", "output", "hallucination")
        assert isinstance(result, DetectionResult)
        # SLM NK should have returned None (model unavailable -> graceful degradation)
        assert result.anomaly_type != "slm_nk_semantic_disconnect"
        assert 0.0 <= result.score <= 1.0


class TestMultiDomainCoverage:
    """All 4 domains produce valid results through the full pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain", ["hallucination", "code_security", "reasoning", "bio"])
    async def test_domain_produces_valid_result(self, domain):
        """Each domain produces a valid DetectionResult with score in [0.0, 1.0]."""
        features = np.zeros(10, dtype=np.float64)
        extractor = _mock_extractor(features)
        nk = _mock_nk_cell(None)
        dc = _mock_dendritic_cell()
        danger = DangerTheoryModulator(alpha=0.0, enabled=False)

        mock_mm = _mock_model_manager(score_value=0.8, available=True)
        slm_nk = SLMNKCell(model_manager=mock_mm, similarity_threshold=0.3, enabled=True)

        bcell = BCell(
            embedding_mode=True,
            model_manager=mock_mm,
            bcell_weight=0.2,
        )

        det = _mock_detector("OCSVMDetector", score=0.4)

        orchestrator = ImmuneOrchestrator(
            feature_extractor=extractor,
            nk_cell=nk,
            dendritic_cell=dc,
            danger_modulator=danger,
            detectors={domain: [det]},
            slm_nk_cell=slm_nk,
            b_cell=bcell,
            model_manager=mock_mm,
        )

        result = await orchestrator.process("prompt", "output", domain)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
