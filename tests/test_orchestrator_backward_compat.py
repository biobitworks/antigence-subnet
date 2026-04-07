"""Backward compatibility: orchestrator disabled produces identical scores to flat ensemble."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigence_subnet.miner.data import load_training_samples
from antigence_subnet.miner.detectors.negsel import NegSelAISDetector
from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector
from antigence_subnet.miner.ensemble import ensemble_detect
from antigence_subnet.miner.orchestrator.config import OrchestratorConfig
from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"


@pytest.fixture(scope="module")
def fitted_detectors():
    """Create and fit OCSVM+NegSel for hallucination domain."""
    training = load_training_samples(str(DATA_DIR), "hallucination")
    ocsvm = OCSVMDetector()
    negsel = NegSelAISDetector()
    ocsvm.fit(training)
    negsel.fit(training)
    return [ocsvm, negsel]


@pytest.fixture(scope="module")
def samples_and_manifest():
    """Load hallucination evaluation data."""
    with open(DATA_DIR / "hallucination" / "samples.json") as f:
        samples = json.load(f)["samples"]
    with open(DATA_DIR / "hallucination" / "manifest.json") as f:
        manifest = json.load(f)
    return samples, manifest


class TestOrchestratorDisabledIdentical:
    """When orchestrator.enabled=False, forward.py falls through to ensemble."""

    def test_disabled_config_default(self):
        config = OrchestratorConfig()
        assert config.enabled is False

    def test_from_empty_toml(self):
        config = OrchestratorConfig.from_toml_raw({})
        assert config.enabled is False

    def test_from_toml_explicit_false(self):
        config = OrchestratorConfig.from_toml_raw({"miner": {"orchestrator": {"enabled": False}}})
        assert config.enabled is False


class TestOrchestratorEnabledNoRegression:
    """Orchestrator enabled with no NK/danger produces same scores as ensemble."""

    def test_orchestrator_no_nk_no_danger_matches_ensemble(
        self, fitted_detectors, samples_and_manifest
    ):
        """Orchestrator with empty NK stats and disabled danger should match flat ensemble."""
        samples, manifest = samples_and_manifest
        config = OrchestratorConfig(
            enabled=True,
            nk_config={"z_threshold": 100.0},  # Very high threshold = no triggers
            danger_config={"alpha": 0.0, "enabled": False},
        )
        orchestrator = ImmuneOrchestrator.from_config(config, {"hallucination": fitted_detectors})

        async def compare():
            mismatches = 0
            for sample in samples[:10]:
                flat_r = await ensemble_detect(
                    fitted_detectors,
                    sample.get("prompt", ""),
                    sample.get("output", ""),
                )
                orch_r = await orchestrator.process(
                    sample.get("prompt", ""),
                    sample.get("output", ""),
                    "hallucination",
                )
                if abs(flat_r.score - orch_r.score) > 0.01:
                    mismatches += 1
            return mismatches

        mismatches = asyncio.run(compare())
        assert mismatches == 0, (
            f"{mismatches} score mismatches between flat ensemble and orchestrator"
        )


# ---------------------------------------------------------------------------
# Phase 37 BCell Backward Compatibility
# ---------------------------------------------------------------------------


class TestBCellBackwardCompat:
    """Phase 37: BCellStub still works, v7.0 config produces identical behavior."""

    def test_bcell_stub_still_works(self):
        """BCellStub() creates valid instance, isinstance(BCellStub(), ImmuneCellType)."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell
        from antigence_subnet.miner.orchestrator.cells import BCellStub, ImmuneCellType

        stub = BCellStub()
        # BCellStub is a subclass of BCell
        assert isinstance(stub, BCell)
        # BCellStub satisfies ImmuneCellType Protocol
        assert isinstance(stub, ImmuneCellType)
        # process() returns None (B Cell never gates)
        import numpy as np

        assert stub.process(np.zeros(10), "prompt", "output") is None
        # memory_size is 0 (default init = cold start)
        assert stub.memory_size == 0

    def test_no_bcell_config_identical_to_v7(self, fitted_detectors, samples_and_manifest):
        """OrchestratorConfig with no bcell section produces orchestrator that
        behaves identically to v7.0 (empty bcell_config -> BCell with empty
        memory -> no influence)."""
        samples, manifest = samples_and_manifest

        # v7.0-style config: no bcell_config at all
        config_v7 = OrchestratorConfig(
            enabled=True,
            nk_config={"z_threshold": 100.0},  # Very high = no triggers
            danger_config={"alpha": 0.0, "enabled": False},
            # bcell_config defaults to {} -> no BCell created
        )

        # v8.0-style config: explicit bcell_config but empty memory
        config_v8 = OrchestratorConfig(
            enabled=True,
            nk_config={"z_threshold": 100.0},
            danger_config={"alpha": 0.0, "enabled": False},
            bcell_config={"bcell_weight": 0.2, "max_memory": 500},
        )

        orch_v7 = ImmuneOrchestrator.from_config(
            config_v7,
            {"hallucination": fitted_detectors},
        )
        orch_v8 = ImmuneOrchestrator.from_config(
            config_v8,
            {"hallucination": fitted_detectors},
        )

        # v7 has no BCell, v8 has BCell with empty memory
        assert orch_v7._b_cell is None
        assert orch_v8._b_cell is not None
        assert orch_v8._b_cell.memory_size == 0

        async def compare():
            mismatches = 0
            for sample in samples[:10]:
                v7_r = await orch_v7.process(
                    sample.get("prompt", ""),
                    sample.get("output", ""),
                    "hallucination",
                )
                v8_r = await orch_v8.process(
                    sample.get("prompt", ""),
                    sample.get("output", ""),
                    "hallucination",
                )
                if abs(v7_r.score - v8_r.score) > 0.01:
                    mismatches += 1
            return mismatches

        mismatches = asyncio.run(compare())
        assert mismatches == 0, (
            f"{mismatches} score mismatches between v7 (no BCell) and "
            f"v8 (empty BCell) -- cold start should produce zero influence"
        )


# ---------------------------------------------------------------------------
# Phase 43 Embedding Mode Backward Compatibility
# ---------------------------------------------------------------------------


class TestEmbeddingModeBackwardCompat:
    """Phase 43: v8.0 configs (no embedding_mode) produce identical behavior."""

    def test_v8_config_no_embedding_identical_behavior(self):
        """OrchestratorConfig with bcell_config={} (v8.0 style) produces
        orchestrator with no BCell and no model loading."""
        config = OrchestratorConfig(
            enabled=True,
            bcell_config={},  # v8.0 style: no embedding_mode key
        )
        detectors = {"hallucination": []}

        orchestrator = ImmuneOrchestrator.from_config(config, detectors)
        # Empty bcell_config = no BCell created
        assert orchestrator._b_cell is None

    def test_v8_config_feature_only_bcell_no_embedding(self):
        """OrchestratorConfig with bcell_config having feature-only keys
        (no embedding_mode) creates feature-only BCell."""
        config = OrchestratorConfig(
            enabled=True,
            bcell_config={
                "max_memory": 500,
                "bcell_weight": 0.2,
                # No embedding_mode key = defaults to False
            },
        )
        detectors = {"hallucination": []}

        orchestrator = ImmuneOrchestrator.from_config(config, detectors)
        assert orchestrator._b_cell is not None
        assert orchestrator._b_cell._embedding_mode is False

    def test_no_model_config_no_crash(self):
        """OrchestratorConfig without model_config attribute (simulating
        pre-Phase-41 config) does not crash from_config()."""
        # OrchestratorConfig always has model_config (default factory),
        # but verify from_config still works with default model_config
        config = OrchestratorConfig(
            enabled=True,
            bcell_config={"bcell_weight": 0.1},
        )
        # model_config is defaulted via field default_factory
        assert hasattr(config, "model_config")

        detectors = {"hallucination": []}
        orchestrator = ImmuneOrchestrator.from_config(config, detectors)
        assert isinstance(orchestrator, ImmuneOrchestrator)
        # model_manager should still be created (ModelManager with defaults)
        assert orchestrator._model_manager is not None
