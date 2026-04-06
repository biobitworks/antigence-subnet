"""Integration tests for dual NK Cell gate in ImmuneOrchestrator.

Tests SLMNKCell wiring into the orchestrator pipeline, DomainConfig extension
with slm_nk_similarity_threshold, from_config() SLM NK creation, package
exports, and per-domain SLM threshold overrides.

Phase 42, Plan 02.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.orchestrator.config import (
    DomainConfig,
    OrchestratorConfig,
    SLMNKConfig,
)
from antigence_subnet.miner.orchestrator.nk_cell import NKCell
from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator
from antigence_subnet.miner.orchestrator.slm_nk_cell import SLMNKCell


# -- Fixtures ----------------------------------------------------------------

def _make_mock_extractor():
    """Create mock DendriticFeatureExtractor returning 10-dim zeros."""
    mock = MagicMock()
    mock.extract.return_value = np.zeros(10)
    return mock


def _make_mock_dc():
    """Create mock DendriticCell that passes through (safe tier)."""
    mock = MagicMock()
    result = MagicMock()
    result.recommended_tier = []  # empty = full ensemble
    mock.classify.return_value = result
    return mock


def _make_mock_danger():
    """Create mock DangerTheoryModulator that returns input unchanged."""
    mock = MagicMock()
    mock.modulate_result.side_effect = lambda result, features, **kw: result
    return mock


def _make_mock_detector(score=0.5):
    """Create mock detector returning fixed score."""
    mock = MagicMock()
    mock.detect = AsyncMock(return_value=DetectionResult(
        score=score, confidence=0.8, anomaly_type="test",
    ))
    return mock


def _make_mock_slm_nk_cell(trigger_result=None):
    """Create mock SLMNKCell that returns a fixed result or None."""
    mock = MagicMock(spec=SLMNKCell)
    mock.process.return_value = trigger_result
    return mock


def _make_nk_cell_stub():
    """Create NKCell with empty stats (never triggers)."""
    return NKCell(feature_stats=[])


# -- TestDomainConfigSLMNK --------------------------------------------------

class TestDomainConfigSLMNK:
    """DomainConfig extended with slm_nk_similarity_threshold field."""

    def test_domain_config_has_slm_nk_threshold(self):
        """DomainConfig accepts slm_nk_similarity_threshold keyword."""
        dc = DomainConfig(slm_nk_similarity_threshold=0.5)
        assert dc.slm_nk_similarity_threshold == 0.5

    def test_domain_config_slm_nk_threshold_defaults_none(self):
        """DomainConfig() default slm_nk_similarity_threshold is None."""
        dc = DomainConfig()
        assert dc.slm_nk_similarity_threshold is None

    def test_domain_config_toml_parsing(self):
        """from_toml_raw reads slm_nk_similarity_threshold from domain dict."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {
                            "nk_z_threshold": 2.0,
                            "slm_nk_similarity_threshold": 0.4,
                        },
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        dc = config.get_domain_config("hallucination")
        assert dc is not None
        assert dc.slm_nk_similarity_threshold == 0.4

    def test_domain_config_validation_above_1(self):
        """slm_nk_similarity_threshold > 1.0 raises ValueError."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"slm_nk_similarity_threshold": 1.5},
                    },
                }
            }
        }
        with pytest.raises(ValueError, match="slm_nk_similarity_threshold"):
            OrchestratorConfig.from_toml_raw(raw)

    def test_domain_config_validation_below_0(self):
        """slm_nk_similarity_threshold < 0.0 raises ValueError."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"slm_nk_similarity_threshold": -0.1},
                    },
                }
            }
        }
        with pytest.raises(ValueError, match="slm_nk_similarity_threshold"):
            OrchestratorConfig.from_toml_raw(raw)

    def test_domain_config_valid_boundary_0(self):
        """slm_nk_similarity_threshold=0.0 is valid."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"slm_nk_similarity_threshold": 0.0},
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        dc = config.get_domain_config("hallucination")
        assert dc.slm_nk_similarity_threshold == 0.0

    def test_domain_config_valid_boundary_1(self):
        """slm_nk_similarity_threshold=1.0 is valid."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"slm_nk_similarity_threshold": 1.0},
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        dc = config.get_domain_config("hallucination")
        assert dc.slm_nk_similarity_threshold == 1.0


# -- TestOrchestratorDualNKGate -----------------------------------------------

class TestOrchestratorDualNKGate:
    """Orchestrator with both rule-based NKCell and SLMNKCell."""

    def test_rule_based_triggers_slm_not_called(self):
        """When rule-based NK triggers, SLM NK is NOT called."""
        nk_result = DetectionResult(score=0.9, confidence=0.95, anomaly_type="nk_stat_outlier")
        nk_cell = MagicMock()
        nk_cell.process.return_value = nk_result

        slm_nk = _make_mock_slm_nk_cell()

        orch = ImmuneOrchestrator(
            feature_extractor=_make_mock_extractor(),
            nk_cell=nk_cell,
            dendritic_cell=_make_mock_dc(),
            danger_modulator=_make_mock_danger(),
            detectors={"hallucination": [_make_mock_detector()]},
            slm_nk_cell=slm_nk,
        )

        result = asyncio.run(orch.process("prompt", "output", "hallucination"))
        assert result.anomaly_type == "nk_stat_outlier"
        slm_nk.process.assert_not_called()

    def test_rule_based_passes_slm_triggers(self):
        """When rule-based NK passes, SLM NK triggers -> SLM result returned."""
        slm_result = DetectionResult(
            score=0.8, confidence=0.8, anomaly_type="slm_nk_semantic_disconnect",
        )
        slm_nk = _make_mock_slm_nk_cell(trigger_result=slm_result)

        orch = ImmuneOrchestrator(
            feature_extractor=_make_mock_extractor(),
            nk_cell=_make_nk_cell_stub(),
            dendritic_cell=_make_mock_dc(),
            danger_modulator=_make_mock_danger(),
            detectors={"hallucination": [_make_mock_detector()]},
            slm_nk_cell=slm_nk,
        )

        result = asyncio.run(orch.process("prompt", "output", "hallucination"))
        assert result.anomaly_type == "slm_nk_semantic_disconnect"
        assert result.score == 0.8

    def test_both_pass_continues_to_ensemble(self):
        """When both NK cells pass, detection continues to DCA/ensemble pipeline."""
        slm_nk = _make_mock_slm_nk_cell(trigger_result=None)

        orch = ImmuneOrchestrator(
            feature_extractor=_make_mock_extractor(),
            nk_cell=_make_nk_cell_stub(),
            dendritic_cell=_make_mock_dc(),
            danger_modulator=_make_mock_danger(),
            detectors={"hallucination": [_make_mock_detector(score=0.3)]},
            slm_nk_cell=slm_nk,
        )

        result = asyncio.run(orch.process("prompt", "output", "hallucination"))
        # Should get result from ensemble (mock detector returns 0.3)
        assert result is not None
        assert result.anomaly_type != "slm_nk_semantic_disconnect"
        assert result.anomaly_type != "nk_stat_outlier"

    def test_no_slm_nk_backward_compat(self):
        """slm_nk_cell=None behaves identically to pre-Phase-42."""
        orch = ImmuneOrchestrator(
            feature_extractor=_make_mock_extractor(),
            nk_cell=_make_nk_cell_stub(),
            dendritic_cell=_make_mock_dc(),
            danger_modulator=_make_mock_danger(),
            detectors={"hallucination": [_make_mock_detector(score=0.5)]},
            slm_nk_cell=None,
        )

        result = asyncio.run(orch.process("prompt", "output", "hallucination"))
        assert result is not None
        # No crash, no SLM NK involvement

    def test_default_slm_nk_is_none(self):
        """ImmuneOrchestrator() without slm_nk_cell arg defaults to None."""
        orch = ImmuneOrchestrator(
            feature_extractor=_make_mock_extractor(),
            nk_cell=_make_nk_cell_stub(),
            dendritic_cell=_make_mock_dc(),
            danger_modulator=_make_mock_danger(),
            detectors={},
        )

        # Should not crash when processing
        assert orch._slm_nk is None


# -- TestOrchestratorFromConfig -----------------------------------------------

class TestOrchestratorFromConfig:
    """from_config() creates SLMNKCell from slm_nk_config + model_config."""

    def test_enabled_creates_slm_nk_cell(self):
        """from_config with slm_nk_config.enabled=True creates SLMNKCell."""
        config = OrchestratorConfig(
            enabled=True,
            slm_nk_config=SLMNKConfig(enabled=True, similarity_threshold=0.4),
        )
        orch = ImmuneOrchestrator.from_config(config, {"hallucination": []})
        assert orch._slm_nk is not None
        assert isinstance(orch._slm_nk, SLMNKCell)

    def test_disabled_results_in_none(self):
        """from_config with slm_nk_config.enabled=False sets slm_nk_cell=None."""
        config = OrchestratorConfig(
            enabled=True,
            slm_nk_config=SLMNKConfig(enabled=False),
        )
        orch = ImmuneOrchestrator.from_config(config, {"hallucination": []})
        assert orch._slm_nk is None

    def test_from_config_passes_threshold(self):
        """from_config passes similarity_threshold from SLMNKConfig to SLMNKCell."""
        config = OrchestratorConfig(
            enabled=True,
            slm_nk_config=SLMNKConfig(enabled=True, similarity_threshold=0.7),
        )
        orch = ImmuneOrchestrator.from_config(config, {"hallucination": []})
        assert orch._slm_nk._similarity_threshold == 0.7


# -- TestPackageExports -------------------------------------------------------

class TestPackageExports:
    """Verify SLMNKCell and SLMNKConfig importable from orchestrator package."""

    def test_import_slm_nk_cell(self):
        """SLMNKCell importable from antigence_subnet.miner.orchestrator."""
        from antigence_subnet.miner.orchestrator import SLMNKCell as Imported
        assert Imported is SLMNKCell

    def test_import_slm_nk_config(self):
        """SLMNKConfig importable from antigence_subnet.miner.orchestrator."""
        from antigence_subnet.miner.orchestrator import SLMNKConfig as Imported
        assert Imported is SLMNKConfig


# -- TestPerDomainSLMThreshold ------------------------------------------------

class TestPerDomainSLMThreshold:
    """Per-domain slm_nk_similarity_threshold override passed to SLMNKCell."""

    def test_domain_threshold_passed_to_slm_nk(self):
        """Domain-specific slm_nk_similarity_threshold is passed as kwarg to SLMNKCell.process()."""
        slm_nk = _make_mock_slm_nk_cell(trigger_result=None)

        config = OrchestratorConfig(
            enabled=True,
            domain_configs={
                "hallucination": DomainConfig(slm_nk_similarity_threshold=0.6),
            },
        )

        orch = ImmuneOrchestrator(
            feature_extractor=_make_mock_extractor(),
            nk_cell=_make_nk_cell_stub(),
            dendritic_cell=_make_mock_dc(),
            danger_modulator=_make_mock_danger(),
            detectors={"hallucination": [_make_mock_detector()]},
            config=config,
            slm_nk_cell=slm_nk,
        )

        asyncio.run(orch.process("prompt", "output", "hallucination"))
        slm_nk.process.assert_called_once()
        _, kwargs = slm_nk.process.call_args
        assert kwargs.get("similarity_threshold") == 0.6

    def test_no_domain_threshold_no_kwarg(self):
        """When domain has no slm_nk_similarity_threshold, no override kwarg is passed."""
        slm_nk = _make_mock_slm_nk_cell(trigger_result=None)

        config = OrchestratorConfig(
            enabled=True,
            domain_configs={
                "hallucination": DomainConfig(nk_z_threshold=2.0),
            },
        )

        orch = ImmuneOrchestrator(
            feature_extractor=_make_mock_extractor(),
            nk_cell=_make_nk_cell_stub(),
            dendritic_cell=_make_mock_dc(),
            danger_modulator=_make_mock_danger(),
            detectors={"hallucination": [_make_mock_detector()]},
            config=config,
            slm_nk_cell=slm_nk,
        )

        asyncio.run(orch.process("prompt", "output", "hallucination"))
        slm_nk.process.assert_called_once()
        _, kwargs = slm_nk.process.call_args
        assert "similarity_threshold" not in kwargs

    def test_unconfigured_domain_no_override(self):
        """Domain with no config entry does not pass slm_nk override."""
        slm_nk = _make_mock_slm_nk_cell(trigger_result=None)

        orch = ImmuneOrchestrator(
            feature_extractor=_make_mock_extractor(),
            nk_cell=_make_nk_cell_stub(),
            dendritic_cell=_make_mock_dc(),
            danger_modulator=_make_mock_danger(),
            detectors={"code_security": [_make_mock_detector()]},
            slm_nk_cell=slm_nk,
        )

        asyncio.run(orch.process("prompt", "output", "code_security"))
        slm_nk.process.assert_called_once()
        _, kwargs = slm_nk.process.call_args
        assert "similarity_threshold" not in kwargs
