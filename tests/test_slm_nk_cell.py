"""Tests for SLMNKCell semantic anomaly gate and SLMNKConfig.

Covers: SLMNKConfig defaults and TOML parsing, SLMNKCell Protocol conformance,
core detection logic with mock ModelManager, graceful degradation when
ModelManager is unavailable or raises, disabled state behavior, per-request
similarity_threshold overrides.
"""

import logging

import numpy as np
import pytest

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.orchestrator.cells import ImmuneCellType
from antigence_subnet.miner.orchestrator.config import OrchestratorConfig, SLMNKConfig
from antigence_subnet.miner.orchestrator.slm_nk_cell import SLMNKCell


class MockModelManager:
    """Mock ModelManager for testing SLMNKCell without real model loading."""

    def __init__(self, similarity: float = 0.5, available: bool = True):
        self._similarity = similarity
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def score(self, prompt: str, output: str) -> float:
        return self._similarity


class RaisingModelManager:
    """Mock ModelManager that raises on score()."""

    def is_available(self) -> bool:
        return True

    def score(self, prompt: str, output: str) -> float:
        raise RuntimeError("Model load failed")


# ---------------------------------------------------------------------------
# SLMNKConfig tests
# ---------------------------------------------------------------------------


class TestSLMNKConfig:
    """SLMNKConfig dataclass defaults, TOML parsing, and validation."""

    def test_defaults(self):
        """SLMNKConfig() defaults: enabled=True, similarity_threshold=0.3."""
        cfg = SLMNKConfig()
        assert cfg.enabled is True
        assert cfg.similarity_threshold == 0.3

    def test_from_toml_raw_empty_returns_defaults(self):
        """SLMNKConfig.from_toml_raw({}) returns defaults (empty TOML = safe defaults)."""
        cfg = SLMNKConfig.from_toml_raw({})
        assert cfg.enabled is True
        assert cfg.similarity_threshold == 0.3

    def test_from_toml_raw_with_values(self):
        """SLMNKConfig.from_toml_raw() parses [miner.orchestrator.slm_nk] values."""
        raw = {
            "miner": {
                "orchestrator": {
                    "slm_nk": {
                        "enabled": False,
                        "similarity_threshold": 0.5,
                    }
                }
            }
        }
        cfg = SLMNKConfig.from_toml_raw(raw)
        assert cfg.enabled is False
        assert cfg.similarity_threshold == 0.5

    def test_orchestrator_config_has_slm_nk_config(self):
        """OrchestratorConfig has slm_nk_config field of type SLMNKConfig."""
        config = OrchestratorConfig()
        assert hasattr(config, "slm_nk_config")
        assert isinstance(config.slm_nk_config, SLMNKConfig)

    def test_orchestrator_from_toml_raw_parses_slm_nk(self):
        """OrchestratorConfig.from_toml_raw() parses [miner.orchestrator.slm_nk]."""
        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "slm_nk": {
                        "enabled": False,
                        "similarity_threshold": 0.7,
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.slm_nk_config.enabled is False
        assert config.slm_nk_config.similarity_threshold == 0.7

    def test_v8_toml_without_slm_nk_backward_compat(self):
        """v8.0 TOML without [miner.orchestrator.slm_nk] produces default SLMNKConfig."""
        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "nk": {"z_threshold": 3.0},
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.slm_nk_config.enabled is True
        assert config.slm_nk_config.similarity_threshold == 0.3

    def test_similarity_threshold_above_1_raises_valueerror(self):
        """similarity_threshold > 1.0 raises ValueError."""
        raw = {
            "miner": {
                "orchestrator": {
                    "slm_nk": {"similarity_threshold": 1.5}
                }
            }
        }
        with pytest.raises(ValueError, match="similarity_threshold"):
            OrchestratorConfig.from_toml_raw(raw)

    def test_similarity_threshold_negative_raises_valueerror(self):
        """similarity_threshold < 0.0 raises ValueError."""
        raw = {
            "miner": {
                "orchestrator": {
                    "slm_nk": {"similarity_threshold": -0.1}
                }
            }
        }
        with pytest.raises(ValueError, match="similarity_threshold"):
            OrchestratorConfig.from_toml_raw(raw)

    def test_similarity_threshold_at_boundaries_valid(self):
        """similarity_threshold at 0.0 and 1.0 are valid."""
        for val in [0.0, 1.0]:
            raw = {
                "miner": {
                    "orchestrator": {
                        "slm_nk": {"similarity_threshold": val}
                    }
                }
            }
            config = OrchestratorConfig.from_toml_raw(raw)
            assert config.slm_nk_config.similarity_threshold == val


# ---------------------------------------------------------------------------
# SLMNKCell Protocol conformance
# ---------------------------------------------------------------------------


class TestSLMNKCellProtocol:
    """SLMNKCell satisfies ImmuneCellType Protocol."""

    def test_satisfies_protocol(self):
        """SLMNKCell instance passes isinstance(cell, ImmuneCellType)."""
        mm = MockModelManager()
        cell = SLMNKCell(model_manager=mm)
        assert isinstance(cell, ImmuneCellType)


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------


class TestSLMNKCellDetection:
    """SLMNKCell.process() detection logic with mock ModelManager."""

    def test_similarity_above_threshold_returns_none(self):
        """Similarity >= threshold -> returns None (no anomaly, passes gate)."""
        mm = MockModelManager(similarity=0.8)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.3)
        features = np.zeros(10)
        result = cell.process(features, "What is Python?", "Python is a language.")
        assert result is None

    def test_similarity_at_threshold_returns_none(self):
        """Similarity exactly at threshold -> returns None (not below)."""
        mm = MockModelManager(similarity=0.3)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.3)
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_similarity_below_threshold_returns_detection_result(self):
        """Similarity < threshold -> returns DetectionResult."""
        mm = MockModelManager(similarity=0.1)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.3)
        features = np.zeros(10)
        result = cell.process(features, "What is Python?", "The moon is blue.")
        assert result is not None
        assert isinstance(result, DetectionResult)
        assert result.anomaly_type == "slm_nk_semantic_disconnect"

    def test_score_is_1_minus_similarity(self):
        """DetectionResult.score == 1 - similarity."""
        mm = MockModelManager(similarity=0.1)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.3)
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output")
        assert result is not None
        assert result.score == pytest.approx(0.9)

    def test_confidence_is_1_minus_similarity(self):
        """DetectionResult.confidence == 1 - similarity."""
        mm = MockModelManager(similarity=0.2)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.3)
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output")
        assert result is not None
        assert result.confidence == pytest.approx(0.8)

    def test_zero_similarity_max_score(self):
        """Similarity=0.0 -> score=1.0, confidence=1.0."""
        mm = MockModelManager(similarity=0.0)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.3)
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output")
        assert result is not None
        assert result.score == 1.0
        assert result.confidence == 1.0

    def test_per_request_threshold_override(self):
        """similarity_threshold kwarg overrides default for that call."""
        # Default threshold 0.3, similarity 0.2 -> would detect
        mm = MockModelManager(similarity=0.2)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.3)
        features = np.zeros(10)

        # Override threshold to 0.1 -> similarity 0.2 >= 0.1 -> should pass
        result = cell.process(features, "prompt", "output", similarity_threshold=0.1)
        assert result is None

    def test_per_request_threshold_triggers_detection(self):
        """Per-request threshold can make previously-passing samples fail."""
        mm = MockModelManager(similarity=0.5)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.3)
        features = np.zeros(10)

        # Default: similarity 0.5 >= 0.3 -> passes
        result_default = cell.process(features, "prompt", "output")
        assert result_default is None

        # Override threshold to 0.8 -> similarity 0.5 < 0.8 -> detects
        result_override = cell.process(features, "prompt", "output", similarity_threshold=0.8)
        assert result_override is not None
        assert result_override.anomaly_type == "slm_nk_semantic_disconnect"

    def test_custom_default_threshold(self):
        """SLMNKCell with custom default threshold works correctly."""
        mm = MockModelManager(similarity=0.4)
        cell = SLMNKCell(model_manager=mm, similarity_threshold=0.5)
        features = np.zeros(10)
        # similarity 0.4 < threshold 0.5 -> detect
        result = cell.process(features, "prompt", "output")
        assert result is not None
        assert result.score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestSLMNKCellGracefulDegradation:
    """SLMNKCell returns None when ModelManager is unavailable or errors."""

    def test_model_unavailable_returns_none(self):
        """When model_manager.is_available() is False -> returns None."""
        mm = MockModelManager(available=False)
        cell = SLMNKCell(model_manager=mm)
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_model_unavailable_logs_warning(self, caplog):
        """When model_manager.is_available() is False -> logs warning."""
        mm = MockModelManager(available=False)
        cell = SLMNKCell(model_manager=mm)
        features = np.zeros(10)
        with caplog.at_level(logging.WARNING):
            cell.process(features, "prompt", "output")
        assert any("not available" in rec.message.lower() or "unavailable" in rec.message.lower() for rec in caplog.records)

    def test_score_raises_returns_none(self):
        """When model_manager.score() raises exception -> returns None."""
        mm = RaisingModelManager()
        cell = SLMNKCell(model_manager=mm)
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_score_raises_logs_warning(self, caplog):
        """When model_manager.score() raises -> logs warning."""
        mm = RaisingModelManager()
        cell = SLMNKCell(model_manager=mm)
        features = np.zeros(10)
        with caplog.at_level(logging.WARNING):
            cell.process(features, "prompt", "output")
        assert any("error" in rec.message.lower() or "failed" in rec.message.lower() or "exception" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Disabled state
# ---------------------------------------------------------------------------


class TestSLMNKCellDisabled:
    """SLMNKCell with enabled=False always returns None."""

    def test_disabled_returns_none(self):
        """When enabled=False, process() returns None regardless of similarity."""
        mm = MockModelManager(similarity=0.0)  # Would normally detect
        cell = SLMNKCell(model_manager=mm, enabled=False)
        features = np.zeros(10)
        result = cell.process(features, "prompt", "output")
        assert result is None

    def test_disabled_does_not_call_model(self):
        """When enabled=False, model_manager.score() is not called."""

        class TrackingModelManager:
            def __init__(self):
                self.score_called = False

            def is_available(self) -> bool:
                return True

            def score(self, prompt: str, output: str) -> float:
                self.score_called = True
                return 0.0

        mm = TrackingModelManager()
        cell = SLMNKCell(model_manager=mm, enabled=False)
        features = np.zeros(10)
        cell.process(features, "prompt", "output")
        assert mm.score_called is False
