"""Tests for ModelManager and ModelConfig.

Validates lazy model initialization, embed()/score() API, device auto-detection,
graceful degradation when sentence-transformers is unavailable, TOML config
parsing, and backward compatibility with v8.0 configs.
"""

import numpy as np
import pytest

# Skip tests that require actual model loading if sentence-transformers
# is not installed.
st = pytest.importorskip(
    "sentence_transformers", reason="model manager tests require sentence-transformers"
)


class TestModelConfig:
    """Tests for ModelConfig dataclass and TOML parsing."""

    def test_defaults(self):
        """ModelConfig() defaults: model_name='all-MiniLM-L6-v2', cache_dir=None, device='auto'."""
        from antigence_subnet.miner.orchestrator.config import ModelConfig

        cfg = ModelConfig()
        assert cfg.model_name == "all-MiniLM-L6-v2"
        assert cfg.cache_dir is None
        assert cfg.device == "auto"

    def test_from_toml_raw_empty(self):
        """ModelConfig.from_toml_raw({}) returns defaults (empty TOML = safe defaults)."""
        from antigence_subnet.miner.orchestrator.config import ModelConfig

        cfg = ModelConfig.from_toml_raw({})
        assert cfg.model_name == "all-MiniLM-L6-v2"
        assert cfg.cache_dir is None
        assert cfg.device == "auto"

    def test_from_toml_raw_with_values(self):
        """ModelConfig.from_toml_raw with [miner.model] section returns those values."""
        from antigence_subnet.miner.orchestrator.config import ModelConfig

        raw = {
            "miner": {
                "model": {
                    "model_name": "custom-model",
                    "cache_dir": "/tmp/models",
                    "device": "cpu",
                }
            }
        }
        cfg = ModelConfig.from_toml_raw(raw)
        assert cfg.model_name == "custom-model"
        assert cfg.cache_dir == "/tmp/models"
        assert cfg.device == "cpu"

    def test_orchestrator_config_has_model_config(self):
        """OrchestratorConfig.from_toml_raw() parses [miner.model] into model_config field."""
        from antigence_subnet.miner.orchestrator.config import (
            ModelConfig,
            OrchestratorConfig,
        )

        raw = {
            "miner": {
                "orchestrator": {"enabled": True},
                "model": {"model_name": "test-model", "device": "cpu"},
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert isinstance(config.model_config, ModelConfig)
        assert config.model_config.model_name == "test-model"
        assert config.model_config.device == "cpu"

    def test_orchestrator_config_default_model_config(self):
        """v8.0 TOML without [miner.model] produces OrchestratorConfig with default ModelConfig."""
        from antigence_subnet.miner.orchestrator.config import (
            ModelConfig,
            OrchestratorConfig,
        )

        raw = {"miner": {"orchestrator": {"enabled": True}}}
        config = OrchestratorConfig.from_toml_raw(raw)
        assert isinstance(config.model_config, ModelConfig)
        assert config.model_config.model_name == "all-MiniLM-L6-v2"
        assert config.model_config.cache_dir is None
        assert config.model_config.device == "auto"


class TestModelManager:
    """Tests for ModelManager embed()/score() API and lazy initialization."""

    def test_construct_without_loading(self):
        """ModelManager(config=ModelConfig()) constructs without loading any model (lazy)."""
        from antigence_subnet.miner.orchestrator.config import ModelConfig
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager(config=ModelConfig())
        assert mgr._model is None

    def test_model_none_before_embed(self):
        """ModelManager._model is None before first embed() or score() call."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        assert mgr._model is None
        assert mgr.loaded is False

    def test_embed_returns_correct_shape(self):
        """embed('hello world') returns np.ndarray of shape (384,) and dtype float32."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        result = mgr.embed("hello world")
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        assert result.dtype == np.float32

    def test_embed_empty_string(self):
        """embed('') returns np.ndarray of shape (384,) (handles empty string)."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        result = mgr.embed("")
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    def test_score_returns_float_in_range(self):
        """score('prompt text', 'output text') returns float in [0.0, 1.0]."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        s = mgr.score("prompt text", "output text")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_score_identical_texts_high(self):
        """score('identical text', 'identical text') returns float close to 1.0 (>0.95)."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        s = mgr.score("identical text", "identical text")
        assert s > 0.95

    def test_score_dissimilar_texts_low(self):
        """score('cats are pets', 'quantum physics equations') returns float < 0.5."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        s = mgr.score("cats are pets", "quantum physics equations")
        assert s < 0.5

    def test_model_loaded_after_embed(self):
        """ModelManager._model is not None after first embed() call (lazy loaded)."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        mgr.embed("test")
        assert mgr._model is not None
        assert mgr.loaded is True

    def test_model_cache_shared(self):
        """Two ModelManager instances with same config share the underlying SentenceTransformer."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr1 = ModelManager()
        mgr2 = ModelManager()
        mgr1.embed("test")
        mgr2.embed("test")
        assert mgr1._model is mgr2._model

    def test_cpu_device(self):
        """ModelManager works with device='cpu'."""
        from antigence_subnet.miner.orchestrator.config import ModelConfig
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        cfg = ModelConfig(device="cpu")
        mgr = ModelManager(config=cfg)
        result = mgr.embed("cpu test")
        assert result.shape == (384,)

    def test_is_available(self):
        """is_available() returns True when sentence-transformers installed."""
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        assert mgr.is_available() is True


class TestModelManagerUnavailable:
    """Tests for graceful degradation when sentence-transformers not available."""

    def test_is_available_false(self, monkeypatch):
        """is_available() returns False when sentence-transformers not installed."""
        import antigence_subnet.miner.orchestrator.model_manager as mm_module

        monkeypatch.setattr(mm_module, "_sbert_available", False)
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        assert mgr.is_available() is False

    def test_embed_raises_runtime_error(self, monkeypatch):
        """embed() raises RuntimeError when sentence-transformers not installed."""
        import antigence_subnet.miner.orchestrator.model_manager as mm_module

        monkeypatch.setattr(mm_module, "_sbert_available", False)
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            mgr.embed("test")

    def test_score_raises_runtime_error(self, monkeypatch):
        """score() raises RuntimeError when sentence-transformers not installed."""
        import antigence_subnet.miner.orchestrator.model_manager as mm_module

        monkeypatch.setattr(mm_module, "_sbert_available", False)
        from antigence_subnet.miner.orchestrator.model_manager import ModelManager

        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            mgr.score("prompt", "output")


class TestBackwardCompat:
    """Tests for package exports and backward compatibility."""

    def test_import_from_orchestrator_package(self):
        """Verify ModelManager and ModelConfig importable from orchestrator package."""
        from antigence_subnet.miner.orchestrator import ModelConfig, ModelManager

        assert ModelConfig is not None
        assert ModelManager is not None

    def test_orchestrator_config_without_model_section(self):
        """OrchestratorConfig.from_toml_raw with orchestrator but no [miner.model]
        produces model_config with defaults."""
        from antigence_subnet.miner.orchestrator import ModelConfig, OrchestratorConfig

        raw = {"miner": {"orchestrator": {"enabled": True}}}
        config = OrchestratorConfig.from_toml_raw(raw)
        assert isinstance(config.model_config, ModelConfig)
        assert config.model_config.model_name == "all-MiniLM-L6-v2"
        assert config.model_config.device == "auto"
        assert config.model_config.cache_dir is None

    def test_v8_toml_full_backward_compat(self):
        """v8.0-style TOML dict (orchestrator section, no [miner.model]) produces
        identical OrchestratorConfig behavior to pre-Phase-41."""
        from antigence_subnet.miner.orchestrator import OrchestratorConfig

        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "nk": {"z_threshold": 2.5},
                    "dca": {"escalation_tiers": 3},
                    "danger": {"pamp_weight": 0.6},
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.enabled is True
        assert config.nk_config == {"z_threshold": 2.5}
        assert config.dca_config == {"escalation_tiers": 3}
        assert config.danger_config == {"pamp_weight": 0.6}
        assert config.domain_configs == {}

    def test_model_config_in_full_toml(self):
        """TOML with both [miner.orchestrator] and [miner.model] parses both."""
        from antigence_subnet.miner.orchestrator import ModelConfig, OrchestratorConfig

        raw = {
            "miner": {
                "orchestrator": {"enabled": True, "nk": {"z_threshold": 3.0}},
                "model": {"model_name": "custom-embed", "device": "cpu"},
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.enabled is True
        assert config.nk_config == {"z_threshold": 3.0}
        assert isinstance(config.model_config, ModelConfig)
        assert config.model_config.model_name == "custom-embed"
        assert config.model_config.device == "cpu"

    def test_orchestrator_config_model_config_attribute(self):
        """OrchestratorConfig.from_toml_raw({}) has .model_config attribute that is ModelConfig."""
        from antigence_subnet.miner.orchestrator import ModelConfig, OrchestratorConfig

        config = OrchestratorConfig.from_toml_raw({})
        assert hasattr(config, "model_config")
        assert isinstance(config.model_config, ModelConfig)
