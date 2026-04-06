"""Tests for OrchestratorConfig dataclass and TOML backward compatibility.

Covers: from_toml_raw() with various inputs, safe defaults, sub-table access,
backward compatibility with v6.0 TOML files that lack [miner.orchestrator],
DomainConfig dataclass, domain config validation, and get_domain_config().
"""

import logging

import pytest

from antigence_subnet.miner.orchestrator.config import DomainConfig, OrchestratorConfig


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig.from_toml_raw() with dict inputs."""

    def test_empty_dict_returns_disabled(self):
        """OrchestratorConfig.from_toml_raw({}) returns config with enabled=False."""
        config = OrchestratorConfig.from_toml_raw({})
        assert config.enabled is False

    def test_enabled_true_from_toml(self):
        """from_toml_raw with enabled=true returns config with enabled=True."""
        raw = {"miner": {"orchestrator": {"enabled": True}}}
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.enabled is True

    def test_v6_style_no_orchestrator_key(self):
        """v6.0 style TOML (no orchestrator key) returns enabled=False."""
        raw = {"miner": {"detectors": {"hallucination": "some.module.path"}}}
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.enabled is False

    def test_nk_config_defaults_empty(self):
        """nk_config defaults to empty dict when [miner.orchestrator.nk] absent."""
        raw = {"miner": {"orchestrator": {"enabled": True}}}
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.nk_config == {}

    def test_nk_config_from_toml(self):
        """nk_config returns values when set in TOML."""
        raw = {"miner": {"orchestrator": {"nk": {"z_threshold": 2.5}}}}
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.nk_config == {"z_threshold": 2.5}

    def test_dca_config_defaults_empty(self):
        """dca_config defaults to empty dict when absent."""
        config = OrchestratorConfig.from_toml_raw({})
        assert config.dca_config == {}

    def test_danger_config_defaults_empty(self):
        """danger_config defaults to empty dict when absent."""
        config = OrchestratorConfig.from_toml_raw({})
        assert config.danger_config == {}

    def test_all_sub_configs_populated(self):
        """All sub-configs populated from full orchestrator TOML."""
        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "nk": {"z_threshold": 2.5, "skip_binary": True},
                    "dca": {"escalation_tiers": 3},
                    "danger": {"pamp_weight": 0.6, "danger_weight": 0.4},
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.enabled is True
        assert config.nk_config == {"z_threshold": 2.5, "skip_binary": True}
        assert config.dca_config == {"escalation_tiers": 3}
        assert config.danger_config == {"pamp_weight": 0.6, "danger_weight": 0.4}

    def test_dataclass_default_construction(self):
        """OrchestratorConfig() direct construction uses correct defaults."""
        config = OrchestratorConfig()
        assert config.enabled is False
        assert config.nk_config == {}
        assert config.dca_config == {}
        assert config.danger_config == {}


class TestTomlBackwardCompat:
    """Tests that v6.0 TOML files load without breaking."""

    def test_v6_config_loads_without_orchestrator(self, tmp_path):
        """Existing v6.0 TOML with [miner.detectors] but no [miner.orchestrator]
        loads via load_toml_config and produces OrchestratorConfig with defaults."""
        from antigence_subnet.utils.config_file import load_toml_config

        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[miner]\n'
            '[miner.detectors]\n'
            'hallucination = "antigence_subnet.miner.detectors.HallucinationDetector"\n'
            '\n'
            '[validator]\n'
            'timeout = 12.0\n'
        )

        raw = load_toml_config(toml_file)
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.enabled is False
        assert config.nk_config == {}
        assert config.dca_config == {}
        assert config.danger_config == {}

    def test_full_orchestrator_toml_integration(self, tmp_path):
        """Full TOML file with [miner.orchestrator] section loads and produces
        valid OrchestratorConfig with correct sub-configs."""
        from antigence_subnet.utils.config_file import load_toml_config

        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[miner]\n'
            '[miner.detectors]\n'
            'hallucination = "antigence_subnet.miner.detectors.HallucinationDetector"\n'
            '\n'
            '[miner.orchestrator]\n'
            'enabled = true\n'
            '\n'
            '[miner.orchestrator.nk]\n'
            'z_threshold = 3.0\n'
            '\n'
            '[miner.orchestrator.dca]\n'
            'escalation_tiers = 2\n'
            '\n'
            '[miner.orchestrator.danger]\n'
            'pamp_weight = 0.7\n'
            '\n'
            '[validator]\n'
            'timeout = 12.0\n'
        )

        raw = load_toml_config(toml_file)
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.enabled is True
        assert config.nk_config == {"z_threshold": 3.0}
        assert config.dca_config == {"escalation_tiers": 2}
        assert config.danger_config == {"pamp_weight": 0.7}


class TestDomainConfig:
    """Tests for DomainConfig dataclass and per-domain overrides."""

    def test_domain_config_stores_nk_z_threshold(self):
        """DomainConfig(nk_z_threshold=2.0) stores override, other fields are None."""
        dc = DomainConfig(nk_z_threshold=2.0)
        assert dc.nk_z_threshold == 2.0
        assert dc.dca_pamp_threshold is None
        assert dc.danger_alpha is None
        assert dc.danger_enabled is None

    def test_domain_config_all_none_means_defaults(self):
        """DomainConfig() with all None means 'use global defaults'."""
        dc = DomainConfig()
        assert dc.nk_z_threshold is None
        assert dc.dca_pamp_threshold is None
        assert dc.danger_alpha is None
        assert dc.danger_enabled is None

    def test_from_toml_with_domain_hallucination(self):
        """from_toml_raw with [miner.orchestrator.domains.hallucination] nk_z_threshold=2.0
        produces domain_configs['hallucination'].nk_z_threshold == 2.0."""
        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "domains": {
                        "hallucination": {"nk_z_threshold": 2.0},
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert "hallucination" in config.domain_configs
        assert config.domain_configs["hallucination"].nk_z_threshold == 2.0

    def test_from_toml_no_domains_produces_empty_dict(self):
        """from_toml_raw with no domains section produces domain_configs == {}."""
        raw = {"miner": {"orchestrator": {"enabled": True}}}
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.domain_configs == {}

    def test_partial_domain_override_leaves_others_none(self):
        """TOML with domains.hallucination only overriding nk_z_threshold
        leaves dca_pamp_threshold and danger_alpha as None."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"nk_z_threshold": 2.5},
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        dc = config.domain_configs["hallucination"]
        assert dc.nk_z_threshold == 2.5
        assert dc.dca_pamp_threshold is None
        assert dc.danger_alpha is None
        assert dc.danger_enabled is None

    def test_negative_nk_z_threshold_raises_valueerror(self):
        """from_toml_raw with nk_z_threshold=-1.0 raises ValueError."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"nk_z_threshold": -1.0},
                    },
                }
            }
        }
        with pytest.raises(ValueError, match="nk_z_threshold"):
            OrchestratorConfig.from_toml_raw(raw)

    def test_danger_alpha_above_1_raises_valueerror(self):
        """from_toml_raw with danger_alpha=1.5 raises ValueError."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"danger_alpha": 1.5},
                    },
                }
            }
        }
        with pytest.raises(ValueError, match="danger_alpha"):
            OrchestratorConfig.from_toml_raw(raw)

    def test_danger_alpha_negative_raises_valueerror(self):
        """from_toml_raw with danger_alpha=-0.1 raises ValueError."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"danger_alpha": -0.1},
                    },
                }
            }
        }
        with pytest.raises(ValueError, match="danger_alpha"):
            OrchestratorConfig.from_toml_raw(raw)

    def test_unknown_domain_logs_warning(self, caplog):
        """from_toml_raw with unknown domain 'fake_domain' does not raise
        but logs warning."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "fake_domain": {"nk_z_threshold": 2.0},
                    },
                }
            }
        }
        with caplog.at_level(logging.WARNING):
            config = OrchestratorConfig.from_toml_raw(raw)
        assert "fake_domain" in config.domain_configs
        assert any("fake_domain" in rec.message for rec in caplog.records)

    def test_v7_toml_backward_compat(self):
        """v7.0 TOML (no domains section) produces identical config to v7.0."""
        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "nk": {"z_threshold": 3.0},
                    "dca": {"pamp_threshold": 0.3},
                    "danger": {"alpha": 0.3, "enabled": True},
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.domain_configs == {}
        assert config.enabled is True
        assert config.nk_config == {"z_threshold": 3.0}

    def test_get_domain_config_returns_override(self):
        """get_domain_config('hallucination') returns DomainConfig with overrides."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"nk_z_threshold": 2.0, "danger_alpha": 0.5},
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        dc = config.get_domain_config("hallucination")
        assert dc is not None
        assert dc.nk_z_threshold == 2.0
        assert dc.danger_alpha == 0.5

    def test_get_domain_config_unconfigured_returns_none(self):
        """get_domain_config('unconfigured') returns None."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"nk_z_threshold": 2.0},
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.get_domain_config("unconfigured") is None

    def test_negative_dca_pamp_threshold_raises_valueerror(self):
        """from_toml_raw with dca_pamp_threshold=-0.5 raises ValueError."""
        raw = {
            "miner": {
                "orchestrator": {
                    "domains": {
                        "hallucination": {"dca_pamp_threshold": -0.5},
                    },
                }
            }
        }
        with pytest.raises(ValueError, match="dca_pamp_threshold"):
            OrchestratorConfig.from_toml_raw(raw)


class TestBCellConfigParsing:
    """Tests for bcell_config TOML parsing with embedding_mode keys (Phase 43)."""

    def test_bcell_embedding_mode_from_toml(self):
        """from_toml_raw with [miner.orchestrator.bcell] embedding_mode=true parses correctly."""
        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "bcell": {
                        "embedding_mode": True,
                        "embedding_sigma": 0.02,
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.bcell_config["embedding_mode"] is True
        assert config.bcell_config["embedding_sigma"] == 0.02

    def test_bcell_no_embedding_mode_produces_empty_key(self):
        """from_toml_raw without embedding_mode in bcell still parses bcell_config."""
        raw = {
            "miner": {
                "orchestrator": {
                    "enabled": True,
                    "bcell": {
                        "max_memory": 500,
                        "k": 10,
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        assert "embedding_mode" not in config.bcell_config
        assert config.bcell_config["max_memory"] == 500
        assert config.bcell_config["k"] == 10

    def test_bcell_config_defaults_empty_when_absent(self):
        """bcell_config defaults to empty dict when [miner.orchestrator.bcell] absent."""
        raw = {"miner": {"orchestrator": {"enabled": True}}}
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.bcell_config == {}

    def test_bcell_config_backward_compat_no_orchestrator(self):
        """v6.0 TOML (no orchestrator) produces empty bcell_config."""
        config = OrchestratorConfig.from_toml_raw({})
        assert config.bcell_config == {}

    def test_bcell_config_full_toml_integration(self, tmp_path):
        """Full TOML file with bcell embedding keys parses via load_toml_config."""
        from antigence_subnet.utils.config_file import load_toml_config

        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[miner]\n'
            '[miner.orchestrator]\n'
            'enabled = true\n'
            '\n'
            '[miner.orchestrator.bcell]\n'
            'max_memory = 500\n'
            'embedding_mode = true\n'
            'embedding_sigma = 0.015\n'
            'bcell_weight = 0.3\n'
        )

        raw = load_toml_config(toml_file)
        config = OrchestratorConfig.from_toml_raw(raw)
        assert config.bcell_config["embedding_mode"] is True
        assert config.bcell_config["embedding_sigma"] == 0.015
        assert config.bcell_config["max_memory"] == 500
        assert config.bcell_config["bcell_weight"] == 0.3

    def test_bcell_from_config_roundtrip(self):
        """BCell.from_config with parsed bcell_config produces correct parameters."""
        from antigence_subnet.miner.orchestrator.b_cell import BCell

        raw = {
            "miner": {
                "orchestrator": {
                    "bcell": {
                        "embedding_mode": True,
                        "embedding_sigma": 0.02,
                        "max_memory": 500,
                    },
                }
            }
        }
        config = OrchestratorConfig.from_toml_raw(raw)
        cell = BCell.from_config(config.bcell_config)
        # Without model_manager, embedding_mode falls back to False
        assert cell._embedding_mode is False
        assert cell._embedding_sigma == pytest.approx(0.02)
        assert cell._max_memory == 500
