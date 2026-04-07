"""Unit tests for orchestrator auto-tuning sweep script.

Validates:
- Grid dimensions (4 z * 4 pamp * 5 alpha = 80)
- TOML output format with [miner.orchestrator.domains.<domain>] section
- Sweep JSON format with required keys
- Best config selection (max F1)
- TOML loadable by OrchestratorConfig.from_toml_raw()
- Default config values match benchmark_orchestrator.py defaults
"""

import json
import sys
from pathlib import Path

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestGridDimensions:
    """Test 1: Grid constants produce expected 80 combinations."""

    def test_grid_dimensions(self):
        from scripts.tune_orchestrator import DANGER_ALPHAS, PAMP_THRESHOLDS, Z_THRESHOLDS

        assert len(Z_THRESHOLDS) == 4, f"Expected 4 z_thresholds, got {len(Z_THRESHOLDS)}"
        assert len(PAMP_THRESHOLDS) == 4, f"Expected 4 pamp_thresholds, got {len(PAMP_THRESHOLDS)}"
        assert len(DANGER_ALPHAS) == 5, f"Expected 5 danger_alphas, got {len(DANGER_ALPHAS)}"
        total = len(Z_THRESHOLDS) * len(PAMP_THRESHOLDS) * len(DANGER_ALPHAS)
        assert total == 80, f"Expected 80 total combinations, got {total}"

    def test_grid_values(self):
        from scripts.tune_orchestrator import DANGER_ALPHAS, PAMP_THRESHOLDS, Z_THRESHOLDS

        assert Z_THRESHOLDS == [2.0, 3.0, 5.0, 8.0]
        assert PAMP_THRESHOLDS == [0.1, 0.3, 0.5, 0.7]
        assert DANGER_ALPHAS == [0.0, 0.05, 0.1, 0.2, 0.3]


class TestWriteTunedToml:
    """Test 4-5: write_tuned_toml produces valid TOML with correct structure."""

    def test_write_tuned_toml_format(self, tmp_path):
        from scripts.tune_orchestrator import write_tuned_toml

        best_config = {
            "nk_z_threshold": 3.0,
            "dca_pamp_threshold": 0.3,
            "danger_alpha": 0.1,
        }
        write_tuned_toml("hallucination", best_config, config_dir=str(tmp_path))

        toml_path = tmp_path / "hallucination.toml"
        assert toml_path.exists(), "TOML file not created"

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        # Verify nested structure
        assert "miner" in data
        assert "orchestrator" in data["miner"]
        assert "domains" in data["miner"]["orchestrator"]
        assert "hallucination" in data["miner"]["orchestrator"]["domains"]

        domain_section = data["miner"]["orchestrator"]["domains"]["hallucination"]
        assert domain_section["nk_z_threshold"] == 3.0
        assert domain_section["dca_pamp_threshold"] == 0.3
        assert domain_section["danger_alpha"] == 0.1
        assert domain_section["danger_enabled"] is True

    def test_toml_keys_present(self, tmp_path):
        from scripts.tune_orchestrator import write_tuned_toml

        best_config = {
            "nk_z_threshold": 5.0,
            "dca_pamp_threshold": 0.5,
            "danger_alpha": 0.2,
        }
        write_tuned_toml("code_security", best_config, config_dir=str(tmp_path))

        toml_path = tmp_path / "code_security.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        domain_section = data["miner"]["orchestrator"]["domains"]["code_security"]
        required_keys = {"nk_z_threshold", "dca_pamp_threshold", "danger_alpha", "danger_enabled"}
        assert set(domain_section.keys()) == required_keys


class TestWriteSweepJson:
    """Test 3: write_sweep_json produces valid JSON with required keys."""

    def test_write_sweep_json_format(self, tmp_path):
        from scripts.tune_orchestrator import write_sweep_json

        sweep_result = {
            "best_config": {"nk_z_threshold": 3.0, "dca_pamp_threshold": 0.3, "danger_alpha": 0.1},
            "best_f1": 0.97,
            "default_f1": 0.95,
            "all_results": [
                {
                    "z_threshold": 2.0,
                    "pamp_threshold": 0.1,
                    "alpha": 0.0,
                    "f1": 0.90,
                    "precision": 0.95,
                    "recall": 0.86,
                },
                {
                    "z_threshold": 3.0,
                    "pamp_threshold": 0.3,
                    "alpha": 0.1,
                    "f1": 0.97,
                    "precision": 0.98,
                    "recall": 0.96,
                },
            ],
        }
        write_sweep_json("hallucination", sweep_result, output_dir=str(tmp_path))

        json_path = tmp_path / "hallucination_sweep.json"
        assert json_path.exists(), "Sweep JSON file not created"

        with open(json_path) as f:
            data = json.load(f)

        assert "best_config" in data
        assert "best_f1" in data
        assert "all_results" in data
        assert "timestamp" in data
        assert isinstance(data["all_results"], list)
        assert isinstance(data["best_config"], dict)
        assert isinstance(data["best_f1"], float)


class TestBestConfigSelection:
    """Test 4: Selection logic picks the max-F1 entry."""

    def test_best_config_selection(self):
        from scripts.tune_orchestrator import select_best_config

        all_results = [
            {"z_threshold": 2.0, "pamp_threshold": 0.1, "alpha": 0.0, "f1": 0.85},
            {"z_threshold": 3.0, "pamp_threshold": 0.3, "alpha": 0.1, "f1": 0.97},
            {"z_threshold": 5.0, "pamp_threshold": 0.5, "alpha": 0.2, "f1": 0.92},
            {"z_threshold": 8.0, "pamp_threshold": 0.7, "alpha": 0.3, "f1": 0.88},
        ]
        best = select_best_config(all_results)
        assert best["z_threshold"] == 3.0
        assert best["pamp_threshold"] == 0.3
        assert best["alpha"] == 0.1
        assert best["f1"] == 0.97

    def test_best_config_returns_highest_f1(self):
        from scripts.tune_orchestrator import select_best_config

        all_results = [
            {"z_threshold": 2.0, "pamp_threshold": 0.1, "alpha": 0.0, "f1": 0.50},
            {"z_threshold": 8.0, "pamp_threshold": 0.7, "alpha": 0.3, "f1": 1.00},
        ]
        best = select_best_config(all_results)
        assert best["f1"] == 1.00


class TestTomlLoadableByOrchestratorConfig:
    """Test 5: Tuned TOML structure is parseable by OrchestratorConfig."""

    def test_toml_loadable_by_orchestrator_config(self, tmp_path):
        """Write tuned TOML, parse it, and load via OrchestratorConfig.from_toml_raw().

        Verifies the TOML structure is valid and keys match the expected
        Phase 36 domain config schema.
        """
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig
        from scripts.tune_orchestrator import write_tuned_toml

        best_config = {
            "nk_z_threshold": 3.0,
            "dca_pamp_threshold": 0.5,
            "danger_alpha": 0.2,
        }
        write_tuned_toml("hallucination", best_config, config_dir=str(tmp_path))

        toml_path = tmp_path / "hallucination.toml"
        with open(toml_path, "rb") as f:
            toml_raw = tomllib.load(f)

        # from_toml_raw should parse the domain config without error
        config = OrchestratorConfig.from_toml_raw(toml_raw)
        domain_cfg = config.get_domain_config("hallucination")
        assert domain_cfg is not None, "Domain config not parsed"
        assert domain_cfg.nk_z_threshold == 3.0
        assert domain_cfg.dca_pamp_threshold == 0.5
        assert domain_cfg.danger_alpha == 0.2
        assert domain_cfg.danger_enabled is True

    def test_toml_zero_alpha_disables_danger(self, tmp_path):
        """Verify alpha=0.0 produces danger_enabled=false in TOML."""
        from antigence_subnet.miner.orchestrator.config import OrchestratorConfig
        from scripts.tune_orchestrator import write_tuned_toml

        best_config = {
            "nk_z_threshold": 5.0,
            "dca_pamp_threshold": 0.3,
            "danger_alpha": 0.0,
        }
        write_tuned_toml("bio", best_config, config_dir=str(tmp_path))

        with open(tmp_path / "bio.toml", "rb") as f:
            toml_raw = tomllib.load(f)

        config = OrchestratorConfig.from_toml_raw(toml_raw)
        domain_cfg = config.get_domain_config("bio")
        assert domain_cfg is not None
        assert domain_cfg.danger_enabled is False
        assert domain_cfg.danger_alpha == 0.0


class TestDefaultConfigValues:
    """Test 6: Default parameters match benchmark_orchestrator.py defaults."""

    def test_default_config_values(self):
        from scripts.tune_orchestrator import (
            DEFAULT_DANGER_ALPHA,
            DEFAULT_DANGER_ENABLED,
            DEFAULT_PAMP_THRESHOLD,
            DEFAULT_Z_THRESHOLD,
        )

        assert DEFAULT_Z_THRESHOLD == 5.0, f"Expected default z=5.0, got {DEFAULT_Z_THRESHOLD}"
        assert DEFAULT_PAMP_THRESHOLD == 0.3, (
            f"Expected default pamp=0.3, got {DEFAULT_PAMP_THRESHOLD}"
        )
        assert DEFAULT_DANGER_ALPHA == 0.0, (
            f"Expected default alpha=0.0, got {DEFAULT_DANGER_ALPHA}"
        )
        assert DEFAULT_DANGER_ENABLED is False, (
            f"Expected default danger_enabled=False, got {DEFAULT_DANGER_ENABLED}"
        )
