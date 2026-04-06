"""
Tests for validator cold-start protocol (VHARD-03).

Verifies ColdStartManager readiness checks, warmup lifecycle,
state persistence, TOML config parsing, and integration with
BaseValidatorNeuron.
"""

import json
import os

import numpy as np
import pytest

from antigence_subnet.validator.cold_start import (
    ColdStartConfig,
    ColdStartManager,
    parse_cold_start_config,
)


# ------------------------------------------------------------------ #
# ColdStartConfig
# ------------------------------------------------------------------ #


class TestColdStartConfig:
    """Tests for ColdStartConfig dataclass."""

    def test_defaults(self):
        """Config has documented defaults."""
        cfg = ColdStartConfig()
        assert cfg.max_startup_seconds == 60.0
        assert cfg.min_miners_required == 1
        assert cfg.warmup_rounds == 5

    def test_custom_values(self):
        """Config accepts custom values."""
        cfg = ColdStartConfig(
            max_startup_seconds=30.0,
            min_miners_required=3,
            warmup_rounds=10,
        )
        assert cfg.max_startup_seconds == 30.0
        assert cfg.min_miners_required == 3
        assert cfg.warmup_rounds == 10


# ------------------------------------------------------------------ #
# TOML parsing
# ------------------------------------------------------------------ #


class TestTomlParsing:
    """Tests for parse_cold_start_config TOML integration."""

    def test_empty_toml_uses_defaults(self):
        """Empty dict produces default config."""
        cfg = parse_cold_start_config({})
        assert cfg.max_startup_seconds == 60.0
        assert cfg.warmup_rounds == 5

    def test_partial_toml(self):
        """Partial TOML overrides only specified keys."""
        toml = {"validator": {"cold_start": {"warmup_rounds": 10}}}
        cfg = parse_cold_start_config(toml)
        assert cfg.warmup_rounds == 10
        assert cfg.max_startup_seconds == 60.0  # default

    def test_full_toml(self):
        """Full TOML section populates all fields."""
        toml = {
            "validator": {
                "cold_start": {
                    "max_startup_seconds": 30.0,
                    "min_miners_required": 4,
                    "warmup_rounds": 3,
                }
            }
        }
        cfg = parse_cold_start_config(toml)
        assert cfg.max_startup_seconds == 30.0
        assert cfg.min_miners_required == 4
        assert cfg.warmup_rounds == 3

    def test_missing_validator_section(self):
        """Missing [validator] section uses defaults."""
        cfg = parse_cold_start_config({"other": {}})
        assert cfg.max_startup_seconds == 60.0


# ------------------------------------------------------------------ #
# ColdStartManager readiness
# ------------------------------------------------------------------ #


class TestReadiness:
    """Tests for ColdStartManager readiness checks."""

    def test_all_checks_pass(self):
        """All checks True -> is_ready True."""
        m = ColdStartManager()
        assert m.is_ready(
            metagraph_synced=True,
            eval_dataset_loaded=True,
            dendrite_connected=True,
            n_miners=5,
        )

    def test_partial_checks_fail(self):
        """Missing eval dataset -> not ready."""
        m = ColdStartManager()
        assert not m.is_ready(
            metagraph_synced=True,
            eval_dataset_loaded=False,
            dendrite_connected=True,
            n_miners=5,
        )

    def test_no_checks_pass(self):
        """All False -> not ready."""
        m = ColdStartManager()
        assert not m.is_ready(
            metagraph_synced=False,
            eval_dataset_loaded=False,
            dendrite_connected=False,
            n_miners=0,
        )

    def test_not_enough_miners(self):
        """All checks pass but not enough miners -> not ready."""
        cfg = ColdStartConfig(min_miners_required=3)
        m = ColdStartManager(config=cfg)
        assert not m.is_ready(
            metagraph_synced=True,
            eval_dataset_loaded=True,
            dendrite_connected=True,
            n_miners=2,
        )

    def test_readiness_checks_returns_dict(self):
        """readiness_checks returns proper dict with expected keys."""
        m = ColdStartManager()
        checks = m.readiness_checks(
            metagraph_synced=True,
            eval_dataset_loaded=False,
            dendrite_connected=True,
        )
        assert checks == {
            "metagraph_synced": True,
            "eval_dataset_loaded": False,
            "dendrite_connected": True,
        }

    def test_startup_time_positive(self):
        """startup_time_seconds returns a positive value."""
        m = ColdStartManager()
        assert m.startup_time_seconds() >= 0.0


# ------------------------------------------------------------------ #
# Warmup lifecycle
# ------------------------------------------------------------------ #


class TestWarmup:
    """Tests for warmup round management."""

    def test_begin_warmup_reduces_sample_size(self):
        """begin_warmup returns min(4, original)."""
        m = ColdStartManager()
        reduced = m.begin_warmup(16)
        assert reduced == 4

    def test_begin_warmup_small_original(self):
        """begin_warmup with small original keeps original."""
        m = ColdStartManager()
        reduced = m.begin_warmup(2)
        assert reduced == 2

    def test_get_current_sample_size_during_warmup(self):
        """During warmup, effective sample_size is reduced."""
        m = ColdStartManager()
        m.begin_warmup(16)
        assert m.get_current_sample_size(16) == 4

    def test_get_current_sample_size_after_warmup(self):
        """After warmup complete, effective sample_size is restored."""
        cfg = ColdStartConfig(warmup_rounds=2)
        m = ColdStartManager(config=cfg)
        m.begin_warmup(16)
        m.record_warmup_round()
        m.record_warmup_round()
        assert m.warmup_complete is True
        assert m.get_current_sample_size(16) == 16

    def test_record_warmup_round_counts(self):
        """record_warmup_round increments and returns done status."""
        cfg = ColdStartConfig(warmup_rounds=3)
        m = ColdStartManager(config=cfg)
        m.begin_warmup(8)

        assert m.record_warmup_round() is False  # 1/3
        assert m.warmup_rounds_remaining == 2
        assert m.record_warmup_round() is False  # 2/3
        assert m.warmup_rounds_remaining == 1
        assert m.record_warmup_round() is True   # 3/3
        assert m.warmup_complete is True
        assert m.warmup_rounds_remaining == 0

    def test_warmup_rounds_remaining_starts_correct(self):
        """warmup_rounds_remaining starts at config value."""
        cfg = ColdStartConfig(warmup_rounds=7)
        m = ColdStartManager(config=cfg)
        assert m.warmup_rounds_remaining == 7


# ------------------------------------------------------------------ #
# State persistence
# ------------------------------------------------------------------ #


class TestStatePersistence:
    """Tests for cold-start state save/load round-trip."""

    def test_state_dict_round_trip(self):
        """get_state_dict -> load_state_dict preserves cold-start state."""
        m1 = ColdStartManager()
        m1.bootstrap_step = 42
        m1.warmup_complete = True
        m1._warmup_rounds_done = 5

        state = m1.get_state_dict()

        m2 = ColdStartManager()
        m2.load_state_dict(state)

        assert m2.bootstrap_step == 42
        assert m2.warmup_complete is True
        assert m2._warmup_rounds_done == 5

    def test_state_dict_no_bootstrap(self):
        """State dict with no bootstrap_step uses sentinel -1."""
        m = ColdStartManager()
        state = m.get_state_dict()
        assert state["bootstrap_step"] == -1
        assert state["warmup_complete"] is False

    def test_load_state_dict_with_sentinel(self):
        """Loading sentinel -1 sets bootstrap_step to None."""
        m = ColdStartManager()
        m.load_state_dict({"bootstrap_step": -1, "warmup_complete": False})
        assert m.bootstrap_step is None

    def test_state_dict_has_duration(self):
        """State dict includes startup_duration_seconds."""
        m = ColdStartManager()
        state = m.get_state_dict()
        assert "startup_duration_seconds" in state
        assert state["startup_duration_seconds"] >= 0.0


# ------------------------------------------------------------------ #
# Timeout behavior
# ------------------------------------------------------------------ #


class TestTimeout:
    """Tests for max_startup_seconds timeout."""

    def test_timeout_not_exceeded_initially(self):
        """Fresh manager has not exceeded timeout."""
        m = ColdStartManager()
        assert m.has_exceeded_startup_timeout() is False

    def test_timeout_exceeded_with_zero(self):
        """Zero timeout means immediately exceeded."""
        cfg = ColdStartConfig(max_startup_seconds=0.0)
        m = ColdStartManager(config=cfg)
        assert m.has_exceeded_startup_timeout() is True


# ------------------------------------------------------------------ #
# Validator integration: warmup complete on restart
# ------------------------------------------------------------------ #


class TestValidatorIntegration:
    """Integration tests with BaseValidatorNeuron."""

    def test_cold_start_attr_exists(self, mock_config):
        """Validator has cold_start attribute after init."""
        from neurons.validator import Validator

        v = Validator(config=mock_config)
        assert hasattr(v, "cold_start")
        assert isinstance(v.cold_start, ColdStartManager)

    def test_warmup_complete_skips_on_restart(self, mock_config):
        """If warmup_complete=True in saved state, skip warmup on load."""
        from neurons.validator import Validator

        v = Validator(config=mock_config)
        # Simulate completed warmup
        v.cold_start.warmup_complete = True
        v.cold_start.bootstrap_step = 10
        v.cold_start._warmup_rounds_done = 5

        # Save and reload
        v.save_state()
        v2 = Validator(config=mock_config)
        v2.load_state()

        assert v2.cold_start.warmup_complete is True
        assert v2.cold_start.bootstrap_step == 10
        assert v2.cold_start.get_current_sample_size(16) == 16

    def test_warmup_in_progress_resumes(self, mock_config):
        """If warmup was in-progress, resume from remaining rounds."""
        from neurons.validator import Validator

        v = Validator(config=mock_config)
        v.cold_start.warmup_complete = False
        v.cold_start._warmup_rounds_done = 3
        v.cold_start.bootstrap_step = 5

        v.save_state()
        v2 = Validator(config=mock_config)
        v2.load_state()

        assert v2.cold_start.warmup_complete is False
        assert v2.cold_start._warmup_rounds_done == 3
        assert v2.cold_start.warmup_rounds_remaining == 2  # 5 - 3

    def test_cold_start_state_in_npz(self, mock_config, tmp_path):
        """Verify cold_start key is present in saved .npz file."""
        from neurons.validator import Validator

        v = Validator(config=mock_config)
        v.cold_start.bootstrap_step = 7
        v.save_state()

        state_file = os.path.join(str(tmp_path), "state.npz")
        state = np.load(state_file, allow_pickle=True)
        assert "cold_start" in state.files
        cs_data = json.loads(str(state["cold_start"]))
        assert cs_data["bootstrap_step"] == 7

    def test_min_miners_skips_forward(self, mock_config):
        """With min_miners=100, is_ready returns False (not enough miners)."""
        from neurons.validator import Validator

        v = Validator(config=mock_config)
        v.cold_start.config.min_miners_required = 100
        # Mock metagraph has ~17 neurons
        n_miners = max(0, v.metagraph.n - 1)
        assert not v.cold_start.is_ready(
            metagraph_synced=True,
            eval_dataset_loaded=True,
            dendrite_connected=True,
            n_miners=n_miners,
        )
