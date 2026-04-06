"""Tests for TOML config file loading utility.

Covers: find_config_file, load_toml_config, flatten_toml, apply_toml_defaults.
Ensures CLI > TOML > code defaults precedence chain.
"""

import argparse
import sys

import pytest


class TestLoadTomlConfig:
    """Tests for load_toml_config()."""

    def test_load_toml(self, tmp_path):
        """load_toml_config(path) returns dict from valid TOML file."""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text('[neuron]\ntimeout = 12.0\ndevice = "cpu"\n')

        from antigence_subnet.utils.config_file import load_toml_config

        result = load_toml_config(toml_file)
        assert result == {"neuron": {"timeout": 12.0, "device": "cpu"}}

    def test_malformed_toml_error(self, tmp_path):
        """load_toml_config on invalid TOML raises TOMLDecodeError."""
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text("[neuron\ntimeout = ")  # malformed

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        from antigence_subnet.utils.config_file import load_toml_config

        with pytest.raises(tomllib.TOMLDecodeError):
            load_toml_config(toml_file)


class TestFlattenToml:
    """Tests for flatten_toml()."""

    def test_flatten_toml(self):
        """flatten_toml flattens single-level nested dict."""
        from antigence_subnet.utils.config_file import flatten_toml

        result = flatten_toml({"neuron": {"timeout": 12.0}})
        assert result == {"neuron.timeout": 12.0}

    def test_flatten_nested(self):
        """flatten_toml flattens multi-level nested dict."""
        from antigence_subnet.utils.config_file import flatten_toml

        result = flatten_toml({"validator": {"microglia": {"enabled": True}}})
        assert result == {"validator.microglia.enabled": True}

    def test_flatten_preserves_non_dict_values(self):
        """flatten_toml does not recurse into non-dict values."""
        from antigence_subnet.utils.config_file import flatten_toml

        result = flatten_toml({"miner": {"detectors": {"hallucination": "path"}}})
        assert result == {"miner.detectors.hallucination": "path"}

    def test_flatten_top_level_scalars(self):
        """flatten_toml preserves top-level scalar keys."""
        from antigence_subnet.utils.config_file import flatten_toml

        result = flatten_toml({"port": 8080, "debug": True})
        assert result == {"port": 8080, "debug": True}


class TestFindConfigFile:
    """Tests for find_config_file()."""

    def test_find_config_explicit(self, tmp_path):
        """find_config_file with explicit path returns Path if exists."""
        from antigence_subnet.utils.config_file import find_config_file

        toml_file = tmp_path / "custom.toml"
        toml_file.write_text("[neuron]\n")
        result = find_config_file(str(toml_file))
        assert result == toml_file

    def test_find_config_explicit_missing(self, tmp_path):
        """find_config_file with explicit nonexistent path returns None."""
        from antigence_subnet.utils.config_file import find_config_file

        result = find_config_file(str(tmp_path / "nonexistent.toml"))
        assert result is None

    def test_find_config_cwd(self, tmp_path, monkeypatch):
        """find_config_file(None) finds antigence_subnet.toml in CWD."""
        from antigence_subnet.utils.config_file import find_config_file

        toml_file = tmp_path / "antigence_subnet.toml"
        toml_file.write_text("[neuron]\n")
        monkeypatch.chdir(tmp_path)
        result = find_config_file(None)
        assert result is not None
        assert result.name == "antigence_subnet.toml"

    def test_find_config_none(self, tmp_path, monkeypatch):
        """find_config_file(None) returns None when no config exists."""
        from antigence_subnet.utils.config_file import find_config_file

        monkeypatch.chdir(tmp_path)
        # Also prevent home directory discovery
        monkeypatch.setattr(
            "antigence_subnet.utils.config_file.CONFIG_SEARCH_PATHS",
            [tmp_path / "nonexistent.toml"],
        )
        result = find_config_file(None)
        assert result is None


class TestApplyTomlDefaults:
    """Tests for apply_toml_defaults()."""

    def test_apply_toml_defaults(self, tmp_path):
        """TOML values override code defaults in argparse."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[neuron]\ntimeout = 30.0\n")

        parser = argparse.ArgumentParser()
        parser.add_argument("--neuron.timeout", type=float, default=12.0)

        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args([])
        assert getattr(args, "neuron.timeout") == 30.0

    def test_cli_overrides_toml(self, tmp_path):
        """CLI args override TOML values."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[neuron]\ntimeout = 30.0\n")

        parser = argparse.ArgumentParser()
        parser.add_argument("--neuron.timeout", type=float, default=12.0)

        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args(["--neuron.timeout", "60.0"])
        assert getattr(args, "neuron.timeout") == 60.0

    def test_missing_config_uses_defaults(self, tmp_path, monkeypatch):
        """apply_toml_defaults with no config file returns None and leaves defaults."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "antigence_subnet.utils.config_file.CONFIG_SEARCH_PATHS",
            [tmp_path / "nonexistent.toml"],
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--neuron.timeout", type=float, default=12.0)

        result = apply_toml_defaults(parser, config_path=None)
        assert result is None
        args = parser.parse_args([])
        assert getattr(args, "neuron.timeout") == 12.0

    def test_unknown_toml_keys_ignored(self, tmp_path):
        """TOML keys not in parser are silently ignored."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            "[neuron]\ntimeout = 30.0\nunknown_key = 999\n"
            "[completely_unknown]\nfoo = 1\n"
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--neuron.timeout", type=float, default=12.0)

        # Should not raise
        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args([])
        assert getattr(args, "neuron.timeout") == 30.0
        assert not hasattr(args, "unknown_key")

    def test_config_file_arg(self, tmp_path):
        """Parser with --config-file arg: apply_toml_defaults reads explicit path."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        toml_file = tmp_path / "custom.toml"
        toml_file.write_text("[neuron]\ntimeout = 42.0\n")

        parser = argparse.ArgumentParser()
        parser.add_argument("--config-file", type=str, default=None)
        parser.add_argument("--neuron.timeout", type=float, default=12.0)

        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args([])
        assert getattr(args, "neuron.timeout") == 42.0

    def test_validator_policy_section_maps_to_parser_defaults(self, tmp_path):
        """[validator.policy] keys map onto policy argparse aliases."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        toml_file = tmp_path / "policy.toml"
        toml_file.write_text(
            "[validator.policy]\n"
            'mode = "operator_multiband"\n'
            "high_threshold = 0.5\n"
            "low_threshold = 0.493536\n"
            "min_confidence = 0.6\n"
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("--policy.mode", "--validator.policy.mode", type=str, default="global_threshold")
        parser.add_argument("--policy.high_threshold", "--validator.policy.high_threshold", type=float, default=0.5)
        parser.add_argument("--policy.low_threshold", "--validator.policy.low_threshold", type=float, default=0.5)
        parser.add_argument("--policy.min_confidence", "--validator.policy.min_confidence", type=float, default=0.0)

        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args([])

        assert getattr(args, "policy.mode") == "operator_multiband"
        assert getattr(args, "policy.high_threshold") == 0.5
        assert getattr(args, "policy.low_threshold") == pytest.approx(0.493536)
        assert getattr(args, "policy.min_confidence") == 0.6


class TestCompatShim:
    """Test that the tomllib/tomli compat shim works."""

    def test_compat_shim_imports(self):
        """config_file module imports successfully with tomllib compat shim."""
        from antigence_subnet.utils.config_file import (
            apply_toml_defaults,
            find_config_file,
            flatten_toml,
            load_toml_config,
        )

        # All functions should be callable
        assert callable(find_config_file)
        assert callable(load_toml_config)
        assert callable(flatten_toml)
        assert callable(apply_toml_defaults)


class TestIntegration:
    """Integration tests for TOML config with neuron components."""

    def test_validator_toml_defaults(self, tmp_path):
        """Validator-style parser picks up TOML defaults."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[validator]\ntimeout = 99.0\n")

        parser = argparse.ArgumentParser()
        parser.add_argument("--validator.timeout", type=float, default=12.0)

        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args([])
        assert getattr(args, "validator.timeout") == 99.0

    def test_api_toml_defaults(self, tmp_path):
        """API server parser picks up TOML defaults."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[api]\nport = 9090\n")

        parser = argparse.ArgumentParser()
        parser.add_argument("--api.port", type=int, default=8080)

        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args([])
        assert getattr(args, "api.port") == 9090

    def test_restart_applies_config_change(self, tmp_path):
        """Changing config and re-applying picks up new values."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        toml_file = tmp_path / "config.toml"

        # First "startup" with value A
        toml_file.write_text("[neuron]\ntimeout = 30.0\n")
        parser_a = argparse.ArgumentParser()
        parser_a.add_argument("--neuron.timeout", type=float, default=12.0)
        apply_toml_defaults(parser_a, config_path=str(toml_file))
        args_a = parser_a.parse_args([])
        assert getattr(args_a, "neuron.timeout") == 30.0

        # Second "startup" with value B (simulates config change + restart)
        toml_file.write_text("[neuron]\ntimeout = 60.0\n")
        parser_b = argparse.ArgumentParser()
        parser_b.add_argument("--neuron.timeout", type=float, default=12.0)
        apply_toml_defaults(parser_b, config_path=str(toml_file))
        args_b = parser_b.parse_args([])
        assert getattr(args_b, "neuron.timeout") == 60.0

    def test_config_file_explicit_path(self, tmp_path):
        """apply_toml_defaults with explicit config_path reads that file."""
        from antigence_subnet.utils.config_file import apply_toml_defaults

        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        toml_file = custom_dir / "my_config.toml"
        toml_file.write_text("[neuron]\ntimeout = 77.0\n")

        parser = argparse.ArgumentParser()
        parser.add_argument("--neuron.timeout", type=float, default=12.0)

        result = apply_toml_defaults(parser, config_path=str(toml_file))
        assert result == toml_file
        args = parser.parse_args([])
        assert getattr(args, "neuron.timeout") == 77.0
