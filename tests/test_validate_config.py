"""Tests for antigence_subnet.validate_config CLI module.

Validates TOML structure checks, parameter range validation, unknown key
warnings, dry-run feature extraction, and exit codes for valid/invalid configs.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from antigence_subnet.validate_config import (
    dry_run,
    main,
    validate_config,
)

PROFILES_DIR = Path(__file__).resolve().parent.parent / "configs" / "profiles"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestValidConfig:
    """Valid TOML config produces exit code 0 and 'Config valid' message."""

    def test_balanced_profile_valid(self, tmp_path: Path):
        """Balanced profile passes validation with zero errors."""
        issues = validate_config(PROFILES_DIR / "balanced.toml")
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0

    def test_valid_config_exit_code_zero(self, capsys):
        """Valid config produces exit code 0 via main()."""
        with (
            pytest.raises(SystemExit) as exc_info,
            patch("sys.argv", ["validate_config", str(PROFILES_DIR / "balanced.toml")]),
        ):
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Config valid" in captured.out


class TestFileErrors:
    """Non-existent and malformed files produce exit code 1."""

    def test_nonexistent_file(self, tmp_path: Path):
        """Non-existent file produces error issue."""
        issues = validate_config(tmp_path / "nonexistent.toml")
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any(
            "not found" in e.message.lower() or "does not exist" in e.message.lower()
            for e in errors
        )

    def test_nonexistent_file_exit_code(self, capsys):
        """Non-existent file produces exit code 1 via main()."""
        with (
            pytest.raises(SystemExit) as exc_info,
            patch("sys.argv", ["validate_config", "/tmp/nonexistent_abc123.toml"]),
        ):
            main()
        assert exc_info.value.code == 1

    def test_malformed_toml(self, tmp_path: Path):
        """Malformed TOML produces error issue."""
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("this is [ not valid toml ]]]]")
        issues = validate_config(bad_toml)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1


class TestMissingSections:
    """Missing optional sections produce warnings, not errors."""

    def test_missing_orchestrator_section_warning(self, tmp_path: Path):
        """Missing [miner.orchestrator] is a warning (not error) -- optional section."""
        minimal = tmp_path / "minimal.toml"
        minimal.write_text("[miner]\n")
        issues = validate_config(minimal)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0  # No errors -- section is optional


class TestParameterRangeValidation:
    """Out-of-range parameter values produce validation errors."""

    def test_nk_z_threshold_negative(self, tmp_path: Path):
        """nk.z_threshold = -1.0 produces validation error."""
        toml_content = textwrap.dedent("""\
            [miner.orchestrator]
            enabled = true
            [miner.orchestrator.nk]
            z_threshold = -1.0
        """)
        path = tmp_path / "bad_nk.toml"
        path.write_text(toml_content)
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any("z_threshold" in e.message for e in errors)

    def test_danger_alpha_out_of_range(self, tmp_path: Path):
        """danger.alpha = 2.0 produces validation error (out of [0,1] range)."""
        toml_content = textwrap.dedent("""\
            [miner.orchestrator]
            enabled = true
            [miner.orchestrator.danger]
            alpha = 2.0
        """)
        path = tmp_path / "bad_danger.toml"
        path.write_text(toml_content)
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any("alpha" in e.message for e in errors)

    def test_dca_pamp_threshold_negative(self, tmp_path: Path):
        """dca.pamp_threshold = -0.5 produces validation error."""
        toml_content = textwrap.dedent("""\
            [miner.orchestrator]
            enabled = true
            [miner.orchestrator.dca]
            pamp_threshold = -0.5
        """)
        path = tmp_path / "bad_dca.toml"
        path.write_text(toml_content)
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any("pamp_threshold" in e.message for e in errors)

    def test_bcell_weight_out_of_range(self, tmp_path: Path):
        """bcell.bcell_weight = 1.5 produces validation error (out of [0,1])."""
        toml_content = textwrap.dedent("""\
            [miner.orchestrator]
            enabled = true
            [miner.orchestrator.bcell]
            bcell_weight = 1.5
        """)
        path = tmp_path / "bad_bcell.toml"
        path.write_text(toml_content)
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any("bcell_weight" in e.message for e in errors)

    def test_slm_nk_similarity_threshold_negative(self, tmp_path: Path):
        """slm_nk.similarity_threshold = -0.1 produces validation error."""
        toml_content = textwrap.dedent("""\
            [miner.orchestrator]
            enabled = true
            [miner.orchestrator.slm_nk]
            similarity_threshold = -0.1
        """)
        path = tmp_path / "bad_slm_nk.toml"
        path.write_text(toml_content)
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) >= 1
        assert any("similarity_threshold" in e.message for e in errors)


class TestUnknownKeys:
    """Unknown TOML keys produce warnings but not errors."""

    def test_unknown_top_level_key(self, tmp_path: Path):
        """Unknown top-level key produces warning, not error."""
        toml_content = textwrap.dedent("""\
            [miner.orchestrator]
            enabled = true
            [completely_unknown_section]
            foo = "bar"
        """)
        path = tmp_path / "unknown_key.toml"
        path.write_text(toml_content)
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]
        assert len(errors) == 0
        assert len(warnings) >= 1
        assert any("unknown" in w.message.lower() for w in warnings)

    def test_unknown_key_inside_orchestrator(self, tmp_path: Path):
        """Unknown key inside [miner.orchestrator] produces warning."""
        toml_content = textwrap.dedent("""\
            [miner.orchestrator]
            enabled = true
            bogus_param = 42
        """)
        path = tmp_path / "unknown_orch.toml"
        path.write_text(toml_content)
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]
        assert len(errors) == 0
        assert len(warnings) >= 1


class TestValidatorScoringValidation:
    """[validator.scoring] is recognized and range-checked."""

    def test_invalid_scoring_mode_is_error(self, tmp_path: Path):
        path = tmp_path / "bad_scoring_mode.toml"
        path.write_text(
            textwrap.dedent(
                """\
                [validator.scoring]
                mode = "stochastic"
                """
            )
        )
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert any("mode" in issue.message for issue in errors)

    def test_invalid_scoring_ranges_are_errors(self, tmp_path: Path):
        path = tmp_path / "bad_scoring_ranges.toml"
        path.write_text(
            textwrap.dedent(
                """\
                [validator.scoring]
                repeats = 0
                ci_level = 1.5
                """
            )
        )
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert any("repeats" in issue.message for issue in errors)
        assert any("ci_level" in issue.message for issue in errors)

    def test_unknown_validator_scoring_key_is_warning_only(self, tmp_path: Path):
        path = tmp_path / "unknown_scoring_key.toml"
        path.write_text(
            textwrap.dedent(
                """\
                [validator.scoring]
                mode = "exact"
                unsupported = true
                """
            )
        )
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]
        assert len(errors) == 0
        assert any("unsupported" in issue.message for issue in warnings)


class TestValidatorPolicyValidation:
    """[validator.policy] is recognized, range-checked, and warns on extras."""

    def test_invalid_policy_mode_is_error(self, tmp_path: Path):
        path = tmp_path / "bad_policy_mode.toml"
        path.write_text(
            textwrap.dedent(
                """\
                [validator.policy]
                mode = "manual_review"
                """
            )
        )
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert any(
            issue.section == "validator.policy" and "mode" in issue.message for issue in errors
        )

    def test_invalid_policy_ranges_are_errors(self, tmp_path: Path):
        path = tmp_path / "bad_policy_ranges.toml"
        path.write_text(
            textwrap.dedent(
                """\
                [validator.policy]
                mode = "operator_multiband"
                high_threshold = 1.1
                low_threshold = -0.1
                min_confidence = 1.5
                """
            )
        )
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        assert any("high_threshold" in issue.message for issue in errors)
        assert any("low_threshold" in issue.message for issue in errors)
        assert any("min_confidence" in issue.message for issue in errors)

    def test_unknown_validator_policy_key_is_warning_only(self, tmp_path: Path):
        path = tmp_path / "unknown_policy_key.toml"
        path.write_text(
            textwrap.dedent(
                """\
                [validator.policy]
                mode = "global_threshold"
                high_threshold = 0.5
                fallback = "review"
                """
            )
        )
        issues = validate_config(path)
        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]
        assert len(errors) == 0
        assert any("fallback" in issue.message for issue in warnings)


class TestDryRun:
    """--dry-run flag loads config and reports score range output."""

    def test_dry_run_returns_dict(self, tmp_path: Path):
        """dry_run() returns a dict with domain keys."""
        result = dry_run(PROFILES_DIR / "balanced.toml")
        assert isinstance(result, dict)
        # Should have entries for domains with available evaluation data
        assert len(result) > 0

    def test_dry_run_contains_nk_triggers(self):
        """Dry-run result contains nk_triggers for domains with eval data."""
        result = dry_run(PROFILES_DIR / "balanced.toml")
        for _domain, info in result.items():
            assert "nk_triggers" in info
            assert "feature_stats" in info

    def test_dry_run_flag_via_main(self, capsys):
        """--dry-run flag via main() doesn't raise and exits 0."""
        with (
            pytest.raises(SystemExit) as exc_info,
            patch(
                "sys.argv",
                ["validate_config", "--dry-run", str(PROFILES_DIR / "balanced.toml")],
            ),
        ):
            main()
        assert exc_info.value.code == 0


class TestAllProfilesValid:
    """All three profile TOMLs pass validation with exit code 0."""

    @pytest.mark.parametrize("profile", ["balanced", "conservative", "aggressive"])
    def test_profile_passes_validation(self, profile: str):
        """Profile TOML passes validation with zero errors."""
        issues = validate_config(PROFILES_DIR / f"{profile}.toml")
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0, f"{profile} profile has errors: {errors}"
