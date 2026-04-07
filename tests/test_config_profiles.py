"""Tests for config profile TOML files (conservative, balanced, aggressive).

Verifies that all three profiles:
- Are valid TOML (parseable by tomllib)
- Parse successfully via OrchestratorConfig.from_toml_raw()
- Have parameter values matching the D-02 specification
- Include all 4 known domain override sections
- Have orchestrator.enabled = true
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from antigence_subnet.miner.orchestrator.config import OrchestratorConfig

PROFILES_DIR = Path(__file__).resolve().parent.parent / "configs" / "profiles"

PROFILE_NAMES = ["conservative", "balanced", "aggressive"]


@pytest.fixture(params=PROFILE_NAMES)
def profile_name(request: pytest.FixtureRequest) -> str:
    """Parameterized fixture yielding each profile name."""
    return request.param


@pytest.fixture
def profile_toml(profile_name: str) -> dict:
    """Load and return parsed TOML dict for a profile."""
    path = PROFILES_DIR / f"{profile_name}.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


@pytest.fixture
def profile_config(profile_toml: dict) -> OrchestratorConfig:
    """Parse profile TOML into OrchestratorConfig."""
    return OrchestratorConfig.from_toml_raw(profile_toml)


class TestProfileParsing:
    """All profiles must be valid TOML and parseable by OrchestratorConfig."""

    def test_profile_file_exists(self, profile_name: str):
        """Profile TOML file exists in configs/profiles/."""
        path = PROFILES_DIR / f"{profile_name}.toml"
        assert path.exists(), f"{path} does not exist"

    def test_profile_is_valid_toml(self, profile_toml: dict):
        """Profile is valid TOML (fixture implicitly tests this)."""
        assert isinstance(profile_toml, dict)

    def test_from_toml_raw_succeeds(self, profile_config: OrchestratorConfig):
        """OrchestratorConfig.from_toml_raw() succeeds without raising."""
        assert profile_config is not None

    def test_orchestrator_enabled(self, profile_config: OrchestratorConfig):
        """All profiles have orchestrator.enabled = true."""
        assert profile_config.enabled is True


class TestAllProfilesDomains:
    """All profiles must include all 4 known domain override sections."""

    EXPECTED_DOMAINS = {"hallucination", "code_security", "reasoning", "bio"}

    def test_has_all_domain_sections(self, profile_config: OrchestratorConfig):
        """Profile includes domain overrides for all 4 known domains."""
        assert set(profile_config.domain_configs.keys()) == self.EXPECTED_DOMAINS


class TestConservativeProfile:
    """Conservative profile: high thresholds, minimal false positives."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = PROFILES_DIR / "conservative.toml"
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        self.config = OrchestratorConfig.from_toml_raw(raw)

    def test_nk_z_threshold_high(self):
        """Conservative z_threshold >= 5.0 (very high bar)."""
        assert self.config.nk_config.get("z_threshold", 0) >= 5.0

    def test_danger_alpha_zero(self):
        """Conservative danger alpha == 0.0 (disabled)."""
        assert self.config.danger_config.get("alpha", -1) == 0.0

    def test_danger_disabled(self):
        """Conservative danger.enabled is false."""
        assert self.config.danger_config.get("enabled") is False

    def test_bcell_weight_low(self):
        """Conservative bcell_weight <= 0.1."""
        assert self.config.bcell_config.get("bcell_weight", 1.0) <= 0.1

    def test_slm_nk_threshold_high(self):
        """Conservative SLM NK similarity_threshold >= 0.5."""
        assert self.config.slm_nk_config.similarity_threshold >= 0.5

    def test_dca_pamp_threshold_high(self):
        """Conservative DCA pamp_threshold is high (needs strong signal)."""
        assert self.config.dca_config.get("pamp_threshold", 0) >= 0.4


class TestBalancedProfile:
    """Balanced profile: matches current code defaults exactly."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = PROFILES_DIR / "balanced.toml"
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        self.config = OrchestratorConfig.from_toml_raw(raw)

    def test_nk_z_threshold_default(self):
        """Balanced z_threshold == 3.0 (matches code default)."""
        assert self.config.nk_config.get("z_threshold") == 3.0

    def test_danger_alpha_default(self):
        """Balanced danger alpha == 0.3 (matches code default)."""
        assert self.config.danger_config.get("alpha") == 0.3

    def test_danger_enabled(self):
        """Balanced danger.enabled is true."""
        assert self.config.danger_config.get("enabled") is True

    def test_bcell_weight_default(self):
        """Balanced bcell_weight == 0.2 (matches code default)."""
        assert self.config.bcell_config.get("bcell_weight") == 0.2

    def test_slm_nk_threshold_default(self):
        """Balanced SLM NK similarity_threshold == 0.3."""
        assert self.config.slm_nk_config.similarity_threshold == 0.3

    def test_dca_pamp_threshold_default(self):
        """Balanced DCA pamp_threshold == 0.3 (matches code default)."""
        assert self.config.dca_config.get("pamp_threshold") == 0.3

    def test_balanced_profile_declares_phase93_policy_defaults(self):
        """Balanced profile pins the ADR-selected validator policy defaults."""
        path = PROFILES_DIR / "balanced.toml"
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        policy = raw.get("validator", {}).get("policy", {})
        assert policy.get("mode") == "operator_multiband"
        assert policy.get("high_threshold") == 0.5
        assert policy.get("low_threshold") == pytest.approx(0.493536)
        assert policy.get("min_confidence") == 0.6


class TestAggressiveProfile:
    """Aggressive profile: low thresholds, maximum recall."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = PROFILES_DIR / "aggressive.toml"
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        self.config = OrchestratorConfig.from_toml_raw(raw)

    def test_nk_z_threshold_low(self):
        """Aggressive z_threshold <= 2.5."""
        assert self.config.nk_config.get("z_threshold", 99) <= 2.5

    def test_danger_alpha_active(self):
        """Aggressive danger alpha >= 0.3 (full modulation)."""
        assert self.config.danger_config.get("alpha", 0) >= 0.3

    def test_danger_enabled(self):
        """Aggressive danger.enabled is true."""
        assert self.config.danger_config.get("enabled") is True

    def test_bcell_weight_high(self):
        """Aggressive bcell_weight >= 0.3 (strong memory influence)."""
        assert self.config.bcell_config.get("bcell_weight", 0) >= 0.3

    def test_slm_nk_threshold_low(self):
        """Aggressive SLM NK similarity_threshold <= 0.2."""
        assert self.config.slm_nk_config.similarity_threshold <= 0.2

    def test_dca_pamp_threshold_low(self):
        """Aggressive DCA pamp_threshold low (low bar for mature classification)."""
        assert self.config.dca_config.get("pamp_threshold", 1) <= 0.25

    def test_dca_adaptive_enabled(self):
        """Aggressive DCA adaptive mode is enabled."""
        assert self.config.dca_config.get("adaptive") is True

    def test_feedback_enabled(self):
        """Aggressive profile has feedback section (in TOML, parsed separately)."""
        path = PROFILES_DIR / "aggressive.toml"
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        feedback = raw.get("miner", {}).get("orchestrator", {}).get("feedback", {})
        assert feedback.get("enabled") is True


class TestValidatorPolicyProfiles:
    """All shipped validator profiles declare policy explicitly."""

    @pytest.mark.parametrize("profile_name", PROFILE_NAMES)
    def test_profile_declares_validator_policy(self, profile_name: str):
        path = PROFILES_DIR / f"{profile_name}.toml"
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        policy = raw.get("validator", {}).get("policy")
        assert isinstance(policy, dict), f"{profile_name} missing [validator.policy]"
        assert policy.get("mode") in {
            "global_threshold",
            "domain_thresholds",
            "operator_multiband",
        }


class TestProfileValidationIntegration:
    """Integration tests: profile configs pass validate_config CLI end-to-end."""

    def test_balanced_validate_zero_errors(self):
        """Balanced profile passes validate_config() with zero errors."""
        from antigence_subnet.validate_config import validate_config

        issues = validate_config(PROFILES_DIR / "balanced.toml")
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_conservative_validate_zero_errors(self):
        """Conservative profile passes validate_config() with zero errors."""
        from antigence_subnet.validate_config import validate_config

        issues = validate_config(PROFILES_DIR / "conservative.toml")
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_aggressive_validate_zero_errors(self):
        """Aggressive profile passes validate_config() with zero errors."""
        from antigence_subnet.validate_config import validate_config

        issues = validate_config(PROFILES_DIR / "aggressive.toml")
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_dry_run_returns_all_domains(self):
        """dry_run() on balanced.toml returns dict with all 4 domain keys."""
        from antigence_subnet.validate_config import dry_run

        result = dry_run(PROFILES_DIR / "balanced.toml")
        assert isinstance(result, dict)
        expected_domains = {"hallucination", "code_security", "reasoning", "bio"}
        assert set(result.keys()) == expected_domains

    def test_dry_run_has_nk_triggers_and_feature_stats(self):
        """dry_run result contains nk_triggers and feature_stats for domains with eval data."""
        from antigence_subnet.validate_config import dry_run

        result = dry_run(PROFILES_DIR / "balanced.toml")
        for domain, info in result.items():
            assert "nk_triggers" in info, f"{domain} missing nk_triggers"
            assert "feature_stats" in info, f"{domain} missing feature_stats"
            if info.get("status") == "ok" and info.get("samples_tested", 0) > 0:
                assert isinstance(info["feature_stats"], dict)
                assert len(info["feature_stats"]) > 0
