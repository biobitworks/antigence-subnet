from __future__ import annotations

import argparse
import textwrap

import bittensor as bt
import pytest

from antigence_subnet.protocol import VerificationSynapse
from antigence_subnet.utils.config_file import apply_toml_defaults


class TestValidatorPolicyConfigSurface:
    def test_validator_policy_toml_maps_to_parser_defaults(self, tmp_path):
        toml_file = tmp_path / "policy.toml"
        toml_file.write_text(
            textwrap.dedent(
                """\
                [validator.policy]
                mode = "operator_multiband"
                high_threshold = 0.5
                low_threshold = 0.493536
                min_confidence = 0.6
                """
            )
        )

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--policy.mode", "--validator.policy.mode", type=str, default="global_threshold"
        )
        parser.add_argument(
            "--policy.high_threshold", "--validator.policy.high_threshold", type=float, default=0.5
        )
        parser.add_argument(
            "--policy.low_threshold", "--validator.policy.low_threshold", type=float, default=0.5
        )
        parser.add_argument(
            "--policy.min_confidence", "--validator.policy.min_confidence", type=float, default=0.0
        )

        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args([])

        assert getattr(args, "policy.mode") == "operator_multiband"
        assert getattr(args, "policy.high_threshold") == pytest.approx(0.5)
        assert getattr(args, "policy.low_threshold") == pytest.approx(0.493536)
        assert getattr(args, "policy.min_confidence") == pytest.approx(0.6)

    def test_cli_alias_overrides_validator_policy_toml(self, tmp_path):
        toml_file = tmp_path / "policy.toml"
        toml_file.write_text(
            textwrap.dedent(
                """\
                [validator.policy]
                mode = "operator_multiband"
                high_threshold = 0.5
                low_threshold = 0.493536
                min_confidence = 0.6
                """
            )
        )

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--policy.mode", "--validator.policy.mode", type=str, default="global_threshold"
        )
        parser.add_argument(
            "--policy.high_threshold", "--validator.policy.high_threshold", type=float, default=0.5
        )
        parser.add_argument(
            "--policy.low_threshold", "--validator.policy.low_threshold", type=float, default=0.5
        )
        parser.add_argument(
            "--policy.min_confidence", "--validator.policy.min_confidence", type=float, default=0.0
        )

        apply_toml_defaults(parser, config_path=str(toml_file))
        args = parser.parse_args(
            [
                "--validator.policy.mode",
                "global_threshold",
                "--validator.policy.high_threshold",
                "0.61",
            ]
        )

        assert getattr(args, "policy.mode") == "global_threshold"
        assert getattr(args, "policy.high_threshold") == pytest.approx(0.61)
        assert getattr(args, "policy.low_threshold") == pytest.approx(0.493536)
        assert getattr(args, "policy.min_confidence") == pytest.approx(0.6)

    def test_policy_namespace_does_not_change_synapse_response_fields(self):
        response_fields = {
            "anomaly_score",
            "confidence",
            "anomaly_type",
            "feature_attribution",
        }
        assert response_fields.issubset(VerificationSynapse.model_fields)
        assert "decision" not in VerificationSynapse.model_fields
        assert "policy" not in VerificationSynapse.model_fields


class TestValidatorPolicyCompatibilityBridge:
    def test_legacy_reward_threshold_maps_to_global_threshold_when_policy_missing(
        self, mock_config
    ):
        from antigence_subnet.base.validator import BaseValidatorNeuron

        if not hasattr(mock_config, "reward"):
            mock_config.reward = bt.Config()
        mock_config.reward.decision_threshold = 0.65

        validator = BaseValidatorNeuron(config=mock_config)

        assert validator.config.policy.mode == "global_threshold"
        assert validator.config.policy.high_threshold == pytest.approx(0.65)
        assert validator.config.policy.low_threshold == pytest.approx(0.65)
        assert validator.config.policy.min_confidence == pytest.approx(0.0)

    def test_explicit_policy_wins_over_legacy_reward_threshold(self, mock_config):
        from antigence_subnet.base.validator import BaseValidatorNeuron

        if not hasattr(mock_config, "reward"):
            mock_config.reward = bt.Config()
        mock_config.reward.decision_threshold = 0.77

        mock_config.policy = bt.Config()
        mock_config.policy.mode = "operator_multiband"
        mock_config.policy.high_threshold = 0.5
        mock_config.policy.low_threshold = 0.493536
        mock_config.policy.min_confidence = 0.6

        validator = BaseValidatorNeuron(config=mock_config)

        assert validator.config.policy.mode == "operator_multiband"
        assert validator.config.policy.high_threshold == pytest.approx(0.5)
        assert validator.config.policy.low_threshold == pytest.approx(0.493536)
        assert validator.config.policy.min_confidence == pytest.approx(0.6)
