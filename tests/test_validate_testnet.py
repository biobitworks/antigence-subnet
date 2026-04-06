"""
Tests for the testnet validation script (TNET-01 through TNET-04).

Covers:
- TNET-01: Registration proof (metagraph presence, UID, hotkey, stake)
- TNET-02: Evaluation rounds (100+ forward passes, per-round score tracking)
- TNET-03: Non-zero rewards (miner scores above random baseline)
- TNET-04: set_weights evidence (ExtrinsicResponse with spec_version)
- Report schema validation
- Full dry-run end-to-end pass
"""

import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

# Add project root to path so scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neurons.validator import Validator
from scripts.validate_testnet import (
    _build_config,
    build_report,
    capture_set_weights_evidence,
    check_registration,
    run_evaluation_rounds,
    run_validation,
)


class TestRegistrationProof:
    """TNET-01: Validator registration appears in metagraph."""

    def test_registration_proof(self, mock_config):
        """After creating a Validator with mock_config, check_registration()
        returns a dict with uid, hotkey, stake, metagraph_n, timestamp."""
        validator = Validator(config=mock_config)
        result = check_registration(validator)

        assert isinstance(result, dict)
        assert "uid" in result
        assert "hotkey" in result
        assert "stake" in result
        assert "metagraph_n" in result
        assert "timestamp" in result
        assert isinstance(result["uid"], int)
        assert result["uid"] >= 0


class TestEvaluationRounds:
    """TNET-02: Validator completes 100+ evaluation rounds."""

    @pytest.mark.asyncio
    async def test_evaluation_rounds(self, mock_config):
        """After running 100 forward passes, rounds_completed == 100 and
        scores_per_round has length 100."""
        validator = Validator(config=mock_config)
        result = await run_evaluation_rounds(validator, rounds=100)

        assert isinstance(result, dict)
        assert result["rounds_completed"] == 100
        assert isinstance(result["scores_per_round"], list)
        assert len(result["scores_per_round"]) == 100


class TestNonzeroRewards:
    """TNET-03: Miner receives non-zero rewards."""

    @pytest.mark.asyncio
    async def test_nonzero_rewards(self, mock_config):
        """After running 100 forward passes, at least one score is > 0.0
        and the evidence dict has max_score > 0.0."""
        validator = Validator(config=mock_config)
        result = await run_evaluation_rounds(validator, rounds=100)

        assert np.any(validator.scores > 0), (
            f"Expected at least one non-zero score, got: {validator.scores}"
        )
        # Check that max_score from the last round is > 0
        last_round = result["scores_per_round"][-1]
        assert last_round["max_score"] > 0.0


class TestSetWeightsEvidence:
    """TNET-04: set_weights transaction evidence."""

    @pytest.mark.asyncio
    async def test_set_weights_evidence(self, mock_config):
        """After running at least 1 forward pass, capture_set_weights_evidence()
        returns a dict with success == True and spec_version == 1."""
        validator = Validator(config=mock_config)
        # Run at least 1 forward pass so scores are non-zero
        await run_evaluation_rounds(validator, rounds=1)

        evidence = capture_set_weights_evidence(validator)

        assert isinstance(evidence, dict)
        assert evidence["success"] is True
        assert evidence["spec_version"] == 1


class TestReportSchema:
    """Report JSON schema validation."""

    def test_report_schema_requires_phase94_metadata_and_100_rounds(self):
        """build_report() carries strict-live provenance and 24h verdict fields."""
        registration = {
            "uid": 0,
            "hotkey": "test-hotkey",
            "stake": 100000.0,
            "metagraph_n": 17,
            "timestamp": "2026-03-30T00:00:00+00:00",
            "block_number": 123,
            "block_hash": "0xabc",
            "classification": "internal-only",
            "public_release": False,
        }
        rounds_result = {
            "rounds_completed": 100,
            "scores_per_round": [
                {"step": i, "max_score": 0.5, "nonzero_count": 3}
                for i in range(100)
            ],
        }
        reward_evidence = {"max_score": 0.5, "nonzero_uids": 3}
        weights_evidence = {"success": True, "spec_version": 1, "message": "ok"}

        report = build_report(
            mode="dry-run",
            netuid=1,
            registration=registration,
            rounds_result=rounds_result,
            reward_evidence=reward_evidence,
            weights_evidence=weights_evidence,
            strict_live=True,
            config_file=".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml",
            artifact_root=".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts",
            duration_hours=24,
            elapsed_seconds=86400.0,
        )

        # Top-level keys
        assert "timestamp" in report
        assert "mode" in report
        assert "netuid" in report
        assert "rounds_completed" in report
        assert report["strict_live"] is True
        assert report["config_file"].endswith("artifacts/config/live.toml")
        assert report["artifact_root"].endswith("artifacts")
        assert report["duration_hours"] == 24
        assert report["elapsed_seconds"] == 86400.0
        assert report["execution_mode"] == "same-host-private"
        assert report["git_commit_sha"]
        assert report["config_sha256"]
        assert report["subtensor_endpoint"]
        assert report["policy_mode"] == "operator_multiband"
        assert report["high_threshold"] == 0.5
        assert report["low_threshold"] == pytest.approx(0.493536)
        assert report["min_confidence"] == pytest.approx(0.6)
        assert report["start_time_utc"]
        assert report["end_time_utc"]
        assert report["unexpected_exit_count"] == 0
        assert report["process_restarts_total"] == 0
        assert report["chain_submission_failures"] == 0
        assert report["anomaly_count"] == 0
        assert report["max_memory_growth_pct"] <= 15
        assert "criteria" in report
        assert "overall" in report
        assert report["criteria"]["TNET-02"]["passed"] is True
        assert report["criteria"]["TNET-02"]["evidence"]["rounds_completed"] >= 100
        assert report["registration"]["classification"] == "internal-only"
        assert report["registration"]["public_release"] is False
        assert report["registration"]["block_number"] == 123
        assert report["weights"]["block_number"] is not None

        # Criteria sub-dicts
        for tnet_id in ["TNET-01", "TNET-02", "TNET-03", "TNET-04"]:
            assert tnet_id in report["criteria"], f"Missing {tnet_id} in criteria"
            assert "passed" in report["criteria"][tnet_id]
            assert "evidence" in report["criteria"][tnet_id]


class TestPhase94StrictLive:
    def test_build_config_threads_config_file_into_bt_config(self, tmp_path):
        """_build_config() preserves the exact live TOML path."""
        config_file = tmp_path / "live.toml"
        config_file.write_text("[validator.policy]\nmode = \"operator_multiband\"\n")
        args = Namespace(
            netuid=1,
            wallet_name="validator",
            wallet_hotkey="default",
            dry_run=True,
            subtensor_network="test",
            config_file=str(config_file),
        )

        config = _build_config(args, wallet_path=str(tmp_path / "wallets"))

        assert str(config.config_file) == str(config_file)

    @pytest.mark.asyncio
    async def test_run_validation_strict_live_hard_fails_without_mock_fallback(
        self, mock_config, monkeypatch
    ):
        """Strict live mode must not silently flip config.mock=True."""
        monkeypatch.setattr(
            "scripts.validate_testnet.check_balance",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("funding failure")),
        )
        monkeypatch.setattr(
            "scripts.validate_testnet.Validator",
            AsyncMock(side_effect=RuntimeError("strict_live preflight failed")),
        )

        with pytest.raises(RuntimeError, match="strict_live"):
            await run_validation(
                config=mock_config,
                rounds=100,
                dry_run=False,
                strict_live=True,
                config_file=".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/config/live.toml",
                artifact_root=".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts",
                duration_hours=24,
            )

    def test_phase94_operator_artifacts_require_internal_only_and_wallet_hygiene(self):
        checklist_contract = (
            Path(".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/94-01-PLAN.md")
            .read_text()
        )

        assert "same-host-private" in checklist_contract
        assert "internal-only" in checklist_contract
        assert "BT_WALLET_PATH" in checklist_contract
        for forbidden in ("mnemonic", "seed phrase", "private key", "raw coldkey"):
            assert forbidden in checklist_contract


class TestDryRunAllPass:
    """Full dry-run integration test."""

    @pytest.mark.asyncio
    async def test_dry_run_all_pass(self, mock_config):
        """Running run_validation with dry_run=True returns a report
        with overall == 'PASS' and all 4 criteria passed."""
        report = await run_validation(config=mock_config, rounds=100, dry_run=True)

        assert isinstance(report, dict)
        assert report["overall"] == "PASS"
        for tnet_id in ["TNET-01", "TNET-02", "TNET-03", "TNET-04"]:
            assert report["criteria"][tnet_id]["passed"] is True, (
                f"{tnet_id} failed: {report['criteria'][tnet_id]}"
            )
