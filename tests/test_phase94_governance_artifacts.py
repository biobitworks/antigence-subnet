"""Tests for Phase 94 governance artifact generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import generate_phase94_governance_artifacts as phase94_governance  # noqa: E402


def test_bootstrap_phase94_env_loads_shell_style_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.phase94"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "export PHASE94_NETUID=94",
                "PHASE94_SUBTENSOR_ENDPOINT='wss://example.test'",
                "PHASE94_MINER_WALLET_NAME=miner_wallet # trailing comment",
                'PHASE94_MINER_WALLET_HOTKEY="miner_hotkey"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for key in (
        "PHASE94_NETUID",
        "PHASE94_SUBTENSOR_ENDPOINT",
        "PHASE94_MINER_WALLET_NAME",
        "PHASE94_MINER_WALLET_HOTKEY",
    ):
        monkeypatch.delenv(key, raising=False)

    loaded = phase94_governance._bootstrap_phase94_env(env_file)

    assert loaded == env_file
    assert phase94_governance.os.environ["PHASE94_NETUID"] == "94"
    assert phase94_governance.os.environ["PHASE94_SUBTENSOR_ENDPOINT"] == "wss://example.test"
    assert phase94_governance.os.environ["PHASE94_MINER_WALLET_NAME"] == "miner_wallet"
    assert phase94_governance.os.environ["PHASE94_MINER_WALLET_HOTKEY"] == "miner_hotkey"


def test_governance_generator_keeps_approval_pending_even_with_env_file(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    config_dir = artifact_root / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "live.toml"
    config_file.write_text(
        "\n".join(
            [
                "[validator.policy]",
                'mode = "operator_multiband"',
                "high_threshold = 0.5",
                "low_threshold = 0.493536",
                "min_confidence = 0.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env.phase94"
    env_file.write_text(
        "\n".join(
            [
                "PHASE94_NETUID=94",
                "PHASE94_SUBTENSOR_ENDPOINT=wss://test.finney.opentensor.ai:443",
                "PHASE94_MINER_WALLET_NAME=miner_wallet",
                "PHASE94_MINER_WALLET_HOTKEY=miner_hotkey",
                "PHASE94_VALIDATOR_WALLET_NAME=validator_wallet",
                "PHASE94_VALIDATOR_WALLET_HOTKEY=validator_hotkey",
                "BT_WALLET_PATH=/tmp/wallets",
                "PHASE94_MINER_METRICS_PORT=9100",
                "PHASE94_VALIDATOR_METRICS_PORT=9101",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for key in phase94_governance.REQUIRED_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    loaded = phase94_governance._bootstrap_phase94_env(env_file)
    candidate, context = phase94_governance._generate_payloads(
        artifact_root=artifact_root,
        config_file=config_file,
        project_root=Path(__file__).resolve().parent.parent,
    )

    assert loaded == env_file
    assert candidate["ready_for_approval"] is True
    assert candidate["missing_required_env_vars"] == []
    assert candidate["netuid"] == 94
    assert candidate["subtensor_endpoint"] == "wss://test.finney.opentensor.ai:443"
    assert candidate["miner_metrics_port"] == "9100"
    assert candidate["validator_metrics_port"] == "9101"

    candidate_path = artifact_root / "governance" / "deployment-candidate.json"
    phase94_governance.atomic_write_json(candidate_path, candidate)
    candidate_sha256 = phase94_governance._json_sha256(candidate_path)
    approval = {
        "approved": False,
        "approval_status": "pending-human-approval",
        "generated_at_utc": context["generated_at_utc"],
        "generator": "scripts/generate_phase94_governance_artifacts.py",
        "scope": "94-01-live-gate",
        "deployment_candidate_sha256": candidate_sha256,
        "approved_at_utc": None,
        "approver": None,
        "ready_for_live_action": False,
    }

    assert approval["approved"] is False
    assert approval["approved_at_utc"] is None
    assert approval["approver"] is None
    assert candidate_sha256
    assert json.loads(candidate_path.read_text(encoding="utf-8"))["ready_for_approval"] is True
