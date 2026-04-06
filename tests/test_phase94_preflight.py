"""Tests for Phase 94 governance preflight validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.phase94_preflight import validate_phase94_preflight  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_preflight_fails_when_candidate_hash_is_stale(tmp_path):
    artifact_root = tmp_path / "artifacts"
    governance_dir = artifact_root / "governance"
    candidate_path = governance_dir / "deployment-candidate.json"
    candidate = {
        "classification": "internal-only",
        "public_release": False,
        "execution_mode": "same-host-private",
        "missing_required_env_vars": [],
        "ready_for_approval": True,
    }
    _write_json(candidate_path, candidate)
    _write_json(
        governance_dir / "no-start-checklist.json",
        {
            "execution_mode": "same-host-private",
            "deployment_candidate_path": str(candidate_path),
            "deployment_candidate_sha256": "stale-hash",
            "required_env_presence": {
                "BT_WALLET_PATH": True,
            },
            "approved_metrics_ports": {
                "PHASE94_MINER_METRICS_PORT": "9100",
                "PHASE94_VALIDATOR_METRICS_PORT": "9101",
            },
            "bt_wallet_path": {
                "present": True,
                "redacted_basename": ".../wallets",
            },
            "btcli": {
                "available": True,
            },
            "wallets": {
                "miner": {
                    "wallet_name": "miner",
                    "wallet_hotkey": "default",
                    "hotkey_ss58": "5miner",
                    "funded_wallet_confirmed": True,
                },
                "validator": {
                    "wallet_name": "validator",
                    "wallet_hotkey": "default",
                    "hotkey_ss58": "5validator",
                    "funded_wallet_confirmed": True,
                },
            },
        },
    )
    _write_json(
        governance_dir / "operator-approval.json",
        {
            "scope": "94-01-live-gate",
            "deployment_candidate_sha256": "stale-hash",
            "approved": False,
            "approval_status": "pending-human-approval",
            "approver": None,
            "approved_at_utc": None,
            "ready_for_live_action": False,
        },
    )

    ok, errors = validate_phase94_preflight(artifact_root, require_approval=False)

    assert ok is False
    assert any("stale deployment candidate hash" in error for error in errors)


def test_preflight_requires_live_action_fields_when_approval_is_requested(tmp_path):
    artifact_root = tmp_path / "artifacts"
    governance_dir = artifact_root / "governance"
    candidate_path = governance_dir / "deployment-candidate.json"
    candidate = {
        "classification": "internal-only",
        "public_release": False,
        "execution_mode": "same-host-private",
        "missing_required_env_vars": [],
        "ready_for_approval": True,
    }
    _write_json(candidate_path, candidate)
    candidate_sha256 = __import__("hashlib").sha256(candidate_path.read_bytes()).hexdigest()

    _write_json(
        governance_dir / "no-start-checklist.json",
        {
            "execution_mode": "same-host-private",
            "deployment_candidate_path": str(candidate_path),
            "deployment_candidate_sha256": candidate_sha256,
            "required_env_presence": {
                "BT_WALLET_PATH": True,
                "PHASE94_NETUID": True,
            },
            "approved_metrics_ports": {
                "PHASE94_MINER_METRICS_PORT": "9100",
                "PHASE94_VALIDATOR_METRICS_PORT": "9101",
            },
            "bt_wallet_path": {
                "present": True,
                "redacted_basename": ".../wallets",
            },
            "btcli": {
                "available": True,
            },
            "wallets": {
                "miner": {
                    "wallet_name": "miner",
                    "wallet_hotkey": "default",
                    "hotkey_ss58": "5miner",
                    "funded_wallet_confirmed": True,
                },
                "validator": {
                    "wallet_name": "validator",
                    "wallet_hotkey": "default",
                    "hotkey_ss58": "5validator",
                    "funded_wallet_confirmed": True,
                },
            },
        },
    )
    _write_json(
        governance_dir / "operator-approval.json",
        {
            "scope": "94-01-live-gate",
            "deployment_candidate_sha256": candidate_sha256,
            "approved": False,
            "approval_status": "pending-human-approval",
            "approver": None,
            "approved_at_utc": None,
            "ready_for_live_action": False,
        },
    )

    ok, errors = validate_phase94_preflight(artifact_root, require_approval=True)

    assert ok is False
    assert any("approved=true" in error for error in errors)
    assert any("record approver" in error for error in errors)


def test_preflight_passes_when_artifacts_are_consistent_and_approved(tmp_path):
    artifact_root = tmp_path / "artifacts"
    governance_dir = artifact_root / "governance"
    candidate_path = governance_dir / "deployment-candidate.json"
    candidate = {
        "classification": "internal-only",
        "public_release": False,
        "execution_mode": "same-host-private",
        "missing_required_env_vars": [],
        "ready_for_approval": True,
    }
    _write_json(candidate_path, candidate)
    candidate_sha256 = __import__("hashlib").sha256(candidate_path.read_bytes()).hexdigest()

    _write_json(
        governance_dir / "no-start-checklist.json",
        {
            "execution_mode": "same-host-private",
            "deployment_candidate_path": str(candidate_path),
            "deployment_candidate_sha256": candidate_sha256,
            "required_env_presence": {
                "BT_WALLET_PATH": True,
                "PHASE94_NETUID": True,
                "PHASE94_SUBTENSOR_ENDPOINT": True,
                "PHASE94_MINER_WALLET_NAME": True,
                "PHASE94_MINER_WALLET_HOTKEY": True,
                "PHASE94_VALIDATOR_WALLET_NAME": True,
                "PHASE94_VALIDATOR_WALLET_HOTKEY": True,
                "PHASE94_MINER_METRICS_PORT": True,
                "PHASE94_VALIDATOR_METRICS_PORT": True,
            },
            "approved_metrics_ports": {
                "PHASE94_MINER_METRICS_PORT": "9100",
                "PHASE94_VALIDATOR_METRICS_PORT": "9101",
            },
            "bt_wallet_path": {
                "present": True,
                "redacted_basename": ".../wallets",
            },
            "btcli": {
                "available": True,
            },
            "wallets": {
                "miner": {
                    "wallet_name": "miner",
                    "wallet_hotkey": "default",
                    "hotkey_ss58": "5miner",
                    "funded_wallet_confirmed": True,
                },
                "validator": {
                    "wallet_name": "validator",
                    "wallet_hotkey": "default",
                    "hotkey_ss58": "5validator",
                    "funded_wallet_confirmed": True,
                },
            },
        },
    )
    _write_json(
        governance_dir / "operator-approval.json",
        {
            "scope": "94-01-live-gate",
            "deployment_candidate_sha256": candidate_sha256,
            "approved": True,
            "approval_status": "approved",
            "approver": "byron",
            "approved_at_utc": "2026-04-05T15:04:20Z",
            "ready_for_live_action": True,
        },
    )

    ok, errors = validate_phase94_preflight(artifact_root, require_approval=True)

    assert ok is True
    assert errors == []
