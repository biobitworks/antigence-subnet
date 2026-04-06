#!/usr/bin/env python3
"""Generate pending governance artifacts for the Phase 94 live gate."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tomllib

from antigence_subnet.utils.runtime_metrics import atomic_write_json

DEFAULT_ARTIFACT_ROOT = Path(
    ".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts"
)
DEFAULT_ENV_FILE = Path(".env.phase94")
DEFAULT_POLICY = {
    "execution_mode": "same-host-private",
    "policy_mode": "operator_multiband",
    "high_threshold": 0.5,
    "low_threshold": 0.493536,
    "min_confidence": 0.6,
}
REDACTION_RULES = [
    "Never record mnemonic phrases, seed values, private keys, or raw coldkey material.",
    "Record wallet readiness as booleans plus public hotkey SS58 only.",
    "Record BT_WALLET_PATH as presence-only or a redacted basename, never the full raw path.",
]
REQUIRED_ENV_VARS = (
    "PHASE94_NETUID",
    "PHASE94_SUBTENSOR_ENDPOINT",
    "PHASE94_MINER_WALLET_NAME",
    "PHASE94_MINER_WALLET_HOTKEY",
    "PHASE94_VALIDATOR_WALLET_NAME",
    "PHASE94_VALIDATOR_WALLET_HOTKEY",
    "BT_WALLET_PATH",
    "PHASE94_MINER_METRICS_PORT",
    "PHASE94_VALIDATOR_METRICS_PORT",
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _normalize_env_value(value: str) -> str:
    if " #" in value and not value.lstrip().startswith(('"', "'")):
        value = value.split(" #", 1)[0].rstrip()
    return _strip_wrapping_quotes(value.strip())


def _load_env_file(path: Path) -> dict[str, str]:
    env_values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid env assignment in {path}:{line_number}")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing env key in {path}:{line_number}")
        env_values[key] = _normalize_env_value(raw_value)
    return env_values


def _bootstrap_phase94_env(env_file: Path | None) -> Path | None:
    candidate = env_file
    if candidate is None and DEFAULT_ENV_FILE.exists():
        candidate = DEFAULT_ENV_FILE
    if candidate is None or not candidate.exists():
        return None
    for key, value in _load_env_file(candidate).items():
        os.environ.setdefault(key, value)
    return candidate


def _current_git_commit_sha(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_policy(config_file: Path) -> dict[str, float | str]:
    if not config_file.exists():
        return DEFAULT_POLICY.copy()
    with config_file.open("rb") as handle:
        config = tomllib.load(handle)
    policy = config.get("validator", {}).get("policy", {})
    return {
        "execution_mode": DEFAULT_POLICY["execution_mode"],
        "policy_mode": policy.get("mode", DEFAULT_POLICY["policy_mode"]),
        "high_threshold": policy.get(
            "high_threshold",
            DEFAULT_POLICY["high_threshold"],
        ),
        "low_threshold": policy.get(
            "low_threshold",
            DEFAULT_POLICY["low_threshold"],
        ),
        "min_confidence": policy.get(
            "min_confidence",
            DEFAULT_POLICY["min_confidence"],
        ),
    }


def _btcli_info() -> dict[str, str | bool | None]:
    executable = shutil.which("btcli")
    if not executable:
        return {
            "available": False,
            "path": None,
            "version": None,
        }
    version = None
    for args in (["btcli", "--version"], ["btcli", "version"]):
        try:
            version = subprocess.check_output(
                args,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            if version:
                break
        except Exception:  # noqa: BLE001
            continue
    return {
        "available": True,
        "path": executable,
        "version": version or "unknown",
    }


def _redacted_wallet_path() -> str | None:
    wallet_path = os.getenv("BT_WALLET_PATH")
    if not wallet_path:
        return None
    return f".../{Path(wallet_path).name}"


def _env_presence() -> dict[str, bool]:
    return {name: bool(os.getenv(name)) for name in REQUIRED_ENV_VARS}


def _wallet_payload(prefix: str) -> dict[str, str | bool | None]:
    wallet_name = os.getenv(f"PHASE94_{prefix}_WALLET_NAME")
    wallet_hotkey = os.getenv(f"PHASE94_{prefix}_WALLET_HOTKEY")
    return {
        "wallet_name": wallet_name or None,
        "wallet_hotkey": wallet_hotkey or None,
        "hotkey_ss58": None,
        "funded_wallet_confirmed": False,
        "ready_for_live_action": bool(wallet_name and wallet_hotkey),
    }


def _json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate_payloads(
    *,
    artifact_root: Path,
    config_file: Path,
    project_root: Path,
) -> tuple[dict, dict]:
    governance_dir = artifact_root / "governance"
    governance_dir.mkdir(parents=True, exist_ok=True)

    env_presence = _env_presence()
    policy = _load_policy(config_file)
    git_commit_sha = _current_git_commit_sha(project_root)
    config_sha256 = _sha256_file(config_file)
    generated_at = _iso_now()
    candidate = {
        "status": "pending-human-input",
        "generated_at_utc": generated_at,
        "generator": "scripts/generate_phase94_governance_artifacts.py",
        "git_commit_sha": git_commit_sha,
        "config_file": str(config_file),
        "config_sha256": config_sha256,
        "netuid": int(os.getenv("PHASE94_NETUID", "0")),
        "subtensor_endpoint": os.getenv("PHASE94_SUBTENSOR_ENDPOINT", "unknown"),
        "execution_mode": policy["execution_mode"],
        "policy_mode": policy["policy_mode"],
        "high_threshold": policy["high_threshold"],
        "low_threshold": policy["low_threshold"],
        "min_confidence": policy["min_confidence"],
        "miner_metrics_port": os.getenv("PHASE94_MINER_METRICS_PORT"),
        "validator_metrics_port": os.getenv("PHASE94_VALIDATOR_METRICS_PORT"),
        "classification": "internal-only",
        "public_release": False,
        "ready_for_approval": all(env_presence.values()),
        "missing_required_env_vars": [
            name for name, is_present in env_presence.items() if not is_present
        ],
    }
    return candidate, {
        "generated_at_utc": generated_at,
        "policy": policy,
        "env_presence": env_presence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate pending Phase 94 governance artifacts.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Artifact root containing config/, governance/, validation/, and stability-24h/.",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to the copied live TOML config. Defaults to <artifact-root>/config/live.toml.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help=(
            "Optional shell-style env file with Phase 94 values. If omitted, "
            f"{DEFAULT_ENV_FILE} is loaded automatically when present."
        ),
    )
    args = parser.parse_args()

    loaded_env_file = _bootstrap_phase94_env(Path(args.env_file) if args.env_file else None)
    artifact_root = Path(args.artifact_root)
    config_file = (
        Path(args.config_file)
        if args.config_file
        else artifact_root / "config" / "live.toml"
    )
    project_root = Path(__file__).resolve().parent.parent
    artifact_root.mkdir(parents=True, exist_ok=True)

    candidate, context = _generate_payloads(
        artifact_root=artifact_root,
        config_file=config_file,
        project_root=project_root,
    )
    governance_dir = artifact_root / "governance"
    candidate_path = governance_dir / "deployment-candidate.json"
    atomic_write_json(candidate_path, candidate)
    candidate_sha256 = _json_sha256(candidate_path)

    checklist = {
        "status": "pending-human-input",
        "generated_at_utc": context["generated_at_utc"],
        "generator": "scripts/generate_phase94_governance_artifacts.py",
        "execution_mode": context["policy"]["execution_mode"],
        "deployment_candidate_path": str(candidate_path),
        "deployment_candidate_sha256": candidate_sha256,
        "approved_metrics_ports": {
            "PHASE94_MINER_METRICS_PORT": os.getenv("PHASE94_MINER_METRICS_PORT"),
            "PHASE94_VALIDATOR_METRICS_PORT": os.getenv("PHASE94_VALIDATOR_METRICS_PORT"),
        },
        "required_env_presence": context["env_presence"],
        "wallets": {
            "miner": _wallet_payload("MINER"),
            "validator": _wallet_payload("VALIDATOR"),
        },
        "bt_wallet_path": {
            "present": context["env_presence"]["BT_WALLET_PATH"],
            "redacted_basename": _redacted_wallet_path(),
        },
        "btcli": _btcli_info(),
        "approval_record_path": str(governance_dir / "operator-approval.json"),
        "redaction_rules": REDACTION_RULES,
        "ready_for_live_action": False,
        "notes": [
            "Fill hotkey_ss58 and funded_wallet_confirmed after checking real wallets.",
            (
                "Keep operator-approval.json approved=false until a human "
                "approves the exact candidate."
            ),
        ],
    }
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
        "notes": [
            "Human must inspect deployment-candidate.json and no-start-checklist.json first.",
            "Set approved=true only after confirming the exact candidate hash is still current.",
        ],
    }
    atomic_write_json(governance_dir / "no-start-checklist.json", checklist)
    atomic_write_json(governance_dir / "operator-approval.json", approval)

    print("Generated Phase 94 governance artifacts:")
    if loaded_env_file is not None:
        print(f"Loaded env file: {loaded_env_file}")
    for filename in (
        "deployment-candidate.json",
        "no-start-checklist.json",
        "operator-approval.json",
    ):
        print(f"  - {governance_dir / filename}")
    if candidate["missing_required_env_vars"]:
        print("Missing required env vars:")
        for env_var in candidate["missing_required_env_vars"]:
            print(f"  - {env_var}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
