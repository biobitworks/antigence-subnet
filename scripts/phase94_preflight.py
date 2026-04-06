#!/usr/bin/env python3
"""Validate Phase 94 governance artifacts before approval or live action."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_ARTIFACT_ROOT = Path(
    ".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts"
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_truthy(value: object) -> bool:
    return value is True


def validate_phase94_preflight(
    artifact_root: Path,
    *,
    require_approval: bool,
) -> tuple[bool, list[str]]:
    governance_dir = artifact_root / "governance"
    candidate_path = governance_dir / "deployment-candidate.json"
    checklist_path = governance_dir / "no-start-checklist.json"
    approval_path = governance_dir / "operator-approval.json"

    errors: list[str] = []
    for path in (candidate_path, checklist_path, approval_path):
        if not path.exists():
            errors.append(f"missing artifact: {path}")
    if errors:
        return False, errors

    candidate = _load_json(candidate_path)
    checklist = _load_json(checklist_path)
    approval = _load_json(approval_path)
    candidate_sha256 = _sha256_file(candidate_path)

    if candidate.get("classification") != "internal-only":
        errors.append("deployment-candidate.json must be classification=internal-only")
    if candidate.get("public_release") is not False:
        errors.append("deployment-candidate.json must set public_release=false")
    if candidate.get("execution_mode") != "same-host-private":
        errors.append("deployment-candidate.json must set execution_mode=same-host-private")
    if candidate.get("missing_required_env_vars"):
        errors.append(
            "deployment-candidate.json still lists missing_required_env_vars"
        )
    if candidate.get("ready_for_approval") is not True:
        errors.append("deployment-candidate.json must set ready_for_approval=true")

    if checklist.get("execution_mode") != "same-host-private":
        errors.append("no-start-checklist.json must set execution_mode=same-host-private")
    if checklist.get("deployment_candidate_path") != str(candidate_path):
        errors.append("no-start-checklist.json points at the wrong candidate path")
    if checklist.get("deployment_candidate_sha256") != candidate_sha256:
        errors.append("no-start-checklist.json has a stale deployment candidate hash")

    required_env_presence = checklist.get("required_env_presence", {})
    missing_env_flags = sorted(
        name for name, present in required_env_presence.items() if not _is_truthy(present)
    )
    if missing_env_flags:
        errors.append(
            "no-start-checklist.json still has missing env presence flags: "
            + ", ".join(missing_env_flags)
        )

    approved_metrics_ports = checklist.get("approved_metrics_ports", {})
    if not approved_metrics_ports.get("PHASE94_MINER_METRICS_PORT"):
        errors.append("no-start-checklist.json is missing PHASE94_MINER_METRICS_PORT")
    if not approved_metrics_ports.get("PHASE94_VALIDATOR_METRICS_PORT"):
        errors.append("no-start-checklist.json is missing PHASE94_VALIDATOR_METRICS_PORT")

    wallet_path = checklist.get("bt_wallet_path", {})
    if wallet_path.get("present") is not True:
        errors.append("no-start-checklist.json must record BT_WALLET_PATH as present")
    if not wallet_path.get("redacted_basename"):
        errors.append("no-start-checklist.json must store a redacted wallet basename")

    btcli = checklist.get("btcli", {})
    if btcli.get("available") is not True:
        errors.append("no-start-checklist.json must confirm btcli availability")

    wallets = checklist.get("wallets", {})
    for role in ("miner", "validator"):
        wallet = wallets.get(role, {})
        if not wallet.get("wallet_name"):
            errors.append(f"no-start-checklist.json is missing {role} wallet_name")
        if not wallet.get("wallet_hotkey"):
            errors.append(f"no-start-checklist.json is missing {role} wallet_hotkey")
        if not wallet.get("hotkey_ss58"):
            errors.append(f"no-start-checklist.json is missing {role} hotkey_ss58")
        if wallet.get("funded_wallet_confirmed") is not True:
            errors.append(
                f"no-start-checklist.json must confirm funded_wallet_confirmed for {role}"
            )

    if require_approval:
        if approval.get("scope") != "94-01-live-gate":
            errors.append("operator-approval.json must scope approval to 94-01-live-gate")
        if approval.get("deployment_candidate_sha256") != candidate_sha256:
            errors.append("operator-approval.json has a stale deployment candidate hash")
        if approval.get("approved") is not True:
            errors.append("operator-approval.json must set approved=true")
        if approval.get("approval_status") not in {"approved", "human-approved"}:
            errors.append(
                "operator-approval.json must set approval_status to approved or human-approved"
            )
        if not approval.get("approver"):
            errors.append("operator-approval.json must record approver")
        if not approval.get("approved_at_utc"):
            errors.append("operator-approval.json must record approved_at_utc")
        if approval.get("ready_for_live_action") is not True:
            errors.append("operator-approval.json must set ready_for_live_action=true")

    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Artifact root containing the Phase 94 governance artifacts.",
    )
    parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Fail unless operator-approval.json records an approved live gate.",
    )
    args = parser.parse_args()

    ok, errors = validate_phase94_preflight(
        Path(args.artifact_root),
        require_approval=args.require_approval,
    )
    if ok:
        mode = "ready-for-live-action" if args.require_approval else "ready-for-approval"
        print(f"Phase 94 preflight PASS ({mode})")
        return 0

    mode = "ready-for-live-action" if args.require_approval else "ready-for-approval"
    print(f"Phase 94 preflight FAIL ({mode})")
    for error in errors:
        print(f" - {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
