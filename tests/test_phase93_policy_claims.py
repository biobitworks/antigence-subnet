"""Contract tests for Phase 93 claim-bundle and writeback artifacts."""

from __future__ import annotations

import json
from pathlib import Path


CLAIMS_PATH = Path("data/overwatch/phase93-policy-claims.json")
WRITEBACK_PATH = Path("data/overwatch/phase93-policy-writeback-report.json")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase93_claim_bundle_matches_local_first_overwatch_contract():
    payload = _load_json(CLAIMS_PATH)

    assert payload["run_id"]
    experiment = payload["experiment"]
    assert experiment["_key"] == "antigence_phase93_policy"
    assert experiment["exp_id"] == "EXP-093"
    assert experiment["project"] == "antigence-bittensor"
    assert experiment["status"] == "completed"
    assert ".planning/phases/93-decision-policy-adr-operator-migration/93-ADR.md" == experiment["note_path"]
    assert "data/benchmarks/phase92-continuous-benchmark.json" in experiment["result_paths"]
    assert ".planning/phases/93-decision-policy-adr-operator-migration/93-ADR.md" in experiment["result_paths"]
    assert ".planning/phases/93-decision-policy-adr-operator-migration/93-migration-guide.md" in experiment["result_paths"]
    assert "data/overwatch/phase93-policy-claims.json" in experiment["result_paths"]

    claims = {claim["_key"]: claim for claim in payload["claims"]}
    assert set(claims) == {
        "phase93_policy_parity",
        "phase93_operator_multiband_default",
        "phase93_detector_contract_unchanged",
    }
    assert claims["phase93_policy_parity"]["classification"] == "MEASURED"
    assert claims["phase93_operator_multiband_default"]["classification"] == "INFERRED"
    assert claims["phase93_detector_contract_unchanged"]["classification"] == "MEASURED"
    assert "operator_multiband" in claims["phase93_operator_multiband_default"]["text"]

    for claim in claims.values():
        derived_from = claim["derived_from"]
        evidence_refs = derived_from["evidence_refs"]
        assert any(
            ref == "local:data/benchmarks/phase92-continuous-benchmark.json"
            for ref in evidence_refs
        )
        assert any(
            ref == "local:.planning/phases/93-decision-policy-adr-operator-migration/93-ADR.md"
            for ref in evidence_refs
        )


def test_phase93_writeback_report_allows_deferred_or_successful_promotion():
    payload = _load_json(WRITEBACK_PATH)

    assert payload["run_id"]
    assert "created_at" in payload
    assert "errors" in payload
    assert "counts" in payload

    if payload.get("skipped") is True:
        assert payload["reason"]
        assert payload["replay_command"]
        assert "overwatch_" in payload["reason"]
        assert "phase93-policy-claims.json" in payload["replay_command"]
    else:
        assert payload["skipped"] is False
        assert isinstance(payload["counts"], dict)
        assert payload["errors"] == []
