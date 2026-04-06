from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE95_DIR = (
    ROOT
    / ".planning"
    / "phases"
    / "95-simulation-performance-deployment-deferral-report"
)
REQUIREMENTS_PATH = ROOT / ".planning" / "REQUIREMENTS.md"
MATRIX_PATH = PHASE95_DIR / "95-evidence-matrix.md"
REPORT_PATH = PHASE95_DIR / "95-report.md"
APPENDIX_PATH = PHASE95_DIR / "95-future-live-appendix.md"

REPAIRED_DEPLOY_02 = (
    "DEPLOY-02: Simulation performance and deployment-deferral report "
    "(simulation / localhost-backed behavior), "
    "simulation-vs-deferred-real-network gap analysis, and issue list with "
    "severity ratings plus explicit public-safe deferral rationale"
)

CLAIM_BUCKETS = (
    "observed under simulation / localhost-backed resources",
    "inferred but not proven for real-network behavior",
    "intentionally unknown because live deployment was deferred",
)


def _read(path: Path) -> str:
    assert path.exists(), f"Missing required Phase 95 artifact: {path}"
    return path.read_text(encoding="utf-8")


def test_requirements_repairs_deploy_02_to_simulation_only_contract():
    text = _read(REQUIREMENTS_PATH)
    deploy_line = next(
        line for line in text.splitlines() if "DEPLOY-02" in line
    ).replace("**", "")

    assert REPAIRED_DEPLOY_02 in deploy_line
    assert (
        "DEPLOY-02: Testnet performance report (real network behavior), "
        "local vs testnet gap analysis, issue list with severity ratings"
        not in text
    )


def test_evidence_matrix_fixes_runtime_stack_and_claim_buckets():
    text = _read(MATRIX_PATH)

    assert "weighted ensemble + operator_multiband" in text
    assert "92-benchmark-report.md" in text
    assert "93-ADR.md" in text
    assert "94-VERIFICATION.md" in text

    for bucket in ("observed", "inferred", "unknown"):
        assert f"| {bucket} |" in text

    table_rows = [
        line
        for line in text.splitlines()
        if line.startswith("|")
        and "---" not in line
        and "Bucket" not in line
    ]
    assert table_rows, "Expected at least one evidence-matrix row"

    for row in table_rows:
        bucket_count = sum(f"| {bucket} |" in row for bucket in ("observed", "inferred", "unknown"))
        assert bucket_count == 1, f"Expected exactly one claim bucket in row: {row}"


def test_report_contains_required_sections_without_live_validation_overclaim():
    text = _read(REPORT_PATH)
    lowered = text.lower()

    assert "# Phase 95" in text
    assert "## Executive Summary" in text
    for section in CLAIM_BUCKETS:
        assert f"## {section}" in text
    assert "## Severity-Ranked Issue Ledger" in text
    assert "## Public-Safe Deferral Rationale" in text
    assert "92-benchmark-report.md" in text
    assert "93-ADR.md" in text
    assert "94-VERIFICATION.md" in text

    for forbidden in ("testnet validated", "real-world validated"):
        assert forbidden not in lowered


def test_future_live_appendix_lists_prerequisites_without_blocking_current_milestone():
    text = _read(APPENDIX_PATH)
    lowered = text.lower()

    assert "Future Live" in text
    assert "funded wallet" in lowered
    assert "outside git" in lowered
    assert "approved netuid and endpoint" in lowered
    assert "immutable candidate manifest" in lowered
    assert "pre-action approvals" in lowered
    assert "registration proof" in lowered
    assert "24h/72h artifacts" in lowered
    assert "non-public-safe storage boundaries" in lowered
    assert "not part of the current milestone blocker" in lowered
