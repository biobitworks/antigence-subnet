from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_PHASE97_DIR = REPO_ROOT / ".planning/phases/97-publication-submission"
pytestmark = pytest.mark.skipif(
    not _PHASE97_DIR.exists(), reason="phase 97 artifacts archived"
)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_manuscript_mentions_final_v10_story() -> None:
    manuscript = _read("docs/manuscript.md")

    assert "**Version:** 10.0.0 | **Date:** 2026-04-05" in manuscript
    assert "Phase 84" in manuscript
    assert "NO-GO" in manuscript
    assert "operator_multiband" in manuscript
    assert "simulation / localhost-backed" in manuscript
    assert "Computers & Security" in manuscript


def test_publication_strategy_sets_active_no_go_target() -> None:
    strategy = _read("docs/publication-strategy.md")

    assert "## Active Submission Path" in strategy
    assert "Computers & Security" in strategy
    assert "NO-GO branch" in strategy


def test_submission_package_files_exist_with_honest_status() -> None:
    checklist = _read("docs/submission/computers-security-package/CHECKLIST.md")
    cover_letter = _read("docs/submission/computers-security-package/COVER_LETTER.md")
    supplementary = _read("docs/submission/computers-security-package/SUPPLEMENTARY.md")

    assert "Submission status: pending human portal upload" in checklist
    assert "Computers & Security" in checklist
    assert "pending human portal submission" in cover_letter
    assert "Phase 92" in supplementary
    assert "Phase 95" in supplementary


def test_supplementary_archive_and_traceability_report_exist() -> None:
    archive_path = REPO_ROOT / "docs/submission/computers-security-package/supplementary-data.zip"
    traceability_report = _read(
        ".planning/phases/97-publication-submission/97-traceability-report.md"
    )

    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    expected = {
        "docs/submission/computers-security-package/SUPPLEMENTARY.md",
        "data/benchmarks/phase92-continuous-benchmark.json",
        ".planning/phases/92-continuous-antibody-benchmark-canon/92-benchmark-report.md",
        ".planning/phases/95-simulation-performance-deployment-deferral-report/95-report.md",
    }
    assert expected.issubset(names)
    assert "Overwatch Traceability Validation" in traceability_report
