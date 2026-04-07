from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "CHANGELOG.md").exists(),
    reason="CHANGELOG.md not in public mirror",
)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _require(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"missing {label}: {pattern}"
    return match


def test_release_versions_are_synced_to_v10() -> None:
    pyproject = _read("pyproject.toml")
    citation = _read("CITATION.cff")
    zenodo = json.loads(_read(".zenodo.json"))
    manuscript = _read("docs/manuscript.md")

    pyproject_version = _require(
        r'^version = "(?P<version>[^"]+)"$', pyproject, "pyproject version"
    ).group("version")
    citation_version = _require(
        r"^version: (?P<version>[0-9.]+)$", citation, "citation version"
    ).group("version")
    manuscript_version = _require(
        r"\*\*Version:\*\* (?P<version>[0-9.]+) \| \*\*Date:\*\* 2026-04-05",
        manuscript,
        "manuscript version line",
    ).group("version")

    assert pyproject_version == "10.0.0"
    assert citation_version == "10.0.0"
    assert zenodo["version"] == "10.0.0"
    assert manuscript_version == "10.0.0"


def test_changelog_contains_v10_release_entry() -> None:
    changelog = _read("CHANGELOG.md")

    assert "## [10.0.0] - 2026-04-05" in changelog
    assert "Phase 84 swarm NO-GO" in changelog
    assert "weighted ensemble + operator_multiband" in changelog
    assert "simulation-only deployment evidence" in changelog


def test_v10_release_notes_cover_required_topics() -> None:
    release_notes = _read("docs/release/v10.0-release-notes.md")

    for section in (
        "## What Shipped",
        "## Evidence Chain",
        "## Release Verification",
        "## Known Human-Only Follow-Ups",
    ):
        assert section in release_notes

    assert "Phase 84 NO-GO" in release_notes
    assert "operator_multiband" in release_notes
    assert "simulation / localhost-backed" in release_notes


def test_zenodo_handoff_is_explicitly_pending() -> None:
    handoff = _read("docs/release/v10.0-zenodo-handoff.md")

    assert "Zenodo status: pending human action" in handoff
    assert "Do not mint or record a fake DOI" in handoff
    assert "CITATION.cff" in handoff
    assert ".zenodo.json" in handoff
