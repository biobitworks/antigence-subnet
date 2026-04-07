"""Tests for the public mirror manifest policy."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts.build_public_mirror import _classify_path, _load_manifest

_DEFAULT_MANIFEST_PATH = Path(
    ".planning/phases/99-public-mirror-curation-history-sanitization/99-public-mirror-manifest.json"
)
_BUILD_SUMMARY_PATH = Path("dist/public-mirror/.build-summary.json")


def _resolve_manifest_path() -> Path | None:
    """Return the manifest path, falling back to the build summary when the
    default path no longer exists on disk (e.g. after Commit 2 deletes it).
    Returns None if the manifest cannot be found (e.g. in public repo CI)."""
    if _DEFAULT_MANIFEST_PATH.exists():
        return _DEFAULT_MANIFEST_PATH
    # The default path was committed-as-deleted. Restore it from git history.
    # Try extracting from build summary first (most reliable after a rebuild).
    if _BUILD_SUMMARY_PATH.exists():
        summary = json.loads(_BUILD_SUMMARY_PATH.read_text(encoding="utf-8"))
        recorded = Path(summary.get("manifest", ""))
        if recorded.exists():
            return recorded
    # Last resort: restore from git history to a temp path.
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "99-public-mirror-manifest.json"
    result = subprocess.run(
        [
            "git",
            "show",
            f"HEAD~1:{_DEFAULT_MANIFEST_PATH}",
        ],
        stdout=tmp.open("wb"),
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return tmp


def test_manifest_excludes_internal_release_surfaces():
    manifest_path = _resolve_manifest_path()
    if manifest_path is None:
        pytest.skip("manifest not available (public repo CI)")
    manifest = _load_manifest(manifest_path)

    include_release_doc, release_reason = _classify_path("PUBLIC_RELEASE.md", manifest)
    include_pitch_doc, pitch_reason = _classify_path(
        "docs/pitch/TAO5-APPLICATION.md", manifest
    )
    include_readme, readme_reason = _classify_path("README.md", manifest)
    include_phase94_template, template_reason = _classify_path(
        ".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/governance/deployment-candidate.json",
        manifest,
    )

    assert include_release_doc
    assert release_reason == "included"
    assert not include_pitch_doc
    assert pitch_reason == "excluded"
    assert include_readme
    assert readme_reason == "included"
    assert include_phase94_template
    assert template_reason == "included"
