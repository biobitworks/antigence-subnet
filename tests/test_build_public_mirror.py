"""Tests for the public mirror manifest policy."""

from pathlib import Path

from scripts.build_public_mirror import _classify_path, _load_manifest


def test_manifest_excludes_internal_release_surfaces():
    manifest = _load_manifest(
        Path(
            ".planning/phases/99-public-mirror-curation-history-sanitization/99-public-mirror-manifest.json"
        )
    )

    include_release_doc, release_reason = _classify_path("PUBLIC_RELEASE.md", manifest)
    include_pitch_doc, pitch_reason = _classify_path(
        "docs/pitch/TAO5-APPLICATION.md", manifest
    )
    include_readme, readme_reason = _classify_path("README.md", manifest)
    include_phase94_template, template_reason = _classify_path(
        ".planning/phases/94-testnet-deployment-of-winning-ensemble-policy/artifacts/governance/deployment-candidate.json",
        manifest,
    )

    assert not include_release_doc
    assert release_reason in {"excluded", "not-included"}
    assert not include_pitch_doc
    assert pitch_reason == "excluded"
    assert include_readme
    assert readme_reason == "included"
    assert include_phase94_template
    assert template_reason == "included"
