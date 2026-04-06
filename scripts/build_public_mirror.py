#!/usr/bin/env python3
"""Build a bounded public mirror tree from the Phase 99 manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MANIFEST = Path(
    ".planning/phases/99-public-mirror-curation-history-sanitization/99-public-mirror-manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("dist/public-mirror")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _git_tracked_files(repo_root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"],
        cwd=repo_root,
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _matches_prefix(path: str, rule: str) -> bool:
    if rule.endswith("/"):
        return path == rule[:-1] or path.startswith(rule)
    return path == rule or path.startswith(f"{rule}/")


def _longest_match(path: str, rules: list[str]) -> int:
    matches = [len(rule.rstrip("/")) for rule in rules if _matches_prefix(path, rule)]
    return max(matches, default=-1)


def _classify_path(path: str, manifest: dict) -> tuple[bool, str]:
    include_len = _longest_match(path, manifest["include_paths"])
    if include_len < 0:
        return False, "not-included"

    exclude_rules = manifest["exclude_paths"] + manifest["mirror_only_paths"]
    exclude_len = _longest_match(path, exclude_rules)
    if exclude_len >= include_len:
        return False, "excluded"
    return True, "included"


def _copy_file(repo_root: Path, relative_path: str, output_dir: Path) -> None:
    source = repo_root / relative_path
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_public_mirror(manifest_path: Path, output_dir: Path, clean: bool) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    manifest = _load_manifest(manifest_path)

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracked_files = _git_tracked_files(repo_root)
    included_files: list[str] = []
    excluded_files: list[dict[str, str]] = []

    for relative_path in tracked_files:
        include_file, reason = _classify_path(relative_path, manifest)
        if include_file:
            _copy_file(repo_root, relative_path, output_dir)
            included_files.append(relative_path)
        else:
            excluded_files.append({"path": relative_path, "reason": reason})

    summary = {
        "generated_at_utc": _iso_now(),
        "manifest": str(manifest_path),
        "generated_tree": str(output_dir),
        "included_count": len(included_files),
        "excluded_count": len(excluded_files),
        "included_files": included_files,
        "excluded_files": excluded_files,
        "manifest_rules": {
            "include_paths": manifest["include_paths"],
            "exclude_paths": manifest["exclude_paths"],
            "mirror_only_paths": manifest["mirror_only_paths"],
            "rewrite_later_paths": manifest["rewrite_later_paths"],
            "approval_later_paths": manifest["approval_later_paths"],
        },
    }
    _write_summary(output_dir / ".build-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the Phase 99 public mirror manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for the generated public mirror tree.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing output directory before rebuilding it.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    build_public_mirror(manifest_path=manifest_path, output_dir=output_dir, clean=args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
