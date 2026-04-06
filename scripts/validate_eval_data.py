#!/usr/bin/env python3
"""Cross-validation script for Antigence Subnet evaluation datasets.

Checks schema integrity, class balance, duplicate IDs, required fields,
and cross-references between samples.json and manifest.json for each domain.

Usage:
    python scripts/validate_eval_data.py --domain all
    python scripts/validate_eval_data.py --domain hallucination --data-dir data/evaluation
"""

import argparse
import json
import sys
from pathlib import Path

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_SOURCES = {"synthetic", "template", "llm"}
VALID_LABELS = {"normal", "anomalous"}
REQUIRED_SAMPLE_FIELDS = {"id", "prompt", "output", "domain", "metadata"}
REQUIRED_MANIFEST_FIELDS = {"ground_truth_label", "ground_truth_type", "is_honeypot"}

DOMAINS = ["hallucination", "code_security", "reasoning", "bio"]


def validate_schema(samples: list[dict], manifest: dict, domain: str) -> list[str]:
    """Validate schema of samples and manifest entries.

    Checks:
    - Every sample has required fields (id, prompt, output, domain, metadata)
    - code_security samples also have "code" field
    - Every sample ID has a manifest entry
    - Every manifest entry has required fields
    - metadata has difficulty and source with valid values
    - No orphan manifest entries (every manifest key has a sample)

    Args:
        samples: List of sample dicts.
        manifest: Dict mapping sample_id to ground truth entry.
        domain: Domain name for domain-specific checks.

    Returns:
        List of error strings (empty if valid).
    """
    errors = []
    sample_ids = set()

    for i, sample in enumerate(samples):
        # Check required fields
        for field in REQUIRED_SAMPLE_FIELDS:
            if field not in sample:
                errors.append(f"Sample {i}: missing required field '{field}'")

        # code_security requires "code" field
        if domain == "code_security" and "code" not in sample:
            errors.append(f"Sample {i}: code_security sample missing 'code' field")

        sid = sample.get("id")
        if sid is None:
            continue

        sample_ids.add(sid)

        # Check manifest entry exists
        if sid not in manifest:
            errors.append(f"Sample '{sid}': not found in manifest")
        else:
            entry = manifest[sid]
            # Check manifest fields
            for field in REQUIRED_MANIFEST_FIELDS:
                if field not in entry:
                    errors.append(f"Manifest '{sid}': missing field '{field}'")

            if "ground_truth_label" in entry:
                if entry["ground_truth_label"] not in VALID_LABELS:
                    errors.append(
                        f"Manifest '{sid}': invalid ground_truth_label "
                        f"'{entry['ground_truth_label']}'"
                    )

            if "is_honeypot" in entry:
                if not isinstance(entry["is_honeypot"], bool):
                    errors.append(
                        f"Manifest '{sid}': is_honeypot must be bool, "
                        f"got {type(entry['is_honeypot']).__name__}"
                    )

        # Check metadata
        meta = sample.get("metadata", {})
        if "difficulty" not in meta:
            errors.append(f"Sample '{sid}': metadata missing 'difficulty'")
        elif meta["difficulty"] not in VALID_DIFFICULTIES:
            errors.append(
                f"Sample '{sid}': invalid difficulty '{meta['difficulty']}'"
            )

        if "source" not in meta:
            errors.append(f"Sample '{sid}': metadata missing 'source'")
        elif meta["source"] not in VALID_SOURCES:
            errors.append(
                f"Sample '{sid}': invalid source '{meta['source']}'"
            )

    # Check for orphan manifest entries
    for mid in manifest:
        if mid not in sample_ids:
            errors.append(f"Orphan manifest entry '{mid}': no corresponding sample")

    return errors


def validate_balance(manifest: dict) -> list[str]:
    """Check that anomalous ratio is between 40% and 60%.

    Args:
        manifest: Dict mapping sample_id to ground truth entry.

    Returns:
        List of error strings (empty if balanced).
    """
    errors = []
    if not manifest:
        errors.append("Empty manifest -- cannot check balance")
        return errors

    total = len(manifest)
    anomalous = sum(
        1 for entry in manifest.values()
        if entry.get("ground_truth_label") == "anomalous"
    )
    ratio = anomalous / total

    if ratio < 0.40 or ratio > 0.60:
        errors.append(
            f"Class balance error: {anomalous}/{total} anomalous "
            f"({ratio:.1%}), expected 40-60%"
        )
    elif ratio < 0.45 or ratio > 0.55:
        # Warning but not error
        pass  # Could log warning but not treated as error

    return errors


def validate_duplicates(samples: list[dict]) -> list[str]:
    """Check for duplicate sample IDs.

    Args:
        samples: List of sample dicts.

    Returns:
        List of error strings (empty if no duplicates).
    """
    errors = []
    seen = {}
    for i, sample in enumerate(samples):
        sid = sample.get("id")
        if sid is None:
            continue
        if sid in seen:
            errors.append(
                f"Duplicate ID '{sid}': found at indices {seen[sid]} and {i}"
            )
        else:
            seen[sid] = i

    return errors


def validate_domain(
    data_dir: Path, domain: str
) -> tuple[list[str], dict]:
    """Run all validations on a single domain.

    Args:
        data_dir: Root evaluation data directory.
        domain: Domain name.

    Returns:
        Tuple of (errors_list, stats_dict).
    """
    errors = []
    domain_dir = data_dir / domain

    samples_path = domain_dir / "samples.json"
    manifest_path = domain_dir / "manifest.json"

    if not samples_path.exists():
        errors.append(f"Missing {samples_path}")
        return errors, {}

    if not manifest_path.exists():
        errors.append(f"Missing {manifest_path}")
        return errors, {}

    with open(samples_path) as f:
        data = json.load(f)
    samples = data.get("samples", [])

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Run all checks
    errors.extend(validate_schema(samples, manifest, domain))
    errors.extend(validate_balance(manifest))
    errors.extend(validate_duplicates(samples))

    # Compute stats
    total = len(samples)
    normal = sum(
        1 for entry in manifest.values()
        if entry.get("ground_truth_label") == "normal"
    )
    anomalous = sum(
        1 for entry in manifest.values()
        if entry.get("ground_truth_label") == "anomalous"
    )
    honeypots = sum(
        1 for entry in manifest.values()
        if entry.get("is_honeypot", False)
    )
    ratio = anomalous / total if total > 0 else 0

    stats = {
        "domain": domain,
        "sample_count": total,
        "normal_count": normal,
        "anomalous_count": anomalous,
        "honeypot_count": honeypots,
        "anomalous_ratio": f"{ratio:.1%}",
        "errors": len(errors),
    }

    return errors, stats


def print_summary_table(all_stats: list[dict]) -> None:
    """Print a summary table of validation results."""
    header = f"{'Domain':<20} {'Samples':>8} {'Normal':>8} {'Anomalous':>10} {'Honeypots':>10} {'Anom.Ratio':>11} {'Errors':>7}"
    print("\n" + "=" * len(header))
    print("Evaluation Data Validation Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for stats in all_stats:
        print(
            f"{stats['domain']:<20} "
            f"{stats['sample_count']:>8} "
            f"{stats['normal_count']:>8} "
            f"{stats['anomalous_count']:>10} "
            f"{stats['honeypot_count']:>10} "
            f"{stats['anomalous_ratio']:>11} "
            f"{stats['errors']:>7}"
        )

    print("-" * len(header))
    total_samples = sum(s["sample_count"] for s in all_stats)
    total_errors = sum(s["errors"] for s in all_stats)
    print(f"{'TOTAL':<20} {total_samples:>8} {'':>8} {'':>10} {'':>10} {'':>11} {total_errors:>7}")
    print("=" * len(header))

    if total_errors == 0:
        print("\nResult: ALL VALID")
    else:
        print(f"\nResult: {total_errors} ERROR(S) FOUND")


def main() -> None:
    """CLI entry point for evaluation data validation."""
    parser = argparse.ArgumentParser(
        description="Validate Antigence Subnet evaluation datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python validate_eval_data.py --domain all\n"
            "  python validate_eval_data.py --domain hallucination\n"
            "  python validate_eval_data.py --domain bio --data-dir /path/to/data\n"
        ),
    )
    parser.add_argument(
        "--domain",
        default="all",
        choices=["hallucination", "code_security", "reasoning", "bio", "all"],
        help="Domain to validate (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Data directory (default: data/evaluation).",
    )

    args = parser.parse_args()

    domains = DOMAINS if args.domain == "all" else [args.domain]

    all_errors = []
    all_stats = []

    for domain in domains:
        errors, stats = validate_domain(args.data_dir, domain)
        all_errors.extend(errors)
        if stats:
            all_stats.append(stats)

        if errors:
            print(f"\n[{domain}] {len(errors)} error(s):")
            for e in errors:
                print(f"  - {e}")

    if all_stats:
        print_summary_table(all_stats)

    sys.exit(1 if all_errors else 0)


if __name__ == "__main__":
    main()
