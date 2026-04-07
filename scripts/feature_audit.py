"""Feature distribution audit across all evaluation domains.

Characterizes all 10 dendritic features across all 4 evaluation domains,
producing machine-readable JSON reports and a human-readable markdown summary.

Phase 32 NK Cell needs per-feature mean/std for z-score thresholds.
Phase 33 DCA needs binary/continuous classification and correlation data
for signal mapping.

Usage:
    python3 scripts/feature_audit.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so we can import antigence_subnet
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from antigence_subnet.miner.detectors.dendritic_features import (  # noqa: E402
    DendriticFeatureExtractor,
)

DOMAINS = ["hallucination", "code_security", "reasoning", "bio"]


def audit_domain(
    domain: str,
    data_dir: str = "data/evaluation",
) -> dict:
    """Audit dendritic feature distributions for a single evaluation domain.

    Args:
        domain: Name of the evaluation domain (e.g. "hallucination").
        data_dir: Base directory containing domain subdirectories with
            ``samples.json`` files.

    Returns:
        Dictionary with keys: domain, n_samples, feature_stats,
        correlation_matrix, high_correlations, feature_names, metadata.
    """
    # Resolve data_dir relative to project root if not absolute
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = _project_root / data_path

    samples_path = data_path / domain / "samples.json"
    with open(samples_path) as f:
        data = json.load(f)

    samples = data["samples"]

    # Extract text from each sample: prefer 'output', fall back to 'code'
    texts = []
    for sample in samples:
        text = sample.get("output", "")
        if not text and "code" in sample:
            text = sample["code"]
        if not text:
            text = sample.get("prompt", "")
        texts.append(text)

    # Extract features using DendriticFeatureExtractor
    extractor = DendriticFeatureExtractor()
    features = extractor.extract_batch(texts)  # shape (n_samples, 10)

    n_samples = len(texts)
    feature_names = DendriticFeatureExtractor.FEATURE_NAMES

    # Compute per-feature statistics
    feature_stats = {}
    for idx, name in enumerate(feature_names):
        col = features[:, idx]
        unique_vals = np.unique(col).tolist()
        unique_count = len(unique_vals)

        # Binary: unique values are subset of {0.0, 1.0}
        is_binary = unique_count <= 2 and all(v in (0.0, 1.0) for v in unique_vals)
        # Constant: only one unique value
        is_constant = unique_count == 1

        feature_stats[name] = {
            "index": idx,
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "percentiles": {
                "p5": float(np.percentile(col, 5)),
                "p25": float(np.percentile(col, 25)),
                "p50": float(np.percentile(col, 50)),
                "p75": float(np.percentile(col, 75)),
                "p95": float(np.percentile(col, 95)),
            },
            "unique_count": unique_count,
            "unique_values": unique_vals,
            "is_binary": is_binary,
            "is_constant": is_constant,
        }

    # Compute correlation matrix
    corr_raw = np.corrcoef(features.T)  # 10x10

    # Replace NaN with None (NaN arises from constant features)
    correlation_matrix = []
    for row in corr_raw:
        cleaned_row = []
        for v in row:
            if v != v or np.isnan(v):  # NaN check (NaN != NaN)
                cleaned_row.append(None)
            else:
                cleaned_row.append(round(float(v), 6))
        correlation_matrix.append(cleaned_row)

    # Flag high correlations (|r| > 0.7, upper triangle only)
    high_correlations = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            val = correlation_matrix[i][j]
            if val is not None and abs(val) > 0.7:
                high_correlations.append(
                    {
                        "features": [feature_names[i], feature_names[j]],
                        "r": val,
                        "note": _correlation_note(feature_names[i], feature_names[j], val),
                    }
                )

    return {
        "domain": domain,
        "n_samples": n_samples,
        "feature_names": list(feature_names),
        "feature_stats": feature_stats,
        "correlation_matrix": correlation_matrix,
        "high_correlations": high_correlations,
        "metadata": {
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "extractor_class": "DendriticFeatureExtractor",
            "data_dir": str(data_path),
        },
    }


def _correlation_note(feat_a: str, feat_b: str, r: float) -> str:
    """Generate an explanatory note for a high correlation pair."""
    pair = frozenset({feat_a, feat_b})

    if pair == frozenset({"pamp_score", "danger_signal"}):
        return (
            "Algebraic dependency: both derived from same danger pattern matches "
            f"(pamp=matches/3, danger=matches/5). r={r:.4f}. "
            "DCA must assign to same signal category to avoid double-counting."
        )

    if pair == frozenset({"claim_density", "hedging_ratio"}):
        return (
            "claim_density includes +0.1 for hedging_ratio presence. "
            f"r={r:.4f}. Partial algebraic dependency."
        )

    if pair == frozenset({"claim_density", "pamp_score"}):
        return (
            "claim_density includes -0.3 when pamp_score>0.5. "
            f"r={r:.4f}. Partial algebraic dependency."
        )

    if pair == frozenset({"claim_density", "citation_count"}):
        return (
            "claim_density includes +0.2 when citation_count present. "
            f"r={r:.4f}. Partial algebraic dependency."
        )

    if pair == frozenset({"claim_density", "exaggeration"}):
        return (
            "claim_density includes -0.2 when exaggeration>0.5. "
            f"r={r:.4f}. Partial algebraic dependency."
        )

    strength = "strong" if abs(r) > 0.9 else "moderate"
    direction = "positive" if r > 0 else "negative"
    return f"{strength.capitalize()} {direction} correlation (r={r:.4f})."


def generate_summary_md(all_results: list[dict]) -> str:
    """Generate a consolidated markdown summary of all audit results.

    Args:
        all_results: List of per-domain audit result dicts.

    Returns:
        Markdown string with feature tables, classification, and warnings.
    """
    lines = [
        "# Feature Audit Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "This report characterizes all 10 dendritic features across all evaluation "
        "domains. It informs Phase 32 NK Cell z-score thresholds and Phase 33 DCA "
        "signal mapping.",
        "",
    ]

    feature_names = DendriticFeatureExtractor.FEATURE_NAMES

    # ---- Per-domain feature tables ----
    lines.append("## Per-Domain Feature Statistics")
    lines.append("")

    for result in all_results:
        domain = result["domain"]
        n = result["n_samples"]
        lines.append(f"### {domain} (n={n})")
        lines.append("")
        lines.append("| Feature | Type | Mean | Std | Min | Max |")
        lines.append("|---------|------|------|-----|-----|-----|")

        for name in feature_names:
            stats = result["feature_stats"][name]
            ftype = (
                "constant"
                if stats["is_constant"]
                else ("binary" if stats["is_binary"] else "continuous")
            )
            lines.append(
                f"| {name} | {ftype} | {stats['mean']:.4f} | "
                f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
            )
        lines.append("")

    # ---- Binary vs Continuous classification table ----
    lines.append("## Feature Classification (Binary vs Continuous)")
    lines.append("")
    lines.append("| Feature | " + " | ".join(r["domain"] for r in all_results) + " |")
    lines.append("|---------|" + "|".join(["------"] * len(all_results)) + "|")

    for name in feature_names:
        row = f"| {name} |"
        for result in all_results:
            stats = result["feature_stats"][name]
            if stats["is_constant"]:
                row += " constant |"
            elif stats["is_binary"]:
                row += " binary |"
            else:
                row += " continuous |"
        lines.append(row)
    lines.append("")

    # ---- High Correlations ----
    lines.append("## High Correlations (|r| > 0.7)")
    lines.append("")

    any_high = False
    for result in all_results:
        if result["high_correlations"]:
            any_high = True
            lines.append(f"### {result['domain']}")
            lines.append("")
            for pair in result["high_correlations"]:
                features = " / ".join(pair["features"])
                lines.append(f"- **{features}**: r = {pair['r']:.4f}")
                lines.append(f"  - {pair['note']}")
            lines.append("")

    if not any_high:
        lines.append("No feature pairs with |r| > 0.7 found.")
        lines.append("")

    # ---- pamp_score / danger_signal explicit call-out ----
    lines.append("## Key Warnings")
    lines.append("")
    lines.append("### pamp_score / danger_signal Algebraic Relationship")
    lines.append("")
    lines.append(
        "Both `pamp_score` and `danger_signal` are derived from the same underlying "
        "danger pattern match count:"
    )
    lines.append("")
    lines.append("- `pamp_score = min(danger_matches / 3.0, 1.0)`")
    lines.append("- `danger_signal = min(danger_matches / 5.0, 1.0)`")
    lines.append("")
    lines.append(
        "This creates a perfect or near-perfect linear correlation (r=1.0 when "
        "neither saturates). Phase 33 DCA **must** assign these to the same signal "
        "category to avoid double-counting."
    )
    lines.append("")

    # ---- claim_density dependency chain ----
    lines.append("### claim_density Dependency Chain")
    lines.append("")
    lines.append("`claim_density` (source credibility) is computed from 4 other features:")
    lines.append("")
    lines.append("```")
    lines.append("credibility = 0.5")
    lines.append("  + 0.2 * (citation_count > 0)")
    lines.append("  + 0.1 * (hedging_ratio > 0)")
    lines.append("  - 0.2 * (exaggeration > 0.5)")
    lines.append("  - 0.3 * (pamp_score > 0.5)")
    lines.append("```")
    lines.append("")
    lines.append(
        "This means claim_density is algebraically dependent on citation_count, "
        "hedging_ratio, exaggeration, and pamp_score. Phase 33 DCA should group "
        "claim_density with its input features or treat it as a derived signal."
    )
    lines.append("")

    # ---- Constant feature warnings ----
    lines.append("### Constant Feature Warnings")
    lines.append("")

    for result in all_results:
        constant_features = [
            name for name in feature_names if result["feature_stats"][name]["is_constant"]
        ]
        if constant_features:
            lines.append(
                f"- **{result['domain']}**: {len(constant_features)} constant feature(s): "
                f"{', '.join(constant_features)}"
            )
            if len(constant_features) == len(feature_names):
                lines.append(
                    f"  - **WARNING**: All features are constant in {result['domain']}. "
                    "This domain produces degenerate feature vectors. Z-score "
                    "thresholds (Phase 32) will be undefined for this domain. "
                    "NK Cell must handle this gracefully (skip z-score, pass-through to ensemble)."
                )
    lines.append("")

    # ---- Downstream consumption notes ----
    lines.append("## Downstream Consumption")
    lines.append("")
    lines.append("### Phase 32 (NK Cell)")
    lines.append("")
    lines.append(
        "Use `feature_stats.{name}.mean` and `feature_stats.{name}.std` from per-domain "
        "JSON files to compute z-score thresholds. Skip features with `std == 0.0` "
        "(constant features produce undefined z-scores)."
    )
    lines.append("")
    lines.append("### Phase 33 (DCA)")
    lines.append("")
    lines.append(
        "Use `feature_stats.{name}.is_binary` for signal mapping. Use "
        "`high_correlations` to identify features that must be in the same "
        "signal category (especially pamp_score/danger_signal)."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run feature audit across all domains and write output artifacts."""
    # Ensure output directory exists
    audit_dir = _project_root / "data" / "audit"
    os.makedirs(audit_dir, exist_ok=True)

    all_results = []

    for domain in DOMAINS:
        print(f"Auditing domain: {domain}...")
        result = audit_domain(domain)
        all_results.append(result)

        # Write per-domain JSON
        output_path = audit_dir / f"{domain}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {output_path}")

    # Write consolidated summary
    summary_md = generate_summary_md(all_results)
    summary_path = audit_dir / "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(summary_md)
    print(f"\nSummary written to {summary_path}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print(summary_md)


if __name__ == "__main__":
    main()
