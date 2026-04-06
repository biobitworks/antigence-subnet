"""Config validation CLI for Antigence subnet TOML configuration files.

Validates TOML structure, parameter ranges, detects unknown keys and
conflicting settings. Provides a dry-run mode that loads config, creates
feature extractors, and runs NK Cell evaluation on sample data.

Usage:
    python -m antigence_subnet.validate_config config.toml
    python -m antigence_subnet.validate_config --dry-run config.toml
    python -m antigence_subnet.validate_config --json config.toml
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import numpy as np

from antigence_subnet.miner.orchestrator.config import OrchestratorConfig
from antigence_subnet.protocol import KNOWN_DOMAINS
from antigence_subnet.utils.config_file import load_toml_config

# ---------------------------------------------------------------------------
# Known TOML sections for unknown-key detection
# ---------------------------------------------------------------------------

# Top-level known keys
_KNOWN_TOP_LEVEL = {"neuron", "validator", "miner", "api"}

# Known keys under [miner]
_KNOWN_MINER = {"detectors", "orchestrator", "model"}

# Known keys under [miner.orchestrator]
_KNOWN_ORCHESTRATOR = {
    "enabled", "nk", "dca", "danger", "bcell", "slm_nk",
    "domains", "feedback", "telemetry",
}

# Known keys under [miner.orchestrator.nk]
_KNOWN_NK = {"z_threshold"}

# Known keys under [miner.orchestrator.dca]
_KNOWN_DCA = {"pamp_threshold", "signal_weights", "adaptive", "adapt_alpha"}

# Known keys under [miner.orchestrator.danger]
_KNOWN_DANGER = {"alpha", "enabled"}

# Known keys under [miner.orchestrator.bcell]
_KNOWN_BCELL = {
    "max_memory", "k", "bcell_weight", "half_life",
    "eviction_threshold", "jitter_sigma", "memory_dir",
    "embedding_mode", "embedding_sigma",
}

# Known keys under [miner.orchestrator.slm_nk]
_KNOWN_SLM_NK = {"enabled", "similarity_threshold"}

# Known keys under [miner.model]
_KNOWN_MODEL = {"model_name", "cache_dir", "device"}

# Known keys under [miner.orchestrator.feedback]
_KNOWN_FEEDBACK = {"enabled", "lookback_rounds"}

# Known keys under [miner.orchestrator.telemetry]
_KNOWN_TELEMETRY = {"window_size"}

# Known keys per domain override
_KNOWN_DOMAIN_KEYS = {
    "nk_z_threshold", "dca_pamp_threshold", "danger_alpha",
    "danger_enabled", "slm_nk_similarity_threshold",
}

# Known keys under [validator]
_KNOWN_VALIDATOR = {"rotation", "scoring", "policy"}

# Known keys under [validator.rotation]
_KNOWN_ROTATION = {"enabled", "window"}

# Known keys under [validator.scoring]
_KNOWN_SCORING = {"mode", "repeats", "ci_level"}

# Known keys under [validator.policy]
_KNOWN_POLICY = {"mode", "high_threshold", "low_threshold", "min_confidence"}


@dataclass
class ConfigIssue:
    """A single config validation issue.

    Attributes:
        level: Issue severity -- "error" or "warning".
        section: Dotted TOML section path (e.g., "miner.orchestrator.nk").
        message: Human-readable description of the issue.
    """

    level: str  # "error" | "warning"
    section: str
    message: str


def _check_unknown_keys(
    data: dict[str, Any],
    known: set[str],
    section_path: str,
    issues: list[ConfigIssue],
) -> None:
    """Check for unknown keys in a TOML section and emit warnings."""
    for key in data:
        if key not in known:
            issues.append(ConfigIssue(
                level="warning",
                section=section_path,
                message=f"Unknown key '{key}' in [{section_path}]",
            ))


def _validate_ranges(toml_data: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """Validate parameter ranges beyond what OrchestratorConfig catches.

    OrchestratorConfig.from_toml_raw() validates some ranges via ValueError,
    but additional checks are needed for completeness.
    """
    orch = toml_data.get("miner", {}).get("orchestrator", {})

    # NK checks
    nk = orch.get("nk", {})
    z_threshold = nk.get("z_threshold")
    if z_threshold is not None and z_threshold < 0:
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.nk",
            message=f"z_threshold must be >= 0, got {z_threshold}",
        ))

    # DCA checks
    dca = orch.get("dca", {})
    pamp_threshold = dca.get("pamp_threshold")
    if pamp_threshold is not None and pamp_threshold < 0:
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.dca",
            message=f"pamp_threshold must be >= 0, got {pamp_threshold}",
        ))

    adapt_alpha = dca.get("adapt_alpha")
    if adapt_alpha is not None and (adapt_alpha <= 0.0 or adapt_alpha > 1.0):
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.dca",
            message=f"adapt_alpha must be in (0.0, 1.0], got {adapt_alpha}",
        ))

    # Danger checks
    danger = orch.get("danger", {})
    alpha = danger.get("alpha")
    if alpha is not None and (alpha < 0.0 or alpha > 1.0):
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.danger",
            message=f"alpha must be in [0.0, 1.0], got {alpha}",
        ))

    # B Cell checks
    bcell = orch.get("bcell", {})
    bcell_weight = bcell.get("bcell_weight")
    if bcell_weight is not None and (bcell_weight < 0.0 or bcell_weight > 1.0):
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.bcell",
            message=f"bcell_weight must be in [0.0, 1.0], got {bcell_weight}",
        ))

    max_memory = bcell.get("max_memory")
    if max_memory is not None and (not isinstance(max_memory, int) or max_memory <= 0):
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.bcell",
            message=f"max_memory must be int > 0, got {max_memory}",
        ))

    k = bcell.get("k")
    if k is not None and (not isinstance(k, int) or k <= 0):
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.bcell",
            message=f"k must be int > 0, got {k}",
        ))

    half_life = bcell.get("half_life")
    if half_life is not None and (half_life <= 0.0 or half_life > 1.0):
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.bcell",
            message=f"half_life must be in (0.0, 1.0], got {half_life}",
        ))

    # SLM NK checks
    slm_nk = orch.get("slm_nk", {})
    sim_thresh = slm_nk.get("similarity_threshold")
    if sim_thresh is not None and (sim_thresh < 0.0 or sim_thresh > 1.0):
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.slm_nk",
            message=f"similarity_threshold must be in [0.0, 1.0], got {sim_thresh}",
        ))

    # Model checks
    model = toml_data.get("miner", {}).get("model", {})
    device = model.get("device")
    if device is not None and device not in {"auto", "cpu", "cuda"}:
        issues.append(ConfigIssue(
            level="error",
            section="miner.model",
            message=f"device must be one of {{auto, cpu, cuda}}, got '{device}'",
        ))

    # Feedback checks
    feedback = orch.get("feedback", {})
    lookback = feedback.get("lookback_rounds")
    if lookback is not None and (not isinstance(lookback, int) or lookback <= 0):
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator.feedback",
            message=f"lookback_rounds must be int > 0, got {lookback}",
        ))

    # Domain override range checks
    domains = orch.get("domains", {})
    for domain_name, domain_dict in domains.items():
        if not isinstance(domain_dict, dict):
            continue
        section = f"miner.orchestrator.domains.{domain_name}"
        d_z = domain_dict.get("nk_z_threshold")
        if d_z is not None and d_z < 0:
            issues.append(ConfigIssue(
                level="error", section=section,
                message=f"nk_z_threshold must be >= 0, got {d_z}",
            ))
        d_pamp = domain_dict.get("dca_pamp_threshold")
        if d_pamp is not None and d_pamp < 0:
            issues.append(ConfigIssue(
                level="error", section=section,
                message=f"dca_pamp_threshold must be >= 0, got {d_pamp}",
            ))
        d_alpha = domain_dict.get("danger_alpha")
        if d_alpha is not None and (d_alpha < 0.0 or d_alpha > 1.0):
            issues.append(ConfigIssue(
                level="error", section=section,
                message=f"danger_alpha must be in [0.0, 1.0], got {d_alpha}",
            ))
        d_sim = domain_dict.get("slm_nk_similarity_threshold")
        if d_sim is not None and (d_sim < 0.0 or d_sim > 1.0):
            issues.append(ConfigIssue(
                level="error", section=section,
                message=f"slm_nk_similarity_threshold must be in [0.0, 1.0], got {d_sim}",
            ))

    # Rotation checks (VHARD-01)
    rotation = toml_data.get("validator", {}).get("rotation", {})
    rot_window = rotation.get("window")
    if rot_window is not None and (not isinstance(rot_window, int) or rot_window <= 0):
        issues.append(ConfigIssue(
            level="error",
            section="validator.rotation",
            message=f"window must be int > 0, got {rot_window}",
        ))

    scoring = toml_data.get("validator", {}).get("scoring", {})
    scoring_mode = scoring.get("mode")
    if scoring_mode is not None and scoring_mode not in {
        "exact",
        "statistical",
        "semantic",
    }:
        issues.append(ConfigIssue(
            level="error",
            section="validator.scoring",
            message=(
                "mode must be one of {exact, statistical, semantic}, "
                f"got '{scoring_mode}'"
            ),
        ))

    scoring_repeats = scoring.get("repeats")
    if scoring_repeats is not None and (
        not isinstance(scoring_repeats, int) or scoring_repeats <= 0
    ):
        issues.append(ConfigIssue(
            level="error",
            section="validator.scoring",
            message=f"repeats must be int > 0, got {scoring_repeats}",
        ))

    scoring_ci_level = scoring.get("ci_level")
    if scoring_ci_level is not None and (
        scoring_ci_level <= 0.0 or scoring_ci_level >= 1.0
    ):
        issues.append(ConfigIssue(
            level="error",
            section="validator.scoring",
            message=f"ci_level must be in (0.0, 1.0), got {scoring_ci_level}",
        ))

    policy = toml_data.get("validator", {}).get("policy", {})
    policy_mode = policy.get("mode")
    if policy_mode is not None and policy_mode not in {
        "global_threshold",
        "domain_thresholds",
        "operator_multiband",
    }:
        issues.append(ConfigIssue(
            level="error",
            section="validator.policy",
            message=(
                "mode must be one of {{global_threshold, domain_thresholds, "
                f"operator_multiband}}, got '{policy_mode}'"
            ),
        ))

    high_threshold = policy.get("high_threshold")
    if high_threshold is not None and (high_threshold < 0.0 or high_threshold > 1.0):
        issues.append(ConfigIssue(
            level="error",
            section="validator.policy",
            message=f"high_threshold must be in [0.0, 1.0], got {high_threshold}",
        ))

    low_threshold = policy.get("low_threshold")
    if low_threshold is not None and (low_threshold < 0.0 or low_threshold > 1.0):
        issues.append(ConfigIssue(
            level="error",
            section="validator.policy",
            message=f"low_threshold must be in [0.0, 1.0], got {low_threshold}",
        ))

    min_confidence = policy.get("min_confidence")
    if min_confidence is not None and (min_confidence < 0.0 or min_confidence > 1.0):
        issues.append(ConfigIssue(
            level="error",
            section="validator.policy",
            message=f"min_confidence must be in [0.0, 1.0], got {min_confidence}",
        ))

    if (
        high_threshold is not None
        and low_threshold is not None
        and low_threshold > high_threshold
    ):
        issues.append(ConfigIssue(
            level="error",
            section="validator.policy",
            message=(
                "low_threshold must be <= high_threshold, "
                f"got low_threshold={low_threshold}, high_threshold={high_threshold}"
            ),
        ))


def _check_conflicts(toml_data: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """Detect conflicting settings and emit warnings."""
    orch = toml_data.get("miner", {}).get("orchestrator", {})
    danger = orch.get("danger", {})

    # Warn if danger.enabled=false but alpha > 0 (alpha has no effect)
    if danger.get("enabled") is False and danger.get("alpha", 0) > 0:
        issues.append(ConfigIssue(
            level="warning",
            section="miner.orchestrator.danger",
            message=(
                f"danger.enabled is false but alpha={danger.get('alpha')} > 0 "
                "(alpha has no effect when danger is disabled)"
            ),
        ))


def _check_unknown_keys_deep(toml_data: dict[str, Any], issues: list[ConfigIssue]) -> None:
    """Walk the TOML tree and check for unknown keys at each level."""
    # Top level
    _check_unknown_keys(toml_data, _KNOWN_TOP_LEVEL, "root", issues)

    # Under [miner]
    miner = toml_data.get("miner", {})
    if isinstance(miner, dict):
        _check_unknown_keys(miner, _KNOWN_MINER, "miner", issues)

        # Under [miner.orchestrator]
        orch = miner.get("orchestrator", {})
        if isinstance(orch, dict):
            _check_unknown_keys(orch, _KNOWN_ORCHESTRATOR, "miner.orchestrator", issues)

            # Sub-sections
            nk = orch.get("nk", {})
            if isinstance(nk, dict):
                _check_unknown_keys(nk, _KNOWN_NK, "miner.orchestrator.nk", issues)

            dca = orch.get("dca", {})
            if isinstance(dca, dict):
                _check_unknown_keys(dca, _KNOWN_DCA, "miner.orchestrator.dca", issues)

            danger = orch.get("danger", {})
            if isinstance(danger, dict):
                _check_unknown_keys(danger, _KNOWN_DANGER, "miner.orchestrator.danger", issues)

            bcell = orch.get("bcell", {})
            if isinstance(bcell, dict):
                _check_unknown_keys(bcell, _KNOWN_BCELL, "miner.orchestrator.bcell", issues)

            slm_nk = orch.get("slm_nk", {})
            if isinstance(slm_nk, dict):
                _check_unknown_keys(slm_nk, _KNOWN_SLM_NK, "miner.orchestrator.slm_nk", issues)

            feedback = orch.get("feedback", {})
            if isinstance(feedback, dict):
                _check_unknown_keys(
                    feedback,
                    _KNOWN_FEEDBACK,
                    "miner.orchestrator.feedback",
                    issues,
                )

            telemetry = orch.get("telemetry", {})
            if isinstance(telemetry, dict):
                _check_unknown_keys(
                    telemetry,
                    _KNOWN_TELEMETRY,
                    "miner.orchestrator.telemetry",
                    issues,
                )

            # Domain override keys
            domains = orch.get("domains", {})
            if isinstance(domains, dict):
                for domain_name, domain_dict in domains.items():
                    if isinstance(domain_dict, dict):
                        _check_unknown_keys(
                            domain_dict,
                            _KNOWN_DOMAIN_KEYS,
                            f"miner.orchestrator.domains.{domain_name}",
                            issues,
                        )

        # Under [miner.model]
        model = miner.get("model", {})
        if isinstance(model, dict):
            _check_unknown_keys(model, _KNOWN_MODEL, "miner.model", issues)

    # Under [validator]
    validator = toml_data.get("validator", {})
    if isinstance(validator, dict):
        _check_unknown_keys(validator, _KNOWN_VALIDATOR, "validator", issues)

        # Under [validator.rotation]
        rotation = validator.get("rotation", {})
        if isinstance(rotation, dict):
            _check_unknown_keys(rotation, _KNOWN_ROTATION, "validator.rotation", issues)

        scoring = validator.get("scoring", {})
        if isinstance(scoring, dict):
            _check_unknown_keys(scoring, _KNOWN_SCORING, "validator.scoring", issues)

        policy = validator.get("policy", {})
        if isinstance(policy, dict):
            _check_unknown_keys(policy, _KNOWN_POLICY, "validator.policy", issues)


def validate_config(toml_path: Path) -> list[ConfigIssue]:
    """Validate a TOML config file and return a list of issues.

    Checks TOML parseability, parameter ranges, unknown keys, and
    conflicting settings.

    Args:
        toml_path: Path to the TOML config file.

    Returns:
        List of ConfigIssue instances. Empty list means valid config.
    """
    issues: list[ConfigIssue] = []

    # Check file exists
    if not toml_path.exists():
        issues.append(ConfigIssue(
            level="error",
            section="file",
            message=f"Config file does not exist: {toml_path}",
        ))
        return issues

    # Check TOML parseable
    try:
        toml_data = load_toml_config(toml_path)
    except tomllib.TOMLDecodeError as e:
        issues.append(ConfigIssue(
            level="error",
            section="file",
            message=f"Malformed TOML: {e}",
        ))
        return issues

    # Check unknown keys (deep walk)
    _check_unknown_keys_deep(toml_data, issues)

    # Check parameter ranges
    _validate_ranges(toml_data, issues)

    # Check conflicting settings
    _check_conflicts(toml_data, issues)

    # Try OrchestratorConfig.from_toml_raw() to catch its built-in validation
    try:
        OrchestratorConfig.from_toml_raw(toml_data)
    except ValueError as e:
        issues.append(ConfigIssue(
            level="error",
            section="miner.orchestrator",
            message=f"OrchestratorConfig validation failed: {e}",
        ))

    return issues


def dry_run(toml_path: Path) -> dict[str, dict[str, Any]]:
    """Run a dry validation: load config, extract features, run NK Cell check.

    Loads the config, creates a DendriticFeatureExtractor, and runs
    NK Cell evaluation on up to 5 samples per domain from evaluation data.

    Does NOT require a trained detector or model download -- tests config
    plumbing with real feature extraction only.

    Args:
        toml_path: Path to the TOML config file.

    Returns:
        Dict mapping domain -> {nk_triggers, feature_stats, samples_tested, ...}
    """
    from antigence_subnet.miner.detectors.dendritic_features import (
        DendriticFeatureExtractor,
    )
    from antigence_subnet.miner.orchestrator.nk_cell import NKCell

    toml_data = load_toml_config(toml_path)
    config = OrchestratorConfig.from_toml_raw(toml_data)

    # Find project root (where data/ directory lives)
    project_root = toml_path.resolve().parent
    # Walk up to find data/evaluation/ if not at project root
    for _ in range(5):
        if (project_root / "data" / "evaluation").exists():
            break
        project_root = project_root.parent

    eval_dir = project_root / "data" / "evaluation"
    audit_dir = project_root / "data" / "audit"

    extractor = DendriticFeatureExtractor()
    results: dict[str, dict[str, Any]] = {}

    for domain in sorted(KNOWN_DOMAINS):
        samples_path = eval_dir / domain / "samples.json"
        if not samples_path.exists():
            results[domain] = {
                "nk_triggers": 0,
                "feature_stats": {},
                "samples_tested": 0,
                "status": "no eval data",
            }
            continue

        # Load up to 5 samples
        with open(samples_path) as f:
            data = json.load(f)
        samples = data.get("samples", [])[:5]

        # Try to load NK Cell from audit data if available
        audit_path = audit_dir / f"{domain}.json"
        nk_cell: NKCell | None = None
        z_threshold = config.nk_config.get("z_threshold", 3.0)

        # Check for per-domain override
        domain_cfg = config.get_domain_config(domain)
        if domain_cfg and domain_cfg.nk_z_threshold is not None:
            z_threshold = domain_cfg.nk_z_threshold

        if audit_path.exists():
            with suppress(json.JSONDecodeError, KeyError):
                nk_cell = NKCell.from_audit_json(audit_path, z_threshold=z_threshold)

        # Extract features and run NK Cell if available
        all_features: list[np.ndarray] = []
        nk_triggers = 0
        for sample in samples:
            text = sample.get("output", "") or ""
            features = extractor.extract(text)
            all_features.append(features)

            if nk_cell is not None:
                result = nk_cell.process(
                    features=features,
                    prompt=sample.get("prompt", ""),
                    output=text,
                )
                if result is not None:
                    nk_triggers += 1

        # Compute feature stats
        feature_stats: dict[str, Any] = {}
        if all_features:
            stacked = np.stack(all_features)
            feature_names = DendriticFeatureExtractor.FEATURE_NAMES
            for i, name in enumerate(feature_names):
                col = stacked[:, i]
                feature_stats[name] = {
                    "mean": round(float(np.mean(col)), 4),
                    "std": round(float(np.std(col)), 4),
                    "min": round(float(np.min(col)), 4),
                    "max": round(float(np.max(col)), 4),
                }

        results[domain] = {
            "nk_triggers": nk_triggers,
            "feature_stats": feature_stats,
            "samples_tested": len(samples),
            "z_threshold": z_threshold,
            "status": "ok",
        }

    return results


def main() -> None:
    """CLI entry point for config validation."""
    parser = argparse.ArgumentParser(
        prog="validate_config",
        description="Validate Antigence subnet TOML configuration files.",
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the TOML config file to validate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extended validation with feature extraction on sample data.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results in machine-readable JSON format.",
    )

    args = parser.parse_args()

    # Run validation
    issues = validate_config(args.config_path)
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    if args.json_output:
        output: dict[str, Any] = {
            "valid": len(errors) == 0,
            "errors": [asdict(e) for e in errors],
            "warnings": [asdict(w) for w in warnings],
        }
        if args.dry_run and len(errors) == 0:
            output["dry_run"] = dry_run(args.config_path)
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        if errors:
            print(f"ERRORS ({len(errors)}):")
            for e in errors:
                print(f"  [{e.section}] {e.message}")
            print()

        if warnings:
            print(f"WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"  [{w.section}] {w.message}")
            print()

        if len(errors) == 0:
            print("Config valid.")

        if args.dry_run and len(errors) == 0:
            print("\n--- Dry Run Results ---\n")
            dr_results = dry_run(args.config_path)
            for domain, info in sorted(dr_results.items()):
                status = info.get("status", "ok")
                if status == "no eval data":
                    print(f"  {domain}: no evaluation data available")
                    continue
                print(f"  {domain}:")
                print(f"    Samples tested: {info['samples_tested']}")
                print(f"    NK triggers: {info['nk_triggers']}")
                print(f"    z_threshold: {info.get('z_threshold', 'N/A')}")
                if info.get("feature_stats"):
                    print("    Feature stats:")
                    for fname, fstats in info["feature_stats"].items():
                        print(
                            f"      {fname}: "
                            f"mean={fstats['mean']}, std={fstats['std']}, "
                            f"range=[{fstats['min']}, {fstats['max']}]"
                        )
                print()

    # Exit code: 0 if valid, 1 if errors
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
