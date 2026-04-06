#!/usr/bin/env python3
"""Orchestrator auto-tuning: grid search over NK/DCA/Danger parameters.

Sweeps NK z_threshold, DCA pamp_threshold, and danger alpha against
evaluation data to find the F1-maximizing configuration per domain.
Writes tuned configs to per-domain TOML files loadable by miners.

Grid (D-01):
    NK z_threshold:      [2.0, 3.0, 5.0, 8.0]       (4 values)
    DCA pamp_threshold:  [0.1, 0.3, 0.5, 0.7]       (4 values)
    Danger alpha:        [0.0, 0.05, 0.1, 0.2, 0.3]  (5 values)
    Total: 4 * 4 * 5 = 80 combinations per domain

Usage:
    python scripts/tune_orchestrator.py
    python scripts/tune_orchestrator.py --domains hallucination reasoning
    python scripts/tune_orchestrator.py --output-dir /tmp/tuning --config-dir /tmp/configs
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from antigence_subnet.miner.data import load_training_samples
from antigence_subnet.miner.detectors.dendritic_features import DendriticFeatureExtractor
from antigence_subnet.miner.detectors.negsel import NegSelAISDetector
from antigence_subnet.miner.detectors.sklearn_backends import OCSVMDetector
from antigence_subnet.miner.orchestrator.config import OrchestratorConfig
from antigence_subnet.miner.orchestrator.danger import DangerTheoryModulator
from antigence_subnet.miner.orchestrator.dendritic_cell import DendriticCell
from antigence_subnet.miner.orchestrator.nk_cell import NKCell
from antigence_subnet.miner.orchestrator.orchestrator import ImmuneOrchestrator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"
AUDIT_DIR = Path(__file__).resolve().parent.parent / "data" / "audit"
DOMAINS = ["hallucination", "code_security", "reasoning", "bio"]
DECISION_THRESHOLD = 0.5

# Grid search parameters (D-01)
Z_THRESHOLDS = [2.0, 3.0, 5.0, 8.0]
PAMP_THRESHOLDS = [0.1, 0.3, 0.5, 0.7]
DANGER_ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3]

# Default parameters matching benchmark_orchestrator.py (lines 204-209)
DEFAULT_Z_THRESHOLD = 5.0
DEFAULT_PAMP_THRESHOLD = 0.3
DEFAULT_DANGER_ALPHA = 0.0
DEFAULT_DANGER_ENABLED = False


# ---------------------------------------------------------------------------
# Reused from benchmark_orchestrator.py
# ---------------------------------------------------------------------------


def load_eval_data(domain: str) -> tuple[list[dict], dict]:
    """Load evaluation samples and manifest for a domain."""
    samples_path = DATA_DIR / domain / "samples.json"
    manifest_path = DATA_DIR / domain / "manifest.json"
    with open(samples_path) as f:
        samples = json.load(f)["samples"]
    with open(manifest_path) as f:
        manifest = json.load(f)
    return samples, manifest


def compute_metrics(scores: list[float], labels: list[str], threshold: float = DECISION_THRESHOLD) -> dict:
    """Compute precision, recall, F1, accuracy from scores and ground truth labels."""
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        predicted_anomalous = score >= threshold
        actual_anomalous = label == "anomalous"
        if predicted_anomalous and actual_anomalous:
            tp += 1
        elif predicted_anomalous and not actual_anomalous:
            fp += 1
        elif not predicted_anomalous and actual_anomalous:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def create_detectors(domain: str) -> list:
    """Create OCSVM + NegSel detector pair for a domain."""
    ocsvm = OCSVMDetector()
    negsel = NegSelAISDetector()
    return [ocsvm, negsel]


async def run_orchestrator(orchestrator: ImmuneOrchestrator, samples: list[dict], manifest: dict, domain: str) -> tuple[list[float], list[str]]:
    """Run orchestrator on all samples, return scores and labels."""
    scores = []
    labels = []
    for sample in samples:
        sid = sample["id"]
        gt = manifest.get(sid, {})
        label = gt.get("ground_truth_label", "normal")
        result = await orchestrator.process(
            sample.get("prompt", ""),
            sample.get("output", ""),
            domain,
            sample.get("code"),
            sample.get("context"),
        )
        scores.append(result.score)
        labels.append(label)
    return scores, labels


# ---------------------------------------------------------------------------
# Core tuning functions
# ---------------------------------------------------------------------------


def select_best_config(all_results: list[dict]) -> dict:
    """Select the result entry with highest F1 score.

    Args:
        all_results: List of dicts, each with at least an "f1" key.

    Returns:
        The dict from all_results with the highest F1 value.
    """
    return max(all_results, key=lambda r: r["f1"])


def _build_orchestrator(
    domain: str,
    z_threshold: float,
    pamp_threshold: float,
    danger_alpha: float,
    detectors: list,
) -> ImmuneOrchestrator:
    """Build an ImmuneOrchestrator with specified parameters.

    Follows the exact pattern from benchmark_orchestrator.py lines 204-228.
    """
    config = OrchestratorConfig(
        enabled=True,
        nk_config={"z_threshold": z_threshold},
        dca_config={"pamp_threshold": pamp_threshold},
        danger_config={"alpha": danger_alpha, "enabled": danger_alpha > 0.0},
    )

    audit_json = AUDIT_DIR / f"{domain}.json"
    nk_cell = (
        NKCell.from_audit_json(str(audit_json), z_threshold=z_threshold)
        if audit_json.exists()
        else NKCell(feature_stats=[])
    )
    dc = DendriticCell.from_config(config.dca_config)
    danger = DangerTheoryModulator.from_config(config.danger_config)

    return ImmuneOrchestrator(
        feature_extractor=DendriticFeatureExtractor(),
        nk_cell=nk_cell,
        dendritic_cell=dc,
        danger_modulator=danger,
        detectors={domain: detectors},
    )


async def run_default(domain: str, detectors: list, samples: list[dict], manifest: dict) -> float:
    """Run orchestrator with default params and return F1 for baseline comparison.

    Default params match benchmark_orchestrator.py: z=5.0, pamp=0.3,
    alpha=0.0, danger_enabled=False (D-05).
    """
    orchestrator = _build_orchestrator(
        domain,
        z_threshold=DEFAULT_Z_THRESHOLD,
        pamp_threshold=DEFAULT_PAMP_THRESHOLD,
        danger_alpha=DEFAULT_DANGER_ALPHA,
        detectors=detectors,
    )
    scores, labels = await run_orchestrator(orchestrator, samples, manifest, domain)
    metrics = compute_metrics(scores, labels)
    return metrics["f1"]


async def sweep_domain(domain: str) -> dict:
    """Run grid search over orchestrator parameters for a single domain.

    For each of 80 parameter combinations, constructs an OrchestratorConfig,
    builds an ImmuneOrchestrator, runs it against eval samples, and computes F1.

    Detectors are fitted once outside the sweep loop for efficiency.

    Args:
        domain: Domain to sweep (e.g., "hallucination").

    Returns:
        Dict with keys:
        - "best_config": dict with nk_z_threshold, dca_pamp_threshold, danger_alpha
        - "best_f1": float
        - "default_f1": float
        - "all_results": list of 80 dicts with z_threshold, pamp_threshold, alpha, f1, precision, recall
    """
    # Load data
    samples, manifest = load_eval_data(domain)

    # Fit detectors once (outside sweep loop)
    detectors = create_detectors(domain)
    training = load_training_samples(str(DATA_DIR), domain)
    for det in detectors:
        det.fit(training)

    # Run default baseline (D-05)
    default_f1 = await run_default(domain, detectors, samples, manifest)

    # Grid search
    all_results = []
    total = len(Z_THRESHOLDS) * len(PAMP_THRESHOLDS) * len(DANGER_ALPHAS)
    count = 0

    for z in Z_THRESHOLDS:
        for pamp in PAMP_THRESHOLDS:
            for alpha in DANGER_ALPHAS:
                count += 1
                orchestrator = _build_orchestrator(domain, z, pamp, alpha, detectors)
                scores, labels = await run_orchestrator(orchestrator, samples, manifest, domain)
                metrics = compute_metrics(scores, labels)

                all_results.append({
                    "z_threshold": z,
                    "pamp_threshold": pamp,
                    "alpha": alpha,
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                })

                if count % 20 == 0 or count == total:
                    print(f"    [{domain}] {count}/{total} combinations evaluated...", flush=True)

    # Select best (D-02)
    best_entry = select_best_config(all_results)
    best_config = {
        "nk_z_threshold": best_entry["z_threshold"],
        "dca_pamp_threshold": best_entry["pamp_threshold"],
        "danger_alpha": best_entry["alpha"],
    }

    return {
        "best_config": best_config,
        "best_f1": best_entry["f1"],
        "default_f1": default_f1,
        "all_results": all_results,
    }


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------


def write_sweep_json(domain: str, sweep_result: dict, output_dir: str | None = None) -> str:
    """Write full sweep results to JSON (D-03).

    Args:
        domain: Domain name.
        sweep_result: Dict from sweep_domain().
        output_dir: Output directory. Defaults to data/tuning/.

    Returns:
        Path to the written JSON file.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "data" / "tuning")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / f"{domain}_sweep.json"
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "grid": {
            "z_thresholds": Z_THRESHOLDS,
            "pamp_thresholds": PAMP_THRESHOLDS,
            "danger_alphas": DANGER_ALPHAS,
            "total_combinations": len(Z_THRESHOLDS) * len(PAMP_THRESHOLDS) * len(DANGER_ALPHAS),
        },
        "best_config": sweep_result["best_config"],
        "best_f1": sweep_result["best_f1"],
        "default_f1": sweep_result["default_f1"],
        "all_results": sweep_result["all_results"],
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    return str(json_path)


def write_tuned_toml(domain: str, best_config: dict, config_dir: str | None = None) -> str:
    """Write best config to loadable TOML (D-04).

    Format:
        [miner.orchestrator.domains.<domain>]
        nk_z_threshold = <value>
        dca_pamp_threshold = <value>
        danger_alpha = <value>
        danger_enabled = true

    Args:
        domain: Domain name.
        best_config: Dict with nk_z_threshold, dca_pamp_threshold, danger_alpha.
        config_dir: Output directory. Defaults to configs/tuned/.

    Returns:
        Path to the written TOML file.
    """
    if config_dir is None:
        config_dir = str(Path(__file__).resolve().parent.parent / "configs" / "tuned")

    out_path = Path(config_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    toml_path = out_path / f"{domain}.toml"

    # Determine danger_enabled: enabled if alpha > 0
    danger_enabled = best_config["danger_alpha"] > 0.0

    lines = [
        f"[miner.orchestrator.domains.{domain}]",
        f"nk_z_threshold = {best_config['nk_z_threshold']}",
        f"dca_pamp_threshold = {best_config['dca_pamp_threshold']}",
        f"danger_alpha = {best_config['danger_alpha']}",
        f"danger_enabled = {'true' if danger_enabled else 'false'}",
        "",
    ]

    with open(toml_path, "w") as f:
        f.write("\n".join(lines))

    return str(toml_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune orchestrator parameters via grid search"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DOMAINS,
        help="Domains to tune (default: all 4)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sweep JSON output dir (default: data/tuning)",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Tuned TOML output dir (default: configs/tuned)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Orchestrator Auto-Tuning (Grid Search)")
    print(f"Grid: {len(Z_THRESHOLDS)} z * {len(PAMP_THRESHOLDS)} pamp * {len(DANGER_ALPHAS)} alpha = {len(Z_THRESHOLDS) * len(PAMP_THRESHOLDS) * len(DANGER_ALPHAS)} combinations/domain")
    print("=" * 70)

    results = {}
    start = time.time()

    for domain in args.domains:
        print(f"\n  [{domain}] Starting sweep...", flush=True)
        domain_start = time.time()
        sweep_result = await sweep_domain(domain)
        domain_elapsed = time.time() - domain_start
        results[domain] = sweep_result

        # Write outputs
        json_path = write_sweep_json(domain, sweep_result, output_dir=args.output_dir)
        toml_path = write_tuned_toml(domain, sweep_result["best_config"], config_dir=args.config_dir)

        print(f"  [{domain}] Done in {domain_elapsed:.1f}s", flush=True)
        print(f"    Sweep JSON: {json_path}", flush=True)
        print(f"    Tuned TOML: {toml_path}", flush=True)

    elapsed = time.time() - start

    # Summary table (D-05)
    print(f"\n{'=' * 70}")
    print(f"Auto-Tuning Complete in {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print()
    header = f"{'Domain':<18} | {'Default F1':>10} | {'Tuned F1':>10} | {'Improvement':>11} | {'Best z':>6} | {'Best pamp':>9} | {'Best alpha':>10}"
    print(header)
    print("-" * len(header))

    for domain in args.domains:
        r = results[domain]
        improvement = r["best_f1"] - r["default_f1"]
        bc = r["best_config"]
        print(
            f"{domain:<18} | {r['default_f1']:>10.4f} | {r['best_f1']:>10.4f} | "
            f"{improvement:>+10.4f} | {bc['nk_z_threshold']:>6.1f} | "
            f"{bc['dca_pamp_threshold']:>9.1f} | {bc['danger_alpha']:>10.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
