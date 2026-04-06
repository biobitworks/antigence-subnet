#!/usr/bin/env python3
"""
Benchmark all detectors across all domains.

Measures F1, precision, recall, throughput, and detection quality
for every detector × domain combination. Produces a summary table.

Usage:
    python scripts/benchmark_detectors.py
    python scripts/benchmark_detectors.py --domains hallucination code_security
    python scripts/benchmark_detectors.py --detectors isolation_forest lof ocsvm
    python scripts/benchmark_detectors.py --rounds 5  # repeat for confidence intervals
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigence_subnet.miner.data import load_training_samples
from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector
from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector, OCSVMDetector
from antigence_subnet.miner.detectors.fractal_complexity import FractalComplexityDetector
from antigence_subnet.miner.detectors.negsel import NegSelAISDetector
from antigence_subnet.miner.detectors import (
    HallucinationDetector,
    CodeSecurityDetector,
    ReasoningDetector,
    BioDetector,
)

try:
    from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector
except ImportError:
    AutoencoderDetector = None

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"
DOMAINS = ["hallucination", "code_security", "reasoning", "bio"]

# Generic detectors work on any domain via TF-IDF
GENERIC_DETECTORS = {
    "IsolationForest": IsolationForestDetector,
    "LOF": LOFDetector,
    "OCSVM": OCSVMDetector,
    "Fractal": FractalComplexityDetector,
    "NegSel": NegSelAISDetector,
}
if AutoencoderDetector:
    GENERIC_DETECTORS["Autoencoder"] = AutoencoderDetector

# Domain-specific detectors (only for their domain)
DOMAIN_DETECTORS = {
    "hallucination": {"HallucinationPack": HallucinationDetector},
    "code_security": {"CodeSecurityPack": CodeSecurityDetector},
    "reasoning": {"ReasoningPack": ReasoningDetector},
    "bio": {"BioPack": BioDetector},
}


def load_eval_data(domain: str):
    """Load all samples and ground truth for a domain."""
    samples_path = DATA_DIR / domain / "samples.json"
    manifest_path = DATA_DIR / domain / "manifest.json"

    with open(samples_path) as f:
        all_samples = json.load(f)["samples"]
    with open(manifest_path) as f:
        manifest = json.load(f)

    return all_samples, manifest


async def benchmark_detector(detector, all_samples, manifest, threshold=0.5):
    """Run detector on all samples, compute metrics."""
    tp = fp = fn = tn = 0
    scores = []
    latencies = []

    for sample in all_samples:
        truth = manifest.get(sample["id"], {}).get("ground_truth_label", "normal")
        is_anomalous = truth == "anomalous"

        t0 = time.perf_counter()
        try:
            result = await detector.detect(
                prompt=sample.get("prompt", ""),
                output=sample.get("output", ""),
                code=sample.get("code"),
                context=sample.get("context"),
            )
            latency = time.perf_counter() - t0
            latencies.append(latency)

            predicted_anomalous = result.score >= threshold
            scores.append(result.score)

            if predicted_anomalous and is_anomalous:
                tp += 1
            elif predicted_anomalous and not is_anomalous:
                fp += 1
            elif not predicted_anomalous and is_anomalous:
                fn += 1
            else:
                tn += 1
        except Exception as e:
            latency = time.perf_counter() - t0
            latencies.append(latency)
            # Detection failure counts as missed anomaly or correct normal
            if is_anomalous:
                fn += 1
            else:
                tn += 1
            scores.append(0.0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(all_samples) if all_samples else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = len(all_samples) / sum(latencies) if sum(latencies) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency * 1000,
        "throughput_per_sec": throughput,
        "samples": len(all_samples),
    }


async def run_benchmarks(domains, detector_filter, rounds):
    """Run all benchmarks."""
    results = []

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"  Domain: {domain} ({DATA_DIR / domain})")
        print(f"{'='*60}")

        all_samples, manifest = load_eval_data(domain)
        normal_samples = [
            s for s in all_samples
            if manifest.get(s["id"], {}).get("ground_truth_label") == "normal"
        ]
        anomalous_count = sum(
            1 for s in all_samples
            if manifest.get(s["id"], {}).get("ground_truth_label") == "anomalous"
        )
        print(f"  Samples: {len(all_samples)} total ({len(normal_samples)} normal, {anomalous_count} anomalous)")

        # Collect detectors for this domain
        detectors_to_test = {}

        # Generic detectors
        for name, cls in GENERIC_DETECTORS.items():
            if detector_filter and name.lower() not in detector_filter:
                continue
            detectors_to_test[name] = cls

        # Domain-specific detector
        domain_dets = DOMAIN_DETECTORS.get(domain, {})
        for name, cls in domain_dets.items():
            if detector_filter and name.lower() not in detector_filter:
                continue
            detectors_to_test[name] = cls

        for det_name, det_cls in detectors_to_test.items():
            round_results = []

            for r in range(rounds):
                try:
                    detector = det_cls()
                    # Override domain for generic detectors
                    if det_name in GENERIC_DETECTORS:
                        detector.domain = domain

                    t_fit_start = time.perf_counter()
                    detector.fit(normal_samples)
                    fit_time = time.perf_counter() - t_fit_start

                    metrics = await benchmark_detector(detector, all_samples, manifest)
                    metrics["fit_time_ms"] = fit_time * 1000
                    metrics["detector"] = det_name
                    metrics["domain"] = domain
                    metrics["round"] = r + 1
                    round_results.append(metrics)

                except Exception as e:
                    print(f"  {det_name:25s} | ERROR: {e}")
                    round_results.append({
                        "detector": det_name, "domain": domain, "round": r + 1,
                        "f1": 0, "precision": 0, "recall": 0, "error": str(e),
                    })

            # Average across rounds
            if round_results and "f1" in round_results[0]:
                avg = {
                    "detector": det_name,
                    "domain": domain,
                    "f1": sum(r["f1"] for r in round_results) / len(round_results),
                    "precision": sum(r["precision"] for r in round_results) / len(round_results),
                    "recall": sum(r["recall"] for r in round_results) / len(round_results),
                    "accuracy": sum(r.get("accuracy", 0) for r in round_results) / len(round_results),
                    "avg_latency_ms": sum(r.get("avg_latency_ms", 0) for r in round_results) / len(round_results),
                    "throughput_per_sec": sum(r.get("throughput_per_sec", 0) for r in round_results) / len(round_results),
                    "fit_time_ms": sum(r.get("fit_time_ms", 0) for r in round_results) / len(round_results),
                    "rounds": rounds,
                }
                results.append(avg)

                status = "PASS" if avg["f1"] > 0.3 else "WARN" if avg["f1"] > 0 else "FAIL"
                print(
                    f"  {det_name:25s} | F1={avg['f1']:.3f} | P={avg['precision']:.3f} "
                    f"| R={avg['recall']:.3f} | {avg['avg_latency_ms']:.1f}ms/sample "
                    f"| fit={avg['fit_time_ms']:.0f}ms | [{status}]"
                )

    return results


def print_summary_table(results):
    """Print final summary as markdown table."""
    print(f"\n{'='*80}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    print("| Detector | Domain | F1 | Precision | Recall | Latency (ms) | Throughput |")
    print("|----------|--------|----|-----------|--------|--------------|------------|")

    for r in sorted(results, key=lambda x: (x["domain"], -x["f1"])):
        print(
            f"| {r['detector']:20s} | {r['domain']:14s} | {r['f1']:.3f} | "
            f"{r['precision']:.3f}     | {r['recall']:.3f}  | "
            f"{r['avg_latency_ms']:>10.1f}  | {r['throughput_per_sec']:>8.1f}/s  |"
        )

    # Best per domain
    print(f"\n{'='*60}")
    print("  BEST DETECTOR PER DOMAIN (by F1)")
    print(f"{'='*60}\n")

    domains_seen = {}
    for r in sorted(results, key=lambda x: -x["f1"]):
        if r["domain"] not in domains_seen:
            domains_seen[r["domain"]] = r
            print(f"  {r['domain']:15s} → {r['detector']:20s} (F1={r['f1']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all detectors across domains")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, help="Domains to test")
    parser.add_argument("--detectors", nargs="+", default=None, help="Filter detectors (lowercase)")
    parser.add_argument("--rounds", type=int, default=1, help="Repeat N times for confidence")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    args = parser.parse_args()

    detector_filter = [d.lower() for d in args.detectors] if args.detectors else None

    print("=" * 60)
    print("  ANTIGENCE SUBNET — DETECTOR BENCHMARK")
    print("=" * 60)
    print(f"  Domains: {args.domains}")
    print(f"  Rounds:  {args.rounds}")
    if detector_filter:
        print(f"  Filter:  {detector_filter}")

    results = asyncio.run(run_benchmarks(args.domains, detector_filter, args.rounds))

    print_summary_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Exit code: 1 if any detector has F1=0
    if any(r["f1"] == 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
