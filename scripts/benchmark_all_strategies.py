#!/usr/bin/env python3
"""
Comprehensive strategy benchmark: singles, ensembles, and all combinations.

Compares:
A. Single detectors (baseline)
B. Every 2-detector ensemble
C. Every 3-detector ensemble
D. Best-possible ensemble (all detectors)
E. NegSl-AIS (real Umair et al. 2025 negative selection on dendritic features)

Usage:
    python scripts/benchmark_all_strategies.py
    python scripts/benchmark_all_strategies.py --domain hallucination
"""

import argparse
import asyncio
import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigence_subnet.miner.data import load_training_samples
from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector
from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector, OCSVMDetector
from antigence_subnet.miner.detectors.fractal_complexity import FractalComplexityDetector
from antigence_subnet.miner.detectors.negsel import NegSelAISDetector
from antigence_subnet.miner.ensemble import ensemble_detect

try:
    from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector
except ImportError:
    AutoencoderDetector = None

from antigence_subnet.miner.detectors import (
    HallucinationDetector,
    CodeSecurityDetector,
    ReasoningDetector,
    BioDetector,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"
DOMAINS = ["hallucination", "code_security", "reasoning", "bio"]

# Domain pack mapping
DOMAIN_PACK = {
    "hallucination": ("HallPack", HallucinationDetector),
    "code_security": ("CodePack", CodeSecurityDetector),
    "reasoning": ("ReasPack", ReasoningDetector),
    "bio": ("BioPack", BioDetector),
}



def load_eval_data(domain):
    with open(DATA_DIR / domain / "samples.json") as f:
        samples = json.load(f)["samples"]
    with open(DATA_DIR / domain / "manifest.json") as f:
        manifest = json.load(f)
    return samples, manifest


async def eval_detector(detector, samples, manifest, threshold=0.5):
    """Evaluate a single detector. Returns metrics dict."""
    tp = fp = fn = tn = 0
    latencies = []

    for s in samples:
        truth = manifest.get(s["id"], {}).get("ground_truth_label", "normal")
        is_anom = truth == "anomalous"

        t0 = time.perf_counter()
        result = await detector.detect(
            prompt=s.get("prompt", ""),
            output=s.get("output", ""),
            code=s.get("code"),
            context=s.get("context"),
        )
        latencies.append(time.perf_counter() - t0)

        pred = result.score >= threshold
        if pred and is_anom:
            tp += 1
        elif pred and not is_anom:
            fp += 1
        elif not pred and is_anom:
            fn += 1
        else:
            tn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        "precision": p, "recall": r, "f1": f1,
        "accuracy": (tp + tn) / len(samples),
        "avg_ms": sum(latencies) / len(latencies) * 1000,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


async def eval_ensemble(detectors, samples, manifest, threshold=0.5):
    """Evaluate an ensemble of fitted detectors."""
    tp = fp = fn = tn = 0
    latencies = []

    for s in samples:
        truth = manifest.get(s["id"], {}).get("ground_truth_label", "normal")
        is_anom = truth == "anomalous"

        t0 = time.perf_counter()
        result = await ensemble_detect(
            detectors,
            prompt=s.get("prompt", ""),
            output=s.get("output", ""),
            code=s.get("code"),
            context=s.get("context"),
        )
        latencies.append(time.perf_counter() - t0)

        pred = result.score >= threshold
        if pred and is_anom:
            tp += 1
        elif pred and not is_anom:
            fp += 1
        elif not pred and is_anom:
            fn += 1
        else:
            tn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        "precision": p, "recall": r, "f1": f1,
        "accuracy": (tp + tn) / len(samples),
        "avg_ms": sum(latencies) / len(latencies) * 1000,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


async def run_domain(domain):
    """Run all strategies for one domain."""
    print(f"\n{'='*70}")
    print(f"  DOMAIN: {domain}")
    print(f"{'='*70}")

    samples, manifest = load_eval_data(domain)
    normal = [s for s in samples if manifest.get(s["id"], {}).get("ground_truth_label") == "normal"]
    anom_count = len(samples) - len(normal)
    print(f"  {len(samples)} samples ({len(normal)} normal, {anom_count} anomalous)\n")

    # Build all detectors
    detector_map = {}

    generic = {
        "IF": IsolationForestDetector,
        "LOF": LOFDetector,
        "OCSVM": OCSVMDetector,
        "Fractal": FractalComplexityDetector,
        "NegSel": NegSelAISDetector,
    }
    if AutoencoderDetector:
        generic["AE"] = AutoencoderDetector

    # Domain pack
    pack_name, pack_cls = DOMAIN_PACK[domain]

    all_classes = {**generic, pack_name: pack_cls}

    # Fit all detectors
    print("  Fitting detectors...")
    for name, cls in all_classes.items():
        try:
            det = cls()
            if hasattr(det, "domain"):
                det.domain = domain
            det.fit(normal)
            detector_map[name] = det
        except Exception as e:
            print(f"    {name}: FAILED to fit ({e})")

    results = []

    # === A. Singles ===
    print(f"\n  --- A. SINGLE DETECTORS ({len(detector_map)}) ---")
    for name, det in detector_map.items():
        m = await eval_detector(det, samples, manifest)
        m["strategy"] = "single"
        m["name"] = name
        m["domain"] = domain
        m["detector_count"] = 1
        results.append(m)
        tag = "***" if m["f1"] >= 0.9 else "**" if m["f1"] >= 0.7 else "*" if m["f1"] >= 0.5 else ""
        print(f"    {name:12s} | F1={m['f1']:.3f} | P={m['precision']:.3f} | R={m['recall']:.3f} | {m['avg_ms']:.1f}ms {tag}")

    # === B. Every 2-combo ensemble ===
    names = list(detector_map.keys())
    print(f"\n  --- B. 2-DETECTOR ENSEMBLES ({len(list(combinations(names, 2)))}) ---")
    for combo in combinations(names, 2):
        dets = [detector_map[n] for n in combo]
        label = "+".join(combo)
        m = await eval_ensemble(dets, samples, manifest)
        m["strategy"] = "ensemble-2"
        m["name"] = label
        m["domain"] = domain
        m["detector_count"] = 2
        results.append(m)
        tag = "***" if m["f1"] >= 0.9 else "**" if m["f1"] >= 0.7 else ""
        print(f"    {label:25s} | F1={m['f1']:.3f} | P={m['precision']:.3f} | R={m['recall']:.3f} | {m['avg_ms']:.1f}ms {tag}")

    # === C. Every 3-combo ensemble ===
    print(f"\n  --- C. 3-DETECTOR ENSEMBLES (top 10 by F1) ---")
    three_results = []
    for combo in combinations(names, 3):
        dets = [detector_map[n] for n in combo]
        label = "+".join(combo)
        m = await eval_ensemble(dets, samples, manifest)
        m["strategy"] = "ensemble-3"
        m["name"] = label
        m["domain"] = domain
        m["detector_count"] = 3
        three_results.append(m)

    three_results.sort(key=lambda x: -x["f1"])
    for m in three_results[:10]:
        results.append(m)
        tag = "***" if m["f1"] >= 0.9 else "**" if m["f1"] >= 0.7 else ""
        print(f"    {m['name']:35s} | F1={m['f1']:.3f} | P={m['precision']:.3f} | R={m['recall']:.3f} | {m['avg_ms']:.1f}ms {tag}")

    # === D. All detectors ensemble ===
    print(f"\n  --- D. ALL-DETECTOR ENSEMBLE ---")
    all_dets = list(detector_map.values())
    label = "+".join(names)
    m = await eval_ensemble(all_dets, samples, manifest)
    m["strategy"] = "ensemble-all"
    m["name"] = f"ALL({len(all_dets)})"
    m["domain"] = domain
    m["detector_count"] = len(all_dets)
    results.append(m)
    print(f"    {m['name']:35s} | F1={m['f1']:.3f} | P={m['precision']:.3f} | R={m['recall']:.3f} | {m['avg_ms']:.1f}ms")

    # === Summary ===
    print(f"\n  --- BEST STRATEGY ---")
    best = max(results, key=lambda x: x["f1"])
    print(f"    {best['name']} (F1={best['f1']:.3f}, {best['strategy']})")

    return results


async def main(domains):
    all_results = []
    for domain in domains:
        domain_results = await run_domain(domain)
        all_results.extend(domain_results)

    # === Global summary ===
    print(f"\n{'='*70}")
    print(f"  GLOBAL SUMMARY — BEST STRATEGY PER DOMAIN")
    print(f"{'='*70}\n")

    print("| Domain | Best Strategy | F1 | Precision | Recall | Detectors | Latency |")
    print("|--------|--------------|-----|-----------|--------|-----------|---------|")

    for domain in domains:
        dr = [r for r in all_results if r["domain"] == domain]
        best = max(dr, key=lambda x: x["f1"])
        print(
            f"| {domain:14s} | {best['name']:25s} | {best['f1']:.3f} | "
            f"{best['precision']:.3f}     | {best['recall']:.3f}  | "
            f"{best['detector_count']}         | {best['avg_ms']:.1f}ms   |"
        )

    # Best single vs best ensemble comparison
    print(f"\n{'='*70}")
    print(f"  SINGLE vs ENSEMBLE COMPARISON")
    print(f"{'='*70}\n")

    print("| Domain | Best Single | F1 | Best Ensemble | F1 | Delta |")
    print("|--------|-------------|-----|---------------|-----|-------|")

    for domain in domains:
        dr = [r for r in all_results if r["domain"] == domain]
        best_single = max([r for r in dr if r["strategy"] == "single"], key=lambda x: x["f1"])
        best_ens = max([r for r in dr if r["strategy"] != "single"], key=lambda x: x["f1"])
        delta = best_ens["f1"] - best_single["f1"]
        sign = "+" if delta >= 0 else ""
        print(
            f"| {domain:14s} | {best_single['name']:11s} | {best_single['f1']:.3f} | "
            f"{best_ens['name']:20s} | {best_ens['f1']:.3f} | {sign}{delta:.3f} |"
        )

    # NegSel vs others
    print(f"\n{'='*70}")
    print(f"  NEGSEL CONTRIBUTION")
    print(f"{'='*70}\n")

    for domain in domains:
        dr = [r for r in all_results if r["domain"] == domain]
        negsel = [r for r in dr if r["strategy"] == "single" and r["name"] == "NegSel"]
        negsel_ensembles = [r for r in dr if "NegSel" in r["name"] and r["strategy"] != "single"]
        non_negsel_ensembles = [r for r in dr if "NegSel" not in r["name"] and r["strategy"] != "single"]

        if negsel:
            ns = negsel[0]
            best_with = max(negsel_ensembles, key=lambda x: x["f1"]) if negsel_ensembles else None
            best_without = max(non_negsel_ensembles, key=lambda x: x["f1"]) if non_negsel_ensembles else None

            print(f"  {domain}:")
            print(f"    NegSel solo:          F1={ns['f1']:.3f}")
            if best_with:
                print(f"    Best w/ NegSel:       F1={best_with['f1']:.3f} ({best_with['name']})")
            if best_without:
                print(f"    Best w/o NegSel:      F1={best_without['f1']:.3f} ({best_without['name']})")
            if best_with and best_without:
                delta = best_with["f1"] - best_without["f1"]
                print(f"    NegSel adds:          {'+' if delta >= 0 else ''}{delta:.3f}")
            print()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default=None, help="Single domain to test")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else DOMAINS
    results = asyncio.run(main(domains))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
