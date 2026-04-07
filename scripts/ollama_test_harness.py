#!/usr/bin/env python3
"""
Ollama Test Harness — multi-round eval simulation pipeline.

Uses a local Ollama LLM to generate prompts, runs miner anomaly detection
on evaluation samples, and scores results with the validator reward function.
Produces F1/precision/recall/accuracy/reward metrics and latency data.

Runs N rounds unattended, aggregates per-round metrics into summary
statistics (avg/std), and writes timestamped JSON reports to data/benchmarks/.

This is the fundamental building block for E2E testing without testnet.
Every subsequent phase (80+ baselines, 81+ non-determinism) depends on
this harness producing reliable metrics.

Usage:
    python scripts/ollama_test_harness.py
    python scripts/ollama_test_harness.py --model qwen2.5:1.5b --rounds 50
    python scripts/ollama_test_harness.py --detector IsolationForest --samples-per-round 20
    python scripts/ollama_test_harness.py --output data/benchmarks/custom_run.json
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add project root to path (same pattern as benchmark_detectors.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama

from antigence_subnet.miner.data import load_training_samples
from antigence_subnet.miner.detectors import (
    BioDetector,
    CodeSecurityDetector,
    HallucinationDetector,
    ReasoningDetector,
)
from antigence_subnet.miner.detectors.fractal_complexity import FractalComplexityDetector
from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector
from antigence_subnet.miner.detectors.negsel import NegSelAISDetector
from antigence_subnet.miner.detectors.sklearn_backends import LOFDetector, OCSVMDetector

# NOTE: compute_reward import is deferred to run_single_round() to avoid
# bittensor hijacking argparse at import time. Importing bittensor at
# module level prevents our CLI argparse from working correctly.

try:
    from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector
except ImportError:
    AutoencoderDetector = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"
BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"
DOMAINS = ["hallucination", "code_security", "reasoning", "bio"]

# Generic detectors work on any domain via TF-IDF features
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

# Combined registry for CLI lookup
DETECTOR_REGISTRY = {}
DETECTOR_REGISTRY.update(GENERIC_DETECTORS)
for domain_dets in DOMAIN_DETECTORS.values():
    DETECTOR_REGISTRY.update(domain_dets)

# Prompt templates per domain for Ollama generation
DOMAIN_PROMPTS = {
    "hallucination": (
        "Generate a factual question and a plausible but INCORRECT answer. "
        "Format as 'Q: [question]\\nA: [wrong answer]'. "
        "Make the wrong answer sound convincing."
    ),
    "code_security": (
        "Generate a short Python code snippet that contains a subtle "
        "security vulnerability. Include a brief description of what "
        "the code does, but do NOT mention the vulnerability."
    ),
    "reasoning": (
        "Generate a logical reasoning problem with an INCORRECT conclusion. "
        "Present premises that sound logical but lead to a wrong answer. "
        "Make the flawed reasoning subtle."
    ),
    "bio": (
        "Generate a biology claim that sounds plausible but is factually WRONG. "
        "Include specific biological terms and mechanisms to make it convincing."
    ),
}

DECISION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def check_ollama_available(model: str) -> bool:
    """Check if Ollama server is running and the requested model is available.

    Args:
        model: Model name to check for (e.g., "qwen2.5:1.5b").

    Returns:
        True if server is up and model is pulled, False otherwise.
        Prints actionable error messages on failure.
    """
    try:
        response = ollama.list()
    except (ConnectionError, Exception) as e:
        print(f"Ollama not available: {e}")
        print("Start with: ollama serve")
        return False

    # Check if model is in the available list
    # Models may be listed with or without ":latest" suffix
    available_models = []
    for m in response.models:
        name = m.model
        available_models.append(name)
        # Also track without :latest suffix
        if name.endswith(":latest"):
            available_models.append(name[: -len(":latest")])

    if model not in available_models and f"{model}:latest" not in available_models:
        print(f"Model '{model}' not found. Available models: {available_models}")
        print(f"Run: ollama pull {model}")
        return False

    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_data(domain: str) -> tuple[list[dict], dict]:
    """Load all samples and ground truth manifest for a domain.

    Args:
        domain: Evaluation domain (e.g., "hallucination").

    Returns:
        Tuple of (samples_list, manifest_dict).
    """
    samples_path = DATA_DIR / domain / "samples.json"
    manifest_path = DATA_DIR / domain / "manifest.json"

    with open(samples_path) as f:
        all_samples = json.load(f)["samples"]
    with open(manifest_path) as f:
        manifest = json.load(f)

    return all_samples, manifest


# ---------------------------------------------------------------------------
# Ollama prompt generation
# ---------------------------------------------------------------------------


def generate_ollama_prompt(model: str, domain: str, seed: int) -> dict:
    """Generate a prompt+output pair using Ollama.

    Calls ollama.chat with a domain-specific prompt template.
    Ollama durations are in NANOSECONDS -- divide by 1_000_000 for ms.

    Args:
        model: Ollama model name.
        domain: Domain for prompt template selection.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys: text, latency_ms, eval_ms, tokens.
    """
    prompt_text = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["hallucination"])

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        options={"temperature": 0.8, "seed": seed, "num_predict": 256},
    )

    return {
        "text": response.message.content,
        # Ollama durations are nanoseconds -> convert to milliseconds
        "latency_ms": (response.total_duration or 0) / 1_000_000,
        "eval_ms": (response.eval_duration or 0) / 1_000_000,
        "tokens": response.eval_count or 0,
    }


# ---------------------------------------------------------------------------
# Single-round pipeline
# ---------------------------------------------------------------------------


async def run_single_round(
    domain: str,
    detector_name: str,
    model: str,
    samples_per_round: int = 20,
    warmup: bool = True,
    seed: int = 42,
) -> dict:
    """Run a single evaluation round: generate -> detect -> score.

    1. Optionally warm up Ollama (loads model into memory)
    2. Generate prompt+output pair using Ollama
    3. Load evaluation data and select samples (seeded)
    4. Instantiate and fit detector on normal samples
    5. Run async detection on each sample
    6. Score with compute_reward and manual TP/FP/FN/TN metrics

    Args:
        domain: Evaluation domain.
        detector_name: Detector class name from DETECTOR_REGISTRY.
        model: Ollama model name.
        samples_per_round: Number of eval samples to use.
        warmup: If True, warm up Ollama before timed generation.
        seed: RNG seed for sample selection and Ollama.

    Returns:
        Structured dict with round, domain, detector, model, metrics,
        latency, samples_evaluated, and seed.
    """
    round_start = time.perf_counter()

    # 1. Optional warmup: load model into Ollama memory
    if warmup:
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "warmup"}],
            options={"num_predict": 1},
        )

    # 2. Generate prompt using Ollama
    ollama_result = generate_ollama_prompt(model, domain, seed)
    ollama_generate_ms = ollama_result["latency_ms"]

    # 3. Load evaluation data
    all_samples, manifest = load_eval_data(domain)
    normal_samples = load_training_samples(str(DATA_DIR), domain)

    # 4. Select samples (seeded for reproducibility)
    rng = random.Random(seed)
    n_select = min(samples_per_round, len(all_samples))
    selected_samples = rng.sample(all_samples, n_select)

    # 5. Instantiate and fit detector
    if detector_name in GENERIC_DETECTORS:
        det_cls = GENERIC_DETECTORS[detector_name]
    elif detector_name in DETECTOR_REGISTRY:
        det_cls = DETECTOR_REGISTRY[detector_name]
    else:
        raise ValueError(
            f"Unknown detector '{detector_name}'. Available: {list(DETECTOR_REGISTRY.keys())}"
        )

    detector = det_cls()
    # Set domain for generic detectors
    if detector_name in GENERIC_DETECTORS:
        detector.domain = domain
    detector.fit(normal_samples)

    # 6. Run detection on each selected sample
    anomaly_scores = []
    ground_truths = []
    detection_latencies = []

    for sample in selected_samples:
        t0 = time.perf_counter()
        try:
            result = await detector.detect(
                prompt=sample.get("prompt", ""),
                output=sample.get("output", ""),
                code=sample.get("code"),
                context=sample.get("context"),
            )
            latency = time.perf_counter() - t0
            detection_latencies.append(latency)
            anomaly_scores.append(result.score)
        except Exception:
            latency = time.perf_counter() - t0
            detection_latencies.append(latency)
            anomaly_scores.append(0.0)

        # Get ground truth from manifest
        gt_label = manifest.get(sample["id"], {}).get("ground_truth_label", "normal")
        ground_truths.append(gt_label)

    # 7. Score with compute_reward (deferred import to avoid bittensor argparse hijack)
    from antigence_subnet.validator.reward import compute_reward

    reward = compute_reward(anomaly_scores, ground_truths)

    # Manual TP/FP/FN/TN for F1/precision/recall/accuracy
    tp = fp = fn = tn = 0
    for score, truth in zip(anomaly_scores, ground_truths, strict=False):
        predicted_anomalous = score >= DECISION_THRESHOLD
        actually_anomalous = truth == "anomalous"
        if predicted_anomalous and actually_anomalous:
            tp += 1
        elif predicted_anomalous and not actually_anomalous:
            fp += 1
        elif not predicted_anomalous and actually_anomalous:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(selected_samples) if selected_samples else 0.0

    # Latency computation
    detection_total_ms = sum(detection_latencies) * 1000
    detection_avg_ms = (
        (detection_total_ms / len(detection_latencies)) if detection_latencies else 0.0
    )
    round_total_ms = (time.perf_counter() - round_start) * 1000

    # 8. Structured result
    return {
        "round": 1,
        "domain": domain,
        "detector": detector_name,
        "model": model,
        "metrics": {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "reward": reward,
        },
        "latency": {
            "ollama_generate_ms": ollama_generate_ms,
            "detection_avg_ms": detection_avg_ms,
            "detection_total_ms": detection_total_ms,
            "round_total_ms": round_total_ms,
        },
        "samples_evaluated": len(selected_samples),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Multi-round runner
# ---------------------------------------------------------------------------


async def run_multi_round(
    domain: str,
    detector_name: str,
    model: str,
    rounds: int = 10,
    samples_per_round: int = 20,
    warmup: bool = True,
    seed: int = 42,
) -> dict:
    """Run multiple evaluation rounds and aggregate statistics.

    Executes run_single_round N times with incrementing seeds, collects
    all per-round results, computes summary statistics (avg/std), and
    returns a full report dict ready for JSON serialization.

    Args:
        domain: Evaluation domain.
        detector_name: Detector class name from DETECTOR_REGISTRY.
        model: Ollama model name.
        rounds: Number of evaluation rounds to run.
        samples_per_round: Number of eval samples per round.
        warmup: If True, warm up Ollama before first round.
        seed: Base RNG seed (incremented per round).

    Returns:
        Full report dict with harness_version, timestamp, config, rounds, summary.
    """
    loop_start = time.perf_counter()

    round_results = []
    for i in range(rounds):
        # Warmup only on the first round
        do_warmup = warmup and (i == 0)
        round_seed = seed + i

        result = await run_single_round(
            domain=domain,
            detector_name=detector_name,
            model=model,
            samples_per_round=samples_per_round,
            warmup=do_warmup,
            seed=round_seed,
        )

        # Update round number and seed in the result
        result["round"] = i + 1
        result["seed"] = round_seed
        round_results.append(result)

    total_time = time.perf_counter() - loop_start

    # Compute summary statistics with numpy
    f1_scores = [r["metrics"]["f1"] for r in round_results]
    precision_scores = [r["metrics"]["precision"] for r in round_results]
    recall_scores = [r["metrics"]["recall"] for r in round_results]
    reward_scores = [r["metrics"]["reward"] for r in round_results]
    ollama_latencies = [r["latency"]["ollama_generate_ms"] for r in round_results]
    detection_latencies = [r["latency"]["detection_avg_ms"] for r in round_results]

    summary = {
        "avg_f1": float(np.mean(f1_scores)),
        "avg_precision": float(np.mean(precision_scores)),
        "avg_recall": float(np.mean(recall_scores)),
        "std_f1": float(np.std(f1_scores)),
        "std_precision": float(np.std(precision_scores)),
        "std_recall": float(np.std(recall_scores)),
        "avg_reward": float(np.mean(reward_scores)),
        "std_reward": float(np.std(reward_scores)),
        "avg_ollama_latency_ms": float(np.mean(ollama_latencies)),
        "avg_detection_latency_ms": float(np.mean(detection_latencies)),
        "total_rounds": rounds,
        "total_samples": sum(r["samples_evaluated"] for r in round_results),
        "total_time_s": float(total_time),
    }

    return {
        "harness_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": model,
            "detector": detector_name,
            "domain": domain,
            "rounds": rounds,
            "samples_per_round": samples_per_round,
        },
        "rounds": round_results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# JSON reporting
# ---------------------------------------------------------------------------


def write_report(report: dict, output_path: str | None) -> str:
    """Write report dict to JSON file.

    Args:
        report: Full report dict from run_multi_round.
        output_path: File path to write. If None, auto-generates a
            timestamped filename in BENCHMARKS_DIR.

    Returns:
        The path the report was written to.
    """
    if output_path is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ollama_harness_{timestamp_str}.json"
        output_path = str(BENCHMARKS_DIR / filename)

    # Create parent directory if needed
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def print_summary(report: dict) -> None:
    """Print human-readable summary to stdout.

    Args:
        report: Full report dict from run_multi_round.
    """
    config = report["config"]
    summary = report["summary"]

    print("=" * 48)
    print("  OLLAMA TEST HARNESS - SUMMARY")
    print("=" * 48)
    print(f"  Model:    {config['model']}")
    print(f"  Detector: {config['detector']}")
    print(f"  Domain:   {config['domain']}")
    print(f"  Rounds:   {config['rounds']}")
    print("-" * 48)
    print(f"  F1:        {summary['avg_f1']:.3f} +/- {summary['std_f1']:.3f}")
    print(f"  Precision: {summary['avg_precision']:.3f} +/- {summary['std_precision']:.3f}")
    print(f"  Recall:    {summary['avg_recall']:.3f} +/- {summary['std_recall']:.3f}")
    print(f"  Reward:    {summary['avg_reward']:.3f} +/- {summary['std_reward']:.3f}")
    print("-" * 48)
    print(f"  Ollama latency:    {summary['avg_ollama_latency_ms']:.1f} ms/request")
    print(f"  Detection latency:   {summary['avg_detection_latency_ms']:.1f} ms/sample")
    print(f"  Total time: {summary['total_time_s']:.1f}s")
    print(f"  Total samples: {summary['total_samples']}")
    print("=" * 48)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the Ollama test harness."""
    parser = argparse.ArgumentParser(
        description="Ollama Test Harness -- multi-round eval simulation pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:1.5b",
        help="Ollama model name (default: qwen2.5:1.5b)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of evaluation rounds (default: 10)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="hallucination",
        choices=DOMAINS,
        help="Evaluation domain (default: hallucination)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="IsolationForest",
        help="Detector name from registry (default: IsolationForest)",
    )
    parser.add_argument(
        "--samples-per-round",
        type=int,
        default=20,
        help="Number of eval samples per round (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (default: auto-generated in data/benchmarks/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip Ollama warmup (model may not be preloaded)",
    )
    args = parser.parse_args()

    # Pre-flight check
    if not check_ollama_available(args.model):
        sys.exit(1)

    # Run multi-round evaluation
    report = asyncio.run(
        run_multi_round(
            domain=args.domain,
            detector_name=args.detector,
            model=args.model,
            rounds=args.rounds,
            samples_per_round=args.samples_per_round,
            warmup=not args.no_warmup,
            seed=args.seed,
        )
    )

    # Print human-readable summary
    print_summary(report)

    # Write JSON report (auto-generates path if --output not specified)
    path = write_report(report, args.output)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
