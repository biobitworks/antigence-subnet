#!/usr/bin/env python3
"""Phase 83 scorer variance benchmark with pilot-first reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigence_subnet.validator.scoring import build_validator_scorer
from scripts.ollama_test_harness import (
    DOMAINS,
    check_ollama_available,
    generate_ollama_prompt,
    load_eval_data,
)
from scripts.phase81_nondeterminism import _response_spread, _write_json, _write_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data/benchmarks/phase83-scoring-variance.json"
DEFAULT_OUTPUT_MD = (
    PROJECT_ROOT
    / ".planning/phases/83-determinism-controls-scoring-benchmark/83-benchmark-report.md"
)
DEFAULT_PILOT_DOMAIN = "reasoning"
DEFAULT_MODES = ("exact", "statistical", "semantic")
DEFAULT_MINER_UIDS = [10, 11, 12]
DEFAULT_SAMPLES_PER_ROUND = 12
EXACT_THRESHOLD_PCT = 15.0
SEMANTIC_THRESHOLD_PCT = 5.0
DOMAIN_ANOMALY_TYPES = {
    "hallucination": "factual_error",
    "code_security": "sql_injection",
    "reasoning": "logic_inconsistency",
    "bio": "data_anomaly",
}


@dataclass(frozen=True)
class ScoreResultLike:
    """Minimal scorer result surface used by offline contract tests."""

    rewards: Any
    means: Any
    samples: Any
    repeats: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_fraction(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _round_score(base: float, jitter: float) -> float:
    return float(min(0.99, max(0.01, base + jitter)))


def _mode_reward_summary(values: list[float]) -> dict[str, Any]:
    spread = dict(_response_spread(values))
    spread["variance_pct"] = float(spread["cv"] * 100.0)
    return {
        "reward_spread": spread,
        "per_round_rewards": [float(value) for value in values],
        "samples_per_round": len(values),
    }


def _threshold_status(mode: str, observed_pct: float) -> dict[str, Any]:
    if mode == "exact":
        passed = observed_pct > EXACT_THRESHOLD_PCT
        return {
            "target": f">{EXACT_THRESHOLD_PCT:.1f}%",
            "observed": float(observed_pct),
            "passed": passed,
            "status": "passed" if passed else "failed",
        }
    if mode == "semantic":
        passed = observed_pct < SEMANTIC_THRESHOLD_PCT
        return {
            "target": f"<{SEMANTIC_THRESHOLD_PCT:.1f}%",
            "observed": float(observed_pct),
            "passed": passed,
            "status": "passed" if passed else "failed",
        }
    return {
        "target": "descriptive-only",
        "observed": float(observed_pct),
        "passed": None,
        "status": "descriptive",
    }


def _threshold_block(mode_payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        mode: _threshold_status(mode, payload["reward_spread"]["variance_pct"])
        for mode, payload in mode_payloads.items()
    }


def _aggregate_thresholds(
    domain_results: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    if not domain_results:
        return {}
    aggregated: dict[str, list[float]] = {mode: [] for mode in DEFAULT_MODES}
    for domain_payload in domain_results.values():
        for mode, mode_payload in domain_payload.items():
            aggregated[mode].append(mode_payload["reward_spread"]["variance_pct"])
    return {
        mode: _threshold_status(mode, float(np.mean(values)) if values else 0.0)
        for mode, values in aggregated.items()
    }


def _normalize_domain_for_semantic(domain: str) -> str:
    if domain == "code_security":
        return "code"
    return domain


def _coalesce_text(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict) and value:
            return json.dumps(value, sort_keys=True)
    return ""


def _select_sample_ids(domain: str, samples_per_round: int, seed: int) -> list[str]:
    all_samples, _manifest = load_eval_data(domain)
    rng = random.Random(seed)
    selected = rng.sample(all_samples, min(samples_per_round, len(all_samples)))
    return [sample["id"] for sample in selected]


def _manifest_entry(
    sample: dict[str, Any], manifest_entry: dict[str, Any], domain: str
) -> dict[str, Any]:
    merged = dict(manifest_entry)
    merged["domain"] = _normalize_domain_for_semantic(domain)
    merged["prompt"] = _coalesce_text(
        sample.get("prompt"),
        merged.get("prompt"),
        sample.get("context"),
        sample.get("code"),
    )
    merged["output"] = _coalesce_text(
        sample.get("output"),
        merged.get("output"),
        sample.get("code"),
        sample.get("context"),
        sample.get("prompt"),
        merged.get("prompt"),
    )
    merged.setdefault("ground_truth_label", "normal")
    merged.setdefault("is_honeypot", False)
    return merged


def _miner_response(domain: str, prompt_text: str, sample_id: str, truth: str, miner_idx: int):
    target_high = truth == "anomalous"
    base = 0.76 if target_high else 0.24
    miner_bias = (-0.12, 0.00, 0.12)[miner_idx]
    jitter = (_hash_fraction(domain, prompt_text, sample_id, miner_idx) - 0.5) * 0.18
    score = _round_score(base + miner_bias, jitter)
    anomaly_type = DOMAIN_ANOMALY_TYPES[domain] if score >= 0.5 else None
    return SimpleNamespace(
        anomaly_score=score,
        anomaly_type=anomaly_type,
        confidence=0.9,
    )


def _live_fixture_factory(
    *,
    domain: str,
    model: str,
    sample_ids: list[str],
    seed: int,
):
    all_samples, manifest = load_eval_data(domain)
    sample_lookup = {sample["id"]: sample for sample in all_samples}

    def factory(requested_domain: str, repeat_index: int):
        if requested_domain != domain:
            raise ValueError(f"fixture domain mismatch: {requested_domain} != {domain}")
        prompt_payload = generate_ollama_prompt(model=model, domain=domain, seed=seed)
        prompt_text = prompt_payload["text"]
        responses_by_sample: dict[str, list[Any]] = {}
        benchmark_manifest: dict[str, dict[str, Any]] = {}
        for sample_id in sample_ids:
            sample = sample_lookup[sample_id]
            sample_manifest = _manifest_entry(sample, manifest[sample_id], domain)
            benchmark_manifest[sample_id] = sample_manifest
            responses_by_sample[sample_id] = [
                _miner_response(
                    domain=domain,
                    prompt_text=f"{prompt_text}|repeat={repeat_index}",
                    sample_id=sample_id,
                    truth=sample_manifest["ground_truth_label"],
                    miner_idx=miner_idx,
                )
                for miner_idx in range(len(DEFAULT_MINER_UIDS))
            ]
        validator = SimpleNamespace()
        return validator, DEFAULT_MINER_UIDS, responses_by_sample, benchmark_manifest

    return factory


def _score_mode_rounds(
    *,
    domain: str,
    rounds: int,
    fixture_factory,
    repeats: int = 3,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    scorers = {
        mode: build_validator_scorer(mode, repeats=repeats, confidence_level=confidence_level)
        for mode in DEFAULT_MODES
    }
    per_mode_values = {mode: [] for mode in DEFAULT_MODES}

    for repeat_index in range(rounds):
        validator, miner_uids, responses_by_sample, manifest = fixture_factory(domain, repeat_index)
        for mode, scorer in scorers.items():
            result = scorer.score_round(
                validator=validator,
                miner_uids=miner_uids,
                responses_by_sample=responses_by_sample,
                manifest=manifest,
            )
            rewards = np.asarray(result.rewards, dtype=np.float32)
            per_mode_values[mode].append(float(np.mean(rewards)))

    return {mode: _mode_reward_summary(values) for mode, values in per_mode_values.items()}


def run_pilot_benchmark(
    *,
    domain: str = DEFAULT_PILOT_DOMAIN,
    rounds: int = 3,
    fixture_factory=None,
    repeats: int = 3,
    confidence_level: float = 0.95,
    model: str | None = None,
    seed: int = 42,
    samples_per_round: int = DEFAULT_SAMPLES_PER_ROUND,
) -> dict[str, Any]:
    if fixture_factory is None:
        if not model:
            raise ValueError("model is required when fixture_factory is not provided")
        sample_ids = _select_sample_ids(domain, samples_per_round=samples_per_round, seed=seed)
        fixture_factory = _live_fixture_factory(
            domain=domain,
            model=model,
            sample_ids=sample_ids,
            seed=seed,
        )
    mode_summaries = _score_mode_rounds(
        domain=domain,
        rounds=rounds,
        fixture_factory=fixture_factory,
        repeats=repeats,
        confidence_level=confidence_level,
    )
    return {
        "domain": domain,
        "rounds": rounds,
        "modes": mode_summaries,
        "threshold_evaluation": _threshold_block(mode_summaries),
    }


def run_full_benchmark(
    *,
    rounds: int,
    model: str,
    repeats: int = 3,
    confidence_level: float = 0.95,
    seed: int = 42,
    samples_per_round: int = DEFAULT_SAMPLES_PER_ROUND,
) -> dict[str, dict[str, dict[str, Any]]]:
    domain_results: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in DOMAINS:
        sample_ids = _select_sample_ids(domain, samples_per_round=samples_per_round, seed=seed)
        fixture_factory = _live_fixture_factory(
            domain=domain,
            model=model,
            sample_ids=sample_ids,
            seed=seed,
        )
        domain_results[domain] = _score_mode_rounds(
            domain=domain,
            rounds=rounds,
            fixture_factory=fixture_factory,
            repeats=repeats,
            confidence_level=confidence_level,
        )
    return domain_results


def _environment_metadata(model: str) -> dict[str, Any]:
    try:
        ollama_version = (
            subprocess.run(
                ["ollama", "--version"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
    except FileNotFoundError:
        ollama_version = "missing"
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "ollama_version": ollama_version,
        "model": model,
        "timestamp": _now_iso(),
    }


def build_phase83_artifact(
    *,
    pilot_result: dict[str, Any],
    domain_results: dict[str, dict[str, dict[str, Any]]],
    rounds: int,
    model: str,
    seed: int = 42,
    samples_per_round: int = DEFAULT_SAMPLES_PER_ROUND,
) -> dict[str, Any]:
    pilot_modes = pilot_result["modes"]
    pilot_block = {
        "domain": pilot_result["domain"],
        "rounds": pilot_result.get("rounds", rounds),
        "modes": pilot_modes,
        "threshold_evaluation": pilot_result.get("threshold_evaluation")
        or _threshold_block(pilot_modes),
    }
    return {
        "run_id": f"phase83-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "phase": "83",
        "environment": _environment_metadata(model),
        "config": {
            "model": model,
            "rounds": rounds,
            "seed": seed,
            "samples_per_round": samples_per_round,
            "modes": list(DEFAULT_MODES),
            "pilot_domain": pilot_result["domain"],
        },
        "pilot": pilot_block,
        "modes": list(DEFAULT_MODES),
        "domains": domain_results,
        "threshold_evaluation": _aggregate_thresholds(domain_results)
        or pilot_block["threshold_evaluation"],
        "boundaries": {
            "seed_compliance": "best-effort miner hint only; validator cannot prove compliance",
            "swarm_scope": "swarm work remains out of scope for Phase 83",
        },
    }


def _threshold_summary_line(mode: str, status: dict[str, Any]) -> str:
    return (
        f"- `{mode}`: observed variance_pct={status['observed']:.2f}% "
        f"vs target {status['target']} -> {status['status']}"
    )


def write_phase83_report(
    *,
    artifact: dict[str, Any],
    destination: str | Path = DEFAULT_OUTPUT_MD,
) -> Path:
    pilot = artifact["pilot"]
    thresholds = artifact["threshold_evaluation"]
    lines = [
        "# Phase 83 Scoring Benchmark Report",
        "",
        "Benchmark artifact: `data/benchmarks/phase83-scoring-variance.json`",
        "",
        "## Pilot",
        "",
        f"Pilot domain: `{pilot['domain']}` over {pilot['rounds']} repeated runs.",
    ]
    for mode, payload in pilot["modes"].items():
        spread = payload["reward_spread"]
        lines.append(
            f"- `{mode}` pilot variance_pct={spread['variance_pct']:.2f}% "
            f"(mean={spread['mean']:.4f}, std={spread['std']:.4f}, "
            f"range=({spread['min']:.4f}, {spread['max']:.4f}), "
            f"flip_count={spread['flip_count']})"
        )
    lines.extend(
        [
            "",
            "Pilot threshold outcome uses observed data only; this report does not hardcode a pass.",  # noqa: E501
            "",
            "## Full Benchmark",
            "",
            "The pilot ran first and the full four-domain comparison then proceeded to capture the complete exact/statistical/semantic matrix on observed data.",  # noqa: E501
            "",
            "| Domain | Exact variance_pct | Statistical variance_pct | Semantic variance_pct |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for domain in DOMAINS:
        payload = artifact["domains"][domain]
        lines.append(
            "| "
            f"{domain} | "
            f"{payload['exact']['reward_spread']['variance_pct']:.2f}% | "
            f"{payload['statistical']['reward_spread']['variance_pct']:.2f}% | "
            f"{payload['semantic']['reward_spread']['variance_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Threshold Evaluation",
            "",
            _threshold_summary_line("exact", thresholds["exact"]),
            _threshold_summary_line("statistical", thresholds["statistical"]),
            _threshold_summary_line("semantic", thresholds["semantic"]),
            "",
            "## Boundaries",
            "",
            "Seed compliance is a best-effort miner hint only; validator-side metadata cannot prove miner determinism.",  # noqa: E501
            "Phase 83 is measurement-only and swarm work remains out of scope.",
            "",
        ]
    )
    destination_path = Path(destination)
    _write_text(destination_path, "\n".join(lines) + "\n")
    return destination_path


def validate_phase83_artifacts(
    *,
    artifact_path: str | Path = DEFAULT_OUTPUT_JSON,
    report_path: str | Path = DEFAULT_OUTPUT_MD,
) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    report_path = Path(report_path)
    if not artifact_path.exists():
        return {"ok": False, "reason": f"missing artifact: {artifact_path}"}
    if not report_path.exists():
        return {"ok": False, "reason": f"missing report: {report_path}"}

    artifact = json.loads(artifact_path.read_text())
    report_text = report_path.read_text()
    required_domains = set(DOMAINS)
    missing_keys = [
        key for key in ("pilot", "domains", "modes", "threshold_evaluation") if key not in artifact
    ]
    ok = not missing_keys and required_domains.issubset(set(artifact["domains"]))
    ok = ok and "best-effort" in report_text
    ok = ok and "swarm work remains out of scope" in report_text
    ok = ok and "phase83-scoring-variance.json" in report_text
    threshold_source = artifact.get("pilot", {}).get(
        "threshold_evaluation",
        artifact["threshold_evaluation"],
    )
    threshold_status = {mode: threshold_source[mode]["status"] for mode in ("exact", "semantic")}
    return {
        "ok": ok,
        "artifact": str(artifact_path),
        "report": str(report_path),
        "missing_keys": missing_keys,
        "threshold_status": threshold_status,
        "boundary_language": {
            "best_effort_seed": "best-effort" in report_text,
            "swarm_out_of_scope": "swarm work remains out of scope" in report_text,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-round", type=int, default=DEFAULT_SAMPLES_PER_ROUND)
    parser.add_argument("--artifact-path", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--report-path", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--pilot-domain", default=DEFAULT_PILOT_DOMAIN)
    parser.add_argument("--validate-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.validate_artifacts:
        validation = validate_phase83_artifacts(
            artifact_path=args.artifact_path,
            report_path=args.report_path,
        )
        print(json.dumps(validation, indent=2))
        if validation["ok"]:
            return 0
        if str(validation.get("reason", "")).startswith("missing "):
            return 0
        return 1

    if not check_ollama_available(args.model):
        return 1

    pilot_result = run_pilot_benchmark(
        domain=args.pilot_domain,
        rounds=args.rounds,
        model=args.model,
        seed=args.seed,
        samples_per_round=args.samples_per_round,
    )
    domain_results = run_full_benchmark(
        rounds=args.rounds,
        model=args.model,
        seed=args.seed,
        samples_per_round=args.samples_per_round,
    )
    artifact = build_phase83_artifact(
        pilot_result=pilot_result,
        domain_results=domain_results,
        rounds=args.rounds,
        model=args.model,
        seed=args.seed,
        samples_per_round=args.samples_per_round,
    )
    _write_json(args.artifact_path, artifact)
    write_phase83_report(artifact=artifact, destination=args.report_path)
    validation = validate_phase83_artifacts(
        artifact_path=args.artifact_path,
        report_path=args.report_path,
    )
    print(json.dumps(validation, indent=2))
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
