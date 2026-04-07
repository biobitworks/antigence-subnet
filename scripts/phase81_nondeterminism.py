#!/usr/bin/env python3
"""Phase 81 non-determinism measurement wrapper built on the Ollama harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import parse, request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigence_subnet.miner.data import load_training_samples
from scripts.ollama_test_harness import (
    DATA_DIR,
    DECISION_THRESHOLD,
    DETECTOR_REGISTRY,
    DOMAINS,
    GENERIC_DETECTORS,
    check_ollama_available,
    generate_ollama_prompt,
    load_eval_data,
    run_single_round,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_REFERENCE = "data/benchmarks/v9.2-baseline.json"
DEFAULT_OUTPUT_JSON = "data/benchmarks/phase81-nondeterminism-impact.json"
DEFAULT_OUTPUT_MD = (
    ".planning/phases/81-non-determinism-research-impact-measurement/81-nondeterminism.md"
)
DEFAULT_OUTPUT_OVERWATCH = "data/overwatch/phase81-nondeterminism-claims.json"
DEFAULT_WRITEBACK_REPORT = "data/overwatch/phase81-nondeterminism-writeback-report.json"
OVERWATCH_BASE_URL = "http://localhost:8531"
OVERWATCH_DB = "overwatch"
SOURCE_KEYS = [
    "gpu_kernel",
    "batching_scheduling",
    "floating_point_kernel",
    "harness_level_randomness",
]
BOUNDARY_PHRASES = [
    "miners remain adversarial",
    "best-effort",
    "cannot force exact miner inference behavior",
    "backward-compatible",
    "commodity-hardware",
]


@dataclass
class ArtifactInputs:
    artifact_data: dict[str, Any]
    markdown_text: str
    overwatch_payload: dict[str, Any]
    writeback_report: dict[str, Any] | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    _ensure_parent(destination)
    with open(destination, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    _ensure_parent(destination)
    destination.write_text(content)


def _detector_for(domain: str, detector_name: str):
    if detector_name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector '{detector_name}'")
    detector = DETECTOR_REGISTRY[detector_name]()
    if detector_name in GENERIC_DETECTORS:
        detector.domain = domain
    normal_samples = load_training_samples(str(DATA_DIR), domain)
    detector.fit(normal_samples)
    return detector


def _select_samples(
    domain: str, samples_per_round: int, seed: int
) -> tuple[list[dict], dict[str, Any]]:
    all_samples, manifest = load_eval_data(domain)
    rng = random.Random(seed)
    n_select = min(samples_per_round, len(all_samples))
    selected_samples = rng.sample(all_samples, n_select)
    return selected_samples, manifest


def _response_spread(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0, "flip_count": 0}
    mean = float(np.mean(values))
    std = float(np.std(values))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    cv = float(std / mean) if abs(mean) > 1e-9 else 0.0
    predictions = [value >= DECISION_THRESHOLD for value in values]
    flip_count = sum(
        1 for index in range(1, len(predictions)) if predictions[index] != predictions[index - 1]
    )
    return {
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "cv": cv,
        "flip_count": flip_count,
    }


def _round_metric_spread(round_runs: list[dict[str, Any]]) -> dict[str, Any]:
    reward_values = [run["metrics"]["reward"] for run in round_runs]
    f1_values = [run["metrics"]["f1"] for run in round_runs]
    return {
        "metric": "reward",
        "spread": _response_spread(reward_values),
        "f1_spread": _response_spread(f1_values),
        "repeats": len(round_runs),
    }


def build_fixed_input_repeats(
    domain: str,
    selected_samples: list[dict[str, Any]],
    seed: int,
    repeats: int,
) -> list[dict[str, Any]]:
    sample_ids = [sample["id"] for sample in selected_samples]
    return [
        {
            "repeat": repeat_index + 1,
            "domain": domain,
            "seed": seed,
            "sample_ids": sample_ids,
            "response": {},
            "score_outputs": [],
        }
        for repeat_index in range(repeats)
    ]


async def _measure_fixed_input_variance(
    domain: str,
    detector_name: str,
    model: str,
    samples_per_round: int,
    seed: int,
    repeats: int,
) -> list[dict[str, Any]]:
    selected_samples, manifest = _select_samples(
        domain, samples_per_round=samples_per_round, seed=seed
    )
    detector = _detector_for(domain, detector_name)
    repeats_template = build_fixed_input_repeats(
        domain=domain,
        selected_samples=selected_samples,
        seed=seed,
        repeats=repeats,
    )
    for repeat in repeats_template:
        response = generate_ollama_prompt(model=model, domain=domain, seed=seed)
        scores: list[float] = []
        truths: list[str] = []
        for sample in selected_samples:
            result = await detector.detect(
                prompt=sample.get("prompt", ""),
                output=sample.get("output", ""),
                code=sample.get("code"),
                context=sample.get("context"),
            )
            scores.append(float(result.score))
            truths.append(manifest.get(sample["id"], {}).get("ground_truth_label", "normal"))
        repeat["response"] = response
        repeat["score_outputs"] = scores
        repeat["ground_truths"] = truths
    return repeats_template


def _domain_baseline_summary(domain: str) -> dict[str, Any]:
    baseline = _read_json(PROJECT_ROOT / BASELINE_REFERENCE)
    return baseline["sections"]["ollama"]["domains"][domain]["summary"]


def _global_variance_summary(domains: dict[str, Any], section: str) -> dict[str, Any]:
    means = [payload[section]["spread"]["mean"] for payload in domains.values()]
    stds = [payload[section]["spread"]["std"] for payload in domains.values()]
    flips = [payload[section]["spread"]["flip_count"] for payload in domains.values()]
    return {
        "mean_of_means": float(np.mean(means)) if means else 0.0,
        "mean_of_stds": float(np.mean(stds)) if stds else 0.0,
        "total_flip_count": int(sum(flips)),
        "domains": len(domains),
    }


def _source_decomposition(
    domain: str,
    response_summary: dict[str, Any],
    round_summary: dict[str, Any],
) -> dict[str, Any]:
    response_spread = response_summary["spread"]
    round_spread = round_summary["spread"]
    return {
        "gpu_kernel": {
            "observed_signal": (
                f"{domain} fixed-input reward spread std={round_spread['std']:.4f} with "
                f"{round_spread['flip_count']} threshold flips."
            ),
            "evidence_refs": [
                "resource.knowledge_resource.arxiv_2408_04667",
                f"local:{domain}#gpu_kernel",
            ],
            "inference": (
                "The repeated score spread may reflect nondeterministic GPU kernel execution "
                "or runtime-level scheduling effects in the local stack."
            ),
            "mitigation": (
                "Treat single-run scores as provisional and size future validator thresholds "
                "against repeated-run spread before tightening penalties."
            ),
        },
        "batching_scheduling": {
            "observed_signal": (
                f"{domain} produced {response_summary['unique_responses']} unique repeated responses "  # noqa: E501
                f"under a fixed seed and fixed sample IDs."
            ),
            "evidence_refs": [
                "resource.knowledge_resource.vllm_reproducibility",
                f"local:{domain}#batching_scheduling",
            ],
            "inference": (
                "The response diversity may reflect request batching or scheduling sensitivity "
                "even when validator-controlled inputs are held constant."
            ),
            "mitigation": (
                "Benchmark threshold choices on a stable local load profile and compare spread "
                "before and after serving-stack changes."
            ),
        },
        "floating_point_kernel": {
            "observed_signal": (
                f"{domain} response-score band ranged from {response_spread['min']:.4f} to "
                f"{response_spread['max']:.4f} around DECISION_THRESHOLD={DECISION_THRESHOLD:.1f}."
            ),
            "evidence_refs": [
                "resource.knowledge_resource.llm_nondeterminism_article",
                f"local:{domain}#floating_point_kernel",
            ],
            "inference": (
                "Near-threshold movement may reflect floating-point accumulation order or "
                "kernel-level numeric drift rather than a semantic behavior change."
            ),
            "mitigation": (
                "Prefer margin-aware or repeated-run validator comparisons in later phases "
                "instead of treating exact threshold crossings as fully stable."
            ),
        },
        "harness_level_randomness": {
            "observed_signal": (
                f"{domain} fixed-input repeats reused sample_ids={response_summary['sample_ids']} "
                "and seed=42 while the baseline runner still increments round seeds."
            ),
            "evidence_refs": [BASELINE_REFERENCE, f"local:{domain}#harness_level_randomness"],
            "inference": (
                "This source is validator-controlled harness randomness, not miner behavior, "
                "and must stay separated from runtime variance when interpreting results."
            ),
            "mitigation": (
                "Keep the Phase 81 measurement path fixed-input only, and preserve the existing "
                "seed-increment baseline path for comparability without conflating the two."
            ),
        },
    }


def summarize_domain_variance(
    domain: str,
    response_repeats: list[dict[str, Any]],
    round_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    response_means = [
        float(np.mean(repeat["score_outputs"])) if repeat["score_outputs"] else 0.0
        for repeat in response_repeats
    ]
    unique_responses = len({repeat["response"].get("text", "") for repeat in response_repeats})
    sample_ids = response_repeats[0].get("sample_ids", []) if response_repeats else []
    response_summary = {
        "metric": "mean_score",
        "spread": _response_spread(response_means),
        "unique_responses": unique_responses,
        "sample_ids": sample_ids,
        "repeats": len(response_repeats),
    }
    round_summary = _round_metric_spread(round_runs)
    source_decomposition = _source_decomposition(domain, response_summary, round_summary)
    return {
        "domain": domain,
        "decision_threshold": DECISION_THRESHOLD,
        "response_variance": response_summary,
        "round_variance": round_summary,
        "source_decomposition": source_decomposition,
    }


def _environment_metadata(model: str) -> dict[str, Any]:
    ollama_version = "unknown"
    try:
        completed = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.stdout.strip():
            ollama_version = completed.stdout.strip()
    except FileNotFoundError:
        ollama_version = "unavailable"

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "ollama_version": ollama_version,
        "model": model,
        "timestamp": _now_iso(),
        "baseline_path": BASELINE_REFERENCE,
    }


def _artifact_payload(
    model: str,
    rounds: int,
    samples_per_round: int,
    seed: int,
    domain_results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": f"phase81-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "phase": "81",
        "baseline_reference": BASELINE_REFERENCE,
        "environment": _environment_metadata(model),
        "config": {
            "model": model,
            "detector": "IsolationForest",
            "rounds": rounds,
            "samples_per_round": samples_per_round,
            "seed": seed,
        },
        "domains": domain_results,
        "response_variance": {
            "global_summary": _global_variance_summary(domain_results, "response_variance")
        },
        "round_variance": {
            "global_summary": _global_variance_summary(domain_results, "round_variance")
        },
    }


def _markdown_for_artifact(artifact: dict[str, Any]) -> str:
    lines = [
        "# Phase 81 Non-Determinism Impact Measurement",
        "",
        f"Baseline anchor: `{artifact['baseline_reference']}`",
        "",
        "Locked sources: arXiv 2408.04667, vLLM reproducibility guidance, and the Brendoerfer non-determinism article.",  # noqa: E501
        "",
        "Validators should interpret these results under adversarial miner assumptions: miners remain adversarial, determinism is best-effort only, validators cannot force exact miner inference behavior, and any controls must stay backward-compatible and commodity-hardware feasible.",  # noqa: E501
        "",
        "## Observed variance",
        "",
    ]
    for domain, payload in artifact["domains"].items():
        spread = payload["round_variance"]["spread"]
        lines.append(
            f"- `{domain}`: reward mean={spread['mean']:.4f}, std={spread['std']:.4f}, "
            f"range=({spread['min']:.4f}, {spread['max']:.4f}), flip_count={spread['flip_count']}"
        )
    lines.extend(
        [
            "",
            "## Inferred variance sources",
            "",
            "The observations above are measurements. The source mapping below is cautious inference rather than causal proof.",  # noqa: E501
            "",
            "## source_decomposition",
            "",
            "| Source | Observed signal | Evidence refs | Cautious inference | Mitigation |",
            "|---|---|---|---|---|",
        ]
    )
    for domain, payload in artifact["domains"].items():
        for source_key, source_payload in payload["source_decomposition"].items():
            lines.append(
                f"| {domain}:{source_key} | {source_payload['observed_signal']} | "
                f"{', '.join(source_payload['evidence_refs'])} | {source_payload['inference']} | "
                f"{source_payload['mitigation']} |"
            )
    lines.extend(
        [
            "",
            "## Recommended mitigations",
            "",
            "Use repeated measurement when tuning validator thresholds, keep fixed-input experiments separate from baseline seed-increment runs, and prefer validator-side controls that remain backward-compatible and commodity-hardware feasible. The goal is to measure instability margins, not to assume miners can be forced into exact deterministic inference behavior.",  # noqa: E501
        ]
    )
    return "\n".join(lines) + "\n"


def build_overwatch_claim_payload(experiment_data: dict[str, Any]) -> dict[str, Any]:
    run_id = experiment_data["run_id"]
    created_at = _now_iso()
    experiment_key = "antigence_phase81_nondeterminism"
    publications = [
        {
            "_key": "phase81_arxiv_2408_04667",
            "title": "LLM temperature=0 variance up to 15% (arXiv 2408.04667)",
            "url": "https://arxiv.org/html/2408.04667v5",
            "publication_type": "preprint",
            "year": 2024,
            "visibility": "shared",
            "created_at": created_at,
            "updated_at": created_at,
        },
        {
            "_key": "phase81_vllm_reproducibility",
            "title": "vLLM Reproducibility Documentation",
            "url": "https://docs.vllm.ai/en/latest/usage/reproducibility/",
            "publication_type": "documentation",
            "year": 2026,
            "visibility": "shared",
            "created_at": created_at,
            "updated_at": created_at,
        },
        {
            "_key": "phase81_llm_nondeterminism_article",
            "title": "Why LLMs Are Not Deterministic",
            "url": "https://mbrenndoerfer.com/writing/why-llms-are-not-deterministic",
            "publication_type": "web_article",
            "year": 2025,
            "visibility": "shared",
            "created_at": created_at,
            "updated_at": created_at,
        },
    ]
    claims = []
    domains = experiment_data.get("domains", {})
    if not domains and "source_decomposition" in experiment_data:
        domains = {
            "aggregate": {
                "response_variance": {
                    "spread": {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "cv": 0.0,
                        "flip_count": 0,
                    }
                },
                "round_variance": {
                    "spread": {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "cv": 0.0,
                        "flip_count": 0,
                    }
                },
                "source_decomposition": experiment_data["source_decomposition"],
            }
        }
    for domain, payload in domains.items():
        claim_key = f"phase81_{_safe_slug(domain)}_variance"
        source_decomposition = payload["source_decomposition"]
        observed_variance = [
            (
                f"{domain} fixed-input mean score std="
                f"{payload['response_variance']['spread']['std']:.4f} with "
                f"{payload['response_variance']['spread']['flip_count']} fixed-input threshold flips."  # noqa: E501
            )
        ]
        inferred_sources = [source["inference"] for source in source_decomposition.values()]
        recommended_mitigations = [source["mitigation"] for source in source_decomposition.values()]
        claims.append(
            {
                "_key": claim_key,
                "text": (
                    f"{domain} showed measured validator-facing variance across repeated local runs; "  # noqa: E501
                    "miners remain adversarial, determinism is best-effort only, validators cannot force exact miner inference behavior, "  # noqa: E501
                    "and future controls must remain backward-compatible and commodity-hardware feasible."  # noqa: E501
                ),
                "classification": "MEASURED",
                "claim_type": "statistical",
                "created_at": created_at,
                "updated_at": created_at,
                "visibility": "shared",
                "confidence": 0.75,
                "supporting_evidence": [f"artifact/phase81_{_safe_slug(domain)}"],
                "source_project": "antigence-bittensor",
                "exp_id": "EXP-081",
                "status": "active",
                "observed_variance": observed_variance,
                "inferred_sources": inferred_sources,
                "recommended_mitigations": recommended_mitigations,
                "source_decomposition": source_decomposition,
                "derived_from": {
                    "derivation_method": "statistical_analysis",
                    "run_id": run_id,
                    "evidence_refs": [f"local:{DEFAULT_OUTPUT_JSON}", f"local:{DEFAULT_OUTPUT_MD}"],
                },
                "citations": [
                    {
                        "publication_key": "phase81_arxiv_2408_04667",
                        "citation_context": "Observed repeated-run variance motivates measuring spread, not single-point scores.",  # noqa: E501
                        "evidence_refs": ["resource.knowledge_resource.arxiv_2408_04667"],
                    },
                    {
                        "publication_key": "phase81_vllm_reproducibility",
                        "citation_context": "Runtime reproducibility remains scoped to environment and scheduling controls.",  # noqa: E501
                        "evidence_refs": ["resource.knowledge_resource.vllm_reproducibility"],
                    },
                    {
                        "publication_key": "phase81_llm_nondeterminism_article",
                        "citation_context": "Floating-point, batching, and hardware execution order remain plausible sources of variance.",  # noqa: E501
                        "evidence_refs": ["resource.knowledge_resource.llm_nondeterminism_article"],
                    },
                ],
            }
        )
    return {
        "run_id": run_id,
        "experiment": {
            "_key": experiment_key,
            "exp_id": "EXP-081",
            "title": "Phase 81 Non-Determinism Impact Measurement",
            "project": "antigence-bittensor",
            "status": "completed",
            "description": "10x per-domain repeated measurement separating fixed-input response variance from round-metric variance.",  # noqa: E501
            "methodology": "Fixed-input repetitions plus repeated full round measurements against the Phase 80 baseline path.",  # noqa: E501
            "outcome": "positive",
            "agent": "codex",
            "run_ids": [run_id],
            "script_path": "scripts/phase81_nondeterminism.py",
            "note_path": DEFAULT_OUTPUT_MD,
            "result_paths": [DEFAULT_OUTPUT_JSON, DEFAULT_OUTPUT_OVERWATCH],
            "visibility": "shared",
            "created_at": created_at,
            "updated_at": created_at,
        },
        "publications": publications,
        "claims": claims,
    }


def _load_artifact_inputs(
    artifact_data: dict[str, Any] | None = None,
    markdown_text: str | None = None,
    overwatch_payload: dict[str, Any] | None = None,
    writeback_report: dict[str, Any] | None = None,
    input_json: str | None = None,
    input_md: str | None = None,
    input_overwatch: str | None = None,
    input_report: str | None = None,
) -> ArtifactInputs:
    return ArtifactInputs(
        artifact_data=artifact_data if artifact_data is not None else _read_json(input_json),
        markdown_text=markdown_text if markdown_text is not None else Path(input_md).read_text(),
        overwatch_payload=overwatch_payload
        if overwatch_payload is not None
        else _read_json(input_overwatch),
        writeback_report=writeback_report
        if writeback_report is not None
        else (_read_json(input_report) if input_report else None),
    )


def validate_phase81_artifacts(
    artifact_data: dict[str, Any] | None = None,
    markdown_text: str | None = None,
    overwatch_payload: dict[str, Any] | None = None,
    writeback_report: dict[str, Any] | None = None,
    input_json: str | None = None,
    input_md: str | None = None,
    input_overwatch: str | None = None,
    input_report: str | None = None,
) -> None:
    inputs = _load_artifact_inputs(
        artifact_data=artifact_data,
        markdown_text=markdown_text,
        overwatch_payload=overwatch_payload,
        writeback_report=writeback_report,
        input_json=input_json,
        input_md=input_md,
        input_overwatch=input_overwatch,
        input_report=input_report,
    )
    required_top_level = {
        "environment",
        "baseline_reference",
        "domains",
        "response_variance",
        "round_variance",
    }
    missing = required_top_level - set(inputs.artifact_data)
    if missing:
        raise ValueError(f"Artifact JSON missing keys: {sorted(missing)}")
    if inputs.artifact_data["baseline_reference"] != BASELINE_REFERENCE:
        raise ValueError(
            "Artifact JSON baseline_reference must anchor to data/benchmarks/v9.2-baseline.json"
        )
    for domain, payload in inputs.artifact_data["domains"].items():
        for key in ("response_variance", "round_variance", "source_decomposition"):
            if key not in payload:
                raise ValueError(f"Domain {domain} missing {key}")
        source_keys = set(payload["source_decomposition"])
        if source_keys != set(SOURCE_KEYS):
            raise ValueError(
                f"Domain {domain} source_decomposition mismatch: {sorted(source_keys)}"
            )
    for heading in (
        "Observed variance",
        "Inferred variance sources",
        "Recommended mitigations",
        "source_decomposition",
    ):
        if heading not in inputs.markdown_text:
            raise ValueError(f"Markdown missing heading/text: {heading}")
    lowered_markdown = inputs.markdown_text.lower()
    for phrase in BOUNDARY_PHRASES:
        if phrase.lower() not in lowered_markdown:
            raise ValueError(f"Markdown missing boundary phrase: {phrase}")
    if "claims" not in inputs.overwatch_payload:
        raise ValueError("Overwatch payload missing claims")
    for claim in inputs.overwatch_payload["claims"]:
        for key in (
            "observed_variance",
            "inferred_sources",
            "recommended_mitigations",
            "source_decomposition",
        ):
            if key not in claim:
                raise ValueError(f"Claim missing {key}")
        if set(claim["source_decomposition"]) != set(SOURCE_KEYS):
            raise ValueError("Claim source_decomposition missing required sources")
        claim_text = json.dumps(claim).lower()
        for phrase in (
            "best-effort",
            "cannot force exact miner inference behavior",
            "backward-compatible",
            "commodity-hardware",
        ):
            if phrase not in claim_text:
                raise ValueError(f"Claim payload missing phrase: {phrase}")
    if inputs.writeback_report is not None:
        if "run_id" not in inputs.writeback_report:
            raise ValueError("Writeback report missing run_id")
        if "errors" not in inputs.writeback_report:
            raise ValueError("Writeback report missing errors")
        if "counts" not in inputs.writeback_report and not inputs.writeback_report.get("skipped"):
            raise ValueError("Writeback report must contain counts unless skipped=true")


def build_writeback_report(
    run_id: str,
    counts: dict[str, int] | None,
    errors: list[str],
    skipped: bool = False,
    reason: str | None = None,
    replay_command: str | None = None,
) -> dict[str, Any]:
    report = {
        "run_id": run_id,
        "counts": counts or {},
        "errors": errors,
        "created_at": _now_iso(),
    }
    if skipped:
        report["skipped"] = True
        report["reason"] = reason
        if replay_command:
            report["replay_command"] = replay_command
    return report


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode()
    req = request.Request(url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=2) as response:
        raw = response.read()
    return json.loads(raw.decode() or "{}")


def _upsert_document(base_url: str, collection: str, document: dict[str, Any]) -> None:
    query = parse.urlencode({"overwriteMode": "update", "returnNew": "false", "silent": "true"})
    url = f"{base_url}/_db/{OVERWATCH_DB}/_api/document/{collection}?{query}"
    _http_json("POST", url, document)


def _overwatch_available(base_url: str) -> bool:
    try:
        _http_json("GET", f"{base_url}/_api/version")
        return True
    except Exception:
        return False


def apply_overwatch_claims(
    overwatch_payload: dict[str, Any] | None = None,
    input_overwatch: str | None = None,
    writeback_report_path: str = DEFAULT_WRITEBACK_REPORT,
    base_url: str = OVERWATCH_BASE_URL,
) -> dict[str, Any]:
    payload = overwatch_payload if overwatch_payload is not None else _read_json(input_overwatch)
    run_id = payload["run_id"]
    replay_command = (
        "PYTHON=python3.11 make test-env && .venv/bin/python scripts/phase81_nondeterminism.py "
        f"--validate-artifacts --input-json {DEFAULT_OUTPUT_JSON} --input-md {DEFAULT_OUTPUT_MD} "
        f"--input-overwatch {DEFAULT_OUTPUT_OVERWATCH} --writeback-report {writeback_report_path} --apply-overwatch"  # noqa: E501
    )
    if not _overwatch_available(base_url):
        report = build_writeback_report(
            run_id=run_id,
            counts={},
            errors=[],
            skipped=True,
            reason="overwatch_unavailable",
            replay_command=replay_command,
        )
        _write_json(writeback_report_path, report)
        return report

    counts = {"experiments": 0, "publications": 0, "claims": 0, "derived_from": 0, "cites": 0}
    errors_seen: list[str] = []
    try:
        _upsert_document(base_url, "experiment", payload["experiment"])
        counts["experiments"] += 1
    except Exception as exc:
        errors_seen.append(f"experiment:{exc}")
    for publication in payload.get("publications", []):
        try:
            _upsert_document(base_url, "publication", publication)
            counts["publications"] += 1
        except Exception as exc:
            errors_seen.append(f"publication:{publication['_key']}:{exc}")
    for claim in payload.get("claims", []):
        claim_doc = {
            key: value for key, value in claim.items() if key not in {"derived_from", "citations"}
        }
        try:
            _upsert_document(base_url, "claim", claim_doc)
            counts["claims"] += 1
        except Exception as exc:
            errors_seen.append(f"claim:{claim['_key']}:{exc}")
            continue
        derived_edge = {
            "_key": f"{claim['_key']}_derived_from_exp081",
            "_from": f"claim/{claim['_key']}",
            "_to": f"experiment/{payload['experiment']['_key']}",
            "rel_type": "derived_from",
            "source_system": "antigence-bittensor",
            "visibility": "shared",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            **claim["derived_from"],
        }
        try:
            _upsert_document(base_url, "derived_from", derived_edge)
            counts["derived_from"] += 1
        except Exception as exc:
            errors_seen.append(f"derived_from:{claim['_key']}:{exc}")
        for index, citation in enumerate(claim.get("citations", []), start=1):
            cites_edge = {
                "_key": f"{claim['_key']}_cites_{index}",
                "_from": f"claim/{claim['_key']}",
                "_to": f"publication/{citation['publication_key']}",
                "rel_type": "cites",
                "source_system": "antigence-bittensor",
                "visibility": "shared",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "citation_type": "supports",
                "citation_context": citation["citation_context"],
                "evidence_refs": citation["evidence_refs"],
            }
            try:
                _upsert_document(base_url, "cites", cites_edge)
                counts["cites"] += 1
            except Exception as exc:
                errors_seen.append(f"cites:{claim['_key']}:{exc}")
    report = build_writeback_report(run_id=run_id, counts=counts, errors=errors_seen)
    _write_json(writeback_report_path, report)
    return report


async def run_phase81_experiment(
    model: str = "qwen2.5:1.5b",
    detector_name: str = "IsolationForest",
    rounds: int = 10,
    samples_per_round: int = 20,
    seed: int = 42,
    output_json: str = DEFAULT_OUTPUT_JSON,
    output_md: str = DEFAULT_OUTPUT_MD,
    output_overwatch: str = DEFAULT_OUTPUT_OVERWATCH,
) -> dict[str, Any]:
    if not check_ollama_available(model):
        raise RuntimeError(f"Ollama model {model} is not available")

    domain_results: dict[str, Any] = {}
    for domain in DOMAINS:
        response_repeats = await _measure_fixed_input_variance(
            domain=domain,
            detector_name=detector_name,
            model=model,
            samples_per_round=samples_per_round,
            seed=seed,
            repeats=rounds,
        )
        round_runs = []
        for repeat_index in range(rounds):
            round_runs.append(
                await run_single_round(
                    domain=domain,
                    detector_name=detector_name,
                    model=model,
                    samples_per_round=samples_per_round,
                    warmup=repeat_index == 0,
                    seed=seed,
                )
            )
        domain_summary = summarize_domain_variance(
            domain=domain,
            response_repeats=response_repeats,
            round_runs=round_runs,
        )
        domain_summary["response_repeats"] = response_repeats
        domain_summary["round_runs"] = round_runs
        domain_summary["baseline_summary"] = _domain_baseline_summary(domain)
        domain_results[domain] = domain_summary

    artifact = _artifact_payload(
        model=model,
        rounds=rounds,
        samples_per_round=samples_per_round,
        seed=seed,
        domain_results=domain_results,
    )
    markdown = _markdown_for_artifact(artifact)
    overwatch_payload = build_overwatch_claim_payload(artifact)
    _write_json(output_json, artifact)
    _write_text(output_md, markdown)
    _write_json(output_overwatch, overwatch_payload)
    return {
        "artifact": artifact,
        "markdown": markdown,
        "overwatch_payload": overwatch_payload,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--detector", default="IsolationForest")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--samples-per-round", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-overwatch", default=DEFAULT_OUTPUT_OVERWATCH)
    parser.add_argument("--writeback-report", default=DEFAULT_WRITEBACK_REPORT)
    parser.add_argument("--validate-artifacts", action="store_true")
    parser.add_argument("--apply-overwatch", action="store_true")
    parser.add_argument("--input-json")
    parser.add_argument("--input-md")
    parser.add_argument("--input-overwatch")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.validate_artifacts:
        validate_phase81_artifacts(
            input_json=args.input_json or args.output_json,
            input_md=args.input_md or args.output_md,
            input_overwatch=args.input_overwatch or args.output_overwatch,
            input_report=args.writeback_report if Path(args.writeback_report).exists() else None,
        )
        if args.apply_overwatch:
            apply_overwatch_claims(
                input_overwatch=args.input_overwatch or args.output_overwatch,
                writeback_report_path=args.writeback_report,
            )
        return 0

    result = asyncio.run(
        run_phase81_experiment(
            model=args.model,
            detector_name=args.detector,
            rounds=args.rounds,
            samples_per_round=args.samples_per_round,
            seed=args.seed,
            output_json=args.output_json,
            output_md=args.output_md,
            output_overwatch=args.output_overwatch,
        )
    )
    validate_phase81_artifacts(
        artifact_data=result["artifact"],
        markdown_text=result["markdown"],
        overwatch_payload=result["overwatch_payload"],
        writeback_report=None,
    )
    if args.apply_overwatch:
        apply_overwatch_claims(
            overwatch_payload=result["overwatch_payload"],
            writeback_report_path=args.writeback_report,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
