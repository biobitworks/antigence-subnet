"""Baseline integrity tests for committed v9.2 benchmark JSONs.

These tests validate the IMMUTABLE v9.2 baseline files that are committed
to the repository. They do NOT regenerate baselines -- they verify that
the committed files have the expected schema, completeness, and value ranges.
"""

import json
from pathlib import Path

import pytest

# Project root (two levels up from tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DETECTOR_BASELINE = PROJECT_ROOT / "data" / "benchmarks" / "v9.2-baseline-detectors.json"
ORCHESTRATOR_BASELINE = PROJECT_ROOT / "data" / "benchmarks" / "v9.2-baseline-orchestrator.json"
OVERWATCH_PAYLOAD = PROJECT_ROOT / "data" / "overwatch" / "registration-payload.json"

EXPECTED_DOMAINS = {"hallucination", "code_security", "reasoning", "bio"}
EXPECTED_GENERIC_DETECTORS = {
    "IsolationForest", "LOF", "OCSVM", "Fractal", "NegSel", "Autoencoder"
}
DETECTOR_REQUIRED_KEYS = {
    "detector", "domain", "f1", "precision", "recall", "accuracy",
    "avg_latency_ms", "throughput_per_sec", "fit_time_ms", "rounds"
}
EXPECTED_PIPELINES = {"flat_ensemble", "orchestrator", "v9_orchestrator"}
EXPECTED_RESOURCE_IDS = {
    "resource.knowledge_resource.miroshark",
    "resource.knowledge_resource.mirofish",
    "resource.knowledge_resource.pentagi",
    "resource.knowledge_resource.llm_nondeterminism_article",
    "resource.knowledge_resource.vllm_reproducibility",
    "resource.knowledge_resource.arxiv_2408_04667",
}


# --- Detector Baseline Tests ---

def test_detector_baseline_exists():
    """v9.2 detector baseline JSON exists and is valid JSON."""
    assert DETECTOR_BASELINE.exists(), f"Missing: {DETECTOR_BASELINE}"
    data = json.loads(DETECTOR_BASELINE.read_text())
    assert isinstance(data, list), "Detector baseline must be a JSON array"
    assert len(data) > 0, "Detector baseline must not be empty"


def test_detector_baseline_schema():
    """Each entry has all required keys."""
    data = json.loads(DETECTOR_BASELINE.read_text())
    for i, entry in enumerate(data):
        missing = DETECTOR_REQUIRED_KEYS - set(entry.keys())
        assert not missing, f"Entry {i} ({entry.get('detector', '?')}/{entry.get('domain', '?')}) missing keys: {missing}"


def test_detector_baseline_domains():
    """All 4 domains present in detector baseline."""
    data = json.loads(DETECTOR_BASELINE.read_text())
    domains = {entry["domain"] for entry in data}
    missing = EXPECTED_DOMAINS - domains
    assert not missing, f"Missing domains: {missing}"


def test_detector_baseline_detectors():
    """At least 6 generic detectors present plus domain packs."""
    data = json.loads(DETECTOR_BASELINE.read_text())
    detectors = {entry["detector"] for entry in data}
    missing = EXPECTED_GENERIC_DETECTORS - detectors
    assert not missing, f"Missing generic detectors: {missing}"
    # Should have more than 6 (domain packs add extras)
    assert len(detectors) >= 6, f"Expected at least 6 detectors, got {len(detectors)}"


def test_detector_baseline_metrics_range():
    """All F1/precision/recall values in [0.0, 1.0]."""
    data = json.loads(DETECTOR_BASELINE.read_text())
    for i, entry in enumerate(data):
        for metric in ("f1", "precision", "recall"):
            val = entry[metric]
            assert 0.0 <= val <= 1.0, (
                f"Entry {i} ({entry['detector']}/{entry['domain']}): "
                f"{metric}={val} out of range [0.0, 1.0]"
            )


def test_detector_baseline_entry_count():
    """Baseline should have at least 28 entries (7 detectors x 4 domains)."""
    data = json.loads(DETECTOR_BASELINE.read_text())
    # 6 generic detectors + 1 domain pack per domain x 4 domains = 28 minimum
    assert len(data) >= 28, f"Expected >= 28 entries, got {len(data)}"


# --- Orchestrator Baseline Tests ---

def test_orchestrator_baseline_exists():
    """v9.2 orchestrator baseline JSON exists and is valid JSON."""
    assert ORCHESTRATOR_BASELINE.exists(), f"Missing: {ORCHESTRATOR_BASELINE}"
    data = json.loads(ORCHESTRATOR_BASELINE.read_text())
    assert isinstance(data, dict), "Orchestrator baseline must be a JSON object"
    assert "domains" in data, "Orchestrator baseline must have 'domains' key"


def test_orchestrator_baseline_schema():
    """Each domain has flat_ensemble, orchestrator, and v9_orchestrator."""
    data = json.loads(ORCHESTRATOR_BASELINE.read_text())
    domains = data["domains"]
    for domain_name in EXPECTED_DOMAINS:
        assert domain_name in domains, f"Missing domain: {domain_name}"
        pipelines = set(domains[domain_name].keys())
        missing = EXPECTED_PIPELINES - pipelines
        assert not missing, f"Domain {domain_name} missing pipelines: {missing}"


def test_orchestrator_baseline_has_f1():
    """Each pipeline entry has f1, precision, recall keys."""
    data = json.loads(ORCHESTRATOR_BASELINE.read_text())
    domains = data["domains"]
    for domain_name, domain_data in domains.items():
        for pipeline_name, pipeline_data in domain_data.items():
            if pipeline_name in EXPECTED_PIPELINES:
                for metric in ("f1", "precision", "recall"):
                    assert metric in pipeline_data, (
                        f"{domain_name}/{pipeline_name} missing '{metric}'"
                    )


# --- Overwatch Registration Payload Tests ---

def test_overwatch_payload_exists():
    """Overwatch registration payload exists and is valid JSON."""
    assert OVERWATCH_PAYLOAD.exists(), f"Missing: {OVERWATCH_PAYLOAD}"
    data = json.loads(OVERWATCH_PAYLOAD.read_text())
    assert isinstance(data, dict), "Payload must be a JSON object"
    assert "resources" in data, "Payload must have 'resources' key"
    assert "claims" in data, "Payload must have 'claims' key"


def test_overwatch_payload_resources_count():
    """Resources list has exactly 6 entries."""
    data = json.loads(OVERWATCH_PAYLOAD.read_text())
    resources = data["resources"]
    assert len(resources) == 6, f"Expected 6 resources, got {len(resources)}"


def test_overwatch_payload_resource_ids():
    """All 6 resource_ids present."""
    data = json.loads(OVERWATCH_PAYLOAD.read_text())
    resource_ids = {r["resource_id"] for r in data["resources"]}
    missing = EXPECTED_RESOURCE_IDS - resource_ids
    assert not missing, f"Missing resource_ids: {missing}"


def test_overwatch_payload_claims():
    """Claims list has at least 1 entry with claim_type 'project-registration'."""
    data = json.loads(OVERWATCH_PAYLOAD.read_text())
    claims = data["claims"]
    assert len(claims) >= 1, "Expected at least 1 claim"
    registration_claims = [c for c in claims if c.get("claim_type") == "project-registration"]
    assert len(registration_claims) >= 1, "Expected at least 1 project-registration claim"
    # Verify claim has required fields
    claim = registration_claims[0]
    assert "claim_id" in claim, "Claim missing 'claim_id'"
    assert "evidence_tier" in claim, "Claim missing 'evidence_tier'"
    assert claim["evidence_tier"] == "MEASURED", f"Expected evidence_tier=MEASURED, got {claim['evidence_tier']}"


# --- Ollama Harness Baseline Tests ---

OLLAMA_DOMAINS = ["hallucination", "code_security", "reasoning", "bio"]
OLLAMA_SUMMARY_REQUIRED_KEYS = {
    "avg_f1", "std_f1", "avg_precision", "std_precision",
    "avg_recall", "std_recall", "avg_reward", "std_reward",
    "total_rounds", "total_samples", "total_time_s",
}


def _ollama_baseline_path(domain: str) -> Path:
    return PROJECT_ROOT / "data" / "benchmarks" / f"v9.2-baseline-ollama-{domain}.json"


@pytest.mark.parametrize("domain", OLLAMA_DOMAINS)
def test_ollama_baseline_exists(domain):
    """Per-domain Ollama harness baseline JSON exists and is valid JSON."""
    path = _ollama_baseline_path(domain)
    assert path.exists(), f"Missing: {path}"
    data = json.loads(path.read_text())
    assert isinstance(data, dict), f"Ollama {domain} baseline must be a JSON object"
    assert "summary" in data, f"Ollama {domain} baseline must have 'summary' key"


@pytest.mark.parametrize("domain", OLLAMA_DOMAINS)
def test_ollama_baseline_schema(domain):
    """Per-domain Ollama baseline summary has all required keys."""
    path = _ollama_baseline_path(domain)
    data = json.loads(path.read_text())
    summary = data["summary"]
    missing = OLLAMA_SUMMARY_REQUIRED_KEYS - set(summary.keys())
    assert not missing, f"Ollama {domain} summary missing keys: {missing}"


@pytest.mark.parametrize("domain", OLLAMA_DOMAINS)
def test_ollama_baseline_50_rounds(domain):
    """Per-domain Ollama baseline ran exactly 50 rounds."""
    path = _ollama_baseline_path(domain)
    data = json.loads(path.read_text())
    assert data["summary"]["total_rounds"] == 50, (
        f"Expected 50 rounds for {domain}, got {data['summary']['total_rounds']}"
    )


@pytest.mark.parametrize("domain", OLLAMA_DOMAINS)
def test_ollama_baseline_metrics_range(domain):
    """Ollama baseline F1/precision/recall in [0.0, 1.0]."""
    path = _ollama_baseline_path(domain)
    data = json.loads(path.read_text())
    summary = data["summary"]
    for metric in ("avg_f1", "avg_precision", "avg_recall"):
        val = summary[metric]
        assert 0.0 <= val <= 1.0, f"Ollama {domain} {metric}={val} out of [0.0, 1.0]"


# --- Consolidated Baseline Tests ---

COMBINED_BASELINE = PROJECT_ROOT / "data" / "benchmarks" / "v9.2-baseline.json"


def test_combined_baseline_exists():
    """Consolidated v9.2-baseline.json exists and is valid JSON."""
    assert COMBINED_BASELINE.exists(), f"Missing: {COMBINED_BASELINE}"
    data = json.loads(COMBINED_BASELINE.read_text())
    assert isinstance(data, dict), "Combined baseline must be a JSON object"
    assert data.get("version") == "9.2", f"Expected version=9.2, got {data.get('version')}"
    assert data.get("immutable") is True, "Combined baseline must be immutable"


def test_combined_baseline_has_all_sections():
    """Combined baseline has detectors, orchestrator, and ollama sections."""
    data = json.loads(COMBINED_BASELINE.read_text())
    sections = data.get("sections", {})
    for section in ("detectors", "orchestrator", "ollama"):
        assert section in sections, f"Combined baseline missing section: {section}"


def test_combined_baseline_ollama_domains():
    """Combined baseline ollama section has all 4 domains."""
    data = json.loads(COMBINED_BASELINE.read_text())
    ollama = data["sections"]["ollama"]
    domains = set(ollama.get("domains", {}).keys())
    missing = EXPECTED_DOMAINS - domains
    assert not missing, f"Combined baseline ollama missing domains: {missing}"
