"""Cross-domain data expansion validation tests.

Validates MAIN-05 acceptance criteria: all 4 evaluation domains have 220 samples
each, balanced classes (>=45% per class), correct honeypot ratios, consistent
schema, adversarial edge cases, and that EvaluationDataset loads all domains
without error.
"""

import json
from pathlib import Path

import pytest

from antigence_subnet.validator.evaluation import EvaluationDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/evaluation")
DOMAINS = ["bio", "code_security", "hallucination", "reasoning"]
DOMAIN_PREFIXES = {
    "hallucination": "hall",
    "code_security": "cs",
    "reasoning": "rea",
    "bio": "bio",
}
EXPECTED_HONEYPOTS = {
    "hallucination": 20,
    "code_security": 15,
    "reasoning": 11,
    "bio": 21,
}


# ---------------------------------------------------------------------------
# Test 1: All domains have exactly 220 samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_all_domains_220_samples(domain):
    """Each domain must have exactly 220 samples in samples.json and manifest.json."""
    samples_path = DATA_DIR / domain / "samples.json"
    manifest_path = DATA_DIR / domain / "manifest.json"

    with open(samples_path) as f:
        data = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    samples = data["samples"]
    assert len(samples) == 220, f"{domain}: expected 220 samples, got {len(samples)}"
    assert len(manifest) == 220, f"{domain}: expected 220 manifest entries, got {len(manifest)}"

    # All sample IDs exist in manifest
    sample_ids = [s["id"] for s in samples]
    assert len(set(sample_ids)) == 220, f"{domain}: duplicate IDs found"
    for sid in sample_ids:
        assert sid in manifest, f"{domain}: sample {sid} missing from manifest"


# ---------------------------------------------------------------------------
# Test 2: Class balance across all domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_class_balance_all_domains(domain):
    """Each domain must have balanced normal/anomalous classes (>=45% per class)."""
    manifest_path = DATA_DIR / domain / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    normal = sum(1 for v in manifest.values() if v["ground_truth_label"] == "normal")
    anomalous = sum(1 for v in manifest.values() if v["ground_truth_label"] == "anomalous")

    total = normal + anomalous
    assert total == 220, f"{domain}: normal ({normal}) + anomalous ({anomalous}) != 220"
    # 55/45 balance contract: each class must hold at least 45% of total samples.
    min_class_count = int(total * 0.45)
    assert normal >= min_class_count, (
        f"{domain}: too few normal samples ({normal}), min {min_class_count} "
        f"(45% of {total})"
    )
    assert anomalous >= min_class_count, (
        f"{domain}: too few anomalous samples ({anomalous}), min {min_class_count} "
        f"(45% of {total})"
    )


# ---------------------------------------------------------------------------
# Test 3: Honeypot ratios across all domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_honeypot_ratio_all_domains(domain):
    """Each domain must have honeypots within ~5-10% of total samples."""
    manifest_path = DATA_DIR / domain / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    honeypots = sum(1 for v in manifest.values() if v.get("is_honeypot"))
    expected = EXPECTED_HONEYPOTS[domain]

    assert honeypots == expected, f"{domain}: expected {expected} honeypots, got {honeypots}"
    assert honeypots >= 6, f"{domain}: too few honeypots ({honeypots}), min 6"
    assert honeypots <= 25, f"{domain}: too many honeypots ({honeypots}), max 25"


# ---------------------------------------------------------------------------
# Test 4: Sample schema validation across all domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_sample_schema_all_domains(domain):
    """Each sample must have required keys with correct format."""
    prefix = DOMAIN_PREFIXES[domain]
    samples_path = DATA_DIR / domain / "samples.json"
    with open(samples_path) as f:
        data = json.load(f)

    for sample in data["samples"]:
        # Required keys
        assert "id" in sample, f"{domain}: sample missing 'id'"
        assert sample["id"].startswith(f"eval-{prefix}-"), (
            f"{domain}: ID {sample['id']} doesn't start with eval-{prefix}-"
        )
        assert "prompt" in sample, f"{domain}: {sample['id']} missing 'prompt'"
        assert isinstance(sample["prompt"], str) and len(sample["prompt"]) > 0, (
            f"{domain}: {sample['id']} prompt must be non-empty string"
        )
        assert "domain" in sample, f"{domain}: {sample['id']} missing 'domain'"
        assert sample["domain"] == domain, (
            f"{domain}: {sample['id']} domain is {sample['domain']}, expected {domain}"
        )
        assert "metadata" in sample, f"{domain}: {sample['id']} missing 'metadata'"
        assert "difficulty" in sample["metadata"], (
            f"{domain}: {sample['id']} metadata missing 'difficulty'"
        )
        assert "source" in sample["metadata"], f"{domain}: {sample['id']} metadata missing 'source'"

        # Domain-specific content field
        if domain == "code_security":
            assert "code" in sample, f"{domain}: {sample['id']} missing 'code' field"
        else:
            assert "output" in sample, f"{domain}: {sample['id']} missing 'output' field"


# ---------------------------------------------------------------------------
# Test 5: Manifest schema validation across all domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_manifest_schema_all_domains(domain):
    """Each manifest entry must have valid ground_truth_label and is_honeypot."""
    manifest_path = DATA_DIR / domain / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    for sid, entry in manifest.items():
        assert "ground_truth_label" in entry, f"{domain}: {sid} missing 'ground_truth_label'"
        assert entry["ground_truth_label"] in ("normal", "anomalous"), (
            f"{domain}: {sid} has invalid label '{entry['ground_truth_label']}'"
        )
        assert "is_honeypot" in entry, f"{domain}: {sid} missing 'is_honeypot'"
        assert isinstance(entry["is_honeypot"], bool), (
            f"{domain}: {sid} is_honeypot must be boolean"
        )

        if entry["ground_truth_label"] == "anomalous":
            assert entry.get("ground_truth_type") is not None, (
                f"{domain}: anomalous {sid} missing 'ground_truth_type'"
            )
            assert (
                isinstance(entry["ground_truth_type"], str) and len(entry["ground_truth_type"]) > 0
            ), f"{domain}: anomalous {sid} ground_truth_type must be non-empty string"


# ---------------------------------------------------------------------------
# Test 6: Adversarial samples present in relevant domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["code_security", "reasoning", "bio"])
def test_adversarial_samples_present(domain):
    """code_security, reasoning, and bio must each have at least 2 adversarial samples."""
    manifest_path = DATA_DIR / domain / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    adversarial = [
        sid
        for sid, entry in manifest.items()
        if entry.get("ground_truth_type") and "adversarial" in entry["ground_truth_type"]
    ]
    assert len(adversarial) >= 2, (
        f"{domain}: expected at least 2 adversarial samples, found {len(adversarial)}: {adversarial}"  # noqa: E501
    )


# ---------------------------------------------------------------------------
# Test 7: EvaluationDataset loads all domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_evaluation_dataset_loads_all_domains(domain):
    """EvaluationDataset must load each domain with correct sample and honeypot counts."""
    dataset = EvaluationDataset(DATA_DIR, domain)

    assert len(dataset.samples) == 220, (
        f"{domain}: EvaluationDataset loaded {len(dataset.samples)} samples, expected 220"
    )
    expected_hp = EXPECTED_HONEYPOTS[domain]
    assert len(dataset._honeypot_samples) == expected_hp, (
        f"{domain}: EvaluationDataset has {len(dataset._honeypot_samples)} honeypots, expected {expected_hp}"  # noqa: E501
    )

    # get_round_samples returns correct count
    round_samples = dataset.get_round_samples(round_num=42, n=10, n_honeypots=2)
    assert len(round_samples) == 10, (
        f"{domain}: get_round_samples returned {len(round_samples)}, expected 10"
    )
