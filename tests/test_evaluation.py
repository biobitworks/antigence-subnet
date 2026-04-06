"""Tests for the evaluation dataset manager.

Covers RWRD-02 (hidden evaluation datasets) and RWRD-03 (honeypot injection).
"""

import json
from pathlib import Path

import pytest

from antigence_subnet.validator.evaluation import EvaluationDataset

# --- Helper to create test data fixtures ---


def _create_test_data(data_dir: Path, n_regular: int = 8, n_honeypots: int = 4):
    """Create minimal test evaluation data in a temporary directory."""
    domain_dir = data_dir / "test_domain"
    domain_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    manifest = {}

    for i in range(n_regular):
        sid = f"test-reg-{i:03d}"
        label = "anomalous" if i % 2 == 0 else "normal"
        samples.append(
            {
                "id": sid,
                "prompt": f"Prompt {i}",
                "output": f"Output {i}",
                "domain": "test_domain",
                "metadata": {"difficulty": "easy", "source": "synthetic"},
            }
        )
        manifest[sid] = {
            "ground_truth_label": label,
            "ground_truth_type": "factual_error" if label == "anomalous" else None,
            "is_honeypot": False,
        }

    for i in range(n_honeypots):
        sid = f"test-hp-{i:03d}"
        label = "anomalous" if i % 2 == 0 else "normal"
        samples.append(
            {
                "id": sid,
                "prompt": f"Honeypot prompt {i}",
                "output": f"Honeypot output {i}",
                "domain": "test_domain",
                "metadata": {"difficulty": "medium", "source": "synthetic"},
            }
        )
        manifest[sid] = {
            "ground_truth_label": label,
            "ground_truth_type": "factual_error" if label == "anomalous" else None,
            "is_honeypot": True,
        }

    (domain_dir / "samples.json").write_text(json.dumps({"samples": samples}, indent=2))
    (domain_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return domain_dir


# --- Tests ---


class TestSamplesNoLabels:
    """Loaded samples contain only id, prompt, output, domain, metadata -- no ground_truth_label."""

    def test_samples_no_labels(self, tmp_path):
        _create_test_data(tmp_path)
        ds = EvaluationDataset(data_dir=tmp_path, domain="test_domain")
        for sample in ds.samples:
            assert "ground_truth_label" not in sample
            assert "id" in sample
            assert "prompt" in sample
            assert "output" in sample
            assert "domain" in sample
            assert "metadata" in sample


class TestManifestSeparation:
    """Manifest is loaded separately and keyed by sample ID."""

    def test_manifest_separation(self, tmp_path):
        _create_test_data(tmp_path)
        ds = EvaluationDataset(data_dir=tmp_path, domain="test_domain")

        # Manifest is a dict keyed by sample ID
        assert isinstance(ds.manifest, dict)

        # Every sample has a corresponding manifest entry
        for sample in ds.samples:
            sid = sample["id"]
            assert sid in ds.manifest
            entry = ds.manifest[sid]
            assert "ground_truth_label" in entry
            assert "is_honeypot" in entry


class TestRoundRotation:
    """Different round numbers produce different sample sets."""

    def test_round_rotation(self, tmp_path):
        _create_test_data(tmp_path, n_regular=20, n_honeypots=6)
        ds = EvaluationDataset(data_dir=tmp_path, domain="test_domain")

        samples_r0 = ds.get_round_samples(round_num=0, n=10, n_honeypots=2)
        samples_r1 = ds.get_round_samples(round_num=1, n=10, n_honeypots=2)

        ids_r0 = {s["id"] for s in samples_r0}
        ids_r1 = {s["id"] for s in samples_r1}

        # Different rounds should produce different sample sets
        assert ids_r0 != ids_r1

    def test_same_round_same_samples(self, tmp_path):
        _create_test_data(tmp_path, n_regular=20, n_honeypots=6)
        ds = EvaluationDataset(data_dir=tmp_path, domain="test_domain")

        samples_a = ds.get_round_samples(round_num=5, n=10, n_honeypots=2)
        samples_b = ds.get_round_samples(round_num=5, n=10, n_honeypots=2)

        ids_a = [s["id"] for s in samples_a]
        ids_b = [s["id"] for s in samples_b]

        # Same round number should produce identical samples in same order
        assert ids_a == ids_b


class TestHoneypotInjection:
    """Honeypot injection rate and manifest marking."""

    def test_honeypot_injection_rate(self, tmp_path):
        _create_test_data(tmp_path, n_regular=20, n_honeypots=6)
        ds = EvaluationDataset(data_dir=tmp_path, domain="test_domain")

        samples = ds.get_round_samples(round_num=0, n=10, n_honeypots=2)
        assert len(samples) == 10

        honeypots = [
            s for s in samples if ds.manifest[s["id"]].get("is_honeypot", False)
        ]
        regulars = [
            s for s in samples if not ds.manifest[s["id"]].get("is_honeypot", False)
        ]

        assert len(honeypots) == 2
        assert len(regulars) == 8

    def test_honeypots_marked_in_manifest(self, tmp_path):
        _create_test_data(tmp_path, n_regular=8, n_honeypots=4)
        ds = EvaluationDataset(data_dir=tmp_path, domain="test_domain")

        # All honeypot samples must have is_honeypot=True in manifest
        for sample in ds._honeypot_samples:
            entry = ds.manifest[sample["id"]]
            assert entry["is_honeypot"] is True


class TestDatasetVersionTracking:
    """Dataset exposes version derived from content hash."""

    def test_dataset_version_tracking(self, tmp_path):
        _create_test_data(tmp_path)
        ds = EvaluationDataset(data_dir=tmp_path, domain="test_domain")

        # Version should be a non-empty hex string (12 chars)
        assert isinstance(ds.dataset_version, str)
        assert len(ds.dataset_version) == 12
        # Should be valid hex
        int(ds.dataset_version, 16)


class TestSeedDataLoadable:
    """EvaluationDataset can load from the real seed data directory."""

    def test_seed_data_loadable(self):
        seed_dir = Path("data/evaluation")
        if not seed_dir.exists():
            pytest.skip("Seed data not yet created")

        ds = EvaluationDataset(data_dir=seed_dir, domain="hallucination")

        # Should have 60 samples total (30 original + 30 domain pack expansion)
        assert len(ds.samples) == 60

        # Should have 9 honeypots (6 original + 3 new)
        assert len(ds._honeypot_samples) == 9

        # Should have 51 regular samples
        assert len(ds._regular_samples) == 51

        # All samples should have required fields
        for sample in ds.samples:
            assert "id" in sample
            assert "prompt" in sample
            assert "output" in sample
            assert "domain" in sample
            assert sample["domain"] == "hallucination"
            assert "ground_truth_label" not in sample

        # All sample IDs should be in manifest
        for sample in ds.samples:
            assert sample["id"] in ds.manifest

        # Version should be set
        assert ds.dataset_version
