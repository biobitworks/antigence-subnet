"""Smoke tests for NegSel integration into benchmark scripts.

Validates:
  1. NegSelAISDetector present in benchmark_detectors GENERIC_DETECTORS
  2. Simulated NegSelDetector class fully removed from benchmark_all_strategies
  3. NegSelAISDetector runs single-domain detection on hallucination data
  4. NegSel ensemble pair (NegSel + IsolationForest) produces valid result
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from antigence_subnet.miner.detector import DetectionResult
from antigence_subnet.miner.detectors.negsel import NegSelAISDetector
from antigence_subnet.miner.ensemble import ensemble_detect

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "hallucination"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hallucination_data():
    """Load hallucination evaluation data (samples + manifest)."""
    with open(DATA_DIR / "samples.json") as f:
        samples = json.load(f)["samples"]
    with open(DATA_DIR / "manifest.json") as f:
        manifest = json.load(f)
    return samples, manifest


@pytest.fixture
def normal_samples(hallucination_data):
    """Return only normal-labeled samples for training."""
    samples, manifest = hallucination_data
    return [
        s for s in samples
        if manifest.get(s["id"], {}).get("ground_truth_label") == "normal"
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_negsel_in_generic_detectors():
    """GENERIC_DETECTORS in benchmark_detectors.py includes NegSel -> NegSelAISDetector."""
    from scripts.benchmark_detectors import GENERIC_DETECTORS

    assert "NegSel" in GENERIC_DETECTORS, "NegSel key missing from GENERIC_DETECTORS"
    assert GENERIC_DETECTORS["NegSel"] is NegSelAISDetector, (
        f"Expected NegSelAISDetector, got {GENERIC_DETECTORS['NegSel']}"
    )


def test_simulated_negsel_removed():
    """Simulated NegSelDetector class must not exist in benchmark_all_strategies.py."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_all_strategies.py"
    content = script_path.read_text()
    assert "class NegSelDetector(BaseDetector)" not in content, (
        "Simulated NegSelDetector class still present in benchmark_all_strategies.py"
    )
    # Confirm real import is present
    assert "from antigence_subnet.miner.detectors.negsel import NegSelAISDetector" in content


async def test_negsel_runs_single_domain(hallucination_data, normal_samples):
    """NegSelAISDetector fits and detects on hallucination domain samples."""
    samples, manifest = hallucination_data

    negsel = NegSelAISDetector(num_detectors=50, random_state=42)
    negsel.fit(normal_samples)

    # Mix of 3 normal + 2 anomalous samples
    test_samples = []
    normal_count = 0
    anom_count = 0
    for s in samples:
        label = manifest.get(s["id"], {}).get("ground_truth_label", "normal")
        if label == "normal" and normal_count < 3:
            test_samples.append(s)
            normal_count += 1
        elif label == "anomalous" and anom_count < 2:
            test_samples.append(s)
            anom_count += 1
        if normal_count >= 3 and anom_count >= 2:
            break

    assert len(test_samples) == 5, f"Expected 5 test samples, got {len(test_samples)}"

    for s in test_samples:
        result = await negsel.detect(
            prompt=s.get("prompt", ""),
            output=s.get("output", ""),
            code=s.get("code"),
            context=s.get("context"),
        )
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of range"
        assert 0.0 <= result.confidence <= 1.0, f"Confidence {result.confidence} out of range"

    # Verify this is the real detector, not the simulated one
    info = negsel.get_info()
    assert info["backend"] == "negsel-ais", (
        f"Expected backend 'negsel-ais', got '{info.get('backend')}'"
    )


async def test_negsel_ensemble_pair(normal_samples):
    """NegSel + IsolationForest ensemble produces valid DetectionResult."""
    from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector

    negsel = NegSelAISDetector(num_detectors=50, random_state=42)
    negsel.fit(normal_samples)

    iforest = IsolationForestDetector(random_state=42)
    iforest.fit(normal_samples)

    result = await ensemble_detect(
        [negsel, iforest],
        prompt="What is the capital of France?",
        output="The capital of France is Berlin, which was established in 1823.",
    )
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.score <= 1.0, f"Ensemble score {result.score} out of range"
    assert 0.0 <= result.confidence <= 1.0, f"Ensemble confidence {result.confidence} out of range"
