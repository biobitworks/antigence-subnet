"""Tests for the agent reasoning domain pack.

Covers chain-of-thought feature extraction, evaluation dataset integrity,
ReasoningDetector class, and registry integration.
"""

import json
import shutil
import tempfile

import pytest

# ------------------------------------------------------------------
# Test: Reasoning feature extraction
# ------------------------------------------------------------------


class TestReasoningFeatures:
    """Tests for extract_reasoning_features()."""

    def test_returns_dict_with_seven_keys(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        result = extract_reasoning_features("prompt", "output text")
        expected_keys = {
            "step_count",
            "logical_connective_density",
            "negation_count",
            "contradiction_score",
            "premise_conclusion_ratio",
            "avg_step_length",
            "total_length",
        }
        assert set(result.keys()) == expected_keys
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_step_count_with_numbered_steps(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        text = "Step 1: Identify the problem. Step 2: Analyze data. Therefore: conclude."
        result = extract_reasoning_features("", text)
        assert result["step_count"] >= 2

    def test_step_count_with_ordinal_markers(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        text = "First, we note the premise. Then, we derive the implication. Finally, we conclude."
        result = extract_reasoning_features("", text)
        assert result["step_count"] >= 2

    def test_logical_connective_density(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        text = "Because A is true, therefore B follows. However, since C, consequently D."
        result = extract_reasoning_features("", text)
        assert result["logical_connective_density"] > 0.0

    def test_negation_count(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        text = "This is not correct. We cannot assume this. It doesn't follow."
        result = extract_reasoning_features("", text)
        assert result["negation_count"] >= 3

    def test_contradiction_score_with_contradicting_text(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        text = "X is true. However, X is not true."
        result = extract_reasoning_features("", text)
        assert result["contradiction_score"] > 0

    def test_empty_input_returns_defaults(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        result = extract_reasoning_features("", "")
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"
        assert result["total_length"] == 0.0

    def test_premise_conclusion_ratio(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        text = "Given A, assuming B, suppose C. Therefore D."
        result = extract_reasoning_features("", text)
        # More premises than conclusions -> ratio > 1
        assert result["premise_conclusion_ratio"] >= 1.0

    def test_total_length(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.features import (
            extract_reasoning_features,
        )

        result = extract_reasoning_features("hello", "world")
        assert result["total_length"] == float(len("hello") + len("world"))


# ------------------------------------------------------------------
# Test: ReasoningDetector class
# ------------------------------------------------------------------


@pytest.fixture
def rea_normal_samples():
    """Load normal-only samples from the reasoning evaluation data."""
    with open("data/evaluation/reasoning/samples.json") as f:
        samples = json.load(f)["samples"]
    with open("data/evaluation/reasoning/manifest.json") as f:
        manifest = json.load(f)
    return [s for s in samples if manifest[s["id"]]["ground_truth_label"] == "normal"]


@pytest.fixture
def rea_anomalous_sample():
    """Return a single anomalous reasoning sample for detection testing."""
    return {
        "prompt": "If all mammals breathe air, and whales are mammals, what can we conclude?",
        "output": (
            "Step 1: All mammals breathe air. Step 2: Whales are mammals. "
            "Step 3: Therefore, whales breathe air. "
            "However, in Step 2 we proved whales are not mammals. "
            "This contradicts step 1, so mammals do not breathe air."
        ),
    }


class TestReasoningDetector:
    """Tests for ReasoningDetector class."""

    def test_domain_is_reasoning(self):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
            ReasoningDetector,
        )

        assert ReasoningDetector.domain == "reasoning"

    def test_fit_sets_is_fitted(self, rea_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
            ReasoningDetector,
        )

        detector = ReasoningDetector()
        assert not detector._is_fitted
        detector.fit(rea_normal_samples)
        assert detector._is_fitted

    @pytest.mark.asyncio
    async def test_detect_returns_valid_result(self, rea_normal_samples, rea_anomalous_sample):
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
            ReasoningDetector,
        )

        detector = ReasoningDetector()
        detector.fit(rea_normal_samples)
        result = await detector.detect(
            prompt=rea_anomalous_sample["prompt"],
            output=rea_anomalous_sample["output"],
        )
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.anomaly_type, str)

    @pytest.mark.asyncio
    async def test_detect_returns_feature_attribution(
        self, rea_normal_samples, rea_anomalous_sample,
    ):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
            ReasoningDetector,
        )

        detector = ReasoningDetector()
        detector.fit(rea_normal_samples)
        result = await detector.detect(
            prompt=rea_anomalous_sample["prompt"],
            output=rea_anomalous_sample["output"],
        )
        assert result.feature_attribution is not None
        domain_features = {
            "step_count",
            "logical_connective_density",
            "negation_count",
            "contradiction_score",
            "premise_conclusion_ratio",
            "avg_step_length",
            "total_length",
        }
        assert domain_features.issubset(set(result.feature_attribution.keys()))

    def test_get_info_after_fit(self, rea_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
            ReasoningDetector,
        )

        detector = ReasoningDetector()
        detector.fit(rea_normal_samples)
        info = detector.get_info()
        assert info["backend"] == "scikit-learn"
        assert info["is_fitted"] is True
        assert info["domain"] == "reasoning"
        assert info["name"] == "ReasoningDetector"

    def test_save_load_round_trip(self, rea_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
            ReasoningDetector,
        )

        detector = ReasoningDetector()
        detector.fit(rea_normal_samples)
        tmpdir = tempfile.mkdtemp()
        try:
            detector.save_state(tmpdir)
            detector2 = ReasoningDetector()
            assert not detector2._is_fitted
            detector2.load_state(tmpdir)
            assert detector2._is_fitted
            assert len(detector2._baseline_scores_sorted) == len(
                detector._baseline_scores_sorted
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_registry_integration(self):
        from antigence_subnet.miner.detectors import get_detector
        from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
            ReasoningDetector,
        )

        cls = get_detector("reasoning")
        assert cls is ReasoningDetector


# ------------------------------------------------------------------
# Test: Evaluation dataset integrity
# ------------------------------------------------------------------


class TestReasoningDataset:
    """Tests for reasoning evaluation data files."""

    SAMPLES_PATH = "data/evaluation/reasoning/samples.json"
    MANIFEST_PATH = "data/evaluation/reasoning/manifest.json"

    def test_samples_has_30_entries(self):
        with open(self.SAMPLES_PATH) as f:
            data = json.load(f)
        assert len(data["samples"]) == 220

    def test_manifest_has_entry_for_every_sample(self):
        with open(self.SAMPLES_PATH) as f:
            samples = json.load(f)["samples"]
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        sample_ids = {s["id"] for s in samples}
        manifest_ids = set(manifest.keys())
        assert sample_ids == manifest_ids

    def test_ground_truth_types_cover_required_categories(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        types = {v.get("ground_truth_type") for v in manifest.values()}
        required = {"logical_contradiction", "non_sequitur", "constraint_violation"}
        assert required.issubset(types)
        assert None in types

    def test_at_least_2_honeypots(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        honeypot_count = sum(1 for v in manifest.values() if v.get("is_honeypot"))
        assert honeypot_count >= 2

    def test_all_samples_have_reasoning_domain(self):
        with open(self.SAMPLES_PATH) as f:
            samples = json.load(f)["samples"]

        for s in samples:
            assert s["domain"] == "reasoning", f"Sample {s['id']} has domain={s['domain']}"

    def test_manifest_entries_have_correct_format(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        for _sample_id, entry in manifest.items():
            assert "ground_truth_label" in entry
            assert "ground_truth_type" in entry
            assert "is_honeypot" in entry
            assert entry["ground_truth_label"] in ("anomalous", "normal")
