"""Tests for the bio pipelines domain pack.

Covers bio feature extraction, evaluation dataset integrity,
BioDetector class, and registry integration.
"""

import json
import shutil
import tempfile

import pytest

# ------------------------------------------------------------------
# Test: Bio feature extraction
# ------------------------------------------------------------------


class TestBioFeatures:
    """Tests for extract_bio_features()."""

    def test_returns_dict_with_seven_keys(self):
        from antigence_subnet.miner.detectors.domain_packs.bio.features import (
            extract_bio_features,
        )

        result = extract_bio_features("Analyze protein levels", "concentration: 0.5 mM")
        expected_keys = {
            "numeric_value_count",
            "out_of_range_count",
            "z_score_outlier_count",
            "negative_value_count",
            "unit_mention_count",
            "value_magnitude_range",
            "statistical_summary_count",
        }
        assert set(result.keys()) == expected_keys
        # All values are floats
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_out_of_range_ph_and_negative_expression(self):
        from antigence_subnet.miner.detectors.domain_packs.bio.features import (
            extract_bio_features,
        )

        result = extract_bio_features("", "pH: 15.2, gene expression: -3.5")
        assert result["out_of_range_count"] >= 1.0  # pH > 14
        assert result["negative_value_count"] >= 1.0  # -3.5 is negative

    def test_unit_mentions(self):
        from antigence_subnet.miner.detectors.domain_packs.bio.features import (
            extract_bio_features,
        )

        result = extract_bio_features("", "concentration: 0.5 mM, temperature: 37 Celsius")
        assert result["unit_mention_count"] >= 2.0  # mM and Celsius

    def test_empty_input_returns_defaults_without_error(self):
        from antigence_subnet.miner.detectors.domain_packs.bio.features import (
            extract_bio_features,
        )

        result = extract_bio_features("", "")
        # Should not raise
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"
        # No numeric values in empty input
        assert result["numeric_value_count"] == 0.0

    def test_z_score_outlier_detection(self):
        from antigence_subnet.miner.detectors.domain_packs.bio.features import (
            extract_bio_features,
        )

        # Many similar values plus one extreme outlier -- 999 has z > 3 with enough peers
        result = extract_bio_features(
            "", "values: 0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 999.0"
        )
        assert result["z_score_outlier_count"] >= 1.0

    def test_statistical_summary_count(self):
        from antigence_subnet.miner.detectors.domain_packs.bio.features import (
            extract_bio_features,
        )

        result = extract_bio_features(
            "",
            "The mean value was 5.2 with a standard deviation of 0.3. "
            "ANOVA showed significance at p-value 0.01.",
        )
        assert result["statistical_summary_count"] >= 3.0  # mean, standard deviation, ANOVA/p-value

    def test_value_magnitude_range(self):
        from antigence_subnet.miner.detectors.domain_packs.bio.features import (
            extract_bio_features,
        )

        # Values: 0.001 and 1000 => log10(1000/0.001) = log10(1e6) = 6.0
        result = extract_bio_features("", "conc: 0.001 mM vs 1000 mM")
        assert result["value_magnitude_range"] > 5.0


# ------------------------------------------------------------------
# Test: Evaluation dataset integrity
# ------------------------------------------------------------------


class TestBioDataset:
    """Tests for bio evaluation data files."""

    SAMPLES_PATH = "data/evaluation/bio/samples.json"
    MANIFEST_PATH = "data/evaluation/bio/manifest.json"

    def test_samples_has_220_entries(self):
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
        required = {"value_out_of_range", "statistical_anomaly", "unit_inconsistency"}
        assert required.issubset(types)
        # Normal samples have null type
        assert None in types

    def test_honeypots_present(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        honeypot_count = sum(1 for v in manifest.values() if v.get("is_honeypot"))
        assert honeypot_count >= 2  # eval-bio-004 and eval-bio-022

    def test_all_samples_have_bio_domain(self):
        with open(self.SAMPLES_PATH) as f:
            samples = json.load(f)["samples"]

        for s in samples:
            assert s["domain"] == "bio", f"Sample {s['id']} has domain={s['domain']}"


# ------------------------------------------------------------------
# Test: BioDetector class
# ------------------------------------------------------------------


@pytest.fixture
def bio_normal_samples():
    """Load normal-only samples from the bio evaluation data."""
    with open("data/evaluation/bio/samples.json") as f:
        samples = json.load(f)["samples"]
    with open("data/evaluation/bio/manifest.json") as f:
        manifest = json.load(f)
    return [s for s in samples if manifest[s["id"]]["ground_truth_label"] == "normal"]


@pytest.fixture
def bio_anomalous_input():
    """Return an anomalous bio input for detection testing."""
    return {
        "prompt": "Report the pH measurement results for the protein buffer.",
        "output": (
            "The pH of the buffer solution was measured at 15.8, "
            "with concentration of -5.2 mM."
        ),
    }


class TestBioDetector:
    """Tests for BioDetector class."""

    def test_domain_is_bio(self):
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import (
            BioDetector,
        )

        assert BioDetector.domain == "bio"

    def test_fit_sets_is_fitted(self, bio_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import (
            BioDetector,
        )

        detector = BioDetector()
        assert not detector._is_fitted
        detector.fit(bio_normal_samples)
        assert detector._is_fitted

    @pytest.mark.asyncio
    async def test_detect_returns_valid_result(self, bio_normal_samples, bio_anomalous_input):
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import (
            BioDetector,
        )

        detector = BioDetector()
        detector.fit(bio_normal_samples)
        result = await detector.detect(
            prompt=bio_anomalous_input["prompt"],
            output=bio_anomalous_input["output"],
        )
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.anomaly_type, str)

    @pytest.mark.asyncio
    async def test_detect_returns_feature_attribution(
        self, bio_normal_samples, bio_anomalous_input,
    ):
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import (
            BioDetector,
        )

        detector = BioDetector()
        detector.fit(bio_normal_samples)
        result = await detector.detect(
            prompt=bio_anomalous_input["prompt"],
            output=bio_anomalous_input["output"],
        )
        assert result.feature_attribution is not None
        # Should contain domain-specific feature names
        domain_features = {
            "numeric_value_count",
            "out_of_range_count",
            "z_score_outlier_count",
            "negative_value_count",
            "unit_mention_count",
            "value_magnitude_range",
            "statistical_summary_count",
        }
        assert domain_features.issubset(set(result.feature_attribution.keys()))

    def test_get_info_after_fit(self, bio_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import (
            BioDetector,
        )

        detector = BioDetector()
        detector.fit(bio_normal_samples)
        info = detector.get_info()
        assert info["backend"] == "scikit-learn"
        assert info["is_fitted"] is True
        assert info["domain"] == "bio"
        assert info["name"] == "BioDetector"

    def test_save_load_round_trip(self, bio_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import (
            BioDetector,
        )

        detector = BioDetector()
        detector.fit(bio_normal_samples)
        tmpdir = tempfile.mkdtemp()
        try:
            detector.save_state(tmpdir)
            detector2 = BioDetector()
            assert not detector2._is_fitted
            detector2.load_state(tmpdir)
            assert detector2._is_fitted
            # Verify baseline scores match
            assert len(detector2._baseline_scores_sorted) == len(
                detector._baseline_scores_sorted
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_registry_integration(self):
        """Test that BioDetector is accessible via direct import."""
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import (
            BioDetector,
        )

        assert BioDetector.domain == "bio"
        # Verify it subclasses BaseDetector
        from antigence_subnet.miner.detector import BaseDetector

        assert issubclass(BioDetector, BaseDetector)
