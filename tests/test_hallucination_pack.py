"""Tests for the hallucination domain pack.

Covers feature extraction, evaluation dataset integrity,
HallucinationDetector class, and registry integration.
"""

import json
import shutil
import tempfile

import pytest

# ------------------------------------------------------------------
# Test: Hallucination feature extraction
# ------------------------------------------------------------------


class TestHallucinationFeatures:
    """Tests for extract_hallucination_features()."""

    def test_returns_dict_with_seven_keys(self):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.features import (
            extract_hallucination_features,
        )

        result = extract_hallucination_features("prompt", "output text")
        expected_keys = {
            "claim_density",
            "citation_count",
            "citation_pattern_score",
            "hedging_ratio",
            "numeric_claim_density",
            "avg_sentence_length",
            "text_length",
        }
        assert set(result.keys()) == expected_keys
        # All values are floats
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_citation_extraction(self):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.features import (
            extract_hallucination_features,
        )

        text = "According to Smith et al. (2020) and Jones (2019), the result is 42."
        result = extract_hallucination_features("", text)
        assert result["citation_count"] >= 2.0
        assert result["numeric_claim_density"] > 0.0

    def test_empty_input_returns_defaults(self):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.features import (
            extract_hallucination_features,
        )

        result = extract_hallucination_features("", "")
        # Should not raise
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"
        # Empty input should have zero text_length for the combined text
        assert result["text_length"] == 0.0

    def test_hedging_ratio_with_hedging_words(self):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.features import (
            extract_hallucination_features,
        )

        text = "This may be correct. It might work. Possibly true."
        result = extract_hallucination_features("", text)
        # All 3 sentences have hedging words
        assert result["hedging_ratio"] > 0.5

    def test_numeric_claim_density(self):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.features import (
            extract_hallucination_features,
        )

        text = "The temperature is 37.5 degrees and the pressure is 101325 pascals."
        result = extract_hallucination_features("", text)
        assert result["numeric_claim_density"] > 0.0

    def test_bracket_citations(self):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.features import (
            extract_hallucination_features,
        )

        text = "This is supported by evidence [1] and further confirmed [2]."
        result = extract_hallucination_features("", text)
        assert result["citation_count"] >= 2.0


# ------------------------------------------------------------------
# Test: Evaluation dataset integrity
# ------------------------------------------------------------------


class TestHallucinationDataset:
    """Tests for hallucination evaluation data files."""

    SAMPLES_PATH = "data/evaluation/hallucination/samples.json"
    MANIFEST_PATH = "data/evaluation/hallucination/manifest.json"

    def test_samples_has_at_least_45_entries(self):
        with open(self.SAMPLES_PATH) as f:
            data = json.load(f)
        assert len(data["samples"]) >= 45

    def test_manifest_has_entry_for_every_sample(self):
        with open(self.SAMPLES_PATH) as f:
            samples = json.load(f)["samples"]
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        sample_ids = {s["id"] for s in samples}
        manifest_ids = set(manifest.keys())
        # Every sample must have a manifest entry
        assert sample_ids == manifest_ids

    def test_ground_truth_types_cover_required_categories(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        types = {v.get("ground_truth_type") for v in manifest.values()}
        required = {
            "fabricated_citation", "unsupported_claim",
            "hallucinated_fact", "factual_error",
        }
        assert required.issubset(types)
        # Normal samples have null type
        assert None in types

    def test_at_least_5_honeypots(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        honeypot_count = sum(1 for v in manifest.values() if v.get("is_honeypot"))
        assert honeypot_count >= 5

    def test_all_samples_have_hallucination_domain(self):
        with open(self.SAMPLES_PATH) as f:
            samples = json.load(f)["samples"]

        for s in samples:
            assert s["domain"] == "hallucination", f"Sample {s['id']} has domain={s['domain']}"


# ------------------------------------------------------------------
# Test: HallucinationDetector class
# ------------------------------------------------------------------


@pytest.fixture
def normal_samples():
    """Load normal-only samples from the hallucination evaluation data."""
    with open("data/evaluation/hallucination/samples.json") as f:
        samples = json.load(f)["samples"]
    with open("data/evaluation/hallucination/manifest.json") as f:
        manifest = json.load(f)
    return [s for s in samples if manifest[s["id"]]["ground_truth_label"] == "normal"]


@pytest.fixture
def anomalous_sample():
    """Return a single anomalous sample for detection testing."""
    return {
        "prompt": "What are the key findings of recent research on transformer architectures?",
        "output": (
            "According to Richardson and Yamamoto (2023), transformer attention mechanisms "
            "exhibit logarithmic scaling. Petersfield et al. (2024) confirmed a 47% improvement. "
            "Karenina and Oblensky (2021) predicted these results within 3% margin of error."
        ),
    }


class TestHallucinationDetector:
    """Tests for HallucinationDetector class."""

    def test_domain_is_hallucination(self):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        assert HallucinationDetector.domain == "hallucination"

    def test_fit_sets_is_fitted(self, normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector()
        assert not detector._is_fitted
        detector.fit(normal_samples)
        assert detector._is_fitted

    @pytest.mark.asyncio
    async def test_detect_returns_valid_result(self, normal_samples, anomalous_sample):
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector()
        detector.fit(normal_samples)
        result = await detector.detect(
            prompt=anomalous_sample["prompt"],
            output=anomalous_sample["output"],
        )
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.anomaly_type, str)

    @pytest.mark.asyncio
    async def test_detect_returns_feature_attribution(self, normal_samples, anomalous_sample):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector()
        detector.fit(normal_samples)
        result = await detector.detect(
            prompt=anomalous_sample["prompt"],
            output=anomalous_sample["output"],
        )
        assert result.feature_attribution is not None
        # Should contain domain-specific feature names
        domain_features = {
            "claim_density",
            "citation_count",
            "citation_pattern_score",
            "hedging_ratio",
            "numeric_claim_density",
            "avg_sentence_length",
            "text_length",
        }
        assert domain_features.issubset(set(result.feature_attribution.keys()))

    def test_get_info_after_fit(self, normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector()
        detector.fit(normal_samples)
        info = detector.get_info()
        assert info["backend"] == "scikit-learn"
        assert info["is_fitted"] is True
        assert info["domain"] == "hallucination"
        assert info["name"] == "HallucinationDetector"

    def test_save_load_round_trip(self, normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector()
        detector.fit(normal_samples)
        tmpdir = tempfile.mkdtemp()
        try:
            detector.save_state(tmpdir)
            detector2 = HallucinationDetector()
            assert not detector2._is_fitted
            detector2.load_state(tmpdir)
            assert detector2._is_fitted
            # Verify baseline scores match
            assert len(detector2._baseline_scores_sorted) == len(detector._baseline_scores_sorted)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_registry_integration(self):
        from antigence_subnet.miner.detectors import get_detector
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        cls = get_detector("hallucination")
        assert cls is HallucinationDetector


# ------------------------------------------------------------------
# Test: HallucinationDetector sbert integration
# ------------------------------------------------------------------


class TestHallucinationDetectorSbert:
    """Tests for HallucinationDetector with sbert embedding path."""

    def _skip_if_no_sbert(self):
        pytest.importorskip(
            "sentence_transformers",
            reason="sbert tests require sentence-transformers",
        )

    def test_sbert_fit_and_detect(self, normal_samples, anomalous_sample):
        """Test 1: HallucinationDetector(embedding_method='sbert') fits and detects."""
        self._skip_if_no_sbert()
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector(embedding_method="sbert")
        detector.fit(normal_samples)
        assert detector._is_fitted
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            detector.detect(prompt=anomalous_sample["prompt"], output=anomalous_sample["output"])
        )
        assert 0.0 <= result.score <= 1.0

    def test_tfidf_backward_compat(self, normal_samples, anomalous_sample):
        """Test 2: HallucinationDetector(embedding_method='tfidf') works as before."""
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector(embedding_method="tfidf")
        detector.fit(normal_samples)
        assert detector._is_fitted
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            detector.detect(prompt=anomalous_sample["prompt"], output=anomalous_sample["output"])
        )
        assert 0.0 <= result.score <= 1.0

    def test_default_is_sbert_when_available(self):
        """Test 3: Default embedding_method is sbert when sentence-transformers installed."""
        self._skip_if_no_sbert()
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector()
        assert detector.embedding_method == "sbert"

    def test_fallback_to_tfidf_with_warning(self):
        """Test 4: Falls back to tfidf with warning when sbert not available."""
        import warnings

        import antigence_subnet.miner.detectors.embeddings as emb_module
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        original = emb_module._sbert_available
        try:
            emb_module._sbert_available = False
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                detector = HallucinationDetector(embedding_method="sbert")
                assert detector.embedding_method == "tfidf"
                assert len(w) == 1
                assert "sentence-transformers not installed" in str(w[0].message)
        finally:
            emb_module._sbert_available = original

    def test_get_info_includes_embedding_method(self, normal_samples):
        """Test 5: get_info() includes embedding_method key."""
        self._skip_if_no_sbert()
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector(embedding_method="sbert")
        detector.fit(normal_samples)
        info = detector.get_info()
        assert "embedding_method" in info
        assert info["embedding_method"] == "sbert"

    def test_save_load_preserves_embedding_method(self, normal_samples):
        """Test 6: save_state/load_state round-trip preserves embedding_method."""
        self._skip_if_no_sbert()
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detector = HallucinationDetector(embedding_method="sbert")
        detector.fit(normal_samples)
        tmpdir = tempfile.mkdtemp()
        try:
            detector.save_state(tmpdir)
            detector2 = HallucinationDetector(embedding_method="sbert")
            detector2.load_state(tmpdir)
            assert detector2.embedding_method == "sbert"
            assert detector2._is_fitted
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ------------------------------------------------------------------
# Test: sbert vs tfidf F1/AUC comparison
# ------------------------------------------------------------------


def test_sbert_vs_tfidf_f1():
    """Compare F1 and AUC for sbert vs tfidf on hallucination eval data.

    Trains both methods on normal samples, evaluates on all samples,
    and asserts sbert F1 >= tfidf F1 within 0.05 tolerance.
    """
    pytest.importorskip(
        "sentence_transformers",
        reason="sbert comparison requires sentence-transformers",
    )

    import asyncio

    from sklearn.metrics import f1_score, roc_auc_score

    from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
        HallucinationDetector,
    )

    # Load eval data
    with open("data/evaluation/hallucination/samples.json") as f:
        all_samples = json.load(f)["samples"]
    with open("data/evaluation/hallucination/manifest.json") as f:
        manifest = json.load(f)

    normal_samples = [s for s in all_samples if manifest[s["id"]]["ground_truth_label"] == "normal"]
    all_labels = [
        0 if manifest[s["id"]]["ground_truth_label"] == "normal" else 1 for s in all_samples
    ]

    # Train both detectors
    det_sbert = HallucinationDetector(embedding_method="sbert")
    det_tfidf = HallucinationDetector(embedding_method="tfidf")
    det_sbert.fit(normal_samples)
    det_tfidf.fit(normal_samples)

    # Detect on all samples
    loop = asyncio.get_event_loop()
    scores_sbert = []
    scores_tfidf = []
    for s in all_samples:
        r_sbert = loop.run_until_complete(
            det_sbert.detect(prompt=s.get("prompt", ""), output=s.get("output", ""))
        )
        r_tfidf = loop.run_until_complete(
            det_tfidf.detect(prompt=s.get("prompt", ""), output=s.get("output", ""))
        )
        scores_sbert.append(r_sbert.score)
        scores_tfidf.append(r_tfidf.score)

    # Compute metrics
    preds_sbert = [1 if s >= 0.5 else 0 for s in scores_sbert]
    preds_tfidf = [1 if s >= 0.5 else 0 for s in scores_tfidf]

    f1_sbert = f1_score(all_labels, preds_sbert, zero_division=0.0)
    f1_tfidf = f1_score(all_labels, preds_tfidf, zero_division=0.0)

    # AUC requires both classes in y_true
    try:
        auc_sbert = roc_auc_score(all_labels, scores_sbert)
        auc_tfidf = roc_auc_score(all_labels, scores_tfidf)
    except ValueError:
        auc_sbert = 0.0
        auc_tfidf = 0.0

    print(f"TF-IDF: F1={f1_tfidf:.3f} AUC={auc_tfidf:.3f}")
    print(f"SBERT:  F1={f1_sbert:.3f} AUC={auc_sbert:.3f}")

    # sbert must be at least as good within tolerance
    assert f1_sbert >= f1_tfidf - 0.05, (
        f"SBERT F1 ({f1_sbert:.3f}) worse than TF-IDF F1 ({f1_tfidf:.3f}) by more than 0.05"
    )
