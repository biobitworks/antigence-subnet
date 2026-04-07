"""Integration tests for all four domain packs working together.

Verifies that hallucination, code_security, reasoning, and bio domain packs
can all be loaded, fitted, and used for detection simultaneously. Also checks
evaluation dataset consistency across all domains.

Note: Some tests may be skipped if domain packs from parallel plans (07-02)
are not yet merged. All tests will pass once 07-02 and 07-03 are merged.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from antigence_subnet.miner.detector import BaseDetector
from antigence_subnet.protocol import KNOWN_DOMAINS, VerificationSynapse

# ---------------------------------------------------------------------------
# Helpers: dynamic domain pack loading with graceful skip
# ---------------------------------------------------------------------------

def _try_import_detector(domain: str):
    """Try to import a domain pack's detector class. Returns (class, skip_reason)."""
    try:
        if domain == "hallucination":
            from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
                HallucinationDetector,
            )
            return HallucinationDetector, None
        elif domain == "code_security":
            from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
                CodeSecurityDetector,
            )
            return CodeSecurityDetector, None
        elif domain == "reasoning":
            from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
                ReasoningDetector,
            )
            return ReasoningDetector, None
        elif domain == "bio":
            from antigence_subnet.miner.detectors.domain_packs.bio.detector import BioDetector
            return BioDetector, None
        else:
            return None, f"Unknown domain: {domain}"
    except ImportError as e:
        return None, f"Domain pack '{domain}' not yet available: {e}"


def _all_domains_available() -> tuple[bool, str]:
    """Check if all 4 domain packs are importable."""
    missing = []
    for domain in KNOWN_DOMAINS:
        cls, reason = _try_import_detector(domain)
        if cls is None:
            missing.append(domain)
    if missing:
        return False, f"Missing domain packs: {', '.join(sorted(missing))}"
    return True, ""


# Test inputs per domain for detection testing
_TEST_INPUTS = {
    "hallucination": {
        "prompt": "Who wrote Hamlet?",
        "output": "Hamlet was written by Charles Dickens.",
        "code": None,
    },
    "code_security": {
        "prompt": "Write a query function",
        "output": "Here is a function that queries user input",
        "code": "import os; os.system(input())",
    },
    "reasoning": {
        "prompt": "If A > B and B > C, what is the relationship between A and C?",
        "output": "Step 1: A > B. Step 2: B > C. Therefore C > A.",
        "code": None,
    },
    "bio": {
        "prompt": "Report pH measurement results",
        "output": "The pH of the sample was measured at 15.8.",
        "code": None,
    },
}


# Evaluation data root
_EVAL_DATA_DIR = "data/evaluation"


# ---------------------------------------------------------------------------
# Conditional skip for tests that need all 4 domains
# ---------------------------------------------------------------------------

_all_available, _skip_reason = _all_domains_available()
requires_all_domains = pytest.mark.skipif(
    not _all_available,
    reason=_skip_reason or "Not all domain packs available",
)


# ---------------------------------------------------------------------------
# Test: All domains registered (requires all 4 packs merged)
# ---------------------------------------------------------------------------


@requires_all_domains
class TestAllDomainsRegistered:
    """Assert DETECTOR_REGISTRY contains exactly the 4 domains."""

    def test_registry_has_all_four_domains(self):
        from antigence_subnet.miner.detectors import DETECTOR_REGISTRY

        registered = set(DETECTOR_REGISTRY.keys())
        expected = {"hallucination", "code_security", "reasoning", "bio"}
        assert registered == expected, (
            f"Registry mismatch: expected {expected}, got {registered}"
        )

    def test_registry_maps_to_correct_classes(self):
        # Registry is now list-based (ensemble support) -- use get_detector() for backward compat
        from antigence_subnet.miner.detectors import get_detector
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import BioDetector
        from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
            CodeSecurityDetector,
        )
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )
        from antigence_subnet.miner.detectors.domain_packs.reasoning.detector import (
            ReasoningDetector,
        )

        assert get_detector("hallucination") is HallucinationDetector
        assert get_detector("code_security") is CodeSecurityDetector
        assert get_detector("reasoning") is ReasoningDetector
        assert get_detector("bio") is BioDetector


# ---------------------------------------------------------------------------
# Test: Registered domains match KNOWN_DOMAINS (requires all 4 packs)
# ---------------------------------------------------------------------------


@requires_all_domains
class TestAllDomainsMatchKnownDomains:
    """Assert registered domains equals KNOWN_DOMAINS from protocol.py."""

    def test_registered_equals_known(self):
        from antigence_subnet.miner.detectors import DETECTOR_REGISTRY

        registered = set(DETECTOR_REGISTRY.keys())
        assert registered == KNOWN_DOMAINS, (
            f"Mismatch: registered={registered}, KNOWN_DOMAINS={KNOWN_DOMAINS}"
        )


# ---------------------------------------------------------------------------
# Test: All domains fit and detect (requires all 4 packs + eval data)
# ---------------------------------------------------------------------------


@requires_all_domains
class TestAllDomainsFitAndDetect:
    """For each domain, instantiate detector, fit, and detect."""

    @pytest.mark.asyncio
    async def test_all_domains_produce_valid_results(self):
        from antigence_subnet.miner.data import load_training_samples

        for domain in sorted(KNOWN_DOMAINS):
            cls, reason = _try_import_detector(domain)
            assert cls is not None, f"Failed to import {domain}: {reason}"

            # Load training data
            samples = load_training_samples(_EVAL_DATA_DIR, domain)
            assert len(samples) > 0, f"No training samples for {domain}"

            # Instantiate and fit
            detector = cls()
            detector.fit(samples)

            # Detect
            test_input = _TEST_INPUTS[domain]
            result = await detector.detect(
                prompt=test_input["prompt"],
                output=test_input["output"],
                code=test_input.get("code"),
            )

            # Validate result
            assert 0.0 <= result.score <= 1.0, (
                f"{domain}: score {result.score} out of range"
            )
            assert 0.0 <= result.confidence <= 1.0, (
                f"{domain}: confidence {result.confidence} out of range"
            )
            assert isinstance(result.anomaly_type, str) and len(result.anomaly_type) > 0, (
                f"{domain}: anomaly_type empty or not string"
            )
            assert (
                isinstance(result.feature_attribution, dict)
                and len(result.feature_attribution) >= 1
            ), (
                f"{domain}: feature_attribution missing or empty"
            )


# ---------------------------------------------------------------------------
# Test: Domain routing through forward.py (requires all 4 packs)
# ---------------------------------------------------------------------------


@requires_all_domains
class TestDomainRouting:
    """Create a mock miner with all 4 domain detectors and test routing."""

    @pytest.mark.asyncio
    async def test_forward_routes_all_domains(self):
        from antigence_subnet.miner.data import load_training_samples
        from antigence_subnet.miner.forward import forward as miner_forward

        # Build detectors dict with all 4 domain packs fitted
        detectors = {}
        for domain in sorted(KNOWN_DOMAINS):
            cls, reason = _try_import_detector(domain)
            assert cls is not None, f"Failed to import {domain}: {reason}"
            samples = load_training_samples(_EVAL_DATA_DIR, domain)
            detector = cls()
            detector.fit(samples)
            detectors[domain] = detector

        # Mock miner
        miner = MagicMock()
        miner.detectors = detectors
        miner.supported_domains = set(detectors.keys())
        miner.orchestrator = None  # No orchestrator (use ensemble path)

        # Route each domain
        for domain in sorted(KNOWN_DOMAINS):
            test_input = _TEST_INPUTS[domain]
            synapse = VerificationSynapse(
                prompt=test_input["prompt"],
                output=test_input["output"],
                domain=domain,
                code=test_input.get("code"),
            )
            result_synapse = await miner_forward(miner, synapse)
            assert result_synapse.anomaly_score is not None, (
                f"{domain}: anomaly_score not populated after forward"
            )
            assert 0.0 <= result_synapse.anomaly_score <= 1.0, (
                f"{domain}: anomaly_score {result_synapse.anomaly_score} out of range"
            )


# ---------------------------------------------------------------------------
# Test: Evaluation dataset consistency across all domains
# ---------------------------------------------------------------------------


class TestEvaluationDatasetConsistency:
    """Verify eval datasets for each available domain."""

    @pytest.mark.parametrize("domain", sorted(KNOWN_DOMAINS))
    def test_dataset_structure(self, domain):
        samples_path = Path(_EVAL_DATA_DIR) / domain / "samples.json"
        manifest_path = Path(_EVAL_DATA_DIR) / domain / "manifest.json"

        if not samples_path.exists():
            pytest.skip(f"Evaluation data for '{domain}' not yet available")

        # samples.json exists and is valid
        with open(samples_path) as f:
            data = json.load(f)
        assert "samples" in data, f"{domain}: samples.json missing 'samples' key"
        samples = data["samples"]
        assert len(samples) > 0, f"{domain}: no samples"

        # manifest.json exists and covers all sample ids
        assert manifest_path.exists(), f"{domain}: manifest.json missing"
        with open(manifest_path) as f:
            manifest = json.load(f)

        sample_ids = {s["id"] for s in samples}
        manifest_ids = set(manifest.keys())
        assert sample_ids == manifest_ids, (
            f"{domain}: sample/manifest ID mismatch. "
            f"In samples only: {sample_ids - manifest_ids}. "
            f"In manifest only: {manifest_ids - sample_ids}"
        )

        # At least one honeypot per domain
        honeypot_count = sum(1 for v in manifest.values() if v.get("is_honeypot"))
        assert honeypot_count >= 1, f"{domain}: no honeypots found"

        # At least one normal and one anomalous label
        labels = {v.get("ground_truth_label") for v in manifest.values()}
        assert "normal" in labels, f"{domain}: no normal samples"
        assert "anomalous" in labels, f"{domain}: no anomalous samples"


# ---------------------------------------------------------------------------
# Test: Domains available for this plan (bio + hallucination at minimum)
# ---------------------------------------------------------------------------


class TestAvailableDomainPacks:
    """Tests that don't require all 4 domains -- verifies what's available now."""

    def test_hallucination_and_bio_importable(self):
        """At minimum, hallucination (07-01) and bio (07-03) are available."""
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import BioDetector
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        assert HallucinationDetector.domain == "hallucination"
        assert BioDetector.domain == "bio"
        assert issubclass(HallucinationDetector, BaseDetector)
        assert issubclass(BioDetector, BaseDetector)

    @pytest.mark.asyncio
    async def test_hallucination_and_bio_fit_detect(self):
        """Verify hallucination and bio packs fit and detect together."""
        from antigence_subnet.miner.data import load_training_samples
        from antigence_subnet.miner.detectors.domain_packs.bio.detector import BioDetector
        from antigence_subnet.miner.detectors.domain_packs.hallucination.detector import (
            HallucinationDetector,
        )

        detectors = {}
        for domain, cls in [("hallucination", HallucinationDetector), ("bio", BioDetector)]:
            samples = load_training_samples(_EVAL_DATA_DIR, domain)
            det = cls()
            det.fit(samples)
            detectors[domain] = det

        # Detect with hallucination
        h_result = await detectors["hallucination"].detect(
            prompt="Who wrote Hamlet?",
            output="Hamlet was written by Charles Dickens.",
        )
        assert 0.0 <= h_result.score <= 1.0

        # Detect with bio
        b_result = await detectors["bio"].detect(
            prompt="Report pH measurement results",
            output="The pH of the sample was measured at 15.8.",
        )
        assert 0.0 <= b_result.score <= 1.0
