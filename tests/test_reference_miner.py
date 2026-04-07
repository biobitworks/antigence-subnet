"""
Tests for reference miner with config-based detector loading (DET-06, DET-08).

Verifies training data loader, config-based detector loading via importlib,
reference miner end-to-end synapse processing, state persistence, and
third-party detector pluggability.
"""

import json
import os
import sys

import pytest

from antigence_subnet.miner.detector import BaseDetector, DetectionResult

_torch_available = True
try:
    import torch as _torch  # noqa: F401
except ImportError:
    _torch_available = False


# ---------------------------------------------------------------------------
# TestLoadTrainingSamples
# ---------------------------------------------------------------------------
class TestLoadTrainingSamples:
    """Tests for antigence_subnet.miner.data.load_training_samples."""

    def test_loads_normal_only(self):
        """load_training_samples returns only normal samples from seed data."""
        from antigence_subnet.miner.data import load_training_samples

        samples = load_training_samples("data/evaluation", "hallucination")
        assert len(samples) == 108  # 108 normal samples in expanded 220-sample dataset

    def test_samples_have_required_keys(self):
        """Each returned sample has 'prompt' and 'output' keys."""
        from antigence_subnet.miner.data import load_training_samples

        samples = load_training_samples("data/evaluation", "hallucination")
        for s in samples:
            assert "prompt" in s, f"Sample missing 'prompt' key: {s.get('id')}"
            assert "output" in s, f"Sample missing 'output' key: {s.get('id')}"

    def test_file_not_found(self):
        """Raises FileNotFoundError for nonexistent directory."""
        from antigence_subnet.miner.data import load_training_samples

        with pytest.raises(FileNotFoundError):
            load_training_samples("/nonexistent/path", "hallucination")

    def test_empty_when_no_normal(self, tmp_path):
        """Returns empty list when all samples are anomalous."""
        from antigence_subnet.miner.data import load_training_samples

        domain_dir = tmp_path / "test_domain"
        domain_dir.mkdir()

        samples_data = {
            "samples": [
                {"id": "s1", "prompt": "p", "output": "o", "domain": "test"},
                {"id": "s2", "prompt": "p", "output": "o", "domain": "test"},
            ]
        }
        manifest_data = {
            "s1": {
                "ground_truth_label": "anomalous",
                "ground_truth_type": "error", "is_honeypot": False,
            },
            "s2": {
                "ground_truth_label": "anomalous",
                "ground_truth_type": "error", "is_honeypot": False,
            },
        }

        with open(domain_dir / "samples.json", "w") as f:
            json.dump(samples_data, f)
        with open(domain_dir / "manifest.json", "w") as f:
            json.dump(manifest_data, f)

        result = load_training_samples(str(tmp_path), "test_domain")
        assert result == []


# ---------------------------------------------------------------------------
# TestLoadDetector
# ---------------------------------------------------------------------------
class TestLoadDetector:
    """Tests for neurons.miner.load_detector."""

    def test_load_iforest(self):
        """Loads IsolationForestDetector by full class path."""
        from antigence_subnet.miner.detectors.isolation_forest import IsolationForestDetector
        from neurons.miner import load_detector

        cls_path = "antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector"
        det = load_detector(cls_path)
        assert isinstance(det, IsolationForestDetector)
        assert isinstance(det, BaseDetector)

    @pytest.mark.skipif(
        not _torch_available,
        reason="torch not installed",
    )
    def test_load_autoencoder(self):
        """Loads AutoencoderDetector by full class path."""
        from antigence_subnet.miner.detectors.autoencoder import AutoencoderDetector
        from neurons.miner import load_detector

        cls_path = "antigence_subnet.miner.detectors.autoencoder.AutoencoderDetector"
        det = load_detector(cls_path)
        assert isinstance(det, AutoencoderDetector)
        assert isinstance(det, BaseDetector)

    def test_invalid_module(self):
        """Raises ImportError for bad module path."""
        from neurons.miner import load_detector

        with pytest.raises(ImportError):
            load_detector("nonexistent.module.Detector")

    def test_invalid_class(self):
        """Raises AttributeError for bad class name."""
        from neurons.miner import load_detector

        with pytest.raises(AttributeError):
            load_detector(
                "antigence_subnet.miner.detectors.isolation_forest.NonexistentClass"
            )

    def test_returns_base_detector(self):
        """Loaded instance passes isinstance(result, BaseDetector) check."""
        from neurons.miner import load_detector

        cls_path = "antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector"
        det = load_detector(cls_path)
        assert isinstance(det, BaseDetector)


# ---------------------------------------------------------------------------
# TestReferenceMiner
# ---------------------------------------------------------------------------
class TestReferenceMiner:
    """Tests for reference Miner with config-based detector loading."""

    def test_miner_init_fits_detector(self, mock_config):
        """Miner instantiation with mock config loads and fits detector."""
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        assert "hallucination" in miner.detectors
        detector = miner.detectors["hallucination"]
        assert detector.get_info()["is_fitted"] is True

    @pytest.mark.asyncio
    async def test_forward_populates_score(self, mock_config):
        """forward() populates anomaly_score as float in [0.0, 1.0]."""
        from antigence_subnet.protocol import VerificationSynapse
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        synapse = VerificationSynapse(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
            domain="hallucination",
        )
        result = await miner.forward(synapse)
        assert result.anomaly_score is not None
        assert isinstance(result.anomaly_score, float)
        assert 0.0 <= result.anomaly_score <= 1.0

    @pytest.mark.asyncio
    async def test_forward_populates_confidence(self, mock_config):
        """forward() populates confidence as float in [0.0, 1.0]."""
        from antigence_subnet.protocol import VerificationSynapse
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        synapse = VerificationSynapse(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
            domain="hallucination",
        )
        result = await miner.forward(synapse)
        assert result.confidence is not None
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_forward_populates_anomaly_type(self, mock_config):
        """forward() populates anomaly_type as non-empty string."""
        from antigence_subnet.protocol import VerificationSynapse
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        synapse = VerificationSynapse(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
            domain="hallucination",
        )
        result = await miner.forward(synapse)
        assert result.anomaly_type is not None
        assert isinstance(result.anomaly_type, str)

    def test_supported_domains(self, mock_config):
        """Miner supports 'hallucination' in supported_domains."""
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        assert "hallucination" in miner.supported_domains

    @pytest.mark.asyncio
    async def test_rejects_unsupported_domain(self, mock_config):
        """forward() rejects unsupported domain with 400 status."""
        from antigence_subnet.protocol import VerificationSynapse
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        synapse = VerificationSynapse(
            prompt="Test",
            output="Test output",
            domain="unknown_domain",
        )
        result = await miner.forward(synapse)
        assert result.axon.status_code == 400

    @pytest.mark.asyncio
    async def test_forward_feature_attribution(self, mock_config):
        """forward() populates feature_attribution (may be None or dict)."""
        from antigence_subnet.protocol import VerificationSynapse
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        synapse = VerificationSynapse(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
            domain="hallucination",
        )
        result = await miner.forward(synapse)
        # feature_attribution is either None or a dict of str -> float
        if result.feature_attribution is not None:
            assert isinstance(result.feature_attribution, dict)
            for k, v in result.feature_attribution.items():
                assert isinstance(k, str)
                assert isinstance(v, float)


# ---------------------------------------------------------------------------
# TestMinerStatePersistence
# ---------------------------------------------------------------------------
class TestMinerStatePersistence:
    """Tests for miner save_state/load_state with detector persistence."""

    def test_save_creates_directory(self, mock_config):
        """save_state() creates the detector_state directory."""
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        miner.save_state()
        state_dir = os.path.join(mock_config.neuron.full_path, "detector_state")
        assert os.path.isdir(state_dir)

    @pytest.mark.asyncio
    async def test_load_restores_detector(self, mock_config):
        """load_state() restores detector that produces valid scores."""
        from antigence_subnet.protocol import VerificationSynapse
        from neurons.miner import Miner

        # Save state from first miner
        miner1 = Miner(config=mock_config)
        miner1.save_state()

        # Create second miner and load state
        miner2 = Miner(config=mock_config)
        miner2.load_state()

        # Verify the loaded detector produces valid scores
        synapse = VerificationSynapse(
            prompt="What is the capital of France?",
            output="The capital of France is Berlin.",
            domain="hallucination",
        )
        result = await miner2.forward(synapse)
        assert result.anomaly_score is not None
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_load_state_no_directory(self, mock_config):
        """load_state() handles missing directory gracefully."""
        from neurons.miner import Miner

        miner = Miner(config=mock_config)
        # Should not raise -- just logs warning
        miner.load_state()


# ---------------------------------------------------------------------------
# TestThirdPartyDetector
# ---------------------------------------------------------------------------

class CustomTestDetector(BaseDetector):
    """Custom detector for testing third-party pluggability."""

    domain = "test_custom"

    def __init__(self):
        self._is_fitted = False

    def fit(self, samples: list[dict]) -> None:
        self._is_fitted = True

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        return DetectionResult(
            score=0.42,
            confidence=0.9,
            anomaly_type="custom_test",
            feature_attribution={"test_feature": 1.0},
        )

    def get_info(self) -> dict:
        return {
            "name": "CustomTestDetector",
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "test",
            "is_fitted": self._is_fitted,
        }


class TestThirdPartyDetector:
    """Tests for third-party detector pluggability via load_detector."""

    def test_custom_detector_is_base_detector(self):
        """Custom detector inherits from BaseDetector."""
        det = CustomTestDetector()
        assert isinstance(det, BaseDetector)

    def test_custom_detector_loadable(self):
        """Custom detector can be loaded via load_detector when importable."""
        # Register this test module's CustomTestDetector as importable
        # by temporarily inserting it into a fake module path
        import types

        from neurons.miner import load_detector
        fake_module = types.ModuleType("_test_custom_detectors")
        fake_module.CustomTestDetector = CustomTestDetector
        sys.modules["_test_custom_detectors"] = fake_module

        try:
            det = load_detector("_test_custom_detectors.CustomTestDetector")
            assert isinstance(det, BaseDetector)
            assert det.domain == "test_custom"
        finally:
            del sys.modules["_test_custom_detectors"]

    @pytest.mark.asyncio
    async def test_custom_detector_end_to_end(self, mock_config):
        """Custom detector works end-to-end with miner forward pipeline."""
        from antigence_subnet.protocol import VerificationSynapse
        from neurons.miner import Miner

        # Create miner with standard detector, then inject custom
        miner = Miner(config=mock_config)
        custom_det = CustomTestDetector()
        custom_det.fit([])
        miner.detectors["test_custom"] = custom_det
        miner.supported_domains.add("test_custom")

        synapse = VerificationSynapse(
            prompt="Custom test",
            output="Custom output",
            domain="test_custom",
        )
        result = await miner.forward(synapse)
        assert result.anomaly_score == 0.42
        assert result.confidence == 0.9
        assert result.anomaly_type == "custom_test"
        assert result.feature_attribution == {"test_feature": 1.0}
