"""
Tests for multi-domain miner detector loading from TOML config (MINE-01).

Verifies:
- Multiple detectors loaded from [miner.detectors] TOML table
- Synapse routing to correct domain detector
- Backward compatibility with --detector CLI arg
- CLI-loaded detector takes precedence over TOML for same domain
- Unfitted domain detector warning when training data missing
- Per-domain subdirectory state persistence (save_state / load_state)
- Empty TOML detectors table falls back to CLI behavior
"""

import argparse
import os
import sys
import types
from unittest.mock import patch

import bittensor as bt
import pytest

from antigence_subnet.base.neuron import _DEFAULT_TRAINING_DATA_DIR
from antigence_subnet.miner.detector import BaseDetector, DetectionResult

# ---------------------------------------------------------------------------
# Fake detectors for testing
# ---------------------------------------------------------------------------

class FakeDetectorA(BaseDetector):
    """Fake detector for domain 'domain_a'."""

    domain = "domain_a"

    def __init__(self):
        self._is_fitted = False
        self._fit_count = 0
        self._detect_calls = []
        self._saved_path = None
        self._loaded_path = None

    def fit(self, samples: list[dict]) -> None:
        self._is_fitted = True
        self._fit_count += 1

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        self._detect_calls.append({"prompt": prompt, "output": output})
        return DetectionResult(
            score=0.11,
            confidence=0.91,
            anomaly_type="fake_a",
        )

    def save_state(self, path: str) -> None:
        self._saved_path = path
        # Write a marker file so we can verify the path
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "fake_a_state.txt"), "w") as f:
            f.write("saved")

    def load_state(self, path: str) -> None:
        self._loaded_path = path
        marker = os.path.join(path, "fake_a_state.txt")
        if os.path.exists(marker):
            self._is_fitted = True

    def get_info(self) -> dict:
        return {
            "name": "FakeDetectorA",
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "fake",
            "is_fitted": self._is_fitted,
        }


class FakeDetectorB(BaseDetector):
    """Fake detector for domain 'domain_b'."""

    domain = "domain_b"

    def __init__(self):
        self._is_fitted = False
        self._detect_calls = []
        self._saved_path = None
        self._loaded_path = None

    def fit(self, samples: list[dict]) -> None:
        self._is_fitted = True

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        self._detect_calls.append({"prompt": prompt, "output": output})
        return DetectionResult(
            score=0.22,
            confidence=0.82,
            anomaly_type="fake_b",
        )

    def save_state(self, path: str) -> None:
        self._saved_path = path
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "fake_b_state.txt"), "w") as f:
            f.write("saved")

    def load_state(self, path: str) -> None:
        self._loaded_path = path
        marker = os.path.join(path, "fake_b_state.txt")
        if os.path.exists(marker):
            self._is_fitted = True

    def get_info(self) -> dict:
        return {
            "name": "FakeDetectorB",
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "fake",
            "is_fitted": self._is_fitted,
        }


class FakeHallucinationDetector(BaseDetector):
    """Fake detector for domain 'hallucination' (same domain as CLI default)."""

    domain = "hallucination"

    def __init__(self):
        self._is_fitted = False
        self._source = "toml"  # To track which path created it

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
            score=0.99,
            confidence=0.99,
            anomaly_type="fake_hallucination",
        )

    def get_info(self) -> dict:
        return {
            "name": "FakeHallucinationDetector",
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "fake",
            "is_fitted": self._is_fitted,
        }


# Register fake detector modules so load_detector() can find them
def _register_fake_modules():
    """Register fake detector classes as importable modules for load_detector()."""
    mod_a = types.ModuleType("_fake_detectors_a")
    mod_a.FakeDetectorA = FakeDetectorA
    sys.modules["_fake_detectors_a"] = mod_a

    mod_b = types.ModuleType("_fake_detectors_b")
    mod_b.FakeDetectorB = FakeDetectorB
    sys.modules["_fake_detectors_b"] = mod_b

    mod_h = types.ModuleType("_fake_detectors_hallucination")
    mod_h.FakeHallucinationDetector = FakeHallucinationDetector
    sys.modules["_fake_detectors_hallucination"] = mod_h


def _unregister_fake_modules():
    """Clean up fake detector modules."""
    for name in ["_fake_detectors_a", "_fake_detectors_b", "_fake_detectors_hallucination"]:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def register_fakes():
    """Register and clean up fake detector modules for all tests."""
    _register_fake_modules()
    yield
    _unregister_fake_modules()


def _make_miner_config(tmp_path, mock_wallet, toml_raw=None, detector_class=None):
    """Create a mock config for Miner instantiation with optional TOML raw data.

    Args:
        tmp_path: Temporary path for state directory.
        mock_wallet: Wallet fixture.
        toml_raw: Dict to set as _toml_raw on the neuron instance.
        detector_class: Override the --detector CLI arg class path.
    """
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)
    parser.add_argument("--netuid", type=int, default=1)
    parser.add_argument("--mock", action="store_true", default=False)
    parser.add_argument("--neuron.full_path", type=str, default=str(tmp_path))
    parser.add_argument("--neuron.device", type=str, default="cpu")
    parser.add_argument("--neuron.shutdown_timeout", type=int, default=30)
    parser.add_argument(
        "--detector",
        type=str,
        default=detector_class or "_fake_detectors_a.FakeDetectorA",
    )
    parser.add_argument("--neuron.training_data_dir", type=str, default=_DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--neuron.training_domain", type=str, default="domain_a")
    parser.add_argument(
        "--logging.level", type=str, default="info",
        choices=["debug", "info", "warning", "error"],
    )
    parser.add_argument("--config-file", type=str, default=None)

    args = [
        "--mock",
        "--netuid", "1",
        "--wallet.name", mock_wallet.name,
        "--wallet.hotkey", mock_wallet.hotkey_str,
        "--wallet.path", mock_wallet.path,
        "--neuron.full_path", str(tmp_path),
        "--no_prompt",
    ]

    if detector_class:
        args.extend(["--detector", detector_class])

    config = bt.Config(parser, args=args)
    return config, toml_raw


# ---------------------------------------------------------------------------
# TestMultiDomainLoad
# ---------------------------------------------------------------------------

class TestMultiDomainLoad:
    """Tests for loading multiple detectors from TOML [miner.detectors] table."""

    def test_multi_domain_load(self, mock_wallet, tmp_path):
        """When _toml_raw has [miner][detectors] with 2 entries, Miner loads all."""
        from neurons.miner import Miner

        config, _ = _make_miner_config(tmp_path, mock_wallet)

        toml_path = tmp_path / "test.toml"
        toml_content = b"""
[miner.detectors]
domain_a = "_fake_detectors_a.FakeDetectorA"
domain_b = "_fake_detectors_b.FakeDetectorB"
"""
        toml_path.write_bytes(toml_content)

        with patch("antigence_subnet.base.neuron.find_config_file", return_value=toml_path):
            miner = Miner(config=config)

        assert "domain_a" in miner.detectors
        assert "domain_b" in miner.detectors
        assert len(miner.detectors) == 2
        assert miner.supported_domains == {"domain_a", "domain_b"}

    def test_empty_detectors_table(self, mock_wallet, tmp_path):
        """Empty [miner.detectors] table does not crash; falls back to CLI --detector."""
        from neurons.miner import Miner

        toml_path = tmp_path / "test.toml"
        toml_content = b"""
[miner.detectors]
"""
        toml_path.write_bytes(toml_content)

        config, _ = _make_miner_config(tmp_path, mock_wallet)

        with patch("antigence_subnet.base.neuron.find_config_file", return_value=toml_path):
            miner = Miner(config=config)

        # CLI --detector loaded domain_a
        assert "domain_a" in miner.detectors
        assert len(miner.detectors) == 1


# ---------------------------------------------------------------------------
# TestDomainRouting
# ---------------------------------------------------------------------------

class TestDomainRouting:
    """Tests for synapse routing to correct domain detector."""

    @pytest.mark.asyncio
    async def test_routes_to_correct_domain(self, mock_wallet, tmp_path):
        """Miner with 2 detectors routes synapse to matching domain detector."""
        from antigence_subnet.protocol import VerificationSynapse
        from neurons.miner import Miner

        toml_path = tmp_path / "test.toml"
        toml_content = b"""
[miner.detectors]
domain_a = "_fake_detectors_a.FakeDetectorA"
domain_b = "_fake_detectors_b.FakeDetectorB"
"""
        toml_path.write_bytes(toml_content)

        config, _ = _make_miner_config(tmp_path, mock_wallet)

        with patch("antigence_subnet.base.neuron.find_config_file", return_value=toml_path):
            miner = Miner(config=config)

        # Route to domain_a
        synapse_a = VerificationSynapse(
            prompt="Test prompt A",
            output="Test output A",
            domain="domain_a",
        )
        result_a = await miner.forward(synapse_a)
        assert result_a.anomaly_score == 0.11
        assert result_a.anomaly_type == "fake_a"

        # Route to domain_b
        synapse_b = VerificationSynapse(
            prompt="Test prompt B",
            output="Test output B",
            domain="domain_b",
        )
        result_b = await miner.forward(synapse_b)
        assert result_b.anomaly_score == 0.22
        assert result_b.anomaly_type == "fake_b"


# ---------------------------------------------------------------------------
# TestBackwardCompat
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Tests for backward compatibility with --detector CLI arg."""

    def test_single_detector_backward_compat(self, mock_wallet, tmp_path):
        """When no [miner.detectors] TOML table exists, --detector CLI loads a single detector."""
        from neurons.miner import Miner

        config, _ = _make_miner_config(tmp_path, mock_wallet)

        with patch("antigence_subnet.base.neuron.find_config_file", return_value=None):
            miner = Miner(config=config)

        assert "domain_a" in miner.detectors
        assert len(miner.detectors) == 1
        assert miner.supported_domains == {"domain_a"}

    def test_cli_detector_not_overridden(self, mock_wallet, tmp_path):
        """CLI-loaded detector takes precedence over TOML for the same domain."""
        from neurons.miner import Miner

        # CLI loads domain_a via _fake_detectors_a.FakeDetectorA
        # TOML also specifies domain_a (with a different class that also has domain "domain_a")
        # But since the fake hallucination detector has domain "hallucination", let's
        # use a scenario where CLI loads hallucination and TOML also wants hallucination
        toml_path = tmp_path / "test.toml"
        toml_content = b"""
[miner.detectors]
hallucination = "_fake_detectors_hallucination.FakeHallucinationDetector"
domain_b = "_fake_detectors_b.FakeDetectorB"
"""
        toml_path.write_bytes(toml_content)

        # CLI loads the real IsolationForestDetector for hallucination domain
        config, _ = _make_miner_config(
            tmp_path,
            mock_wallet,
            detector_class="antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector",
        )
        # Override training domain to match IsolationForest default
        config.neuron.training_domain = "hallucination"

        with patch("antigence_subnet.base.neuron.find_config_file", return_value=toml_path):
            miner = Miner(config=config)

        # CLI detector should be IsolationForestDetector, not FakeHallucinationDetector
        assert "hallucination" in miner.detectors
        assert miner.detectors["hallucination"].__class__.__name__ == "IsolationForestDetector"
        # TOML domain_b should also be loaded
        assert "domain_b" in miner.detectors
        assert len(miner.detectors) == 2


# ---------------------------------------------------------------------------
# TestUnfittedDomain
# ---------------------------------------------------------------------------

class TestUnfittedDomain:
    """Tests for unfitted domain detector handling."""

    def test_unfitted_domain_warning(self, mock_wallet, tmp_path):
        """When TOML detector's training domain has no data, detector registers unfitted."""
        from neurons.miner import Miner

        toml_path = tmp_path / "test.toml"
        # domain_b has no training data in data/evaluation/domain_b/
        toml_content = b"""
[miner.detectors]
domain_b = "_fake_detectors_b.FakeDetectorB"
"""
        toml_path.write_bytes(toml_content)

        config, _ = _make_miner_config(tmp_path, mock_wallet)

        with patch("antigence_subnet.base.neuron.find_config_file", return_value=toml_path):
            miner = Miner(config=config)

        # domain_b should be registered but unfitted (no training data at data/evaluation/domain_b/)
        assert "domain_b" in miner.detectors
        assert miner.detectors["domain_b"].get_info()["is_fitted"] is False


# ---------------------------------------------------------------------------
# TestMultiDomainStatePersistence
# ---------------------------------------------------------------------------

class TestMultiDomainStatePersistence:
    """Tests for per-domain subdirectory state persistence."""

    def test_multi_domain_save_state(self, mock_wallet, tmp_path):
        """save_state() with multiple detectors uses per-domain subdirectories."""
        from neurons.miner import Miner

        toml_path = tmp_path / "test.toml"
        toml_content = b"""
[miner.detectors]
domain_a = "_fake_detectors_a.FakeDetectorA"
domain_b = "_fake_detectors_b.FakeDetectorB"
"""
        toml_path.write_bytes(toml_content)

        config, _ = _make_miner_config(tmp_path, mock_wallet)

        with patch("antigence_subnet.base.neuron.find_config_file", return_value=toml_path):
            miner = Miner(config=config)

        miner.save_state()

        # Check per-domain subdirectories exist
        base_state_dir = os.path.join(str(tmp_path), "detector_state")
        assert os.path.isdir(os.path.join(base_state_dir, "domain_a"))
        assert os.path.isdir(os.path.join(base_state_dir, "domain_b"))
        # Check marker files from fake detectors
        assert os.path.exists(os.path.join(base_state_dir, "domain_a", "fake_a_state.txt"))
        assert os.path.exists(os.path.join(base_state_dir, "domain_b", "fake_b_state.txt"))

    def test_multi_domain_load_state(self, mock_wallet, tmp_path):
        """load_state() with multiple detectors restores from per-domain subdirectories."""
        from neurons.miner import Miner

        toml_path = tmp_path / "test.toml"
        toml_content = b"""
[miner.detectors]
domain_a = "_fake_detectors_a.FakeDetectorA"
domain_b = "_fake_detectors_b.FakeDetectorB"
"""
        toml_path.write_bytes(toml_content)

        config, _ = _make_miner_config(tmp_path, mock_wallet)

        # First miner: save state
        with patch("antigence_subnet.base.neuron.find_config_file", return_value=toml_path):
            miner1 = Miner(config=config)
        miner1.save_state()

        # Second miner: load state
        with patch("antigence_subnet.base.neuron.find_config_file", return_value=toml_path):
            miner2 = Miner(config=config)
        miner2.load_state()

        # Verify detectors loaded from per-domain subdirectories
        base_state_dir = os.path.join(str(tmp_path), "detector_state")
        assert miner2.detectors["domain_a"]._loaded_path == os.path.join(base_state_dir, "domain_a")
        assert miner2.detectors["domain_b"]._loaded_path == os.path.join(base_state_dir, "domain_b")
