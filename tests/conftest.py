"""Shared test fixtures for the Antigence subnet test suite."""

import argparse
import os
import shutil
import tempfile

import bittensor as bt
import pytest
from bittensor_wallet import Wallet

from antigence_subnet.base.neuron import _DEFAULT_EVAL_DATA_DIR, _DEFAULT_TRAINING_DATA_DIR
from antigence_subnet.miner.data import load_training_samples
from antigence_subnet.mock import MockDendrite, MockMetagraph, MockSubtensor
from antigence_subnet.protocol import VerificationSynapse


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.ollama tests when OLLAMA_SKIP=1."""
    if os.environ.get("OLLAMA_SKIP", "0") == "1":
        skip_ollama = pytest.mark.skip(reason="OLLAMA_SKIP=1 -- Ollama tests disabled")
        for item in items:
            if "ollama" in item.keywords:
                item.add_marker(skip_ollama)


@pytest.fixture
def mock_wallet():
    """Create a wallet with real keys in a temp directory for testing."""
    tmpdir = tempfile.mkdtemp()
    wallet = Wallet(name="test_mock", hotkey="test_mock", path=tmpdir)
    wallet.create_if_non_existent(
        coldkey_use_password=False, hotkey_use_password=False
    )
    yield wallet
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def miner_wallet():
    """Create a wallet with real keys for the miner in integration tests."""
    tmpdir = tempfile.mkdtemp()
    wallet = Wallet(name="test_miner", hotkey="test_miner", path=tmpdir)
    wallet.create_if_non_existent(
        coldkey_use_password=False, hotkey_use_password=False
    )
    yield wallet
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def validator_wallet():
    """Create a wallet with real keys for the validator in integration tests."""
    tmpdir = tempfile.mkdtemp()
    wallet = Wallet(name="test_validator", hotkey="test_validator", path=tmpdir)
    wallet.create_if_non_existent(
        coldkey_use_password=False, hotkey_use_password=False
    )
    yield wallet
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_config(mock_wallet, tmp_path):
    """Create a mock config for neuron instantiation.

    Provides --mock --netuid 1 configuration with a temp directory
    for state persistence.
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
    parser.add_argument("--neuron.sample_size", type=int, default=16)
    parser.add_argument("--neuron.timeout", type=float, default=12.0)
    parser.add_argument("--neuron.moving_average_alpha", type=float, default=0.1)
    parser.add_argument("--neuron.eval_data_dir", type=str, default=_DEFAULT_EVAL_DATA_DIR)
    parser.add_argument("--neuron.eval_domain", type=str, default="hallucination")
    parser.add_argument("--neuron.samples_per_round", type=int, default=10)
    parser.add_argument("--neuron.n_honeypots", type=int, default=2)
    parser.add_argument("--neuron.set_weights_interval", type=int, default=100)
    parser.add_argument("--neuron.set_weights_retries", type=int, default=3)
    parser.add_argument("--neuron.shutdown_timeout", type=int, default=30)
    parser.add_argument(
        "--detector",
        type=str,
        default="antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector",
    )
    parser.add_argument("--neuron.training_data_dir", type=str, default=_DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--neuron.training_domain", type=str, default="hallucination")
    # Microglia surveillance CLI args
    parser.add_argument("--microglia.interval", type=int, default=100)
    parser.add_argument("--microglia.webhook_url", type=str, default=None)
    parser.add_argument("--microglia.inactive_threshold", type=int, default=10)
    parser.add_argument("--microglia.stale_threshold", type=int, default=5)
    parser.add_argument("--microglia.deregistration_threshold", type=int, default=50)
    parser.add_argument("--microglia.alert_cooldown", type=int, default=10)
    parser.add_argument("--microglia.enabled", action="store_true", default=True)
    # Reward weight configuration (Phase 26 - MAIN-06)
    parser.add_argument("--reward.base_weight", type=float, default=0.70)
    parser.add_argument("--reward.calibration_weight", type=float, default=0.10)
    parser.add_argument("--reward.robustness_weight", type=float, default=0.10)
    parser.add_argument("--reward.diversity_weight", type=float, default=0.10)
    parser.add_argument("--reward.decision_threshold", type=float, default=0.5)
    parser.add_argument(
        "--logging.level", type=str, default="info",
        choices=["debug", "info", "warning", "error"],
    )
    parser.add_argument("--config-file", type=str, default=None)
    # Cold-start configuration (Phase 50 - VHARD-03)
    parser.add_argument("--cold-start.max-startup-seconds", type=float, default=60.0,
                        dest="cold_start.max_startup_seconds")
    parser.add_argument("--cold-start.min-miners-required", type=int, default=1,
                        dest="cold_start.min_miners_required")
    parser.add_argument("--cold-start.warmup-rounds", type=int, default=5,
                        dest="cold_start.warmup_rounds")

    config = bt.Config(
        parser,
        args=[
            "--mock",
            "--netuid", "1",
            "--wallet.name", mock_wallet.name,
            "--wallet.hotkey", mock_wallet.hotkey_str,
            "--wallet.path", mock_wallet.path,
            "--neuron.full_path", str(tmp_path),
            "--no_prompt",
        ],
    )
    return config


@pytest.fixture
def sample_synapse():
    """Create a sample VerificationSynapse with only required fields."""
    return VerificationSynapse(
        prompt="What is the capital of France?",
        output="The capital of France is Berlin.",
        domain="hallucination",
    )


@pytest.fixture
def sample_synapse_with_response(sample_synapse):
    """Create a VerificationSynapse with response fields populated."""
    sample_synapse.anomaly_score = 0.95
    sample_synapse.confidence = 0.88
    sample_synapse.anomaly_type = "factual_error"
    sample_synapse.feature_attribution = {"city_name": 0.95, "country_match": 0.85}
    return sample_synapse


@pytest.fixture
def mock_subtensor(mock_wallet):
    """Create a MockSubtensor with subnet and registered neurons."""
    return MockSubtensor(netuid=1, n=16, wallet=mock_wallet)


@pytest.fixture
def mock_metagraph(mock_subtensor):
    """Create a MockMetagraph synced from MockSubtensor."""
    return MockMetagraph(netuid=1, subtensor=mock_subtensor)


@pytest.fixture
def mock_dendrite(mock_wallet):
    """Create a MockDendrite for testing verification responses."""
    return MockDendrite(wallet=mock_wallet)


@pytest.fixture
def training_samples():
    """Load normal-only samples from seed evaluation data."""
    return load_training_samples("data/evaluation", "hallucination")


@pytest.fixture
def all_seed_samples():
    """Load ALL samples (normal + anomalous) from seed evaluation data."""
    import json

    with open("data/evaluation/hallucination/samples.json") as f:
        return json.load(f)["samples"]
