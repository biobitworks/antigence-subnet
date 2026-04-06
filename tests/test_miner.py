"""
Tests for miner neuron (PROTO-02, PROTO-05).

Verifies miner instantiation, axon setup, forward signature,
forward response population, neuron type, and CLI args.
"""

import inspect

import pytest

from antigence_subnet.protocol import VerificationSynapse
from neurons.miner import Miner


def test_miner_instantiation_mock(mock_config):
    """Test Miner creates successfully with mock config."""
    miner = Miner(config=mock_config)
    assert miner is not None
    assert miner.config.mock is True


def test_miner_has_axon(mock_config):
    """Test miner has an axon."""
    miner = Miner(config=mock_config)
    assert miner.axon is not None


@pytest.mark.asyncio
async def test_miner_forward_signature(mock_config):
    """Test Miner.forward accepts (self, synapse: VerificationSynapse)."""
    miner = Miner(config=mock_config)
    sig = inspect.signature(miner.forward)
    params = list(sig.parameters.keys())
    assert "synapse" in params


@pytest.mark.asyncio
async def test_miner_forward_populates_response(mock_config):
    """Test miner forward populates response fields."""
    miner = Miner(config=mock_config)
    synapse = VerificationSynapse(
        prompt="Test prompt",
        output="Test output",
        domain="hallucination",
    )
    result = await miner.forward(synapse)

    assert result.anomaly_score is not None
    assert result.confidence is not None
    assert result.anomaly_type is not None
    # Real detector produces scores in [0, 1] range (MockDetector removed in 03-02)
    assert 0.0 <= result.anomaly_score <= 1.0


def test_miner_neuron_type(mock_config):
    """Test miner neuron type is 'MinerNeuron'."""
    miner = Miner(config=mock_config)
    assert miner.neuron_type == "MinerNeuron"


def test_miner_cli_args():
    """Test Miner.add_args() produces config with netuid and mock attributes."""
    config = Miner.add_args()
    # Config should have netuid and mock attributes from parsed args
    assert hasattr(config, "netuid")
    assert hasattr(config, "mock")
