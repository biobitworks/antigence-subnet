"""
Tests for blacklist and priority functions on the reference miner (NET-02).

Verifies that inbound VerificationSynapse requests are filtered by
registration and stake, and that priority is based on caller stake.
"""

import numpy as np
import pytest
from bittensor.core.synapse import TerminalInfo

from antigence_subnet.protocol import VerificationSynapse
from neurons.miner import Miner


def make_synapse_with_caller(hotkey: str) -> VerificationSynapse:
    """Create a VerificationSynapse with caller identity set via TerminalInfo."""
    synapse = VerificationSynapse(
        prompt="test prompt",
        output="test output",
        domain="hallucination",
    )
    synapse.dendrite = TerminalInfo(hotkey=hotkey)
    return synapse


# ---------------------------------------------------------------------------
# MockMetagraph property tests
# ---------------------------------------------------------------------------


def test_mock_metagraph_has_stakes(mock_config):
    """MockMetagraph.S returns numpy float32 array of shape (n,)."""
    miner = Miner(config=mock_config)
    stakes = miner.metagraph.S
    assert isinstance(stakes, np.ndarray)
    assert stakes.dtype == np.float32
    assert stakes.shape == (miner.metagraph.n,)


def test_mock_metagraph_has_validator_permit(mock_config):
    """MockMetagraph.validator_permit returns numpy boolean array of shape (n,)."""
    miner = Miner(config=mock_config)
    permits = miner.metagraph.validator_permit
    assert isinstance(permits, np.ndarray)
    assert permits.dtype == bool
    assert permits.shape == (miner.metagraph.n,)


# ---------------------------------------------------------------------------
# Blacklist tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blacklist_unregistered(mock_config):
    """Blacklist rejects requests from hotkeys not in metagraph."""
    miner = Miner(config=mock_config)
    synapse = make_synapse_with_caller("totally-unknown-hotkey-xyz")

    should_blacklist, reason = await miner.blacklist(synapse)
    assert should_blacklist is True
    assert reason == "Unrecognized hotkey"


@pytest.mark.asyncio
async def test_blacklist_zero_stake(mock_config):
    """Blacklist rejects requests from registered callers with zero stake."""
    miner = Miner(config=mock_config)

    # Pick a registered hotkey and set its stake to 0
    registered_hotkey = miner.metagraph.hotkeys[1]
    caller_uid = miner.metagraph.hotkeys.index(registered_hotkey)
    miner.metagraph._stakes[caller_uid] = 0.0

    synapse = make_synapse_with_caller(registered_hotkey)

    should_blacklist, reason = await miner.blacklist(synapse)
    assert should_blacklist is True
    assert reason == "Insufficient stake"


@pytest.mark.asyncio
async def test_blacklist_allows_valid(mock_config):
    """Blacklist allows requests from registered callers with stake > 0."""
    miner = Miner(config=mock_config)

    # Pick a registered hotkey (default stake is >0)
    registered_hotkey = miner.metagraph.hotkeys[1]

    synapse = make_synapse_with_caller(registered_hotkey)

    should_blacklist, reason = await miner.blacklist(synapse)
    assert should_blacklist is False
    assert reason == "Hotkey recognized"


@pytest.mark.asyncio
async def test_blacklist_missing_dendrite(mock_config):
    """Blacklist rejects when synapse.dendrite is None."""
    miner = Miner(config=mock_config)
    synapse = VerificationSynapse(
        prompt="test", output="test", domain="hallucination"
    )
    synapse.dendrite = None

    should_blacklist, reason = await miner.blacklist(synapse)
    assert should_blacklist is True
    assert reason == "Missing dendrite or hotkey"


@pytest.mark.asyncio
async def test_blacklist_missing_hotkey(mock_config):
    """Blacklist rejects when synapse.dendrite.hotkey is None."""
    miner = Miner(config=mock_config)
    synapse = VerificationSynapse(
        prompt="test", output="test", domain="hallucination"
    )
    synapse.dendrite = TerminalInfo(hotkey=None)

    should_blacklist, reason = await miner.blacklist(synapse)
    assert should_blacklist is True
    assert reason == "Missing dendrite or hotkey"


# ---------------------------------------------------------------------------
# Priority tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_priority_returns_stake(mock_config):
    """Priority returns caller's stake as float for registered callers."""
    miner = Miner(config=mock_config)

    registered_hotkey = miner.metagraph.hotkeys[1]
    caller_uid = miner.metagraph.hotkeys.index(registered_hotkey)
    expected_stake = float(miner.metagraph.S[caller_uid])

    synapse = make_synapse_with_caller(registered_hotkey)

    priority = await miner.priority(synapse)
    assert priority == expected_stake
    assert isinstance(priority, float)


@pytest.mark.asyncio
async def test_priority_missing_dendrite(mock_config):
    """Priority returns 0.0 when synapse.dendrite is None."""
    miner = Miner(config=mock_config)
    synapse = VerificationSynapse(
        prompt="test", output="test", domain="hallucination"
    )
    synapse.dendrite = None

    priority = await miner.priority(synapse)
    assert priority == 0.0


@pytest.mark.asyncio
async def test_priority_unregistered(mock_config):
    """Priority returns 0.0 for an unregistered caller."""
    miner = Miner(config=mock_config)
    synapse = make_synapse_with_caller("totally-unknown-hotkey-xyz")

    priority = await miner.priority(synapse)
    assert priority == 0.0
