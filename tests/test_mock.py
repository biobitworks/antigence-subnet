"""Tests for mock infrastructure (PROTO-03).

Verifies MockSubtensor, MockMetagraph, and MockDendrite work correctly
with SDK v10 parameter naming and VerificationSynapse response fields.

Note: MockSubtensor's substrate is a MagicMock, so methods like
subnet_exists() and is_hotkey_registered() that rely on substrate.query()
always return truthy values. Tests verify registration by inspecting
the underlying chain_state dict directly.
"""

import pytest
from bittensor.utils.balance import Balance

from antigence_subnet.mock import MockMetagraph, MockSubtensor
from antigence_subnet.protocol import VerificationSynapse


def _hotkey_registered(subtensor, netuid: int, hotkey_ss58: str) -> bool:
    """Check if a hotkey is registered by inspecting chain state directly."""
    uids = subtensor.chain_state["SubtensorModule"]["Uids"]
    return hotkey_ss58 in uids.get(netuid, {})


class TestMockSubtensor:
    """Test MockSubtensor subnet and neuron registration."""

    def test_mock_subtensor_creates_subnet(self):
        """MockSubtensor creates the specified subnet."""
        subtensor = MockSubtensor(netuid=1, n=0)
        # Check chain state directly -- subnet_exists() returns MagicMock
        assert 1 in subtensor.chain_state["SubtensorModule"]["NetworksAdded"]

    def test_mock_subtensor_registers_wallet(self, mock_wallet):
        """MockSubtensor registers the wallet's hotkey on the subnet."""
        subtensor = MockSubtensor(netuid=1, n=0, wallet=mock_wallet)
        assert _hotkey_registered(
            subtensor, 1, mock_wallet.hotkey.ss58_address
        )

    def test_mock_subtensor_registers_n_miners(self):
        """MockSubtensor registers n additional mock miners."""
        n = 8
        subtensor = MockSubtensor(netuid=1, n=n)
        for i in range(1, n + 1):
            assert _hotkey_registered(subtensor, 1, f"miner-hotkey-{i}")

    def test_mock_subtensor_uses_balance_objects(self, mock_wallet):
        """MockSubtensor uses Balance objects for stake and balance (no TypeError)."""
        # If Balance objects were not used, force_register_neuron would have
        # raised a TypeError or produced incorrect results during __init__
        subtensor = MockSubtensor(netuid=1, n=4, wallet=mock_wallet)
        assert 1 in subtensor.chain_state["SubtensorModule"]["NetworksAdded"]
        assert _hotkey_registered(
            subtensor, 1, mock_wallet.hotkey.ss58_address
        )

    def test_mock_v10_parameter_naming(self):
        """MockSubtensor uses hotkey_ss58 not hotkey (v10 naming convention)."""
        subtensor = MockSubtensor(netuid=1, n=0)
        # The v10 force_register_neuron signature requires hotkey_ss58
        uid = subtensor.force_register_neuron(
            netuid=1,
            hotkey_ss58="test-v10-hotkey",
            coldkey_ss58="test-v10-coldkey",
            balance=Balance(1000),
            stake=Balance(1000),
        )
        assert uid >= 0
        assert _hotkey_registered(subtensor, 1, "test-v10-hotkey")


class TestMockMetagraph:
    """Test MockMetagraph syncing from MockSubtensor."""

    def test_mock_metagraph_sync(self, mock_subtensor):
        """MockMetagraph synced from MockSubtensor has neurons populated."""
        metagraph = MockMetagraph(netuid=1, subtensor=mock_subtensor)
        assert metagraph.n > 0

    def test_mock_metagraph_has_hotkeys(self, mock_subtensor):
        """MockMetagraph exposes a non-empty list of hotkeys."""
        metagraph = MockMetagraph(netuid=1, subtensor=mock_subtensor)
        assert isinstance(metagraph.hotkeys, list)
        assert len(metagraph.hotkeys) > 0
        for hk in metagraph.hotkeys:
            assert isinstance(hk, str)
            assert len(hk) > 0

    def test_mock_metagraph_has_axons(self, mock_subtensor):
        """MockMetagraph exposes AxonInfo objects for each neuron."""
        metagraph = MockMetagraph(netuid=1, subtensor=mock_subtensor)
        assert len(metagraph.axons) == metagraph.n
        for axon in metagraph.axons:
            assert hasattr(axon, "hotkey")
            assert hasattr(axon, "ip")
            assert hasattr(axon, "port")


class TestMockDendrite:
    """Test MockDendrite returns VerificationSynapse responses."""

    @pytest.mark.asyncio
    async def test_mock_dendrite_returns_responses(
        self, mock_dendrite, sample_synapse, mock_metagraph
    ):
        """MockDendrite returns a list of responses matching the number of axons."""
        axons = mock_metagraph.axons[:3]
        responses = await mock_dendrite.forward(
            axons=axons, synapse=sample_synapse, deserialize=False
        )
        assert isinstance(responses, list)
        assert len(responses) == len(axons)

    @pytest.mark.asyncio
    async def test_mock_dendrite_populates_verification_fields(
        self, mock_dendrite, sample_synapse, mock_metagraph
    ):
        """Deserialized=False responses have anomaly fields populated (not None)."""
        axons = mock_metagraph.axons[:3]
        responses = await mock_dendrite.forward(
            axons=axons, synapse=sample_synapse, deserialize=False
        )
        for resp in responses:
            assert isinstance(resp, VerificationSynapse)
            assert resp.anomaly_score is not None
            assert 0.0 <= resp.anomaly_score <= 1.0
            assert resp.confidence is not None
            assert 0.5 <= resp.confidence <= 1.0
            assert resp.anomaly_type is not None
            assert resp.feature_attribution is not None
            assert isinstance(resp.feature_attribution, dict)

    @pytest.mark.asyncio
    async def test_mock_dendrite_deserialize_returns_floats(
        self, mock_dendrite, sample_synapse, mock_metagraph
    ):
        """Deserialized=True responses are floats (anomaly_score values)."""
        axons = mock_metagraph.axons[:3]
        responses = await mock_dendrite.forward(
            axons=axons, synapse=sample_synapse, deserialize=True
        )
        for resp in responses:
            assert isinstance(resp, float)
            assert 0.0 <= resp <= 1.0

    @pytest.mark.asyncio
    async def test_mock_dendrite_callable(
        self, mock_dendrite, sample_synapse, mock_metagraph
    ):
        """MockDendrite supports being called directly (matches SDK interface)."""
        axons = mock_metagraph.axons[:2]
        responses = await mock_dendrite(
            axons=axons, synapse=sample_synapse, deserialize=False
        )
        assert len(responses) == 2
        assert all(isinstance(r, VerificationSynapse) for r in responses)
