"""
Mock infrastructure for offline testing.

Provides MockSubtensor, MockMetagraph, and MockDendrite with
SDK v10 compatibility (hotkey_ss58, Balance objects).

Note: The SDK v10.2.0 MockSubtensor.neuron_for_uid_lite() has a bug where
it references non-existent fields (rank, trust, pruning_score) on NeuronInfo.
MockMetagraph works around this by building metagraph data directly from
the subtensor's chain state rather than using the broken sync() path.
"""

import random
from typing import Any

import bittensor as bt
import numpy as np
from bittensor.core.chain_data import AxonInfo
from bittensor.utils.balance import Balance

from antigence_subnet.protocol import VerificationSynapse


def _reset_mock_state() -> None:
    """Reset the SDK's global mock state so each MockSubtensor starts clean.

    The SDK's MockSubtensor uses a module-level __GLOBAL_MOCK_STATE__ dict
    that is shared across ALL instances via ``self.__dict__ = __GLOBAL_MOCK_STATE__``.
    Without clearing it, creating a second MockSubtensor in the same process
    inherits all previously registered subnets and neurons.
    """
    from bittensor.utils.mock.subtensor_mock import __GLOBAL_MOCK_STATE__

    __GLOBAL_MOCK_STATE__.clear()


class MockSubtensor(bt.MockSubtensor):
    """Mock subtensor with v10-compatible neuron registration.

    Creates a subnet and registers neurons using the v10 parameter naming
    convention (hotkey_ss58, coldkey_ss58) and Balance objects.

    Each instance resets the SDK's global mock state to ensure test isolation.
    """

    def __init__(
        self,
        netuid: int = 1,
        n: int = 16,
        wallet: Any | None = None,
        network: str = "mock",
    ):
        # Reset global state so each MockSubtensor instance starts clean
        _reset_mock_state()
        super().__init__(network=network)

        # Note: self.subnet_exists() is unreliable on MockSubtensor because
        # it calls substrate.query() on a MagicMock, which always returns truthy.
        # Check the chain state dict directly instead.
        if netuid not in self.chain_state["SubtensorModule"]["NetworksAdded"]:
            self.create_subnet(netuid)

        # Register the owner wallet if provided
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkey.ss58_address,
                balance=Balance(100000),
                stake=Balance(100000),
            )

        # Register n additional mock miners
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=f"miner-hotkey-{i}",
                coldkey_ss58="mock-coldkey",
                balance=Balance(100000),
                stake=Balance(100000),
            )


class MockMetagraph:
    """Mock metagraph that builds neuron data from a MockSubtensor.

    The SDK v10.2.0 metagraph.sync() path is broken for MockSubtensor
    (neuron_for_uid_lite references non-existent NeuronInfo fields).
    This class works around the bug by constructing metagraph data directly
    from the subtensor's registered neuron information.
    """

    def __init__(
        self,
        netuid: int = 1,
        subtensor: MockSubtensor | None = None,
        network: str = "mock",
    ):
        self.netuid = netuid
        self.network = network
        self._axons: list[AxonInfo] = []
        self._n: int = 0
        self._stakes: np.ndarray = np.array([], dtype=np.float32)
        self._validator_permits: np.ndarray = np.array([], dtype=bool)

        if subtensor is not None:
            self._sync_from_subtensor(subtensor, netuid)

    def _sync_from_subtensor(self, subtensor: MockSubtensor, netuid: int) -> None:
        """Build metagraph data from MockSubtensor chain state."""
        # Count registered neurons
        subnet_n_key = "SubtensorModule"
        chain = subtensor.chain_state
        n_neurons = chain.get(subnet_n_key, {}).get("SubnetworkN", {}).get(netuid, {})

        # Get the actual count from the chain state
        if isinstance(n_neurons, dict):
            # Get the latest block's value
            n_val = max(n_neurons.values()) if n_neurons else 0
        else:
            n_val = int(n_neurons) if n_neurons else 0

        self._n = n_val

        # Initialize stake and validator permit arrays
        self._stakes = np.ones(self._n, dtype=np.float32) * 100000.0
        self._validator_permits = np.ones(self._n, dtype=bool)

        # Build axon info for each registered neuron
        self._axons = []
        for uid in range(n_val):
            neuron = subtensor.neuron_for_uid(uid=uid, netuid=netuid)
            if neuron is not None and not neuron.is_null:
                self._axons.append(
                    AxonInfo(
                        ip="127.0.0.1",
                        port=8091 + uid,
                        hotkey=neuron.hotkey,
                        coldkey=neuron.coldkey,
                        version=0,
                        ip_type=4,
                    )
                )
            else:
                # Null placeholder
                self._axons.append(
                    AxonInfo(
                        ip="0.0.0.0",
                        port=0,
                        hotkey="",
                        coldkey="",
                        version=0,
                        ip_type=4,
                    )
                )

    @property
    def n(self) -> int:
        """Number of neurons in the metagraph."""
        return self._n

    @property
    def hotkeys(self) -> list[str]:
        """List of hotkey ss58 addresses for registered neurons."""
        return [axon.hotkey for axon in self._axons]

    @property
    def coldkeys(self) -> list[str]:
        """List of coldkey ss58 addresses for registered neurons."""
        return [axon.coldkey for axon in self._axons]

    @property
    def axons(self) -> list[AxonInfo]:
        """List of AxonInfo for registered neurons."""
        return self._axons

    @property
    def uids(self) -> np.ndarray:
        """Array of UIDs."""
        return np.arange(self._n, dtype=np.int64)

    @property
    def S(self) -> np.ndarray:  # noqa: N802
        """Stake array for registered neurons."""
        return self._stakes

    @property
    def validator_permit(self) -> np.ndarray:
        """Boolean array of validator permits."""
        return self._validator_permits

    def sync(self, subtensor: MockSubtensor | None = None) -> None:
        """Re-sync from subtensor."""
        if subtensor is not None:
            self._sync_from_subtensor(subtensor, self.netuid)


class MockDendrite:
    """Mock dendrite that returns VerificationSynapse responses with populated fields.

    Unlike the SDK template's MockDendrite (which populates dummy_output),
    this implementation populates VerificationSynapse-specific response fields
    (anomaly_score, confidence, anomaly_type, feature_attribution).
    """

    def __init__(self, wallet: Any | None = None):
        self.wallet = wallet

    async def forward(
        self,
        axons: list[Any] | Any,
        synapse: Any = None,
        timeout: float = 12.0,
        deserialize: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        """Return mock responses with VerificationSynapse fields populated."""
        if not isinstance(axons, list):
            axons = [axons]

        responses: list[Any] = []
        for _axon in axons:
            resp = synapse.model_copy()
            if isinstance(resp, VerificationSynapse):
                resp.anomaly_score = round(random.uniform(0.0, 1.0), 4)
                resp.confidence = round(random.uniform(0.5, 1.0), 4)
                resp.anomaly_type = "mock_anomaly"
                resp.feature_attribution = {"mock_feature": 0.5}
            responses.append(resp)

        if deserialize:
            return [r.deserialize() for r in responses]
        return responses

    async def __call__(
        self,
        axons: list[Any] | Any,
        synapse: Any = None,
        timeout: float = 12.0,
        deserialize: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        """Allow calling dendrite directly (matches SDK Dendrite interface)."""
        return await self.forward(
            axons=axons,
            synapse=synapse,
            timeout=timeout,
            deserialize=deserialize,
            **kwargs,
        )
