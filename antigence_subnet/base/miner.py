"""
Base miner neuron module.

Provides BaseMinerNeuron with axon setup, forward/blacklist/priority
attachment, and lifecycle management. Concrete miners subclass this.
"""

import time
from abc import abstractmethod
from typing import Tuple  # noqa: UP035 -- SDK Axon.attach() validates against typing.Tuple

import bittensor as bt

from antigence_subnet.base.neuron import BaseNeuron
from antigence_subnet.protocol import VerificationSynapse


class BaseMinerNeuron(BaseNeuron):
    """Base class for miner neurons.

    Sets up axon with forward, blacklist, and priority functions attached.
    Subclasses must implement the forward method to handle VerificationSynapse.
    """

    neuron_type: str = "MinerNeuron"

    def __init__(self, config=None):
        super().__init__(config=config)

        # Set up axon
        self.axon = bt.Axon(wallet=self.wallet, config=self.config)

        # Attach handlers to axon
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

        bt.logging.info(f"Axon created and handlers attached for {self.neuron_type}")

    @abstractmethod
    async def forward(
        self, synapse: VerificationSynapse
    ) -> VerificationSynapse:
        """Process a VerificationSynapse request.

        Must be implemented by subclasses. Routes synapse to the
        appropriate detector based on domain field.

        Args:
            synapse: The incoming verification request.

        Returns:
            The synapse with response fields populated.
        """
        ...

    async def blacklist(
        self, synapse: VerificationSynapse
    ) -> Tuple[bool, str]:  # noqa: UP006 -- SDK Axon.attach() validates against typing.Tuple
        """Check if a request should be blacklisted.

        Default: allow all requests. Phase 4 adds real filtering
        based on validator registration, stake, and rate limits.

        Args:
            synapse: The incoming verification request.

        Returns:
            Tuple of (should_blacklist, reason).
        """
        return (False, "")

    async def priority(self, synapse: VerificationSynapse) -> float:
        """Determine request priority.

        Default: flat priority (0.0). Phase 4 adds stake-based priority.

        Args:
            synapse: The incoming verification request.

        Returns:
            Priority value (higher = more priority).
        """
        return 0.0

    def run(self) -> None:
        """Start the miner: serve axon, sync, and loop."""
        bt.logging.info(f"Starting {self.neuron_type}...")
        self.load_state()

        # Serve the axon
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info(f"Miner axon serving on port {self.axon.port}")

        try:
            while not self.should_exit:
                self.sync()
                if not self.should_exit:
                    time.sleep(12)  # Sync every block (~12s)
        finally:
            self.axon.stop()
            self.save_state()
            bt.logging.info("Miner graceful shutdown complete")

    def save_state(self) -> None:
        """Save miner state. Subclasses override for detector-specific state."""
        pass

    def load_state(self) -> None:
        """Load miner state. Subclasses override for detector-specific state."""
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit -- stop axon."""
        if hasattr(self, "axon"):
            self.axon.stop()
        self.save_state()
        bt.logging.info(f"Shutting down {self.neuron_type}.")
        return False
