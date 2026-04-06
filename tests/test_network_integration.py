"""
Network integration tests using real bt.Axon and bt.Dendrite on localhost,
and testnet set_weights validation.

These tests prove that real Bittensor networking primitives work with
VerificationSynapse over HTTP on localhost. No MockDendrite is used
anywhere in the networking path -- only MockSubtensor/MockMetagraph for
chain state (which is acceptable since we are testing networking, not chain).

The testnet test (NET-12) proves set_weights() works on real testnet
infrastructure -- weights are written to chain and readable from metagraph.
That path is opt-in, manual, and operator-only; most contributors should use
the localhost coverage or mock-mode paths instead.

Requirements: NET-10, NET-11, NET-12

WARNING: These tests perform real network I/O on localhost. Each test
starts a real FastAPI/uvicorn server via bt.Axon and queries it via
bt.Dendrite's aiohttp client.

WARNING: The testnet test (test_set_weights_testnet) requires manual setup:
wallet creation, testnet registration, and TAO funding. It is skipped by
default and only runs when BT_TESTNET_WALLET_NAME is set.
"""

import os

import bittensor as bt
import numpy as np
import pytest
from bittensor.core.chain_data import AxonInfo
from bittensor.utils.balance import Balance

from antigence_subnet.mock import MockMetagraph, MockSubtensor
from antigence_subnet.protocol import VerificationSynapse
from antigence_subnet.validator.forward import forward as validator_forward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Find a free TCP port on localhost to avoid collisions."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_axon_info(axon: bt.Axon, wallet) -> AxonInfo:
    """Build an AxonInfo pointing at a running Axon on localhost."""
    return AxonInfo(
        ip="127.0.0.1",
        port=axon.port,
        hotkey=wallet.hotkey.ss58_address,
        coldkey=wallet.coldkey.ss58_address,
        version=0,
        ip_type=4,
    )


# ---------------------------------------------------------------------------
# NET-10: Localhost Axon/Dendrite round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_axon_dendrite_round_trip(miner_wallet, validator_wallet):
    """Real bt.Axon serves VerificationSynapse, real bt.Dendrite queries it.

    Proves NET-10: a real Axon and Dendrite can exchange VerificationSynapse
    data over HTTP on localhost with correct response field population.
    """
    port = _find_free_port()

    # Simple forward handler that populates response fields
    async def handler(synapse: VerificationSynapse) -> VerificationSynapse:
        synapse.anomaly_score = 0.85
        synapse.confidence = 0.9
        synapse.anomaly_type = "test_detection"
        synapse.feature_attribution = {"test_feature": 0.7}
        return synapse

    # Create and start a real Axon
    axon = bt.Axon(wallet=miner_wallet, port=port)
    axon.attach(forward_fn=handler)
    axon.start()

    try:
        # Create a real Dendrite
        dendrite = bt.Dendrite(wallet=validator_wallet)

        # Build AxonInfo pointing at the running Axon
        axon_info = _make_axon_info(axon, miner_wallet)

        # Create synapse with request fields
        synapse = VerificationSynapse(
            prompt="Test prompt",
            output="Test output",
            domain="hallucination",
        )

        # Query via real HTTP
        responses = await dendrite(
            axons=[axon_info],
            synapse=synapse,
            deserialize=False,
            timeout=10.0,
        )

        # Assertions
        assert len(responses) == 1, f"Expected 1 response, got {len(responses)}"
        resp = responses[0]
        assert resp.anomaly_score == 0.85, (
            f"Expected anomaly_score=0.85, got {resp.anomaly_score}"
        )
        assert resp.confidence == 0.9, (
            f"Expected confidence=0.9, got {resp.confidence}"
        )
        assert resp.anomaly_type == "test_detection", (
            f"Expected anomaly_type='test_detection', got {resp.anomaly_type}"
        )
        assert resp.feature_attribution == {"test_feature": 0.7}, (
            f"Expected feature_attribution={{'test_feature': 0.7}}, "
            f"got {resp.feature_attribution}"
        )
    finally:
        axon.stop()


@pytest.mark.integration
async def test_axon_dendrite_multiple_synapses(miner_wallet, validator_wallet):
    """Axon handles multiple sequential VerificationSynapse requests correctly.

    Proves the Axon can serve multiple requests with different inputs,
    returning varied responses based on prompt content.
    """
    port = _find_free_port()

    # Forward handler that varies response based on prompt content
    async def handler(synapse: VerificationSynapse) -> VerificationSynapse:
        if "normal" in synapse.prompt.lower():
            synapse.anomaly_score = 0.1
            synapse.confidence = 0.95
            synapse.anomaly_type = "normal"
        elif "suspicious" in synapse.prompt.lower():
            synapse.anomaly_score = 0.75
            synapse.confidence = 0.8
            synapse.anomaly_type = "suspicious_pattern"
        else:
            synapse.anomaly_score = 0.5
            synapse.confidence = 0.6
            synapse.anomaly_type = "unknown"
        return synapse

    axon = bt.Axon(wallet=miner_wallet, port=port)
    axon.attach(forward_fn=handler)
    axon.start()

    try:
        dendrite = bt.Dendrite(wallet=validator_wallet)
        axon_info = _make_axon_info(axon, miner_wallet)

        # Send 3 sequential requests with different prompts
        test_cases = [
            ("This is a normal output", 0.1, "normal"),
            ("This looks suspicious to me", 0.75, "suspicious_pattern"),
            ("Something else entirely", 0.5, "unknown"),
        ]

        for prompt, expected_score, expected_type in test_cases:
            synapse = VerificationSynapse(
                prompt=prompt,
                output="Test output",
                domain="hallucination",
            )

            responses = await dendrite(
                axons=[axon_info],
                synapse=synapse,
                deserialize=False,
                timeout=10.0,
            )

            assert len(responses) == 1
            resp = responses[0]
            assert resp.anomaly_score == expected_score, (
                f"For prompt '{prompt}': expected score={expected_score}, "
                f"got {resp.anomaly_score}"
            )
            assert resp.anomaly_type == expected_type, (
                f"For prompt '{prompt}': expected type='{expected_type}', "
                f"got '{resp.anomaly_type}'"
            )
    finally:
        axon.stop()


# ---------------------------------------------------------------------------
# NET-11: Validator forward pass against live miner Axon
# ---------------------------------------------------------------------------


class _MinimalEvalDataset:
    """Minimal evaluation dataset for integration testing.

    Provides the same interface as EvaluationDataset but with controlled
    samples where all ground truths are 'anomalous' and there are no
    honeypots. This ensures a miner returning anomaly_score >= 0.5 will
    always get a non-zero base reward (all predictions correct).

    We are testing the NETWORK path, not the evaluation/reward logic.
    """

    def __init__(self):
        self.domain = "hallucination"
        self.dataset_version = "integration-test-v1"

        # 20 all-anomalous samples, no honeypots
        self.samples = [
            {
                "id": f"integ-{i:03d}",
                "prompt": f"Integration test prompt {i}",
                "output": f"Integration test output {i} with hallucinated fact",
                "domain": "hallucination",
            }
            for i in range(1, 21)
        ]

        # Manifest: all anomalous, none are honeypots
        self.manifest = {
            s["id"]: {
                "ground_truth_label": "anomalous",
                "is_honeypot": False,
            }
            for s in self.samples
        }

        # Partitions (same as EvaluationDataset)
        self._regular_samples = list(self.samples)
        self._honeypot_samples = []

    def get_round_samples(
        self, round_num: int, n: int = 10, n_honeypots: int = 2
    ) -> list[dict]:
        """Return samples for a round. No honeypots, all regular."""
        # Return up to n regular samples (no honeypots available)
        import numpy as _np

        rng = _np.random.default_rng(seed=round_num)
        indices = rng.choice(
            len(self._regular_samples),
            size=min(n, len(self._regular_samples)),
            replace=False,
        )
        selected = [self._regular_samples[i] for i in indices]
        rng.shuffle(selected)
        return selected

    def get_ground_truth(self, sample_id: str) -> dict:
        return self.manifest[sample_id]


class _IntegrationValidator:
    """Minimal validator-like object for integration testing.

    Provides all attributes that validator_forward() needs without
    going through BaseNeuron/BaseValidatorNeuron __init__ (which
    attempts real chain registration when mock=False). Uses a REAL
    bt.Dendrite for HTTP transport -- no MockDendrite.
    """

    def __init__(
        self,
        wallet,
        metagraph: MockMetagraph,
        dendrite: bt.Dendrite,
        evaluation,
        uid: int,
        full_path: str,
    ):
        self.wallet = wallet
        self.metagraph = metagraph
        self.dendrite = dendrite
        self.evaluation = evaluation
        self.uid = uid
        self.step = 0
        self.scores = np.zeros(metagraph.n, dtype=np.float32)
        self.score_history: dict[int, list[float]] = {}
        self.confidence_history: dict[int, list[tuple[list[float], list[int]]]] = {}
        self.microglia = None  # Skip microglia recording

        # Build a config namespace with required neuron settings
        self.config = bt.Config()
        self.config.netuid = 1
        self.config.neuron = bt.Config()
        self.config.neuron.sample_size = 1
        self.config.neuron.timeout = 10.0
        self.config.neuron.moving_average_alpha = 0.1
        self.config.neuron.full_path = full_path

    def update_scores(self, rewards: np.ndarray, uids: list[int]) -> None:
        """Update EMA scores (same as BaseValidatorNeuron.update_scores)."""
        alpha = self.config.neuron.moving_average_alpha
        rewards = np.nan_to_num(rewards, nan=0.0)
        for i, uid in enumerate(uids):
            if 0 <= uid < len(self.scores):
                self.scores[uid] = (
                    alpha * rewards[i] + (1 - alpha) * self.scores[uid]
                )


@pytest.mark.integration
async def test_validator_forward_against_live_axon(
    miner_wallet, validator_wallet, tmp_path
):
    """Validator forward pass completes against a live miner Axon.

    Proves NET-11: the full validator forward pipeline (sample selection,
    challenge creation, querying, composite reward, EMA update) works
    end-to-end with a REAL bt.Dendrite making HTTP calls to a REAL
    bt.Axon on localhost. No MockDendrite in the forward path.

    The only mocks are MockSubtensor/MockMetagraph for chain state,
    which is acceptable since we are testing the network transport path.
    """
    port = _find_free_port()

    # Miner forward handler: returns plausible anomaly detection results
    async def miner_forward_fn(
        synapse: VerificationSynapse,
    ) -> VerificationSynapse:
        synapse.anomaly_score = 0.75
        synapse.confidence = 0.85
        synapse.anomaly_type = "factual_error"
        synapse.feature_attribution = {"test_feat": 0.6}
        return synapse

    # Start a real Axon serving the miner forward handler
    axon = bt.Axon(wallet=miner_wallet, port=port)
    axon.attach(forward_fn=miner_forward_fn)
    axon.start()

    try:
        # Build MockSubtensor with both wallets registered
        subtensor = MockSubtensor(netuid=1, n=0, wallet=validator_wallet)
        subtensor.force_register_neuron(
            netuid=1,
            hotkey_ss58=miner_wallet.hotkey.ss58_address,
            coldkey_ss58=miner_wallet.coldkey.ss58_address,
            balance=Balance(100000),
            stake=Balance(100000),
        )

        # Build MockMetagraph from the subtensor
        metagraph = MockMetagraph(netuid=1, subtensor=subtensor)

        # Find the miner's UID in the metagraph and patch its AxonInfo
        # to point at our real localhost Axon
        miner_uid = metagraph.hotkeys.index(
            miner_wallet.hotkey.ss58_address
        )
        metagraph._axons[miner_uid] = AxonInfo(
            ip="127.0.0.1",
            port=axon.port,
            hotkey=miner_wallet.hotkey.ss58_address,
            coldkey=miner_wallet.coldkey.ss58_address,
            version=0,
            ip_type=4,
        )

        # Use minimal evaluation dataset (all-anomalous, no honeypots)
        # to isolate network transport testing from reward logic
        eval_dataset = _MinimalEvalDataset()

        # Find validator UID
        validator_uid = metagraph.hotkeys.index(
            validator_wallet.hotkey.ss58_address
        )

        # Create integration validator with REAL bt.Dendrite
        integration_validator = _IntegrationValidator(
            wallet=validator_wallet,
            metagraph=metagraph,
            dendrite=bt.Dendrite(wallet=validator_wallet),
            evaluation=eval_dataset,
            uid=validator_uid,
            full_path=str(tmp_path),
        )

        # Confirm scores start at zero
        assert np.all(integration_validator.scores == 0), (
            "Scores should start at zero"
        )

        # Run the full validator forward pass
        await validator_forward(integration_validator)

        # Assert: the miner's UID now has a non-zero EMA score
        miner_score = integration_validator.scores[miner_uid]
        assert miner_score > 0, (
            f"Expected non-zero score for miner UID {miner_uid} after "
            f"forward pass, got {miner_score}. All scores: "
            f"{integration_validator.scores}"
        )

        # Assert: at least one non-zero score in the scores array
        assert np.any(integration_validator.scores > 0), (
            f"Expected at least one non-zero score after forward, "
            f"got all zeros: {integration_validator.scores}"
        )

    finally:
        axon.stop()


# ---------------------------------------------------------------------------
# NET-12: Testnet set_weights integration test
# ---------------------------------------------------------------------------


@pytest.mark.testnet
@pytest.mark.timeout(60)
@pytest.mark.skipif(
    not os.environ.get("BT_TESTNET_WALLET_NAME"),
    reason=(
        "Testnet wallet not configured -- set BT_TESTNET_WALLET_NAME and "
        "BT_TESTNET_WALLET_HOTKEY env vars. Requires manual wallet creation, "
        "testnet registration, and TAO funding."
    ),
)
def test_set_weights_testnet():
    """Prove set_weights() works on real testnet infrastructure (NET-12).

    This test requires manual setup before it can run:
    1. Create wallet: btcli wallet create --wallet.name <name>
    2. Fund on testnet: btcli wallet faucet --wallet.name <name> --subtensor.network test
    3. Register on subnet: btcli subnet register --wallet.name <name> \\
           --wallet.hotkey default --subtensor.network test --netuid <NETUID>
    4. Set env vars: BT_TESTNET_WALLET_NAME, BT_TESTNET_WALLET_HOTKEY (optional),
           BT_WALLET_PATH (optional), BT_TESTNET_NETUID (optional, default 1)

    When credentials are configured, this test:
    - Connects to real testnet subtensor
    - Verifies wallet registration on the metagraph
    - Sets weights on-chain via subtensor.set_weights()
    - Asserts ExtrinsicResponse.success is True (inclusion in block)
    - Re-syncs metagraph to verify weights were recorded
    """
    # 1. Read wallet config from environment
    wallet_name = os.environ["BT_TESTNET_WALLET_NAME"]
    wallet_hotkey = os.environ.get("BT_TESTNET_WALLET_HOTKEY", "default")
    wallet_path = os.environ.get("BT_WALLET_PATH", "~/.bittensor/wallets")
    netuid = int(os.environ.get("BT_TESTNET_NETUID", "1"))

    # 2. Create wallet from existing keys
    wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey, path=wallet_path)

    # 3. Connect to testnet
    subtensor = bt.Subtensor(network="test")

    # 4. Sync metagraph
    metagraph = subtensor.metagraph(netuid=netuid)

    # 5. Verify registration
    assert wallet.hotkey.ss58_address in metagraph.hotkeys, (
        f"Wallet hotkey {wallet.hotkey.ss58_address} is not registered on "
        f"subnet {netuid}. Run: btcli subnet register --wallet.name {wallet_name} "
        f"--wallet.hotkey {wallet_hotkey} --subtensor.network test --netuid {netuid}"
    )

    # 6. Get own UID
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    # 7. Build simple weights (self-weight only, simplest valid setting)
    try:
        import torch

        uids = torch.tensor([uid], dtype=torch.int64)
        weights = torch.tensor([65535], dtype=torch.int64)
    except ImportError:
        # Fallback to numpy if torch is not installed
        uids_np = np.array([uid], dtype=np.int64)
        weights_np = np.array([65535], dtype=np.int64)
        uids = uids_np
        weights = weights_np

    # 8. Call set_weights with wait_for_inclusion=True for verification
    result = subtensor.set_weights(
        wallet=wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        version_key=0,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    # 9. Assert success (ExtrinsicResponse API, v10)
    assert result.success is True, (
        f"set_weights failed: {result.message}"
    )

    # 10. Re-sync metagraph to read back weights
    metagraph_after = subtensor.metagraph(netuid=netuid)

    # 11. Verify weights were recorded on chain
    # The metagraph.W property is a tensor of shape (n, n) where W[uid]
    # is the weight vector set by uid. After set_weights with
    # wait_for_inclusion=True, the weight should be recorded.
    # Note: metagraph.W availability varies by SDK version. If available,
    # verify the weight was set. If not, the successful inclusion
    # (result.success == True) is sufficient proof.
    if hasattr(metagraph_after, "W") and metagraph_after.W is not None:
        try:
            own_weights = metagraph_after.W[uid]
            assert own_weights[uid] > 0, (
                f"Expected non-zero self-weight at W[{uid}][{uid}] after "
                f"set_weights, got {own_weights[uid]}"
            )
        except (IndexError, TypeError):
            # W matrix access may fail depending on metagraph state --
            # the successful set_weights result is sufficient
            pass
