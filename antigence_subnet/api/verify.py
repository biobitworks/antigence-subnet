"""
Standalone verify() library function for verification-as-a-service (NET-04).

Provides a simple synchronous API for external consumers to query the
Antigence trust signal without running a full validator node.

Usage:
    from antigence_subnet.api.verify import verify

    result = verify(
        prompt="What is 2+2?",
        output="2+2 is 5",
        domain="hallucination",
        subtensor_network="test",
    )
    print(result.trust_score, result.confidence)
"""

import asyncio
from dataclasses import dataclass, field

import bittensor as bt
import numpy as np

from antigence_subnet.mock import MockDendrite, MockMetagraph, MockSubtensor
from antigence_subnet.protocol import VerificationSynapse

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of a verification query against the subnet.

    Attributes:
        trust_score: Aggregated anomaly score (0.0=normal, 1.0=anomalous).
        confidence: Aggregated confidence of contributing miners.
        anomaly_types: Unique anomaly type labels from responding miners.
        contributing_miners: Number of miners that returned valid responses.
        raw_responses: Per-miner response dicts with anomaly_score, confidence, anomaly_type.
    """

    trust_score: float
    confidence: float
    anomaly_types: list[str]
    contributing_miners: int
    raw_responses: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Factory helpers (patchable for testing)
# ---------------------------------------------------------------------------


def _create_subtensor(network: str):
    """Create a Subtensor instance (real or mock)."""
    if network == "mock":
        return MockSubtensor(netuid=1, n=8)
    return bt.Subtensor(network=network)


def _create_metagraph(subtensor, netuid: int):
    """Create a metagraph from a subtensor."""
    if isinstance(subtensor, MockSubtensor):
        return MockMetagraph(netuid=netuid, subtensor=subtensor)
    return subtensor.metagraph(netuid=netuid)


def _create_dendrite(wallet):
    """Create a Dendrite instance (real or mock)."""
    if wallet is None:
        return MockDendrite()
    return bt.Dendrite(wallet=wallet)


# ---------------------------------------------------------------------------
# Internal async query
# ---------------------------------------------------------------------------


async def _query_validators(
    metagraph,
    dendrite,
    synapse: VerificationSynapse,
    top_k: int = 5,
    timeout: float = 12.0,
) -> list:
    """Select top-K neurons by stake and query them.

    Args:
        metagraph: Metagraph with stakes and axon info.
        dendrite: Dendrite for sending queries.
        synapse: VerificationSynapse with request fields filled.
        top_k: Number of top-stake miners to query.
        timeout: Query timeout in seconds.

    Returns:
        List of VerificationSynapse response objects.
    """
    k = min(top_k, metagraph.n)
    top_uids = np.argsort(metagraph.S)[-k:]

    axons = [metagraph.axons[uid] for uid in top_uids]
    responses = await dendrite(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
        deserialize=False,
    )
    return responses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify(
    prompt: str,
    output: str,
    domain: str,
    subtensor_network: str = "test",
    netuid: int = 1,
    top_k: int = 5,
    timeout: float = 12.0,
    code: str | None = None,
    context: str | None = None,
) -> VerificationResult:
    """Query the Antigence subnet for a trust score on an AI output.

    This is the main entry point for external consumers. It connects to the
    Bittensor network, selects top miners by stake, queries them, and
    aggregates the results into a VerificationResult.

    Args:
        prompt: Original prompt or input text.
        output: AI-generated output to verify.
        domain: Detection domain identifier (e.g., "hallucination").
        subtensor_network: Network to connect to ("test", "finney", "mock").
        netuid: Subnet UID to query.
        top_k: Number of top-stake miners to query.
        timeout: Query timeout in seconds.
        code: Optional code content for code_security domain.
        context: Optional JSON-serialized metadata.

    Returns:
        VerificationResult with aggregated trust score, confidence,
        anomaly types, and raw per-miner responses.
    """
    # --- Create network objects ---
    subtensor = _create_subtensor(subtensor_network)
    metagraph = _create_metagraph(subtensor, netuid)

    # Use default wallet for signing queries (may be None in mock mode)
    try:
        wallet = bt.Wallet()
    except Exception:
        wallet = None

    dendrite = _create_dendrite(wallet)

    # --- Build synapse ---
    synapse = VerificationSynapse(
        prompt=prompt,
        output=output,
        domain=domain,
        code=code,
        context=context,
    )

    # --- Query miners ---
    loop = asyncio.new_event_loop()
    try:
        responses = loop.run_until_complete(
            _query_validators(
                metagraph=metagraph,
                dendrite=dendrite,
                synapse=synapse,
                top_k=top_k,
                timeout=timeout,
            )
        )
    finally:
        loop.close()

    # --- Aggregate results ---
    valid_scores = []
    valid_confidences = []
    valid_weights = []
    anomaly_types_set: set[str] = set()
    raw_responses: list[dict] = []

    k = min(top_k, metagraph.n)
    top_uids = np.argsort(metagraph.S)[-k:]

    for i, resp in enumerate(responses):
        if resp.anomaly_score is not None:
            uid = top_uids[i]
            valid_scores.append(resp.anomaly_score)
            valid_confidences.append(
                resp.confidence if resp.confidence is not None else 0.5
            )
            valid_weights.append(float(metagraph.S[uid]))
            if resp.anomaly_type is not None:
                anomaly_types_set.add(resp.anomaly_type)
            raw_responses.append(
                {
                    "anomaly_score": resp.anomaly_score,
                    "confidence": resp.confidence,
                    "anomaly_type": resp.anomaly_type,
                }
            )

    if not valid_scores:
        return VerificationResult(
            trust_score=0.5,
            confidence=0.0,
            anomaly_types=[],
            contributing_miners=0,
            raw_responses=[],
        )

    # Weighted average by stake
    weights = np.array(valid_weights, dtype=np.float64)
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(len(valid_scores)) / len(valid_scores)

    scores_arr = np.array(valid_scores, dtype=np.float64)
    conf_arr = np.array(valid_confidences, dtype=np.float64)

    trust_score = float(np.dot(weights, scores_arr))
    confidence = float(np.dot(weights, conf_arr))

    return VerificationResult(
        trust_score=round(trust_score, 6),
        confidence=round(confidence, 6),
        anomaly_types=sorted(anomaly_types_set),
        contributing_miners=len(valid_scores),
        raw_responses=raw_responses,
    )
