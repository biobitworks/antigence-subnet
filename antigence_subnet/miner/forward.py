"""
Miner forward logic with domain routing.

Routes incoming VerificationSynapse requests to the correct detector
based on the domain field. Rejects unsupported domains with status
code 400 per design decision D-07 (fail closed).
"""

import bittensor as bt

from antigence_subnet.miner.ensemble import ensemble_detect
from antigence_subnet.protocol import VerificationSynapse


async def forward(
    miner, synapse: VerificationSynapse
) -> VerificationSynapse:
    """Route synapse to correct detector based on domain field.

    Args:
        miner: The miner neuron instance with detectors and supported_domains.
        synapse: Incoming verification request.

    Returns:
        The synapse with response fields populated by the detector,
        or with error status if domain is unsupported.
    """
    # Input validation -- reject empty required fields
    if not synapse.prompt or not synapse.prompt.strip():
        synapse.axon.status_code = 400
        synapse.axon.status_message = "Rejected: empty prompt"
        bt.logging.warning("Rejected request: empty prompt")
        return synapse

    if not synapse.output or not synapse.output.strip():
        synapse.axon.status_code = 400
        synapse.axon.status_message = "Rejected: empty output"
        bt.logging.warning("Rejected request: empty output")
        return synapse

    # D-07: Reject unknown/unsupported domains (fail closed)
    if synapse.domain not in miner.supported_domains:
        synapse.axon.status_code = 400
        synapse.axon.status_message = (
            f"Unsupported domain: {synapse.domain}. "
            f"Supported: {miner.supported_domains}"
        )
        bt.logging.warning(
            f"Rejected request for unsupported domain: {synapse.domain}"
        )
        return synapse

    # Orchestrator path (D-10): if miner has orchestrator, route through it
    orchestrator = getattr(miner, "orchestrator", None)
    if orchestrator is not None:
        try:
            result = await orchestrator.process(
                prompt=synapse.prompt,
                output=synapse.output,
                domain=synapse.domain,
                code=synapse.code,
                context=synapse.context,
            )
            synapse.anomaly_score = result.score
            synapse.confidence = result.confidence
            synapse.anomaly_type = result.anomaly_type
            synapse.feature_attribution = result.feature_attribution

            bt.logging.debug(
                f"Orchestrator detection for '{synapse.domain}' | "
                f"Score: {result.score:.4f} | Confidence: {result.confidence:.4f}"
            )
            # Telemetry (Phase 40)
            telemetry = getattr(miner, "telemetry", None)
            if telemetry is not None:
                telemetry.record(synapse.domain, result.score, result.confidence)
                telemetry.update_prometheus(synapse.domain)
            return synapse
        except Exception as e:
            bt.logging.error(f"Orchestrator error for '{synapse.domain}': {e}")
            synapse.axon.status_code = 500
            synapse.axon.status_message = f"Orchestrator error: {type(e).__name__}"
            return synapse

    detector_or_ensemble = miner.detectors[synapse.domain]
    try:
        if isinstance(detector_or_ensemble, list):
            # Ensemble mode: multiple detectors for this domain
            result = await ensemble_detect(
                detectors=detector_or_ensemble,
                prompt=synapse.prompt,
                output=synapse.output,
                code=synapse.code,
                context=synapse.context,
            )
        else:
            # Single detector mode (backward compatible)
            result = await detector_or_ensemble.detect(
                prompt=synapse.prompt,
                output=synapse.output,
                code=synapse.code,
                context=synapse.context,
            )

        synapse.anomaly_score = result.score
        synapse.confidence = result.confidence
        synapse.anomaly_type = result.anomaly_type
        synapse.feature_attribution = result.feature_attribution
    except Exception as e:
        bt.logging.error(f"Detection error for domain '{synapse.domain}': {e}")
        synapse.axon.status_code = 500
        synapse.axon.status_message = f"Detection error: {type(e).__name__}"
        # Response fields remain None -- validator will score as zero
        return synapse

    bt.logging.debug(
        f"Domain '{synapse.domain}' detection complete | "
        f"Score: {result.score:.4f} | Confidence: {result.confidence:.4f}"
    )

    # Telemetry (Phase 40)
    telemetry = getattr(miner, "telemetry", None)
    if telemetry is not None:
        telemetry.record(synapse.domain, result.score, result.confidence)
        telemetry.update_prometheus(synapse.domain)

    return synapse
