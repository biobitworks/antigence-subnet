"""
Validator entry point for the Antigence verification subnet.

Creates a Validator neuron that queries miners, scores responses,
and updates EMA scores. Phase 2 replaces the placeholder reward
with precision-first scoring.
"""

from datetime import datetime, timezone
from pathlib import Path

from antigence_subnet.base.validator import BaseValidatorNeuron
from antigence_subnet.utils.runtime_metrics import (
    bootstrap_phase94_prometheus_exporter,
    build_runtime_snapshot,
    get_process_rss_bytes,
    load_phase94_runtime_config,
    start_periodic_json_export,
)
from antigence_subnet.validator.forward import forward as validator_forward


class Validator(BaseValidatorNeuron):
    """Concrete validator for the Antigence subnet.

    Queries miners with VerificationSynapse, scores responses,
    and updates weights on the metagraph.
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    async def forward(self):
        """Run a single forward pass -- no synapse arg (Pitfall 3)."""
        return await validator_forward(self)


if __name__ == "__main__":
    with Validator() as validator:
        # Phase 94 env surface: PHASE94_VALIDATOR_METRICS_PORT,
        # PHASE94_VALIDATOR_RUNTIME_EXPORT_DIR,
        # PHASE94_TELEMETRY_EXPORT_INTERVAL_SECONDS.
        runtime_cfg = load_phase94_runtime_config("validator")
        runtime_started_at = datetime.now(timezone.utc).isoformat()
        baseline_rss_bytes = get_process_rss_bytes()
        if runtime_cfg.metrics_port is not None:
            from antigence_subnet.api.metrics import get_collector

            bootstrap_phase94_prometheus_exporter(
                runtime_cfg.metrics_port,
                collector_factory=get_collector,
            )

        def _build_phase94_validator_snapshot() -> dict:
            connected_miners_count = int((validator.scores > 0).sum())
            return build_runtime_snapshot(
                role="validator",
                metrics_port=runtime_cfg.metrics_port,
                started_at_utc=runtime_started_at,
                baseline_rss_bytes=baseline_rss_bytes,
                extra_fields={
                    "hotkey_ss58": validator.wallet.hotkey.ss58_address,
                    "step": int(validator.step),
                    "connected_miners_count": connected_miners_count,
                    "policy": {
                        "mode": validator.config.policy.mode,
                        "high_threshold": validator.config.policy.high_threshold,
                        "low_threshold": validator.config.policy.low_threshold,
                        "min_confidence": validator.config.policy.min_confidence,
                    },
                },
            )

        stop_event, export_thread = start_periodic_json_export(
            (Path(runtime_cfg.export_dir) / "runtime.json")
            if runtime_cfg.export_dir is not None
            else None,
            runtime_cfg.export_interval_seconds,
            _build_phase94_validator_snapshot,
        )
        try:
            validator.run()
        finally:
            if stop_event is not None:
                stop_event.set()
            if export_thread is not None:
                export_thread.join(timeout=2)
