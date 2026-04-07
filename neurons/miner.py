"""
Reference miner entry point for the Antigence verification subnet.

Loads a detector via --detector CLI argument (importlib dynamic loading),
fits on normal-only seed samples at startup, and serves VerificationSynapse
responses. Third-party miners can swap in their own detector by specifying
a different --detector class path.
"""

import importlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple  # noqa: UP035 -- SDK Axon.attach() validates against typing.Tuple

import bittensor as bt

from antigence_subnet.base.miner import BaseMinerNeuron
from antigence_subnet.base.neuron import _DEFAULT_TRAINING_DATA_DIR
from antigence_subnet.miner.data import load_training_samples
from antigence_subnet.miner.detector import BaseDetector
from antigence_subnet.miner.forward import forward as miner_forward
from antigence_subnet.protocol import VerificationSynapse
from antigence_subnet.utils.runtime_metrics import (
    bootstrap_phase94_prometheus_exporter,
    build_runtime_snapshot,
    get_process_rss_bytes,
    load_phase94_runtime_config,
    start_periodic_json_export,
)

# Default detector class path
DEFAULT_DETECTOR = "antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector"


def load_detector(class_path: str, **kwargs) -> BaseDetector:
    """Dynamically load a detector class by its fully-qualified class path.

    Uses importlib to import the module and getattr to retrieve the class.
    This enables third-party miners to plug in their own BaseDetector
    implementations without modifying validator code.

    Args:
        class_path: Fully-qualified Python class path, e.g.
            'antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector'
        **kwargs: Additional keyword arguments passed to the detector constructor.
            Unknown kwargs are silently ignored for detectors that don't accept them.

    Returns:
        An instantiated BaseDetector subclass.

    Raises:
        ImportError: If the module cannot be found.
        AttributeError: If the class doesn't exist in the module.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    detector_cls = getattr(module, class_name)

    # Pass kwargs that the constructor accepts, ignore the rest
    import inspect

    sig = inspect.signature(detector_cls.__init__)
    accepted_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
    return detector_cls(**filtered_kwargs)


class Miner(BaseMinerNeuron):
    """Reference miner with config-based detector loading and fit-on-startup.

    Loads detector class via --detector CLI arg, fits on normal samples from
    seed evaluation data, and serves VerificationSynapse responses through
    the standard forward pipeline.
    """

    @classmethod
    def _add_args_to_parser(cls, parser):
        """Add miner-specific args including --detector and training data args."""
        super()._add_args_to_parser(parser)
        parser.add_argument(
            "--detector",
            type=str,
            default=DEFAULT_DETECTOR,
            help="Fully-qualified class path for detector (e.g., "
            "'antigence_subnet.miner.detectors.isolation_forest.IsolationForestDetector')",
        )
        parser.add_argument(
            "--neuron.training_data_dir",
            type=str,
            default=None,
            help="Path to evaluation data root directory (default: <project_root>/data/evaluation)",
        )
        parser.add_argument(
            "--neuron.training_domain",
            type=str,
            default="hallucination",
            help="Domain subdirectory for training data",
        )
        parser.add_argument(
            "--embedding-method",
            type=str,
            default="sbert",
            choices=["sbert", "tfidf"],
            help="Embedding method for hallucination detection (default: sbert, fallback: tfidf)",
        )

    def __init__(self, config=None):
        super().__init__(config=config)
        self.detectors = {}
        self.supported_domains = set()

        # Load detector via config-based class path
        detector_class_path = getattr(self.config, "detector", None) or DEFAULT_DETECTOR
        bt.logging.info(f"Loading detector: {detector_class_path}")
        embedding_method = getattr(self.config, "embedding_method", "sbert")
        detector = load_detector(detector_class_path, embedding_method=embedding_method)

        # Try restoring saved detector state before fitting (PROD-05)
        training_data_dir = (
            getattr(self.config.neuron, "training_data_dir", None) or _DEFAULT_TRAINING_DATA_DIR
        )
        training_domain = getattr(self.config.neuron, "training_domain", None) or "hallucination"

        state_dir = os.path.join(
            os.path.expanduser(getattr(self.config.neuron, "full_path", "~/.bittensor/neurons")),
            "detector_state",
        )
        state_loaded = False

        # Try per-domain subdirectory first, then flat dir (backward compat)
        single_detector_state_dir = os.path.join(state_dir, training_domain)
        for candidate_dir in [single_detector_state_dir, state_dir]:
            if os.path.exists(candidate_dir):
                try:
                    detector.load_state(candidate_dir)
                    if detector.get_info().get("is_fitted", False):
                        state_loaded = True
                        bt.logging.info(
                            f"Loaded saved detector state from {candidate_dir} -- skipping fit"
                        )
                        break
                except Exception as e:
                    bt.logging.warning(f"Failed to load saved detector state: {e}")

        # Only fit if no saved state was successfully loaded
        if not state_loaded:
            try:
                training_samples = load_training_samples(training_data_dir, training_domain)
                if training_samples:
                    detector.fit(training_samples)
                    bt.logging.info(
                        f"Detector fitted on {len(training_samples)} normal samples "
                        f"from {training_data_dir}/{training_domain}"
                    )
                else:
                    bt.logging.warning(
                        f"No normal training samples found in "
                        f"{training_data_dir}/{training_domain}. "
                        "Detector is unfitted -- scores will be meaningless."
                    )
            except FileNotFoundError:
                bt.logging.warning(
                    f"Training data not found at {training_data_dir}/{training_domain}. "
                    "Detector is unfitted -- scores will be meaningless."
                )

        # Register CLI-loaded detector
        self.detectors[detector.domain] = detector
        self.supported_domains.add(detector.domain)

        # Multi-domain detector loading from TOML [miner.detectors] table (MINE-01)
        # Supports both string (single) and list (ensemble) values
        detectors_table = self._toml_raw.get("miner", {}).get("detectors", {})
        if detectors_table:
            for domain_name, class_paths in detectors_table.items():
                # Support both string (single) and list (ensemble) values
                if isinstance(class_paths, str):
                    class_paths = [class_paths]

                if domain_name in self.detectors and len(class_paths) == 1:
                    bt.logging.info(
                        f"Skipping TOML detector for '{domain_name}' -- already loaded via CLI"
                    )
                    continue

                domain_detectors = []
                for class_path in class_paths:
                    bt.logging.info(
                        f"Loading TOML detector for domain '{domain_name}': {class_path}"
                    )
                    try:
                        domain_detector = load_detector(
                            class_path, embedding_method=embedding_method
                        )
                    except (ImportError, AttributeError) as e:
                        bt.logging.warning(
                            f"Failed to load detector for domain '{domain_name}': {e}"
                        )
                        continue

                    # Try restore saved state first (same pattern as single detector)
                    domain_state_dir = os.path.join(
                        state_dir, f"{domain_name}_{len(domain_detectors)}"
                    )
                    domain_state_loaded = False
                    if os.path.exists(domain_state_dir):
                        try:
                            domain_detector.load_state(domain_state_dir)
                            if domain_detector.get_info().get("is_fitted", False):
                                domain_state_loaded = True
                                bt.logging.info(
                                    f"Loaded saved state for '{domain_name}' "
                                    f"detector [{len(domain_detectors)}]"
                                )
                        except Exception as e:
                            bt.logging.warning(f"Failed to load state for '{domain_name}': {e}")

                    if not domain_state_loaded:
                        try:
                            domain_samples = load_training_samples(training_data_dir, domain_name)
                            if domain_samples:
                                domain_detector.fit(domain_samples)
                                bt.logging.info(
                                    f"Detector '{domain_name}' [{len(domain_detectors)}] "
                                    f"fitted on {len(domain_samples)} samples"
                                )
                            else:
                                bt.logging.warning(
                                    f"No training samples for domain '{domain_name}' "
                                    f"-- detector unfitted"
                                )
                        except FileNotFoundError:
                            bt.logging.warning(
                                f"Training data not found for domain '{domain_name}' "
                                f"-- detector unfitted"
                            )

                    domain_detectors.append(domain_detector)

                if len(domain_detectors) == 1:
                    # Single detector -- store directly (backward compat)
                    self.detectors[domain_name] = domain_detectors[0]
                elif len(domain_detectors) > 1:
                    # Ensemble -- store as list
                    self.detectors[domain_name] = domain_detectors
                    bt.logging.info(
                        f"Ensemble loaded for '{domain_name}': {len(domain_detectors)} detectors"
                    )
                self.supported_domains.add(domain_name)

        # Telemetry (Phase 40)
        from antigence_subnet.miner.orchestrator.telemetry import MinerTelemetry

        telemetry_config = self._toml_raw.get("miner", {}).get("telemetry", {})
        self.telemetry = MinerTelemetry(
            window_size=telemetry_config.get("window_size", 100),
        )
        self.telemetry.register_prometheus()
        bt.logging.info(f"Telemetry initialized | window_size={self.telemetry._window_size}")

        bt.logging.info(
            f"Miner ready | Detectors: {len(self.detectors)} | "
            f"Domains: {sorted(self.supported_domains)} | "
            f"Multi-domain: {len(self.detectors) > 1}"
        )

    async def forward(self, synapse: VerificationSynapse) -> VerificationSynapse:
        """Route synapse to detector based on domain."""
        return await miner_forward(self, synapse)

    async def blacklist(self, synapse: VerificationSynapse) -> Tuple[bool, str]:  # noqa: UP006 -- SDK Axon.attach() validates against typing.Tuple
        """Reject unregistered callers and zero-stake callers.

        Per user decision: blacklist rejects callers not registered on subnet
        OR with zero stake. Applied per-synapse on VerificationSynapse only.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Blacklist: request missing dendrite/hotkey")
            return (True, "Missing dendrite or hotkey")

        caller_hotkey = synapse.dendrite.hotkey
        if caller_hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f"Blacklist: unregistered hotkey {caller_hotkey}")
            return (True, "Unrecognized hotkey")

        caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
        caller_stake = float(self.metagraph.S[caller_uid])
        if caller_stake <= 0:
            bt.logging.trace(f"Blacklist: zero stake for {caller_hotkey}")
            return (True, "Insufficient stake")

        bt.logging.trace(f"Allowing request from {caller_hotkey} (stake={caller_stake})")
        return (False, "Hotkey recognized")

    async def priority(self, synapse: VerificationSynapse) -> float:
        """Return caller's stake as priority value.

        Per user decision: higher stake = higher priority in request queue.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0

        caller_hotkey = synapse.dendrite.hotkey
        if caller_hotkey not in self.metagraph.hotkeys:
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
        return float(self.metagraph.S[caller_uid])

    def save_state(self) -> None:
        """Save detector state to disk using per-domain subdirectories."""
        base_state_dir = os.path.join(
            os.path.expanduser(getattr(self.config.neuron, "full_path", "~/.bittensor/neurons")),
            "detector_state",
        )
        os.makedirs(base_state_dir, exist_ok=True)

        for domain, detector_or_list in self.detectors.items():
            if isinstance(detector_or_list, list):
                for i, detector in enumerate(detector_or_list):
                    det_dir = os.path.join(base_state_dir, f"{domain}_{i}")
                    os.makedirs(det_dir, exist_ok=True)
                    try:
                        detector.save_state(det_dir)
                        bt.logging.info(f"Saved ensemble detector state for '{domain}[{i}]'")
                    except Exception as e:
                        bt.logging.warning(
                            f"Failed to save ensemble detector state for '{domain}[{i}]': {e}"
                        )
            else:
                domain_dir = os.path.join(base_state_dir, domain)
                os.makedirs(domain_dir, exist_ok=True)
                try:
                    detector_or_list.save_state(domain_dir)
                    bt.logging.info(f"Saved detector state for domain '{domain}'")
                except Exception as e:
                    bt.logging.warning(f"Failed to save detector state for '{domain}': {e}")

    def load_state(self) -> None:
        """Load detector state from per-domain subdirectories."""
        base_state_dir = os.path.join(
            os.path.expanduser(getattr(self.config.neuron, "full_path", "~/.bittensor/neurons")),
            "detector_state",
        )

        if not os.path.exists(base_state_dir):
            bt.logging.info("No saved detector state found")
            return

        for domain, detector_or_list in self.detectors.items():
            if isinstance(detector_or_list, list):
                for i, detector in enumerate(detector_or_list):
                    det_dir = os.path.join(base_state_dir, f"{domain}_{i}")
                    if not os.path.exists(det_dir):
                        bt.logging.debug(f"No saved state for ensemble '{domain}[{i}]'")
                        continue
                    try:
                        detector.load_state(det_dir)
                        bt.logging.info(f"Loaded ensemble detector state for '{domain}[{i}]'")
                    except Exception as e:
                        bt.logging.warning(
                            f"Failed to load ensemble detector state for '{domain}[{i}]': {e}"
                        )
            else:
                domain_dir = os.path.join(base_state_dir, domain)
                if not os.path.exists(domain_dir):
                    bt.logging.debug(f"No saved state for domain '{domain}'")
                    continue
                try:
                    detector_or_list.load_state(domain_dir)
                    bt.logging.info(f"Loaded detector state for domain '{domain}'")
                except Exception as e:
                    bt.logging.warning(f"Failed to load detector state for '{domain}': {e}")


if __name__ == "__main__":
    with Miner() as miner:
        # Phase 94 env surface: PHASE94_MINER_METRICS_PORT,
        # PHASE94_MINER_TELEMETRY_EXPORT_DIR,
        # PHASE94_TELEMETRY_EXPORT_INTERVAL_SECONDS.
        runtime_cfg = load_phase94_runtime_config("miner")
        runtime_started_at = datetime.now(timezone.utc).isoformat()
        baseline_rss_bytes = get_process_rss_bytes()
        if runtime_cfg.metrics_port is not None:
            bootstrap_phase94_prometheus_exporter(runtime_cfg.metrics_port)

        def _build_phase94_miner_snapshot() -> dict:
            telemetry_dir = runtime_cfg.export_dir
            domain_stats = {}
            anomaly_count = 0
            if telemetry_dir is not None:
                telemetry_dir.mkdir(parents=True, exist_ok=True)
            for domain in sorted(miner.supported_domains):
                stats = miner.telemetry.get_stats(domain)
                domain_stats[domain] = stats
                anomaly_count += int(stats.get("anomaly_count", 0))
                if telemetry_dir is not None:
                    miner.telemetry.export_json(
                        domain,
                        str(telemetry_dir / f"{domain}.json"),
                    )
            return build_runtime_snapshot(
                role="miner",
                metrics_port=runtime_cfg.metrics_port,
                started_at_utc=runtime_started_at,
                baseline_rss_bytes=baseline_rss_bytes,
                anomaly_count=anomaly_count,
                extra_fields={
                    "hotkey_ss58": miner.wallet.hotkey.ss58_address,
                    "supported_domains": sorted(miner.supported_domains),
                    "telemetry": domain_stats,
                },
            )

        stop_event, export_thread = start_periodic_json_export(
            (Path(runtime_cfg.export_dir) / "runtime.json")
            if runtime_cfg.export_dir is not None
            else None,
            runtime_cfg.export_interval_seconds,
            _build_phase94_miner_snapshot,
        )
        try:
            miner.run()
        finally:
            if stop_event is not None:
                stop_event.set()
            if export_thread is not None:
                export_thread.join(timeout=2)
