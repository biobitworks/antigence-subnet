"""
Base neuron module adapted for Bittensor SDK v10.

Provides BaseNeuron with v10-compatible configuration, mock support,
and lifecycle management. All neuron classes (miner, validator) inherit
from this base.
"""

import argparse
import logging
import signal as signal_module
import sys
from pathlib import Path

import bittensor as bt

from antigence_subnet import __spec_version__ as spec_version
from antigence_subnet.mock import MockMetagraph, MockSubtensor
from antigence_subnet.utils.config_file import (
    apply_toml_defaults,
    find_config_file,
    load_toml_config,
)

# Project root = directory containing antigence_subnet/ and data/
# neuron.py lives at antigence_subnet/base/neuron.py -> 3 parents up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_EVAL_DATA_DIR = str(_PROJECT_ROOT / "data" / "evaluation")
_DEFAULT_TRAINING_DATA_DIR = str(_PROJECT_ROOT / "data" / "evaluation")


class BaseNeuron:
    """Base class for all Antigence subnet neurons.

    Handles configuration, wallet setup, subtensor/metagraph initialization,
    and provides context manager support for lifecycle management.
    """

    neuron_type: str = "BaseNeuron"
    spec_version: int = spec_version

    def __init__(self, config=None):
        # Parse config from CLI args if not provided
        self.config = config if config is not None else self.add_args()

        # Store raw TOML config for structured access (e.g., [miner.detectors] table)
        config_file_explicit = getattr(self.config, "config_file", None)
        toml_file = find_config_file(config_file_explicit)
        self._toml_raw = load_toml_config(toml_file) if toml_file else {}

        # Set up logging
        bt.logging(config=self.config)

        # Apply --logging.level if set
        log_level = getattr(getattr(self.config, "logging", None), "level", "info") or "info"
        if log_level == "debug":
            bt.logging.set_debug(True)
        elif log_level == "trace":
            bt.logging.set_trace(True)
        elif log_level in ("warning", "error"):
            logging.getLogger("bittensor").setLevel(getattr(logging, log_level.upper()))

        # Set up wallet (auto-create in mock mode for fresh-clone quick-start)
        self.wallet = bt.Wallet(config=self.config)
        if getattr(self.config, "mock", False):
            self.wallet.create_if_non_existent()

        # Set up subtensor (mock or real)
        if getattr(self.config, "mock", False):
            self.subtensor = MockSubtensor(
                netuid=self.config.netuid, wallet=self.wallet
            )
        else:
            self.subtensor = bt.Subtensor(config=self.config)

        # Set up metagraph (mock or real)
        if getattr(self.config, "mock", False):
            self.metagraph = MockMetagraph(
                netuid=self.config.netuid, subtensor=self.subtensor
            )
        else:
            self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)

        # Verify registration before proceeding (skip in mock mode)
        if not getattr(self.config, "mock", False) and not self.check_registered():
                bt.logging.error(
                    f"Hotkey {self.wallet.hotkey.ss58_address} not registered "
                    f"on subnet {self.config.netuid}. "
                    "Run: btcli subnet register --wallet.name <name> "
                    "--wallet.hotkey <hotkey> --subtensor.network test "
                    f"--netuid {self.config.netuid}"
                )
                sys.exit(1)

        # Determine own UID from metagraph
        self.uid = self._get_uid()

        # Graceful shutdown support (PROD-01)
        self.should_exit = False
        self._shutdown_timeout = (
            getattr(getattr(self.config, "neuron", None), "shutdown_timeout", 30) or 30
        )

        # Register signal handlers (main thread only)
        try:
            signal_module.signal(signal_module.SIGTERM, self._handle_shutdown_signal)
            signal_module.signal(signal_module.SIGINT, self._handle_shutdown_signal)
        except (OSError, ValueError):
            # ValueError: signal only works in main thread
            bt.logging.debug("Signal handlers not registered (not main thread)")

        network = "mock" if getattr(self.config, "mock", False) else self.config.subtensor.network
        bt.logging.info(
            f"Initialized {self.neuron_type} | UID: {self.uid} | Network: {network}"
        )

    def _get_uid(self) -> int:
        """Get the neuron's UID from the metagraph."""
        try:
            return self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
        except (ValueError, IndexError):
            bt.logging.warning(
                "Wallet hotkey not found in metagraph. "
                "Neuron may not be registered. Using UID 0."
            )
            return 0

    @classmethod
    def _add_args_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Add standard Bittensor and custom neuron args to parser.

        Subclasses can override to add more args, calling super first.
        """
        bt.Wallet.add_args(parser)
        bt.Subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.Axon.add_args(parser)

        # Custom neuron arguments
        parser.add_argument(
            "--netuid", type=int, default=1, help="Subnet netuid"
        )
        parser.add_argument(
            "--mock",
            action="store_true",
            default=False,
            help="Run in mock mode for testing",
        )
        parser.add_argument(
            "--neuron.full_path",
            type=str,
            default="~/.bittensor/neurons",
            help="Full path to neuron state directory",
        )
        parser.add_argument(
            "--neuron.device",
            type=str,
            default="cpu",
            help="Device to run on (cpu or cuda)",
        )
        parser.add_argument(
            "--neuron.shutdown_timeout",
            type=int,
            default=30,
            help="Graceful shutdown timeout in seconds (default: 30)",
        )
        parser.add_argument(
            "--logging.level",
            type=str,
            default="info",
            choices=["debug", "info", "warning", "error"],
            help="Logging level (debug/info/warning/error)",
        )
        parser.add_argument(
            "--config-file",
            type=str,
            default=None,
            help="Path to TOML config file (default: auto-discover antigence_subnet.toml)",
        )

    @classmethod
    def add_args(cls) -> bt.Config:
        """Create argument parser with all args and return Config.

        Integrates TOML config file defaults before bt.Config parses
        CLI args. Precedence: CLI args > TOML config > code defaults.
        """
        parser = argparse.ArgumentParser()
        cls._add_args_to_parser(parser)
        # Pre-parse to extract --config-file before TOML injection
        pre_args, _ = parser.parse_known_args()
        config_file_path = getattr(pre_args, "config_file", None)
        toml_path = apply_toml_defaults(parser, config_path=config_file_path)
        if toml_path:
            logging.getLogger("bittensor").info(f"Loaded config from {toml_path}")
        return bt.Config(parser)

    def check_registered(self) -> bool:
        """Verify the wallet hotkey is registered on the metagraph."""
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Wallet hotkey {self.wallet.hotkey.ss58_address} is not "
                f"registered on netuid {self.config.netuid}. "
                "Please register the hotkey before running."
            )
            return False
        return True

    def sync(self) -> None:
        """Re-sync metagraph from subtensor."""
        bt.logging.info("Syncing metagraph...")
        if getattr(self.config, "mock", False):
            self.metagraph.sync(subtensor=self.subtensor)
        else:
            self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        bt.logging.info(f"Metagraph synced. Neurons: {self.metagraph.n}")

    def _handle_shutdown_signal(self, signum, frame):
        """Set should_exit flag on SIGTERM/SIGINT for graceful shutdown."""
        sig_name = signal_module.Signals(signum).name
        bt.logging.info(f"Received {sig_name} -- initiating graceful shutdown")
        self.should_exit = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        bt.logging.info(f"Shutting down {self.neuron_type}.")
        return False
