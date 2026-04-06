"""
Configuration helpers for miner and validator neurons.

Provides functions to add custom CLI arguments for each neuron type.
"""


def add_miner_args(parser):
    """Add miner-specific CLI arguments.

    Args:
        parser: argparse.ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--neuron.name",
        type=str,
        default="miner",
        help="Neuron name",
    )


def add_validator_args(parser):
    """Add validator-specific CLI arguments.

    Args:
        parser: argparse.ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--neuron.name",
        type=str,
        default="validator",
        help="Neuron name",
    )
    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        default=16,
        help="Number of miners to query per forward pass",
    )
    parser.add_argument(
        "--neuron.timeout",
        type=float,
        default=12.0,
        help="Dendrite query timeout in seconds",
    )
    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        default=0.1,
        help="EMA alpha for score updates",
    )
