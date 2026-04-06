"""
TOML configuration file loading for Antigence subnet neurons.

Provides a config file layer between code defaults and CLI args:
    CLI args > TOML config file > code defaults

Uses tomllib (Python 3.11+) with tomli backport for Python 3.10.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Search order for config file when no explicit path is given
CONFIG_SEARCH_PATHS = [
    Path("antigence_subnet.toml"),
    Path.home() / ".antigence" / "config.toml",
]


def find_config_file(explicit_path: str | None = None) -> Path | None:
    """Find a TOML config file.

    Search order:
    1. Explicit path (if provided and exists)
    2. antigence_subnet.toml in current working directory
    3. ~/.antigence/config.toml

    Returns the Path if found, None otherwise.
    """
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None
    for candidate in CONFIG_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    return None


def load_toml_config(path: Path) -> dict[str, Any]:
    """Load and return parsed TOML config from a file path.

    Raises tomllib.TOMLDecodeError on malformed TOML.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def flatten_toml(data: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested TOML dict to dotted keys for argparse.

    Example: {"neuron": {"timeout": 12.0}} -> {"neuron.timeout": 12.0}
    Non-dict values (scalars, lists) are kept as-is.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_toml(value, full_key))
        else:
            flat[full_key] = value
    return flat


def apply_toml_defaults(
    parser: argparse.ArgumentParser,
    config_path: str | None = None,
) -> Path | None:
    """Load TOML config and inject matching values as argparse defaults.

    Only sets defaults for keys that match existing parser arguments.
    Unknown TOML keys are silently ignored (forward compatibility).

    Returns the config file Path if loaded, None if no config found.
    """
    path = find_config_file(config_path)
    if path is None:
        return None

    data = load_toml_config(path)
    flat = flatten_toml(data)

    # Build mapping from --option.string (without leading dashes) to argparse dest
    known_options: dict[str, str] = {}
    known_dests: set[str] = set()
    for action in parser._actions:
        known_dests.add(action.dest)
        for opt in action.option_strings:
            clean = opt.lstrip("-")
            known_options[clean] = action.dest

    defaults: dict[str, Any] = {}
    for toml_key, value in flat.items():
        if toml_key in known_options:
            defaults[known_options[toml_key]] = value
        elif toml_key.replace(".", "_") in known_dests:
            defaults[toml_key.replace(".", "_")] = value

    if defaults:
        parser.set_defaults(**defaults)

    return path
