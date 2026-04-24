"""Pytest configuration for deterministic_scoring test subpackage.

Forces PYTHONHASHSEED=0 at session scope so any accidental reliance on
dict/set iteration order is stable across test runs. The canonical JSON
encoder sorts keys explicitly, so correctness does not depend on this
fixture -- it is a belt-and-suspenders guard and mirrors the environment
the cross-process determinism subprocess test establishes for its child.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _force_pythonhashseed_zero() -> None:
    # This only affects hashes created inside this process; subprocess tests
    # set PYTHONHASHSEED=0 in the child env explicitly.
    os.environ.setdefault("PYTHONHASHSEED", "0")
