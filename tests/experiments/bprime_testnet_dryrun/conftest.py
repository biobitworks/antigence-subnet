"""Pytest configuration for the B' testnet dry-run scaffolding (Phase 1104).

Responsibilities:

1. **sys.path wiring.** Prepend the dry-run package directory so
   ``import scripted_adversarial_miner`` resolves to
   ``experiments/bprime-testnet-dryrun/scripted_adversarial_miner.py``.
   Also prepend the repo root for any optional cross-imports from the
   v13.0 ``deterministic_scoring`` public API.

2. **PYTHONHASHSEED=0** is set (belt-and-suspenders; mirrors the
   deterministic_scoring conftest pattern). The miner itself does not
   rely on hash order, but tests that compare canonical JSON byte-strings
   across Python invocations do.

Non-responsibilities (explicit):

* Does NOT install a bittensor stub. The scripted miner is bittensor-free
  and its tests must remain so.
* Does NOT import anything from ``antigence_subnet.miner`` or the v13.1
  migration experiment. The dry-run has no transitive dependency on
  either.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

_CONFTEST_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _CONFTEST_DIR
for _ in range(8):
    if (_REPO_ROOT / "antigence_subnet").is_dir():
        break
    _REPO_ROOT = _REPO_ROOT.parent
else:  # pragma: no cover
    raise RuntimeError("Could not locate repo root from conftest location")

_DRYRUN_DIR = _REPO_ROOT / "experiments" / "bprime-testnet-dryrun"

for _p in (str(_REPO_ROOT), str(_DRYRUN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The dry-run source (experiments/bprime-testnet-dryrun/) is excluded from
# the public repo mirror. Skip collection cleanly when the module is absent,
# matching the established public-mirror skip pattern (see commits
# "fix: skip mirror manifest test in public repo CI" and
# "fix: skip phase95 report contract tests (artifacts archived)").
collect_ignore: list[str] = []
if not (_DRYRUN_DIR / "scripted_adversarial_miner.py").is_file():
    collect_ignore.append("test_scripted_miner.py")


@pytest.fixture(autouse=True, scope="session")
def _force_pythonhashseed_zero() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")


@pytest.fixture
def repo_root() -> pathlib.Path:
    return _REPO_ROOT


@pytest.fixture
def dryrun_dir() -> pathlib.Path:
    return _DRYRUN_DIR


@pytest.fixture
def fixture_path(dryrun_dir: pathlib.Path) -> pathlib.Path:
    return dryrun_dir / "fixtures" / "adversarial_pattern_v1.json"
