"""Convergence-detector hook for v13.1 migration (experiment-first, Phase 1101).

Reads the SHA-256 linked audit chain produced by Phase 1100's
``bridge_get_rewards`` + ``AuditChainWriter``, extracts per-UID trajectory
windows via the Phase 1001 public API, runs the three deterministic
detectors (``detect_oscillation``, ``detect_metastability``,
``detect_convergence_failure``), and emits each event to both a dedicated
Prometheus counter and a structured JSON log line.

Hard contracts (enforced by tests under ``tests/experiments/v13_1_migration/``):

* **Zero bittensor imports.** This module must remain importable in
  ``/tmp/chi-exp-venv`` without bittensor. ``convergence_hook`` depends
  only on ``antigence_subnet.validator.deterministic_scoring`` (pure stdlib),
  plus ``prometheus_client`` (a standard dependency of
  ``antigence_subnet.api.metrics`` already).
* **Schema fidelity.** Events are returned/emitted exactly as the
  Phase 1001 detectors produce them -- ``schema_version == 1``,
  no wrapping, no mutation, no extra keys.
* **Non-blocking by default at the call site.** This module itself MAY
  raise (``ChainIntegrityError``, ``ValueError``) when the chain is
  tampered or malformed -- that is diagnostically useful. The caller (the
  forward-loop copy in ``production_copy/forward.py``) wraps invocation
  in ``try/except`` and logs at WARNING level rather than propagating.

CLI / TOML wiring (``--convergence.*`` flags, ``[validator.convergence]``
keys) is DEFERRED to Phase 1102. This phase exposes configurability via
the ``config`` argument of :func:`run_convergence_checks`, and via the
module-level :data:`DEFAULT_CONVERGENCE_CONFIG` mapping.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Callable, Mapping
from typing import Any

from prometheus_client import CollectorRegistry, Counter

from antigence_subnet.validator.deterministic_scoring import (
    detect_convergence_failure,
    detect_metastability,
    detect_oscillation,
    extract_trajectories,
)

__all__ = [
    "DEFAULT_CONVERGENCE_CONFIG",
    "CONVERGENCE_REGISTRY",
    "CONVERGENCE_EVENTS_COUNTER",
    "emit_events",
    "run_convergence_checks",
    "default_log_fn",
]


# --------------------------------------------------------------------- #
# Thresholds (Phase 1002 defaults; mirror Phase 1001 spec)               #
# --------------------------------------------------------------------- #
DEFAULT_CONVERGENCE_CONFIG: dict[str, Any] = {
    "window_size": 20,
    "sign_change_threshold": 4,
    "variance_bound": 1e-4,
    "top_quantile_cut": 0.5,
    "min_consecutive_rounds": 10,
    "epsilon": 0.05,
}


# --------------------------------------------------------------------- #
# Prometheus surface                                                     #
# --------------------------------------------------------------------- #
# Dedicated registry -- keeps the hook's counter out of the default
# prometheus_client global REGISTRY so tests can assert on label values
# without leakage across test runs or from other exporters. Production
# promotion (Phase 1103) can merge this into the existing
# ``antigence_subnet.api.metrics`` registry via ``registry.register(...)``.
CONVERGENCE_REGISTRY: CollectorRegistry = CollectorRegistry()
CONVERGENCE_EVENTS_COUNTER: Counter = Counter(
    "antigence_convergence_events_total",
    "Convergence-detector events by type (oscillation, metastability, convergence_failure).",
    labelnames=("event",),
    registry=CONVERGENCE_REGISTRY,
)


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #
def default_log_fn(event: Mapping[str, Any]) -> None:
    """Serialize ``event`` as a single-line JSON record on stderr."""
    print(json.dumps(dict(event), sort_keys=True), file=sys.stderr)


def _resolve_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONVERGENCE_CONFIG)
    if config:
        merged.update(config)
    return merged


# --------------------------------------------------------------------- #
# Emission                                                               #
# --------------------------------------------------------------------- #
def emit_events(
    events: list[dict],
    *,
    counter: Counter | None = None,
    log_fn: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    """Increment Prometheus counter (per event label) and log each event as JSON.

    Parameters
    ----------
    events
        List of event dicts as returned by the Phase 1001 detectors.
    counter
        Override for :data:`CONVERGENCE_EVENTS_COUNTER`. Tests pass their own
        to isolate labels; production should use the module-level default.
    log_fn
        Override for :func:`default_log_fn`. Accepts a single event mapping.
    """
    c = counter if counter is not None else CONVERGENCE_EVENTS_COUNTER
    lg = log_fn if log_fn is not None else default_log_fn
    for event in events:
        event_type = event.get("event", "unknown")
        c.labels(event=event_type).inc()
        lg(event)


# --------------------------------------------------------------------- #
# Top-level entry point                                                  #
# --------------------------------------------------------------------- #
def run_convergence_checks(
    chain_path: pathlib.Path | str,
    *,
    config: Mapping[str, Any] | None = None,
    replica_chain_path: pathlib.Path | str | None = None,
    emit: bool = True,
    counter: Counter | None = None,
    log_fn: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict]:
    """Run the three Phase 1001 detectors over an audit-chain window.

    Pipeline:

    1. :func:`extract_trajectories` on ``chain_path`` using
       ``config["window_size"]``.
    2. :func:`detect_oscillation` + :func:`detect_metastability` over the
       single trajectory map.
    3. If ``replica_chain_path`` is provided, :func:`extract_trajectories`
       on it too, then :func:`detect_convergence_failure` across the pair.
       Otherwise the convergence-failure detector is skipped (nothing to
       compare).
    4. If ``emit`` is True, push events through :func:`emit_events`.
    5. Return the concatenated event list (in invocation order:
       oscillation → metastability → convergence_failure).

    Raises:
        ChainIntegrityError: if the chain file is tampered or malformed
            (propagated from :func:`extract_trajectories`).
        ValueError: on invalid window_size (propagated).
    """
    cfg = _resolve_config(config)
    trajectories = extract_trajectories(chain_path, window_size=cfg["window_size"])

    events: list[dict] = []
    events.extend(
        detect_oscillation(
            trajectories,
            sign_change_threshold=cfg["sign_change_threshold"],
        )
    )
    events.extend(
        detect_metastability(
            trajectories,
            variance_bound=cfg["variance_bound"],
            top_quantile_cut=cfg["top_quantile_cut"],
            min_consecutive_rounds=cfg["min_consecutive_rounds"],
        )
    )
    if replica_chain_path is not None:
        replica_trajectories = extract_trajectories(
            replica_chain_path, window_size=cfg["window_size"]
        )
        events.extend(
            detect_convergence_failure(
                trajectories,
                replica_trajectories,
                epsilon=cfg["epsilon"],
            )
        )

    if emit and events:
        emit_events(events, counter=counter, log_fn=log_fn)
    return events
