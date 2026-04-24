"""BP convergence detectors over score trajectories (BPCONV-02..05).

Three pure detectors, each consuming :class:`TrajectoryWindow` dicts:

- :func:`detect_oscillation` -- sign-change count of first differences
  (gaming / thrashing signal).
- :func:`detect_metastability` -- low-variance, below-top-quantile plateau
  (mediocre stuck state).
- :func:`detect_convergence_failure` -- pairwise divergence across two
  replica trajectory maps (cross-validator disagreement).

Every detector emits a list of dicts validating against the shared event
schema (``schema_version == EVENT_SCHEMA_VERSION``). All events JSON
round-trip without loss -- verified in tests.

Pure stdlib. No numpy. No bittensor. No randomness.
"""

from __future__ import annotations

import statistics
from typing import Mapping, Sequence

from antigence_subnet.validator.deterministic_scoring.trajectory import (
    TrajectoryWindow,
)

EVENT_SCHEMA_VERSION = 1


def _sign_changes(seq: Sequence[float]) -> int:
    """Count first-difference sign changes in ``seq``.

    Zero differences are skipped (no sign, no state reset). A change is
    counted each time the current nonzero sign is the opposite of the
    previously observed nonzero sign.
    """
    prev_sign: int | None = None
    count = 0
    for i in range(len(seq) - 1):
        diff = seq[i + 1] - seq[i]
        if diff == 0:
            continue
        sign = 1 if diff > 0 else -1
        if prev_sign is not None and sign != prev_sign:
            count += 1
        prev_sign = sign
    return count


def detect_oscillation(
    trajectories: Mapping[int, TrajectoryWindow],
    sign_change_threshold: int = 4,
) -> list[dict]:
    """Emit one ``oscillation`` event per miner whose trajectory flips too often.

    A miner's window is flagged when ``_sign_changes(ema_scores) >=
    sign_change_threshold``. Events are emitted in ascending UID order for
    deterministic output.
    """
    events: list[dict] = []
    for uid in sorted(trajectories.keys()):
        window = trajectories[uid]
        count = _sign_changes(window.ema_scores)
        if count >= sign_change_threshold:
            events.append(
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "event": "oscillation",
                    "uid": uid,
                    "round_range": [window.round_start, window.round_end],
                    "details": {
                        "sign_changes": count,
                        "window_size": len(window.ema_scores),
                        "threshold": sign_change_threshold,
                    },
                }
            )
    return events


def detect_metastability(
    trajectories: Mapping[int, TrajectoryWindow],
    variance_bound: float = 1e-4,
    top_quantile_cut: float = 0.5,
    min_consecutive_rounds: int = 10,
) -> list[dict]:
    """Emit ``metastability`` events for miners stuck at a low-variance plateau.

    A miner window triggers when:
        * ``len(ema_scores) >= min_consecutive_rounds``, AND
        * ``statistics.pvariance(ema_scores) < variance_bound``, AND
        * ``statistics.median(ema_scores) < top_quantile_cut``.

    Population variance (pvariance, not sample variance) is used so the
    result is deterministic and does not introduce Bessel's correction.
    Events are emitted in ascending UID order.
    """
    events: list[dict] = []
    for uid in sorted(trajectories.keys()):
        window = trajectories[uid]
        if len(window.ema_scores) < min_consecutive_rounds:
            continue
        variance = statistics.pvariance(window.ema_scores)
        median_val = statistics.median(window.ema_scores)
        if variance < variance_bound and median_val < top_quantile_cut:
            events.append(
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "event": "metastability",
                    "uid": uid,
                    "round_range": [window.round_start, window.round_end],
                    "details": {
                        "variance": variance,
                        "median": median_val,
                        "variance_bound": variance_bound,
                        "top_quantile_cut": top_quantile_cut,
                        "window_size": len(window.ema_scores),
                    },
                }
            )
    return events


def detect_convergence_failure(
    replica_a: Mapping[int, TrajectoryWindow],
    replica_b: Mapping[int, TrajectoryWindow],
    epsilon: float = 0.05,
) -> list[dict]:
    """Emit ``convergence_failure`` events where two replicas disagree on a UID.

    For each UID present in BOTH replicas, compute
    ``max_i |a.ema_scores[i] - b.ema_scores[i]|`` over the intersection of
    round ranges. If ``max_divergence > epsilon``, emit an event. Disjoint
    UIDs and non-overlapping windows are silently skipped. Events emitted
    in ascending UID order for deterministic output.
    """
    events: list[dict] = []
    shared_uids = sorted(set(replica_a.keys()) & set(replica_b.keys()))
    for uid in shared_uids:
        a = replica_a[uid]
        b = replica_b[uid]
        round_start = max(a.round_start, b.round_start)
        round_end = min(a.round_end, b.round_end)
        if round_end < round_start:
            continue
        a_slice = a.ema_scores[
            round_start - a.round_start : round_end - a.round_start + 1
        ]
        b_slice = b.ema_scores[
            round_start - b.round_start : round_end - b.round_start + 1
        ]
        max_diff = max(abs(x - y) for x, y in zip(a_slice, b_slice))
        if max_diff > epsilon:
            events.append(
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "event": "convergence_failure",
                    "uid": uid,
                    "round_range": [round_start, round_end],
                    "details": {
                        "max_divergence": max_diff,
                        "epsilon": epsilon,
                        "window_size": len(a_slice),
                    },
                }
            )
    return events


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "detect_convergence_failure",
    "detect_metastability",
    "detect_oscillation",
]
