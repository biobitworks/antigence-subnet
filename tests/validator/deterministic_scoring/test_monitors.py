"""Tests for the BP convergence detectors (oscillation / metastability / convergence-failure)."""

from __future__ import annotations

import json

import pytest

from antigence_subnet.validator.deterministic_scoring.monitors import (
    EVENT_SCHEMA_VERSION,
    detect_convergence_failure,
    detect_metastability,
    detect_oscillation,
)
from antigence_subnet.validator.deterministic_scoring.trajectory import (
    TrajectoryWindow,
)


# ---------------------------------------------------------------------------
# Oscillation detector
# ---------------------------------------------------------------------------

def _win(uid: int, scores: tuple[float, ...], start: int = 0) -> TrajectoryWindow:
    return TrajectoryWindow(
        uid=uid,
        round_start=start,
        round_end=start + len(scores) - 1,
        ema_scores=scores,
    )


def test_oscillation_triggers_on_sawtooth():
    sawtooth = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9)
    trajs = {42: _win(42, sawtooth)}
    events = detect_oscillation(trajs)
    assert len(events) == 1
    ev = events[0]
    assert ev["uid"] == 42
    assert ev["event"] == "oscillation"
    assert ev["schema_version"] == 1
    assert ev["details"]["sign_changes"] >= 4


def test_oscillation_does_not_trigger_on_monotone():
    monotone = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    trajs = {1: _win(1, monotone)}
    assert detect_oscillation(trajs) == []


def test_oscillation_does_not_trigger_on_flat():
    flat = tuple([0.5] * 10)
    trajs = {1: _win(1, flat)}
    assert detect_oscillation(trajs) == []


def test_oscillation_threshold_respected():
    sawtooth = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9)
    trajs = {1: _win(1, sawtooth)}
    assert detect_oscillation(trajs, sign_change_threshold=100) == []


def test_oscillation_event_json_serializable():
    sawtooth = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9)
    trajs = {3: _win(3, sawtooth)}
    events = detect_oscillation(trajs)
    assert len(events) == 1
    blob = json.dumps(events[0])
    reparsed = json.loads(blob)
    assert reparsed == events[0]
    assert reparsed["schema_version"] == EVENT_SCHEMA_VERSION


def test_oscillation_round_range_correct():
    sawtooth = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9)
    trajs = {9: _win(9, sawtooth, start=5)}
    events = detect_oscillation(trajs)
    assert len(events) == 1
    assert events[0]["round_range"] == [5, 14]


# ---------------------------------------------------------------------------
# Metastability detector
# ---------------------------------------------------------------------------

def test_metastability_triggers_on_flat_low_score():
    # Exactly flat: variance = 0, median = 0.30 < cut 0.5.
    trajs = {1: _win(1, tuple([0.30] * 12))}
    events = detect_metastability(trajs)
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "metastability"
    assert ev["uid"] == 1
    assert ev["schema_version"] == 1
    assert ev["details"]["variance"] < 1e-4
    assert ev["details"]["median"] == 0.30


def test_metastability_triggers_on_jittered_low_score():
    # Plan-checker concern #1: exercise the variance bound non-trivially
    # with a tiny jitter around 0.30 so pvariance > 0 but still < 1e-4.
    # pvariance of 12 samples with ±0.003 jitter is roughly 6e-6 -- well
    # below the 1e-4 bound. Median remains ~0.30 < 0.5.
    jittered = (
        0.300, 0.303, 0.297, 0.302, 0.298, 0.301,
        0.299, 0.302, 0.298, 0.301, 0.299, 0.300,
    )
    trajs = {2: _win(2, jittered)}
    events = detect_metastability(trajs)
    assert len(events) == 1
    ev = events[0]
    assert ev["details"]["variance"] > 0.0          # truly nontrivial
    assert ev["details"]["variance"] < 1e-4         # but still below bound
    assert ev["details"]["median"] < 0.5


def test_metastability_does_not_trigger_on_top_quantile():
    trajs = {1: _win(1, tuple([0.95] * 12))}
    assert detect_metastability(trajs) == []


def test_metastability_does_not_trigger_on_high_variance():
    # Orthogonal to oscillation detector: flipping trajectory has high
    # variance and therefore MUST NOT trigger metastability.
    trajs = {1: _win(1, (0.1, 0.9) * 6)}
    assert detect_metastability(trajs) == []


def test_metastability_respects_min_consecutive_rounds():
    trajs = {1: _win(1, tuple([0.30] * 5))}
    assert detect_metastability(trajs, min_consecutive_rounds=10) == []


def test_metastability_event_json_serializable():
    trajs = {7: _win(7, tuple([0.30] * 12), start=3)}
    events = detect_metastability(trajs)
    assert len(events) == 1
    blob = json.dumps(events[0])
    reparsed = json.loads(blob)
    assert reparsed == events[0]
    assert reparsed["schema_version"] == EVENT_SCHEMA_VERSION
    assert reparsed["round_range"] == [3, 14]


def test_metastability_custom_thresholds():
    # Flat 0.7 is above default 0.5 cut (no trigger), but honoring the
    # custom cut=0.8 must trigger (0.7 < 0.8, variance = 0).
    trajs = {1: _win(1, tuple([0.7] * 12))}
    assert detect_metastability(trajs) == []
    events = detect_metastability(trajs, top_quantile_cut=0.8)
    assert len(events) == 1
    assert events[0]["details"]["top_quantile_cut"] == 0.8


# ---------------------------------------------------------------------------
# Convergence-failure detector
# ---------------------------------------------------------------------------

def test_convergence_failure_triggers_when_replicas_diverge():
    a = {5: TrajectoryWindow(uid=5, round_start=0, round_end=2, ema_scores=(0.1, 0.2, 0.3))}
    b = {5: TrajectoryWindow(uid=5, round_start=0, round_end=2, ema_scores=(0.1, 0.2, 0.9))}
    events = detect_convergence_failure(a, b)
    assert len(events) == 1
    ev = events[0]
    assert ev["uid"] == 5
    assert ev["event"] == "convergence_failure"
    assert ev["details"]["max_divergence"] == pytest.approx(0.6)
    assert ev["details"]["epsilon"] == 0.05
    assert ev["schema_version"] == 1


def test_convergence_failure_no_trigger_when_aligned():
    a = {1: _win(1, (0.5, 0.5, 0.5))}
    b = {1: _win(1, (0.5, 0.5, 0.5))}
    assert detect_convergence_failure(a, b) == []


def test_convergence_failure_respects_epsilon():
    a = {1: _win(1, (0.50, 0.50, 0.50))}
    b = {1: _win(1, (0.52, 0.50, 0.50))}  # diverges by 0.02 < default 0.05
    assert detect_convergence_failure(a, b) == []
    # Custom epsilon that DOES trigger (0.02 > 0.01).
    events = detect_convergence_failure(a, b, epsilon=0.01)
    assert len(events) == 1
    assert events[0]["details"]["max_divergence"] == pytest.approx(0.02)


def test_convergence_failure_skips_disjoint_uids():
    a = {1: _win(1, (0.1, 0.2, 0.3))}
    b = {2: _win(2, (0.7, 0.8, 0.9))}
    assert detect_convergence_failure(a, b) == []


def test_convergence_failure_event_json_serializable():
    a = {5: _win(5, (0.1, 0.2, 0.3))}
    b = {5: _win(5, (0.1, 0.2, 0.9))}
    events = detect_convergence_failure(a, b)
    assert len(events) == 1
    blob = json.dumps(events[0])
    reparsed = json.loads(blob)
    assert reparsed == events[0]
    assert reparsed["schema_version"] == EVENT_SCHEMA_VERSION


def test_convergence_failure_round_range_is_intersection():
    # A spans rounds 5..14 (10 entries), B spans 7..14 (8 entries).
    a_scores = tuple(0.1 * i for i in range(10))           # rounds 5..14
    b_scores = tuple(0.9 * i for i in range(8))            # rounds 7..14
    a = {9: TrajectoryWindow(uid=9, round_start=5, round_end=14, ema_scores=a_scores)}
    b = {9: TrajectoryWindow(uid=9, round_start=7, round_end=14, ema_scores=b_scores)}
    events = detect_convergence_failure(a, b)
    assert len(events) == 1
    ev = events[0]
    assert ev["round_range"] == [7, 14]
    assert ev["details"]["window_size"] == 8
    # The slice of A for rounds 7..14 is the last 8 values of a_scores.
    a_slice = a_scores[-8:]
    expected_max = max(abs(x - y) for x, y in zip(a_slice, b_scores))
    assert ev["details"]["max_divergence"] == pytest.approx(expected_max)
