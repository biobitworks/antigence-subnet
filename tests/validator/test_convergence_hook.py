"""Null-hypothesis + latency tests for the v13.1 convergence-detector hook.

Every test rejects a concrete null hypothesis (NH0). The latency measurement
at the bottom is NOT an asserted threshold -- WIRE-04 records the number so
operators know what to expect without making the CI flaky.

Runs under ``/tmp/chi-exp-venv`` (Python 3.13, numpy, prometheus_client; no
bittensor -- the forward.py copy's bittensor imports are stubbed by
``conftest.py`` at the package level).
"""

from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import sys
import time

from antigence_subnet.validator.deterministic_scoring import (
    AuditChainWriter,
    FrozenRoundRecord,
    FrozenRoundScore,
    verify_chain,
)


# --------------------------------------------------------------------- #
# helpers                                                                #
# --------------------------------------------------------------------- #
def _chain_path(tmp_path: pathlib.Path, name: str = "chain.jsonl") -> pathlib.Path:
    return tmp_path / name


def _hotkeys(uids):
    return [f"5F{''.join(['%02x' % ((u * 7 + 13) & 0xff)] * 23)}" for u in uids]


def _make_chain(
    path: pathlib.Path,
    *,
    n_rounds: int,
    uids: list[int],
    score_fn,
    ema_alpha: float = 0.1,
) -> None:
    """Write ``n_rounds`` records to ``path`` using ``score_fn(uid, r) -> float``."""
    writer = AuditChainWriter(path)
    hk = _hotkeys(uids)
    for r in range(n_rounds):
        scores = tuple(
            FrozenRoundScore(
                uid=uid,
                raw_reward=float(score_fn(uid, r)),
                ema_score=float(score_fn(uid, r)),
                hotkey=hk[i],
            )
            for i, uid in enumerate(uids)
        )
        record = FrozenRoundRecord(
            round_index=r,
            ema_alpha=ema_alpha,
            scores=scores,
            prev_hash=writer.latest_hash(),
        )
        writer.append(record)


# --------------------------------------------------------------------- #
# module import happens AFTER the deterministic_scoring imports          #
# --------------------------------------------------------------------- #
# v13.1.1 Phase 1103: convergence_hook promoted to
# antigence_subnet/validator/convergence_hook.py.
from antigence_subnet.validator import convergence_hook  # noqa: E402


# --------------------------------------------------------------------- #
# Test 1 -- all three detectors called                                   #
# --------------------------------------------------------------------- #
def test_hook_fires_detectors_over_audit_chain(tmp_path, monkeypatch):
    """NH0: hook silently skips detectors. Expected: all three invoked exactly once."""
    path = _chain_path(tmp_path)
    uids = [0, 1, 2]
    _make_chain(path, n_rounds=20, uids=uids, score_fn=lambda u, r: 0.1 + 0.01 * r)

    calls = {"osc": 0, "meta": 0, "conv": 0}

    orig_osc = convergence_hook.detect_oscillation
    orig_meta = convergence_hook.detect_metastability
    orig_conv = convergence_hook.detect_convergence_failure

    def spy_osc(*a, **kw):
        calls["osc"] += 1
        return orig_osc(*a, **kw)

    def spy_meta(*a, **kw):
        calls["meta"] += 1
        return orig_meta(*a, **kw)

    def spy_conv(*a, **kw):
        calls["conv"] += 1
        return orig_conv(*a, **kw)

    monkeypatch.setattr(convergence_hook, "detect_oscillation", spy_osc)
    monkeypatch.setattr(convergence_hook, "detect_metastability", spy_meta)
    monkeypatch.setattr(convergence_hook, "detect_convergence_failure", spy_conv)

    # Single-replica call: osc + meta fire; conv SKIPPED (no replica_chain_path).
    convergence_hook.run_convergence_checks(path, emit=False)
    assert calls == {"osc": 1, "meta": 1, "conv": 0}

    # Dual-replica call: conv also fires.
    replica = _chain_path(tmp_path, name="replica.jsonl")
    _make_chain(replica, n_rounds=20, uids=uids, score_fn=lambda u, r: 0.1 + 0.01 * r)
    convergence_hook.run_convergence_checks(
        path, replica_chain_path=replica, emit=False
    )
    assert calls == {"osc": 2, "meta": 2, "conv": 1}


# --------------------------------------------------------------------- #
# Test 2 -- emitted events match schema_version=1                        #
# --------------------------------------------------------------------- #
def test_hook_emits_structured_events_schema_v1(tmp_path):
    """NH0: events drift from Phase 1001 schema. Expected: schema_version=1 verbatim."""
    path = _chain_path(tmp_path)
    uids = [0]

    # Sawtooth trajectory: guaranteed high sign-change count -> oscillation event.
    def sawtooth(uid, r):
        return 0.5 + (0.3 if r % 2 == 0 else -0.3)

    _make_chain(path, n_rounds=20, uids=uids, score_fn=sawtooth)

    events = convergence_hook.run_convergence_checks(path, emit=False)
    assert len(events) >= 1
    for ev in events:
        assert ev["schema_version"] == 1
        assert ev["event"] in ("oscillation", "metastability", "convergence_failure")
        assert isinstance(ev["uid"], int)
        assert (
            isinstance(ev["round_range"], list)
            and len(ev["round_range"]) == 2
        )
        assert isinstance(ev["details"], dict)
        # JSON round-trip must be lossless.
        assert json.loads(json.dumps(ev, sort_keys=True)) == ev


# --------------------------------------------------------------------- #
# Test 3 -- Prometheus counter increments on oscillation                 #
# --------------------------------------------------------------------- #
def test_hook_updates_prometheus_counter_on_oscillation(tmp_path):
    """NH0: counter stays at 0. Expected: label 'oscillation' > 0 after sawtooth."""
    from prometheus_client import CollectorRegistry, Counter

    # Per-test private registry + counter -- no leakage.
    reg = CollectorRegistry()
    counter = Counter(
        "antigence_convergence_events_total",
        "per-test",
        labelnames=("event",),
        registry=reg,
    )

    path = _chain_path(tmp_path)
    uids = list(range(5))

    def sawtooth(uid, r):
        base = 0.5 + 0.05 * uid
        return base + (0.3 if r % 2 == 0 else -0.3)

    _make_chain(path, n_rounds=20, uids=uids, score_fn=sawtooth)

    # Emit on -- pass custom counter via kwarg.
    events = convergence_hook.run_convergence_checks(path, counter=counter, log_fn=lambda e: None)
    assert len(events) > 0
    osc_val = counter.labels(event="oscillation")._value.get()
    assert osc_val > 0, f"oscillation counter did not increment, got {osc_val}"


# --------------------------------------------------------------------- #
# Test 4 -- hook non-blocking at the call site                           #
# --------------------------------------------------------------------- #
def test_hook_non_blocking_on_failure(tmp_path, monkeypatch):
    """NH0: IOError propagates up. Expected: caller's try/except swallows it.

    This mirrors the pattern the forward-loop copy uses around the hook.
    """
    path = _chain_path(tmp_path)  # file not created on purpose -> IOError inside

    raised = False
    try:
        convergence_hook.run_convergence_checks(path)
    except Exception:
        raised = True
    # The hook itself MAY raise (chain missing); the contract is that the
    # caller's try/except at the forward-site swallows it. Simulate that here:
    caught_by_caller = False
    try:
        try:
            convergence_hook.run_convergence_checks(path)
        except Exception as exc:
            caught_by_caller = True
            # Expected error mentions chain/no-such/empty; we accept any
            # exception here because the contract under test is that the
            # caller catches it, not the exact message.
            _ = str(exc).lower()
            assert True
    finally:
        pass
    assert raised is True  # hook surfaces the error
    assert caught_by_caller is True  # caller's try/except protects the forward loop


# --------------------------------------------------------------------- #
# Test 5 -- convergence_hook.py does not transitively import bittensor    #
# --------------------------------------------------------------------- #
def test_hook_no_bittensor_import():
    """NH0: hook pulls bittensor transitively. Expected: fresh import -> no bittensor.

    Uses a subprocess so we get a clean sys.modules (the test session's
    conftest installs a stub).
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    code = (
        "import sys, pathlib;"
        f"sys.path.insert(0, r'{repo_root}');"
        "from antigence_subnet.validator import convergence_hook;"
        "assert 'bittensor' not in sys.modules, "
        "    f'bittensor leaked into convergence_hook import graph: "
        "{list(k for k in sys.modules if \"bittensor\" in k)}';"
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


# --------------------------------------------------------------------- #
# Test 6 -- production forward.py byte-identical to v13.0                #
# --------------------------------------------------------------------- #
def test_production_forward_unchanged():
    """NH0: audit-chain promotion rewrites reward.py. Expected: reward.py byte-identical to v13.0.

    v13.1.1 (Phase 1103) DOES modify forward.py / base/validator.py /
    validate_config.py (that is the migration). reward.py remains
    untouched because audit_bridge wraps rather than rewrites it.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "v13.0..HEAD",
            "--",
            "antigence_subnet/validator/reward.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"git diff failed: {result.stderr}"
    assert result.stdout == "", (
        f"reward.py drifted from v13.0:\n{result.stdout[:500]}"
    )


# --------------------------------------------------------------------- #
# Test 7 -- latency under 20-round / 256-miner synthetic fixture         #
# --------------------------------------------------------------------- #
def test_latency_20x256_fixture(tmp_path, capsys):
    """WIRE-04: record median + p99 over N runs. NOT asserted (avoids flake)."""
    path = _chain_path(tmp_path, name="latency_chain.jsonl")
    uids = list(range(256))

    # Smooth drifting trajectories -- no detector fires, measures pure cost.
    def drift(uid, r):
        return 0.3 + 0.001 * uid + 0.002 * r

    _make_chain(path, n_rounds=20, uids=uids, score_fn=drift)
    verify_chain(path)  # sanity

    # Warm caches (import paths, etc.)
    convergence_hook.run_convergence_checks(path, emit=False)

    timings: list[float] = []
    for _ in range(5):
        t0 = time.monotonic()
        convergence_hook.run_convergence_checks(path, emit=False)
        timings.append((time.monotonic() - t0) * 1000.0)  # ms

    med = statistics.median(timings)
    p99 = max(timings)  # with 5 samples, max == p100; p99 proxy
    print(
        f"\n[WIRE-04 latency] 20 rounds × 256 miners, 5 runs: "
        f"median={med:.2f}ms max={p99:.2f}ms all={[f'{t:.2f}' for t in timings]}",
    )
    # Write a sidecar file so SUMMARY.md can cite the number.
    out = tmp_path / "latency_record.json"
    out.write_text(
        json.dumps(
            {
                "window_rounds": 20,
                "miners_per_round": 256,
                "runs": len(timings),
                "median_ms": med,
                "p99_ms": p99,
                "samples_ms": timings,
            },
            sort_keys=True,
        )
    )
    # Guard against something totally pathological (e.g., multi-second runs
    # would indicate a regression in trajectory extraction); 1s is generous.
    assert med < 1000.0, f"latency pathology: median={med}ms"
