"""Phase 1102 STATEPOL-03: mid-session audit enablement integration test.

Scenario:

    1. Instantiate a stub validator with ``config.audit.enabled=False``.
    2. Run 3 "rounds" of the forward path -- simulating reward
       computation WITHOUT the audit bridge (the disabled branch).
       Verify: no ``chain.jsonl`` exists.
    3. Flip ``config.audit.enabled=True`` mid-session.
    4. Run 2 more "rounds" -- now going through the audit bridge.
       Verify: ``chain.jsonl`` contains EXACTLY 2 records, starting from
       ``GENESIS_PREV_HASH`` (no history replay for rounds 1-3).

Null hypotheses rejected:

* NH0_history_replay: enabling mid-session replays rounds 1-3 into the
   chain. Expected: chain has exactly 2 records.
* NH0_continued_prev_hash: the first post-enable record might carry a
   prev_hash claiming continuity with a phantom prior chain.
   Expected: record 1 prev_hash == GENESIS_PREV_HASH.
* NH0_no_chain_when_disabled: chain writes occur even when disabled.
   Expected: no chain.jsonl exists after rounds 1-3.
"""

from __future__ import annotations

import pathlib
import types

import numpy as np
import pytest

from antigence_subnet.validator.deterministic_scoring import (
    AuditChainWriter,
    FrozenRoundRecord,
    GENESIS_PREV_HASH,
    from_canonical_json,
    verify_chain,
)

from antigence_subnet.validator import audit_state
from antigence_subnet.validator.audit_bridge import (
    RewardToAuditAdapter,
    next_round_index,
    resume_chain_prev_hash,
)


def _make_validator(enabled: bool, *, full_path: str):
    v = types.SimpleNamespace()
    v.config = types.SimpleNamespace()
    v.config.audit = types.SimpleNamespace(enabled=enabled, chain_path="")
    v.config.neuron = types.SimpleNamespace(full_path=full_path)
    v.audit_chain_path = None
    v.step = 0
    return v


def _simulate_round(validator, uids, rewards_np):
    """Mimic the forward.py Stage 7.5 block: write to audit chain ONLY
    when ``validator.config.audit.enabled and validator.audit_chain_path``."""
    audit_enabled = bool(getattr(validator.config.audit, "enabled", False))
    chain_path = getattr(validator, "audit_chain_path", None)
    if audit_enabled and chain_path:
        writer = AuditChainWriter(chain_path)
        adapter = RewardToAuditAdapter(writer, ema_alpha=0.1)
        idx = next_round_index(writer)
        hotkeys = [f"hk-{u:04x}" * 10 for u in uids]
        adapter.record_round(
            round_index=idx,
            miner_uids=list(uids),
            rewards=[rewards_np[i] for i in range(len(uids))],
            hotkeys=hotkeys,
        )
    validator.step += 1


def test_mid_session_audit_enable(tmp_path):
    """STATEPOL-03: enable audit mid-session -> clean start, no replay."""
    neuron_dir = tmp_path / "neuron"
    v = _make_validator(enabled=False, full_path=str(neuron_dir))

    # load_audit_state on startup with disabled -> no path resolved.
    audit_state.load_audit_state(v)
    assert v.audit_chain_path is None

    chain_path = neuron_dir / "chain.jsonl"

    # --- Phase A: 3 rounds with audit DISABLED -------------------------
    uids = [0, 1, 2]
    for i in range(3):
        rewards = np.array([0.7 - i * 0.1, 0.5, 0.3 + i * 0.05], dtype=np.float32)
        _simulate_round(v, uids, rewards)

    # STATEPOL-02: no chain file created, no directory created.
    assert not chain_path.exists(), (
        "chain.jsonl created during audit-disabled rounds"
    )
    assert v.step == 3
    # Also: save_audit_state during disabled phase must not create dirs.
    audit_state.save_audit_state(v)
    assert not neuron_dir.exists()

    # --- Phase B: flip enabled=True mid-session ------------------------
    v.config.audit.enabled = True
    audit_state.save_audit_state(v)  # preflight: parent dir creation
    audit_state.load_audit_state(v)  # resolves path, verifies (empty -> GENESIS)
    assert v.audit_chain_path == str(chain_path)

    # Verify resume returns GENESIS (no phantom history to chain to).
    assert resume_chain_prev_hash(v.audit_chain_path) == GENESIS_PREV_HASH

    # --- Phase C: 2 more rounds with audit ENABLED ---------------------
    for i in range(2):
        rewards = np.array([0.9 - i * 0.05, 0.4, 0.6], dtype=np.float32)
        _simulate_round(v, uids, rewards)

    # STATEPOL-03 primary claim: chain has EXACTLY 2 records.
    assert chain_path.exists()
    lines = [ln for ln in chain_path.read_bytes().splitlines() if ln.strip()]
    assert len(lines) == 2, (
        f"Expected 2 chain records post-enablement; got {len(lines)}"
    )

    # Record 0 prev_hash must be GENESIS (clean start, no replay of
    # rounds 1-3 that ran disabled).
    rec0 = from_canonical_json(lines[0], FrozenRoundRecord)
    assert rec0.prev_hash == GENESIS_PREV_HASH, (
        f"First post-enablement record prev_hash != GENESIS; got {rec0.prev_hash!r}"
    )
    assert rec0.round_index == 0
    assert len(rec0.scores) == 3

    # Record 1 chains to record 0 (no skip/gap).
    rec1 = from_canonical_json(lines[1], FrozenRoundRecord)
    assert rec1.round_index == 1
    # prev_hash of rec1 == hash-of-rec0 (verified by verify_chain below).

    # Full chain walk must pass cleanly -- proves hash linkage is correct
    # and no tamper.
    verify_chain(chain_path)

    # v.step should be 5 (3 disabled + 2 enabled rounds).
    assert v.step == 5


def test_mid_session_disable_stops_writes(tmp_path):
    """Bonus STATEPOL-02: flipping enabled=True -> False mid-session
    stops chain writes from that point on."""
    neuron_dir = tmp_path / "neuron"
    v = _make_validator(enabled=True, full_path=str(neuron_dir))
    audit_state.save_audit_state(v)
    audit_state.load_audit_state(v)

    uids = [0, 1]
    _simulate_round(v, uids, np.array([0.7, 0.3], dtype=np.float32))
    _simulate_round(v, uids, np.array([0.6, 0.4], dtype=np.float32))

    chain_path = neuron_dir / "chain.jsonl"
    lines_before = [
        ln for ln in chain_path.read_bytes().splitlines() if ln.strip()
    ]
    assert len(lines_before) == 2

    # Disable mid-session.
    v.config.audit.enabled = False
    audit_state.load_audit_state(v)  # resets audit_chain_path to None
    assert v.audit_chain_path is None

    _simulate_round(v, uids, np.array([0.5, 0.5], dtype=np.float32))

    # Chain unchanged after disable (Stage 7.5 gate short-circuits).
    lines_after = [
        ln for ln in chain_path.read_bytes().splitlines() if ln.strip()
    ]
    assert lines_after == lines_before, (
        "Chain appended after audit.enabled was flipped False"
    )
