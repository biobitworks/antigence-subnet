"""Phase 1100 audit-bridge tests.

Null-hypothesis evidence (per-test intent in docstring):

* test_bridge_converts_numpy_float32_to_python_float
    NH0: the bridge might silently let numpy scalars through, producing
    non-portable canonical JSON. This test proves the boundary enforces
    Python float via explicit ``type(...) is float`` checks post-cast.

* test_bridge_rejects_nan_inf
    NH0: NaN / +-Inf could slip into the chain and corrupt every
    downstream verifier. This test proves ValueError fires BEFORE
    FrozenRoundScore is constructed.

* test_bridge_writes_one_record_per_miner
    NH0: a round might lose or duplicate miners. This test proves one
    FrozenRoundRecord per round, containing exactly N FrozenRoundScore
    entries (one per UID), and verify_chain stays clean.

* test_chain_continuity_across_restart
    NH0: reopening the chain from a second process might drift the
    prev_hash or skip a round_index. This test writes 3 rounds, reopens,
    appends 2 more, and confirms verify_chain still passes with 5
    contiguous records.

* test_bridge_tamper_detection_via_chain
    NH0: tamper on the JSONL could go undetected. This test flips a
    byte in the middle of the chain file and asserts verify_chain
    raises ChainIntegrityError.

* test_production_code_unchanged
    NH0: the experiment might silently edit production. This test runs
    git diff v13.0..HEAD for the four protected files and asserts the
    diff is empty.

A final smoke test exercises bridge_get_rewards end-to-end against the
production_copy reward path (with a bittensor stub already active).
"""

from __future__ import annotations

import pathlib
import subprocess

import numpy as np
import pytest

# audit_bridge is at antigence_subnet/validator/audit_bridge.py (promoted to
# production in Phase 1103 v13.1.1).
from antigence_subnet.validator.audit_bridge import (  # noqa: E402
    RewardToAuditAdapter,
    bridge_get_rewards,
    next_round_index,
    resume_chain_prev_hash,
)
from antigence_subnet.validator.deterministic_scoring import (
    GENESIS_PREV_HASH,
    AuditChainWriter,
    ChainIntegrityError,
    FrozenRoundRecord,
    from_canonical_json,
    verify_chain,
)


# --------------------------------------------------------------------- #
# helpers                                                                #
# --------------------------------------------------------------------- #
def _chain_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "chain.jsonl"


def _hotkeys(uids):
    return [f"5F{''.join(['%02x' % ((u * 7 + 13) & 0xff)] * 23)}" for u in uids]


# --------------------------------------------------------------------- #
# Test 1 -- numpy float32 -> Python float at the boundary                #
# --------------------------------------------------------------------- #
def test_bridge_converts_numpy_float32_to_python_float():
    """NH0: bridge silently passes numpy scalars. Expected: explicit cast."""
    coerce = RewardToAuditAdapter._coerce_float

    # numpy.float32 -> Python float
    result = coerce(np.float32(0.7), field="rewards[0]")
    assert type(result) is float
    assert result == pytest.approx(0.7, rel=1e-6)

    # numpy.float64 -> Python float
    result64 = coerce(np.float64(0.5), field="rewards[0]")
    assert type(result64) is float
    assert result64 == 0.5

    # plain Python float passes through identity
    p = 0.3
    assert coerce(p, field="rewards[0]") is p

    # numpy.int32 is REJECTED (explicit cast required upstream)
    with pytest.raises(TypeError, match="numpy integer"):
        coerce(np.int32(1), field="rewards[0]")

    # bool is REJECTED (subclass-of-int trap)
    with pytest.raises(TypeError, match="bool"):
        coerce(True, field="rewards[0]")

    # Plain Python int is REJECTED (forces explicit conversion upstream)
    with pytest.raises(TypeError):
        coerce(1, field="rewards[0]")


# --------------------------------------------------------------------- #
# Test 2 -- NaN / Inf rejected before FrozenRoundScore                   #
# --------------------------------------------------------------------- #
def test_bridge_rejects_nan_inf():
    """NH0: NaN/Inf slip through. Expected: ValueError at boundary."""
    coerce = RewardToAuditAdapter._coerce_float

    with pytest.raises(ValueError, match="NaN"):
        coerce(np.float32("nan"), field="rewards[0]")
    with pytest.raises(ValueError, match="NaN"):
        coerce(float("nan"), field="rewards[0]")
    with pytest.raises(ValueError, match=r"\+Inf"):
        coerce(np.float64("inf"), field="rewards[0]")
    with pytest.raises(ValueError, match="-Inf"):
        coerce(-float("inf"), field="rewards[0]")
    with pytest.raises(ValueError, match=r"\+Inf"):
        coerce(np.float32(np.inf), field="rewards[0]")


# --------------------------------------------------------------------- #
# Test 3 -- one record per round, N scores per record                    #
# --------------------------------------------------------------------- #
def test_bridge_writes_one_record_per_miner(tmp_path):
    """NH0: rounds lose/duplicate miners. Expected: 1 record, 5 scores."""
    path = _chain_path(tmp_path)
    writer = AuditChainWriter(path)
    adapter = RewardToAuditAdapter(writer, ema_alpha=0.1)

    uids = [0, 1, 2, 3, 4]
    rewards = np.array([0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
    hotkeys = _hotkeys(uids)

    h = adapter.record_round(
        round_index=0,
        miner_uids=uids,
        rewards=[rewards[i] for i in range(5)],
        hotkeys=hotkeys,
    )
    assert len(h) == 64  # sha256 hex

    # verify_chain clean
    verify_chain(path)

    # Exactly one line (one record), containing 5 scores.
    lines = [ln for ln in path.read_bytes().splitlines() if ln.strip()]
    assert len(lines) == 1
    record = from_canonical_json(lines[0], FrozenRoundRecord)
    assert record.round_index == 0
    assert len(record.scores) == 5
    assert [s.uid for s in record.scores] == uids
    assert all(type(s.raw_reward) is float for s in record.scores)
    assert [s.hotkey for s in record.scores] == hotkeys
    assert record.prev_hash == GENESIS_PREV_HASH


# --------------------------------------------------------------------- #
# Test 4 -- restart continuity                                           #
# --------------------------------------------------------------------- #
def test_chain_continuity_across_restart(tmp_path):
    """NH0: reopen breaks prev_hash / round_index. Expected: clean append."""
    path = _chain_path(tmp_path)

    # Session A: 3 rounds via adapter.
    writer_a = AuditChainWriter(path)
    adapter_a = RewardToAuditAdapter(writer_a, ema_alpha=0.1)
    uids = [10, 11, 12]
    hk = _hotkeys(uids)
    for r in range(3):
        rewards = np.array(
            [0.1 * (r + 1), 0.2 * (r + 1), 0.3 * (r + 1)], dtype=np.float32
        )
        adapter_a.record_round(
            round_index=r,
            miner_uids=uids,
            rewards=[rewards[i] for i in range(3)],
            hotkeys=hk,
        )
    del writer_a, adapter_a  # "close" session A

    # Session B: resume. prev_hash MUST match latest_hash.
    prev = resume_chain_prev_hash(path)
    assert prev != GENESIS_PREV_HASH  # chain non-empty
    assert len(prev) == 64

    writer_b = AuditChainWriter(path)
    assert writer_b.latest_hash() == prev
    assert next_round_index(writer_b) == 3

    adapter_b = RewardToAuditAdapter(writer_b, ema_alpha=0.1)
    for r in (3, 4):
        rewards = np.array(
            [0.1 * (r + 1), 0.2 * (r + 1), 0.3 * (r + 1)], dtype=np.float32
        )
        adapter_b.record_round(
            round_index=r,
            miner_uids=uids,
            rewards=[rewards[i] for i in range(3)],
            hotkeys=hk,
        )

    # Clean 5-record chain, round_index contiguous.
    verify_chain(path)
    lines = [ln for ln in path.read_bytes().splitlines() if ln.strip()]
    assert len(lines) == 5
    indices = [
        from_canonical_json(ln, FrozenRoundRecord).round_index for ln in lines
    ]
    assert indices == [0, 1, 2, 3, 4]

    # resume_chain_prev_hash on missing file -> genesis.
    missing = tmp_path / "does_not_exist.jsonl"
    assert resume_chain_prev_hash(missing) == GENESIS_PREV_HASH

    # resume on empty file -> genesis.
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert resume_chain_prev_hash(empty) == GENESIS_PREV_HASH


# --------------------------------------------------------------------- #
# Test 5 -- tamper detection                                             #
# --------------------------------------------------------------------- #
def test_bridge_tamper_detection_via_chain(tmp_path):
    """NH0: tamper undetected. Expected: verify_chain raises."""
    path = _chain_path(tmp_path)
    writer = AuditChainWriter(path)
    adapter = RewardToAuditAdapter(writer, ema_alpha=0.1)
    uids = [0, 1, 2]
    hk = _hotkeys(uids)
    for r in range(3):
        rewards = np.array([0.5, 0.4, 0.3], dtype=np.float32)
        adapter.record_round(
            round_index=r,
            miner_uids=uids,
            rewards=[rewards[i] for i in range(3)],
            hotkeys=hk,
        )
    verify_chain(path)  # sanity: clean before tamper

    # Mutate a digit inside record 1 (the second JSONL line). Specifically
    # we swap a single character in the hotkey string so the JSON remains
    # parseable but the record's content hash changes.
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    target = lines[1]
    # Flip one ASCII char inside the line; ensure it lands in the JSON body
    # (avoid the trailing newline).
    body = target.rstrip(b"\n")
    # pick a stable offset into the body
    idx = len(body) // 2
    flipped_char = bytes([body[idx] ^ 1])
    mutated = body[:idx] + flipped_char + body[idx + 1:]
    lines[1] = mutated + b"\n"
    path.write_bytes(b"".join(lines))

    with pytest.raises(ChainIntegrityError):
        verify_chain(path)

    # resume_chain_prev_hash also surfaces the tamper (does NOT swallow).
    with pytest.raises(ChainIntegrityError):
        resume_chain_prev_hash(path)


# --------------------------------------------------------------------- #
# Test 6 -- production code unchanged                                    #
# --------------------------------------------------------------------- #
def test_production_code_unchanged():
    """NH0: the promotion drifts reward.py. Expected: reward.py stays byte-identical.

    v13.1.1 (Phase 1103) promotes audit-chain integration into
    forward.py, base/validator.py, and validate_config.py -- those three
    files ARE expected to differ from v13.0 after the migration commit.
    The reward.py path MUST remain untouched (audit_bridge wraps it from
    outside), which is the surviving half of this invariant.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    protected = [
        "antigence_subnet/validator/reward.py",
    ]
    proc = subprocess.run(
        ["git", "diff", "v13.0..HEAD", "--"] + protected,
        capture_output=True,
        cwd=str(repo_root),
        check=True,
    )
    assert proc.stdout == b"", (
        f"reward.py drifted from v13.0 (audit_bridge must wrap, not modify):\n"
        f"{proc.stdout.decode()}"
    )

    # Also: the deterministic_scoring package remains bittensor-free.
    ds_dir = pathlib.Path(repo_root) / "antigence_subnet" / "validator" / "deterministic_scoring"
    for py in ds_dir.rglob("*.py"):
        src = py.read_text()
        for line in src.splitlines():
            stripped = line.lstrip()
            assert not stripped.startswith("import bittensor"), (
                f"bittensor import leaked into {py}"
            )
            assert not stripped.startswith("from bittensor"), (
                f"bittensor from-import leaked into {py}"
            )


# --------------------------------------------------------------------- #
# Smoke: bridge_get_rewards end-to-end over production_copy              #
# --------------------------------------------------------------------- #
class _MockResp:
    def __init__(self, score: float) -> None:
        self.anomaly_score = score
        self.anomaly_type = None


class _MockValidator:
    # Minimal surface -- reward.py only uses it for bt.logging context.
    pass


def test_bridge_get_rewards_end_to_end(tmp_path):
    """Smoke: full wrapper path returns numpy array + writes one record."""
    path = _chain_path(tmp_path)
    writer = AuditChainWriter(path)

    uids = [0, 1, 2]
    responses_by_sample = {
        "s1": [_MockResp(0.9), _MockResp(0.1), _MockResp(0.8)],
        "s2": [_MockResp(0.2), _MockResp(0.2), _MockResp(0.7)],
    }
    manifest = {
        "s1": {"ground_truth_label": "anomalous", "is_honeypot": False},
        "s2": {"ground_truth_label": "normal", "is_honeypot": False},
    }
    hotkeys_by_uid = {0: "hk-0-aaa", 1: "hk-1-bbb", 2: "hk-2-ccc"}

    validator = _MockValidator()
    rewards = bridge_get_rewards(
        validator,
        uids,
        responses_by_sample,
        manifest,
        writer,
        hotkeys_by_uid=hotkeys_by_uid,
        round_index=0,
    )
    # Drop-in: numpy.ndarray of length 3.
    assert hasattr(rewards, "shape")
    assert rewards.shape == (3,)

    # One record on chain, containing 3 scores.
    verify_chain(path)
    lines = [ln for ln in path.read_bytes().splitlines() if ln.strip()]
    assert len(lines) == 1
    record = from_canonical_json(lines[0], FrozenRoundRecord)
    assert len(record.scores) == 3
    assert [s.uid for s in record.scores] == uids
    assert [s.hotkey for s in record.scores] == [
        hotkeys_by_uid[u] for u in uids
    ]
    assert all(type(s.raw_reward) is float for s in record.scores)
    # rewards are in [0,1] as the precision-first formula produces.
    for s in record.scores:
        assert 0.0 <= s.raw_reward <= 1.0


# --------------------------------------------------------------------- #
# Extra: adapter length / type guards                                    #
# --------------------------------------------------------------------- #
def test_adapter_rejects_mismatched_lengths(tmp_path):
    writer = AuditChainWriter(_chain_path(tmp_path))
    adapter = RewardToAuditAdapter(writer, ema_alpha=0.1)
    with pytest.raises(ValueError, match="rewards length"):
        adapter.record_round(
            round_index=0,
            miner_uids=[0, 1],
            rewards=[0.1],
            hotkeys=["a", "b"],
        )
    with pytest.raises(ValueError, match="hotkeys length"):
        adapter.record_round(
            round_index=0,
            miner_uids=[0, 1],
            rewards=[0.1, 0.2],
            hotkeys=["a"],
        )


def test_adapter_rejects_bad_ema_alpha(tmp_path):
    writer = AuditChainWriter(_chain_path(tmp_path))
    with pytest.raises(ValueError):
        RewardToAuditAdapter(writer, ema_alpha=0.0)
    with pytest.raises(ValueError):
        RewardToAuditAdapter(writer, ema_alpha=1.0)
    with pytest.raises(TypeError):
        RewardToAuditAdapter(writer, ema_alpha=1)  # plain int


def test_resume_rejects_broken_chain(tmp_path):
    """resume_chain_prev_hash surfaces tampers (does NOT swallow)."""
    path = _chain_path(tmp_path)
    # Write a garbage line.
    path.write_bytes(b'{"not":"a record"}\n')
    with pytest.raises(ChainIntegrityError):
        resume_chain_prev_hash(path)
