"""Tests for SHA-256 audit chain writer and verifier (DETSCORE-03)."""

from __future__ import annotations

import pathlib

import pytest

from antigence_subnet.validator.deterministic_scoring.chain import (
    AuditChainWriter,
    ChainIntegrityError,
    GENESIS_PREV_HASH,
    hash_record,
    verify_chain,
)
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
    FrozenRoundScore,
)


def _make_score(uid: int, reward: float, ema: float) -> FrozenRoundScore:
    return FrozenRoundScore(
        uid=uid, raw_reward=reward, ema_score=ema, hotkey=f"hk{uid}"
    )


def _make_record(
    round_index: int, prev_hash: str, alpha: float = 0.1
) -> FrozenRoundRecord:
    scores = (
        _make_score(0, 0.1 * (round_index + 1), 0.01 * (round_index + 1)),
        _make_score(1, 0.2 * (round_index + 1), 0.02 * (round_index + 1)),
        _make_score(2, 0.3 * (round_index + 1), 0.03 * (round_index + 1)),
    )
    return FrozenRoundRecord(
        round_index=round_index,
        ema_alpha=alpha,
        scores=scores,
        prev_hash=prev_hash,
    )


def _build_ten_round_chain(path: pathlib.Path) -> list[FrozenRoundRecord]:
    writer = AuditChainWriter(path)
    records: list[FrozenRoundRecord] = []
    prev = GENESIS_PREV_HASH
    for i in range(10):
        r = _make_record(i, prev)
        writer.append(r)
        prev = hash_record(r)
        records.append(r)
    return records


# ---- hash_record determinism ------------------------------------------------


def test_hash_record_is_deterministic():
    r = _make_record(0, GENESIS_PREV_HASH)
    assert hash_record(r) == hash_record(r)


def test_hash_record_changes_on_field_change():
    r1 = _make_record(0, GENESIS_PREV_HASH, alpha=0.1)
    r2 = _make_record(0, GENESIS_PREV_HASH, alpha=0.2)
    assert hash_record(r1) != hash_record(r2)


def test_hash_record_returns_64_lowercase_hex():
    h = hash_record(_make_record(0, GENESIS_PREV_HASH))
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---- AuditChainWriter empty state ------------------------------------------


def test_empty_chain_latest_hash_is_genesis(tmp_path):
    writer = AuditChainWriter(tmp_path / "chain.jsonl")
    assert writer.latest_hash() == GENESIS_PREV_HASH


def test_nonexistent_file_latest_hash_is_genesis(tmp_path):
    writer = AuditChainWriter(tmp_path / "subdir" / "chain.jsonl")
    assert writer.latest_hash() == GENESIS_PREV_HASH
    # subdir should have been created by __init__.
    assert (tmp_path / "subdir").exists()


# ---- AuditChainWriter.append linkage enforcement ----------------------------


def test_append_first_record_requires_genesis_prev_hash(tmp_path):
    writer = AuditChainWriter(tmp_path / "chain.jsonl")
    bad = _make_record(0, "a" * 64)  # prev_hash != genesis
    with pytest.raises(ChainIntegrityError, match="prev_hash mismatch"):
        writer.append(bad)


def test_append_rejects_wrong_prev_hash(tmp_path):
    writer = AuditChainWriter(tmp_path / "chain.jsonl")
    r0 = _make_record(0, GENESIS_PREV_HASH)
    writer.append(r0)
    # Second record with wrong prev_hash.
    r1 = _make_record(1, "b" * 64)
    with pytest.raises(ChainIntegrityError, match="prev_hash mismatch"):
        writer.append(r1)


def test_append_rejects_non_contiguous_round_index(tmp_path):
    writer = AuditChainWriter(tmp_path / "chain.jsonl")
    r0 = _make_record(0, GENESIS_PREV_HASH)
    writer.append(r0)
    # Skip round 1, try to append round 2.
    r2 = _make_record(2, hash_record(r0))
    with pytest.raises(ChainIntegrityError, match="non-contiguous round_index"):
        writer.append(r2)


def test_append_returns_hash(tmp_path):
    writer = AuditChainWriter(tmp_path / "chain.jsonl")
    r0 = _make_record(0, GENESIS_PREV_HASH)
    h = writer.append(r0)
    assert h == hash_record(r0)
    assert writer.latest_hash() == h


# ---- 10-round chain build + verify -----------------------------------------


def test_ten_round_chain_verifies(tmp_path):
    path = tmp_path / "chain.jsonl"
    _build_ten_round_chain(path)
    # verify_chain returns None on success, raises otherwise.
    verify_chain(path)


def test_jsonl_format_is_one_record_per_line(tmp_path):
    path = tmp_path / "chain.jsonl"
    _build_ten_round_chain(path)
    raw = path.read_bytes()
    # 10 records, each newline-terminated.
    assert raw.count(b"\n") == 10
    lines = raw.splitlines()
    assert len(lines) == 10
    for line in lines:
        assert line.startswith(b"{") and line.endswith(b"}")


# ---- verify_chain failure modes --------------------------------------------


def test_verify_chain_rejects_missing_file(tmp_path):
    with pytest.raises(ChainIntegrityError, match="does not exist"):
        verify_chain(tmp_path / "nope.jsonl")


def test_verify_chain_rejects_non_contiguous_round_index(tmp_path):
    path = tmp_path / "chain.jsonl"
    records = _build_ten_round_chain(path)
    # Rewrite file with round 3 removed -> rounds become 0,1,2,4,5,6,7,8,9
    # which means at position 3 we expected round 3 but got round 4.
    lines = path.read_bytes().splitlines(keepends=True)
    # Drop the line for round_index=3 (index 3 in the list).
    tampered = b"".join(lines[:3] + lines[4:])
    path.write_bytes(tampered)
    with pytest.raises(ChainIntegrityError, match="non-contiguous round_index"):
        verify_chain(path)
    # Guard against the fixture accidentally matching via prev_hash first:
    # the assertion above is deterministic because round_index is checked
    # before prev_hash in verify_chain.
    assert len(records) == 10


def test_verify_chain_rejects_prev_hash_drift(tmp_path):
    # Build a 3-round chain, then hand-overwrite line 2 with a record that
    # still has the original round_index but a different raw_reward. The
    # line's prev_hash stays pointing at round 0's hash (correct), but the
    # NEXT line's prev_hash still points at the ORIGINAL round 1's hash,
    # not the mutated one -> verify_chain should flag round 2's prev_hash.
    path = tmp_path / "chain.jsonl"
    writer = AuditChainWriter(path)
    r0 = _make_record(0, GENESIS_PREV_HASH)
    writer.append(r0)
    r1 = _make_record(1, hash_record(r0))
    writer.append(r1)
    r2 = _make_record(2, hash_record(r1))
    writer.append(r2)

    # Build a mutated r1 with a different score.
    from antigence_subnet.validator.deterministic_scoring.serialization import canonical_json
    mutated_scores = (
        _make_score(0, 9.99, 0.01),  # changed from 0.2 to 9.99
        _make_score(1, 0.4, 0.04),
        _make_score(2, 0.6, 0.06),
    )
    mutated_r1 = FrozenRoundRecord(
        round_index=1,
        ema_alpha=0.1,
        scores=mutated_scores,
        prev_hash=hash_record(r0),  # still links to round 0
    )
    lines = path.read_bytes().splitlines(keepends=True)
    lines[1] = canonical_json(mutated_r1) + b"\n"
    path.write_bytes(b"".join(lines))

    with pytest.raises(ChainIntegrityError, match="round 2: prev_hash mismatch"):
        verify_chain(path)


def test_verify_chain_rejects_malformed_line(tmp_path):
    path = tmp_path / "chain.jsonl"
    _build_ten_round_chain(path)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    # Truncate line 5 to 80% of its length, breaking the JSON.
    truncated = lines[4][: int(len(lines[4]) * 0.8)]
    lines[4] = truncated + b"\n"  # rejoin with newline to stay JSONL-ish
    path.write_bytes(b"".join(lines))
    with pytest.raises(ChainIntegrityError, match="malformed record at line 5"):
        verify_chain(path)


def test_verify_chain_empty_file_is_valid(tmp_path):
    # Writing nothing is vacuously valid; verify_chain returns without error.
    path = tmp_path / "chain.jsonl"
    path.write_bytes(b"")
    # Should not raise.
    verify_chain(path)
