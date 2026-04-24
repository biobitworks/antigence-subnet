"""Tests for Phase 1002 sibling syndrome audit chain (SYNDROME-04)."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from antigence_subnet.validator.deterministic_scoring.chain import (
    AuditChainWriter,
    ChainIntegrityError,
    GENESIS_PREV_HASH,
    verify_chain,
)
from antigence_subnet.validator.deterministic_scoring.serialization import (
    canonical_json,
)
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
    FrozenRoundScore,
)
from antigence_subnet.validator.deterministic_scoring.syndrome import (
    CODEWORD_DIM,
    SYNDROME_SCHEMA_VERSION,
    Codeword,
    SyndromeChainWriter,
    SyndromeRecord,
    append_syndrome_for_codeword,
    codeword_digest,
    hash_syndrome_record,
    syndrome,
    verify_syndrome_chain,
)

from tests.validator.deterministic_scoring.fixtures.codewords import (
    ALL_FIXTURES,
    CW_SELF,
)


# ---- SyndromeRecord construction & immutability ----------------------------


def _make_valid_record(round_index: int, prev_hash: str) -> SyndromeRecord:
    sv = syndrome(CW_SELF)
    return SyndromeRecord(
        record_type="syndrome",
        schema_version=SYNDROME_SCHEMA_VERSION,
        round_index=round_index,
        prev_hash=prev_hash,
        codeword_digest=codeword_digest(CW_SELF),
        syndrome_digest=sv.digest,
        bucket_signature=sv.bucket_signature,
        anomaly_class="self",
        domain="generic",
    )


def test_syndrome_record_constructs_ok():
    rec = _make_valid_record(0, GENESIS_PREV_HASH)
    assert rec.record_type == "syndrome"
    assert rec.round_index == 0
    assert rec.anomaly_class == "self"


def test_syndrome_record_rejects_wrong_record_type():
    sv = syndrome(CW_SELF)
    with pytest.raises(ValueError, match="record_type"):
        SyndromeRecord(
            record_type="not_syndrome",
            schema_version=1,
            round_index=0,
            prev_hash=GENESIS_PREV_HASH,
            codeword_digest=codeword_digest(CW_SELF),
            syndrome_digest=sv.digest,
            bucket_signature=sv.bucket_signature,
            anomaly_class="self",
            domain="generic",
        )


def test_syndrome_record_rejects_mutation():
    rec = _make_valid_record(0, GENESIS_PREV_HASH)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.anomaly_class = "tampered"  # type: ignore[misc]


def test_syndrome_record_rejects_bad_prev_hash():
    sv = syndrome(CW_SELF)
    with pytest.raises(ValueError):
        SyndromeRecord(
            record_type="syndrome",
            schema_version=1,
            round_index=0,
            prev_hash="short",
            codeword_digest=codeword_digest(CW_SELF),
            syndrome_digest=sv.digest,
            bucket_signature=sv.bucket_signature,
            anomaly_class="self",
            domain="generic",
        )


def test_syndrome_record_rejects_bad_codeword_digest():
    sv = syndrome(CW_SELF)
    with pytest.raises(ValueError):
        SyndromeRecord(
            record_type="syndrome",
            schema_version=1,
            round_index=0,
            prev_hash=GENESIS_PREV_HASH,
            codeword_digest="not_hex_at_all_" * 4,  # 60 chars, not hex
            syndrome_digest=sv.digest,
            bucket_signature=sv.bucket_signature,
            anomaly_class="self",
            domain="generic",
        )


def test_syndrome_record_rejects_negative_round_index():
    sv = syndrome(CW_SELF)
    with pytest.raises(ValueError):
        SyndromeRecord(
            record_type="syndrome",
            schema_version=1,
            round_index=-1,
            prev_hash=GENESIS_PREV_HASH,
            codeword_digest=codeword_digest(CW_SELF),
            syndrome_digest=sv.digest,
            bucket_signature=sv.bucket_signature,
            anomaly_class="self",
            domain="generic",
        )


def test_syndrome_record_rejects_bad_bucket_signature_length():
    sv = syndrome(CW_SELF)
    with pytest.raises(ValueError):
        SyndromeRecord(
            record_type="syndrome",
            schema_version=1,
            round_index=0,
            prev_hash=GENESIS_PREV_HASH,
            codeword_digest=codeword_digest(CW_SELF),
            syndrome_digest=sv.digest,
            bucket_signature=(0, 0, 0),
            anomaly_class="self",
            domain="generic",
        )


def test_syndrome_record_rejects_bucket_out_of_range():
    sv = syndrome(CW_SELF)
    with pytest.raises(ValueError):
        SyndromeRecord(
            record_type="syndrome",
            schema_version=1,
            round_index=0,
            prev_hash=GENESIS_PREV_HASH,
            codeword_digest=codeword_digest(CW_SELF),
            syndrome_digest=sv.digest,
            bucket_signature=(2,) * CODEWORD_DIM,
            anomaly_class="self",
            domain="generic",
        )


# ---- hash_syndrome_record --------------------------------------------------


def test_hash_syndrome_record_is_deterministic():
    rec = _make_valid_record(0, GENESIS_PREV_HASH)
    assert hash_syndrome_record(rec) == hash_syndrome_record(rec)


def test_hash_syndrome_record_is_64_lower_hex():
    rec = _make_valid_record(0, GENESIS_PREV_HASH)
    h = hash_syndrome_record(rec)
    assert len(h) == 64
    assert h == h.lower()


# ---- SyndromeChainWriter: empty / append / linkage -------------------------


def test_empty_syndrome_chain_genesis_hash(tmp_path: pathlib.Path):
    writer = SyndromeChainWriter(tmp_path / "chain.syndromes.jsonl")
    assert writer.latest_hash() == GENESIS_PREV_HASH


def test_missing_file_genesis_hash(tmp_path: pathlib.Path):
    writer = SyndromeChainWriter(tmp_path / "nonexistent" / "chain.syndromes.jsonl")
    assert writer.latest_hash() == GENESIS_PREV_HASH


def test_append_rejects_wrong_prev_hash(tmp_path: pathlib.Path):
    writer = SyndromeChainWriter(tmp_path / "chain.syndromes.jsonl")
    bad = _make_valid_record(0, "f" * 64)
    with pytest.raises(ChainIntegrityError, match="prev_hash"):
        writer.append(bad)


def test_append_rejects_non_contiguous_round_index(tmp_path: pathlib.Path):
    writer = SyndromeChainWriter(tmp_path / "chain.syndromes.jsonl")
    rec = _make_valid_record(5, GENESIS_PREV_HASH)
    with pytest.raises(ChainIntegrityError, match="round_index"):
        writer.append(rec)


def test_single_append_increments_latest_hash(tmp_path: pathlib.Path):
    writer = SyndromeChainWriter(tmp_path / "chain.syndromes.jsonl")
    h = append_syndrome_for_codeword(writer, 0, CW_SELF)
    assert h != GENESIS_PREV_HASH
    assert writer.latest_hash() == h


def _build_ten_round_chain(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "chain.syndromes.jsonl"
    writer = SyndromeChainWriter(path)
    for i in range(10):
        cw = ALL_FIXTURES[i % len(ALL_FIXTURES)]
        append_syndrome_for_codeword(writer, i, cw)
    return path


def test_ten_round_syndrome_chain_verifies(tmp_path: pathlib.Path):
    path = _build_ten_round_chain(tmp_path)
    verify_syndrome_chain(path)  # must not raise


# ---- Tamper detection ------------------------------------------------------


def test_verify_syndrome_chain_detects_mutation(tmp_path: pathlib.Path):
    path = _build_ten_round_chain(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    # Flip the domain on round 4 (5th line). Every record has "generic".
    tampered = lines[4].replace(b'"generic"', b'"tampered"')
    assert tampered != lines[4]
    lines[4] = tampered
    path.write_bytes(b"".join(lines))
    with pytest.raises(ChainIntegrityError):
        verify_syndrome_chain(path)


def test_verify_syndrome_chain_detects_deletion(tmp_path: pathlib.Path):
    path = _build_ten_round_chain(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    del lines[3]
    path.write_bytes(b"".join(lines))
    with pytest.raises(ChainIntegrityError):
        verify_syndrome_chain(path)


def test_verify_syndrome_chain_detects_swap(tmp_path: pathlib.Path):
    path = _build_ten_round_chain(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    lines[2], lines[5] = lines[5], lines[2]
    path.write_bytes(b"".join(lines))
    with pytest.raises(ChainIntegrityError):
        verify_syndrome_chain(path)


def test_verify_syndrome_chain_missing_file(tmp_path: pathlib.Path):
    with pytest.raises(ChainIntegrityError, match="does not exist"):
        verify_syndrome_chain(tmp_path / "nope.jsonl")


def test_verify_syndrome_chain_detects_malformed_line(tmp_path: pathlib.Path):
    path = _build_ten_round_chain(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    lines[1] = b"not valid json\n"
    path.write_bytes(b"".join(lines))
    with pytest.raises(ChainIntegrityError, match="malformed"):
        verify_syndrome_chain(path)


# ---- JSONL format invariants -----------------------------------------------


def test_jsonl_format_one_record_per_line(tmp_path: pathlib.Path):
    path = _build_ten_round_chain(tmp_path)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    assert len(lines) == 10
    for line in lines:
        assert line.endswith(b"\n")
        stripped = line.rstrip(b"\n")
        # Each line is a canonical JSON object (starts with { and ends with }).
        assert stripped.startswith(b"{")
        assert stripped.endswith(b"}")
        # Round-trippable as JSON.
        parsed = json.loads(stripped.decode("utf-8"))
        assert parsed["record_type"] == "syndrome"


def test_jsonl_lines_are_canonical_json(tmp_path: pathlib.Path):
    path = _build_ten_round_chain(tmp_path)
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line.decode("utf-8"))
        assert canonical_json(parsed) == line


# ---- Cross-chain correspondence (SYNDROME-04 core) -------------------------


def _make_score(uid: int, reward: float, ema: float) -> FrozenRoundScore:
    return FrozenRoundScore(
        uid=uid, raw_reward=reward, ema_score=ema, hotkey=f"hk{uid}"
    )


def test_audit_and_syndrome_chains_share_round_index(tmp_path: pathlib.Path):
    """Build a 3-round audit chain and a 3-round syndrome chain in lockstep.

    Both chains verify; their round_index sets are equal.
    """
    audit_path = tmp_path / "chain.jsonl"
    syn_path = tmp_path / "chain.syndromes.jsonl"
    audit_writer = AuditChainWriter(audit_path)
    syn_writer = SyndromeChainWriter(syn_path)

    audit_prev = GENESIS_PREV_HASH
    for r in range(3):
        # Audit side.
        rec = FrozenRoundRecord(
            round_index=r,
            ema_alpha=0.1,
            scores=(_make_score(0, 0.5, 0.5), _make_score(1, 0.6, 0.6)),
            prev_hash=audit_prev,
        )
        audit_prev = audit_writer.append(rec)
        # Syndrome side (shared round_index).
        append_syndrome_for_codeword(syn_writer, r, ALL_FIXTURES[r])

    verify_chain(audit_path)
    verify_syndrome_chain(syn_path)

    # Round-index sets match.
    audit_rounds = {
        json.loads(ln.decode("utf-8"))["round_index"]
        for ln in audit_path.read_bytes().splitlines()
        if ln.strip()
    }
    syn_rounds = {
        json.loads(ln.decode("utf-8"))["round_index"]
        for ln in syn_path.read_bytes().splitlines()
        if ln.strip()
    }
    assert audit_rounds == syn_rounds == {0, 1, 2}


def test_append_syndrome_for_codeword_end_to_end(tmp_path: pathlib.Path):
    path = tmp_path / "chain.syndromes.jsonl"
    writer = SyndromeChainWriter(path)
    h = append_syndrome_for_codeword(writer, 0, CW_SELF)
    verify_syndrome_chain(path)
    # Record was written with the correct anomaly_class.
    line = path.read_bytes().splitlines()[0]
    parsed = json.loads(line.decode("utf-8"))
    assert parsed["anomaly_class"] == "self"
    assert parsed["round_index"] == 0
    assert parsed["domain"] == "generic"
    assert h == hash_syndrome_record(
        SyndromeRecord(
            record_type="syndrome",
            schema_version=1,
            round_index=0,
            prev_hash=GENESIS_PREV_HASH,
            codeword_digest=codeword_digest(CW_SELF),
            syndrome_digest=syndrome(CW_SELF).digest,
            bucket_signature=syndrome(CW_SELF).bucket_signature,
            anomaly_class="self",
            domain="generic",
        )
    )
