"""Tamper-detection test suite (DETSCORE-05).

Four explicit tamper scenarios against a 5-round reference chain:

    1. Mutation  -- change one record's field, leave linkage intact
    2. Reorder   -- swap two records in the JSONL
    3. Delete    -- drop a record from the middle
    4. Truncate  -- truncate the final record mid-bytes (malformed JSON)

Each must be detected by ``verify_chain`` with a specific
``ChainIntegrityError`` message substring. A control test confirms the
untampered fixture verifies cleanly (guards against false positives).
"""

from __future__ import annotations

import pathlib

import pytest

from antigence_subnet.validator.deterministic_scoring.chain import (
    AuditChainWriter,
    ChainIntegrityError,
    verify_chain,
)
from antigence_subnet.validator.deterministic_scoring.replay import (
    RoundInputs,
    replay_chain,
)
from antigence_subnet.validator.deterministic_scoring.serialization import (
    canonical_json,
    from_canonical_json,
)
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
    FrozenRoundScore,
)

# ---- Fixture ----------------------------------------------------------------


def _five_round_inputs() -> list[RoundInputs]:
    return [
        RoundInputs(
            round_index=i,
            ema_alpha=0.2,
            raw_rewards=tuple(
                (uid, 0.1 * (i + 1) + 0.01 * uid, f"hk{uid}") for uid in range(3)
            ),
        )
        for i in range(5)
    ]


@pytest.fixture
def chain_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Deterministic 5-round chain written to ``tmp_path/chain.jsonl``."""
    rounds = _five_round_inputs()
    result = replay_chain(rounds)
    path = tmp_path / "chain.jsonl"
    writer = AuditChainWriter(path)
    for rec in result.records:
        writer.append(rec)
    return path


# ---- Control: untampered fixture verifies cleanly ---------------------------


def test_untampered_chain_verifies(chain_path: pathlib.Path):
    """Sanity check: the fixture itself is a valid chain. Guards against
    false positives in the tamper tests below."""
    # Should not raise.
    verify_chain(chain_path)


# ---- 1. Mutation ------------------------------------------------------------


def test_tamper_mutation_rejected(chain_path: pathlib.Path):
    """Change one score's raw_reward inside round 3. Line 4's prev_hash
    still points at the original round 3's hash -> mismatch at round 4.
    """
    lines = chain_path.read_bytes().splitlines(keepends=True)
    original_r3 = from_canonical_json(lines[3].rstrip(b"\n"), FrozenRoundRecord)

    # Build a mutated record preserving round_index and prev_hash so the
    # attacker's mutation is maximally subtle.
    mutated_scores = tuple(
        FrozenRoundScore(
            uid=s.uid,
            raw_reward=(0.999 if i == 0 else s.raw_reward),
            ema_score=s.ema_score,
            hotkey=s.hotkey,
        )
        for i, s in enumerate(original_r3.scores)
    )
    mutated_r3 = FrozenRoundRecord(
        round_index=original_r3.round_index,
        ema_alpha=original_r3.ema_alpha,
        scores=mutated_scores,
        prev_hash=original_r3.prev_hash,
    )
    lines[3] = canonical_json(mutated_r3) + b"\n"
    chain_path.write_bytes(b"".join(lines))

    with pytest.raises(ChainIntegrityError, match="round 4: prev_hash mismatch"):
        verify_chain(chain_path)


# ---- 2. Reorder -------------------------------------------------------------


def test_tamper_reorder_rejected(chain_path: pathlib.Path):
    """Swap lines 2 and 3 (round_index becomes 0, 1, 3, 2, 4). Accept
    either 'non-contiguous round_index' or 'prev_hash mismatch' in the
    error message, whichever the walk hits first."""
    lines = chain_path.read_bytes().splitlines(keepends=True)
    lines[2], lines[3] = lines[3], lines[2]
    chain_path.write_bytes(b"".join(lines))

    with pytest.raises(ChainIntegrityError) as exc_info:
        verify_chain(chain_path)
    msg = str(exc_info.value)
    assert ("non-contiguous round_index" in msg) or ("prev_hash mismatch" in msg), (
        f"expected non-contiguous or prev_hash mismatch, got: {msg}"
    )


# ---- 3. Delete --------------------------------------------------------------


def test_tamper_delete_rejected(chain_path: pathlib.Path):
    """Drop round_index=2 entirely. Remaining rounds: 0, 1, 3, 4.
    verify_chain should flag non-contiguous round_index at position 2
    (expected 2, got 3)."""
    lines = chain_path.read_bytes().splitlines(keepends=True)
    # Round index 2 corresponds to lines[2] in zero-indexed order.
    del lines[2]
    chain_path.write_bytes(b"".join(lines))

    with pytest.raises(
        ChainIntegrityError, match="non-contiguous round_index"
    ) as exc_info:
        verify_chain(chain_path)
    # Confirm the expected/got pair is the 2->3 transition.
    assert "expected=2" in str(exc_info.value)
    assert "got=3" in str(exc_info.value)


# ---- 4. Truncate ------------------------------------------------------------


def test_tamper_truncate_rejected(chain_path: pathlib.Path):
    """Truncate the final line's JSON to ~60% of its length, making it
    malformed. verify_chain wraps the decode error as 'malformed record
    at line 5'."""
    lines = chain_path.read_bytes().splitlines(keepends=True)
    assert len(lines) == 5
    last_line = lines[4]
    # Keep 60% of the bytes (not counting the trailing newline). 60% is
    # enough to break JSON structure but small enough that an unterminated
    # string / missing brace is guaranteed.
    body = last_line.rstrip(b"\n")
    truncated = body[: int(len(body) * 0.6)] + b"\n"
    lines[4] = truncated
    chain_path.write_bytes(b"".join(lines))

    with pytest.raises(
        ChainIntegrityError, match="malformed record at line 5"
    ):
        verify_chain(chain_path)
