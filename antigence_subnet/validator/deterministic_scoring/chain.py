"""SHA-256 linked audit chain writer and verifier (DETSCORE-03).

The audit chain is a JSONL file: one canonical-JSON record per newline-
terminated line. Each record's ``prev_hash`` must equal the SHA-256 hex
digest of the prior record's canonical bytes. The genesis round uses
``prev_hash = "0" * 64``.

Writer-side enforcement
-----------------------
:class:`AuditChainWriter.append` validates both ``prev_hash`` and
``round_index`` at write time. This means tampering can only happen
*after* the file is written -- which :func:`verify_chain` will then
detect.

Decode-error handling
---------------------
:func:`verify_chain` wraps :class:`json.JSONDecodeError` and any
:class:`ValueError` from :func:`from_canonical_json` into
:class:`ChainIntegrityError`, so truncated or malformed records surface
through the same exception type as hash mismatches (DETSCORE-05 tamper
tests rely on this).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

from antigence_subnet.validator.deterministic_scoring.serialization import (
    canonical_json,
    from_canonical_json,
)
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
)

GENESIS_PREV_HASH = "0" * 64


class ChainIntegrityError(Exception):
    """Raised when the audit chain's hash linkage or format is broken."""


def hash_record(record: FrozenRoundRecord) -> str:
    """Return the SHA-256 hex digest of ``canonical_json(record)``.

    64 lowercase hex characters. Deterministic: equal records produce
    identical digests.
    """
    return hashlib.sha256(canonical_json(record)).hexdigest()


class AuditChainWriter:
    """Append-only writer for the audit chain JSONL file.

    The writer enforces chain linkage: ``append`` rejects records whose
    ``prev_hash`` does not match the current tip, and whose ``round_index``
    is not contiguous with the last appended record.
    """

    def __init__(self, path: pathlib.Path | str) -> None:
        self.path = pathlib.Path(path)
        # Ensure parent directory exists; idempotent.
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all_records(self) -> list[FrozenRoundRecord]:
        if not self.path.exists():
            return []
        records: list[FrozenRoundRecord] = []
        raw = self.path.read_bytes()
        if not raw:
            return []
        for i, line in enumerate(raw.splitlines()):
            if not line.strip():
                continue
            try:
                records.append(from_canonical_json(line, FrozenRoundRecord))
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                raise ChainIntegrityError(
                    f"malformed record at line {i + 1}: {e}"
                ) from e
        return records

    def latest_hash(self) -> str:
        """Return the hash of the last appended record, or genesis if empty."""
        records = self._read_all_records()
        if not records:
            return GENESIS_PREV_HASH
        return hash_record(records[-1])

    def _last_round_index(self) -> int | None:
        records = self._read_all_records()
        if not records:
            return None
        return records[-1].round_index

    def append(self, record: FrozenRoundRecord) -> str:
        """Append ``record`` to the chain, returning its hash.

        Raises:
            ChainIntegrityError: if ``record.prev_hash`` does not equal the
                current tip hash, or if ``record.round_index`` is not
                contiguous with the last appended round.
        """
        current_tip = self.latest_hash()
        if record.prev_hash != current_tip:
            raise ChainIntegrityError(
                f"prev_hash mismatch on append: expected={current_tip}, "
                f"got={record.prev_hash}"
            )
        last_idx = self._last_round_index()
        expected_round = 0 if last_idx is None else last_idx + 1
        if record.round_index != expected_round:
            raise ChainIntegrityError(
                f"non-contiguous round_index on append: expected="
                f"{expected_round}, got={record.round_index}"
            )
        blob = canonical_json(record) + b"\n"
        with open(self.path, "ab") as f:
            f.write(blob)
        return hash_record(record)


def verify_chain(path: pathlib.Path | str) -> None:
    """Verify the JSONL audit chain at ``path``.

    Walks every record, re-derives each hash, and asserts ``prev_hash``
    linkage plus contiguous ``round_index`` starting at 0.

    Raises:
        ChainIntegrityError: on any mismatch (malformed line, non-contiguous
            index, prev_hash mismatch, or missing file).
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise ChainIntegrityError(f"chain file does not exist: {p}")
    raw = p.read_bytes()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        # An empty chain is vacuously valid -- nothing to verify.
        return
    prev_hash_expected = GENESIS_PREV_HASH
    expected_round = 0
    for i, line in enumerate(lines):
        try:
            record = from_canonical_json(line, FrozenRoundRecord)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            raise ChainIntegrityError(
                f"malformed record at line {i + 1}: {e}"
            ) from e
        if record.round_index != expected_round:
            raise ChainIntegrityError(
                f"round {i}: non-contiguous round_index (expected="
                f"{expected_round}, got={record.round_index})"
            )
        if record.prev_hash != prev_hash_expected:
            raise ChainIntegrityError(
                f"round {i}: prev_hash mismatch (expected="
                f"{prev_hash_expected}, got={record.prev_hash})"
            )
        prev_hash_expected = hash_record(record)
        expected_round = record.round_index + 1
