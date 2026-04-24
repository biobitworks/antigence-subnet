"""Syndrome-based anomaly detection layer (Phase 1002 / SYNDROME-01..05).

This module is an ADDITIVE extension of the Phase 1000 deterministic scoring
package. It does NOT modify any Phase 1000 source file. It introduces:

    * :class:`Codeword` (SYNDROME-01) -- frozen fixed-length feature vector.
    * :func:`syndrome` (SYNDROME-02) -- pure deterministic mapping from a
      :class:`Codeword` to a :class:`SyndromeVector`.
    * :class:`SyndromeTable` + :func:`load_default_table` (SYNDROME-03) --
      read-only packaged lookup table mapping per-domain bucket signatures to
      anomaly-class strings.
    * :class:`SyndromeRecord` + :class:`SyndromeChainWriter` +
      :func:`verify_syndrome_chain` (SYNDROME-04) -- a SIBLING JSONL audit
      chain (e.g. ``chain.syndromes.jsonl``) that uses the same SHA-256 +
      prev_hash linkage discipline as Phase 1000's :class:`AuditChainWriter`.
      Reuses :class:`ChainIntegrityError` from :mod:`...chain` so callers
      handle a single exception type across both chains.

Pure stdlib: ``dataclasses``, ``hashlib``, ``importlib.resources``, ``json``,
``math``, ``pathlib``, ``types``, ``typing``. No numpy, no bittensor, no
torch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.resources
import json
import math
import pathlib
import types
from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from antigence_subnet.validator.deterministic_scoring.chain import (
    ChainIntegrityError,
    GENESIS_PREV_HASH,
)
from antigence_subnet.validator.deterministic_scoring.serialization import (
    canonical_json,
)

# ---- Module-level constants ------------------------------------------------

#: Codeword feature-vector length. Baked into the schema; any change is a
#: breaking schema bump.
CODEWORD_DIM: int = 8

#: Current syndrome record / vector schema version.
SYNDROME_SCHEMA_VERSION: int = 1

#: Ternary bucket threshold. ``|f| > _BUCKET_THRESHOLD`` is a deviation;
#: ``|f| <= _BUCKET_THRESHOLD`` is "self". Baked into the schema.
_BUCKET_THRESHOLD: float = 1.0

_HEX_ALPHABET = frozenset("0123456789abcdef")


# ---- Codeword --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Codeword:
    """Frozen fixed-length codeword vector scoped to a ``domain``.

    All features must be exact Python ``float`` (not ``int``, not numpy
    scalars -- enforced via ``type(x) is float``). NaN and +/-Inf are
    rejected at construction time, mirroring ``canonical_json``'s policy.
    """

    schema_version: int
    features: Tuple[float, ...]
    domain: str

    def __post_init__(self) -> None:  # noqa: D401
        # schema_version: exact int equal to 1.
        if type(self.schema_version) is not int or isinstance(
            self.schema_version, bool
        ):
            raise TypeError(
                f"Codeword.schema_version must be int, got "
                f"{type(self.schema_version).__name__}"
            )
        if self.schema_version != 1:
            raise ValueError(
                f"Codeword.schema_version must be 1, got {self.schema_version}"
            )
        # features: tuple of CODEWORD_DIM Python floats, all finite.
        if type(self.features) is not tuple:
            raise TypeError(
                f"Codeword.features must be tuple, got "
                f"{type(self.features).__name__}"
            )
        if len(self.features) != CODEWORD_DIM:
            raise ValueError(
                f"Codeword.features must have length CODEWORD_DIM={CODEWORD_DIM}, "
                f"got {len(self.features)}"
            )
        for i, f in enumerate(self.features):
            # type(f) is float rejects bool, int, numpy.float64 (subclass),
            # and any other float subclass.
            if type(f) is not float:
                raise TypeError(
                    f"Codeword.features[{i}] must be exact float, got "
                    f"{type(f).__name__}"
                )
            if math.isnan(f) or math.isinf(f):
                raise ValueError(
                    f"Codeword.features[{i}] is non-finite (NaN/Inf): {f!r}"
                )
        # domain: non-empty str.
        if type(self.domain) is not str:
            raise TypeError(
                f"Codeword.domain must be str, got {type(self.domain).__name__}"
            )
        if len(self.domain) < 1:
            raise ValueError("Codeword.domain must be non-empty")


def codeword_digest(codeword: Codeword) -> str:
    """Return the SHA-256 hex digest of ``canonical_json(codeword)``.

    64 lowercase hex characters. Deterministic: equal codewords produce
    identical digests.
    """
    return hashlib.sha256(canonical_json(codeword)).hexdigest()


# ---- SyndromeTable ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyndromeTable:
    """Read-only syndrome -> anomaly-class lookup table.

    ``entries`` is wrapped in :class:`types.MappingProxyType` at construction
    so runtime mutation raises ``TypeError``. The wrap happens in
    ``__post_init__`` via ``object.__setattr__`` -- this is the idiomatic
    pattern for coercing a frozen dataclass field after ``__init__`` assigns
    the raw value. It is legal because ``__post_init__`` is allowed to bypass
    the frozen-attribute guard during construction.
    """

    version: int
    # map[domain] -> map[bucket_sig_str] -> anomaly_class
    entries: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if type(self.version) is not int or isinstance(self.version, bool):
            raise TypeError(
                f"SyndromeTable.version must be int, got "
                f"{type(self.version).__name__}"
            )
        if type(self.entries) is not dict and not isinstance(
            self.entries, Mapping
        ):
            raise TypeError(
                f"SyndromeTable.entries must be mapping, got "
                f"{type(self.entries).__name__}"
            )
        # Freeze inner dicts first, then the outer mapping. We use
        # object.__setattr__ here because the dataclass is frozen -- this is
        # a pre-MappingProxyType wrap during __post_init__, which is the
        # documented way to coerce a field value on a frozen dataclass.
        inner_frozen: dict[str, Mapping[str, str]] = {}
        for domain, sub in self.entries.items():
            if type(domain) is not str:
                raise TypeError(
                    f"SyndromeTable.entries domain must be str, got "
                    f"{type(domain).__name__}"
                )
            if not isinstance(sub, Mapping):
                raise TypeError(
                    f"SyndromeTable.entries[{domain!r}] must be mapping"
                )
            inner_frozen[domain] = types.MappingProxyType(dict(sub))
        object.__setattr__(
            self, "entries", types.MappingProxyType(inner_frozen)
        )

    def lookup(
        self, domain: str, bucket_signature: Iterable[int]
    ) -> str:
        """Return the anomaly class for ``(domain, bucket_signature)``.

        Unknown domains and unknown signatures both map to ``"unclassified"``.
        Never raises on input; wrong-length tuples are treated as unknown.
        """
        sig_list = list(bucket_signature)
        key = "sig:" + ",".join(str(i) for i in sig_list)
        sub = self.entries.get(domain)
        if sub is None:
            return "unclassified"
        return sub.get(key, "unclassified")


# ---- Default-table loader (module-level cache) -----------------------------

_DEFAULT_TABLE: SyndromeTable | None = None


def load_default_table() -> SyndromeTable:
    """Return the packaged default :class:`SyndromeTable` (cached).

    Loads ``syndrome_table_v1.json`` via :mod:`importlib.resources`. Repeated
    calls return the same cached instance.
    """
    global _DEFAULT_TABLE
    if _DEFAULT_TABLE is not None:
        return _DEFAULT_TABLE
    raw = (
        importlib.resources.files(
            "antigence_subnet.validator.deterministic_scoring"
        )
        .joinpath("syndrome_table_v1.json")
        .read_bytes()
    )
    text = raw.rstrip(b"\n").decode("utf-8")
    parsed = json.loads(text)
    table = SyndromeTable(
        version=parsed["version"],
        entries=parsed["entries"],
    )
    _DEFAULT_TABLE = table
    return _DEFAULT_TABLE


# ---- SyndromeVector + pure syndrome function -------------------------------


@dataclass(frozen=True, slots=True)
class SyndromeVector:
    """Output of :func:`syndrome` -- frozen, hashable, deterministic."""

    schema_version: int
    bucket_signature: Tuple[int, ...]
    digest: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or isinstance(
            self.schema_version, bool
        ):
            raise TypeError(
                f"SyndromeVector.schema_version must be int, got "
                f"{type(self.schema_version).__name__}"
            )
        if self.schema_version != SYNDROME_SCHEMA_VERSION:
            raise ValueError(
                f"SyndromeVector.schema_version must be "
                f"{SYNDROME_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if type(self.bucket_signature) is not tuple:
            raise TypeError(
                f"SyndromeVector.bucket_signature must be tuple, got "
                f"{type(self.bucket_signature).__name__}"
            )
        if len(self.bucket_signature) != CODEWORD_DIM:
            raise ValueError(
                f"SyndromeVector.bucket_signature must have length "
                f"{CODEWORD_DIM}, got {len(self.bucket_signature)}"
            )
        for i, b in enumerate(self.bucket_signature):
            if type(b) is not int or isinstance(b, bool):
                raise TypeError(
                    f"SyndromeVector.bucket_signature[{i}] must be int, got "
                    f"{type(b).__name__}"
                )
            if b not in (-1, 0, 1):
                raise ValueError(
                    f"SyndromeVector.bucket_signature[{i}] must be in "
                    f"{{-1,0,1}}, got {b}"
                )
        if type(self.digest) is not str:
            raise TypeError(
                f"SyndromeVector.digest must be str, got "
                f"{type(self.digest).__name__}"
            )
        if len(self.digest) != 64:
            raise ValueError(
                f"SyndromeVector.digest must be 64 hex chars, got "
                f"len={len(self.digest)}"
            )
        for c in self.digest:
            if c not in _HEX_ALPHABET:
                raise ValueError(
                    f"SyndromeVector.digest must be lowercase hex, got "
                    f"invalid char {c!r}"
                )


def _bucket(f: float) -> int:
    """Ternary bucket: -1 if f < -T, +1 if f > +T, else 0. Pure comparison."""
    if f > _BUCKET_THRESHOLD:
        return 1
    if f < -_BUCKET_THRESHOLD:
        return -1
    return 0


def syndrome(codeword: Codeword) -> SyndromeVector:
    """Compute the :class:`SyndromeVector` for ``codeword``.

    Pure and deterministic: byte-identical ``canonical_json(syndrome(cw))``
    across two in-process calls and across subprocess boundaries. The hash
    path uses only comparisons (``<``, ``>``) -- no floating-point arithmetic
    -- so associativity and rounding cannot affect the output.
    """
    sig = tuple(_bucket(f) for f in codeword.features)
    payload = {
        "bucket_signature": list(sig),
        "domain": codeword.domain,
        "schema_version": SYNDROME_SCHEMA_VERSION,
    }
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return SyndromeVector(
        schema_version=SYNDROME_SCHEMA_VERSION,
        bucket_signature=sig,
        digest=digest,
    )


def classify(
    codeword: Codeword, table: SyndromeTable | None = None
) -> str:
    """Return the anomaly class for ``codeword``. Never raises.

    Unknown signatures and unknown domains both resolve to
    ``"unclassified"``.
    """
    if table is None:
        table = load_default_table()
    sv = syndrome(codeword)
    return table.lookup(codeword.domain, sv.bucket_signature)


# ---- SyndromeRecord + SyndromeChainWriter + verify_syndrome_chain ----------


@dataclass(frozen=True, slots=True)
class SyndromeRecord:
    """Single entry in the sibling syndrome audit chain.

    Linked to the Phase 1000 audit chain by a shared ``round_index`` value --
    we intentionally do NOT cross-hash into the Phase 1000 chain so that
    Phase 1000's immutability contract is preserved.
    """

    record_type: str  # must equal "syndrome"
    schema_version: int
    round_index: int
    prev_hash: str
    codeword_digest: str
    syndrome_digest: str
    bucket_signature: Tuple[int, ...]
    anomaly_class: str
    domain: str

    def __post_init__(self) -> None:
        if type(self.record_type) is not str:
            raise TypeError(
                f"SyndromeRecord.record_type must be str, got "
                f"{type(self.record_type).__name__}"
            )
        if self.record_type != "syndrome":
            raise ValueError(
                f"SyndromeRecord.record_type must be 'syndrome', got "
                f"{self.record_type!r}"
            )
        if type(self.schema_version) is not int or isinstance(
            self.schema_version, bool
        ):
            raise TypeError(
                f"SyndromeRecord.schema_version must be int, got "
                f"{type(self.schema_version).__name__}"
            )
        if self.schema_version != SYNDROME_SCHEMA_VERSION:
            raise ValueError(
                f"SyndromeRecord.schema_version must be "
                f"{SYNDROME_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if type(self.round_index) is not int or isinstance(
            self.round_index, bool
        ):
            raise TypeError(
                f"SyndromeRecord.round_index must be int, got "
                f"{type(self.round_index).__name__}"
            )
        if self.round_index < 0:
            raise ValueError(
                f"SyndromeRecord.round_index must be >= 0, got {self.round_index}"
            )
        _validate_hex64(self.prev_hash, "prev_hash")
        _validate_hex64(self.codeword_digest, "codeword_digest")
        _validate_hex64(self.syndrome_digest, "syndrome_digest")
        if type(self.bucket_signature) is not tuple:
            raise TypeError(
                f"SyndromeRecord.bucket_signature must be tuple, got "
                f"{type(self.bucket_signature).__name__}"
            )
        if len(self.bucket_signature) != CODEWORD_DIM:
            raise ValueError(
                f"SyndromeRecord.bucket_signature must have length "
                f"{CODEWORD_DIM}, got {len(self.bucket_signature)}"
            )
        for i, b in enumerate(self.bucket_signature):
            if type(b) is not int or isinstance(b, bool):
                raise TypeError(
                    f"SyndromeRecord.bucket_signature[{i}] must be int"
                )
            if b not in (-1, 0, 1):
                raise ValueError(
                    f"SyndromeRecord.bucket_signature[{i}] must be in "
                    f"{{-1,0,1}}, got {b}"
                )
        if type(self.anomaly_class) is not str:
            raise TypeError(
                f"SyndromeRecord.anomaly_class must be str, got "
                f"{type(self.anomaly_class).__name__}"
            )
        if len(self.anomaly_class) < 1:
            raise ValueError("SyndromeRecord.anomaly_class must be non-empty")
        if type(self.domain) is not str:
            raise TypeError(
                f"SyndromeRecord.domain must be str, got "
                f"{type(self.domain).__name__}"
            )
        if len(self.domain) < 1:
            raise ValueError("SyndromeRecord.domain must be non-empty")


def _validate_hex64(value: object, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(
            f"SyndromeRecord.{field_name} must be str, got {type(value).__name__}"
        )
    if len(value) != 64:
        raise ValueError(
            f"SyndromeRecord.{field_name} must be 64 hex chars, got len={len(value)}"
        )
    for c in value:
        if c not in _HEX_ALPHABET:
            raise ValueError(
                f"SyndromeRecord.{field_name} must be lowercase hex, got "
                f"invalid char {c!r}"
            )


def hash_syndrome_record(record: SyndromeRecord) -> str:
    """SHA-256 hex digest of ``canonical_json(record)``."""
    return hashlib.sha256(canonical_json(record)).hexdigest()


def _syndrome_record_from_json(data: bytes) -> SyndromeRecord:
    """Parse a canonical-JSON line into a :class:`SyndromeRecord`."""
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(
            f"expected JSON object for SyndromeRecord, got {type(parsed).__name__}"
        )
    bs = parsed.get("bucket_signature", [])
    if not isinstance(bs, list):
        raise ValueError(
            f"expected JSON array for bucket_signature, got {type(bs).__name__}"
        )
    return SyndromeRecord(
        record_type=parsed["record_type"],
        schema_version=parsed["schema_version"],
        round_index=parsed["round_index"],
        prev_hash=parsed["prev_hash"],
        codeword_digest=parsed["codeword_digest"],
        syndrome_digest=parsed["syndrome_digest"],
        bucket_signature=tuple(bs),
        anomaly_class=parsed["anomaly_class"],
        domain=parsed["domain"],
    )


class SyndromeChainWriter:
    """Append-only writer for the sibling syndrome JSONL chain.

    Mirrors :class:`AuditChainWriter`'s discipline: ``append`` rejects
    records whose ``prev_hash`` does not match the current tip or whose
    ``round_index`` is not contiguous. Raises the Phase 1000
    :class:`ChainIntegrityError` so callers handle one exception type
    across both chains.
    """

    def __init__(self, path: pathlib.Path | str) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all_records(self) -> list[SyndromeRecord]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if not raw:
            return []
        records: list[SyndromeRecord] = []
        for i, line in enumerate(raw.splitlines()):
            if not line.strip():
                continue
            try:
                records.append(_syndrome_record_from_json(line))
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                raise ChainIntegrityError(
                    f"malformed syndrome record at line {i + 1}: {e}"
                ) from e
        return records

    def latest_hash(self) -> str:
        """Return the hash of the last appended record, or genesis if empty."""
        records = self._read_all_records()
        if not records:
            return GENESIS_PREV_HASH
        return hash_syndrome_record(records[-1])

    def _last_round_index(self) -> int | None:
        records = self._read_all_records()
        if not records:
            return None
        return records[-1].round_index

    def append(self, record: SyndromeRecord) -> str:
        """Append ``record``; return its hash. Raises ChainIntegrityError on break."""
        current_tip = self.latest_hash()
        if record.prev_hash != current_tip:
            raise ChainIntegrityError(
                f"prev_hash mismatch on syndrome append: expected={current_tip}, "
                f"got={record.prev_hash}"
            )
        last_idx = self._last_round_index()
        expected_round = 0 if last_idx is None else last_idx + 1
        if record.round_index != expected_round:
            raise ChainIntegrityError(
                f"non-contiguous round_index on syndrome append: expected="
                f"{expected_round}, got={record.round_index}"
            )
        blob = canonical_json(record) + b"\n"
        with open(self.path, "ab") as f:
            f.write(blob)
        return hash_syndrome_record(record)


def verify_syndrome_chain(path: pathlib.Path | str) -> None:
    """Verify the sibling syndrome JSONL chain at ``path``.

    Walks every record, re-derives each hash, and asserts ``prev_hash``
    linkage plus contiguous ``round_index`` starting at 0.

    Raises:
        ChainIntegrityError: on any mismatch (malformed line, non-contiguous
            index, prev_hash mismatch, or missing file).
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise ChainIntegrityError(f"syndrome chain file does not exist: {p}")
    raw = p.read_bytes()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return
    prev_hash_expected = GENESIS_PREV_HASH
    expected_round = 0
    for i, line in enumerate(lines):
        try:
            record = _syndrome_record_from_json(line)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            raise ChainIntegrityError(
                f"malformed syndrome record at line {i + 1}: {e}"
            ) from e
        if record.round_index != expected_round:
            raise ChainIntegrityError(
                f"syndrome round {i}: non-contiguous round_index (expected="
                f"{expected_round}, got={record.round_index})"
            )
        if record.prev_hash != prev_hash_expected:
            raise ChainIntegrityError(
                f"syndrome round {i}: prev_hash mismatch (expected="
                f"{prev_hash_expected}, got={record.prev_hash})"
            )
        prev_hash_expected = hash_syndrome_record(record)
        expected_round = record.round_index + 1


def append_syndrome_for_codeword(
    writer: SyndromeChainWriter,
    round_index: int,
    codeword: Codeword,
    table: SyndromeTable | None = None,
) -> str:
    """Convenience: compute syndrome, classify, build record, append.

    Returns the hash of the appended record.
    """
    sv = syndrome(codeword)
    cls = classify(codeword, table)
    rec = SyndromeRecord(
        record_type="syndrome",
        schema_version=SYNDROME_SCHEMA_VERSION,
        round_index=round_index,
        prev_hash=writer.latest_hash(),
        codeword_digest=codeword_digest(codeword),
        syndrome_digest=sv.digest,
        bucket_signature=sv.bucket_signature,
        anomaly_class=cls,
        domain=codeword.domain,
    )
    return writer.append(rec)
