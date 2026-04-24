"""Canonical JSON encoder and decoder for the deterministic scoring layer.

Design goals (DETSCORE-02):
    * Byte-identical output for equal inputs across independent runs.
    * Sorted keys, no whitespace, UTF-8 encoding (no \\uXXXX escapes for
      non-ASCII; we emit raw UTF-8).
    * Finite-float representation via Python's ``repr(float)``, which the
      language guarantees to round-trip for any finite IEEE-754 double.
    * NaN, +Inf, -Inf are rejected with ``ValueError`` at encode time. A
      deterministic audit chain has no sane way to serialize non-finite
      floats, so silent coercion would defeat the purpose.

The encoder walks dataclass/dict/list/tuple trees. Dataclass instances are
converted to dicts via ``dataclasses.asdict`` (which recurses). Tuples are
converted to lists because JSON has no tuple type -- the decoder restores
the tuple form when reconstructing ``FrozenRoundRecord.scores``.

This module is pure-stdlib: ``json``, ``dataclasses``, ``math``. It does
not import numpy or bittensor, so it can serialize ``FrozenRoundScore`` /
``FrozenRoundRecord`` in any minimal environment.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, TypeVar

from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
    FrozenRoundScore,
)

T = TypeVar("T")


def _walk_reject_nonfinite(obj: Any) -> None:
    """Raise ValueError on any NaN/Inf float found in obj.

    We walk dataclasses, dicts, lists, and tuples. Any other object is
    passed through (primitives are checked by ``_is_bad_float``).
    """
    if _is_bad_float(obj):
        raise ValueError(
            f"canonical_json rejects non-finite float: {obj!r}"
        )
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Walk each field value; avoid dataclasses.asdict() here because we
        # want to catch non-finite floats before allocating the dict copy.
        for f in dataclasses.fields(obj):
            _walk_reject_nonfinite(getattr(obj, f.name))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_reject_nonfinite(v)
            # Keys should be JSON-serializable strings by convention;
            # canonical_json only encodes dataclass trees.
            if _is_bad_float(k):
                raise ValueError(
                    f"canonical_json rejects non-finite float key: {k!r}"
                )
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _walk_reject_nonfinite(item)
        return


def _is_bad_float(obj: Any) -> bool:
    # bool is a subclass of int, not float, so this won't catch True/False.
    if type(obj) is not float:
        return False
    return math.isnan(obj) or math.isinf(obj)


def _to_plain(obj: Any) -> Any:
    """Convert dataclass/tuple trees into plain dict/list trees for json.dumps.

    Tuples become lists (JSON has no tuple). Dataclasses become dicts. The
    result is safe to hand to ``json.dumps`` with ``sort_keys=True``.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> bytes:
    """Encode ``obj`` to canonical JSON bytes.

    Accepts ``FrozenRoundScore``, ``FrozenRoundRecord``, plain dicts/lists,
    tuples (converted to lists), and scalar values.

    Raises:
        ValueError: if any embedded float is NaN or +/-Inf.
    """
    _walk_reject_nonfinite(obj)
    plain = _to_plain(obj)
    # allow_nan=False is redundant after the pre-walk but is a belt-and-
    # suspenders guard: if _walk_reject_nonfinite ever misses a case, the
    # json library will still raise (ValueError) rather than silently
    # emitting "NaN" / "Infinity" (which are non-standard JSON).
    text = json.dumps(
        plain,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def from_canonical_json(data: bytes, target_type: type[T]) -> T:
    """Decode canonical JSON bytes into the requested frozen dataclass type.

    Supports ``FrozenRoundScore`` and ``FrozenRoundRecord``. For generic
    dict/list decoding, callers can use ``json.loads(data.decode("utf-8"))``
    directly.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(
            f"from_canonical_json expects bytes, got {type(data).__name__}"
        )
    parsed = json.loads(data.decode("utf-8"))
    if target_type is FrozenRoundScore:
        if not isinstance(parsed, dict):
            raise ValueError(
                f"expected JSON object for FrozenRoundScore, got "
                f"{type(parsed).__name__}"
            )
        return FrozenRoundScore(**parsed)  # type: ignore[return-value]
    if target_type is FrozenRoundRecord:
        if not isinstance(parsed, dict):
            raise ValueError(
                f"expected JSON object for FrozenRoundRecord, got "
                f"{type(parsed).__name__}"
            )
        scores_list = parsed.get("scores", [])
        if not isinstance(scores_list, list):
            raise ValueError(
                f"expected JSON array for scores, got "
                f"{type(scores_list).__name__}"
            )
        scores = tuple(
            FrozenRoundScore(**s) if isinstance(s, dict) else s
            for s in scores_list
        )
        return FrozenRoundRecord(  # type: ignore[return-value]
            round_index=parsed["round_index"],
            ema_alpha=parsed["ema_alpha"],
            scores=scores,
            prev_hash=parsed["prev_hash"],
        )
    raise TypeError(
        f"from_canonical_json: unsupported target_type {target_type!r}"
    )
