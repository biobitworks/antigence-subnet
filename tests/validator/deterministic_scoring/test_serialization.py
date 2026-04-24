"""Tests for canonical JSON encoder/decoder (DETSCORE-02)."""

from __future__ import annotations

import math

import pytest

from antigence_subnet.validator.deterministic_scoring.serialization import (
    canonical_json,
    from_canonical_json,
)
from antigence_subnet.validator.deterministic_scoring.state import (
    FrozenRoundRecord,
    FrozenRoundScore,
)

_GENESIS = "0" * 64


# ---- Encoder: sorted keys, UTF-8, no whitespace -----------------------------


def test_canonical_json_sorts_keys_no_whitespace():
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_json_returns_bytes():
    out = canonical_json({"x": 1})
    assert isinstance(out, bytes)


def test_canonical_json_utf8_passthrough_not_ascii_escape():
    # ensure_ascii=False emits raw UTF-8 bytes, not \u00e9.
    out = canonical_json({"k": "héllo"})
    assert out == b'{"k":"h\xc3\xa9llo"}'
    # Verify round-trip through utf-8 decode.
    assert out.decode("utf-8") == '{"k":"héllo"}'


def test_canonical_json_nested_dict_sorted():
    obj = {"outer_b": {"inner_b": 1, "inner_a": 2}, "outer_a": 3}
    assert canonical_json(obj) == b'{"outer_a":3,"outer_b":{"inner_a":2,"inner_b":1}}'


# ---- Byte-identity (the DETSCORE-02 core requirement) -----------------------


def test_canonical_json_byte_identity_across_calls():
    s = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    assert canonical_json(s) == canonical_json(s)


def test_canonical_json_byte_identity_across_equal_instances():
    s1 = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    s2 = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    assert s1 is not s2
    assert canonical_json(s1) == canonical_json(s2)


def test_canonical_json_golden_bytes_for_frozen_round_score():
    s = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.3, hotkey="hk")
    # Keys sorted: ema_score, hotkey, raw_reward, uid.
    # Floats use repr(): 0.3 -> "0.3", 0.5 -> "0.5".
    expected = b'{"ema_score":0.3,"hotkey":"hk","raw_reward":0.5,"uid":1}'
    assert canonical_json(s) == expected


def test_canonical_json_golden_bytes_for_frozen_round_record():
    s = FrozenRoundScore(uid=2, raw_reward=1.0, ema_score=0.1, hotkey="h2")
    r = FrozenRoundRecord(
        round_index=0, ema_alpha=0.1, scores=(s,), prev_hash=_GENESIS
    )
    out = canonical_json(r)
    # Validate structure fragments deterministically.
    assert out.startswith(b'{"ema_alpha":0.1,')
    assert b'"prev_hash":"' + _GENESIS.encode() + b'"' in out
    assert b'"round_index":0' in out
    # scores array contains the one score with its own sorted keys.
    assert b'[{"ema_score":0.1,"hotkey":"h2","raw_reward":1.0,"uid":2}]' in out


# ---- NaN / Inf rejection ----------------------------------------------------


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": float("nan")})


def test_canonical_json_rejects_positive_infinity():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": float("inf")})


def test_canonical_json_rejects_negative_infinity():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": -float("inf")})


def test_canonical_json_rejects_nan_in_nested_list():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"items": [1.0, 2.0, float("nan")]})


def test_canonical_json_rejects_nan_in_dataclass_field():
    # Can't construct a FrozenRoundScore with a NaN float directly without
    # violating __post_init__? Actually __post_init__ only checks the type,
    # not finiteness. Confirm that nan *as float* gets past the dataclass
    # and is rejected by canonical_json.
    s = FrozenRoundScore(uid=1, raw_reward=float("nan"), ema_score=0.3, hotkey="hk")
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(s)


# ---- Float round-trip identity ---------------------------------------------


def test_canonical_json_float_roundtrip_exact():
    # 0.1 + 0.2 is the classic binary-float exhibit.
    val = 0.1 + 0.2
    encoded = canonical_json({"x": val})
    import json as _json
    decoded = _json.loads(encoded.decode("utf-8"))["x"]
    # repr() is required to round-trip for finite Python floats.
    assert decoded == val
    # And the hex bit patterns match exactly.
    assert math.isclose(decoded, val, rel_tol=0.0, abs_tol=0.0)


# ---- Decoder round-trip -----------------------------------------------------


def test_from_canonical_json_roundtrips_score():
    s = FrozenRoundScore(uid=7, raw_reward=0.25, ema_score=0.125, hotkey="alpha")
    blob = canonical_json(s)
    restored = from_canonical_json(blob, FrozenRoundScore)
    assert restored == s


def test_from_canonical_json_roundtrips_record():
    s1 = FrozenRoundScore(uid=1, raw_reward=0.5, ema_score=0.1, hotkey="h1")
    s2 = FrozenRoundScore(uid=2, raw_reward=0.4, ema_score=0.05, hotkey="h2")
    r = FrozenRoundRecord(
        round_index=3, ema_alpha=0.2, scores=(s1, s2), prev_hash="a" * 64
    )
    blob = canonical_json(r)
    restored = from_canonical_json(blob, FrozenRoundRecord)
    assert restored == r
    assert isinstance(restored.scores, tuple)
    assert all(isinstance(s, FrozenRoundScore) for s in restored.scores)


def test_from_canonical_json_rejects_wrong_type_arg():
    with pytest.raises(TypeError, match="unsupported target_type"):
        from_canonical_json(b'{}', dict)


def test_from_canonical_json_rejects_non_bytes_input():
    with pytest.raises(TypeError, match="expects bytes"):
        from_canonical_json("not bytes", FrozenRoundScore)  # type: ignore[arg-type]
