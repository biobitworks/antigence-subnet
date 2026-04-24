"""Tests for Phase 1002 syndrome layer (SYNDROME-01..05)."""

from __future__ import annotations

import dataclasses

import pytest

from antigence_subnet.validator.deterministic_scoring.serialization import (
    canonical_json,
)
from antigence_subnet.validator.deterministic_scoring.syndrome import (
    CODEWORD_DIM,
    SYNDROME_SCHEMA_VERSION,
    Codeword,
    SyndromeTable,
    SyndromeVector,
    classify,
    codeword_digest,
    load_default_table,
    syndrome,
)
from tests.validator.deterministic_scoring.fixtures.codewords import (
    ALL_FIXTURES,
    CW_ALTERNATING_SPIKE,
    CW_SATURATION_STORM,
    CW_SELF,
    CW_TOTAL_COLLAPSE,
    CW_UNCLASSIFIED,
    EXPECTED_BUCKET_SIGNATURES,
    EXPECTED_CLASSES,
)

# ---- Codeword construction & immutability ----------------------------------


def test_codeword_constructs_ok():
    cw = Codeword(schema_version=1, features=(0.0,) * 8, domain="generic")
    assert cw.schema_version == 1
    assert cw.features == (0.0,) * 8
    assert cw.domain == "generic"


def test_codeword_rejects_mutation():
    cw = Codeword(schema_version=1, features=(0.0,) * 8, domain="generic")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cw.domain = "other"  # type: ignore[misc]


def test_codeword_rejects_bad_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        Codeword(schema_version=2, features=(0.0,) * 8, domain="generic")


def test_codeword_rejects_schema_version_wrong_type():
    with pytest.raises(TypeError):
        Codeword(schema_version="1", features=(0.0,) * 8, domain="generic")  # type: ignore[arg-type]


def test_codeword_rejects_wrong_length():
    with pytest.raises(ValueError, match="CODEWORD_DIM|8"):
        Codeword(schema_version=1, features=(0.0,) * 7, domain="generic")


def test_codeword_rejects_nan():
    with pytest.raises(ValueError, match="non-finite|NaN"):
        Codeword(
            schema_version=1,
            features=(float("nan"),) + ((0.0,) * 7),
            domain="generic",
        )


def test_codeword_rejects_positive_inf():
    with pytest.raises(ValueError, match="non-finite|Inf"):
        Codeword(
            schema_version=1,
            features=(float("inf"),) + ((0.0,) * 7),
            domain="generic",
        )


def test_codeword_rejects_negative_inf():
    with pytest.raises(ValueError, match="non-finite|Inf"):
        Codeword(
            schema_version=1,
            features=(float("-inf"),) + ((0.0,) * 7),
            domain="generic",
        )


def test_codeword_rejects_int_in_features():
    with pytest.raises(TypeError):
        Codeword(  # type: ignore[arg-type]
            schema_version=1,
            features=(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            domain="generic",
        )


def test_codeword_rejects_numpy_scalar_like():
    """Subclasses of float (including numpy.float64) must be rejected."""
    class NumpyFloatLike(float):
        pass

    with pytest.raises(TypeError):
        Codeword(
            schema_version=1,
            features=(NumpyFloatLike(0.0),) + ((0.0,) * 7),
            domain="generic",
        )


def test_codeword_rejects_empty_domain():
    with pytest.raises(ValueError):
        Codeword(schema_version=1, features=(0.0,) * 8, domain="")


def test_codeword_rejects_domain_wrong_type():
    with pytest.raises(TypeError):
        Codeword(schema_version=1, features=(0.0,) * 8, domain=123)  # type: ignore[arg-type]


def test_codeword_rejects_features_wrong_container_type():
    with pytest.raises(TypeError):
        Codeword(schema_version=1, features=[0.0] * 8, domain="generic")  # type: ignore[arg-type]


# ---- codeword_digest --------------------------------------------------------


def test_codeword_digest_is_64_lower_hex():
    cw = Codeword(schema_version=1, features=(0.0,) * 8, domain="generic")
    d = codeword_digest(cw)
    assert isinstance(d, str)
    assert len(d) == 64
    assert d == d.lower()
    assert all(c in "0123456789abcdef" for c in d)


def test_codeword_digest_is_deterministic():
    cw = Codeword(
        schema_version=1,
        features=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        domain="generic",
    )
    d1 = codeword_digest(cw)
    d2 = codeword_digest(cw)
    assert d1 == d2


def test_codeword_digest_equal_instances_match():
    cw1 = Codeword(schema_version=1, features=(0.0,) * 8, domain="generic")
    cw2 = Codeword(schema_version=1, features=(0.0,) * 8, domain="generic")
    assert codeword_digest(cw1) == codeword_digest(cw2)


def test_codeword_digest_differs_by_domain():
    cw_a = Codeword(schema_version=1, features=(0.0,) * 8, domain="generic")
    cw_b = Codeword(schema_version=1, features=(0.0,) * 8, domain="other")
    assert codeword_digest(cw_a) != codeword_digest(cw_b)


# ---- SyndromeTable / load_default_table ------------------------------------


def test_load_default_table_returns_syndrome_table():
    t = load_default_table()
    assert isinstance(t, SyndromeTable)
    assert t.version == 1


def test_load_default_table_has_generic_domain_with_four_entries():
    t = load_default_table()
    assert "generic" in t.entries
    assert len(t.entries["generic"]) == 4


def test_load_default_table_caches_instance():
    t1 = load_default_table()
    t2 = load_default_table()
    assert t1 is t2


def test_load_default_table_entries_are_readonly():
    t = load_default_table()
    with pytest.raises(TypeError):
        t.entries["generic"]["sig:0,0,0,0,0,0,0,0"] = "attack"  # type: ignore[index]


def test_load_default_table_outer_mapping_is_readonly():
    t = load_default_table()
    with pytest.raises(TypeError):
        t.entries["new_domain"] = {}  # type: ignore[index]


def test_lookup_returns_self_for_zero_signature():
    t = load_default_table()
    assert t.lookup("generic", (0, 0, 0, 0, 0, 0, 0, 0)) == "self"


def test_lookup_returns_total_collapse():
    t = load_default_table()
    assert t.lookup("generic", (-1, -1, -1, -1, -1, -1, -1, -1)) == "total_collapse"


def test_lookup_returns_alternating_spike():
    t = load_default_table()
    assert t.lookup("generic", (1, 0, 1, 0, 1, 0, 1, 0)) == "alternating_spike"


def test_lookup_returns_saturation_storm():
    t = load_default_table()
    assert t.lookup("generic", (1, 1, 1, 1, 1, 1, 1, 1)) == "saturation_storm"


def test_lookup_returns_unclassified_for_unknown_signature():
    t = load_default_table()
    assert t.lookup("generic", (1, -1, 0, 1, -1, 0, 1, -1)) == "unclassified"


def test_lookup_returns_unclassified_for_unknown_domain():
    t = load_default_table()
    assert t.lookup("unknown_domain", (0, 0, 0, 0, 0, 0, 0, 0)) == "unclassified"


def test_lookup_accepts_list_input():
    t = load_default_table()
    assert t.lookup("generic", [0, 0, 0, 0, 0, 0, 0, 0]) == "self"


def test_lookup_table_file_is_canonical_json():
    """Disk bytes of syndrome_table_v1.json must equal canonical_json(parsed)."""
    import importlib.resources as resources
    import json

    raw = (
        resources.files("antigence_subnet.validator.deterministic_scoring")
        .joinpath("syndrome_table_v1.json")
        .read_bytes()
    )
    # Allow (but tolerate absence of) a single trailing newline.
    stripped = raw.rstrip(b"\n")
    parsed = json.loads(stripped.decode("utf-8"))
    assert canonical_json(parsed) == stripped


def test_lookup_never_raises_on_wrong_length_tuple():
    t = load_default_table()
    assert t.lookup("generic", (0, 0, 0)) == "unclassified"


def test_codeword_dim_constant_is_eight():
    assert CODEWORD_DIM == 8


# ---- syndrome() / SyndromeVector (SYNDROME-02) -----------------------------


def test_syndrome_bucket_signatures_match_fixtures():
    got = {
        "CW_SELF": syndrome(CW_SELF).bucket_signature,
        "CW_TOTAL_COLLAPSE": syndrome(CW_TOTAL_COLLAPSE).bucket_signature,
        "CW_ALTERNATING_SPIKE": syndrome(CW_ALTERNATING_SPIKE).bucket_signature,
        "CW_SATURATION_STORM": syndrome(CW_SATURATION_STORM).bucket_signature,
        "CW_UNCLASSIFIED": syndrome(CW_UNCLASSIFIED).bucket_signature,
    }
    assert got == EXPECTED_BUCKET_SIGNATURES


def test_syndrome_returns_syndrome_vector():
    sv = syndrome(CW_SELF)
    assert isinstance(sv, SyndromeVector)
    assert sv.schema_version == SYNDROME_SCHEMA_VERSION


def test_syndrome_digest_is_64_hex():
    sv = syndrome(CW_SELF)
    assert len(sv.digest) == 64
    assert sv.digest == sv.digest.lower()
    assert all(c in "0123456789abcdef" for c in sv.digest)


def test_syndrome_is_pure_in_process():
    sv1 = syndrome(CW_ALTERNATING_SPIKE)
    sv2 = syndrome(CW_ALTERNATING_SPIKE)
    assert sv1 == sv2
    assert canonical_json(sv1) == canonical_json(sv2)


def test_syndrome_is_deterministic_across_processes():
    """Byte-identity of canonical_json(syndrome(cw)) across subprocess boundary.

    This is the SYNDROME-02 core null-hypothesis evidence: a fresh Python
    interpreter computes the same bytes.
    """
    import os
    import subprocess
    import sys

    script = (
        "import sys; "
        "from antigence_subnet.validator.deterministic_scoring.syndrome "
        "import syndrome, Codeword; "
        "from antigence_subnet.validator.deterministic_scoring.serialization "
        "import canonical_json; "
        "cw = Codeword(schema_version=1, "
        "features=(2.0, 0.0, 3.0, 0.0, 1.5, 0.0, 4.0, 0.0), "
        "domain='generic'); "
        "sys.stdout.buffer.write(canonical_json(syndrome(cw)))"
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        env=env,
    )
    in_process = canonical_json(syndrome(CW_ALTERNATING_SPIKE))
    assert result.stdout == in_process


def test_syndrome_digest_matches_hand_computed_golden():
    """Golden-hash test: the digest is sha256(canonical_json(scoped payload))."""
    import hashlib

    cw = CW_SELF
    payload = {
        "bucket_signature": [0, 0, 0, 0, 0, 0, 0, 0],
        "domain": "generic",
        "schema_version": 1,
    }
    expected = hashlib.sha256(canonical_json(payload)).hexdigest()
    assert syndrome(cw).digest == expected


def test_syndrome_vector_rejects_mutation():
    sv = syndrome(CW_SELF)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sv.digest = "0" * 64  # type: ignore[misc]


def test_syndrome_vector_is_hashable():
    sv = syndrome(CW_SELF)
    assert sv in {sv}


def test_bucket_threshold_is_strict_at_boundary():
    """Boundary values 1.0 and -1.0 collapse to bucket 0 (strict < and >)."""
    cw = Codeword(
        schema_version=1,
        features=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
        domain="generic",
    )
    assert syndrome(cw).bucket_signature == (0, 0, 0, 0, 0, 0, 0, 0)


def test_bucket_threshold_just_above_boundary():
    """Values infinitesimally above 1.0 bucket to +1."""
    cw = Codeword(
        schema_version=1,
        features=(1.0000001,) * 8,
        domain="generic",
    )
    assert syndrome(cw).bucket_signature == (1,) * 8


# ---- classify() (SYNDROME-03) ---------------------------------------------


def test_classify_maps_all_five_fixtures_to_expected_classes():
    got = {
        "CW_SELF": classify(CW_SELF),
        "CW_TOTAL_COLLAPSE": classify(CW_TOTAL_COLLAPSE),
        "CW_ALTERNATING_SPIKE": classify(CW_ALTERNATING_SPIKE),
        "CW_SATURATION_STORM": classify(CW_SATURATION_STORM),
        "CW_UNCLASSIFIED": classify(CW_UNCLASSIFIED),
    }
    assert got == EXPECTED_CLASSES


def test_classify_uses_provided_table_when_given():
    """Caller-supplied table overrides the default."""
    custom = SyndromeTable(
        version=1,
        entries={"generic": {"sig:0,0,0,0,0,0,0,0": "custom_self"}},
    )
    assert classify(CW_SELF, table=custom) == "custom_self"


def test_classify_fixtures_iteration_stable():
    """Iterating ALL_FIXTURES in any order yields the expected set of classes."""
    names = (
        "CW_SELF",
        "CW_TOTAL_COLLAPSE",
        "CW_ALTERNATING_SPIKE",
        "CW_SATURATION_STORM",
        "CW_UNCLASSIFIED",
    )
    for name, cw in zip(names, ALL_FIXTURES, strict=True):
        assert classify(cw) == EXPECTED_CLASSES[name]


# ---- SYNDROME-05 end-to-end integration test -------------------------------


def test_end_to_end_phase_1000_plus_1002_chains(tmp_path):
    """Build both the Phase 1000 audit chain and the Phase 1002 syndrome chain
    in lockstep over 5 rounds (one codeword per round). Verify both. Confirm
    round_index sets match and persisted anomaly_class values match classify().
    """
    import json
    import pathlib

    from antigence_subnet.validator.deterministic_scoring import (
        GENESIS_PREV_HASH,
        AuditChainWriter,
        FrozenRoundRecord,
        FrozenRoundScore,
        SyndromeChainWriter,
        append_syndrome_for_codeword,
        verify_chain,
        verify_syndrome_chain,
    )

    audit_path: pathlib.Path = tmp_path / "chain.jsonl"
    syn_path: pathlib.Path = tmp_path / "chain.syndromes.jsonl"
    audit_writer = AuditChainWriter(audit_path)
    syn_writer = SyndromeChainWriter(syn_path)

    names = (
        "CW_SELF",
        "CW_TOTAL_COLLAPSE",
        "CW_ALTERNATING_SPIKE",
        "CW_SATURATION_STORM",
        "CW_UNCLASSIFIED",
    )

    audit_prev = GENESIS_PREV_HASH
    for r, cw in enumerate(ALL_FIXTURES):
        rec = FrozenRoundRecord(
            round_index=r,
            ema_alpha=0.1,
            scores=(FrozenRoundScore(uid=0, raw_reward=0.5, ema_score=0.5, hotkey="hk0"),),
            prev_hash=audit_prev,
        )
        audit_prev = audit_writer.append(rec)
        append_syndrome_for_codeword(syn_writer, r, cw)

    verify_chain(audit_path)
    verify_syndrome_chain(syn_path)

    audit_rounds = [
        json.loads(ln.decode("utf-8"))["round_index"]
        for ln in audit_path.read_bytes().splitlines()
        if ln.strip()
    ]
    syn_lines = [
        json.loads(ln.decode("utf-8"))
        for ln in syn_path.read_bytes().splitlines()
        if ln.strip()
    ]
    syn_rounds = [entry["round_index"] for entry in syn_lines]
    assert audit_rounds == syn_rounds == list(range(5))

    for name, entry in zip(names, syn_lines, strict=True):
        assert entry["anomaly_class"] == EXPECTED_CLASSES[name]


def test_module_api_exports_syndrome_symbols():
    """Top-level package re-exports every new Phase 1002 symbol."""
    import antigence_subnet.validator.deterministic_scoring as mod

    expected = {
        "CODEWORD_DIM",
        "Codeword",
        "SYNDROME_SCHEMA_VERSION",
        "SyndromeChainWriter",
        "SyndromeRecord",
        "SyndromeTable",
        "SyndromeVector",
        "append_syndrome_for_codeword",
        "classify",
        "codeword_digest",
        "hash_syndrome_record",
        "load_default_table",
        "syndrome",
        "verify_syndrome_chain",
    }
    for name in expected:
        assert hasattr(mod, name), f"missing top-level export: {name}"
        assert name in mod.__all__, f"missing from __all__: {name}"


def test_module_api_preserves_phase_1000_and_1001_exports():
    """Every symbol from the earlier phases is still re-exported."""
    import antigence_subnet.validator.deterministic_scoring as mod

    prior = {
        # Phase 1000.
        "AuditChainWriter",
        "ChainIntegrityError",
        "FrozenRoundRecord",
        "FrozenRoundScore",
        "GENESIS_PREV_HASH",
        "ReplayResult",
        "RoundInputs",
        "canonical_json",
        "from_canonical_json",
        "hash_record",
        "replay_chain",
        "verify_chain",
        # Phase 1001.
        "EVENT_SCHEMA_VERSION",
        "TrajectoryWindow",
        "detect_convergence_failure",
        "detect_metastability",
        "detect_oscillation",
        "extract_trajectories",
    }
    for name in prior:
        assert hasattr(mod, name), f"missing prior-phase export: {name}"
        assert name in mod.__all__, f"missing from __all__: {name}"
