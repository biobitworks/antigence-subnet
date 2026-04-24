"""Public-API smoke test for the deterministic_scoring subpackage.

Confirms that every name in ``__all__`` imports cleanly and that a
3-round chain built via the public API verifies without error.
"""

from __future__ import annotations


def test_public_api_imports():
    import antigence_subnet.validator.deterministic_scoring as ds

    expected = {
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
        # Phase 1002.
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
    assert set(ds.__all__) == expected
    for name in expected:
        assert hasattr(ds, name), f"missing public export: {name}"


def test_star_import_works():
    ns: dict[str, object] = {}
    exec(
        "from antigence_subnet.validator.deterministic_scoring import *",
        ns,
    )
    assert "AuditChainWriter" in ns
    assert "FrozenRoundScore" in ns


def test_three_round_end_to_end(tmp_path):
    """Build a 3-round chain via the public API, then verify."""
    from antigence_subnet.validator.deterministic_scoring import (
        GENESIS_PREV_HASH,
        AuditChainWriter,
        FrozenRoundRecord,
        FrozenRoundScore,
        hash_record,
        verify_chain,
    )

    path = tmp_path / "chain.jsonl"
    writer = AuditChainWriter(path)
    prev = GENESIS_PREV_HASH
    for i in range(3):
        scores = (
            FrozenRoundScore(
                uid=0, raw_reward=0.1 * (i + 1), ema_score=0.01 * (i + 1), hotkey="a"
            ),
            FrozenRoundScore(
                uid=1, raw_reward=0.2 * (i + 1), ema_score=0.02 * (i + 1), hotkey="b"
            ),
        )
        record = FrozenRoundRecord(
            round_index=i, ema_alpha=0.1, scores=scores, prev_hash=prev
        )
        writer.append(record)
        prev = hash_record(record)

    # Should not raise.
    verify_chain(path)
