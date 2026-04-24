"""Deterministic validator scoring layer (Phase 1000 + 1001 + 1002).

Additive module providing immutable round-score dataclasses, a byte-stable
canonical JSON serializer, a SHA-256 linked audit chain, convergence
monitoring (Phase 1001), and syndrome-based anomaly detection (Phase 1002).
Sits alongside the existing mutable numpy-based scoring in
``validator/scoring.py`` and ``validator/reward.py`` without modifying them.

Pure-stdlib implementation -- no numpy, no bittensor, no torch -- so the
whole subpackage imports cleanly in a minimal Python environment and is
fully autonomously testable under pytest.

Public API
----------
Frozen dataclasses (DETSCORE-01):
    * :class:`FrozenRoundScore`
    * :class:`FrozenRoundRecord`

Canonical JSON (DETSCORE-02):
    * :func:`canonical_json`
    * :func:`from_canonical_json`

Audit chain (DETSCORE-03):
    * :class:`AuditChainWriter`
    * :class:`ChainIntegrityError`
    * :func:`hash_record`
    * :func:`verify_chain`
    * :data:`GENESIS_PREV_HASH`

Replay harness (DETSCORE-04):
    * :class:`RoundInputs`
    * :class:`ReplayResult`
    * :func:`replay_chain`

Syndrome layer (Phase 1002 / SYNDROME-01..05):
    * :class:`Codeword` (fixed-length codeword vector)
    * :class:`SyndromeVector` (bucketed codeword + digest)
    * :class:`SyndromeTable` (packaged lookup table)
    * :class:`SyndromeRecord` (sibling chain record)
    * :class:`SyndromeChainWriter` (sibling JSONL writer)
    * :func:`syndrome`
    * :func:`classify`
    * :func:`load_default_table`
    * :func:`codeword_digest`
    * :func:`hash_syndrome_record`
    * :func:`verify_syndrome_chain`
    * :func:`append_syndrome_for_codeword`
    * :data:`SYNDROME_SCHEMA_VERSION`
    * :data:`CODEWORD_DIM`
"""

from antigence_subnet.validator.deterministic_scoring.chain import (
    AuditChainWriter,
    ChainIntegrityError,
    GENESIS_PREV_HASH,
    hash_record,
    verify_chain,
)
from antigence_subnet.validator.deterministic_scoring.monitors import (
    EVENT_SCHEMA_VERSION,
    detect_convergence_failure,
    detect_metastability,
    detect_oscillation,
)
from antigence_subnet.validator.deterministic_scoring.replay import (
    ReplayResult,
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
from antigence_subnet.validator.deterministic_scoring.syndrome import (
    CODEWORD_DIM,
    SYNDROME_SCHEMA_VERSION,
    Codeword,
    SyndromeChainWriter,
    SyndromeRecord,
    SyndromeTable,
    SyndromeVector,
    append_syndrome_for_codeword,
    classify,
    codeword_digest,
    hash_syndrome_record,
    load_default_table,
    syndrome,
    verify_syndrome_chain,
)
from antigence_subnet.validator.deterministic_scoring.trajectory import (
    TrajectoryWindow,
    extract_trajectories,
)

__all__ = [
    "AuditChainWriter",
    "CODEWORD_DIM",
    "ChainIntegrityError",
    "Codeword",
    "EVENT_SCHEMA_VERSION",
    "FrozenRoundRecord",
    "FrozenRoundScore",
    "GENESIS_PREV_HASH",
    "ReplayResult",
    "RoundInputs",
    "SYNDROME_SCHEMA_VERSION",
    "SyndromeChainWriter",
    "SyndromeRecord",
    "SyndromeTable",
    "SyndromeVector",
    "TrajectoryWindow",
    "append_syndrome_for_codeword",
    "canonical_json",
    "classify",
    "codeword_digest",
    "detect_convergence_failure",
    "detect_metastability",
    "detect_oscillation",
    "extract_trajectories",
    "from_canonical_json",
    "hash_record",
    "hash_syndrome_record",
    "load_default_table",
    "replay_chain",
    "syndrome",
    "verify_chain",
    "verify_syndrome_chain",
]
