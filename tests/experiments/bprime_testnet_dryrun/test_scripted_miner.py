"""Phase 1104 scripted-miner unit tests.

Null-hypothesis evidence (one per test; intent in docstring):

* test_fixture_parses_cleanly
    NH0: the fixture JSON might be malformed or schema-drifted. Proves
    load_pattern() accepts it and required keys are present.

* test_fixture_matches_miner_contract
    NH0: the miner might parse fields differently than the fixture
    documents. Proves every fixture sample resolves through
    lookup_response() without raising, and returns a valid
    MinerResponse.

* test_known_sample_round_trip
    NH0: given a sample_id in the fixture, the miner might return a
    response not matching the fixture table. Proves round-trip fidelity.

* test_silent_default_is_byte_identical
    NH0: unknown sample_ids might produce non-deterministic defaults.
    Proves two calls with the same unknown id return byte-identical
    MinerResponse JSON.

* test_oscillate_alternates_by_round_parity
    NH0: oscillate mode might not actually alternate. Proves round 0 and
    round 1 produce opposite-sign anomaly_scores.

* test_hmac_accepts_good_signature
    NH0: HMAC verification might reject valid signatures. Proves
    verify_hmac(body, compute_hmac(body, secret), secret) is True.

* test_hmac_rejects_bad_signature
    NH0: HMAC verification might accept forged signatures. Proves a
    one-byte perturbation of the hex digest fails verification.

* test_miner_is_idempotent
    NH0: repeated calls for the same (sample_id, round_index) might
    drift (e.g., timestamp leak). Proves byte-identical responses across
    two calls.

* test_no_bittensor_import
    NH0: the miner module might pull bittensor transitively. Proves
    'bittensor' is not in sys.modules after importing the miner.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# conftest prepends experiments/bprime-testnet-dryrun/ to sys.path.
import scripted_adversarial_miner as miner_mod  # type: ignore[import-not-found]
from scripted_adversarial_miner import (  # type: ignore[import-not-found]
    MinerResponse,
    compute_hmac,
    load_pattern,
    lookup_response,
    verify_hmac,
)


# --------------------------------------------------------------------- #
# Fixture + schema                                                       #
# --------------------------------------------------------------------- #
def test_fixture_parses_cleanly(fixture_path: pathlib.Path) -> None:
    data = load_pattern(fixture_path)
    assert data["schema_version"] == 1
    assert isinstance(data["samples"], dict)
    assert isinstance(data["default_response"], dict)
    assert isinstance(data["attack_modes"], dict)
    # A handful of attack modes must be documented.
    for mode in ("honeypot_fail", "oscillate", "score_flip_replay",
                 "confidence_spike", "silent_default"):
        assert mode in data["attack_modes"], f"{mode} missing from attack_modes"


def test_fixture_matches_miner_contract(fixture_path: pathlib.Path) -> None:
    data = load_pattern(fixture_path)
    for sample_id in data["samples"]:
        # round 0 (even) and round 1 (odd) must both resolve without error.
        r0 = lookup_response(data, sample_id, 0)
        r1 = lookup_response(data, sample_id, 1)
        assert isinstance(r0, MinerResponse)
        assert isinstance(r1, MinerResponse)
        assert r0.sample_id == sample_id
        assert -1.0 <= r0.anomaly_score <= 1.0
        assert 0.0 <= r0.confidence <= 1.0


# --------------------------------------------------------------------- #
# Deterministic lookup                                                   #
# --------------------------------------------------------------------- #
def test_known_sample_round_trip(fixture_path: pathlib.Path) -> None:
    data = load_pattern(fixture_path)
    resp = lookup_response(data, "sample-honeypot-001", round_index=0)
    assert resp.attack_mode == "honeypot_fail"
    assert resp.anomaly_type == "false_positive_honeypot"
    assert pytest.approx(resp.anomaly_score, rel=0, abs=1e-12) == 0.95
    assert pytest.approx(resp.confidence, rel=0, abs=1e-12) == 0.88


def test_silent_default_is_byte_identical(fixture_path: pathlib.Path) -> None:
    data = load_pattern(fixture_path)
    r1 = lookup_response(data, "sample-does-not-exist", round_index=7)
    r2 = lookup_response(data, "sample-does-not-exist", round_index=7)
    # Byte-identical canonical JSON.
    assert r1.model_dump_json() == r2.model_dump_json()
    assert r1.attack_mode == "none"
    assert r1.anomaly_type == "unknown"
    assert r1.anomaly_score == 0.0
    assert r1.confidence == 0.0


def test_oscillate_alternates_by_round_parity(fixture_path: pathlib.Path) -> None:
    data = load_pattern(fixture_path)
    even = lookup_response(data, "sample-osc-001", round_index=0)
    odd = lookup_response(data, "sample-osc-001", round_index=1)
    assert even.attack_mode == "oscillate"
    assert odd.attack_mode == "oscillate"
    # Signs must differ; magnitudes equal.
    assert even.anomaly_score == -odd.anomaly_score
    assert even.anomaly_score != 0.0


# --------------------------------------------------------------------- #
# HMAC                                                                   #
# --------------------------------------------------------------------- #
def test_hmac_accepts_good_signature() -> None:
    secret = b"test-secret-001"
    body = b'{"sample_id":"sample-osc-001","round_index":0}'
    sig = compute_hmac(body, secret)
    assert verify_hmac(body, sig, secret) is True


def test_hmac_rejects_bad_signature() -> None:
    secret = b"test-secret-001"
    body = b'{"sample_id":"sample-osc-001","round_index":0}'
    good = compute_hmac(body, secret)
    # Flip the last hex char; constant-time compare must reject.
    bad = good[:-1] + ("0" if good[-1] != "0" else "1")
    assert verify_hmac(body, bad, secret) is False
    # Different secret also rejected.
    assert verify_hmac(body, good, b"other-secret") is False
    # Empty signature rejected.
    assert verify_hmac(body, "", secret) is False


# --------------------------------------------------------------------- #
# Idempotency                                                            #
# --------------------------------------------------------------------- #
def test_miner_is_idempotent(fixture_path: pathlib.Path) -> None:
    data = load_pattern(fixture_path)
    a = lookup_response(data, "sample-confidence-001", round_index=42)
    b = lookup_response(data, "sample-confidence-001", round_index=42)
    assert a.model_dump_json() == b.model_dump_json()
    # And across a fresh pattern reload:
    data2 = load_pattern(fixture_path)
    c = lookup_response(data2, "sample-confidence-001", round_index=42)
    assert a.model_dump_json() == c.model_dump_json()


# --------------------------------------------------------------------- #
# Isolation                                                              #
# --------------------------------------------------------------------- #
def test_no_bittensor_import() -> None:
    # Run in a clean subprocess so the check is independent of whatever
    # other tests in the same pytest session may have imported.
    # Under the minimal venv bittensor isn't installed (test is trivially true);
    # under the full bittensor venv we must verify the miner module itself
    # does not transitively pull bittensor at import time.
    import subprocess
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    script = (
        "import sys, importlib.util; "
        "spec = importlib.util.spec_from_file_location("
        "'scripted_adversarial_miner', "
        f"r'{repo_root}/experiments/bprime-testnet-dryrun/scripted_adversarial_miner.py'); "
        "m = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(m); "
        "assert 'bittensor' not in sys.modules, 'transitive bittensor import detected'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        "scripted_adversarial_miner transitively imported bittensor at module "
        "load (clean-subprocess check). stderr: " + result.stderr[-400:]
    )


def test_miner_module_has_no_bittensor_source() -> None:
    """Belt-and-suspenders: source of the miner must not reference bittensor."""
    path = pathlib.Path(miner_mod.__file__)
    text = path.read_text(encoding="utf-8")
    # Match top-of-line imports only; the word 'bittensor' may appear in
    # docstrings (as context) and that is fine. An actual import is what
    # we reject.
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import bittensor"), (
            f"found import bittensor in {path}"
        )
        assert not stripped.startswith("from bittensor"), (
            f"found from bittensor in {path}"
        )


def test_fixture_has_no_biological_terms(fixture_path: pathlib.Path) -> None:
    """Scope boundary: no peptide / k-mer / protein / sequence-model terms."""
    text = fixture_path.read_text(encoding="utf-8").lower()
    for term in ("peptide", "k-mer", "kmer", "protein", "sequence-model"):
        assert term not in text, f"fixture contains out-of-scope term {term!r}"
