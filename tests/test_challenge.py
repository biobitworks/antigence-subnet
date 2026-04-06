"""
Tests for per-miner challenge selection, adversarial injection, and dataset refresh detection.

Requirements: CHEAT-01 (per-miner unique challenges), CHEAT-02 (memorization resistance),
CHEAT-03 (adversarial sample injection), MAIN-04 (challenge randomization)
"""


from antigence_subnet.validator.challenge import (
    detect_dataset_refresh,
    get_miner_challenge,
    inject_adversarial_samples,
)


def _make_pool(n: int, domain: str = "hallucination") -> list[dict]:
    """Create a synthetic sample pool for testing."""
    return [
        {
            "id": f"sample_{i:03d}",
            "prompt": f"Prompt {i}",
            "output": f"Output {i}",
            "domain": domain,
        }
        for i in range(n)
    ]


class TestGetMinerChallenge:
    """Tests for get_miner_challenge deterministic selection."""

    def test_different_hotkeys_get_different_subsets(self):
        """Two different hotkeys receiving the same pool get at least one different sample."""
        pool = _make_pool(20)
        hotkey_a = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        hotkey_b = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

        subset_a = get_miner_challenge(pool, hotkey_a, round_num=1, n=10)
        subset_b = get_miner_challenge(pool, hotkey_b, round_num=1, n=10)

        ids_a = {s["id"] for s in subset_a}
        ids_b = {s["id"] for s in subset_b}

        assert len(subset_a) == 10
        assert len(subset_b) == 10
        assert ids_a != ids_b, "Different hotkeys must get different subsets"

    def test_same_hotkey_is_deterministic(self):
        """Same hotkey + same pool + same round always returns identical subset."""
        pool = _make_pool(20)
        hotkey = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"

        subset1 = get_miner_challenge(pool, hotkey, round_num=5, n=10)
        subset2 = get_miner_challenge(pool, hotkey, round_num=5, n=10)

        ids1 = [s["id"] for s in subset1]
        ids2 = [s["id"] for s in subset2]

        assert ids1 == ids2, "Same inputs must produce same output"

    def test_small_pool_returns_all(self):
        """When pool has fewer than n samples, returns all samples without crash."""
        pool = _make_pool(5)
        hotkey = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"

        result = get_miner_challenge(pool, hotkey, round_num=1, n=10)

        assert len(result) == 5
        assert {s["id"] for s in result} == {s["id"] for s in pool}

    def test_memorizing_miner_scores_random(self):
        """A memorizing miner's cached answers become useless after dataset refresh.

        Simulates: miner memorizes sample_id -> answer mappings from pool_v1.
        After refresh to pool_v2 (different IDs), cached lookups return nothing.
        """
        pool_v1 = _make_pool(20)
        pool_v2 = [
            {
                "id": f"new_sample_{i:03d}",
                "prompt": f"New prompt {i}",
                "output": f"New output {i}",
                "domain": "hallucination",
            }
            for i in range(20)
        ]

        hotkey = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"

        # Miner memorizes v1 sample IDs
        v1_subset = get_miner_challenge(pool_v1, hotkey, round_num=1, n=10)
        memorized_ids = {s["id"] for s in v1_subset}

        # After refresh, miner gets v2 samples
        v2_subset = get_miner_challenge(pool_v2, hotkey, round_num=1, n=10)
        v2_ids = {s["id"] for s in v2_subset}

        # Memorized IDs are completely useless
        overlap = memorized_ids & v2_ids
        assert len(overlap) == 0, "Memorized IDs must not match refreshed dataset"


class TestAdversarialInjection:
    """Tests for inject_adversarial_samples."""

    def test_adversarial_injection_adds_samples(self):
        """inject_adversarial_samples adds n_adversarial samples to the list."""
        pool = _make_pool(10)
        result = inject_adversarial_samples(pool, round_num=1, n_adversarial=2)

        assert len(result) == 12  # 10 original + 2 adversarial

    def test_adversarial_samples_have_metadata(self):
        """Each injected adversarial sample has _is_adversarial flag and correct ID format."""
        pool = _make_pool(10)
        result = inject_adversarial_samples(pool, round_num=5, n_adversarial=3)

        adversarial = [s for s in result if s.get("_is_adversarial")]
        assert len(adversarial) == 3

        for _i, adv in enumerate(adversarial):
            assert adv["_is_adversarial"] is True
            assert adv["id"].startswith("adv_5_")  # adv_{round_num}_{i}


class TestDatasetRefreshDetection:
    """Tests for detect_dataset_refresh."""

    def test_dataset_refresh_detection(self):
        """Different versions return True, same versions return False."""
        assert detect_dataset_refresh("abc123", "def456") is True
        assert detect_dataset_refresh("abc123", "abc123") is False
        assert detect_dataset_refresh("", "") is False
        assert detect_dataset_refresh("v1", "v2") is True


class TestEntropyChallenge:
    """Tests for entropy-aware challenge selection (MAIN-04)."""

    def test_entropy_changes_ordering(self):
        """Same (pool, hotkey, round_num) with different entropy_seed produces different subset."""
        pool = _make_pool(30)
        hotkey = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"

        entropy_a = b"\x01" * 32
        entropy_b = b"\x02" * 32

        subset_a = get_miner_challenge(pool, hotkey, round_num=1, n=10, entropy_seed=entropy_a)
        subset_b = get_miner_challenge(pool, hotkey, round_num=1, n=10, entropy_seed=entropy_b)

        ids_a = [s["id"] for s in subset_a]
        ids_b = [s["id"] for s in subset_b]

        assert ids_a != ids_b, "Different entropy seeds must produce different orderings"

    def test_entropy_deterministic(self):
        """Same entropy_seed produces same subset (within-round determinism)."""
        pool = _make_pool(30)
        hotkey = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        entropy = b"\xab\xcd" * 16

        subset_1 = get_miner_challenge(pool, hotkey, round_num=1, n=10, entropy_seed=entropy)
        subset_2 = get_miner_challenge(pool, hotkey, round_num=1, n=10, entropy_seed=entropy)

        ids_1 = [s["id"] for s in subset_1]
        ids_2 = [s["id"] for s in subset_2]

        assert ids_1 == ids_2, "Same entropy_seed must produce identical results"

    def test_no_entropy_backward_compat(self):
        """entropy_seed=None produces same result as before (backward compat)."""
        pool = _make_pool(20)
        hotkey = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"

        # Without entropy_seed (default)
        subset_default = get_miner_challenge(pool, hotkey, round_num=5, n=10)
        # Explicitly passing None
        subset_none = get_miner_challenge(pool, hotkey, round_num=5, n=10, entropy_seed=None)

        ids_default = [s["id"] for s in subset_default]
        ids_none = [s["id"] for s in subset_none]

        assert ids_default == ids_none, "None entropy_seed must match default behavior"
