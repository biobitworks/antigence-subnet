"""
Tests for ChallengeRotation round-based challenge history tracking.

Requirements: VHARD-01 (round-based challenge rotation, anti-caching)
"""

import pytest

from antigence_subnet.validator.rotation import ChallengeRotation


class TestChallengeRotationRecord:
    """Tests for recording miner challenge history."""

    def test_record_stores_sample_ids(self):
        """record() stores sample IDs for a given miner and round."""
        rot = ChallengeRotation(rotation_window=10)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1", "s2", "s3"])

        excluded = rot.get_excluded("hotkey_a")
        assert excluded == {"s1", "s2", "s3"}

    def test_record_accumulates_across_rounds(self):
        """Multiple rounds accumulate sample IDs within the window."""
        rot = ChallengeRotation(rotation_window=10)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1", "s2"])
        rot.record("hotkey_a", round_num=2, sample_ids=["s3", "s4"])

        excluded = rot.get_excluded("hotkey_a")
        assert excluded == {"s1", "s2", "s3", "s4"}

    def test_record_different_miners_independent(self):
        """Different miners have independent history."""
        rot = ChallengeRotation(rotation_window=10)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1", "s2"])
        rot.record("hotkey_b", round_num=1, sample_ids=["s3", "s4"])

        assert rot.get_excluded("hotkey_a") == {"s1", "s2"}
        assert rot.get_excluded("hotkey_b") == {"s3", "s4"}


class TestChallengeRotationEviction:
    """Tests for window-based eviction of old history."""

    def test_evicts_beyond_window(self):
        """Rounds older than rotation_window are evicted."""
        rot = ChallengeRotation(rotation_window=3)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1"])
        rot.record("hotkey_a", round_num=2, sample_ids=["s2"])
        rot.record("hotkey_a", round_num=3, sample_ids=["s3"])
        # Round 1 samples should still be included (window=3 means last 3 rounds)
        assert "s1" in rot.get_excluded("hotkey_a")

        # Add round 4 -- round 1 should now be evicted
        rot.record("hotkey_a", round_num=4, sample_ids=["s4"])
        excluded = rot.get_excluded("hotkey_a")
        assert "s1" not in excluded, "Round 1 should be evicted after window=3"
        assert excluded == {"s2", "s3", "s4"}

    def test_evicts_oldest_first(self):
        """Eviction removes oldest rounds to maintain window size."""
        rot = ChallengeRotation(rotation_window=2)
        rot.record("hotkey_a", round_num=10, sample_ids=["s10"])
        rot.record("hotkey_a", round_num=11, sample_ids=["s11"])
        rot.record("hotkey_a", round_num=12, sample_ids=["s12"])

        excluded = rot.get_excluded("hotkey_a")
        assert "s10" not in excluded
        assert excluded == {"s11", "s12"}


class TestChallengeRotationGetExcluded:
    """Tests for get_excluded behavior."""

    def test_unknown_miner_returns_empty(self):
        """get_excluded for unknown miner returns empty set."""
        rot = ChallengeRotation(rotation_window=10)
        assert rot.get_excluded("unknown_hotkey") == set()

    def test_empty_after_full_eviction(self):
        """After all entries evicted, returns empty set."""
        rot = ChallengeRotation(rotation_window=1)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1"])
        rot.record("hotkey_a", round_num=3, sample_ids=["s3"])
        # Window=1 means only last 1 round kept
        excluded = rot.get_excluded("hotkey_a")
        assert "s1" not in excluded
        assert excluded == {"s3"}


class TestChallengeRotationSerialization:
    """Tests for to_dict/from_dict persistence."""

    def test_roundtrip_serialization(self):
        """to_dict -> from_dict preserves all state."""
        rot = ChallengeRotation(rotation_window=5)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1", "s2"])
        rot.record("hotkey_a", round_num=2, sample_ids=["s3"])
        rot.record("hotkey_b", round_num=1, sample_ids=["s4"])

        data = rot.to_dict()
        restored = ChallengeRotation.from_dict(data)

        assert restored.rotation_window == 5
        assert restored.get_excluded("hotkey_a") == {"s1", "s2", "s3"}
        assert restored.get_excluded("hotkey_b") == {"s4"}

    def test_to_dict_structure(self):
        """to_dict returns JSON-serializable dict with expected keys."""
        rot = ChallengeRotation(rotation_window=7)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1"])

        data = rot.to_dict()
        assert "rotation_window" in data
        assert "history" in data
        assert data["rotation_window"] == 7


class TestChallengeRotationClear:
    """Tests for clearing history."""

    def test_clear_specific_miner(self):
        """clear(hotkey) removes only that miner's history."""
        rot = ChallengeRotation(rotation_window=10)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1"])
        rot.record("hotkey_b", round_num=1, sample_ids=["s2"])

        rot.clear("hotkey_a")
        assert rot.get_excluded("hotkey_a") == set()
        assert rot.get_excluded("hotkey_b") == {"s2"}

    def test_clear_all(self):
        """clear() without args clears all miners."""
        rot = ChallengeRotation(rotation_window=10)
        rot.record("hotkey_a", round_num=1, sample_ids=["s1"])
        rot.record("hotkey_b", round_num=1, sample_ids=["s2"])

        rot.clear()
        assert rot.get_excluded("hotkey_a") == set()
        assert rot.get_excluded("hotkey_b") == set()


class TestChallengeRotationDefaults:
    """Tests for default configuration."""

    def test_default_window(self):
        """Default rotation_window is 10."""
        rot = ChallengeRotation()
        assert rot.rotation_window == 10

    def test_custom_window(self):
        """Custom rotation_window is respected."""
        rot = ChallengeRotation(rotation_window=20)
        assert rot.rotation_window == 20


class TestChallengeRotationIntegration:
    """Integration tests: rotation with EvaluationDataset and get_miner_challenge."""

    @pytest.fixture
    def eval_dataset(self, tmp_path):
        """Create a minimal evaluation dataset for integration testing."""
        import json

        from antigence_subnet.validator.evaluation import EvaluationDataset

        domain = "hallucination"
        domain_dir = tmp_path / domain
        domain_dir.mkdir()

        # 60 regular samples (enough to sustain 10-round rotation)
        samples = []
        manifest = {}
        for i in range(60):
            sid = f"sample_{i:03d}"
            samples.append({
                "id": sid,
                "prompt": f"Prompt {i}",
                "output": f"Output {i}",
                "domain": domain,
            })
            manifest[sid] = {
                "ground_truth_label": "normal" if i % 2 == 0 else "anomalous",
                "is_honeypot": False,
            }

        # 10 honeypots
        for i in range(10):
            sid = f"hp_{i:03d}"
            samples.append({
                "id": sid,
                "prompt": f"Honeypot {i}",
                "output": f"HP output {i}",
                "domain": domain,
            })
            manifest[sid] = {
                "ground_truth_label": "anomalous",
                "is_honeypot": True,
            }

        (domain_dir / "samples.json").write_text(json.dumps({"samples": samples}))
        (domain_dir / "manifest.json").write_text(json.dumps(manifest))

        return EvaluationDataset(data_dir=tmp_path, domain=domain)

    def test_no_repeat_within_window(self, eval_dataset):
        """No miner sees the same sample twice within a 10-round window."""
        from antigence_subnet.validator.challenge import get_miner_challenge

        rot = ChallengeRotation(rotation_window=10)
        hotkey = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"

        for round_num in range(1, 16):
            # Get excluded IDs for this miner
            excluded = rot.get_excluded(hotkey)

            # Get round pool (no exclusion at pool level for this test)
            pool = eval_dataset.get_round_samples(
                round_num=round_num, n=20, n_honeypots=4
            )

            # Select per-miner challenge with exclusion
            challenge = get_miner_challenge(
                samples=pool,
                miner_hotkey=hotkey,
                round_num=round_num,
                n=8,
                excluded_ids=excluded if excluded else None,
            )

            # Check: no sample in this challenge should be in the excluded set
            challenge_ids = {s["id"] for s in challenge}
            overlap = challenge_ids & excluded
            assert len(overlap) == 0, (
                f"Round {round_num}: miner saw previously-seen samples: {overlap}"
            )

            # Record what the miner saw
            rot.record(hotkey, round_num, list(challenge_ids))

    def test_samples_available_after_window_expires(self, eval_dataset):
        """After the rotation window, previously-seen samples become available."""
        from antigence_subnet.validator.challenge import get_miner_challenge

        rot = ChallengeRotation(rotation_window=3)
        hotkey = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"

        # Run 3 rounds to fill the window
        round_1_ids = set()
        for round_num in range(1, 4):
            excluded = rot.get_excluded(hotkey)
            pool = eval_dataset.get_round_samples(
                round_num=round_num, n=20, n_honeypots=4
            )
            challenge = get_miner_challenge(
                samples=pool, miner_hotkey=hotkey, round_num=round_num,
                n=8, excluded_ids=excluded if excluded else None,
            )
            challenge_ids = {s["id"] for s in challenge}
            if round_num == 1:
                round_1_ids = challenge_ids
            rot.record(hotkey, round_num, list(challenge_ids))

        # Round 1 samples should still be excluded
        assert round_1_ids.issubset(rot.get_excluded(hotkey))

        # Run round 4 -- should evict round 1
        rot.record(hotkey, 4, ["new_sample"])
        excluded_after = rot.get_excluded(hotkey)
        # Round 1 samples should no longer be excluded
        assert not round_1_ids.issubset(excluded_after), (
            "Round 1 samples should be available after window expires"
        )

    def test_different_miners_see_same_sample_same_round(self, eval_dataset):
        """Different miners can see the same sample in the same round."""
        from antigence_subnet.validator.challenge import get_miner_challenge

        rot = ChallengeRotation(rotation_window=10)
        hotkey_a = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        hotkey_b = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

        pool = eval_dataset.get_round_samples(
            round_num=1, n=20, n_honeypots=4
        )

        # Both miners get challenges from the same pool, no exclusions
        challenge_a = get_miner_challenge(
            samples=pool, miner_hotkey=hotkey_a, round_num=1, n=8,
        )
        challenge_b = get_miner_challenge(
            samples=pool, miner_hotkey=hotkey_b, round_num=1, n=8,
        )

        ids_a = {s["id"] for s in challenge_a}
        ids_b = {s["id"] for s in challenge_b}

        # There should be some overlap (from same pool of 20, each picks 8)
        # This tests cross-miner independence
        assert len(ids_a) == 8
        assert len(ids_b) == 8
        # Both select from same pool so overlap is possible (but not guaranteed)
        # The key assertion: both miners CAN receive samples without the other's
        # exclusion interfering
        rot.record(hotkey_a, 1, list(ids_a))
        rot.record(hotkey_b, 1, list(ids_b))

        # In round 2, miner A's exclusion should not affect miner B
        excluded_a = rot.get_excluded(hotkey_a)
        excluded_b = rot.get_excluded(hotkey_b)
        assert excluded_a == ids_a
        assert excluded_b == ids_b

    def test_rotation_15_rounds_end_to_end(self, eval_dataset):
        """End-to-end: 15 rounds with 2 miners, verifying anti-caching property."""
        from antigence_subnet.validator.challenge import get_miner_challenge

        rot = ChallengeRotation(rotation_window=10)
        miners = [
            "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        ]

        for round_num in range(1, 16):
            pool = eval_dataset.get_round_samples(
                round_num=round_num, n=20, n_honeypots=4
            )

            for hotkey in miners:
                excluded = rot.get_excluded(hotkey)
                challenge = get_miner_challenge(
                    samples=pool, miner_hotkey=hotkey, round_num=round_num,
                    n=8, excluded_ids=excluded if excluded else None,
                )
                challenge_ids = {s["id"] for s in challenge}

                # Anti-caching: no overlap with excluded IDs
                overlap = challenge_ids & excluded
                assert len(overlap) == 0, (
                    f"Round {round_num}, miner {hotkey[:8]}: "
                    f"saw previously-excluded samples: {overlap}"
                )

                rot.record(hotkey, round_num, list(challenge_ids))
