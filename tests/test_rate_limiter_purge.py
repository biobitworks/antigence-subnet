"""Tests for RateLimiter.purge_expired() -- periodic stale entry cleanup (PROD-03).

Covers:
- purge_expired() removes callers with all timestamps older than window_seconds
- purge_expired() retains callers with at least one unexpired timestamp
- purge_expired() returns the count of removed callers
- check() auto-triggers purge_expired() when purge_interval has elapsed
- check() does NOT trigger purge_expired() before purge_interval elapses
- After purge, callers dict has fewer entries than before
"""

from unittest.mock import patch

from antigence_subnet.api.trust_score import RateLimiter


class TestPurgeExpiredRemovesStalCallers:
    """purge_expired() removes callers with all timestamps older than window."""

    def test_purge_removes_all_expired_callers(self):
        """Add 3 callers, advance time past window, purge removes all 3."""
        rl = RateLimiter(max_requests=60, window_seconds=60.0, purge_interval=300.0)

        # Inject timestamps at t=100 for 3 callers
        rl._requests = {
            "caller-a": [100.0],
            "caller-b": [100.0, 105.0],
            "caller-c": [100.0],
        }
        rl._last_purge = 100.0

        # At t=200, all timestamps are older than 60s window (cutoff=140)
        removed = rl.purge_expired(now=200.0)
        assert removed == 3
        assert len(rl._requests) == 0


class TestPurgeExpiredRetainsActiveCallers:
    """purge_expired() retains callers with at least one unexpired timestamp."""

    def test_purge_retains_caller_with_recent_timestamp(self):
        """One caller expired, one has a recent timestamp -- only 1 removed."""
        rl = RateLimiter(max_requests=60, window_seconds=60.0, purge_interval=300.0)

        rl._requests = {
            "stale-caller": [100.0],
            "active-caller": [100.0, 155.0],  # 155 > 200-60=140
        }
        rl._last_purge = 100.0

        removed = rl.purge_expired(now=200.0)
        assert removed == 1
        assert "stale-caller" not in rl._requests
        assert "active-caller" in rl._requests


class TestPurgeExpiredReturnsCount:
    """purge_expired() returns the count of removed callers."""

    def test_purge_returns_zero_when_nothing_expired(self):
        rl = RateLimiter(max_requests=60, window_seconds=60.0, purge_interval=300.0)
        rl._requests = {
            "caller-a": [195.0],
            "caller-b": [198.0],
        }
        rl._last_purge = 190.0

        removed = rl.purge_expired(now=200.0)
        assert removed == 0
        assert len(rl._requests) == 2


class TestCheckAutoTriggersPurge:
    """check() auto-triggers purge_expired() when purge_interval has elapsed."""

    def test_check_triggers_purge_after_interval(self):
        rl = RateLimiter(max_requests=60, window_seconds=60.0, purge_interval=10.0)

        # Simulate: stale caller added at t=100
        rl._requests = {"stale-caller": [100.0]}
        rl._last_purge = 100.0

        # At t=200, purge_interval (10s) has elapsed; window (60s) makes stale
        with patch("time.monotonic", return_value=200.0):
            result = rl.check("new-caller")

        assert result is True
        # Stale caller should have been purged
        assert "stale-caller" not in rl._requests
        # New caller should be present
        assert "new-caller" in rl._requests


class TestCheckDoesNotPurgeBeforeInterval:
    """check() does NOT trigger purge_expired() before purge_interval elapses."""

    def test_check_skips_purge_before_interval(self):
        rl = RateLimiter(max_requests=60, window_seconds=60.0, purge_interval=100.0)

        # Stale caller at t=100
        rl._requests = {"stale-caller": [100.0]}
        rl._last_purge = 100.0

        # At t=105, only 5s elapsed (< purge_interval=100s)
        # But the stale caller's timestamps are NOT expired yet (105-60=45 < 100)
        # Use t=200 where timestamps are expired BUT purge_interval hasn't elapsed
        # since _last_purge was at 100, and purge_interval is 100
        # At t=150: 50s < 100s purge_interval, timestamps expired (150-60=90 < 100)
        rl._requests = {"stale-caller": [50.0]}  # Expired at t=150 (cutoff=90)
        rl._last_purge = 140.0  # Only 10s ago at t=150

        with patch("time.monotonic", return_value=150.0):
            rl.check("new-caller")

        # Stale caller should NOT have been purged (interval not elapsed)
        assert "stale-caller" in rl._requests


class TestPurgeReducesDictSize:
    """After purge, callers dict has fewer entries than before."""

    def test_dict_size_decreases_after_purge(self):
        rl = RateLimiter(max_requests=60, window_seconds=60.0, purge_interval=300.0)

        rl._requests = {
            f"caller-{i}": [100.0] for i in range(50)
        }
        rl._last_purge = 100.0

        size_before = len(rl._requests)
        assert size_before == 50

        rl.purge_expired(now=200.0)

        size_after = len(rl._requests)
        assert size_after < size_before
        assert size_after == 0
