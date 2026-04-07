"""
Tests for EvaluationDataset.get_round_samples excluded_ids parameter.

Requirements: VHARD-01 (challenge rotation exclusion in sample selection)
"""

import json

import pytest

from antigence_subnet.validator.evaluation import EvaluationDataset


@pytest.fixture
def eval_dataset(tmp_path):
    """Create a minimal evaluation dataset for testing."""
    domain = "hallucination"
    domain_dir = tmp_path / domain
    domain_dir.mkdir()

    # 20 regular samples + 5 honeypots
    samples = []
    manifest = {}
    for i in range(20):
        sid = f"reg_{i:03d}"
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

    for i in range(5):
        sid = f"hp_{i:03d}"
        samples.append({
            "id": sid,
            "prompt": f"Honeypot prompt {i}",
            "output": f"Honeypot output {i}",
            "domain": domain,
        })
        manifest[sid] = {
            "ground_truth_label": "anomalous",
            "is_honeypot": True,
        }

    (domain_dir / "samples.json").write_text(json.dumps({"samples": samples}))
    (domain_dir / "manifest.json").write_text(json.dumps(manifest))

    return EvaluationDataset(data_dir=tmp_path, domain=domain)


class TestGetRoundSamplesExclusion:
    """Tests for excluded_ids parameter in get_round_samples."""

    def test_no_exclusion_backward_compat(self, eval_dataset):
        """Without excluded_ids, behavior matches existing implementation."""
        result = eval_dataset.get_round_samples(round_num=1, n=10, n_honeypots=2)
        assert len(result) == 10

    def test_empty_exclusion_same_as_none(self, eval_dataset):
        """excluded_ids=set() produces same results as None."""
        result_none = eval_dataset.get_round_samples(
            round_num=1, n=10, n_honeypots=2
        )
        result_empty = eval_dataset.get_round_samples(
            round_num=1, n=10, n_honeypots=2, excluded_ids=set()
        )
        # Same round, same seed, should produce identical results
        ids_none = {s["id"] for s in result_none}
        ids_empty = {s["id"] for s in result_empty}
        assert ids_none == ids_empty

    def test_excluded_ids_not_in_result(self, eval_dataset):
        """Samples in excluded_ids are never returned."""
        # Get the first round's results to know what IDs exist
        first_round = eval_dataset.get_round_samples(
            round_num=1, n=10, n_honeypots=2
        )
        first_ids = {s["id"] for s in first_round}

        # Exclude those IDs from next selection
        result = eval_dataset.get_round_samples(
            round_num=2, n=10, n_honeypots=2, excluded_ids=first_ids
        )
        result_ids = {s["id"] for s in result}

        # No overlap with excluded IDs
        overlap = result_ids & first_ids
        assert len(overlap) == 0, f"Excluded IDs appeared in result: {overlap}"

    def test_excluded_honeypots_respected(self, eval_dataset):
        """Honeypot exclusion works alongside regular sample exclusion."""
        # Exclude all honeypots
        honeypot_ids = {f"hp_{i:03d}" for i in range(5)}
        result = eval_dataset.get_round_samples(
            round_num=1, n=10, n_honeypots=2, excluded_ids=honeypot_ids
        )
        result_ids = {s["id"] for s in result}
        # No honeypots should appear
        assert not (result_ids & honeypot_ids)

    def test_small_pool_after_exclusion(self, eval_dataset):
        """When exclusions reduce pool below requested count, returns available samples."""
        # Exclude 18 of 20 regular samples, leaving only 2
        exclude = {f"reg_{i:03d}" for i in range(18)}
        result = eval_dataset.get_round_samples(
            round_num=1, n=10, n_honeypots=2, excluded_ids=exclude
        )
        # Should get at most 2 regular + up to 5 honeypots (but requested 2)
        # = up to 4 total, but may get fewer if pool too small
        assert len(result) <= 10
        # Verify excluded IDs not present
        result_ids = {s["id"] for s in result}
        assert not (result_ids & exclude)
