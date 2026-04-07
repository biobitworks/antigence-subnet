"""
Tests for perturbation stability engine.

Requirements: CHEAT-04 (perturbation stability testing), MAIN-04 (challenge randomization)
"""


from antigence_subnet.validator.perturbation import (
    SYNONYM_MAP,
    compute_stability_bonus,
    generate_perturbation_variants,
    perturb_text,
)


class TestPerturbText:
    """Tests for perturb_text deterministic text mutations."""

    def test_perturb_text_deterministic(self):
        """Same text + same seed produces identical output every time."""
        text = "The model is able to detect anomalies."
        result1 = perturb_text(text, seed=42)
        result2 = perturb_text(text, seed=42)

        assert result1 == result2

    def test_perturb_text_synonym(self):
        """Seed selecting synonym mutation produces word changes."""
        # seed % 3 == 0 -> synonym mutation
        text = "The model is able to detect anomalies."
        result = perturb_text(text, seed=0)

        # Should differ from original (synonym substitution applied)
        assert result != text, f"Synonym mutation should change the text, got: {result}"

    def test_perturb_text_whitespace(self):
        """Seed selecting whitespace mutation modifies spacing."""
        # seed % 3 == 1 -> whitespace mutation
        text = "The model is able to detect anomalies."
        result = perturb_text(text, seed=1)

        assert result != text, f"Whitespace mutation should change the text, got: {result}"

    def test_perturb_text_casing(self):
        """Seed selecting casing mutation uppercases some words."""
        # seed % 3 == 2 -> casing mutation
        text = "The model is able to detect anomalies."
        result = perturb_text(text, seed=2)

        assert result != text, f"Casing mutation should change the text, got: {result}"
        # At least one word should have different casing
        orig_words = text.split()
        result_words = result.split()
        has_case_diff = any(
            o != r for o, r in zip(orig_words, result_words, strict=False)
        )
        assert has_case_diff, "At least one word should have changed casing"


class TestGenerateVariants:
    """Tests for generate_perturbation_variants."""

    def test_generate_variants_count(self):
        """n_variants=3 produces exactly 3 variants."""
        sample = {
            "id": "sample_001",
            "prompt": "What is the capital?",
            "output": "The capital is Berlin.",
            "domain": "hallucination",
        }
        variants = generate_perturbation_variants(sample, round_num=1, n_variants=3)

        assert len(variants) == 3

    def test_generate_variants_metadata(self):
        """Each variant has _is_perturbation=True and _original_id."""
        sample = {
            "id": "sample_001",
            "prompt": "What is the capital?",
            "output": "The capital is Berlin.",
            "domain": "hallucination",
        }
        variants = generate_perturbation_variants(sample, round_num=1, n_variants=2)

        for i, v in enumerate(variants):
            assert v["_is_perturbation"] is True
            assert v["_original_id"] == "sample_001"
            assert v["id"] == f"sample_001_perturb_{i}"

    def test_generate_variants_only_output_changed(self):
        """Prompt and domain fields remain unchanged in variants."""
        sample = {
            "id": "sample_001",
            "prompt": "What is the capital?",
            "output": "The capital is Berlin.",
            "domain": "hallucination",
        }
        variants = generate_perturbation_variants(sample, round_num=1, n_variants=2)

        for v in variants:
            assert v["prompt"] == sample["prompt"]
            assert v["domain"] == sample["domain"]


class TestStabilityBonus:
    """Tests for compute_stability_bonus."""

    def test_stability_bonus_identical_scores(self):
        """Identical scores return perfect stability bonus of 1.0."""
        assert compute_stability_bonus([0.5, 0.5, 0.5]) == 1.0

    def test_stability_bonus_varying_scores(self):
        """Wildly varying scores return a low bonus (< 0.7)."""
        bonus = compute_stability_bonus([0.1, 0.9])
        assert bonus < 0.7, f"Expected < 0.7, got {bonus}"

    def test_stability_bonus_single_score(self):
        """Single score returns 1.0 (cannot measure instability)."""
        assert compute_stability_bonus([0.5]) == 1.0

    def test_stability_bonus_empty(self):
        """Empty scores returns 1.0."""
        assert compute_stability_bonus([]) == 1.0


class TestSynonymMapExpansion:
    """Tests for expanded synonym map (MAIN-04)."""

    def test_synonym_map_has_24_plus_entries(self):
        """SYNONYM_MAP must have at least 24 entries (3x expansion from 8)."""
        assert len(SYNONYM_MAP) >= 24, f"Expected >= 24 entries, got {len(SYNONYM_MAP)}"

    def test_synonym_map_has_original_entries(self):
        """Original 8 entries must still be present."""
        original_keys = {"the", "is", "was", "are", "has", "have", "will", "can"}
        for key in original_keys:
            assert key in SYNONYM_MAP, f"Original entry '{key}' missing from SYNONYM_MAP"

    def test_synonym_map_values_are_lists(self):
        """Every SYNONYM_MAP value must be a non-empty list of strings."""
        for key, synonyms in SYNONYM_MAP.items():
            assert isinstance(synonyms, list), f"SYNONYM_MAP['{key}'] is not a list"
            assert len(synonyms) > 0, f"SYNONYM_MAP['{key}'] is empty"
            for syn in synonyms:
                assert isinstance(syn, str), f"SYNONYM_MAP['{key}'] contains non-string: {syn}"


class TestComposableMutations:
    """Tests for new mutation types and composable mutation application (MAIN-04)."""

    def test_insertion_mutation_adds_filler_word(self):
        """_insertion_mutation inserts a filler word (very, quite, rather, indeed, actually)."""
        import random

        from antigence_subnet.validator.perturbation import _insertion_mutation

        text = "The important result shows clear evidence"
        rng = random.Random(42)
        result = _insertion_mutation(text, rng)

        assert result != text, f"Insertion mutation must change text, got: {result}"
        # Result should have more words than original
        assert len(result.split()) > len(text.split()), "Insertion should add words"

    def test_deletion_mutation_removes_word(self):
        """_deletion_mutation removes a non-critical word from DELETABLE_WORDS set."""
        import random

        from antigence_subnet.validator.perturbation import _deletion_mutation

        text = "The result is very important and quite significant"
        rng = random.Random(42)
        result = _deletion_mutation(text, rng)

        assert result != text, f"Deletion mutation must change text, got: {result}"
        # Result should have fewer words than original
        assert len(result.split()) < len(text.split()), "Deletion should remove words"

    def test_deletion_mutation_no_eligible_words(self):
        """_deletion_mutation returns text unchanged if no deletable words exist."""
        import random

        from antigence_subnet.validator.perturbation import _deletion_mutation

        text = "The cat sat on mat"
        rng = random.Random(42)
        result = _deletion_mutation(text, rng)

        assert result == text, "No eligible words -> text should be unchanged"

    def test_reorder_mutation_swaps_clauses(self):
        """_reorder_mutation swaps clauses around 'and' or 'or'."""
        import random

        from antigence_subnet.validator.perturbation import _reorder_mutation

        text = "The model detects anomalies and the system reports them"
        rng = random.Random(42)
        result = _reorder_mutation(text, rng)

        assert result != text, f"Reorder mutation must change text, got: {result}"

    def test_composable_mutations_apply_multiple(self):
        """With entropy_seed, perturb_text applies 1-3 mutation types (composable)."""
        text = "The model is very important and has significant results"
        # Different entropy seeds should potentially apply different numbers of mutations
        results = set()
        for entropy in range(20):
            result = perturb_text(text, seed=42, entropy_seed=entropy)
            results.add(result)

        # With composable mutations and different entropy seeds, we should get variety
        assert len(results) > 1, "Different entropy seeds must produce different results"


class TestEntropyIntegration:
    """Tests for entropy parameter in perturbation functions (MAIN-04)."""

    def test_perturb_text_deterministic_with_entropy(self):
        """Same (text, seed, entropy_seed) produces identical output."""
        text = "The model is able to detect anomalies."
        r1 = perturb_text(text, seed=42, entropy_seed=12345)
        r2 = perturb_text(text, seed=42, entropy_seed=12345)
        assert r1 == r2, "Same entropy_seed must be deterministic"

    def test_perturb_text_different_entropy_different_output(self):
        """Same (text, seed) but different entropy_seed produces different output."""
        text = "The model is able to detect anomalies and has good results."
        r1 = perturb_text(text, seed=42, entropy_seed=100)
        r2 = perturb_text(text, seed=42, entropy_seed=200)
        assert r1 != r2, "Different entropy_seed must produce different results"

    def test_perturb_text_backward_compat_no_entropy(self):
        """perturb_text without entropy_seed still works (old behavior)."""
        text = "The model is able to detect anomalies."
        r1 = perturb_text(text, seed=42)
        r2 = perturb_text(text, seed=42)
        assert r1 == r2, "Without entropy_seed, old deterministic behavior must hold"

    def test_generate_variants_accepts_entropy_seed(self):
        """generate_perturbation_variants accepts entropy_seed parameter."""
        sample = {
            "id": "sample_001",
            "prompt": "What is the capital?",
            "output": "The capital is Berlin and has large buildings.",
            "domain": "hallucination",
        }
        variants = generate_perturbation_variants(
            sample, round_num=1, n_variants=3, entropy_seed=42,
        )
        assert len(variants) >= 1, "Must produce at least 1 variant"
        assert len(variants) <= 3, "Must produce at most 3 variants"

    def test_generate_variants_different_entropy_different_output(self):
        """Different entropy_seed produces different variants for same (sample, round_num)."""
        sample = {
            "id": "sample_001",
            "prompt": "What is the capital?",
            "output": "The capital is Berlin and has large important buildings.",
            "domain": "hallucination",
        }
        v1 = generate_perturbation_variants(
            sample, round_num=1, n_variants=3, entropy_seed=100,
        )
        v2 = generate_perturbation_variants(
            sample, round_num=1, n_variants=3, entropy_seed=200,
        )
        # Either count or content should differ
        outputs1 = [v["output"] for v in v1]
        outputs2 = [v["output"] for v in v2]
        assert outputs1 != outputs2 or len(v1) != len(v2), (
            "Different entropy seeds must produce different variants"
        )

    def test_generate_variants_n_variants_range(self):
        """With entropy_seed, n_variants can produce 1, 2, or 3 variants."""
        sample = {
            "id": "sample_001",
            "prompt": "Test",
            "output": "The model is able to detect anomalies and has good results.",
            "domain": "hallucination",
        }
        counts = set()
        for entropy in range(30):
            variants = generate_perturbation_variants(
                sample, round_num=1, n_variants=3, entropy_seed=entropy,
            )
            counts.add(len(variants))

        # With entropy_seed % 3 + 1, we should see all of 1, 2, 3
        assert counts == {1, 2, 3}, f"Expected variant counts {{1, 2, 3}}, got {counts}"

    def test_compute_stability_bonus_unchanged(self):
        """compute_stability_bonus behavior is identical to original (MUST NOT CHANGE)."""
        # This is a regression guard: 1 - std(scores), clamped [0, 1]
        import numpy as np

        # Identical scores => 1.0
        assert compute_stability_bonus([0.5, 0.5, 0.5]) == 1.0
        # Single score => 1.0
        assert compute_stability_bonus([0.5]) == 1.0
        # Empty => 1.0
        assert compute_stability_bonus([]) == 1.0
        # Known variance case
        scores = [0.2, 0.8]
        expected = float(max(0.0, 1.0 - np.std(scores)))
        assert abs(compute_stability_bonus(scores) - expected) < 1e-9
