"""Tests for VerificationSynapse protocol definition (PROTO-01)."""

import pytest
from pydantic import ValidationError

from antigence_subnet.protocol import (
    DOMAIN_BIO,
    DOMAIN_CODE_SECURITY,
    DOMAIN_HALLUCINATION,
    DOMAIN_REASONING,
    KNOWN_DOMAINS,
    VerificationSynapse,
)


class TestSynapseInstantiation:
    """Test VerificationSynapse construction."""

    def test_synapse_instantiation(self):
        """VerificationSynapse can be instantiated with required fields."""
        synapse = VerificationSynapse(
            prompt="test", output="test", domain="hallucination"
        )
        assert synapse.prompt == "test"
        assert synapse.output == "test"
        assert synapse.domain == "hallucination"

    def test_synapse_required_fields(self):
        """Constructing without required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            VerificationSynapse()  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            VerificationSynapse(prompt="test")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            VerificationSynapse(prompt="test", output="test")  # type: ignore[call-arg]

    def test_response_fields_optional(self):
        """Synapse with only required fields has None for all response fields."""
        synapse = VerificationSynapse(
            prompt="test", output="test", domain="hallucination"
        )
        assert synapse.anomaly_score is None
        assert synapse.confidence is None
        assert synapse.anomaly_type is None
        assert synapse.feature_attribution is None


class TestSynapseSerializationRoundtrip:
    """Test serialization and deserialization."""

    def test_synapse_serialization_roundtrip(self, sample_synapse_with_response):
        """model_dump() and reconstruction produce identical fields."""
        dump = sample_synapse_with_response.model_dump()
        restored = VerificationSynapse(**dump)
        assert restored.prompt == sample_synapse_with_response.prompt
        assert restored.output == sample_synapse_with_response.output
        assert restored.domain == sample_synapse_with_response.domain
        assert restored.anomaly_score == sample_synapse_with_response.anomaly_score
        assert restored.confidence == sample_synapse_with_response.confidence
        assert restored.anomaly_type == sample_synapse_with_response.anomaly_type
        assert (
            restored.feature_attribution
            == sample_synapse_with_response.feature_attribution
        )

    def test_feature_attribution_dict(self):
        """Dict[str, float] is accepted and survives round-trip."""
        attrs = {"feature_a": 0.9, "feature_b": 0.1}
        synapse = VerificationSynapse(
            prompt="test",
            output="test",
            domain="hallucination",
            feature_attribution=attrs,
        )
        dump = synapse.model_dump()
        restored = VerificationSynapse(**dump)
        assert restored.feature_attribution == attrs


class TestAnomalyScoreRange:
    """Test anomaly_score field validation."""

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_anomaly_score_range_valid(self, score):
        """Scores 0.0, 0.5, 1.0 are accepted."""
        synapse = VerificationSynapse(
            prompt="test",
            output="test",
            domain="hallucination",
            anomaly_score=score,
        )
        assert synapse.anomaly_score == score

    @pytest.mark.parametrize("score", [-0.1, 1.1, -1.0, 2.0])
    def test_anomaly_score_range_invalid(self, score):
        """Scores outside 0.0-1.0 raise ValidationError."""
        with pytest.raises(ValidationError):
            VerificationSynapse(
                prompt="test",
                output="test",
                domain="hallucination",
                anomaly_score=score,
            )


class TestConfidenceRange:
    """Test confidence field validation."""

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_confidence_range_valid(self, conf):
        """Confidence values 0.0, 0.5, 1.0 are accepted."""
        synapse = VerificationSynapse(
            prompt="test",
            output="test",
            domain="hallucination",
            confidence=conf,
        )
        assert synapse.confidence == conf

    @pytest.mark.parametrize("conf", [-0.1, 1.1])
    def test_confidence_range_invalid(self, conf):
        """Confidence values outside 0.0-1.0 raise ValidationError."""
        with pytest.raises(ValidationError):
            VerificationSynapse(
                prompt="test",
                output="test",
                domain="hallucination",
                confidence=conf,
            )


class TestDomainConstants:
    """Test domain constant definitions."""

    def test_domain_constants_defined(self):
        """KNOWN_DOMAINS contains exactly the four expected domains."""
        assert frozenset(
            {"hallucination", "code_security", "reasoning", "bio"}
        ) == KNOWN_DOMAINS
        assert DOMAIN_HALLUCINATION == "hallucination"
        assert DOMAIN_CODE_SECURITY == "code_security"
        assert DOMAIN_REASONING == "reasoning"
        assert DOMAIN_BIO == "bio"


class TestRequiredHashFields:
    """Test required_hash_fields ClassVar."""

    def test_required_hash_fields(self):
        """required_hash_fields is the expected tuple."""
        assert VerificationSynapse.required_hash_fields == (
            "prompt",
            "output",
            "domain",
        )


class TestDeserialize:
    """Test deserialize() method."""

    def test_deserialize_returns_anomaly_score(self):
        """Synapse with anomaly_score=0.8 returns 0.8 from deserialize()."""
        synapse = VerificationSynapse(
            prompt="test",
            output="test",
            domain="hallucination",
            anomaly_score=0.8,
        )
        assert synapse.deserialize() == 0.8

    def test_deserialize_returns_none_when_unset(self):
        """Synapse without response returns None from deserialize()."""
        synapse = VerificationSynapse(
            prompt="test", output="test", domain="hallucination"
        )
        assert synapse.deserialize() is None


class TestContextField:
    """Test optional context field (D-04)."""

    def test_context_field_optional(self):
        """context=None accepted, context with value accepted."""
        synapse_none = VerificationSynapse(
            prompt="test", output="test", domain="hallucination", context=None
        )
        assert synapse_none.context is None

        synapse_with = VerificationSynapse(
            prompt="test",
            output="test",
            domain="hallucination",
            context='{"key": "value"}',
        )
        assert synapse_with.context == '{"key": "value"}'


class TestSeedField:
    """Test optional seed hint contract (NONDET-03)."""

    @pytest.mark.parametrize("seed", [None, 12345])
    def test_seed_field_accepts_none_and_integer_without_changing_required_fields(
        self, seed
    ):
        """Seed remains optional and does not change the base request contract."""
        synapse = VerificationSynapse(
            prompt="test",
            output="test",
            domain="hallucination",
            seed=seed,
        )

        assert synapse.prompt == "test"
        assert synapse.output == "test"
        assert synapse.domain == "hallucination"
        assert synapse.seed == seed
        assert synapse.required_hash_fields == ("prompt", "output", "domain")
        assert synapse.anomaly_score is None
        assert synapse.confidence is None
        assert synapse.anomaly_type is None
        assert synapse.feature_attribution is None

    def test_seed_round_trip_preserves_best_effort_hint_when_present(self):
        """model_dump() round-trips the optional seed hint when explicitly set."""
        synapse = VerificationSynapse(
            prompt="prompt",
            output="output",
            domain="reasoning",
            seed=4242,
        )

        dump = synapse.model_dump(exclude_none=True)
        restored = VerificationSynapse(**dump)

        assert dump["seed"] == 4242
        assert restored.seed == 4242
        assert restored.prompt == synapse.prompt
        assert restored.output == synapse.output
        assert restored.domain == synapse.domain

    def test_seed_absence_preserves_behavior_in_model_dump(self):
        """Absence preserves behavior: no seed key appears when the hint is omitted."""
        synapse = VerificationSynapse(
            prompt="prompt",
            output="output",
            domain="bio",
        )

        dump = synapse.model_dump(exclude_none=True)

        assert "seed" not in dump
        assert dump["prompt"] == "prompt"
        assert dump["output"] == "output"
        assert dump["domain"] == "bio"
