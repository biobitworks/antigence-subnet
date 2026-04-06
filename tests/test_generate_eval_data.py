"""Tests for evaluation data generation and validation scripts."""

import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts dir to path so we can import from it
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Task 1: Generator tests
# ---------------------------------------------------------------------------


class TestHallucinationGenerator:
    def test_returns_correct_count(self):
        from generate_eval_data import generate_hallucination_samples

        samples, manifest = generate_hallucination_samples(n=10, start_id=1, seed=42)
        assert len(samples) == 10

    def test_sample_schema(self):
        from generate_eval_data import generate_hallucination_samples

        samples, manifest = generate_hallucination_samples(n=10, start_id=1, seed=42)
        for s in samples:
            assert "id" in s
            assert "prompt" in s
            assert "output" in s
            assert "domain" in s
            assert s["domain"] == "hallucination"
            assert "metadata" in s

    def test_metadata_fields(self):
        from generate_eval_data import generate_hallucination_samples

        samples, manifest = generate_hallucination_samples(n=10, start_id=1, seed=42)
        for s in samples:
            assert s["metadata"]["difficulty"] in {"easy", "medium", "hard"}
            assert s["metadata"]["source"] == "template"

    def test_manifest_entries(self):
        from generate_eval_data import generate_hallucination_samples

        samples, manifest = generate_hallucination_samples(n=10, start_id=1, seed=42)
        for s in samples:
            assert s["id"] in manifest
            entry = manifest[s["id"]]
            assert entry["ground_truth_label"] in {"normal", "anomalous"}
            assert isinstance(entry["is_honeypot"], bool)
            if entry["ground_truth_label"] == "anomalous":
                assert entry["ground_truth_type"] in {
                    "factual_error",
                    "fabricated_citation",
                    "hallucinated_fact",
                    "unsupported_claim",
                }
            else:
                assert entry["ground_truth_type"] is None

    def test_no_duplicate_ids(self):
        from generate_eval_data import generate_hallucination_samples

        samples, manifest = generate_hallucination_samples(n=20, start_id=1, seed=42)
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids))


class TestCodeSecurityGenerator:
    def test_returns_correct_count(self):
        from generate_eval_data import generate_code_security_samples

        samples, manifest = generate_code_security_samples(n=10, start_id=1, seed=42)
        assert len(samples) == 10

    def test_sample_schema_has_code_field(self):
        from generate_eval_data import generate_code_security_samples

        samples, manifest = generate_code_security_samples(n=10, start_id=1, seed=42)
        for s in samples:
            assert "id" in s
            assert "prompt" in s
            assert "output" in s
            assert "code" in s
            assert "domain" in s
            assert s["domain"] == "code_security"
            assert "metadata" in s

    def test_metadata_fields(self):
        from generate_eval_data import generate_code_security_samples

        samples, manifest = generate_code_security_samples(n=10, start_id=1, seed=42)
        for s in samples:
            assert s["metadata"]["difficulty"] in {"easy", "medium", "hard"}
            assert s["metadata"]["source"] == "template"

    def test_manifest_ground_truth_types(self):
        from generate_eval_data import generate_code_security_samples

        cs_types = {
            "sql_injection", "xss", "command_injection", "path_traversal",
            "hardcoded_credentials", "insecure_deserialization", "ssrf", "xxe",
            "open_redirect", "race_condition", "redos", "mass_assignment",
            "weak_crypto", "code_injection", "adversarial_empty",
            "adversarial_long", "adversarial_unicode",
        }
        samples, manifest = generate_code_security_samples(n=30, start_id=1, seed=42)
        for s in samples:
            entry = manifest[s["id"]]
            if entry["ground_truth_label"] == "anomalous":
                assert entry["ground_truth_type"] in cs_types

    def test_no_duplicate_ids(self):
        from generate_eval_data import generate_code_security_samples

        samples, manifest = generate_code_security_samples(n=20, start_id=1, seed=42)
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids))


class TestReasoningGenerator:
    def test_returns_correct_count(self):
        from generate_eval_data import generate_reasoning_samples

        samples, manifest = generate_reasoning_samples(n=10, start_id=1, seed=42)
        assert len(samples) == 10

    def test_sample_schema(self):
        from generate_eval_data import generate_reasoning_samples

        samples, manifest = generate_reasoning_samples(n=10, start_id=1, seed=42)
        for s in samples:
            assert "id" in s
            assert "prompt" in s
            assert "output" in s
            assert "domain" in s
            assert s["domain"] == "reasoning"
            assert "metadata" in s

    def test_manifest_ground_truth_types(self):
        from generate_eval_data import generate_reasoning_samples

        rea_types = {
            "logical_contradiction", "non_sequitur", "constraint_violation",
            "circular_reasoning", "arithmetic_error", "equivocation",
            "unit_error", "cherry_picking", "interpretation_error",
            "straw_man", "logical_fallacy", "adversarial_empty",
            "adversarial_unicode", "adversarial_repetition",
        }
        samples, manifest = generate_reasoning_samples(n=30, start_id=1, seed=42)
        for s in samples:
            entry = manifest[s["id"]]
            if entry["ground_truth_label"] == "anomalous":
                assert entry["ground_truth_type"] in rea_types

    def test_no_duplicate_ids(self):
        from generate_eval_data import generate_reasoning_samples

        samples, manifest = generate_reasoning_samples(n=20, start_id=1, seed=42)
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids))


class TestBioGenerator:
    def test_returns_correct_count(self):
        from generate_eval_data import generate_bio_samples

        samples, manifest = generate_bio_samples(n=10, start_id=1, seed=42)
        assert len(samples) == 10

    def test_sample_schema(self):
        from generate_eval_data import generate_bio_samples

        samples, manifest = generate_bio_samples(n=10, start_id=1, seed=42)
        for s in samples:
            assert "id" in s
            assert "prompt" in s
            assert "output" in s
            assert "domain" in s
            assert s["domain"] == "bio"
            assert "metadata" in s

    def test_manifest_ground_truth_types(self):
        from generate_eval_data import generate_bio_samples

        bio_types = {
            "value_out_of_range", "statistical_anomaly", "physical_impossibility",
            "species_mismatch", "unit_inconsistency", "p_value_fabrication",
            "pipeline_error", "conservation_violation", "statistical_inconsistency",
            "underpowered_study", "logical_impossibility", "implausible_effect",
            "arithmetic_error", "wrong_test", "adversarial_empty",
            "adversarial_unicode", "adversarial_repetition",
        }
        samples, manifest = generate_bio_samples(n=30, start_id=1, seed=42)
        for s in samples:
            entry = manifest[s["id"]]
            if entry["ground_truth_label"] == "anomalous":
                assert entry["ground_truth_type"] in bio_types

    def test_no_duplicate_ids(self):
        from generate_eval_data import generate_bio_samples

        samples, manifest = generate_bio_samples(n=20, start_id=1, seed=42)
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids))


class TestClassBalance:
    """Verify 40-60% anomalous across all domains."""

    @pytest.mark.parametrize("domain", ["hallucination", "code_security", "reasoning", "bio"])
    def test_anomalous_ratio(self, domain):
        from generate_eval_data import GENERATORS

        gen_fn = GENERATORS[domain]
        samples, manifest = gen_fn(n=50, start_id=1, seed=42)
        anomalous = sum(
            1 for entry in manifest.values()
            if entry["ground_truth_label"] == "anomalous"
        )
        ratio = anomalous / len(samples)
        assert 0.40 <= ratio <= 0.60, f"{domain} anomalous ratio {ratio:.2f} outside 40-60%"


class TestDifficultyDistribution:
    """Verify difficulty is roughly 30/40/30 easy/medium/hard."""

    @pytest.mark.parametrize("domain", ["hallucination", "code_security", "reasoning", "bio"])
    def test_difficulty_spread(self, domain):
        from generate_eval_data import GENERATORS

        gen_fn = GENERATORS[domain]
        samples, _ = gen_fn(n=100, start_id=1, seed=42)
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for s in samples:
            counts[s["metadata"]["difficulty"]] += 1
        # Allow 15-45% for easy, 25-55% for medium, 15-45% for hard (generous tolerance)
        assert 15 <= counts["easy"] <= 45, f"easy={counts['easy']}"
        assert 25 <= counts["medium"] <= 55, f"medium={counts['medium']}"
        assert 15 <= counts["hard"] <= 45, f"hard={counts['hard']}"


class TestCLI:
    def test_cli_generates_files(self, tmp_path):
        """CLI main() with --domain hallucination --count 10 writes samples.json and manifest.json."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--domain", "hallucination",
                "--count", "10",
                "--output-dir", str(tmp_path),
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        samples_path = tmp_path / "hallucination" / "generated_samples.json"
        manifest_path = tmp_path / "hallucination" / "generated_manifest.json"
        assert samples_path.exists(), f"samples.json not found at {samples_path}"
        assert manifest_path.exists(), f"manifest.json not found at {manifest_path}"

        with open(samples_path) as f:
            data = json.load(f)
        assert len(data["samples"]) == 10

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest) == 10

    def test_cli_generates_log(self, tmp_path):
        """CLI writes generation_log.json with metadata."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--domain", "hallucination",
                "--count", "5",
                "--output-dir", str(tmp_path),
                "--seed", "99",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        log_path = tmp_path / "hallucination" / "generation_log.json"
        assert log_path.exists()
        with open(log_path) as f:
            log = json.load(f)
        assert log["count"] == 5
        assert log["seed"] == 99
        assert log["method"] == "template"

    def test_cli_all_domains(self, tmp_path):
        """CLI --domain all generates for all 4 domains."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--domain", "all",
                "--count", "5",
                "--output-dir", str(tmp_path),
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        for domain in ["hallucination", "code_security", "reasoning", "bio"]:
            assert (tmp_path / domain / "generated_samples.json").exists()
            assert (tmp_path / domain / "generated_manifest.json").exists()

    def test_cli_help(self):
        """CLI --help works."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "generate_eval_data.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--domain" in result.stdout
        assert "--count" in result.stdout

    def test_cli_append_mode(self, tmp_path):
        """CLI --append mode appends to existing files."""
        domain_dir = tmp_path / "hallucination"
        domain_dir.mkdir(parents=True)
        # Create initial data
        existing_samples = {
            "samples": [
                {
                    "id": "eval-hall-001",
                    "prompt": "test",
                    "output": "test",
                    "domain": "hallucination",
                    "metadata": {"difficulty": "easy", "source": "synthetic"},
                }
            ]
        }
        existing_manifest = {
            "eval-hall-001": {
                "ground_truth_label": "normal",
                "ground_truth_type": None,
                "is_honeypot": False,
            }
        }
        with open(domain_dir / "samples.json", "w") as f:
            json.dump(existing_samples, f)
        with open(domain_dir / "manifest.json", "w") as f:
            json.dump(existing_manifest, f)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--domain", "hallucination",
                "--count", "5",
                "--output-dir", str(tmp_path),
                "--seed", "42",
                "--append",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        with open(domain_dir / "samples.json") as f:
            data = json.load(f)
        # Should have 1 existing + 5 new
        assert len(data["samples"]) == 6
        # New IDs start from 002 (after existing 001)
        new_ids = [s["id"] for s in data["samples"][1:]]
        assert all(sid.startswith("eval-hall-") for sid in new_ids)
        # No duplicates
        all_ids = [s["id"] for s in data["samples"]]
        assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# Task 2: Validation script tests
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_data_returns_no_errors(self):
        from validate_eval_data import validate_schema

        samples = [
            {"id": "s1", "prompt": "p", "output": "o", "domain": "hallucination",
             "metadata": {"difficulty": "easy", "source": "synthetic"}},
            {"id": "s2", "prompt": "p", "output": "o", "domain": "hallucination",
             "metadata": {"difficulty": "medium", "source": "template"}},
        ]
        manifest = {
            "s1": {"ground_truth_label": "normal", "ground_truth_type": None, "is_honeypot": False},
            "s2": {"ground_truth_label": "anomalous", "ground_truth_type": "factual_error", "is_honeypot": False},
        }
        errors = validate_schema(samples, manifest, "hallucination")
        assert errors == []

    def test_missing_id_field(self):
        from validate_eval_data import validate_schema

        samples = [
            {"prompt": "p", "output": "o", "domain": "hallucination",
             "metadata": {"difficulty": "easy", "source": "synthetic"}},
        ]
        manifest = {}
        errors = validate_schema(samples, manifest, "hallucination")
        assert len(errors) > 0
        assert any("id" in e.lower() for e in errors)

    def test_sample_id_not_in_manifest(self):
        from validate_eval_data import validate_schema

        samples = [
            {"id": "s1", "prompt": "p", "output": "o", "domain": "hallucination",
             "metadata": {"difficulty": "easy", "source": "synthetic"}},
        ]
        manifest = {}
        errors = validate_schema(samples, manifest, "hallucination")
        assert len(errors) > 0
        assert any("manifest" in e.lower() or "s1" in e for e in errors)

    def test_code_security_requires_code_field(self):
        from validate_eval_data import validate_schema

        samples = [
            {"id": "s1", "prompt": "p", "output": "", "domain": "code_security",
             "metadata": {"difficulty": "easy", "source": "synthetic"}},
        ]
        manifest = {
            "s1": {"ground_truth_label": "normal", "ground_truth_type": None, "is_honeypot": False},
        }
        errors = validate_schema(samples, manifest, "code_security")
        assert len(errors) > 0
        assert any("code" in e.lower() for e in errors)

    def test_orphan_manifest_entry(self):
        from validate_eval_data import validate_schema

        samples = [
            {"id": "s1", "prompt": "p", "output": "o", "domain": "hallucination",
             "metadata": {"difficulty": "easy", "source": "synthetic"}},
        ]
        manifest = {
            "s1": {"ground_truth_label": "normal", "ground_truth_type": None, "is_honeypot": False},
            "s_orphan": {"ground_truth_label": "anomalous", "ground_truth_type": "factual_error", "is_honeypot": False},
        }
        errors = validate_schema(samples, manifest, "hallucination")
        assert len(errors) > 0
        assert any("orphan" in e.lower() or "s_orphan" in e for e in errors)


class TestValidateBalance:
    def test_balanced_passes(self):
        from validate_eval_data import validate_balance

        manifest = {
            "s1": {"ground_truth_label": "normal", "ground_truth_type": None, "is_honeypot": False},
            "s2": {"ground_truth_label": "anomalous", "ground_truth_type": "x", "is_honeypot": False},
        }
        errors = validate_balance(manifest)
        assert errors == []

    def test_imbalanced_fails(self):
        from validate_eval_data import validate_balance

        # 90% anomalous
        manifest = {}
        for i in range(9):
            manifest[f"s{i}"] = {"ground_truth_label": "anomalous", "ground_truth_type": "x", "is_honeypot": False}
        manifest["s9"] = {"ground_truth_label": "normal", "ground_truth_type": None, "is_honeypot": False}
        errors = validate_balance(manifest)
        assert len(errors) > 0


class TestValidateDuplicates:
    def test_unique_passes(self):
        from validate_eval_data import validate_duplicates

        samples = [
            {"id": "s1", "prompt": "p", "output": "o"},
            {"id": "s2", "prompt": "p", "output": "o"},
        ]
        errors = validate_duplicates(samples)
        assert errors == []

    def test_duplicate_fails(self):
        from validate_eval_data import validate_duplicates

        samples = [
            {"id": "s1", "prompt": "p", "output": "o"},
            {"id": "s1", "prompt": "q", "output": "r"},
        ]
        errors = validate_duplicates(samples)
        assert len(errors) > 0
        assert any("s1" in e for e in errors)


class TestValidationCLI:
    def test_validates_existing_domains(self):
        """CLI validates all 4 existing domains and exits 0."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "validate_eval_data.py"),
                "--domain", "all",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR.parent),
        )
        assert result.returncode == 0, f"Validation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        # Should mention all 4 domains in output
        for domain in ["hallucination", "code_security", "reasoning", "bio"]:
            assert domain in result.stdout.lower() or domain in result.stderr.lower(), \
                f"Domain {domain} not mentioned in output"

    def test_cli_help(self):
        """Validate CLI --help works."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "validate_eval_data.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--domain" in result.stdout


# ---------------------------------------------------------------------------
# Task 3 (gap closure): LLM generation mode tests
# ---------------------------------------------------------------------------


class TestLLMArgparse:
    """Test that the CLI accepts new LLM-related arguments."""

    def test_argparse_accepts_api_key(self):
        """argparse accepts --api-key optional arg."""
        from generate_eval_data import main
        import argparse

        # Parse known args to verify --api-key is accepted
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--api-key" in result.stdout

    def test_argparse_accepts_api_provider(self):
        """argparse accepts --api-provider optional arg."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--api-provider" in result.stdout

    def test_argparse_accepts_method_with_choices(self):
        """argparse accepts --method with choices ['template', 'llm'], default 'template'."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--method" in result.stdout
        # Verify choices are mentioned
        assert "template" in result.stdout
        assert "llm" in result.stdout

    def test_method_llm_without_api_key_exits(self):
        """--method=llm without --api-key (and no env var) raises SystemExit."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--domain", "hallucination",
                "--count", "5",
                "--method", "llm",
            ],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "ANTIGENCE_LLM_API_KEY": ""},
        )
        assert result.returncode != 0

    def test_api_provider_valid_choices(self):
        """--api-provider accepts 'openai', 'anthropic', 'local' as valid choices."""
        for provider in ["openai", "anthropic", "local"]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "generate_eval_data.py"),
                    "--domain", "hallucination",
                    "--count", "1",
                    "--method", "template",
                    "--api-provider", provider,
                    "--output-dir", "/tmp/test_provider_check",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Provider {provider} rejected: {result.stderr}"


class TestLLMGeneration:
    """Test LLM generation code path with mocked API."""

    def _mock_openai_response(self, content="This is a mocked LLM response."):
        """Create a mock urllib response mimicking OpenAI chat completions."""
        response_data = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode()
        mock_response = MagicMock()
        mock_response.read.return_value = response_data
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        return mock_response

    def test_llm_hallucination_samples_schema(self):
        """LLM generator for hallucination returns samples with correct schema."""
        from generate_eval_data import generate_hallucination_samples_llm

        with patch("generate_eval_data.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_openai_response()
            samples, manifest = generate_hallucination_samples_llm(
                n=5, start_id=1, seed=42, api_key="test-key", api_provider="openai"
            )

        assert len(samples) == 5
        for s in samples:
            assert "id" in s
            assert "prompt" in s
            assert "output" in s
            assert "domain" in s
            assert s["domain"] == "hallucination"
            assert "metadata" in s
            assert s["metadata"]["source"] == "llm"

    def test_llm_code_security_samples_schema(self):
        """LLM generator for code_security returns samples with code field."""
        from generate_eval_data import generate_code_security_samples_llm

        with patch("generate_eval_data.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_openai_response("def func():\n    pass")
            samples, manifest = generate_code_security_samples_llm(
                n=5, start_id=1, seed=42, api_key="test-key", api_provider="openai"
            )

        assert len(samples) == 5
        for s in samples:
            assert "id" in s
            assert "prompt" in s
            assert "output" in s
            assert "code" in s
            assert "domain" in s
            assert s["domain"] == "code_security"
            assert s["metadata"]["source"] == "llm"

    def test_llm_fallback_on_api_failure(self):
        """LLM generation falls back to template when API fails."""
        from generate_eval_data import generate_hallucination_samples_llm

        with patch("generate_eval_data.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection refused")
            samples, manifest = generate_hallucination_samples_llm(
                n=5, start_id=1, seed=42, api_key="test-key", api_provider="openai"
            )

        assert len(samples) == 5
        # All should fall back to template source
        for s in samples:
            assert s["metadata"]["source"] == "template"
            assert "id" in s
            assert "prompt" in s
            assert "output" in s

    def test_template_mode_regression(self, tmp_path):
        """Template-only mode (no --api-key, no --method) produces identical output."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--domain", "hallucination",
                "--count", "10",
                "--output-dir", str(tmp_path),
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        samples_path = tmp_path / "hallucination" / "generated_samples.json"
        assert samples_path.exists()
        with open(samples_path) as f:
            data = json.load(f)
        assert len(data["samples"]) == 10
        for s in data["samples"]:
            assert s["metadata"]["source"] == "template"

    def test_generation_log_records_llm_method(self, tmp_path):
        """generation_log.json records method='llm' when LLM mode used."""
        from generate_eval_data import _write_output
        from pathlib import Path

        _write_output(
            domain="hallucination",
            samples=[],
            manifest={},
            output_dir=tmp_path,
            append=False,
            seed=42,
            count=10,
            method="llm",
        )
        log_path = tmp_path / "hallucination" / "generation_log.json"
        assert log_path.exists()
        with open(log_path) as f:
            log = json.load(f)
        assert log["method"] == "llm"

    def test_llm_generators_registry_exists(self):
        """LLM_GENERATORS registry exists with all 4 domains."""
        from generate_eval_data import LLM_GENERATORS

        assert "hallucination" in LLM_GENERATORS
        assert "code_security" in LLM_GENERATORS
        assert "reasoning" in LLM_GENERATORS
        assert "bio" in LLM_GENERATORS

    def test_call_llm_api_function_exists(self):
        """_call_llm_api function exists and is callable."""
        from generate_eval_data import _call_llm_api

        assert callable(_call_llm_api)

    def test_env_var_api_key_accepted(self):
        """ANTIGENCE_LLM_API_KEY env var is accepted as API key."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_eval_data.py"),
                "--domain", "hallucination",
                "--count", "1",
                "--method", "llm",
                "--output-dir", "/tmp/test_env_key",
            ],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "ANTIGENCE_LLM_API_KEY": "test-key-from-env"},
        )
        # Should not exit with "API key required" error
        # (may still fail due to actual API call, but should not fail at arg parsing)
        if result.returncode != 0:
            assert "api key" not in result.stderr.lower() or "api" not in result.stderr.lower()
