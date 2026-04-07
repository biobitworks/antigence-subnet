"""Tests for the code security domain pack.

Covers AST-based feature extraction, evaluation dataset integrity,
CodeSecurityDetector class, and registry integration.
"""

import json
import shutil
import tempfile

import pytest

# ------------------------------------------------------------------
# Test: Code security feature extraction
# ------------------------------------------------------------------


class TestCodeSecurityFeatures:
    """Tests for extract_code_security_features()."""

    def test_returns_dict_with_seven_keys(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        result = extract_code_security_features("x = 1")
        expected_keys = {
            "dangerous_call_count",
            "string_concat_in_call",
            "hardcoded_credential_score",
            "import_risk_score",
            "ast_node_depth",
            "exec_eval_usage",
            "total_function_calls",
        }
        assert set(result.keys()) == expected_keys
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_dangerous_calls_detected(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        result = extract_code_security_features("import os; os.system('rm -rf /')")
        assert result["dangerous_call_count"] >= 1
        assert result["exec_eval_usage"] >= 0

    def test_string_concat_in_call_sql_injection(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        code = "cursor.execute('SELECT * FROM users WHERE id=' + user_input)"
        result = extract_code_security_features(code)
        assert result["string_concat_in_call"] >= 1

    def test_empty_code_returns_defaults(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        result = extract_code_security_features("")
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_syntax_error_returns_graceful_defaults(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        result = extract_code_security_features("def foo(:")
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"
        # Should not crash, just return defaults

    def test_hardcoded_credentials(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        code = 'password = "mysecret123"\napi_key = "AKIAIOSFODNN7EXAMPLE"'
        result = extract_code_security_features(code)
        assert result["hardcoded_credential_score"] >= 1

    def test_risky_imports(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        code = "import pickle\nimport subprocess\nimport ctypes"
        result = extract_code_security_features(code)
        assert result["import_risk_score"] >= 3

    def test_exec_eval_usage(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        code = "eval(user_input)\nexec(code_str)"
        result = extract_code_security_features(code)
        assert result["exec_eval_usage"] >= 2
        assert result["dangerous_call_count"] >= 2

    def test_ast_node_depth(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        # Nested code should have depth > 1
        code = "if True:\n    if True:\n        if True:\n            x = 1"
        result = extract_code_security_features(code)
        assert result["ast_node_depth"] >= 3

    def test_total_function_calls(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            extract_code_security_features,
        )

        code = "print(len(str(42)))"
        result = extract_code_security_features(code)
        assert result["total_function_calls"] >= 3

    def test_code_from_sample_prefers_code_field(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            code_from_sample,
        )

        sample = {"code": "x = 1", "output": "y = 2"}
        assert code_from_sample(sample) == "x = 1"

    def test_code_from_sample_falls_back_to_output(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            code_from_sample,
        )

        sample = {"output": "y = 2"}
        assert code_from_sample(sample) == "y = 2"

    def test_code_from_sample_empty(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.features import (
            code_from_sample,
        )

        sample = {}
        assert code_from_sample(sample) == ""


# ------------------------------------------------------------------
# Test: CodeSecurityDetector class
# ------------------------------------------------------------------


@pytest.fixture
def cs_normal_samples():
    """Load normal-only samples from the code security evaluation data."""
    with open("data/evaluation/code_security/samples.json") as f:
        samples = json.load(f)["samples"]
    with open("data/evaluation/code_security/manifest.json") as f:
        manifest = json.load(f)
    return [s for s in samples if manifest[s["id"]]["ground_truth_label"] == "normal"]


@pytest.fixture
def cs_anomalous_sample():
    """Return a single anomalous code sample for detection testing."""
    return {
        "prompt": "Write a function to query user data",
        "output": "",
        "code": (
            "def get_user(uid):\n"
            "    cursor.execute('SELECT * FROM users WHERE id=' + uid)\n"
            "    return cursor.fetchone()"
        ),
    }


class TestCodeSecurityDetector:
    """Tests for CodeSecurityDetector class."""

    def test_domain_is_code_security(self):
        from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
            CodeSecurityDetector,
        )

        assert CodeSecurityDetector.domain == "code_security"

    def test_fit_sets_is_fitted(self, cs_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
            CodeSecurityDetector,
        )

        detector = CodeSecurityDetector()
        assert not detector._is_fitted
        detector.fit(cs_normal_samples)
        assert detector._is_fitted

    @pytest.mark.asyncio
    async def test_detect_returns_valid_result(self, cs_normal_samples, cs_anomalous_sample):
        from antigence_subnet.miner.detector import DetectionResult
        from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
            CodeSecurityDetector,
        )

        detector = CodeSecurityDetector()
        detector.fit(cs_normal_samples)
        result = await detector.detect(
            prompt=cs_anomalous_sample["prompt"],
            output=cs_anomalous_sample["output"],
            code=cs_anomalous_sample["code"],
        )
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.anomaly_type, str)

    @pytest.mark.asyncio
    async def test_detect_returns_feature_attribution(self, cs_normal_samples, cs_anomalous_sample):
        from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
            CodeSecurityDetector,
        )

        detector = CodeSecurityDetector()
        detector.fit(cs_normal_samples)
        result = await detector.detect(
            prompt=cs_anomalous_sample["prompt"],
            output=cs_anomalous_sample["output"],
            code=cs_anomalous_sample["code"],
        )
        assert result.feature_attribution is not None
        domain_features = {
            "dangerous_call_count",
            "string_concat_in_call",
            "hardcoded_credential_score",
            "import_risk_score",
            "ast_node_depth",
            "exec_eval_usage",
            "total_function_calls",
        }
        assert domain_features.issubset(set(result.feature_attribution.keys()))

    def test_get_info_after_fit(self, cs_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
            CodeSecurityDetector,
        )

        detector = CodeSecurityDetector()
        detector.fit(cs_normal_samples)
        info = detector.get_info()
        assert info["backend"] == "scikit-learn"
        assert info["is_fitted"] is True
        assert info["domain"] == "code_security"
        assert info["name"] == "CodeSecurityDetector"

    def test_save_load_round_trip(self, cs_normal_samples):
        from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
            CodeSecurityDetector,
        )

        detector = CodeSecurityDetector()
        detector.fit(cs_normal_samples)
        tmpdir = tempfile.mkdtemp()
        try:
            detector.save_state(tmpdir)
            detector2 = CodeSecurityDetector()
            assert not detector2._is_fitted
            detector2.load_state(tmpdir)
            assert detector2._is_fitted
            assert len(detector2._baseline_scores_sorted) == len(
                detector._baseline_scores_sorted
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_registry_integration(self):
        from antigence_subnet.miner.detectors import get_detector
        from antigence_subnet.miner.detectors.domain_packs.code_security.detector import (
            CodeSecurityDetector,
        )

        cls = get_detector("code_security")
        assert cls is CodeSecurityDetector


# ------------------------------------------------------------------
# Test: Evaluation dataset integrity
# ------------------------------------------------------------------


class TestCodeSecurityDataset:
    """Tests for code security evaluation data files."""

    SAMPLES_PATH = "data/evaluation/code_security/samples.json"
    MANIFEST_PATH = "data/evaluation/code_security/manifest.json"

    def test_samples_has_30_entries(self):
        with open(self.SAMPLES_PATH) as f:
            data = json.load(f)
        assert len(data["samples"]) == 220

    def test_manifest_has_entry_for_every_sample(self):
        with open(self.SAMPLES_PATH) as f:
            samples = json.load(f)["samples"]
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        sample_ids = {s["id"] for s in samples}
        manifest_ids = set(manifest.keys())
        assert sample_ids == manifest_ids

    def test_ground_truth_types_cover_required_categories(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        types = {v.get("ground_truth_type") for v in manifest.values()}
        required = {
            "sql_injection",
            "command_injection",
            "hardcoded_credentials",
            "insecure_deserialization",
            "path_traversal",
        }
        assert required.issubset(types)
        # Normal samples have null type
        assert None in types

    def test_at_least_2_honeypots(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        honeypot_count = sum(1 for v in manifest.values() if v.get("is_honeypot"))
        assert honeypot_count >= 2

    def test_all_samples_have_code_security_domain(self):
        with open(self.SAMPLES_PATH) as f:
            samples = json.load(f)["samples"]

        for s in samples:
            assert s["domain"] == "code_security", f"Sample {s['id']} has domain={s['domain']}"

    def test_manifest_entries_have_correct_format(self):
        with open(self.MANIFEST_PATH) as f:
            manifest = json.load(f)

        for _sample_id, entry in manifest.items():
            assert "ground_truth_label" in entry
            assert "ground_truth_type" in entry
            assert "is_honeypot" in entry
            assert entry["ground_truth_label"] in ("anomalous", "normal")
