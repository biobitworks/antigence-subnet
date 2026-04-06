"""Tests for package-relative path resolution (CORR-03).

Verifies that _PROJECT_ROOT, _DEFAULT_EVAL_DATA_DIR, and
_DEFAULT_TRAINING_DATA_DIR resolve to absolute paths under the project
root regardless of the working directory the process is launched from.
"""

from pathlib import Path


class TestProjectRoot:
    """Verify _PROJECT_ROOT resolves to the correct project root."""

    def test_project_root_is_absolute(self):
        """_PROJECT_ROOT is an absolute path."""
        from antigence_subnet.base.neuron import _PROJECT_ROOT

        assert _PROJECT_ROOT.is_absolute(), (
            f"_PROJECT_ROOT must be absolute, got: {_PROJECT_ROOT}"
        )

    def test_project_root_contains_antigence_subnet(self):
        """_PROJECT_ROOT directory contains antigence_subnet/ package."""
        from antigence_subnet.base.neuron import _PROJECT_ROOT

        assert (_PROJECT_ROOT / "antigence_subnet").is_dir(), (
            f"Expected antigence_subnet/ in {_PROJECT_ROOT}"
        )

    def test_project_root_contains_data_dir(self):
        """_PROJECT_ROOT directory contains data/ directory."""
        from antigence_subnet.base.neuron import _PROJECT_ROOT

        assert (_PROJECT_ROOT / "data").is_dir(), (
            f"Expected data/ in {_PROJECT_ROOT}"
        )


class TestDefaultPaths:
    """Verify default eval/training data paths are absolute."""

    def test_default_eval_data_dir_is_absolute(self):
        """_DEFAULT_EVAL_DATA_DIR is an absolute path string."""
        from antigence_subnet.base.neuron import _DEFAULT_EVAL_DATA_DIR

        assert Path(_DEFAULT_EVAL_DATA_DIR).is_absolute(), (
            f"_DEFAULT_EVAL_DATA_DIR must be absolute, got: {_DEFAULT_EVAL_DATA_DIR}"
        )

    def test_default_training_data_dir_is_absolute(self):
        """_DEFAULT_TRAINING_DATA_DIR is an absolute path string."""
        from antigence_subnet.base.neuron import _DEFAULT_TRAINING_DATA_DIR

        assert Path(_DEFAULT_TRAINING_DATA_DIR).is_absolute(), (
            f"_DEFAULT_TRAINING_DATA_DIR must be absolute, got: {_DEFAULT_TRAINING_DATA_DIR}"
        )

    def test_default_eval_data_dir_points_to_data_evaluation(self):
        """_DEFAULT_EVAL_DATA_DIR ends with data/evaluation."""
        from antigence_subnet.base.neuron import _DEFAULT_EVAL_DATA_DIR

        path = Path(_DEFAULT_EVAL_DATA_DIR)
        assert path.parts[-2:] == ("data", "evaluation"), (
            f"Expected path ending in data/evaluation, got: {path}"
        )

    def test_default_training_data_dir_points_to_data_evaluation(self):
        """_DEFAULT_TRAINING_DATA_DIR ends with data/evaluation."""
        from antigence_subnet.base.neuron import _DEFAULT_TRAINING_DATA_DIR

        path = Path(_DEFAULT_TRAINING_DATA_DIR)
        assert path.parts[-2:] == ("data", "evaluation"), (
            f"Expected path ending in data/evaluation, got: {path}"
        )

    def test_default_eval_data_dir_exists(self):
        """_DEFAULT_EVAL_DATA_DIR points to a real directory."""
        from antigence_subnet.base.neuron import _DEFAULT_EVAL_DATA_DIR

        assert Path(_DEFAULT_EVAL_DATA_DIR).is_dir(), (
            f"Expected directory to exist: {_DEFAULT_EVAL_DATA_DIR}"
        )
