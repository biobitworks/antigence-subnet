# Antigence Subnet — Build and Test Targets
# Usage: make help

PYTHON ?= python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest

.PHONY: help test-env test test-quick test-ollama validate-data clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

test-env: $(VENV)/pyvenv.cfg ## Create venv and install all test dependencies (idempotent)

$(VENV)/pyvenv.cfg: requirements-test.txt pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-test.txt
	$(PIP) install -e .
	@touch $(VENV)/pyvenv.cfg

test: test-env ## Run full test suite (creates venv if needed)
	$(PYTEST) tests/ -x --tb=short

test-quick: ## Run tests without venv setup (fast iteration)
	$(PYTEST) tests/ -x --tb=short

test-ollama: test-env ## Run only @pytest.mark.ollama tests
	$(PYTEST) tests/ -m ollama -x --tb=short

validate-data: test-env ## Validate evaluation datasets (all 4 domains)
	$(VENV_BIN)/python scripts/validate_eval_data.py --domain all

clean: ## Remove venv, egg-info, pytest cache, coverage artifacts
	rm -rf $(VENV)
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
