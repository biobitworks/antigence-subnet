# Contributing to Antigence Subnet

Contributions are welcome. There are three main ways to contribute:

1. **Build a custom detector** -- implement your own anomaly detection approach
2. **Add a domain pack** -- extend the subnet to new verification domains
3. **Improve the core** -- fix bugs, optimize performance, improve documentation

## Quick Start for Contributors

```bash
git clone https://github.com/biobitworks/antigence-bittensor.git
cd antigence-bittensor
pip install -e '.[dev]'
pytest tests/ -x -q
```

If all tests pass, the development environment is ready.

## Building a Custom Detector

### The BaseDetector Interface

All detectors implement the `BaseDetector` abstract base class from `antigence_subnet/miner/detector.py`.

**DetectionResult** -- the return type of `detect()`:

```python
@dataclass
class DetectionResult:
    score: float                                    # 0.0 (normal) to 1.0 (anomalous)
    confidence: float                               # 0.0 to 1.0
    anomaly_type: str                               # type identifier (e.g., "hallucination")
    feature_attribution: Optional[Dict[str, float]] = None  # optional feature importance scores
```

**BaseDetector** -- the abstract base class:

```python
class BaseDetector(ABC):
    domain: str  # REQUIRED class attribute -- the domain this detector handles

    @abstractmethod
    def fit(self, samples: list[dict]) -> None:
        """Train/fit the detector on normal (self) samples.
        samples: list of dicts with keys matching VerificationSynapse fields
        (prompt, output, code, context).
        """
        ...

    @abstractmethod
    async def detect(
        self,
        prompt: str,
        output: str,
        code: Optional[str] = None,
        context: Optional[str] = None,
    ) -> DetectionResult:
        """Run detection on a single input. Returns DetectionResult."""
        ...

    def save_state(self, path: str) -> None:
        """Save detector model state. Override for persistent detectors."""
        pass

    def load_state(self, path: str) -> None:
        """Load detector model state. Override for persistent detectors."""
        pass

    def get_info(self) -> dict:
        """Return detector metadata.
        Returns dict with keys: name, domain, version, backend, is_fitted.
        """
        ...
```

Key points:

- `detect()` is **async**. Return a `DetectionResult`, not a tuple.
- `fit()` receives `list[dict]` where each dict has keys matching VerificationSynapse fields: `prompt`, `output`, `code`, `context`.
- `domain` is a **required class attribute** -- set it as a class variable.
- `save_state()`, `load_state()`, and `get_info()` are optional overrides with default implementations.

### Step-by-Step: Your First Detector

Create a file (e.g., `my_detector.py`) with a complete detector implementation:

```python
from typing import Optional
from antigence_subnet.miner.detector import BaseDetector, DetectionResult


class MyDetector(BaseDetector):
    """Minimal detector implementation."""

    domain = "hallucination"  # Required class attribute

    def fit(self, samples: list[dict]) -> None:
        """Train on normal (self) samples.
        samples: list of dicts with keys: prompt, output, code, context
        """
        # Your training logic here
        self._is_fitted = True

    async def detect(
        self,
        prompt: str,
        output: str,
        code: Optional[str] = None,
        context: Optional[str] = None,
    ) -> DetectionResult:
        """Score a single input. Returns DetectionResult."""
        # Your detection logic here
        return DetectionResult(
            score=0.5,            # 0.0=normal, 1.0=anomalous
            confidence=0.8,       # 0.0-1.0
            anomaly_type="normal",
            feature_attribution=None,  # Optional dict
        )

    def get_info(self) -> dict:
        return {
            "name": "MyDetector",
            "domain": self.domain,
            "version": "1.0.0",
            "backend": "custom",
            "is_fitted": getattr(self, "_is_fitted", False),
        }
```

Run your detector with mock infrastructure (no testnet needed):

```bash
python neurons/miner.py --mock --netuid 1 \
  --detector my_package.my_module.MyDetector
```

### Testing Your Detector

Write a test file (e.g., `tests/test_my_detector.py`):

```python
import pytest
from my_detector import MyDetector


@pytest.fixture
def detector():
    d = MyDetector()
    d.fit([{"prompt": "What is 2+2?", "output": "4", "code": None, "context": None}])
    return d


@pytest.mark.asyncio
async def test_detect_returns_result(detector):
    result = await detector.detect(prompt="What is 2+2?", output="4")
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.anomaly_type, str)


@pytest.mark.asyncio
async def test_detect_returns_detection_result_type(detector):
    from antigence_subnet.miner.detector import DetectionResult
    result = await detector.detect(prompt="test", output="test")
    assert isinstance(result, DetectionResult)
```

Run: `pytest tests/test_my_detector.py -v`

### Running CI Locally

```bash
# Lint (same rules as CI)
pip install ruff
ruff check .

# Format check
ruff format --check .

# Tests
pytest tests/ -x --tb=short -q
```

CI runs automatically on pull requests via GitHub Actions: ruff lint + pytest.

## Contributing Domain Packs

### Data Format

See [docs/miner-training-data.md](docs/miner-training-data.md) for the full evaluation data format, available public datasets, and self/non-self mapping guidance.

Each domain pack needs:

1. A detector class implementing `BaseDetector` with a unique `domain` string
2. Evaluation data in `data/evaluation/<domain>/` with `samples.json` and `manifest.json`
3. Registration in `antigence_subnet.toml.example` under `[miner.detectors]`

### Adding a New Domain

Directory structure:

```
antigence_subnet/miner/detectors/domain_packs/<your_domain>/
  __init__.py
  detector.py      # Your BaseDetector subclass
data/evaluation/<your_domain>/
  samples.json     # Prompt/output sample pairs
  manifest.json    # Ground truth labels and honeypot flags
```

The `samples.json` format:

```json
{
  "samples": [
    {
      "id": "unique-id",
      "prompt": "The input prompt",
      "output": "The AI-generated output to evaluate",
      "domain": "your_domain",
      "metadata": {"source": "dataset-name"}
    }
  ]
}
```

### Multi-Domain Miner Config

Register your domain in `antigence_subnet.toml.example` under `[miner.detectors]`:

```toml
[miner.detectors]
hallucination = "antigence_subnet.miner.detectors.domain_packs.hallucination.detector.HallucinationDetector"
your_domain = "your_package.your_module.YourDetector"
```

A multi-domain miner serves multiple detectors simultaneously, increasing scoring opportunities.

## Third-Party Miner Guide

### Choosing a Detection Approach

Two paths:

- **CPU-only** (IsolationForest, scikit-learn): Lower resource requirements. The reference implementation provides a good baseline. No GPU needed.
- **GPU-accelerated** (autoencoder, PyTorch): Higher detection accuracy potential. Requires a CUDA-capable GPU.

### Understanding the Reward Function

The composite reward formula:

- **70% Base reward** = 70% precision + 30% recall. Precision is weighted more heavily because false positives (flagging legitimate output as anomalous) are treated as autoimmune errors.
- **10% Calibration**: Score distribution should clearly separate normal from anomalous inputs (measured by ECE across 10 equal-width bins).
- **10% Robustness**: Consistent scoring on perturbed inputs (synonym substitution, whitespace changes).
- **10% Diversity**: Unique detection approach compared to other miners (measured by cosine similarity of score vectors).
- **Honeypot rule**: Known-answer samples are injected into each round. Failing a honeypot zeroes the entire round's reward.

False positives are penalized more heavily than false negatives. Optimize for precision first, then recall.

See [docs/performance-tuning.md](docs/performance-tuning.md) for the complete reward formula and tuning guidance.

### What Makes a Competitive Miner

- **High precision on honeypot samples** -- never fail honeypots (zeroes entire round)
- **Stable scores on perturbation tests** -- synonym substitutions and whitespace changes should not flip your scores
- **Well-calibrated score distribution** -- clear separation between self (normal) and non-self (anomalous) scores
- **Unique detection approach** -- diversity bonus rewards miners that do not duplicate other miners' methods
- **Multi-domain support** -- serve multiple domain detectors to increase scoring opportunities

## Code Style

### Ruff Configuration

```
Line length: 100
Rules: E (pycodestyle errors), W (warnings), F (pyflakes), I (isort),
       N (pep8-naming), UP (pyupgrade), B (flake8-bugbear), SIM (flake8-simplify)
```

Run: `ruff check .` and `ruff format --check .`

### pytest Patterns

- `asyncio_mode = "auto"` -- no need to mark individual async tests (configured in `pyproject.toml`)
- Mock infrastructure: `MockSubtensor`, `MockMetagraph`, `MockDendrite` from `antigence_subnet.base.mock`
- Test directory: `tests/`

## Submission Process

### PR Checklist

- [ ] All tests pass: `pytest tests/ -x -q`
- [ ] Linting passes: `ruff check .`
- [ ] Detector implements `BaseDetector` (or extends existing domain pack)
- [ ] Test file included with at least: `fit()`, `detect()`, and `get_info()` coverage
- [ ] Domain pack includes evaluation data (if new domain)
- [ ] TOML config example updated (if new domain)

### Review Process

Standard open-source PR workflow. Submit a PR against `main`. Maintainers review for:

- Interface compliance (correct `BaseDetector` implementation)
- Test coverage
- Code style (ruff passing)
- No security concerns (no secrets, no network calls in the detection path)
