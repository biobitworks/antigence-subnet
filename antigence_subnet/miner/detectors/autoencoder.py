"""Autoencoder-based GPU anomaly detector for the hallucination domain.

Uses a symmetric dense autoencoder with TF-IDF features. Trained on normal
(self) samples only. Anomaly scores are percentile-normalized reconstruction
errors mapped to [0, 1] against the training baseline distribution.
"""


import joblib
import numpy as np
import torch
import torch.nn as nn

from antigence_subnet.miner.detector import BaseDetector, DetectionResult
from antigence_subnet.miner.detectors.features import create_vectorizer, samples_to_texts


class TextAutoencoder(nn.Module):
    """Symmetric dense autoencoder for TF-IDF feature reconstruction.

    Architecture:
        Encoder: Linear -> ReLU -> Dropout -> Linear -> ReLU
        Decoder: Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input through encoder-decoder pipeline."""
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseDetector):
    """GPU-capable anomaly detector using a dense autoencoder with TF-IDF features."""

    domain = "hallucination"

    def __init__(
        self,
        max_features: int = 5000,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str | None = None,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = create_vectorizer(max_features)
        self.model: TextAutoencoder | None = None
        self._baseline_errors_sorted: np.ndarray | None = None
        self._is_fitted = False
        self._input_dim: int | None = None

    def fit(self, samples: list[dict]) -> None:
        """Train autoencoder on normal (self) samples.

        Fits the TF-IDF vectorizer, trains the autoencoder to minimize
        reconstruction error on normal data, then stores the sorted
        baseline error distribution for percentile normalization.

        Args:
            samples: List of normal sample dicts with prompt/output keys.
        """
        texts = samples_to_texts(samples)
        X_sparse = self.vectorizer.fit_transform(texts)  # noqa: N806
        X_dense = X_sparse.toarray()  # noqa: N806
        self._input_dim = X_dense.shape[1]

        # Set random seed for reproducible training
        torch.manual_seed(self.random_state)

        X_tensor = torch.FloatTensor(X_dense).to(self.device)  # noqa: N806

        self.model = TextAutoencoder(
            self._input_dim, self.hidden_dim, self.latent_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            reconstructed = self.model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()

        # Compute baseline reconstruction errors
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1).cpu().numpy()

        self._baseline_errors_sorted = np.sort(errors)
        self._is_fitted = True

    async def detect(
        self,
        prompt: str,
        output: str,
        code: str | None = None,
        context: str | None = None,
    ) -> DetectionResult:
        """Run anomaly detection via reconstruction error.

        Transforms input text to TF-IDF features, computes reconstruction
        error, and normalizes to [0, 1] via percentile against baseline.
        Higher reconstruction error = higher anomaly score.

        Args:
            prompt: Original prompt text.
            output: AI-generated output to verify.
            code: Optional code content (unused for hallucination domain).
            context: Optional metadata (unused for hallucination domain).

        Returns:
            DetectionResult with percentile-normalized anomaly score.
        """
        text = f"{prompt} {output}"
        X_sparse = self.vectorizer.transform([text])  # noqa: N806
        X_dense = X_sparse.toarray()  # noqa: N806
        X_tensor = torch.FloatTensor(X_dense).to(self.device)  # noqa: N806

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            error = float(torch.mean((reconstructed - X_tensor) ** 2).cpu().item())

        # Two-sided deviation normalization: distance from baseline median,
        # normalized by baseline spread. Higher deviation = higher anomaly score.
        n = len(self._baseline_errors_sorted)
        median_error = self._baseline_errors_sorted[n // 2]
        baseline_range = self._baseline_errors_sorted[-1] - self._baseline_errors_sorted[0]

        if baseline_range < 1e-10:
            anomaly_score = 0.5
        else:
            deviation = abs(error - median_error) / baseline_range
            anomaly_score = float(np.clip(deviation, 0.0, 1.0))

        confidence = float(min(abs(anomaly_score - 0.5) * 2.0, 1.0))
        anomaly_type = "hallucination" if anomaly_score >= 0.5 else "normal"

        return DetectionResult(
            score=anomaly_score,
            confidence=confidence,
            anomaly_type=anomaly_type,
            feature_attribution=None,
        )

    def get_info(self) -> dict:
        """Return detector metadata."""
        return {
            "name": "AutoencoderDetector",
            "domain": self.domain,
            "version": "0.1.0",
            "backend": "pytorch",
            "is_fitted": self._is_fitted,
        }

    def save_state(self, path: str) -> None:
        """Save model state to disk.

        Saves vectorizer via joblib and model state + config via torch.save.

        Args:
            path: Directory to save state files in.
        """
        joblib.dump(self.vectorizer, f"{path}/ae_vectorizer.joblib")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self._input_dim,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "baseline_errors": self._baseline_errors_sorted,
            },
            f"{path}/ae_state.pt",
        )

    def load_state(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: Directory containing ae_vectorizer.joblib and ae_state.pt.
        """
        self.vectorizer = joblib.load(f"{path}/ae_vectorizer.joblib")
        checkpoint = torch.load(
            f"{path}/ae_state.pt",
            map_location=self.device,
            weights_only=False,
        )
        self._input_dim = checkpoint["input_dim"]
        self.model = TextAutoencoder(
            self._input_dim,
            checkpoint["hidden_dim"],
            checkpoint["latent_dim"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._baseline_errors_sorted = checkpoint["baseline_errors"]
        self._is_fitted = True
