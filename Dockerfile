FROM python:3.12-slim-bookworm AS base

WORKDIR /app

# Install system dependencies (tini for PID 1 signal forwarding, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for runtime security
RUN groupadd --gid 1000 antigence && \
    useradd --uid 1000 --gid antigence --create-home antigence

# Install Python dependencies (must run as root for site-packages access)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e '.[cli]'

# Copy application code with correct ownership
COPY --chown=antigence:antigence . .
RUN pip install --no-cache-dir -e .

# Switch to non-root user for runtime
USER antigence

# ---------------------------------------------------------------------------
# Validator target
# ---------------------------------------------------------------------------
FROM base AS validator

ENTRYPOINT ["tini", "--"]
CMD ["python", "neurons/validator.py", "--subtensor.network", "test", "--netuid", "1"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, signal; os.kill(1, 0)" || exit 1

# ---------------------------------------------------------------------------
# Miner target
# ---------------------------------------------------------------------------
FROM base AS miner

ENTRYPOINT ["tini", "--"]
CMD ["python", "neurons/miner.py", "--subtensor.network", "test", "--netuid", "1"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, signal; os.kill(1, 0)" || exit 1

# ---------------------------------------------------------------------------
# API server target
# ---------------------------------------------------------------------------
FROM base AS api

# Install API dependencies (requires root for pip)
USER root
RUN pip install --no-cache-dir -e '.[api]'
USER antigence

EXPOSE 8080

ENTRYPOINT ["tini", "--"]
CMD ["python", "neurons/api_server.py", "--port", "8080"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
