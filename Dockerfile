# SLM Project - Production Docker Environment
# Build: docker build -t slm-project .
# Run: docker run -p 8000:8000 slm-project

FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN useradd -m -u 1000 slm && \
    mkdir -p /app/slm-comparison /app/benchmarks /app/logs && \
    chown -R slm:slm /app

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files. The .dockerignore keeps training data, adapters and
# checkpoints out; mount them at run time instead (see docker-compose.yml).
COPY --chown=slm:slm . .

# Switch to non-root user
USER slm

# Expose API and metrics ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: start the FastAPI server
CMD ["python", "-m", "uvicorn", "slm_api:app", "--host", "0.0.0.0", "--port", "8000"]
