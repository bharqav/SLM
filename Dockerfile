# SLM Project - Docker Environment
# Build: docker build -t slm-project .
# CPU run: docker run -p 8000:8000 slm-project
# GPU run: docker run --gpus all -p 8000:8000 slm-project

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p slm-comparison benchmarks

# Default command: run the full pipeline
CMD ["python", "run_all.py"]
