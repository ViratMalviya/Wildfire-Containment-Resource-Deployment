# ──────────────────────────────────────────────
# Dockerfile — Wildfire Containment RL + MLOps
# ──────────────────────────────────────────────
# Containerized environment for reproducible training,
# evaluation, and MLflow experiment tracking.
#
# Usage:
#   docker build -t wildfire-rl .
#   docker run -v $(pwd)/results:/app/results wildfire-rl python train.py --config configs/qlearning_v1.yaml
# ──────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="Wildfire-RL Team"
LABEL description="Wildfire Containment & Resource Deployment — RL + MLOps"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY sim/ sim/
COPY src/ src/
COPY configs/ configs/
COPY train.py .
COPY evaluate.py .

# Create output directories
RUN mkdir -p models results mlruns experiments

# Default: train with v1 config
CMD ["python", "train.py", "--config", "configs/qlearning_v1.yaml"]
