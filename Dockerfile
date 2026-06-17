# Slim Python base — small, official, matches your 3.12 requirement.
FROM python:3.12-slim

# System deps some Python libs need at build/runtime (git for a few, build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install ONLY what the pipeline actually imports — not the app/unused deps.
# Pinned loosely; pip resolves compatible versions.
RUN pip install --no-cache-dir \
    "pandas>=2.3" "numpy" "duckdb>=1.5" "pyarrow" \
    "requests>=2.34" "beautifulsoup4>=4.15" \
    "scikit-learn>=1.9" "xgboost>=3.2" "thefuzz>=0.22" \
    "sentence-transformers>=5.5" "bertopic>=0.17" \
    "stop-words"

# Copy the source code and pipeline entrypoint.
COPY src/ ./src/
COPY run_pipeline.py ./

# Make the package importable (we copied src/ but didn't pip-install the project).
ENV PYTHONPATH=/app/src
# Send the HF model cache to a path we'll mount as a volume (avoids re-downloading).
ENV HF_HOME=/app/data/hf_cache

# Default: run the full pipeline. Override with e.g. `... python run_pipeline.py embeddings`
CMD ["python", "run_pipeline.py"]