FROM apache/airflow:2.10.3

# Install the pipeline's Python dependencies into the Airflow image.
# (Same set as your pipeline image, minus what Airflow already has.)
RUN pip install --no-cache-dir \
    "duckdb>=1.5" "pyarrow" "beautifulsoup4>=4.15" \
    "scikit-learn>=1.9" "xgboost>=3.2" "thefuzz>=0.22" \
    "sentence-transformers>=5.5" "bertopic>=0.17" "stop-words"