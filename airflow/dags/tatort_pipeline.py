"""Weekly Tatort data pipeline — orchestrates the full scrape→features flow."""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# Each pipeline step, in dependency order. We run them via the project's
# run_pipeline entrypoint, executed inside the worker with the mounted code.
STEPS = [
    "scrape_list", "imdb", "clean", "load", "team_features",
    "scrape_plots", "load_plots", "embeddings", "death_features",
]

# The worker sees the project code/data at /opt/airflow/project (mounted volumes).
PROJECT = "/opt/airflow/project"
ENV = f"PYTHONPATH={PROJECT}/src HF_HOME={PROJECT}/data/hf_cache"

default_args = {"retries": 1}

with DAG(
    dag_id="tatort_pipeline",
    description="Scrape and process 50+ years of Tatort episodes",
    schedule="0 22 * * 0",          # Sundays 22:00 — after the new episode airs
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["tatort", "portfolio"],
) as dag:

    tasks = {}
    for step in STEPS:
        tasks[step] = BashOperator(
            task_id=step,
           bash_command=(
                f"cd {PROJECT} && PYTHONPATH={PROJECT}/src "
                f"HF_HOME={PROJECT}/data/hf_cache "
                f"python {PROJECT}/src/run_pipeline.py {step}"
            ),
        )

    # Wire them in strict dependency order: each step waits for the previous.
    previous = None
    for step in STEPS:
        if previous is not None:
            previous >> tasks[step]
        previous = tasks[step]  