"""Single entrypoint that runs the full Tatort data pipeline in dependency order.

Each step is a separate callable so it can be run standalone or orchestrated
(e.g. by Airflow) later. Run all: `python run_pipeline.py`
Run one:  `python run_pipeline.py embeddings`
"""

import importlib
import sys
import time

# (step name, module path) in strict dependency order.
STEPS = [
    ("scrape_list",   "tatort.ingestion.wikipedia_list"),
    ("imdb",          "tatort.ingestion.imdb_datasets"),
    ("clean",         "tatort.storage.clean"),
    ("load",          "tatort.storage.load"),
    ("team_features", "tatort.processing.team_features"),
    ("scrape_plots",  "tatort.ingestion.wikipedia_plots"),
    ("load_plots",    "tatort.storage.load_plots"),
    ("embeddings",    "tatort.processing.embeddings"),
    ("death_features","tatort.processing.death_features"),
]


def run_step(name: str, module_path: str) -> None:
    print(f"\n{'='*60}\n▶ STEP: {name}  ({module_path})\n{'='*60}", flush=True)
    t0 = time.time()
    mod = importlib.import_module(module_path)
    mod.main()
    print(f"✓ {name} done in {time.time()-t0:.1f}s", flush=True)


def main() -> None:
    # Optional arg: run a single named step.
    if len(sys.argv) > 1:
        target = sys.argv[1]
        match = [(n, m) for n, m in STEPS if n == target]
        if not match:
            print(f"Unknown step '{target}'. Valid: {[n for n,_ in STEPS]}")
            sys.exit(1)
        run_step(*match[0])
        return
    # Otherwise run the whole pipeline.
    for name, module_path in STEPS:
        run_step(name, module_path)
    print("\n🎉 Full pipeline complete.")


if __name__ == "__main__":
    main()