"""Generate sentence embeddings for each episode's plot summary."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DB = Path("data/processed/tatort.duckdb")
OUT = Path("data/processed/plot_embeddings.parquet")
# Multilingual model — handles German well, small and fast.
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    con = duckdb.connect(str(DB))
    df = con.execute(
        "SELECT Folge, plot FROM plots WHERE plot IS NOT NULL"
    ).fetchdf()
    con.close()
    print(f"Encoding {len(df)} plots with {MODEL} ...")
    print("(First run downloads the model, ~120 MB — be patient.)")

    model = SentenceTransformer(MODEL)
    # encode returns one vector per plot; show a progress bar.
    embeddings = model.encode(
        df["plot"].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")  # (n_plots, 384)

    # Store as a parquet: Folge + one column per embedding dimension.
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    emb_df.insert(0, "Folge", df["Folge"].values)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(OUT, index=False)
    print(f"Saved -> {OUT.resolve()}")


if __name__ == "__main__":
    main()