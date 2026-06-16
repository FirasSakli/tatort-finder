"""Episode recommender based on plot-embedding similarity."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DB = Path("data/processed/tatort.duckdb")
EMB = Path("data/processed/plot_embeddings.parquet")


def load():
    con = duckdb.connect(str(DB))
    meta = con.execute(
        """
        SELECT e.Folge, e.Titel, e.year, e.averageRating,
               t.team_grouped AS team
        FROM episodes e
        LEFT JOIN team_features t USING (Folge)
        """
    ).fetchdf()
    con.close()
    emb = pd.read_parquet(EMB)
    # Keep only episodes that have an embedding.
    meta = meta[meta["Folge"].isin(emb["Folge"])].reset_index(drop=True)
    emb = emb.set_index("Folge").loc[meta["Folge"]].reset_index()
    return meta, emb


def recommend(title_query: str, meta, sim_matrix, k: int = 5):
    """Find k episodes most similar in plot to the queried title."""
    # Case-insensitive partial title match.
    hits = meta[meta["Titel"].str.lower().str.contains(title_query.lower(), na=False)]
    if hits.empty:
        print(f"No episode matching '{title_query}'.")
        return
    idx = hits.index[0]
    ep = meta.loc[idx]
    print(f"\nBecause you watched: '{ep['Titel']}' ({int(ep['year'])}, {ep['team']}, "
          f"rating {ep['averageRating']})")
    print("-" * 60)

    sims = sim_matrix[idx]
    # Sort by similarity, skip the episode itself (highest = itself).
    order = np.argsort(sims)[::-1]
    shown = 0
    for j in order:
        if j == idx:
            continue
        rec = meta.loc[j]
        print(f"  {sims[j]:.3f}  '{rec['Titel']}' ({int(rec['year'])}, "
              f"{rec['team']}, rating {rec['averageRating']})")
        shown += 1
        if shown >= k:
            break


def main():
    meta, emb = load()
    vectors = emb.drop(columns="Folge").values
    print(f"Computing similarity over {len(meta)} episodes...")
    sim_matrix = cosine_similarity(vectors)

    # A few demo queries to sanity-check the recommendations.
    for q in ["Das Nest", "Im Schmerz geboren", "Mord"]:
        recommend(q, meta, sim_matrix, k=5)


if __name__ == "__main__":
    main()