"""Rating predictor: metadata + team + plot embeddings, with ablation."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DB = Path("data/processed/tatort.duckdb")
EMB = Path("data/processed/plot_embeddings.parquet")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(DB))
    df = con.execute(
        """
        SELECT e.Folge, e.Sender, e.year, e.runtimeMinutes, e.averageRating,
               t.team_grouped, t.n_investigators
        FROM episodes e
        LEFT JOIN team_features t USING (Folge)
        WHERE e.averageRating IS NOT NULL
        """
    ).fetchdf()
    con.close()
    emb = pd.read_parquet(EMB)
    return df, emb


def build_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    X = df[["Sender", "year", "decade", "runtimeMinutes",
            "team_grouped", "n_investigators"]].copy()
    return pd.get_dummies(X, columns=["Sender", "team_grouped"], dummy_na=True)


def evaluate(X, y, label, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    model = XGBRegressor(n_estimators=300, max_depth=4,
                         learning_rate=0.05, random_state=seed)
    model.fit(Xtr, ytr)
    mae = mean_absolute_error(yte, model.predict(Xte))
    base = mean_absolute_error(yte, np.full_like(yte, ytr.mean()))
    print(f"{label:32s} MAE {mae:.3f}  (baseline {base:.3f}, +{(1-mae/base)*100:.1f}%)")
    return mae


def main() -> None:
    df, emb = load()

    # Align embeddings to the rated episodes by Folge.
    merged = df.merge(emb, on="Folge", how="inner")
    print(f"Episodes with rating AND embedding: {len(merged)}\n")

    y = merged["averageRating"]
    meta = build_meta(merged)
    emb_cols = [c for c in merged.columns if c.startswith("emb_")]
    emb_only = merged[emb_cols]

    # Ablation: each feature set alone, then combined.
    evaluate(meta, y, "Metadata + team only")
    evaluate(emb_only, y, "Plot embeddings only")
    evaluate(pd.concat([meta.reset_index(drop=True),
                        emb_only.reset_index(drop=True)], axis=1),
             y, "Metadata + team + embeddings")

    # Reduce embeddings to a handful of components to curb overfitting.
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for n in [10, 30, 50]:
        pca = make_pipeline(StandardScaler(), PCA(
            n_components=n, random_state=42))
        emb_reduced = pd.DataFrame(
            pca.fit_transform(emb_only),
            columns=[f"pca_{i}" for i in range(n)],
        )
        combined = pd.concat(
            [meta.reset_index(drop=True), emb_reduced], axis=1)
        evaluate(combined, y, f"Metadata + team + {n} PCA dims")    


if __name__ == "__main__":
    main()
