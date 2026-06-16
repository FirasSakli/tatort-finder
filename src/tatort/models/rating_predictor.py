"""Rating predictor: XGBoost on episode metadata + team features."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DB = Path("data/processed/tatort.duckdb")


def load() -> pd.DataFrame:
    con = duckdb.connect(str(DB))
    # Join the episode table with the parsed team features.
    df = con.execute(
        """
        SELECT e.Folge, e.Sender, e.year, e.runtimeMinutes, e.numVotes,
               e.averageRating,
               t.team_grouped, t.n_investigators
        FROM episodes e
        LEFT JOIN team_features t USING (Folge)
        WHERE e.averageRating IS NOT NULL
        """
    ).fetchdf()
    con.close()
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    X = df[["Sender", "year", "decade", "runtimeMinutes",
            "team_grouped", "n_investigators"]].copy()
    X = pd.get_dummies(X, columns=["Sender", "team_grouped"], dummy_na=True)
    y = df["averageRating"]
    return X, y


def main() -> None:
    df = load()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    baseline = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
    print(f"Baseline MAE (always predict the mean): {baseline:.3f}")
    print(f"XGBoost MAE (metadata + team):          {mae:.3f}")
    print(f"Improvement over baseline:              {(1 - mae/baseline)*100:.1f}%")
    print(f"(Previous metadata-only MAE was 0.450)")


if __name__ == "__main__":
    main()