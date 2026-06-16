"""Parse the free-text 'Ermittler' column into clean team features."""

import re
from pathlib import Path

import duckdb
import pandas as pd

DB = Path("data/processed/tatort.duckdb")


def parse_team(raw: str) -> dict:
    """Turn 'Ballauf und Schenk (Gastauftritt X)' into structured fields."""
    s = str(raw)
    # Remove parenthetical guest-appearance notes.
    s = re.sub(r"\(.*?\)", "", s).strip()
    # Split investigators on 'und' or commas.
    parts = re.split(r"\s+und\s+|,\s*", s)
    investigators = [p.strip() for p in parts if p.strip()]
    # Canonical team key: surnames sorted + joined, so ordering quirks don't
    # create two "different" teams for the same pair.
    team_key = " & ".join(sorted(investigators)) if investigators else "Unknown"
    return {
        "team_key": team_key,
        "n_investigators": len(investigators),
        "lead_investigator": investigators[0] if investigators else "Unknown",
    }


def main(min_episodes: int = 8) -> None:
    con = duckdb.connect(str(DB))
    df = con.execute("SELECT Folge, Ermittler FROM episodes").fetchdf()

    parsed = df["Ermittler"].apply(parse_team).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Bucket rare teams into 'Other' to avoid one-hot explosion.
    counts = df["team_key"].value_counts()
    common = counts[counts >= min_episodes].index
    df["team_grouped"] = df["team_key"].where(df["team_key"].isin(common), "Other")

    n_distinct = df["team_key"].nunique()
    n_kept = len(common)
    n_other = (df["team_grouped"] == "Other").sum()
    print(f"Distinct raw teams: {n_distinct}")
    print(f"Teams kept (>= {min_episodes} episodes): {n_kept}")
    print(f"Episodes bucketed as 'Other': {n_other}")

    # Write the new columns back into DuckDB.
    con.execute("DROP TABLE IF EXISTS team_features")
    con.register("tmp_df", df[["Folge", "team_key", "team_grouped", "n_investigators", "lead_investigator"]])
    con.execute("CREATE TABLE team_features AS SELECT * FROM tmp_df")
    print("\nSaved team_features table to DuckDB.")

    print("\nTop 12 teams by episode count:")
    print(counts.head(12).to_string())
    con.close()


if __name__ == "__main__":
    main()