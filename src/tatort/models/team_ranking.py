"""Rank Tatort teams by violence, solve rate, and cause-of-death profile."""

from pathlib import Path

import duckdb
import pandas as pd

DB = Path("data/processed/tatort.duckdb")


def main(min_episodes: int = 10) -> None:
    con = duckdb.connect(str(DB))
    df = con.execute(
        """
        SELECT t.team_grouped AS team,
               d.death_score,
               d.solved,
               p.plot_len,
               e.averageRating,
               d.cause_erschossen, d.cause_erschlagen, d.cause_vergiftet,
               d.cause_erstochen, d.cause_erwürgt
        FROM team_features t
        JOIN death_features d USING (Folge)
        JOIN plots p USING (Folge)
        JOIN episodes e USING (Folge)
        WHERE t.team_grouped != 'Other' AND p.plot IS NOT NULL
        """
    ).fetchdf()
    con.close()

    # Length-normalized violence: deaths mentioned per 1000 chars of plot.
    df["death_per_1k"] = df["death_score"] / df["plot_len"] * 1000

    agg = df.groupby("team").agg(
        episodes=("death_score", "size"),
        avg_violence=("death_score", "mean"),
        violence_norm=("death_per_1k", "mean"),
        solve_rate=("solved", "mean"),
        avg_rating=("averageRating", "mean"),
    ).round(2)
    agg = agg[agg["episodes"] >= min_episodes]

    print("=== BLOODIEST TEAMS (by raw violence intensity) ===")
    print(agg.sort_values("avg_violence", ascending=False)
          [["episodes", "avg_violence", "violence_norm", "avg_rating"]].head(10).to_string())

    print("\n=== COSIEST TEAMS (lowest violence) ===")
    print(agg.sort_values("avg_violence")
          [["episodes", "avg_violence", "violence_norm", "avg_rating"]].head(5).to_string())

    print("\n=== Does length-normalizing change the top 5? ===")
    raw_top = set(agg.sort_values("avg_violence", ascending=False).head(5).index)
    norm_top = set(agg.sort_values("violence_norm", ascending=False).head(5).index)
    print(f"Raw top-5:  {sorted(raw_top)}")
    print(f"Norm top-5: {sorted(norm_top)}")
    print(f"Overlap: {len(raw_top & norm_top)}/5 teams appear in both")

    # Save the ranking for the dashboard later.
    con = duckdb.connect(str(DB))
    con.execute("DROP TABLE IF EXISTS team_ranking")
    con.register("tmp_rank", agg.reset_index())
    con.execute("CREATE TABLE team_ranking AS SELECT * FROM tmp_rank")
    con.close()
    print("\nSaved team_ranking to DuckDB.")


if __name__ == "__main__":
    main()  