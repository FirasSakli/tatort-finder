"""Load the joined episode dataset into DuckDB for SQL querying."""

from pathlib import Path

import duckdb

SRC = Path("data/interim/episodes_rated.parquet")
DB = Path("data/processed/tatort.duckdb")


def main() -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute("DROP TABLE IF EXISTS episodes")
    con.execute(
        f"CREATE TABLE episodes AS SELECT * FROM read_parquet('{SRC.as_posix()}')")

    n = con.execute("SELECT count(*) FROM episodes").fetchone()[0]
    rated = con.execute(
        "SELECT count(*) FROM episodes WHERE averageRating IS NOT NULL").fetchone()[0]
    print(f"Loaded {n} episodes ({rated} rated) into {DB}")

    # A taste of what SQL queries now look like: best-rated teams.
    print("\nTop 10 teams by average IMDb rating (min 10 episodes):")
    q = """
        SELECT Sender,
               count(*)              AS episodes,
               round(avg(averageRating), 2) AS avg_rating
        FROM episodes
        WHERE averageRating IS NOT NULL
        GROUP BY Sender
        HAVING count(*) >= 10
        ORDER BY avg_rating DESC
        LIMIT 10
    """
    print(con.execute(q).fetchdf().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
