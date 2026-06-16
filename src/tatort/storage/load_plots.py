"""Load scraped plot summaries into DuckDB, joined to episodes by title."""

import json
import re
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd

DB = Path("data/processed/tatort.duckdb")
PLOTS = Path("data/raw/plots.jsonl")


def normalize_title(s: str) -> str:
    s = str(s).lower().strip().replace("ß", "ss")
    s = s.split("(")[0]
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    rows = []
    with open(PLOTS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["status"] == "ok" and rec["plot"]:
                rows.append({"title": rec["title"], "plot": rec["plot"],
                             "plot_len": len(rec["plot"])})
    plots = pd.DataFrame(rows)
    plots["title_norm"] = plots["title"].map(normalize_title)
    plots = plots.drop_duplicates("title_norm")
    print(f"Loaded {len(plots)} unique plots.")

    con = duckdb.connect(str(DB))
    # Build a normalized title on the episodes side to join against.
    eps = con.execute("SELECT Folge, Titel FROM episodes").fetchdf()
    eps["title_norm"] = eps["Titel"].map(normalize_title)

    merged = eps.merge(plots[["title_norm", "plot", "plot_len"]], on="title_norm", how="left")
    matched = merged["plot"].notna().sum()
    print(f"Matched plots to {matched} / {len(eps)} episodes.")

    con.execute("DROP TABLE IF EXISTS plots")
    con.register("tmp_plots", merged[["Folge", "plot", "plot_len"]])
    con.execute("CREATE TABLE plots AS SELECT * FROM tmp_plots")
    print("Saved plots table to DuckDB.")
    con.close()


if __name__ == "__main__":
    main()