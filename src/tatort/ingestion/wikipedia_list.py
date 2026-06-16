"""Scrape the German Wikipedia 'Liste der Tatort-Folgen' into a clean episode table."""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

URL = "https://de.wikipedia.org/wiki/Liste_der_Tatort-Folgen"
OUT = Path("data/raw/episodes_list.parquet")
HEADERS = {
    "User-Agent": "TatortDataScience/0.1 (portfolio project; your_email@example.com)"
}


def fetch_tables() -> list[pd.DataFrame]:
    """Download the page and parse every HTML table into a DataFrame."""
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def find_episode_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Keep only the tables that look like episode lists and stack them."""
    keep = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        has_number = any("nr" in c or "folge" in c for c in cols)
        has_title = any("titel" in c for c in cols)
        if has_number and has_title:
            keep.append(t)
    if not keep:
        raise RuntimeError(
            "No episode tables found — page layout may have changed.")
    return pd.concat(keep, ignore_index=True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tables = fetch_tables()
    print(f"Found {len(tables)} tables on the page.")
    episodes = find_episode_tables(tables)
    print(f"Combined into {len(episodes)} candidate episode rows.")

    # Coerce the episode number to a real integer; rows where this fails
    # (sub-headers, not-yet-aired placeholders with '–') become NaN...
    episodes["Folge"] = pd.to_numeric(episodes["Folge"], errors="coerce")
    before = len(episodes)
    # ...and we drop them, keeping only genuine numbered episodes.
    episodes = episodes.dropna(subset=["Folge"])
    episodes["Folge"] = episodes["Folge"].astype(int)
    print(
        f"Dropped {before - len(episodes)} non-episode rows; {len(episodes)} remain.")

    # Force every other column to string so Parquet has a clean schema.
    for col in episodes.columns:
        if col != "Folge":
            episodes[col] = episodes[col].astype(str)

    episodes = episodes.sort_values("Folge").reset_index(drop=True)
    episodes.to_parquet(OUT, index=False)
    print(f"Saved {len(episodes)} episodes -> {OUT.resolve()}")
    print("\nFirst few rows:")
    print(episodes[["Folge", "Titel", "Sender", "Erstausstrahlung"]].head())


if __name__ == "__main__":
    main()
