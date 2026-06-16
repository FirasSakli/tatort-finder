"""Download IMDb's free datasets and extract Tatort episode ratings.

IMDb publishes daily bulk TSVs at https://datasets.imdbws.com/ for
NON-COMMERCIAL use. We pull three, filter to Tatort's parent series,
and join them into one ratings table keyed by episode number.
"""

import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests

PARENT_TCONST = "tt0806910"  # the Tatort series on IMDb
BASE = "https://datasets.imdbws.com"
FILES = {
    "episode": "title.episode.tsv.gz",   # maps episodes -> parent series + season/ep
    "ratings": "title.ratings.tsv.gz",    # averageRating, numVotes
    "basics": "title.basics.tsv.gz",      # title, year, runtime, genres
}
RAW = Path("data/raw/imdb")
OUT = Path("data/raw/imdb_ratings.parquet")
HEADERS = {"User-Agent": "TatortDataScience/0.1 (portfolio project; firassakli82@gmail.com)"}


def download(name: str, fname: str) -> Path:
    """Download one gzipped TSV if we don't already have it."""
    RAW.mkdir(parents=True, exist_ok=True)
    gz_path = RAW / fname
    if gz_path.exists():
        print(f"  {fname} already downloaded, skipping.")
        return gz_path
    print(f"  downloading {fname} ...")
    with requests.get(f"{BASE}/{fname}", headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return gz_path


def read_tsv(gz_path: Path, usecols: list[str]) -> pd.DataFrame:
    """Read a gzipped IMDb TSV. '\\N' is IMDb's null marker."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        return pd.read_csv(f, sep="\t", usecols=usecols, na_values="\\N", low_memory=False)


def main() -> None:
    print("Step 1: download IMDb datasets (these are large; first run is slow)...")
    paths = {k: download(k, v) for k, v in FILES.items()}

    print("Step 2: find all Tatort episodes...")
    episode = read_tsv(paths["episode"], ["tconst", "parentTconst", "seasonNumber", "episodeNumber"])
    tatort = episode[episode["parentTconst"] == PARENT_TCONST].copy()
    print(f"  found {len(tatort)} Tatort episode entries on IMDb.")

    print("Step 3: join ratings + basics...")
    ratings = read_tsv(paths["ratings"], ["tconst", "averageRating", "numVotes"])
    basics = read_tsv(paths["basics"], ["tconst", "primaryTitle", "startYear", "runtimeMinutes"])

    df = tatort.merge(ratings, on="tconst", how="left").merge(basics, on="tconst", how="left")
    rated = df["averageRating"].notna().sum()
    print(f"  {rated} of {len(df)} episodes have a rating.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nSaved -> {OUT.resolve()}")
    print(df[["primaryTitle", "startYear", "averageRating", "numVotes"]].head())


if __name__ == "__main__":
    main()