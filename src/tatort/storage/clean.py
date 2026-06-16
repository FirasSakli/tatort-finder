"""Join Wikipedia episode metadata with IMDb ratings on normalized title + year."""

import re
import unicodedata
from pathlib import Path
from thefuzz import process as fuzz_process

import pandas as pd

WIKI = Path("data/raw/episodes_list.parquet")
IMDB = Path("data/raw/imdb_ratings.parquet")
OUT = Path("data/interim/episodes_rated.parquet")

GERMAN_MONTHS = {
    "Jan": 1, "Feb": 2, "März": 3, "Mär": 3, "Apr": 4, "Mai": 5, "Juni": 6,
    "Jun": 6, "Juli": 7, "Jul": 7, "Aug": 8, "Sep": 9, "Okt": 10, "Nov": 11, "Dez": 12,
}


def normalize_title(s: str) -> str:
    """Lowercase, strip accents, drop punctuation, collapse whitespace."""
    s = s.replace("ß", "ss")   # unify old/new German spelling for matching
    s = str(s).lower().strip()
    s = str(s).lower().strip()
    # take only the part before any parenthetical note Wikipedia adds
    s = s.split("(")[0]
    # strip accents (ä -> a etc.) for robust matching
    s = "".join(c for c in unicodedata.normalize(
        "NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s)   # drop punctuation incl. ellipses
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_year(date_str: str) -> int | None:
    """Pull the 4-digit year out of a German date like '29. Nov. 1970'."""
    m = re.search(r"(\d{4})", str(date_str))
    return int(m.group(1)) if m else None


def main() -> None:
    wiki = pd.read_parquet(WIKI)
    imdb = pd.read_parquet(IMDB)

    wiki["title_norm"] = wiki["Titel"].map(normalize_title)
    wiki["year"] = wiki["Erstausstrahlung"].map(extract_year)

    imdb["title_norm"] = imdb["primaryTitle"].map(normalize_title)
    imdb["year"] = imdb["startYear"]

    # First try matching on title + year (safest).
    merged = wiki.merge(
        imdb[["title_norm", "year", "averageRating",
              "numVotes", "runtimeMinutes", "tconst"]],
        on=["title_norm", "year"],
        how="left",
    )
    matched_ty = merged["averageRating"].notna().sum()
    print(f"Matched on title+year: {matched_ty} / {len(wiki)}")

    # For the leftovers, retry on title alone (handles year-off-by-one cases).
    unmatched = merged[merged["averageRating"].isna()][["Folge", "title_norm"]]
    retry = unmatched.merge(
        imdb[["title_norm", "averageRating", "numVotes", "runtimeMinutes", "tconst"]],
        on="title_norm",
        how="left",
    ).dropna(subset=["averageRating"]).drop_duplicates("Folge")

    # Patch the retry matches back in.
    merged = merged.set_index("Folge")
    for _, row in retry.iterrows():
        f = row["Folge"]
        for col in ["averageRating", "numVotes", "runtimeMinutes", "tconst"]:
            merged.loc[f, col] = row[col]
    merged = merged.reset_index()

    # Final fallback: fuzzy-match remaining stragglers against IMDb titles.
    imdb_titles = imdb["title_norm"].tolist()
    imdb_lookup = imdb.set_index("title_norm")
    already_used = set(merged["tconst"].dropna())  # don't reuse IMDb entries
    still_unmatched = merged[merged["averageRating"].isna()]
    print(f"\nFuzzy-matching {len(still_unmatched)} stragglers...")
    for _, row in still_unmatched.iterrows():
        best, score = fuzz_process.extractOne(row["title_norm"], imdb_titles)
        if score < 90:
            print(
                f"  SKIP '{row['Titel']}' -> '{best}' (score {score}) — too low")
            continue
        src = imdb_lookup.loc[best]
        if isinstance(src, pd.DataFrame):
            src = src.iloc[0]
        if src["tconst"] in already_used:
            print(
                f"  SKIP '{row['Titel']}' -> '{best}' — IMDb entry already used")
            continue
        already_used.add(src["tconst"])
        for col in ["averageRating", "numVotes", "runtimeMinutes", "tconst"]:
            merged.loc[merged["Folge"] == row["Folge"], col] = src[col]
        print(f"  MATCH '{row['Titel']}' -> '{best}' (score {score})")

    total_matched = merged["averageRating"].notna().sum()
    print(f"Matched after title-only retry: {total_matched} / {len(wiki)}")
    print(f"Still unmatched: {len(wiki) - total_matched}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    print(f"\nSaved -> {OUT.resolve()}")

    # Show a few unmatched titles so we can judge if fuzzy matching is needed.
    still = merged[merged["averageRating"].isna()]
    if len(still):
        print("\nSample of unmatched episodes:")
        print(still[["Folge", "Titel", "year"]].head(
            10).to_string(index=False))


if __name__ == "__main__":
    main()
