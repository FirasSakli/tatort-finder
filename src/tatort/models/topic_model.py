"""Topic modeling over Tatort plots, with names stripped and tuned clustering."""

import re
from pathlib import Path

import duckdb
import pandas as pd
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words

DB = Path("data/processed/tatort.duckdb")
EMB = Path("data/processed/plot_embeddings.parquet")


def build_name_set(con) -> set[str]:
    """Collect investigator surnames so we can strip them from plot text."""
    teams = con.execute(
        "SELECT DISTINCT team_key FROM team_features").fetchdf()
    names = set()
    for key in teams["team_key"].dropna():
        for part in re.split(r"\s*&\s*", key):
            p = part.strip().lower()
            if len(p) > 2:
                names.add(p)
    return names


def strip_names(text: str, names: set[str]) -> str:
    """Remove investigator surnames (and capitalized name-like tokens are kept)."""
    words = re.findall(r"\b\w+\b", text.lower())
    return " ".join(w for w in words if w not in names)


def main(n_topics: int = 14) -> None:
    con = duckdb.connect(str(DB))
    df = con.execute(
        "SELECT Folge, plot FROM plots WHERE plot IS NOT NULL ORDER BY Folge").fetchdf()
    names = build_name_set(con)
    con.close()

    emb = pd.read_parquet(EMB).set_index("Folge").loc[df["Folge"]].values
    docs = [strip_names(p, names) for p in df["plot"]]

    # Proper German stopwords + extra narrative filler.
    stops = get_stop_words("german") + [
        "hatte", "habe", "ihn", "da", "dr", "kommt", "immer", "mehr", "gibt",
        "wurde", "wurden", "sei", "seien", "worden", "macht", "geht", "sagt",
        "findet", "kommt", "zwei", "drei", "schließlich", "schliesslich",
    ]
    vectorizer = CountVectorizer(
        stop_words=stops, min_df=5, ngram_range=(1, 2))

    # KMeans forces every episode into a topic — no giant outlier pile.
    cluster_model = KMeans(n_clusters=n_topics, random_state=42, n_init=10)

    print(
        f"Fitting BERTopic with {n_topics} KMeans clusters (names stripped)...")
    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        hdbscan_model=cluster_model,
        language="multilingual",
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=emb)

    info = topic_model.get_topic_info()
    print(f"\n{len(info)} topics:\n")
    for _, row in info.iterrows():
        words = [w for w, _ in topic_model.get_topic(row["Topic"])][:8]
        print(
            f"Topic {row['Topic']:2d} ({row['Count']:3d} eps): {', '.join(words)}")

    out = pd.DataFrame({"Folge": df["Folge"].values, "topic": topics})
    con = duckdb.connect(str(DB))
    con.execute("DROP TABLE IF EXISTS topics")
    con.register("tmp_topics", out)
    con.execute("CREATE TABLE topics AS SELECT * FROM tmp_topics")
    con.close()
    print("\nSaved per-episode topics to DuckDB.")


if __name__ == "__main__":
    main()
