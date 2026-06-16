"""Extract death/violence features from plot text; validate vs tatort-fundus."""

import re
from pathlib import Path

import duckdb
import pandas as pd

DB = Path("data/processed/tatort.duckdb")

# Death-indicating terms. We count sentences containing a death cue rather than
# raw keyword hits, to avoid double-counting one death described several ways.
DEATH_CUES = [
    r"\bleiche\b", r"\bleichen\b", r"\btote[rn]?\b", r"\btoten\b",
    r"\bermordet", r"\bgetötet", r"\bgetotet", r"\berschossen",
    r"\berstochen", r"\berschlagen", r"\bvergiftet", r"\berdrosselt",
    r"\berwürgt", r"\berwurgt", r"\bumgebracht", r"\bstirbt\b", r"\bstarb\b",
    r"\bmord\b", r"\bopfer\b",
]
# Cause-of-death classification — tightened to specific death terms,
# avoiding greedy stems like 'schlag' that match unrelated words.
CAUSES = {
    "erschossen": r"\berschossen\b|\bniedergeschossen\b|\bkugel\b",
    "erschlagen": r"\berschlagen\b|\btotgeschlagen\b",
    "vergiftet": r"\bvergiftet\b|\bgift\b",
    "erstochen": r"\berstochen\b|\bniedergestochen\b|\berstach\b",
    "erwürgt": r"\berwürgt\b|\berwurgt\b|\berdrosselt\b|\bstranguliert\b",
}
SOLVED_CUES = r"festgenommen|verhaftet|gesteht|geständnis|überführt|uberfuhrt|gefasst|gestellt"


def count_deaths(text: str) -> int:
    """Count sentences that contain at least one death cue (a proxy for deaths)."""
    sentences = re.split(r"[.!?]\s+", text.lower())
    cue_re = re.compile("|".join(DEATH_CUES))
    return sum(1 for s in sentences if cue_re.search(s))


def classify_causes(text: str) -> dict:
    t = text.lower()
    return {f"cause_{name}": int(bool(re.search(pat, t))) for name, pat in CAUSES.items()}


def is_solved(text: str) -> int:
    return int(bool(re.search(SOLVED_CUES, text.lower())))


def main() -> None:
    con = duckdb.connect(str(DB))
    df = con.execute(
        "SELECT Folge, plot FROM plots WHERE plot IS NOT NULL").fetchdf()

    df["death_score"] = df["plot"].map(count_deaths)
    df["solved"] = df["plot"].map(is_solved)
    causes = df["plot"].map(classify_causes).apply(pd.Series)
    feat = pd.concat([df[["Folge", "death_score", "solved"]], causes], axis=1)

    # --- VALIDATION against tatort-fundus benchmark (2.3 deaths/episode) ---
    avg = df["death_score"].mean()
    print("=== Validation vs tatort-fundus ===")
    print(f"Our mean 'death_score' per episode: {avg:.2f}")
    print(f"tatort-fundus benchmark:            2.30")
    print(f"Episodes with zero deaths detected: {(df['death_score'] == 0).sum()} "
          f"(tatort-fundus reports 21 with no corpse)")
    print(f"\nCause-of-death frequency (our detection):")
    for c in CAUSES:
        print(f"  {c:12s}: {causes[f'cause_{c}'].sum()}")
    print("  tatort-fundus order: erschossen(856) > erschlagen(254) > vergiftet(175)")

    con.execute("DROP TABLE IF EXISTS death_features")
    con.register("tmp_death", feat)
    con.execute("CREATE TABLE death_features AS SELECT * FROM tmp_death")
    con.close()
    print("\nSaved death_features to DuckDB.")


if __name__ == "__main__":
    main()
