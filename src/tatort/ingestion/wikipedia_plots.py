"""Scrape plot ('Handlung') sections from individual Tatort Wikipedia articles.

Designed to be polite (rate-limited), resumable (saves incrementally, skips
already-fetched episodes), and failure-tolerant (logs misses, never crashes).
"""

import json
import time
from pathlib import Path

import duckdb
import requests

DB = Path("data/processed/tatort.duckdb")
OUT = Path("data/raw/plots.jsonl")   # one JSON object per line; append-friendly
API = "https://de.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "TatortDataScience/0.1 (portfolio project; your_email@example.com)"}
DELAY = 0.5  # seconds between requests — be a good citizen

# Section headings that contain the plot, in rough order of preference.
PLOT_HEADINGS = ["Handlung", "Inhalt"]


def already_done() -> set[str]:
    """Titles we've already fetched, so a restart skips them."""
    if not OUT.exists():
        return set()
    done = set()
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["title"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def fetch_plot(title: str) -> dict:
    """Fetch one article's plain text and extract the plot section."""
    wiki_title = f"Tatort: {title}"
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "titles": wiki_title, "redirects": 1,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        return {"title": title, "status": "no_article", "plot": None}

    text = page.get("extract", "")
    plot = extract_section(text, PLOT_HEADINGS)
    status = "ok" if plot else "no_plot_section"
    return {"title": title, "status": status, "plot": plot,
            "full_len": len(text)}


def extract_section(text: str, headings: list[str]) -> str | None:
    """Pull the text under the first matching '== Heading ==' until the next heading."""
    lines = text.split("\n")
    capturing = False
    collected = []
    for line in lines:
        stripped = line.strip()
        # Section headers in plaintext extracts look like '== Handlung =='.
        is_header = stripped.startswith("==") and stripped.endswith("==")
        if is_header:
            name = stripped.strip("= ").strip()
            if name in headings:
                capturing = True
                continue
            if capturing:  # hit the next section -> stop
                break
        elif capturing and stripped:
            collected.append(stripped)
    return " ".join(collected) if collected else None


def main() -> None:
    con = duckdb.connect(str(DB))
    titles = [r[0] for r in con.execute("SELECT Titel FROM episodes ORDER BY Folge").fetchall()]
    con.close()

    done = already_done()
    todo = [t for t in titles if t not in done]
    print(f"{len(titles)} episodes total; {len(done)} already fetched; {len(todo)} to go.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "no_article": 0, "no_plot_section": 0, "error": 0}

    with open(OUT, "a", encoding="utf-8") as f:
        for i, title in enumerate(todo, 1):
            try:
                rec = fetch_plot(title)
            except Exception as e:  # network blip etc. — log and continue
                rec = {"title": title, "status": "error", "plot": None, "error": str(e)}
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()  # write immediately so a crash loses nothing
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ... {counts}")
            time.sleep(DELAY)

    print(f"\nDone. Results: {counts}")
    print(f"Saved -> {OUT.resolve()}")


if __name__ == "__main__":
    main()