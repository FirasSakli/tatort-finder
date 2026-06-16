"""Tatort discovery tool — find your next episode by similarity, team, and tone."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

DB = Path("data/processed/tatort.duckdb")
EMB = Path("data/processed/plot_embeddings.parquet")

st.set_page_config(page_title="Tatort Finder", page_icon="🔍", layout="wide")


@st.cache_data
def load_data():
    con = duckdb.connect(str(DB), read_only=True)
    meta = con.execute(
        """
        SELECT e.Folge, e.Titel, e.year, e.Sender, e.averageRating, e.numVotes,
               t.team_grouped AS team,
               d.death_score, p.plot, p.plot_len
        FROM episodes e
        LEFT JOIN team_features t USING (Folge)
        LEFT JOIN death_features d USING (Folge)
        LEFT JOIN plots p USING (Folge)
        """
    ).fetchdf()
    con.close()
    emb = pd.read_parquet(EMB)
    meta = meta[meta["Folge"].isin(emb["Folge"])].reset_index(drop=True)
    emb = emb.set_index("Folge").loc[meta["Folge"]].reset_index()
    # Tone = length-normalized violence, bucketed for a human-friendly label.
    meta["tone"] = meta["death_score"] / meta["plot_len"] * 1000
    meta["tone_label"] = pd.cut(
        meta["tone"], bins=[-1, 0.6, 1.0, 99],
        labels=["🟢 Lighter", "🟡 Moderate", "🔴 Dark"],
    )
    return meta, emb


@st.cache_data
def similarity_matrix(_emb):
    return cosine_similarity(_emb.drop(columns="Folge").values)


def episode_card(row, sim=None):
    rating = f"⭐ {row['averageRating']}" if pd.notna(row["averageRating"]) else "—"
    sim_txt = f"  ·  {sim:.0%} match" if sim is not None else ""
    st.markdown(
        f"**{row['Titel']}** ({int(row['year'])}){sim_txt}  \n"
        f"{row['team']}  ·  {rating}  ·  {row['tone_label']}"
    )


def main():
    meta, emb = load_data()
    sim = similarity_matrix(emb)

    st.title("🔍 Tatort Finder")
    st.caption(
        f"Find your next episode among {len(meta):,} Tatort films (1970–today). "
        "Recommendations are based on plot similarity."
    )

    tab1, tab2 = st.tabs(["Find similar episodes", "Explore the teams"])

    # --- Discovery tab ---
    with tab1:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            choice = st.selectbox(
                "Pick an episode you enjoyed:",
                options=meta["Titel"] + " (" + meta["year"].astype(int).astype(str) + ")",
                index=None,
                placeholder="Start typing a title...",
            )
        with col_b:
            tone_filter = st.multiselect(
                "Tone (optional):",
                options=["🟢 Lighter", "🟡 Moderate", "🔴 Dark"],
            )

        if choice:
            idx = meta.index[
                (meta["Titel"] + " (" + meta["year"].astype(int).astype(str) + ")") == choice
            ][0]
            picked = meta.loc[idx]
            st.divider()
            st.subheader("Because you liked:")
            episode_card(picked)
            with st.expander("Plot"):
                st.write(picked["plot"][:1000] + "...")

            st.subheader("You might also like:")
            order = np.argsort(sim[idx])[::-1]
            shown = 0
            for j in order:
                if j == idx:
                    continue
                rec = meta.loc[j]
                if tone_filter and rec["tone_label"] not in tone_filter:
                    continue
                episode_card(rec, sim=sim[idx][j])
                shown += 1
                if shown >= 6:
                    break

    # --- Teams tab ---
    with tab2:
        team = st.selectbox(
            "Choose a team:",
            sorted(meta[meta["team"] != "Other"]["team"].dropna().unique()),
        )
        sub = meta[meta["team"] == team]
        c1, c2, c3 = st.columns(3)
        c1.metric("Episodes", len(sub))
        c2.metric("Avg. rating", f"{sub['averageRating'].mean():.2f}")
        c3.metric("Typical tone", sub["tone_label"].mode()[0] if len(sub) else "—")

        st.subheader("Highest-rated episodes")
        top = sub.nlargest(5, "averageRating")[["Titel", "year", "averageRating", "tone_label"]]
        top.columns = ["Title", "Year", "Rating", "Tone"]
        st.dataframe(top, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()