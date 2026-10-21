"""
02_eda.py — Exploratory analysis answering the three brief questions.

Q1  Highest-rated movie of 2020, and how to balance rating vs. number of votes
Q2  Most popular actor of 2020 (title.principals if present, else Wikipedia scrape)
Q3  Trends in user-movie preferences over the years

Outputs: charts in outputs/charts/, tables in outputs/tables/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from common import (connect, load_universe, bayesian_rating, save, has_table, OUT, EXT,
                    INK, MUTED, BLUE, ORANGE, LIGHTBLUE, GRID, PAPER)

TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
con = connect()
uni = load_universe()
C_GLOBAL = uni["rating"].mean()


# =============================================================================
# Q1. Highest-rated movie of 2020
# =============================================================================
def q1():
    m20 = con.execute("SELECT * FROM movie_mart WHERE year = 2020 AND votes > 0").df()
    C = C_GLOBAL

    raw_top = m20.sort_values(["rating", "votes"], ascending=False).head(10)
    raw_top[["title", "genres", "rating", "votes"]].to_csv(TABLES / "q1_raw_top10.csv", index=False)

    # Bayesian weighted rating at three credibility thresholds
    ranks = {}
    for m in [1_000, 5_000, 25_000, 100_000]:
        m20[f"wr_{m}"] = bayesian_rating(m20.rating, m20.votes, m, C)
        ranks[m] = m20.sort_values(f"wr_{m}", ascending=False).head(10)
    best = ranks[25_000][["title", "genres", "rating", "votes", "wr_25000"]]
    best.to_csv(TABLES / "q1_weighted_top10_m25000.csv", index=False)
    print("\nQ1 — raw top-3:", raw_top.title.head(3).tolist(),
          "| median votes of raw top-10:", int(raw_top.votes.median()))
    print("Q1 — Bayesian (m=25k) top-5:\n", best.head(5).round(2).to_string(index=False))

    # Chart 1: rating vs votes landscape with iso-WR curves
    fig, ax = plt.subplots(figsize=(13, 7.2))
    d = m20[m20.votes >= 5]
    ax.scatter(d.votes, d.rating, s=8, color=LIGHTBLUE, alpha=0.35, lw=0)
    v = np.logspace(2, 6.2, 300)
    for wr, ls in [(7.5, ":"), (8.0, "--")]:
        # iso-curve: rating needed at v votes to reach WR (m = 25k)
        r = (wr * (v + 25_000) - 25_000 * C) / v
        ok = r <= 10
        ax.plot(v[ok], r[ok], ls, color=MUTED, lw=1.4)
        ax.text(1.5e6, r[-1] + 0.12, f"WR = {wr}", color=MUTED, fontsize=12, ha="right")
    lab = ranks[25_000].head(5)
    ax.scatter(lab.votes, lab.rating, s=70, color=BLUE, zorder=3)
    offsets = [(-150, 22), (10, 18), (-140, -4), (-150, -30), (10, -22)]
    for (_, x), off in zip(lab.iterrows(), offsets):
        ax.annotate(x.title[:24], (x.votes, x.rating), xytext=off, textcoords="offset points",
                    fontsize=12, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    rt = raw_top.head(3)
    ax.scatter(rt.votes, rt.rating, s=70, color=ORANGE, zorder=3)
    ax.annotate("Raw 'top-rated' titles:\n9.5+ from a few dozen votes",
                (rt.votes.iloc[0], rt.rating.iloc[0]), xytext=(60, -95),
                textcoords="offset points", fontsize=12, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.set_xscale("log")
    ax.set_xlim(5, 2e6); ax.set_ylim(1, 10.2)
    ax.set_xlabel("Number of votes (log scale)"); ax.set_ylabel("IMDb rating")
    ax.set_title("2020 films: every dot is a movie. Dashed lines = equal weighted rating (m = 25k)")
    save(fig, "q1_rating_vs_votes_2020")

    # Chart 2: rank stability as m changes (bump chart)
    contenders = pd.concat([ranks[m].head(5) for m in ranks]).title.unique()
    order = {m: m20.sort_values(f"wr_{m}", ascending=False).title.reset_index(drop=True) for m in ranks}
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ms = list(ranks)
    for t in contenders:
        rk = [int(order[m][order[m] == t].index[0]) + 1 for m in ms]
        rk_c = [min(r, 11) for r in rk]
        win = rk[2] == 1
        col = BLUE if win else (ORANGE if rk[0] == 1 else LIGHTBLUE)
        ax.plot(range(len(ms)), rk_c, "-o", lw=3.2 if win else 1.8, color=col)
        if rk[-1] <= 10:
            ax.text(len(ms) - 0.9, rk[-1], t[:30], va="center", fontsize=12.5, color=INK if win else MUTED)
        else:
            ax.text(-0.15, rk_c[0], t[:22] + "…", va="center", ha="right", fontsize=12.5, color=col)
    ax.set_xticks(range(len(ms)), [f"m={m//1000}k" for m in ms])
    ax.set_yticks([1, 3, 5, 7, 9, 11], ["1", "3", "5", "7", "9", "10+"])
    ax.set_ylim(11.5, 0.5); ax.set_xlim(-2.4, len(ms) + 0.9)
    ax.set_ylabel("Rank among 2020 films")
    ax.set_title("Who is #1 depends on how many votes you require")
    save(fig, "q1_rank_sensitivity")
    return best


# =============================================================================
# Q2. Most popular actor of 2020
# =============================================================================
def billed_from_wikipedia() -> pd.DataFrame:
    """2020 billing order scraped from Wikipedia (+6 curated non-US titles), matched to IMDb ids."""
    cast = pd.read_csv(EXT / "cast_2020_billing.tsv", sep="\t")
    con.register("cast_raw", cast)
    nt = lambda x: f"regexp_replace(lower(strip_accents({x})), '[^a-z0-9]', '', 'g')"
    films = con.execute(f"""
        WITH cand AS (
          SELECT c.title AS wiki_title, c.cast_in_billing_order, m.tconst, m.title, m.year, m.votes, m.rating,
                 CASE WHEN {nt('m.title')} = {nt('c.title')}
                        OR {nt('m.original_title')} = {nt('c.title')} THEN 0 ELSE 1 END AS fuzzy
          FROM cast_raw c JOIN movie_mart m
            ON m.year BETWEEN 2019 AND 2021 AND m.votes > 1000
           AND ({nt('m.title')} = {nt('c.title')} OR {nt('m.original_title')} = {nt('c.title')}
                OR {nt('m.title')} LIKE {nt('c.title')} || '%'))
        SELECT * FROM cand
        QUALIFY ROW_NUMBER() OVER (PARTITION BY wiki_title
                                   ORDER BY fuzzy, ABS(year - 2020), votes DESC) = 1""").df()
    rows = []
    for _, f in films.iterrows():
        for i, actor in enumerate([a.strip() for a in f.cast_in_billing_order.split(",")][:4]):
            rows.append({"actor": actor, "billing": i + 1,
                         **f[["tconst", "title", "year", "votes", "rating"]].to_dict()})
    return pd.DataFrame(rows)


def q2():
    # --- 2a. The naive approach: IMDb knownForTitles (no title.principals supplied)
    naive = con.execute("""
        SELECT a.name, SUM(m.votes) AS votes_2020, STRING_AGG(m.title, '; ') AS films
        FROM actor_known_for a JOIN movie_mart m USING (tconst)
        WHERE m.year = 2020 AND m.votes > 0
        GROUP BY 1 ORDER BY 2 DESC LIMIT 8""").df()
    naive.to_csv(TABLES / "q2_naive_knownfor.csv", index=False)
    print("\nQ2 — naive knownFor ranking (data trap):\n", naive.head(5).to_string(index=False))

    # --- 2b. Billing order: IMDb title.principals when available, else the Wikipedia scrape
    wiki = billed_from_wikipedia()
    if has_table("cast_billing"):
        billed = con.execute("""
            SELECT c.nconst AS actor_id, p.name AS actor, c.billing,
                   m.tconst, m.title, m.year, m.votes, m.rating
            FROM cast_billing c
            JOIN movie_mart m USING (tconst)
            JOIN people p USING (nconst)
            WHERE m.year = 2020 AND m.votes > 0 AND c.billing <= 4""").df()
        source = "IMDb title.principals (all countries)"
        # Cross-check: how often does IMDb's top-4 billing agree with Wikipedia's?
        w = wiki[wiki.year == 2020]
        pairs_w = set(zip(w.tconst, w.actor.str.lower()))
        pairs_p = set(zip(billed.tconst, billed.actor.str.lower()))
        films_both = set(w.tconst) & set(billed.tconst)
        pw = {x for x in pairs_w if x[0] in films_both}
        agree = len(pw & pairs_p) / max(1, len(pw))
        pd.DataFrame([{"films_compared": len(films_both), "wiki_pairs": len(pw),
                       "share_found_in_principals_top4": round(agree, 3)}]
                     ).to_csv(TABLES / "q2_crosscheck_wikipedia.csv", index=False)
        print(f"Q2 cross-check: {agree:.0%} of Wikipedia top-4 billings found in principals "
              f"top-4 across {len(films_both)} films")
    else:
        billed = wiki[wiki.year == 2020].assign(actor_id=lambda d: d.actor)
        source = "Wikipedia 'List of American films of 2020' scrape"
    (TABLES / "q2_source.txt").write_text(source)
    print("Q2 source:", source)

    # Star Reach = Σ votes × billing weight (lead = 1, 2nd = ½, 3rd = ⅓, 4th = ¼)
    billed["star_reach"] = billed.votes / billed.billing
    billed["wr"] = bayesian_rating(billed.rating, billed.votes, 25_000, C_GLOBAL)
    g = (billed.groupby("actor_id")
         .apply(lambda d: pd.Series({
             "actor": d.actor.iloc[0], "films": len(d), "lead_roles": int((d.billing == 1).sum()),
             "star_reach": d.star_reach.sum(),
             "reach_weighted_rating": np.average(d.wr, weights=d.star_reach),
             "titles": "; ".join(d.sort_values("votes", ascending=False).title)}),
             include_groups=False)
         .sort_values("star_reach", ascending=False)
         .set_index("actor"))
    g.head(25).to_csv(TABLES / "q2_star_reach_2020.csv")
    print("\nQ2 — Star Reach 2020:\n", g.head(8).round(2).to_string())

    top = g.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12.5, 7))
    colors = [BLUE if r.lead_roles >= 2 else (LIGHTBLUE if r.lead_roles == 1 else GRID)
              for _, r in top.iterrows()]
    ax.barh(top.index, top.star_reach / 1e3, color=colors)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r.star_reach / 1e3 + 8, i, r.titles[:44], va="center", fontsize=11.5, color=MUTED)
    ax.set_xlabel("Star Reach, thousands (votes × billing weight, 2020 releases)")
    ax.set_xlim(0, top.star_reach.max() / 1e3 * 1.75)
    ax.grid(axis="y", visible=False)
    ax.set_title("Most popular actors of 2020 (dark = 2+ lead roles, light = 1 lead, grey = none)")
    save(fig, "q2_star_reach_2020")
    return g


# =============================================================================
# Q3. Trends in user-movie preferences
# =============================================================================
def q3():
    u = uni.copy()
    u["period"] = (u.year.clip(upper=2024) // 5 * 5).astype(int)  # 2025 folds into 2020-24
    g = u.assign(genre=u.genres.str.split(",")).explode("genre").dropna(subset=["genre"])
    main = ["Action", "Adventure", "Sci-Fi", "Fantasy", "Animation", "Comedy",
            "Romance", "Drama", "Crime", "Thriller", "Horror", "Biography"]

    # 3a. Genre share among AAA hits vs. all releases, by 5-year period
    hits = g[g.aaa_hit == 1]
    share_hits = (hits.groupby(["period", "genre"]).size()
                  / hits.groupby("period").tconst.nunique()).unstack().reindex(columns=main)
    share_all = (g.groupby(["period", "genre"]).size()
                 / g.groupby("period").tconst.nunique()).unstack().reindex(columns=main)
    share_hits.round(3).to_csv(TABLES / "q3_genre_share_hits.csv")
    over = (share_hits / share_all)  # over-representation index among hits
    over.round(2).to_csv(TABLES / "q3_genre_overindex_hits.csv")

    fig, ax = plt.subplots(figsize=(13, 6.8))
    show = {"Action": BLUE, "Adventure": "#5B86FF", "Sci-Fi": LIGHTBLUE,
            "Drama": ORANGE, "Romance": "#F5A66E", "Comedy": "#B9B2A6"}
    for gname, col in show.items():
        s = share_hits[gname]
        ax.plot(s.index, s.values * 100, "-o", color=col, lw=3 if gname in ("Action", "Drama") else 2)
        ax.text(s.index[-1] + 0.8, s.values[-1] * 100, gname, va="center", color=col, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlim(1969, 2031)
    ax.set_ylabel("% of AAA hits carrying the genre tag")
    ax.set_title("What audiences flock to: genre mix of each period's AAA hits")
    save(fig, "q3_genre_share_hits")

    # 3b. Attention concentration: share of a year's votes captured by its top 1% films
    def top_share(d, q=0.01):
        k = max(1, int(np.ceil(len(d) * q)))
        return d.votes.nlargest(k).sum() / d.votes.sum()
    conc = u.groupby("year").apply(top_share, include_groups=False)
    conc.to_csv(TABLES / "q3_vote_concentration.csv")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.fill_between(conc.index, conc.values * 100, color=LIGHTBLUE, alpha=0.5, lw=0)
    ax.plot(conc.index, conc.rolling(3, center=True).mean() * 100, color=BLUE, lw=3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel("Share of the year's votes")
    ax.set_title("Every year, roughly a third of all audience attention goes to 1% of films")
    save(fig, "q3_attention_concentration")

    # 3c. Genre economics: hit rate vs median ROI (finance subset)
    fin = g[g.roi_multiple.notna() & (g.budget >= 1e6) & g.genre.isin(main)]
    econ = (g[g.genre.isin(main)].groupby("genre").agg(hit_rate=("aaa_hit", "mean"), films=("tconst", "nunique"))
            .join(fin.groupby("genre").agg(median_roi=("roi_multiple", "median"),
                                           median_budget=("budget", "median"))))
    econ.round(3).to_csv(TABLES / "q3_genre_economics.csv")
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.scatter(econ.hit_rate * 100, econ.median_roi, s=econ.median_budget / 1e5 * 1.2,
               color=BLUE, alpha=0.55, lw=0)
    for gname, r in econ.iterrows():
        ax.annotate(gname, (r.hit_rate * 100, r.median_roi), ha="center", va="center", fontsize=12.5)
    ax.axhline(econ.median_roi.median(), color=MUTED, ls=":", lw=1.2)
    ax.axvline(econ.hit_rate.median() * 100, color=MUTED, ls=":", lw=1.2)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel("AAA hit rate (share of a genre's films that become hits)")
    ax.set_ylabel("Median ROI (worldwide gross ÷ budget)")
    ax.set_title("Genre economics: bubble size = median production budget")
    save(fig, "q3_genre_economics")

    # 3d. Runtime of hits vs all films
    rt = u[u.runtime.notna()].groupby(["period", "aaa_hit"]).runtime.median().unstack()
    rt.to_csv(TABLES / "q3_runtime.csv")
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(rt.index, rt[1], "-o", color=BLUE, lw=3, label="AAA hits")
    ax.plot(rt.index, rt[0], "-o", color=MUTED, lw=2, label="All other releases")
    ax.set_ylabel("Median runtime (min)"); ax.legend()
    ax.set_title("AAA hits run ~15-20 minutes longer than the market")
    save(fig, "q3_runtime")

    print("\nQ3 — genre over-index among hits, latest period:\n",
          over.iloc[-1].sort_values(ascending=False).round(2).to_string())
    print(f"Q3 — top-1pct vote share: 1970s avg {conc.loc[1970:1979].mean()*100:.1f}%, "
          f"2015-25 avg {conc.loc[2015:2025].mean()*100:.1f}%")
    print("Q3 — runtime of hits:\n", rt.round(0).to_string())
    print("Q3 — genre economics:\n", econ.round(2).to_string())


if __name__ == "__main__":
    q1(); q2(); q3()
