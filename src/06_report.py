"""
06_report.py — Collect every number the deck uses into ONE file to send back:

    outputs/RESULTS_FOR_SLIDES.md

Run after 02-05. Paste or upload that file (plus outputs/charts/*.png) to update the deck.
"""
import json
import pandas as pd
from common import OUT, load_universe

T = OUT / "tables"
lines = ["# RESULTS FOR SLIDES", ""]


def table(name, title, n=None, index=False):
    path = T / name
    if not path.exists():
        lines.extend([f"## {title}", f"(missing: {name})", ""])
        return None
    df = pd.read_csv(path, index_col=0 if index else None)
    if n:
        df = df.head(n)
    lines.extend([f"## {title}", "", df.round(3).to_markdown(index=index), ""])
    return df


u = load_universe()
lines += ["## Universe", "",
          f"- films in analysis universe: {len(u):,}",
          f"- AAA hit base rate (all years): {u.aaa_hit.mean():.4f}",
          f"- films with budget+gross (budget >= $1M): {int((u.roi_multiple.notna() & (u.budget >= 1e6)).sum()):,}", ""]

table("q1_raw_top10.csv", "Q1 raw top-10 (2020)")
table("q1_weighted_top10_m25000.csv", "Q1 weighted top-10, m = 25k (2020)")
src = (T / "q2_source.txt").read_text() if (T / "q2_source.txt").exists() else "unknown"
lines += [f"## Q2 source: {src}", ""]
table("q2_naive_knownfor.csv", "Q2 naive knownFor ranking")
table("q2_star_reach_2020.csv", "Q2 Star Reach 2020 (top 15)", n=15, index=True)
table("q2_crosscheck_wikipedia.csv", "Q2 cross-check vs Wikipedia billing")
table("q3_genre_overindex_hits.csv", "Q3 genre over-index among hits (by period)", index=True)
table("q3_genre_share_hits.csv", "Q3 genre share of hits (by period)", index=True)
conc = pd.read_csv(T / "q3_vote_concentration.csv", index_col=0).iloc[:, 0]
lines += ["## Q3 vote concentration (top-1% share)", "",
          f"- 1970s avg: {conc.loc[1970:1979].mean():.3f}", f"- 2015-25 avg: {conc.loc[2015:2025].mean():.3f}", ""]
table("q3_runtime.csv", "Q3 median runtime (0 = other, 1 = AAA hit)", index=True)
table("q3_genre_economics.csv", "Q3 genre economics", index=True)
table("hit_signal_correlations.csv", "Hit metric: Spearman correlations", index=True)
table("hit_definition_comparison.csv", "Hit metric: definitions compared")
table("hit_quality_gate.csv", "Hit metric: ROI by rating band (top-10% reach)", index=True)
table("model_results.csv", "Model results (test 2017-2025)")
table("model_deciles.csv", "Model backtest by decile", index=True)
table("model_roi_top_quartile.csv", "Budgeted films 2013-19: top quartile vs rest", index=True)
fam = table("model_shap_by_family.csv", "SHAP by family (raw)", index=True)
if fam is not None:
    share = (fam.iloc[:, 0] / fam.iloc[:, 0].sum()).sort_values(ascending=False)
    lines += ["SHAP share: " + ", ".join(f"{k} {v:.1%}" for k, v in share.items()), ""]
table("model_shap_importance.csv", "Top 15 individual signals", n=15, index=True)
rec = T / "model_recipe_simulator.csv"
if rec.exists():
    r = pd.read_csv(rec)
    piv = r[r.runtime == 125].pivot(index="concept", columns="franchise", values="p_hit")
    lines += ["## Recipe simulator (125 min)", "", piv.round(3).to_markdown(), ""]
table("model_upcoming_slate_top50.csv", "Upcoming slate (top 15)", n=15)
card = OUT / "model/model_card.json"
if card.exists():
    lines += ["## Model card", "", "```json", json.dumps(json.load(open(card)), indent=1, default=str)[:3000], "```"]

(OUT / "RESULTS_FOR_SLIDES.md").write_text("\n".join(lines))
print("wrote", OUT / "RESULTS_FOR_SLIDES.md")
