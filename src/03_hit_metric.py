"""
03_hit_metric.py — Define and validate the 'AAA Hit' metric.

Definition (see common.add_hit_metric):
    AAA Hit = top 5% of its release year by audience reach (IMDb votes)
              AND IMDb rating >= 6.5

Why: a studio funds films to earn money and build franchises. Box office is
only available for ~9% of films, but votes exist for all of them. We show that
(1) reach is the audience signal most tied to revenue, (2) quality still
matters for *return* once reach is high, and (3) the combined metric isolates
films that pay back far better than the market.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import load_universe, save, OUT, INK, MUTED, BLUE, ORANGE, LIGHTBLUE, GRID

TABLES = OUT / "tables"; TABLES.mkdir(parents=True, exist_ok=True)
u = load_universe()
fin = u[u.roi_multiple.notna() & (u.budget >= 1e6)].copy()

# 1. Which audience signal tracks money?
corr = fin[["votes", "reach_pct", "rating", "budget", "gross_ww", "roi_multiple"]].corr("spearman")
corr.round(3).to_csv(TABLES / "hit_signal_correlations.csv")
print("Spearman vs gross:", corr["gross_ww"].round(2).to_dict())
print("Spearman vs ROI  :", corr["roi_multiple"].round(2).to_dict())

# 2. Candidate definitions compared on the money they make
defs = {
    "All films": np.ones(len(fin), bool),
    "Rating ≥ 7.5 only": (fin.rating >= 7.5).values,
    "Top-5% reach only": (fin.reach_pct >= 0.95).values,
    "AAA Hit (reach + quality gate)": (fin.aaa_hit == 1).values,
}
rows = []
for k, m in defs.items():
    s = fin[m]
    rows.append({"definition": k, "films": int(m.sum()),
                 "p_roi_2_5x": (s.roi_multiple >= 2.5).mean(),
                 "median_roi": s.roi_multiple.median(),
                 "median_gross_musd": s.gross_ww.median() / 1e6})
comp = pd.DataFrame(rows)
comp.round(3).to_csv(TABLES / "hit_definition_comparison.csv", index=False)
print(comp.round(2).to_string(index=False))

# 3. Quality gate: among high-reach films, does rating still move ROI?
top = fin[fin.reach_pct >= 0.90]
bands = pd.cut(top.rating, [0, 5.5, 6.5, 7.5, 10], labels=["≤5.5", "5.6–6.5", "6.6–7.5", ">7.5"])
gate = top.groupby(bands, observed=True).roi_multiple.agg(["count", "median"])
gate.to_csv(TABLES / "hit_quality_gate.csv")
print(gate.round(2))

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
cols = [GRID, LIGHTBLUE, LIGHTBLUE, BLUE]
ax.barh(comp.definition[::-1], comp.median_roi[::-1], color=cols[::-1])
for i, r in enumerate(comp[::-1].itertuples()):
    ax.text(r.median_roi + 0.05, i, f"{r.median_roi:.1f}x   ({r.p_roi_2_5x:.0%} clear 2.5x)",
            va="center", fontsize=13, color=INK)
ax.set_xlim(0, comp.median_roi.max() * 1.6); ax.grid(axis="y", visible=False)
ax.set_xlabel("Median ROI (worldwide gross ÷ budget)")
ax.set_title("Films that meet each definition")
ax = axes[1]
ax.bar(gate.index.astype(str), gate["median"], color=[LIGHTBLUE, LIGHTBLUE, BLUE, BLUE])
for i, v in enumerate(gate["median"]):
    ax.text(i, v + 0.08, f"{v:.1f}x", ha="center", fontsize=13)
ax.set_xlabel("IMDb rating (films already in top-10% reach)")
ax.set_ylabel("Median ROI"); ax.grid(axis="x", visible=False)
ax.set_title("At equal reach, quality ~doubles ROI")
save(fig, "hit_metric_validation")
print("Base rate of AAA Hit in universe:", round(u.aaa_hit.mean(), 4))
