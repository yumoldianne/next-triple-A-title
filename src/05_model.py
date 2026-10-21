"""
05_model.py — Greenlight model: P(AAA Hit | what we know before funding).

Design choices (and why)
  * Out-of-time validation: train 1975-2012, tune 2013-2016, test 2017-2025.
    A studio uses the model on FUTURE films, so random splits would flatter it.
  * Baseline first: a logistic regression on concept-only features (genre,
    runtime, budget) shows what "industry rules of thumb" alone achieve.
  * LightGBM: handles missing budgets natively, captures interactions
    (e.g. genre x franchise), fast to retrain monthly.
  * Leakage audit: budget looks predictive but is a data-coverage artefact,
    so it is reported as a rejected variant, not used.
  * title.principals (optional) adds cast star power and producer records;
    the ablation table shows exactly what they add on top of director/writer.
  * Decision metric: precision in the top decile — a studio greenlights a
    handful of projects, so what matters is the hit rate of the ones we pick.

Outputs: outputs/charts/model_*.png, outputs/tables/model_*.csv,
         outputs/model/lgbm_aaa_hit.txt
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from common import ROOT, OUT, save, INK, MUTED, BLUE, ORANGE, LIGHTBLUE, GRID

TABLES = OUT / "tables"; MODEL_DIR = OUT / "model"
TABLES.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "data/processed/features.parquet")
G = [c for c in df.columns if c.startswith("g_")]
DIRWRI = [c for c in df.columns if c.startswith(("director_", "writer_"))] + ["n_directors", "n_writers"]
CAST = [c for c in df.columns if c.startswith(("cast_", "lead_"))]          # needs title.principals
PROD = [c for c in df.columns if c.startswith("producer_")] + (["n_producers"] if "n_producers" in df else [])
TALENT = DIRWRI + CAST + PROD
HAS_CAST = len(CAST) > 0
print("title.principals features:", "YES" if HAS_CAST else "NO (director/writer only)")
IP = ["is_franchise", "franchise_prior_films", "franchise_prior_hits", "franchise_best_reach", "sequel_marker"]
CONCEPT = G + ["n_genres", "runtime"]
MONEY = ["log_budget"]
# Budget is deliberately EXCLUDED from the production model. Budget exists only
# for films that third-party finance datasets chose to cover, and those are
# overwhelmingly famous films (hit rate 22% when budget is known vs 1.3% when
# not). "Budget is known" therefore leaks the outcome. Dropping it costs nothing:
# test ROC-AUC is equal or better without it (see model_results.csv).
FEATURES = TALENT + IP + CONCEPT + ["year"]
TARGET = "aaa_hit"

rel = df[df.in_universe].copy()
train = rel[rel.year.between(1975, 2012)]
valid = rel[rel.year.between(2013, 2016)]
test = rel[rel.year.between(2017, 2025)]
print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,} | "
      f"hit rate train {train[TARGET].mean():.3f} test {test[TARGET].mean():.3f}")


def precision_at(y, p, frac):
    k = max(1, int(len(p) * frac))
    idx = np.argsort(-p)[:k]
    return np.asarray(y)[idx].mean()


def evaluate(name, y, p):
    base = y.mean()
    r = {"model": name, "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
         "precision_top10pct": precision_at(y, p, 0.10), "precision_top2pct": precision_at(y, p, 0.02),
         "base_rate": base}
    r["lift_top10pct"] = r["precision_top10pct"] / base
    r["brier"] = brier_score_loss(y, p)
    return r


# ----------------------------------------------------------------- 1. Baseline
base_feats = CONCEPT + ["year"]
X_b = lambda d: d[base_feats]
logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
logit.fit(X_b(train), train[TARGET])
p_base = logit.predict_proba(X_b(test))[:, 1]

# ----------------------------------------------------------------- 2. LightGBM
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=42)
dtrain = lgb.Dataset(train[FEATURES], train[TARGET])
dvalid = lgb.Dataset(valid[FEATURES], valid[TARGET])
gbm = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(100, verbose=False)])
print("best iteration:", gbm.best_iteration)

# Refit on train+valid with the tuned number of rounds before touching test
trval = pd.concat([train, valid])
gbm = lgb.train(params, lgb.Dataset(trval[FEATURES], trval[TARGET]),
                num_boost_round=gbm.best_iteration)
gbm.save_model(str(MODEL_DIR / "lgbm_aaa_hit.txt"))
p_gbm = gbm.predict(test[FEATURES])

# Ablations: what does each feature family add?
def fit_eval(cols, name):
    m = lgb.train(params, lgb.Dataset(trval[cols], trval[TARGET]), num_boost_round=gbm.current_iteration())
    return evaluate(name, test[TARGET], m.predict(test[cols]))

results = [evaluate("1 Baseline: logistic on genre + runtime", test[TARGET], p_base),
           fit_eval(CONCEPT + ["year"], "2 GBM: genre + runtime"),
           fit_eval(CONCEPT + IP + ["year"], "3 GBM: + franchise / IP")]
if HAS_CAST:
    results += [fit_eval(CONCEPT + IP + DIRWRI + ["year"], "4 GBM: + director & writer record"),
                evaluate("5 GBM: + cast & producer record (FINAL)", test[TARGET], p_gbm)]
else:
    results += [evaluate("4 GBM: + director & writer record (FINAL)", test[TARGET], p_gbm)]
results += [fit_eval(FEATURES + MONEY, "x Rejected: final + budget (leaks)")]
res = pd.DataFrame(results)
res.round(4).to_csv(TABLES / "model_results.csv", index=False)
print(res.round(3).to_string(index=False))

# ----------------------------------------------------------------- 3. Value realization backtest
test = test.assign(score=p_gbm)
test["decile"] = pd.qcut(test.score.rank(method="first"), 10, labels=range(1, 11)).astype(int)
dec = test.groupby("decile").agg(films=("tconst", "size"), hit_rate=(TARGET, "mean"))
fin = test[test.roi_multiple.notna() & (test.budget >= 1e6)]
dec = dec.join(fin.groupby("decile").agg(fin_films=("tconst", "size"),
                                         median_roi=("roi_multiple", "median"),
                                         p_roi_2_5x=("roi_multiple", lambda s: (s >= 2.5).mean())))
dec.round(3).to_csv(TABLES / "model_deciles.csv")
print(dec.round(2).to_string())

# Same view on the budgeted films only (what a studio actually funds)
fin_all = pd.concat([valid.assign(score=gbm.predict(valid[FEATURES])), test])
fin_all = fin_all[fin_all.roi_multiple.notna() & (fin_all.budget >= 1e6) & (fin_all.year <= 2019)]
fin_all["top_q"] = fin_all.score >= fin_all.score.quantile(0.75)
roi_view = fin_all.groupby("top_q").agg(films=("tconst", "size"), median_roi=("roi_multiple", "median"),
                                        hit_rate=(TARGET, "mean"),
                                        p_roi_2_5x=("roi_multiple", lambda s: (s >= 2.5).mean()))
roi_view.round(3).to_csv(TABLES / "model_roi_top_quartile.csv")
print("Budgeted films 2013-2019 (note: talent, not budget, sorts them)\n", roi_view.round(2))

fig, ax = plt.subplots(figsize=(13, 6.2))
colors = [LIGHTBLUE] * 9 + [BLUE]
ax.bar(dec.index, dec.hit_rate * 100, color=colors)
ax.axhline(test[TARGET].mean() * 100, color=ORANGE, ls="--", lw=2)
ax.text(1, test[TARGET].mean() * 100 + 0.8, f"Market base rate {test[TARGET].mean():.1%}",
        color=ORANGE, fontsize=13)
for d, v in dec.hit_rate.items():
    ax.text(d, v * 100 + 0.5, f"{v:.0%}", ha="center", fontsize=12.5, color=INK)
ax.set_xticks(range(1, 11), [f"D{i}" for i in range(1, 11)])
ax.set_xlabel("Model score decile on unseen 2017–2025 films (D10 = highest)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.set_ylabel("Share that became AAA hits"); ax.grid(axis="x", visible=False)
ax.set_title("Out-of-time backtest: hit rate by model score decile")
save(fig, "model_decile_backtest")

# ----------------------------------------------------------------- 4. Explainability
sample = trval.sample(min(8000, len(trval)), random_state=1)
sv = shap.TreeExplainer(gbm).shap_values(sample[FEATURES])
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
imp.round(4).to_csv(TABLES / "model_shap_importance.csv")
family = {**{c: "Director & writer record" for c in DIRWRI}, **{c: "Cast star power" for c in CAST},
          **{c: "Producer record" for c in PROD}, **{c: "Franchise / IP" for c in IP},
          **{c: "Genre mix" for c in G + ["n_genres"]}, "runtime": "Runtime",
          "year": "Release year"}
fam = imp.groupby(imp.index.map(family)).sum().sort_values()
fam.round(4).to_csv(TABLES / "model_shap_by_family.csv")

LABELS = {"director_best_reach": "Director's best prior reach", "director_mean_reach": "Director's avg prior reach",
          "writer_best_reach": "Writer's best prior reach", "writer_mean_reach": "Writer's avg prior reach",
          "director_hit_rate": "Director's prior hit rate", "writer_hit_rate": "Writer's prior hit rate",
          "director_prior_hits": "Director's prior AAA hits", "writer_prior_hits": "Writer's prior AAA hits",
          "director_prior_films": "Director's prior films", "writer_prior_films": "Writer's prior films",
          "director_experience_yrs": "Director's years active", "writer_experience_yrs": "Writer's years active",
          "franchise_best_reach": "Franchise's best prior reach", "franchise_prior_hits": "Franchise prior hits",
          "is_franchise": "Is a franchise film", "log_budget": "Budget (2020 $)", "runtime": "Runtime",
          "year": "Release year", "n_genres": "Number of genres", "n_writers": "Number of writers"}
for pre, nice in [("cast", "Top-3 cast's"), ("lead", "Lead actor's"), ("producer", "Producer's")]:
    LABELS.update({f"{pre}_best_reach": f"{nice} best prior reach", f"{pre}_mean_reach": f"{nice} avg prior reach",
                   f"{pre}_hit_rate": f"{nice} prior hit rate", f"{pre}_prior_hits": f"{nice} prior AAA hits",
                   f"{pre}_prior_films": f"{nice} prior films", f"{pre}_experience_yrs": f"{nice} years active"})
lab = lambda c: LABELS.get(c, c.replace("g_", "Genre: ").title() if c.startswith("g_") else c)

fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.75})
ax = axes[0]
ax.barh(fam.index, fam.values / fam.sum() * 100, color=[LIGHTBLUE] * (len(fam) - 1) + [BLUE])
for i, v in enumerate(fam.values / fam.sum() * 100):
    ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=13)
ax.set_xlabel("Share of model's explanatory weight (mean |SHAP|)")
ax.grid(axis="y", visible=False); ax.set_title("What drives a hit, by lever")
ax = axes[1]
top = imp.head(12)[::-1]
ax.barh([lab(c) for c in top.index], top.values, color=BLUE)
ax.grid(axis="y", visible=False); ax.set_xlabel("Mean |SHAP| (log-odds)")
ax.set_title("Top 12 individual signals")
save(fig, "model_drivers")
print("Driver families:\n", (fam / fam.sum()).round(3).to_string())

# ----------------------------------------------------------------- 5. Greenlight recipe simulator
# Hold talent at "proven hit-maker" levels (75th pct among directors with a prior hit),
# vary concept. Associational, not causal: it tells us where proven talent tends to win.
proven = trval[trval.director_prior_hits >= 1]
template = trval[FEATURES].median(numeric_only=True)
for c in TALENT:
    template[c] = proven[c].quantile(0.75)
template[G] = 0
template["year"] = 2026
template["is_franchise"] = 0; template["franchise_prior_films"] = 0
template["franchise_prior_hits"] = 0; template["franchise_best_reach"] = np.nan
template["sequel_marker"] = 0
combos = [("Action", "Adventure", "Sci-Fi"), ("Action", "Adventure", "Fantasy"),
          ("Adventure", "Animation", "Comedy"), ("Animation", "Adventure", "Family"),
          ("Action", "Crime", "Thriller"), ("Horror", "Mystery", "Thriller"),
          ("Biography", "Drama", "History"), ("Comedy", "Romance"), ("Drama",),
          ("Action", "Comedy"), ("Sci-Fi", "Thriller"), ("Crime", "Drama", "Mystery")]
rows = []
for combo in combos:
    for rt in [100, 125, 150]:
        for fr in [0, 1]:
            x = template.copy()
            for gname in combo:
                x[f"g_{gname.replace('-', '').lower()}"] = 1
            x["n_genres"] = len(combo); x["runtime"] = rt
            if fr:
                x["is_franchise"] = 1; x["franchise_prior_films"] = 2
                x["franchise_prior_hits"] = 1; x["franchise_best_reach"] = 0.99
                x["sequel_marker"] = 1
            rows.append({"concept": " / ".join(combo), "runtime": rt, "franchise": fr,
                         "p_hit": float(gbm.predict(pd.DataFrame([x])[FEATURES])[0])})
recipe = pd.DataFrame(rows)
recipe.round(4).to_csv(TABLES / "model_recipe_simulator.csv", index=False)
piv = recipe[recipe.runtime == 125].pivot(index="concept", columns="franchise", values="p_hit")
piv.columns = ["Original IP", "Franchise continuation"]
piv = piv.sort_values("Franchise continuation")
fig, ax = plt.subplots(figsize=(13, 7))
y = np.arange(len(piv))
ax.barh(y + 0.2, piv["Franchise continuation"] * 100, height=0.4, color=BLUE, label="Franchise continuation")
ax.barh(y - 0.2, piv["Original IP"] * 100, height=0.4, color=LIGHTBLUE, label="Original IP")
ax.set_yticks(y, piv.index); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.set_xlabel("Predicted P(AAA hit) — proven director & writer, 125 min")
ax.legend(loc="lower right"); ax.set_title("Greenlight simulator: same talent, different concept")
save(fig, "model_recipe_simulator")
print(piv.round(3).to_string())

# ----------------------------------------------------------------- 6. Score the upcoming slate
up = df[df.upcoming & (df.director_prior_films.fillna(0) > 0)].copy()
up["p_hit"] = gbm.predict(up[FEATURES])
names = None
try:
    import duckdb
    con = duckdb.connect(str(ROOT / "data/processed/imdb.duckdb"), read_only=True)
    names = con.execute("""
        SELECT c.tconst, STRING_AGG(p.name, ', ') AS directors
        FROM crew c JOIN people p USING (nconst) WHERE c.role = 'director'
        GROUP BY 1""").df()
    if HAS_CAST:
        stars = con.execute("""
            SELECT c.tconst, STRING_AGG(p.name, ', ' ORDER BY c.billing) AS top_cast
            FROM cast_billing c JOIN people p USING (nconst) WHERE c.billing <= 3
            GROUP BY 1""").df()
        names = names.merge(stars, on="tconst", how="outer")
except Exception as e:  # pragma: no cover
    print("name lookup skipped:", e)
if names is not None:
    up = up.merge(names, on="tconst", how="left")
cols = ["tconst", "title", "year", "directors", "top_cast", "p_hit", "is_franchise", "director_prior_hits",
        "director_best_reach", "cast_best_reach", "runtime"]
slate = up.sort_values("p_hit", ascending=False)[[c for c in cols if c in up.columns]]
slate.head(50).round(3).to_csv(TABLES / "model_upcoming_slate_top50.csv", index=False)
print("Upcoming slate scored:", len(up))
print(slate.head(15).round(3).to_string(index=False))

json.dump({"features": FEATURES, "train_years": "1975-2016", "test_years": "2017-2025",
           "best_iteration": int(gbm.current_iteration()), "results": results},
          open(MODEL_DIR / "model_card.json", "w"), indent=2, default=float)
