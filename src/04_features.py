"""
04_features.py — Build a *pre-greenlight* feature table.

Rule: every feature must be knowable when a studio decides to fund a film.
So no votes, ratings, or release footprint of the film itself. Talent track
records use ONLY films released in earlier years (strictly year < film year).

Feature families
  Talent   director / writer prior films, prior AAA hits, hit rate, best reach
           + (with title.principals) top-3 cast, lead actor and producer records
  IP       franchise continuation (an earlier film shares its title stem),
           prior hits in that franchise, explicit sequel markers
  Concept  genre mix (multi-hot), number of genres, planned runtime
  Money    production budget (≈2020 USD; missing for most films → NaN, which
           the model handles natively)
  Context  release year

Output: data/processed/features.parquet  (released films + upcoming slate)
"""
import re
import numpy as np
import pandas as pd
from common import connect, add_hit_metric, ROOT

con = connect()

# --------------------------------------------------------------------------- base
mart = con.execute("""
    SELECT tconst, title, year, runtime, genres, votes, rating, budget, gross_ww, roi_multiple
    FROM movie_mart WHERE year BETWEEN 1960 AND 2028""").df()
uni = add_hit_metric(con.execute("SELECT tconst, year, votes, rating FROM universe").df())
mart = mart.merge(uni[["tconst", "reach_pct", "aaa_hit"]], on="tconst", how="left")
mart["in_universe"] = mart.aaa_hit.notna()

# Upcoming slate: announced films (2026-2028) not yet rated by audiences
mart["upcoming"] = (mart.year >= 2026) & mart.votes.isna()

# --------------------------------------------------------------------------- talent
# One generic engine for every talent role. For each person we accumulate their
# record on films released in STRICTLY EARLIER years, then aggregate per film.
FILM_COLS = ["tconst", "year", "in_universe", "aaa_hit", "reach_pct"]


def track_record(links: pd.DataFrame, prefix: str, with_count: bool = True) -> pd.DataFrame:
    """links: one row per (tconst, nconst) credit. Returns per-film features named
    <prefix>_prior_films, _prior_hits, _hit_rate, _mean_reach, _best_reach, _experience_yrs."""
    c = links[["tconst", "nconst"]].drop_duplicates().merge(mart[FILM_COLS], on="tconst")
    c["tracked"] = c.in_universe.astype(int)
    c["hit"] = c.aaa_hit.fillna(0)
    c["reach"] = c.reach_pct.fillna(0)
    py = (c.groupby(["nconst", "year"])
            .agg(n=("tconst", "size"), tracked=("tracked", "sum"), hits=("hit", "sum"),
                 reach_sum=("reach", "sum"), reach_max=("reach", "max"))
            .reset_index().sort_values(["nconst", "year"]))
    g = py.groupby("nconst")
    for col in ["n", "tracked", "hits", "reach_sum"]:          # totals BEFORE this year
        py[f"cum_{col}"] = g[col].cumsum() - py[col]
    py["cum_reach_max"] = g["reach_max"].cummax()
    py["cum_reach_max"] = py.groupby("nconst")["cum_reach_max"].shift(1).fillna(0)
    py["first_year"] = g["year"].transform("min")
    film = c[["tconst", "nconst", "year"]].merge(
        py[["nconst", "year", "cum_n", "cum_tracked", "cum_hits", "cum_reach_sum",
            "cum_reach_max", "first_year"]], on=["nconst", "year"])
    film["experience_yrs"] = film.year - film.first_year
    film["hit_rate"] = film.cum_hits / film.cum_tracked.replace(0, np.nan)
    film["mean_reach"] = film.cum_reach_sum / film.cum_tracked.replace(0, np.nan)
    aggs = {f"{prefix}_prior_films": ("cum_n", "max"),
            f"{prefix}_prior_hits": ("cum_hits", "max"),
            f"{prefix}_hit_rate": ("hit_rate", "max"),
            f"{prefix}_mean_reach": ("mean_reach", "max"),
            f"{prefix}_best_reach": ("cum_reach_max", "max"),
            f"{prefix}_experience_yrs": ("experience_yrs", "max")}
    if with_count:
        aggs[f"n_{prefix}s"] = ("nconst", "nunique")
    return film.groupby("tconst").agg(**aggs)


crew = con.execute("SELECT tconst, role, nconst FROM crew").df()
talent = (track_record(crew[crew.role == "director"], "director")
          .join(track_record(crew[crew.role == "writer"], "writer"), how="outer"))

# Cast and producers come from title.principals (optional input).
HAS_PRINCIPALS = con.execute(
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'cast_billing'").fetchone()[0] > 0
if HAS_PRINCIPALS:
    cb = con.execute("SELECT tconst, nconst, billing FROM cast_billing WHERE billing <= 3").df()
    # Star power of the top-3 billed cast: history counts only top-3 billed roles,
    # so an uncredited extra in a blockbuster earns no 'star' credit.
    talent = talent.join(track_record(cb, "cast", with_count=False), how="outer")
    lead = track_record(cb[cb.billing == 1], "lead", with_count=False)
    talent = talent.join(lead, how="outer")
    prod = con.execute("SELECT tconst, nconst FROM producers").df()
    talent = talent.join(track_record(prod, "producer"), how="outer")
    print("principals found: cast (top-3), lead and producer track records added")
else:
    print("no title.principals: model uses director + writer track records only")

# --------------------------------------------------------------------------- franchise / IP
ROMAN = r"(ii|iii|iv|v|vi|vii|viii|ix|x)"


def stem(title: str) -> str:
    t = title.lower()
    t = re.split(r"[:\-–—]", t)[0]                         # "Dune: Part Two" -> "dune"
    t = re.sub(rf"\b(part|chapter|vol\.?|volume)?\s*(\d+|{ROMAN})\s*$", "", t)  # trailing sequel no.
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return re.sub(r"[^a-z0-9]", "", t)


mart["stem"] = mart.title.fillna("").map(stem)
mart["sequel_marker"] = mart.title.fillna("").str.contains(
    rf"(?:\s(?:\d|{ROMAN.upper()})$)|(?:Part\s|Chapter\s)|:\s", regex=True).astype(int)
tracked = mart[mart.in_universe | mart.upcoming][["tconst", "stem", "year", "aaa_hit"]]
fr = (mart[mart.stem.str.len() >= 4][["tconst", "stem", "year"]]
      .merge(mart[mart.in_universe][["stem", "year", "aaa_hit", "reach_pct"]]
             .rename(columns={"year": "y0"}), on="stem"))
fr = fr[fr.y0 < fr.year]
fr_agg = fr.groupby("tconst").agg(franchise_prior_films=("y0", "size"),
                                  franchise_prior_hits=("aaa_hit", "sum"),
                                  franchise_best_reach=("reach_pct", "max"))

# --------------------------------------------------------------------------- concept
GENRES = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary",
          "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"]
gdum = mart.genres.fillna("").str.get_dummies(sep=",").reindex(columns=GENRES, fill_value=0)
gdum.columns = [f"g_{c.replace('-', '').lower()}" for c in gdum.columns]

feat = pd.concat([mart, gdum], axis=1)
feat["n_genres"] = gdum.sum(axis=1)
feat = (feat.merge(talent, left_on="tconst", right_index=True, how="left")
            .merge(fr_agg, left_on="tconst", right_index=True, how="left"))
for c in ["franchise_prior_films", "franchise_prior_hits"]:
    feat[c] = feat[c].fillna(0)
feat["is_franchise"] = (feat.franchise_prior_films > 0).astype(int)

# --------------------------------------------------------------------------- money
# Rough deflator (~3%/yr) to put budgets on a 2020-dollar footing.
feat["budget_2020"] = feat.budget * 1.03 ** (2020 - feat.year)
feat["log_budget"] = np.log10(feat.budget_2020)

keep = feat[(feat.in_universe & (feat.year >= 1975)) | feat.upcoming]
keep = keep.drop(columns=["genres"]).reset_index(drop=True)
keep.to_parquet(ROOT / "data/processed/features.parquet", index=False)
print(f"features: {len(keep):,} rows | released {keep.in_universe.sum():,} | "
      f"upcoming {keep.upcoming.sum():,} | hit rate {keep.aaa_hit.mean():.3f}")
print("franchise share:", round(keep.is_franchise.mean(), 3),
      "| budget known:", round(keep.log_budget.notna().mean(), 3))
