# Give Me The Next AAA Title

End-to-end project: data engineering (SQL on DuckDB) → exploratory analysis →
a business-grounded "hit" metric → a greenlight model → recommendations.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py        # Windows, macOS, Linux.  (./run_all.sh also works on macOS/Linux)
```

Before running, put the IMDb files in `data/raw/`. Names are detected automatically; any of
`title.basics.tsv.gz`, `title_basics.tsv.gz` or `title_basics_tsv.gz` work.

Runtime: ~10–20 minutes on a laptop with 8 GB RAM (title.principals is the largest file; DuckDB
streams it and spills to `data/processed/duckdb_tmp` if memory runs short).

## What to send back after running

1. `outputs/RESULTS_FOR_SLIDES.md` — every number the deck uses, in one file.
2. The `outputs/charts/` folder (11 PNGs), zipped.
3. If anything fails: `outputs/run_log.txt`.

## Repository map

| Path | What it does |
|---|---|
| `sql/01_silver_imdb.sql` | Streams the raw TSV.gz dumps into typed tables (films only, non-adult) |
| `sql/02_silver_principals.sql` | Cast billing (actors re-ranked by credit order) and producers |
| `sql/03_silver_finance.sql` | Joins budget / worldwide gross from TMDB + The Numbers (fuzzy title-year key) |
| `sql/04_gold_movie_mart.sql` | One row per film; defines the analysis universe |
| `src/00_scrape_cast_2020.py` | Wikipedia 2020 billing scrape (fallback + cross-check for Q2) |
| `src/01_build_warehouse.py` | Finds the raw files, runs the SQL layers, prints row counts + coverage |
| `src/02_eda.py` | Q1 highest-rated 2020 film, Q2 most popular 2020 actor, Q3 trends |
| `src/03_hit_metric.py` | Defines and validates the AAA Hit metric against box office |
| `src/04_features.py` | Pre-greenlight features; every track record uses prior years only |
| `src/05_model.py` | Baseline → LightGBM, out-of-time backtest, SHAP, simulator, slate scoring |
| `src/06_report.py` | Writes `outputs/RESULTS_FOR_SLIDES.md` |
| `run_all.py` | Runs 01–06 in order and logs to `outputs/run_log.txt` |

## Data sources

* IMDb non-commercial datasets.
* TMDB 10k movies (budget, revenue, imdb_id; 1960–2015) — GitHub mirror of the Kaggle/Udacity set.
* The Numbers production budgets & worldwide gross (5.8k films to ~2019) — GitHub mirror.
* Wikipedia "List of American films of 2020" for billing order, plus 6 non-US titles curated by hand.

## Key definitions

* **Analysis universe** — feature films 1970–2025, ≥ 500 IMDb votes, 60–240 min.
* **Weighted rating** — `WR = v/(v+m)·R + m/(v+m)·C` (IMDb Top-250 formula; C = universe mean).
* **Star Reach** — Σ over an actor's 2020 films of votes ÷ billing position (top-4 billed).
* **AAA Hit** — top 5% of its release year by votes **and** rating ≥ 6.5.

## Known limitations

* Finance data stops ~2019 and covers ~9% of films; it validates the metric and is never a
  model feature (its coverage itself leaks success — see `05_model.py`).
* IMDb votes are a proxy for audience size, skewed toward online, English-speaking users.
* The model is associational. It ranks projects; it does not prove that changing a genre
  causes a hit.
