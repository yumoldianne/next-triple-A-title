"""Shared paths, plotting style and metric definitions used by every step."""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/processed/imdb.duckdb"
EXT = ROOT / "data/external"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# ---- Palette (matches the deck) --------------------------------------------
INK, PAPER, MUTED, GRID = "#14181F", "#F6F4EF", "#6B7280", "#DEDAD1"
BLUE, ORANGE, LIGHTBLUE = "#1F5EFF", "#F07A2B", "#9DB7FF"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 15, "axes.titlesize": 18,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.family": "DejaVu Sans", "legend.frameon": False,
})


# ---- Raw file discovery ----------------------------------------------------
RAW = ROOT / "data/raw"
RAW_TABLES = ["title_basics", "title_ratings", "title_crew", "name_basics",
              "title_akas", "title_principals"]
OPTIONAL = {"title_principals"}


def find_raw_files(required: bool = True) -> dict:
    """Map 'title_basics' -> Path, accepting title.basics.tsv.gz, title_basics.tsv.gz,
    title_basics_tsv.gz, or an unzipped .tsv, in data/raw/."""
    found = {}
    for key in RAW_TABLES:
        dotted = key.replace("_", ".", 1)
        cands = [f"{dotted}.tsv.gz", f"{key}.tsv.gz", f"{key}_tsv.gz", f"{dotted}.tsv", f"{key}.tsv"]
        hit = next((RAW / c for c in cands if (RAW / c).exists()), None)
        if hit:
            found[key] = hit
        elif required and key not in OPTIONAL:
            raise FileNotFoundError(f"Missing {key}: put e.g. {dotted}.tsv.gz in {RAW}")
    return found


def has_table(name: str) -> bool:
    con = duckdb.connect(str(DB), read_only=True)
    ok = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                     [name]).fetchone()[0] > 0
    con.close()
    return ok


def save(fig, name: str) -> Path:
    path = CHARTS / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def connect():
    return duckdb.connect(str(DB), read_only=True)


# ---- Metric definitions -----------------------------------------------------
def bayesian_rating(rating: pd.Series, votes: pd.Series, m: float, C: float) -> pd.Series:
    """IMDb-style weighted rating:  WR = v/(v+m)·R + m/(v+m)·C.
    m = votes needed before a film's own rating is trusted ~50%; C = prior mean."""
    return votes / (votes + m) * rating + m / (votes + m) * C


HIT_REACH_PCT = 0.95   # top 5% of its release year by audience reach (votes)
HIT_MIN_RATING = 6.5   # ...and audiences did not hate it


def add_hit_metric(df: pd.DataFrame) -> pd.DataFrame:
    """AAA Hit = top-5% audience reach within its release year AND rating >= 6.5.

    Reach is percentile-ranked *within year* so a 1985 film and a 2024 film are
    judged against their own cohort (IMDb's user base and vote accumulation
    differ massively by era). The rating gate removes 'hate-watched' films:
    at equal reach, well-rated films earn ~2x the ROI (see 03_hit_metric.py).
    """
    df = df.copy()
    df["reach_pct"] = df.groupby("year")["votes"].rank(pct=True)
    df["aaa_hit"] = ((df["reach_pct"] >= HIT_REACH_PCT) &
                     (df["rating"] >= HIT_MIN_RATING)).astype(int)
    return df


def load_universe() -> pd.DataFrame:
    con = connect()
    df = con.execute("SELECT * FROM universe").df()
    con.close()
    return add_hit_metric(df)
