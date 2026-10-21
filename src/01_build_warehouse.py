"""
01_build_warehouse.py
---------------------
Builds a local DuckDB warehouse (data/processed/imdb.duckdb) from the raw IMDb
dumps plus two external finance datasets, by running the SQL files in /sql:

    01_silver_imdb.sql        raw TSV -> typed film tables
    02_silver_principals.sql  cast billing + producers   (skipped if no principals file)
    03_silver_finance.sql     budgets / worldwide gross joined to IMDb keys
    04_gold_movie_mart.sql    one-row-per-film analytical mart

File names are auto-detected in data/raw/. Any of these work:
    title.basics.tsv.gz | title_basics.tsv.gz | title_basics_tsv.gz | title.basics.tsv

Run:  python src/01_build_warehouse.py          (full build)
      python src/01_build_warehouse.py 03       (resume from step 03)
"""
import sys
import time
from pathlib import Path
import duckdb

from common import ROOT, find_raw_files

EXT, OUT = ROOT / "data/external", ROOT / "data/processed"
SQL_DIR = ROOT / "sql"
DB_PATH = OUT / "imdb.duckdb"


def run_sql_file(con, path: Path, files: dict) -> None:
    sql = path.read_text().replace("{EXT}", EXT.as_posix())
    for key, p in files.items():
        sql = sql.replace("{" + key + "}", p.as_posix())
    t0 = time.time()
    con.execute(sql)
    print(f"  ✓ {path.name:<28} {time.time() - t0:7.1f}s")


def main() -> None:
    files = find_raw_files(required=True)
    has_principals = "title_principals" in files
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    # Modest settings so this runs on a laptop; spill to disk if needed.
    con.execute(f"SET memory_limit='3GB'; SET preserve_insertion_order=false; "
                f"SET temp_directory='{(OUT / 'duckdb_tmp').as_posix()}';")

    start = sys.argv[1] if len(sys.argv) > 1 else "01"
    print("Building warehouse ->", DB_PATH)
    for f in sorted(SQL_DIR.glob("*.sql")):
        if f.name[:2] < start:
            continue
        if f.name.startswith("02_") and not has_principals:
            print(f"  – {f.name:<28} skipped (no title.principals file in data/raw)")
            continue
        run_sql_file(con, f, files)

    print("\nRow counts")
    tables = ["movies", "ratings", "crew", "people", "actor_known_for", "akas_footprint",
              "finance", "movie_mart"] + (["principals", "cast_billing", "producers"] if has_principals else [])
    for t in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<18} {n:>12,}")

    cov = con.execute(f"""
        SELECT COUNT(*)                                   AS films,
               AVG((lead_director IS NOT NULL)::INT)      AS pct_director,
               AVG((budget IS NOT NULL)::INT)             AS pct_budget,
               AVG((roi_multiple IS NOT NULL)::INT)       AS pct_roi
               {", AVG((tconst IN (SELECT tconst FROM cast_billing))::INT) AS pct_cast" if has_principals else ""}
        FROM universe""").df()
    print("\nCoverage inside analysis universe\n", cov.round(3).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
