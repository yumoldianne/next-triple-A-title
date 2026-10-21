-- =====================================================================
-- 03_silver_finance.sql
-- Augment IMDb with money: production budget + worldwide gross.
--   * TMDB 10k (Kaggle/Udacity mirror on GitHub) : has imdb_id, 1960-2015
--   * The Numbers (Flatiron mirror on GitHub)     : title + date, 1915-2020
-- The Numbers has no IMDb key, so we fuzzy-key on normalised title + year
-- (+/- 1 year) and keep the most-voted IMDb candidate.
-- =====================================================================

CREATE OR REPLACE MACRO norm_title(t) AS
    regexp_replace(lower(strip_accents(t)), '[^a-z0-9]', '', 'g');

-- ---------- TMDB ------------------------------------------------------
CREATE OR REPLACE TABLE fin_tmdb AS
SELECT imdb_id                                  AS tconst,
       NULLIF(CAST(budget  AS DOUBLE), 0)       AS budget,
       NULLIF(CAST(revenue AS DOUBLE), 0)       AS gross_ww
FROM read_csv_auto('{EXT}/tmdb_movies_10k.csv', header = true)
WHERE imdb_id IS NOT NULL;

-- ---------- The Numbers -----------------------------------------------
CREATE OR REPLACE TABLE tn_raw AS
SELECT movie                                                         AS title,
       TRY_CAST(right(release_date, 4) AS INTEGER)                   AS year,
       CAST(replace(replace(production_budget, '$', ''), ',', '') AS DOUBLE) AS budget,
       NULLIF(CAST(replace(replace(worldwide_gross, '$', ''), ',', '') AS DOUBLE), 0) AS gross_ww,
       NULLIF(CAST(replace(replace(domestic_gross,  '$', ''), ',', '') AS DOUBLE), 0) AS gross_dom
FROM read_csv_auto('{EXT}/the_numbers_budgets.csv', header = true, all_varchar = true);

-- Normalised-title lookup for IMDb movies (two keys: primary + original),
-- so the fuzzy match becomes a hash equi-join instead of a nested loop.
CREATE OR REPLACE TABLE movie_title_keys AS
SELECT tconst, year, norm_title(title) AS tkey FROM movies WHERE year IS NOT NULL
UNION
SELECT tconst, year, norm_title(original_title) FROM movies WHERE year IS NOT NULL;

CREATE OR REPLACE TABLE fin_tn AS
WITH tn AS (SELECT *, norm_title(title) AS tkey FROM tn_raw),
cand AS (
    SELECT tn.*, k.tconst, r.votes,
           ROW_NUMBER() OVER (PARTITION BY tn.title, tn.year
                              ORDER BY ABS(k.year - tn.year), r.votes DESC NULLS LAST) AS rk
    FROM tn
    JOIN movie_title_keys k
      ON k.tkey = tn.tkey
     AND ABS(k.year - tn.year) <= 1
    LEFT JOIN ratings r ON r.tconst = k.tconst
)
SELECT DISTINCT ON (tconst) tconst, budget, gross_ww, gross_dom
FROM cand WHERE rk = 1;

-- ---------- Combined: prefer The Numbers (audited-ish), fall back to TMDB
CREATE OR REPLACE TABLE finance AS
SELECT COALESCE(n.tconst, t.tconst)               AS tconst,
       COALESCE(n.budget,   t.budget)             AS budget,
       COALESCE(n.gross_ww, t.gross_ww)           AS gross_ww,
       n.gross_dom,
       CASE WHEN n.tconst IS NOT NULL THEN 'the_numbers' ELSE 'tmdb' END AS fin_source
FROM fin_tn n
FULL OUTER JOIN fin_tmdb t ON n.tconst = t.tconst
QUALIFY ROW_NUMBER() OVER (PARTITION BY COALESCE(n.tconst, t.tconst)
                           ORDER BY n.budget DESC NULLS LAST) = 1;
