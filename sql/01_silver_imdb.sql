-- =====================================================================
-- 01_silver_imdb.sql
-- Raw IMDb TSV dumps  ->  typed, filtered "silver" tables (parquet).
-- Scope: feature films only (titleType = 'movie', non-adult).
-- Engine: DuckDB (streams the .gz files; no need to unzip 1.2 GB).
-- Placeholders like {title_basics} are replaced by the runner with the real file path.
-- =====================================================================

-- ---------- 1. Movies (title.basics) ----------------------------------
CREATE OR REPLACE TABLE movies AS
SELECT
    tconst,
    primaryTitle                                   AS title,
    originalTitle                                  AS original_title,
    TRY_CAST(startYear AS INTEGER)                 AS year,
    TRY_CAST(runtimeMinutes AS INTEGER)            AS runtime,
    NULLIF(genres, '\N')                           AS genres
FROM read_csv('{title_basics}',
              delim = '\t', quote = '', escape = '', header = true,
              nullstr = '\N', all_varchar = true)
WHERE titleType = 'movie'
  AND isAdult = '0';

-- ---------- 2. Ratings (title.ratings), movies only -------------------
CREATE OR REPLACE TABLE ratings AS
SELECT r.tconst,
       CAST(r.averageRating AS DOUBLE)  AS rating,
       CAST(r.numVotes      AS BIGINT)  AS votes
FROM read_csv('{title_ratings}',
              delim = '\t', quote = '', header = true, all_varchar = true) r
SEMI JOIN movies m ON m.tconst = r.tconst;

-- ---------- 3. Crew (title.crew) exploded to one row per person/role --
CREATE OR REPLACE TABLE crew AS
WITH raw AS (
    SELECT c.tconst, c.directors, c.writers
    FROM read_csv('{title_crew}',
                  delim = '\t', quote = '', header = true,
                  nullstr = '\N', all_varchar = true) c
    SEMI JOIN movies m ON m.tconst = c.tconst
)
SELECT tconst, 'director' AS role, UNNEST(string_split(directors, ',')) AS nconst
FROM raw WHERE directors IS NOT NULL
UNION ALL
SELECT tconst, 'writer'   AS role, UNNEST(string_split(writers,   ',')) AS nconst
FROM raw WHERE writers IS NOT NULL;

-- ---------- 4. People (name.basics) -----------------------------------
-- title.principals was not supplied, so the only actor->title link is
-- name.basics.knownForTitles (<= 4 signature titles per person).
CREATE OR REPLACE TABLE people AS
SELECT nconst,
       primaryName                         AS name,
       TRY_CAST(birthYear AS INTEGER)      AS birth_year,
       primaryProfession                   AS professions,
       knownForTitles                      AS known_for
FROM read_csv('{name_basics}',
              delim = '\t', quote = '', escape = '', header = true,
              nullstr = '\N', all_varchar = true);

CREATE OR REPLACE TABLE actor_known_for AS
SELECT p.nconst, p.name,
       CASE WHEN p.professions LIKE '%actress%' THEN 'actress' ELSE 'actor' END AS gender_role,
       UNNEST(string_split(p.known_for, ','))                        AS tconst
FROM people p
WHERE p.known_for IS NOT NULL
  AND (p.professions LIKE '%actor%' OR p.professions LIKE '%actress%');

-- ---------- 5. Release footprint (title.akas) -------------------------
-- Number of distinct regions a film carries a localised title in:
-- a proxy for how widely it was distributed / marketed.
CREATE OR REPLACE TABLE akas_footprint AS
SELECT a.titleId AS tconst,
       COUNT(DISTINCT NULLIF(a.region, '\N')) AS n_regions,
       COUNT(*)                               AS n_akas
FROM read_csv('{title_akas}',
              delim = '\t', quote = '', escape = '', header = true,
              all_varchar = true,
              columns = {'titleId':'VARCHAR','ordering':'VARCHAR','title':'VARCHAR',
                         'region':'VARCHAR','language':'VARCHAR','types':'VARCHAR',
                         'attributes':'VARCHAR','isOriginalTitle':'VARCHAR'}) a
SEMI JOIN movies m ON m.tconst = a.titleId
GROUP BY 1;
