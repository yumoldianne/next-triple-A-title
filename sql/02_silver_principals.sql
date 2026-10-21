-- =====================================================================
-- 02_silver_principals.sql   (runs only if title.principals is present)
-- title.principals = up to ~10 key people per title, in credit order:
--   tconst | ordering | nconst | category | job | characters
-- We keep feature films only and derive:
--   cast_billing  : actors/actresses ranked 1..n by credit order
--                   (ordering also counts directors/writers, so we re-rank)
--   producers     : producer credits per film
-- =====================================================================

CREATE OR REPLACE TABLE principals AS
SELECT p.tconst,
       TRY_CAST(p.ordering AS INTEGER)  AS ordering,
       p.nconst,
       p.category,
       NULLIF(p.job, '\N')              AS job,
       NULLIF(p.characters, '\N')       AS characters
FROM read_csv('{title_principals}',
              delim = '\t', quote = '', escape = '', header = true,
              nullstr = '\N', all_varchar = true) p
SEMI JOIN movies m ON m.tconst = p.tconst;

CREATE OR REPLACE TABLE cast_billing AS
SELECT tconst, nconst, category,
       ROW_NUMBER() OVER (PARTITION BY tconst ORDER BY ordering, nconst) AS billing
FROM principals
WHERE category IN ('actor', 'actress');

CREATE OR REPLACE TABLE producers AS
SELECT DISTINCT tconst, nconst
FROM principals
WHERE category = 'producer';
