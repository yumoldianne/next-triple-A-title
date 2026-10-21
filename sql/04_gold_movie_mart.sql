-- =====================================================================
-- 04_gold_movie_mart.sql
-- One row per feature film: metadata + audience signals + money.
-- This is the single source of truth every downstream step reads.
-- =====================================================================

CREATE OR REPLACE TABLE movie_mart AS
WITH dir AS (
    SELECT tconst,
           COUNT(*) FILTER (WHERE role = 'director') AS n_directors,
           COUNT(*) FILTER (WHERE role = 'writer')   AS n_writers,
           MIN(nconst) FILTER (WHERE role = 'director') AS lead_director
    FROM crew GROUP BY 1
)
SELECT m.tconst, m.title, m.original_title, m.year, m.runtime, m.genres,
       split_part(m.genres, ',', 1)                    AS primary_genre,
       r.rating, r.votes,
       COALESCE(a.n_regions, 0)                        AS n_regions,
       COALESCE(a.n_akas, 0)                           AS n_akas,
       d.n_directors, d.n_writers, d.lead_director,
       f.budget, f.gross_ww, f.gross_dom, f.fin_source,
       CASE WHEN f.budget > 0 AND f.gross_ww > 0
            THEN f.gross_ww / f.budget END             AS roi_multiple
FROM movies m
LEFT JOIN ratings        r USING (tconst)
LEFT JOIN akas_footprint a USING (tconst)
LEFT JOIN dir            d USING (tconst)
LEFT JOIN finance        f USING (tconst);

-- Analysis universe: released, tracked feature films 1970-2025.
-- >= 500 votes is our proxy for "had a real theatrical/streaming release";
-- it removes festival shorts-in-disguise and vote-stuffed micro titles.
CREATE OR REPLACE VIEW universe AS
SELECT * FROM movie_mart
WHERE year BETWEEN 1970 AND 2025
  AND votes >= 500
  AND (runtime IS NULL OR runtime BETWEEN 60 AND 240);
