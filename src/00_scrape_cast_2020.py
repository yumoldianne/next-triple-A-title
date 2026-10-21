"""
00_scrape_cast_2020.py — Augment IMDb with 2020 cast billing order.

Why: the brief's IMDb extract has no title.principals (actor -> film links).
name.basics.knownForTitles is the only fallback and it ranks uncredited extras
of 'Tenet' as 2020's most popular actors. Wikipedia's "List of American films
of 2020" lists each film's cast in billing order, which is what we need.

Output: data/external/cast_2020_billing.tsv  (title, cast_in_billing_order, source)
The committed file was captured on 15 Oct 2026 and topped up by hand with six
widely-voted non-US 2020 titles (source = manual_non_us). Re-running this script
regenerates the Wikipedia rows only.

If you can download title.principals.tsv.gz, prefer it: put it in data/raw and
use sql/99_optional_principals.sql.example instead — it covers every country and year.
"""
import re
from pathlib import Path
import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/List_of_American_films_of_2020"
OUT = Path(__file__).resolve().parents[1] / "data/external/cast_2020_billing_wikipedia.tsv"


def cast_from_cell(cell: str, top_n: int = 4) -> str:
    """'Director (director); Writer (screenplay); A, B, C' -> 'A, B, C' (top-n)."""
    actors = cell.split(";")[-1]
    actors = re.sub(r"\[.*?\]", "", actors)
    names = [a.strip() for a in actors.split(",") if a.strip() and "(" not in a]
    return ", ".join(names[:top_n])


def main():
    html = requests.get(URL, headers={"User-Agent": "aaa-title-research/1.0"}, timeout=30).text
    tables = pd.read_html(html)
    rows = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if not any("cast" in c for c in cols) or not any("title" in c for c in cols):
            continue
        title_col = t.columns[[i for i, c in enumerate(cols) if "title" in c][0]]
        cast_col = t.columns[[i for i, c in enumerate(cols) if "cast" in c][0]]
        for _, r in t.iterrows():
            cast = cast_from_cell(str(r[cast_col]))
            if cast:
                rows.append({"title": str(r[title_col]).strip(), "cast_in_billing_order": cast,
                             "source": "wikipedia_us_2020"})
    df = pd.DataFrame(rows).drop_duplicates("title")
    df.to_csv(OUT, sep="\t", index=False)
    print(f"{len(df)} films -> {OUT}")


if __name__ == "__main__":
    main()
