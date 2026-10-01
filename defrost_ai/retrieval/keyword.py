"""Keyword search: SQLite FTS5 BM25 over heading path (weight 2) and section text (weight 1).
The query becomes an OR of its content words; camelCase is split so `parseArgs` also matches "parse args"."""
from __future__ import annotations

import re

STOP_WORDS = set("a an and are as at be by can do does for from has have how i in is it its of on or the this to was "
                 "what when where which who why will with there their they them into via any all our we you your".split())


def words(s: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w for w in re.findall(r"[A-Za-z0-9]+", s.lower()) if len(w) > 1]


def fts_query(q: str) -> str:
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words(q)) if w not in STOP_WORDS)


def bm25_sections(db, query: str, n: int = 50) -> list[tuple[str, float]]:
    """-> [(section id, score)] best first; score = -bm25 (higher is better)."""
    fq = fts_query(query)
    if not fq:
        return []
    rows = db.execute("SELECT s.id, bm25(sections_fts, 2.0, 1.0) FROM sections_fts f JOIN sections s ON s.rowid=f.rowid "
                      "WHERE sections_fts MATCH ? ORDER BY bm25(sections_fts, 2.0, 1.0) LIMIT ?", (fq, n)).fetchall()
    return [(sid, -float(v)) for sid, v in rows]
